"""PIIGuardLLM — off/shadow/enforce × invoke/stream, tool_calls 복원, 실패 시 텍스트 없이 done(error), 위임 속성, 힌트."""
import logging

import pytest

from agent_arch.agents.context import set_conversation_id, set_user_context
from agent_arch.core.config import settings
from agent_arch.llm.base import BaseLLM, LLMResponse, Message, StreamEvent, ToolCall
from agent_arch.llm.pii.guard_llm import PLACEHOLDER_HINT, PIIGuardLLM

REQUESTER = {"sub": "u1", "name": "백지선", "email": "user@example.com", "username": "jsbaek", "company": "(주)회사"}


class _Inner(BaseLLM):
    """받은 messages 를 기록하고 정해진 응답을 돌려주는 가짜 provider."""

    def __init__(self, reply: str = "", tool_calls=None, events=None, raise_on_invoke=None):
        super().__init__({"model": "fake-1", "stateless": False, "supports_function_calling": True})
        self.calls: list = []
        self.reply, self.reply_calls, self.events = reply, tool_calls, events
        self.raise_on_invoke = raise_on_invoke
        self.client = "provider-specific-attr"

    async def invoke(self, messages, **kwargs):
        self.calls.append(("invoke", messages, kwargs))
        if self.raise_on_invoke:
            raise self.raise_on_invoke
        return LLMResponse(content=self.reply, tool_calls=self.reply_calls, finish_reason="tool_calls" if self.reply_calls else "stop", usage={"total_tokens": 1})

    async def stream(self, messages, **kwargs):
        self.calls.append(("stream", messages, kwargs))
        for ev in self.events or []:
            yield ev

    def _convert_tool_to_schema(self, tool):
        return {"name": getattr(tool, "name", "t")}


@pytest.fixture
def ctx():
    set_user_context(REQUESTER)
    set_conversation_id("conv-1")
    yield
    set_user_context({})
    set_conversation_id(None)


@pytest.fixture
def enforce(monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_mode", "enforce", raising=False)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "(주)회사", raising=False)
    monkeypatch.setattr(settings, "pii_guard_secret", "test-secret", raising=False)


@pytest.fixture
def shadow(monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_mode", "shadow", raising=False)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "(주)회사", raising=False)
    monkeypatch.setattr(settings, "pii_guard_secret", "test-secret", raising=False)


def _msgs():
    return [Message(role="system", content="너는 에이전트다."),
            Message(role="user", content="백지선 입니다. user@example.com 로 010-1234-5678 연락처를 보내 주세요.")]


def _sent(inner):
    return inner.calls[-1][1]


# ── 위임 속성 ────────────────────────────────────────────────────────────────
def test_delegates_public_contract(ctx):
    inner = _Inner()
    g = PIIGuardLLM(inner, "gemini")
    assert g.model == "fake-1" and g.stateless is False and g.supports_function_calling is True
    assert g.config is inner.config and g.client == "provider-specific-attr"
    assert "PIIGuardLLM(" in repr(g)
    with pytest.raises(AttributeError):
        _ = g.no_such_attr


@pytest.mark.asyncio
async def test_bind_tools_returns_wrapper_and_binds_inner(ctx):
    inner = _Inner()
    g = PIIGuardLLM(inner, "gemini")
    tools = [object(), object()]
    assert (await g.bind_tools(tools)) is g
    assert inner.get_bound_tools() == tools and g.get_bound_tools() == tools and g._bound_tools == tools
    assert g._convert_tool_to_schema(None) == {"name": "t"}


# ── off ──────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_off_is_transparent(ctx, monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_mode", "enforce", raising=False)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "다른회사", raising=False)   # 이 회사는 꺼짐
    inner = _Inner(reply="답")
    msgs = _msgs()
    resp = await PIIGuardLLM(inner, "gemini").invoke(msgs)
    assert _sent(inner) is msgs and resp.content == "답"


# ── shadow ───────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_shadow_sends_original_and_logs_without_values(ctx, shadow, caplog):
    inner = _Inner(reply="백지선 님께 010-1234-5678 로 보냈습니다")
    with caplog.at_level(logging.INFO, logger="agent_arch.llm.pii.audit"):
        resp = await PIIGuardLLM(inner, "openai").invoke(_msgs())
    sent = _sent(inner)
    assert "user@example.com" in sent[1].content and "[[" not in sent[1].content
    assert resp.content.startswith("백지선 님께")
    rec = [r for r in caplog.records if "shadow" in r.getMessage()]
    assert len(rec) == 1
    msg = rec[0].getMessage()
    assert "'PERSON': 1" in msg and "echo={'PERSON': 1, 'PHONE': 1}" in msg          # 응답 원문의 이름·전화를 모델이 되돌려 말했다
    assert "jsbaek@" not in msg and "1234" not in msg and "백지선" not in msg


@pytest.mark.asyncio
async def test_shadow_stream_passes_events_through(ctx, shadow):
    events = [StreamEvent(type="text", content="백지선"), StreamEvent(type="text", content=" 님"), StreamEvent(type="done", finish_reason="stop")]
    inner = _Inner(events=events)
    out = [ev async for ev in PIIGuardLLM(inner, "gemini").stream(_msgs())]
    assert [ev.content for ev in out[:2]] == ["백지선", " 님"] and out[-1].type == "done"


# ── enforce ──────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_enforce_invoke_substitutes_and_restores(ctx, enforce):
    inner = _Inner(reply="")
    g = PIIGuardLLM(inner, "gemini")
    msgs = _msgs()
    await g.invoke(msgs)
    sent = _sent(inner)
    user = sent[-1].content
    assert "백지선" not in user and "user@example.com" not in user and "010-1234-5678" not in user
    assert user.count("[[") == 3 and "@example.com" in user
    assert PLACEHOLDER_HINT in sent[0].content and sent[0].content.startswith("너는 에이전트다.")
    assert msgs[1].content.startswith("백지선")                      # 원본 불변
    # 모델이 토큰을 그대로 쓴 답변은 원문으로 복원된다.
    person = [t for t in user.split() if t.startswith("[[PERSON_")][0]
    inner.reply = f"{person} 님께 안내했습니다"
    resp = await g.invoke(msgs)
    assert resp.content == "백지선 님께 안내했습니다"


@pytest.mark.asyncio
async def test_enforce_restores_tool_call_arguments_and_flags_unresolved(ctx, enforce, caplog):
    inner = _Inner(reply="")
    g = PIIGuardLLM(inner, "gemini")
    await g.invoke(_msgs())
    user = _sent(inner)[-1].content
    email_tok = [t for t in user.split() if t.startswith("[[EMAIL_")][0].split("@")[0]
    inner.reply_calls = [ToolCall(id="c1", name="mail_send", arguments={"to": [f"{email_tok}@example.com"], "cc": "[[PERSON_deadbeef]]", "n": 1})]
    with caplog.at_level(logging.WARNING, logger="agent_arch.llm.pii.audit"):
        resp = await g.invoke(_msgs())
    args = resp.tool_calls[0].arguments
    assert args["to"] == ["user@example.com"] and args["cc"] == "[[PERSON_deadbeef]]" and args["n"] == 1
    assert any("unresolved=['PERSON_deadbeef']" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_enforce_stream_restores_across_chunks_and_tool_calls(ctx, enforce):
    probe = _Inner(reply="")
    await PIIGuardLLM(probe, "gemini").invoke(_msgs())
    user = _sent(probe)[-1].content
    person = [t for t in user.split() if t.startswith("[[PERSON_")][0]
    phone = [t for t in user.split() if t.startswith("[[PHONE_")][0]
    a, b = person[:6], person[6:]
    events = [StreamEvent(type="text", content="담당자 "), StreamEvent(type="text", content=a), StreamEvent(type="text", content=b + "님"),
              StreamEvent(type="tool_calls", tool_calls=[ToolCall(id="c", name="sms", arguments={"to": phone})]),
              StreamEvent(type="done", finish_reason="tool_calls", usage={"total_tokens": 3})]
    inner = _Inner(events=events)
    out = [ev async for ev in PIIGuardLLM(inner, "gemini").stream(_msgs())]
    text = "".join(ev.content for ev in out if ev.type == "text")
    assert text == "담당자 백지선님"
    assert "[[" not in text
    tc = [ev for ev in out if ev.type == "tool_calls"][0].tool_calls[0]
    assert tc.arguments == {"to": "010-1234-5678"}
    assert out[-1].type == "done" and out[-1].usage == {"total_tokens": 3}
    assert "백지선" not in _sent(inner)[-1].content


@pytest.mark.asyncio
async def test_enforce_unknown_token_in_text_becomes_label(ctx, enforce):
    inner = _Inner(events=[StreamEvent(type="text", content="[[PERSON_deadbeef]] 님이"), StreamEvent(type="done", finish_reason="stop")])
    out = [ev async for ev in PIIGuardLLM(inner, "gemini").stream(_msgs())]
    assert "".join(ev.content for ev in out if ev.type == "text") == "(확인되지 않은 사람) 님이"


@pytest.mark.asyncio
async def test_enforce_prepare_failure_yields_no_text(ctx, enforce, monkeypatch):
    from agent_arch.llm.pii import guard_llm as mod

    def boom(self, messages, lexicon, scope):
        raise RuntimeError("치환기 고장")

    monkeypatch.setattr(mod.PIIGuardLLM, "_substitute", boom)
    inner = _Inner(reply="원문이 새면 안 됨", events=[StreamEvent(type="text", content="x"), StreamEvent(type="done", finish_reason="stop")])
    g = PIIGuardLLM(inner, "gemini")
    out = [ev async for ev in g.stream(_msgs())]
    assert out == [StreamEvent(type="done", finish_reason="error")] and inner.calls == []
    resp = await g.invoke(_msgs())
    assert resp.content == "" and resp.finish_reason == "error" and inner.calls == []


@pytest.mark.asyncio
async def test_enforce_images_pass_with_warning(ctx, enforce, caplog):
    from agent_arch.llm.base import ImageContent
    inner = _Inner(reply="")
    msgs = [Message(role="user", content="백지선 영수증", images=[ImageContent(data="AAAA", mime_type="image/png")])]
    with caplog.at_level(logging.WARNING):
        await PIIGuardLLM(inner, "gemini").invoke(msgs)
    sent = _sent(inner)[-1]
    assert sent.images and sent.images[0].data == "AAAA" and "백지선" not in sent.content
    assert any("이미지 1장" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_hint_added_once_even_if_no_system(ctx, enforce):
    inner = _Inner(reply="")
    g = PIIGuardLLM(inner, "gemini")
    await g.invoke([Message(role="user", content="백지선 입니다")])
    sent = _sent(inner)
    assert sent[0].role == "system" and sent[0].content == PLACEHOLDER_HINT and len(sent) == 2
    await g.invoke([Message(role="system", content=PLACEHOLDER_HINT), Message(role="user", content="백지선")])
    assert _sent(inner)[0].content.count("[자리표시 안내]") == 1
    await g.invoke([Message(role="user", content="개인정보 없음")])
    assert _sent(inner)[0].role == "user"                               # 치환이 없으면 힌트도 없다


@pytest.mark.asyncio
async def test_same_conversation_same_tokens_across_calls(ctx, enforce):
    inner = _Inner(reply="")
    g = PIIGuardLLM(inner, "gemini")
    await g.invoke([Message(role="user", content="백지선")])
    first = _sent(inner)[-1].content
    await g.invoke([Message(role="user", content="아무개와 백지선")])
    assert first in _sent(inner)[-1].content
    set_conversation_id("conv-2")
    await g.invoke([Message(role="user", content="백지선")])
    assert _sent(inner)[-1].content != first


@pytest.mark.asyncio
async def test_concurrent_requesters_do_not_leak_lexicon(enforce):
    """다른 사용자·대화가 동시에 돌아도 어휘·복원표가 섞이지 않는다(호출별 불변 객체)."""
    import asyncio
    from contextvars import copy_context

    async def run(user, conv, text):
        set_user_context(user)
        set_conversation_id(conv)
        inner = _Inner(reply="")
        await PIIGuardLLM(inner, "gemini").invoke([Message(role="user", content=text)])
        return _sent(inner)[-1].content

    a = {**REQUESTER, "sub": "a", "name": "김보라", "email": "user@example.com", "username": "bkim"}
    b = {**REQUESTER, "sub": "b", "name": "박미래", "email": "user@example.com", "username": "mirae"}
    ra, rb = await asyncio.gather(
        copy_context().run(lambda: asyncio.ensure_future(run(a, "c-a", "김보라 와 박미래"))),
        copy_context().run(lambda: asyncio.ensure_future(run(b, "c-b", "김보라 와 박미래"))),
    )
    assert "김보라" not in ra and "박미래" in ra          # a 의 어휘엔 김보라만
    assert "박미래" not in rb and "김보라" in rb
