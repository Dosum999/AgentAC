"""AnthropicLLM 도구 호출(tool_use) 테스트.

도구 스키마 변환, tool_choice 매핑, 메시지 변환의 tool_use/tool_result 짝 방어,
invoke/stream 의 tool_use 파싱을 검증한다. Messages API 호출은 httpx Mock 으로 대체한다.
도면 레인(도구 바인딩 없음)이 예전과 같은 요청을 내는지도 함께 본다.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent_arch.llm.anthropic_provider import AnthropicLLM, _tool_choice_param
from agent_arch.llm.base import Message
from agent_arch.llm.factory import LLMFactory


def _make_llm(supports_function_calling: bool = True) -> AnthropicLLM:
    return AnthropicLLM({
        "api_key": "sk-ant-test",
        "model": "claude-sonnet-5",
        "temperature": 0.4,
        "max_tokens": 1024,
        "supports_function_calling": supports_function_calling,
        "supports_streaming": True,
    })


class _FakeTool:
    def __init__(self, name: str = "get_weather"):
        self.name = name
        self.description = "도시의 날씨를 조회한다."
        self.parameters = {"city": {"type": "string", "description": "도시명"}}
        self.required_params = ["city"]


def _tool_turn_messages() -> list[Message]:
    """assistant tool_calls → tool 결과 → 다음 user 질문 의 정상 히스토리."""
    return [
        Message(role="system", content="너는 비서다."),
        Message(role="user", content="서울 날씨?"),
        Message(
            role="assistant", content="확인해 보겠습니다.",
            tool_calls=[{"id": "toolu_1", "name": "get_weather", "arguments": {"city": "서울"}}],
        ),
        Message(role="tool", content='{"temp": 21}', name="get_weather", tool_call_id="toolu_1"),
        Message(role="assistant", content="서울은 21도입니다."),
        Message(role="user", content="부산은?"),
    ]


# ── 스키마 · tool_choice ────────────────────────────────────────────────

def test_tool_schema_uses_input_schema():
    schema = _make_llm()._convert_tool_to_schema(_FakeTool())
    assert schema["name"] == "get_weather"
    assert schema["input_schema"]["type"] == "object"
    assert schema["input_schema"]["properties"] == {"city": {"type": "string", "description": "도시명"}}
    assert schema["input_schema"]["required"] == ["city"]
    assert "parameters" not in schema


@pytest.mark.parametrize("mode,allowed,expected", [
    ("AUTO", None, {"type": "auto"}),
    ("ANY", None, {"type": "any"}),
    ("ANY", ["get_weather"], {"type": "tool", "name": "get_weather"}),
    ("ANY", ["a", "b"], {"type": "any"}),
    ("NONE", None, {"type": "none"}),
    ("unknown", None, {"type": "auto"}),
])
def test_tool_choice_mapping(mode, allowed, expected):
    assert _tool_choice_param(mode, allowed) == expected


def test_claude_alias_resolves_to_anthropic():
    assert LLMFactory.normalize_provider("claude") == "anthropic"
    assert LLMFactory.normalize_provider("Claude") == "anthropic"


# ── 메시지 변환: tool_use / tool_result 짝 ───────────────────────────────

def test_convert_messages_pairs_tool_use_and_result():
    system, out = _make_llm()._convert_messages(_tool_turn_messages(), tools_defined=True)
    assert system == "너는 비서다."
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "user", "assistant", "user"]

    assistant_blocks = out[1]["content"]
    assert assistant_blocks[0] == {"type": "text", "text": "확인해 보겠습니다."}
    assert assistant_blocks[1] == {
        "type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "서울"},
    }
    result_blocks = out[2]["content"]
    assert result_blocks == [{"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"temp": 21}'}]


def test_convert_messages_demotes_tool_blocks_when_tools_not_defined():
    """검증·요약 레인처럼 tools 없이 부르면 도구 블록은 전부 텍스트로 내려간다."""
    _, out = _make_llm()._convert_messages(_tool_turn_messages(), tools_defined=False)
    types = [b["type"] for m in out for b in m["content"]]
    assert "tool_use" not in types and "tool_result" not in types
    assistant_text = out[1]["content"][1]["text"]
    assert assistant_text.startswith("[도구 호출: get_weather(")
    assert out[2]["content"][0]["text"].startswith("[도구 결과: get_weather]")


def test_convert_messages_demotes_orphan_tool_use_followed_by_user_text():
    """압축으로 tool 결과가 빠진 tool_use 는 텍스트로 — 그대로 두면 Anthropic 400."""
    msgs = [
        Message(role="user", content="서울 날씨?"),
        Message(
            role="assistant", content="",
            tool_calls=[{"id": "toolu_x", "name": "get_weather", "arguments": {"city": "서울"}}],
        ),
        Message(role="user", content="아 됐고 부산은?"),
    ]
    _, out = _make_llm()._convert_messages(msgs, tools_defined=True)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert out[1]["content"] == [{"type": "text", "text": '[도구 호출: get_weather({"city": "서울"})]'}]


def test_convert_messages_demotes_trailing_orphan_tool_use():
    msgs = [
        Message(role="user", content="서울 날씨?"),
        Message(
            role="assistant", content="",
            tool_calls=[{"id": "toolu_y", "name": "get_weather", "arguments": {}}],
        ),
    ]
    _, out = _make_llm()._convert_messages(msgs, tools_defined=True)
    assert out[-1]["content"][0]["type"] == "text"


def test_convert_messages_demotes_tool_result_without_matching_use():
    """top_k 로 assistant 턴이 잘려 결과만 남은 경우."""
    msgs = [
        Message(role="tool", content="21도", name="get_weather", tool_call_id="toolu_gone"),
        Message(role="user", content="그래서 뭐 입지?"),
    ]
    _, out = _make_llm()._convert_messages(msgs, tools_defined=True)
    assert len(out) == 1 and out[0]["role"] == "user"
    assert out[0]["content"][0] == {"type": "text", "text": "[도구 결과: get_weather]\n21도"}
    assert out[0]["content"][1] == {"type": "text", "text": "그래서 뭐 입지?"}


def test_convert_messages_multiple_tool_results_merge_into_one_user_turn():
    msgs = [
        Message(role="user", content="서울·부산 날씨?"),
        Message(
            role="assistant", content="",
            tool_calls=[
                {"id": "t1", "name": "get_weather", "arguments": {"city": "서울"}},
                {"id": "t2", "name": "get_weather", "arguments": {"city": "부산"}},
            ],
        ),
        Message(role="tool", content="21", name="get_weather", tool_call_id="t1"),
        Message(role="tool", content="23", name="get_weather", tool_call_id="t2"),
    ]
    _, out = _make_llm()._convert_messages(msgs, tools_defined=True)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert [b["tool_use_id"] for b in out[2]["content"]] == ["t1", "t2"]


# ── payload ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_payload_has_tools_only_when_bound():
    llm = _make_llm()
    assert "tools" not in llm._payload([Message(role="user", content="hi")], {}, stream=False)

    await llm.bind_tools([_FakeTool()])
    payload = llm._payload([Message(role="user", content="hi")], {"tool_mode": "ANY"}, stream=False)
    assert [t["name"] for t in payload["tools"]] == ["get_weather"]
    assert payload["tool_choice"] == {"type": "any"}


@pytest.mark.asyncio
async def test_payload_ignores_tools_when_flag_off():
    """yaml 에서 supports_function_calling 을 끄면 바인딩이 있어도 tools 를 안 보낸다."""
    llm = _make_llm(supports_function_calling=False)
    await llm.bind_tools([_FakeTool()])
    payload = llm._payload(_tool_turn_messages(), {}, stream=False)
    assert "tools" not in payload
    # tools 가 없으니 히스토리의 도구 블록도 텍스트로 내려가야 한다.
    types = [b["type"] for m in payload["messages"] for b in m["content"]]
    assert set(types) == {"text"}


# ── invoke ─────────────────────────────────────────────────────────────

def _mock_post_response(content_blocks: list, stop_reason: str) -> MagicMock:
    resp = MagicMock()
    resp.status_code = 200
    body = {
        "content": content_blocks,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    resp.text = json.dumps(body)
    resp.json.return_value = body
    return resp


@pytest.mark.asyncio
async def test_invoke_parses_tool_use_blocks():
    llm = _make_llm()
    await llm.bind_tools([_FakeTool()])
    resp = _mock_post_response(
        [
            {"type": "text", "text": "확인해 볼게요."},
            {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "서울"}},
        ],
        stop_reason="tool_use",
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("agent_arch.llm.anthropic_provider.httpx.AsyncClient", return_value=client), \
         patch("agent_arch.llm.anthropic_provider.meter_llm_call"):
        out = await llm.invoke([Message(role="user", content="서울 날씨?")], tool_mode="AUTO")

    sent = client.post.call_args.kwargs["json"]
    assert sent["tools"][0]["name"] == "get_weather"
    assert sent["tool_choice"] == {"type": "auto"}
    assert out.content == "확인해 볼게요."
    assert out.finish_reason == "tool_calls"
    assert [(tc.id, tc.name, tc.arguments) for tc in out.tool_calls] == [
        ("toolu_1", "get_weather", {"city": "서울"})
    ]
    assert out.usage == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


@pytest.mark.asyncio
async def test_invoke_without_tools_is_unchanged_for_drawing_lane():
    llm = _make_llm()
    resp = _mock_post_response([{"type": "text", "text": "판독 결과"}], stop_reason="end_turn")
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    with patch("agent_arch.llm.anthropic_provider.httpx.AsyncClient", return_value=client), \
         patch("agent_arch.llm.anthropic_provider.meter_llm_call"):
        out = await llm.invoke([Message(role="user", content="도면 읽어줘")], temperature=0.0)

    sent = client.post.call_args.kwargs["json"]
    assert "tools" not in sent and "tool_choice" not in sent
    assert "temperature" not in sent      # 도면이 0.0 을 줘도 보내지 않는다(Claude 5 폐기)
    assert out.tool_calls is None and out.finish_reason == "stop"


# ── stream ─────────────────────────────────────────────────────────────

def _sse(events: list[dict]) -> list[str]:
    lines = []
    for ev in events:
        lines.append(f"event: {ev['type']}")
        lines.append("data: " + json.dumps(ev, ensure_ascii=False))
        lines.append("")
    return lines


def _stream_client(lines: list[str], status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.aread = AsyncMock(return_value=b'{"error":{"message":"bad"}}')

    async def _aiter_lines():
        for line in lines:
            yield line

    resp.aiter_lines = _aiter_lines

    @asynccontextmanager
    async def _stream(*_args, **_kwargs):
        yield resp

    client = MagicMock()
    client.stream = _stream
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=False)
    return client


@pytest.mark.asyncio
async def test_stream_accumulates_tool_use_and_emits_once():
    llm = _make_llm()
    await llm.bind_tools([_FakeTool()])
    lines = _sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 12}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "확인"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "할게요"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {}}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"ci'}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": 'ty": "서울"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
        {"type": "message_stop"},
    ])
    with patch("agent_arch.llm.anthropic_provider.httpx.AsyncClient", return_value=_stream_client(lines)), \
         patch("agent_arch.llm.anthropic_provider.meter_llm_call"):
        events = [e async for e in llm.stream([Message(role="user", content="서울 날씨?")])]

    assert [e.type for e in events] == ["text", "text", "tool_calls", "done"]
    assert "".join(e.content for e in events[:2]) == "확인할게요"
    tc = events[2].tool_calls
    assert len(tc) == 1 and tc[0].id == "toolu_9" and tc[0].arguments == {"city": "서울"}
    assert events[-1].finish_reason == "tool_calls"
    assert events[-1].usage == {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19}


@pytest.mark.asyncio
async def test_stream_text_only_unchanged():
    llm = _make_llm()
    lines = _sse([
        {"type": "message_start", "message": {"usage": {"input_tokens": 3}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "안녕"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
    ])
    with patch("agent_arch.llm.anthropic_provider.httpx.AsyncClient", return_value=_stream_client(lines)), \
         patch("agent_arch.llm.anthropic_provider.meter_llm_call"):
        events = [e async for e in llm.stream([Message(role="user", content="hi")])]
    assert [e.type for e in events] == ["text", "done"]
    assert events[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_stream_malformed_tool_input_falls_back_to_empty_args():
    llm = _make_llm()
    await llm.bind_tools([_FakeTool()])
    lines = _sse([
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "toolu_b", "name": "get_weather", "input": {}}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}},
        # content_block_stop 없이 끊김
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 2}},
    ])
    with patch("agent_arch.llm.anthropic_provider.httpx.AsyncClient", return_value=_stream_client(lines)), \
         patch("agent_arch.llm.anthropic_provider.meter_llm_call"):
        events = [e async for e in llm.stream([Message(role="user", content="?")])]
    assert events[0].type == "tool_calls"
    assert events[0].tool_calls[0].arguments == {}
    assert events[-1].finish_reason == "tool_calls"


# ── 샘플링 인자 미전송 (Claude Opus 4.6 이후 temperature·top_p 폐기) ─────────

def test_payload_never_sends_sampling_params():
    llm = AnthropicLLM({"api_key": "k", "model": "claude-sonnet-5", "max_tokens": 10,
                        "temperature": 0.4, "top_p": 0.95})
    payload = llm._payload([Message(role="user", content="hi")], {"temperature": 0.0, "top_p": 0.9}, stream=False)
    assert "temperature" not in payload and "top_p" not in payload and "top_k" not in payload
    assert payload["max_tokens"] == 10
