"""agent_arch.agents.react.loop (react_loop_events) 단위 테스트.

LLM stream() 응답 패턴에 따른 ProgressEvent / LoopResult yield 흐름,
malformed 처리, ANY 재시도, max_iterations 소진 등 핵심 분기를 검증한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.react.loop import react_loop_events, _SAFE_MSG_FALLBACK
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import Message, StreamEvent, ToolCall


# ── 테스트 헬퍼 ──────────────────────────────────────────────────────────────

_USAGE = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
_MSGS = [Message(role="user", content="질문")]


@pytest.mark.asyncio
async def test_author_lookup_preserves_original_intent_and_pins_first_tool(monkeypatch):
    from agent_arch.agents.react.author_first import AUTHOR_TOOL_NAME
    question = "내가 AI팀에 에이전트 정책 문서라고 공유한 링크 찾아줘"
    calls, arguments = [], []
    count = 0
    async def stream(*args, **kwargs):
        nonlocal count
        calls.append(kwargs)
        count += 1
        if count == 1:
            yield StreamEvent(type="tool_calls", tool_calls=[ToolCall(
                id="author", name=AUTHOR_TOOL_NAME, arguments={"query": "에이전트 정책"})])
            yield StreamEvent(type="done", finish_reason="tool_calls")
        else:
            yield StreamEvent(type="text", content="찾았습니다.")
            yield StreamEvent(type="done", finish_reason="stop")
    async def execute(tcs, tools, **kwargs):
        arguments.extend(tc.arguments for tc in tcs)
        return [(tc, '{"count":1,"results":[{"message":"역할과 권한"}]}') for tc in tcs]
    monkeypatch.setattr("agent_arch.agents.react.loop.execute_tools", execute)
    llm = MagicMock()
    llm.stream = stream
    llm.bind_tools = AsyncMock()
    tool = MagicMock()
    tool.name = AUTHOR_TOOL_NAME
    async for _ in react_loop_events(llm, question, [Message(role="user", content=question)],
                                    [tool], AsyncMock(return_value=[]), 5):
        pass
    assert calls[0]["allowed_tools"] == [AUTHOR_TOOL_NAME]
    assert calls[0]["tool_mode"] == "ANY"
    assert arguments[0]["search_intent"] == question
    assert "must_contain_link" not in arguments[0]


def _text(content: str = "응답", usage: dict | None = None, finish_reason: str = "stop"):
    """텍스트 답변 회차 stream 시퀀스 (text → done) 팩토리."""
    async def _gen(*args, **kwargs):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)
    return _gen


def _tool_calls(tool_calls: list[ToolCall], usage: dict | None = None):
    """도구 호출 회차 stream 시퀀스 (tool_calls → done(tool_calls)) 팩토리."""
    async def _gen(*args, **kwargs):
        yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
        yield StreamEvent(type="done", finish_reason="tool_calls", usage=usage)
    return _gen


def _malformed(usage: dict | None = None):
    """malformed_function_call 회차 stream 시퀀스 팩토리."""
    async def _gen(*args, **kwargs):
        yield StreamEvent(type="done", finish_reason="malformed_function_call", usage=usage)
    return _gen


def _llm(*per_iter_gens) -> MagicMock:
    """각 stream() 호출마다 다음 generator 가 사용되도록 side_effect 로 주입한 LLM."""
    iter_gens = list(per_iter_gens)

    def _next_gen(*args, **kwargs):
        gen = iter_gens.pop(0)
        return gen(*args, **kwargs)

    llm = MagicMock()
    llm.stream = _next_gen
    llm.bind_tools = AsyncMock()
    return llm


def _tc(name: str = "tool1") -> ToolCall:
    """테스트용 ToolCall을 생성한다."""
    return ToolCall(id=f"id_{name}", name=name, arguments={})


def _make_tool(name: str):
    """execute가 성공 결과를 반환하는 모의 도구를 생성한다."""
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(return_value=ToolResult(success=True, output="결과"))
    return tool


async def _run(llm, input="질문", tools=None, search_fn=None, max_iterations=5):
    """react_loop_events를 소비하여 (progress_events, loop_result) 튜플을 반환한다."""
    if tools is None:
        tools = []
    if search_fn is None:
        search_fn = AsyncMock(return_value=[])
    events = [e async for e in react_loop_events(llm, input, list(_MSGS), tools, search_fn, max_iterations)]
    progress = [e for e in events if isinstance(e, ProgressEvent)]
    results = [e for e in events if isinstance(e, LoopResult)]
    assert len(results) == 1, "LoopResult는 정확히 한 번 yield 되어야 한다"
    return progress, results[0]


# ── 직접 답변 경로 ────────────────────────────────────────────────────────────

class TestReactLoopDirectAnswer:
    """도구 호출 없이 바로 답변을 반환하는 경로 검증."""

    @pytest.mark.asyncio
    async def test_yields_loop_result_with_llm_content(self):
        """LLM 이 텍스트 토큰을 흘리면 누적본이 LoopResult.content 에 담긴다."""
        llm = _llm(_text("최종 답변"))
        _, result = await _run(llm)
        assert result.content == "최종 답변"

    @pytest.mark.asyncio
    async def test_token_progress_events_emitted(self):
        """텍스트 응답 회차에서 ProgressEvent('token') 이 yield 된다."""
        llm = _llm(_text("토큰답변"))
        progress, _ = await _run(llm)
        token_events = [e for e in progress if e.event_type == "token"]
        assert token_events, "token 이벤트가 발사되어야 한다"
        assert "".join(e.content for e in token_events) == "토큰답변"

    @pytest.mark.asyncio
    async def test_token_usage_in_loop_result(self):
        """LLM 호출의 usage가 LoopResult.token_usage에 반영된다."""
        llm = _llm(_text("답변", usage=_USAGE))
        _, result = await _run(llm)
        assert result.token_usage["prompt_tokens"] == 5
        assert result.token_usage["completion_tokens"] == 10

    @pytest.mark.asyncio
    async def test_llm_invoke_count_is_one(self):
        """도구 호출 없는 단일 경로에서 llm_invoke_count는 1이다."""
        llm = _llm(_text("답변"))
        _, result = await _run(llm)
        assert result.llm_invoke_count == 1


# ── 도구 호출 후 최종 답변 경로 ───────────────────────────────────────────────

class TestReactLoopWithTool:
    """tool_call → tool 실행 → 최종 답변 경로 검증."""

    @pytest.mark.asyncio
    async def test_tool_then_final_answer_yields_result(self):
        """tool_call 실행 후 최종 답변이 LoopResult로 반환된다."""
        tc = _tc("my_tool")
        llm = _llm(
            _tool_calls([tc], usage=_USAGE),
            _text("최종 답변", usage=_USAGE),
        )
        tool = _make_tool("my_tool")
        _, result = await _run(llm, tools=[tool])
        assert result.content == "최종 답변"
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_token_usage_accumulated_across_invokes(self):
        """여러 번 stream 호출한 token_usage 가 누적된다."""
        tc = _tc("my_tool")
        usage_1 = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        usage_2 = {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
        llm = _llm(
            _tool_calls([tc], usage=usage_1),
            _text("최종 답변", usage=usage_2),
        )
        tool = _make_tool("my_tool")
        _, result = await _run(llm, tools=[tool])
        assert result.token_usage["prompt_tokens"] == 8
        assert result.token_usage["completion_tokens"] == 17
        assert result.token_usage["total_tokens"] == 25

    @pytest.mark.asyncio
    async def test_thinking_event_emitted_on_second_iteration(self):
        """두 번째 iteration에서 ProgressEvent("thinking")가 yield된다."""
        tc = _tc("my_tool")
        llm = _llm(
            _tool_calls([tc]),
            _text("최종 답변"),
        )
        tool = _make_tool("my_tool")
        progress, _ = await _run(llm, tools=[tool])
        assert any(e.event_type == "thinking" for e in progress)


# ── 예외 처리 경로 ────────────────────────────────────────────────────────────

class TestReactLoopEdgeCases:
    """malformed, ANY 재시도, max_iterations 등 예외 경로 검증."""

    @pytest.mark.asyncio
    async def test_malformed_function_call_triggers_none_mode(self):
        """malformed_function_call 시 bind_tools([])로 리셋 후 NONE 모드로 재 스트림한다."""
        llm = _llm(
            _malformed(),
            _text("텍스트 답변"),
        )
        _, result = await _run(llm)
        assert result.content == "텍스트 답변"
        assert result.llm_invoke_count == 2

    @pytest.mark.asyncio
    async def test_empty_content_with_tools_retries_with_any_mode(self):
        """content와 tool_calls 모두 없을 때 도구가 있으면 ANY 모드로 재시도 후 답변을 반환한다."""
        llm = _llm(
            _text(""),                # iter=0: 빈 content
            _text("재시도 후 답변"),  # iter=1: ANY 모드
        )
        tool = _make_tool("dummy")
        _, result = await _run(llm, tools=[tool])
        assert result.content == "재시도 후 답변"

    @pytest.mark.asyncio
    async def test_safe_msg_returned_when_any_retry_also_fails(self):
        """ANY 재시도 후에도 empty content이면 fallback 안전 메시지가 반환된다.

        fallback 문구는 token 이벤트로도 사용자에게 흘려보내 빈 화면을 방지한다.
        """
        llm = _llm(
            _text(""),
            _text(""),
        )
        tool = _make_tool("dummy")
        progress, result = await _run(llm, tools=[tool])
        assert result.content == _SAFE_MSG_FALLBACK
        assert result.finish_reason == "fallback"
        token_events = [e for e in progress if e.event_type == "token"]
        assert "".join(e.content for e in token_events) == _SAFE_MSG_FALLBACK

    @pytest.mark.asyncio
    async def test_empty_after_tools_retries_any_then_answers(self):
        """도구 사용 후 빈 응답이면(무음 종료 금지) ANY 재시도로 답을 받아 반환한다.

        used_tools 때문에 빈 LoopResult 로 무음 종료(token 0개 → 프론트 고착)하던
        버그의 회귀 방지. NONE 합성(도구 unbind → 텍스트 DSL 누출) 대신 ANY 재시도로
        도구 바인딩을 유지한다.
        """
        tc = _tc("tool_search")
        tool = _make_tool("tool_search")
        llm = _llm(
            _tool_calls([tc]),          # iter0: 도구 호출 → used_tools=True
            _text(""),                  # iter1: 빈 응답 → ANY 재시도
            _text("ANY 재시도 답변"),    # iter2(ANY): 답변
        )
        progress, result = await _run(llm, tools=[tool])
        assert result.content == "ANY 재시도 답변"
        token_events = [e for e in progress if e.event_type == "token"]
        assert "".join(e.content for e in token_events) == "ANY 재시도 답변"

    @pytest.mark.asyncio
    async def test_empty_after_tools_any_calls_tool_continues_multistep(self):
        """도구 사용 후 빈 응답 → ANY 재시도가 구조화된 tool_call 을 끌어내 멀티스텝을 잇는다.

        vLLM 다단계(1차 조회 후 빈 응답 → 후속 도구 호출)가 텍스트 DSL 누출 없이
        구조화로 진행되는지 검증.
        """
        tc1 = _tc("hr_find_employee")
        tc2 = _tc("tool_search")
        tools = [_make_tool("hr_find_employee"), _make_tool("tool_search")]
        llm = _llm(
            _tool_calls([tc1]),         # iter0: 1차 도구 → used_tools=True
            _text(""),                  # iter1: 빈 응답 → ANY 재시도
            _tool_calls([tc2]),         # iter2(ANY): 구조화된 후속 도구 호출
            _text("멀티스텝 최종 답변"),  # iter3: 답변
        )
        _, result = await _run(llm, tools=tools)
        assert result.content == "멀티스텝 최종 답변"
        assert len(result.tool_calls) == 2

    @pytest.mark.asyncio
    async def test_empty_after_tools_any_also_empty_yields_fallback(self):
        """도구 사용 후 빈 응답 + ANY 재시도도 빈 응답 → 안전 메시지 fallback (무음 종료 금지)."""
        tc = _tc("tool_search")
        tool = _make_tool("tool_search")
        llm = _llm(
            _tool_calls([tc]),
            _text(""),                  # iter1: 빈 응답 → ANY 재시도
            _text(""),                  # iter2(ANY): 또 빈 응답 → fallback
        )
        progress, result = await _run(llm, tools=[tool])
        assert result.content == _SAFE_MSG_FALLBACK
        assert result.finish_reason == "fallback"
        token_events = [e for e in progress if e.event_type == "token"]
        assert "".join(e.content for e in token_events) == _SAFE_MSG_FALLBACK

    @pytest.mark.asyncio
    async def test_max_iterations_forces_final_answer(self):
        """max_iterations 소진 후 tool_mode=NONE 으로 강제 스트림한다."""
        tc = _tc("tool1")
        tool = _make_tool("tool1")
        # max_iterations=2 → 2번 tool_call 후 강제 stream
        llm = _llm(
            _tool_calls([tc]),
            _tool_calls([tc]),
            _text("강제 최종 답변"),
        )
        _, result = await _run(llm, tools=[tool], max_iterations=2)
        assert result.content == "강제 최종 답변"
        assert result.llm_invoke_count == 3  # 2번 loop + 1번 강제

    @pytest.mark.asyncio
    async def test_max_iterations_with_empty_forced_stream_yields_fallback_token(self):
        """max_iterations 소진 + 강제 NONE 스트림도 빈 응답 → fallback 문구를 token 으로 흘린다."""
        tc = _tc("tool1")
        tool = _make_tool("tool1")
        # 모든 iter 가 tool_call → 강제 NONE 도 빈 응답
        llm = _llm(
            _tool_calls([tc]),
            _tool_calls([tc]),
            _text(""),  # 강제 NONE 빈 응답
        )
        progress, result = await _run(llm, tools=[tool], max_iterations=2)
        # fallback 문구가 LoopResult.content 와 token 이벤트 양쪽에 들어가야 한다
        assert "최대 반복 횟수" in result.content
        token_events = [e for e in progress if e.event_type == "token"]
        assert any("최대 반복 횟수" in e.content for e in token_events)

    @pytest.mark.asyncio
    async def test_oversized_response_triggers_trim(self):
        """30KB 초과 응답은 trim 되어 안내 문구가 붙는다 (재 invoke 하지 않음)."""
        from agent_arch.agents.react.loop import _MAX_RESPONSE_CHARS

        tc = _tc("tool1")
        tool = _make_tool("tool1")
        # 같은 문자 런은 collapse_repeated_runs 가 접으므로, 반복 없는 실제 본문 모양으로 만든다.
        unit = "가나다라마바사아자차카타파하 abcdefghij 0123456789.\n"
        huge_content = (unit * (_MAX_RESPONSE_CHARS // len(unit) + 2))[: _MAX_RESPONSE_CHARS + 1]
        llm = _llm(
            _tool_calls([tc]),
            _text(huge_content),
        )
        _, result = await _run(llm, tools=[tool])
        # trim 적용 → MAX_RESPONSE_CHARS 까지 잘리고 안내문구 추가
        assert len(result.content) <= _MAX_RESPONSE_CHARS + 100
        assert "응답이 너무 길어" in result.content

    @pytest.mark.asyncio
    async def test_repeated_run_collapsed_before_trim(self):
        """모델 반복 루프(표 구분선 대시 12만 자)는 trim 앞에서 접혀 표 본문이 살아남는다.

        2026-09-15 stg 실측 — 접지 않으면 30KB trim 이 대시만 남기고 안내문구를 붙였다.
        """
        from agent_arch.agents.react.loop import _MAX_RESPONSE_CHARS

        tc = _tc("tool1")
        tool = _make_tool("tool1")
        looped = (
            "| 항목 | 넣을 값 |\n| :---- | :" + "-" * 124_000
            + " |\n| 받는 메일 서버(IMAP) | mail.example.com |\n"
        )
        llm = _llm(_tool_calls([tc]), _text(looped))
        _, result = await _run(llm, tools=[tool])
        assert len(result.content) < _MAX_RESPONSE_CHARS
        assert "응답이 너무 길어" not in result.content
        assert "| :---- | :--- |" in result.content
        assert "| 받는 메일 서버(IMAP) | mail.example.com |" in result.content
