"""ReAct StateGraph 패리티 테스트 (Phase 1a).

동일한 mock 입력에 대해 ``react_loop_events`` (기존 커스텀 루프)와
``run_react_graph`` (LangGraph 구현)가 **동일한 LoopResult** 를 산출하는지
시나리오 매트릭스로 검증한다. 두 경로가 loop.py 헬퍼를 공유하므로 패리티는
구조적으로 보장되지만, 노드 분해/라우팅/상태 전파가 어긋나지 않았음을 확인한다.

각 시나리오는 두 경로에 **독립적인 fresh mock llm + messages + all_tools** 를 준다
(루프가 messages/all_tools 를 in-place 변형하고 mock stream 은 1회 소비되므로).
"""
from contextlib import nullcontext

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.core.config import settings
from agent_arch.llm.base import Message, StreamEvent, ToolCall
from agent_arch.agents.tools.base import ToolResult
from agent_arch.agents.react.loop import react_loop_events
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.agents.graphs.react_graph import run_react_graph, stream_react_graph


# ── stream 이벤트 팩토리 (llm.stream 1회 호출당 하나) ──────────────────────────

def stream_text(content, finish="stop", usage=None):
    async def _gen(*a, **k):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason=finish, usage=usage)
    return _gen


def stream_tool_calls(tool_calls, usage=None):
    async def _gen(*a, **k):
        yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
        yield StreamEvent(type="done", finish_reason="tool_calls", usage=usage)
    return _gen


def stream_done_only(finish="stop", usage=None):
    """텍스트/툴콜 없이 done 만 — 빈 응답/ malformed 시나리오용."""
    async def _gen(*a, **k):
        yield StreamEvent(type="done", finish_reason=finish, usage=usage)
    return _gen


def _make_llm(stream_specs):
    """stream_specs = [gen_factory, ...] — llm.stream 호출마다 다음 factory 사용."""
    llm = MagicMock()
    gens = list(stream_specs)

    def _next(*a, **k):
        return gens.pop(0)(*a, **k)

    llm.stream = _next
    llm.bind_tools = AsyncMock()
    return llm


def _make_tool(name, output="결과", success=True, view=None):
    tool = MagicMock()
    tool.name = name
    out = {"view": view} if view is not None else output
    tool.execute = AsyncMock(return_value=ToolResult(success=success, output=out))
    return tool


_U1 = {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}
_U2 = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


async def _run_parity(*, input, max_iterations, stream_specs_factory, tools_factory):
    """두 경로를 fresh 입력으로 각각 실행하고 (loop_result, graph_result) 반환."""
    # (1) 기존 루프
    llm_a = _make_llm(stream_specs_factory())
    msgs_a = [Message(role="user", content=input)]
    loop_res = None
    async for ev in react_loop_events(
        llm_a, input, msgs_a, tools_factory(), AsyncMock(return_value=[]), max_iterations
    ):
        if isinstance(ev, LoopResult):
            loop_res = ev
    # (2) 그래프
    llm_b = _make_llm(stream_specs_factory())
    msgs_b = [Message(role="user", content=input)]
    graph_res = await run_react_graph(
        llm_b, input, msgs_b, tools_factory(), AsyncMock(return_value=[]), max_iterations
    )
    return loop_res, graph_res


def _assert_parity(a: LoopResult, b: LoopResult):
    assert a is not None, "loop 경로가 LoopResult 를 내지 않음"
    assert b is not None, "graph 경로가 LoopResult 를 내지 않음"
    assert a.content == b.content, f"content 불일치:\n  loop={a.content!r}\n  graph={b.content!r}"
    assert a.finish_reason == b.finish_reason, f"finish_reason: {a.finish_reason} != {b.finish_reason}"
    assert a.llm_invoke_count == b.llm_invoke_count, f"invoke_count: {a.llm_invoke_count} != {b.llm_invoke_count}"
    assert a.token_usage == b.token_usage, f"token_usage: {a.token_usage} != {b.token_usage}"
    assert a.tool_calls == b.tool_calls, "tool_calls 불일치"
    assert a.tool_results == b.tool_results, "tool_results 불일치"
    assert a.pending_action == b.pending_action, "pending_action 불일치"
    assert a.pending_queue == b.pending_queue, "pending_queue 불일치"


# ── 시나리오 ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parity_plain_answer():
    """도구 없이 즉시 텍스트 답변."""
    a, b = await _run_parity(
        input="안녕",
        max_iterations=3,
        stream_specs_factory=lambda: [stream_text("안녕하세요, 답변입니다.", usage=_U1)],
        tools_factory=lambda: [],
    )
    _assert_parity(a, b)
    assert b.content == "안녕하세요, 답변입니다."
    assert b.finish_reason == "stop"
    assert b.llm_invoke_count == 1


@pytest.mark.asyncio
async def test_parity_repeated_run_collapsed_not_trimmed():
    """반복 루프 잔해(대시 12만 자)는 두 경로 모두 trim 앞에서 접혀 같은 결과를 낸다."""
    looped = "| 항목 | 값 |\n| :---- | :" + "-" * 124_000 + " |\n| 포트 | 993 |\n"
    a, b = await _run_parity(
        input="아웃룩 설정",
        max_iterations=3,
        stream_specs_factory=lambda: [
            stream_tool_calls([ToolCall(id="c1", name="kb", arguments={})], usage=_U1),
            stream_text(looped, usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("kb")],
    )
    _assert_parity(a, b)
    assert "응답이 너무 길어" not in b.content
    assert "| :---- | :--- |" in b.content
    assert "| 포트 | 993 |" in b.content


@pytest.mark.asyncio
async def test_parity_tool_then_answer():
    """tool_calls → 실행 → 다음 iteration 텍스트 답변."""
    a, b = await _run_parity(
        input="계산해줘",
        max_iterations=3,
        stream_specs_factory=lambda: [
            stream_tool_calls([ToolCall(id="tc1", name="calc", arguments={"x": 1})], usage=_U1),
            stream_text("계산 결과는 42입니다.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc", "결과")],
    )
    _assert_parity(a, b)
    assert b.finish_reason == "stop"
    assert b.llm_invoke_count == 2
    assert len(b.tool_calls) == 1 and len(b.tool_results) == 1


@pytest.mark.asyncio
async def test_parity_malformed_function_call():
    """malformed_function_call → tool_mode=NONE 재스트림."""
    a, b = await _run_parity(
        input="뭔가 요청",
        max_iterations=3,
        stream_specs_factory=lambda: [
            stream_done_only("malformed_function_call", usage=_U1),
            stream_text("복구된 일반 답변입니다.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )
    _assert_parity(a, b)
    assert b.content == "복구된 일반 답변입니다."
    assert b.llm_invoke_count == 2


@pytest.mark.asyncio
async def test_parity_empty_then_any_retry():
    """빈 응답 + 도구 있음 → ANY 재시도 → 답변."""
    a, b = await _run_parity(
        input="애매한 요청",
        max_iterations=3,
        stream_specs_factory=lambda: [
            stream_done_only("stop", usage=_U1),
            stream_text("ANY 재시도 후 답변.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )
    _assert_parity(a, b)
    assert b.content == "ANY 재시도 후 답변."
    assert b.llm_invoke_count == 2


@pytest.mark.asyncio
async def test_parity_fallback_no_tools():
    """빈 응답 + 도구 0개 → 안전 메시지 fallback."""
    a, b = await _run_parity(
        input="도구없는 요청",
        max_iterations=3,
        stream_specs_factory=lambda: [stream_done_only("stop", usage=_U1)],
        tools_factory=lambda: [],
    )
    _assert_parity(a, b)
    assert b.finish_reason == "fallback"
    assert b.llm_invoke_count == 1


@pytest.mark.asyncio
async def test_parity_max_iterations_force_answer():
    """max_iterations 소진 → force_answer(NONE) 강제 답변."""
    a, b = await _run_parity(
        input="반복 소진",
        max_iterations=1,
        stream_specs_factory=lambda: [
            stream_tool_calls([ToolCall(id="t1", name="calc", arguments={})], usage=_U1),
            stream_text("강제 최종 답변.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )
    _assert_parity(a, b)
    assert b.content == "강제 최종 답변."
    assert b.llm_invoke_count == 2


@pytest.mark.asyncio
async def test_hitl_deferred_pause_after_reads():
    """승인 도구는 즉시 pause 안 함(초안 누적) → LLM 이 나머지(조회/요약)를 마치고 최종 답변
    시점에 action_pending 으로 전환. 그래프 전용(레거시 loop 는 즉시 pause — 의도적 분기)."""
    with patch.object(settings, "use_langgraph_hitl", True):
        result = await run_react_graph(
            _make_llm([
                # iter0: 승인 도구(calendar_create) 호출 → 초안 누적(pause 안 함)
                stream_tool_calls([ToolCall(id="a1", name="calendar_create_event", arguments={"title": "회의"})], usage=_U1),
                # iter1: 텍스트 답변(finish) → answer 노드에서 누적 초안으로 action_pending
                stream_text("일정 초안을 준비했습니다.", usage=_U2),
            ]),
            "7월 3일 회의 잡아줘", [Message(role="user", content="7월 3일 회의 잡아줘")],
            [_make_tool("calendar_create_event", view={"kind": "draft", "title": "회의"})],
            AsyncMock(return_value=[]), 3,
        )
    assert result.finish_reason == "action_pending"
    assert result.pending_queue and result.pending_queue[0]["tool_name"] == "calendar_create_event"
    assert "confirm" not in result.pending_queue[0]["arguments"]  # 위조 방지
    assert result.content == "일정 초안을 준비했습니다."           # 조회/요약 완료 후 답변이 초안과 함께


# ── 스트리밍 패리티 (Phase 1b) ────────────────────────────────────────────────
#
# react_loop_events 와 stream_react_graph 가 동일한 ProgressEvent 시퀀스 + 최종
# LoopResult 를 내는지 검증한다. custom stream writer 방출·소비 계약이 노드 분해
# 후에도 토큰 순서를 보존함을 확인.

_STREAM_SCENARIOS = [
    ("plain", dict(
        input="안녕", max_iterations=3, hitl=False,
        stream_specs_factory=lambda: [stream_text("안녕하세요, 답변입니다.", usage=_U1)],
        tools_factory=lambda: [],
    )),
    ("tool_then_answer", dict(
        input="계산해줘", max_iterations=3, hitl=False,
        stream_specs_factory=lambda: [
            stream_tool_calls([ToolCall(id="tc1", name="calc", arguments={"x": 1})], usage=_U1),
            stream_text("계산 결과는 42입니다.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc", "결과")],
    )),
    ("malformed", dict(
        input="뭔가 요청", max_iterations=3, hitl=False,
        stream_specs_factory=lambda: [
            stream_done_only("malformed_function_call", usage=_U1),
            stream_text("복구된 일반 답변입니다.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )),
    ("empty_then_any_retry", dict(
        input="애매한 요청", max_iterations=3, hitl=False,
        stream_specs_factory=lambda: [
            stream_done_only("stop", usage=_U1),
            stream_text("ANY 재시도 후 답변.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )),
    ("fallback_no_tools", dict(
        input="도구없는 요청", max_iterations=3, hitl=False,
        stream_specs_factory=lambda: [stream_done_only("stop", usage=_U1)],
        tools_factory=lambda: [],
    )),
    ("max_iter_force", dict(
        input="반복 소진", max_iterations=1, hitl=False,
        stream_specs_factory=lambda: [
            stream_tool_calls([ToolCall(id="t1", name="calc", arguments={})], usage=_U1),
            stream_text("강제 최종 답변.", usage=_U2),
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )),
    # hitl_pending 은 파리티 제외 — 그래프는 승인 pause 를 지연(초안 누적 후 최종 답변에서),
    # 레거시 loop 는 즉시 pause 라 의도적으로 다름. 그래프 HITL 은 test_hitl_deferred_pause_after_reads 커버.
]


async def _collect_loop_stream(input, max_iterations, stream_specs, tools):
    progress, result = [], None
    async for ev in react_loop_events(
        _make_llm(stream_specs), input, [Message(role="user", content=input)],
        tools, AsyncMock(return_value=[]), max_iterations,
    ):
        if isinstance(ev, ProgressEvent):
            progress.append((ev.event_type, ev.content))
        elif isinstance(ev, LoopResult):
            result = ev
    return progress, result


async def _collect_graph_stream(input, max_iterations, stream_specs, tools):
    progress, result = [], None
    async for ev in stream_react_graph(
        _make_llm(stream_specs), input, [Message(role="user", content=input)],
        tools, AsyncMock(return_value=[]), max_iterations,
    ):
        if isinstance(ev, ProgressEvent):
            progress.append((ev.event_type, ev.content))
        elif isinstance(ev, LoopResult):
            result = ev
    return progress, result


@pytest.mark.parametrize("sid, spec", _STREAM_SCENARIOS, ids=[s[0] for s in _STREAM_SCENARIOS])
@pytest.mark.asyncio
async def test_stream_parity(sid, spec):
    ctx = patch.object(settings, "use_langgraph_hitl", True) if spec["hitl"] else nullcontext()
    with ctx:
        lp, lr = await _collect_loop_stream(
            spec["input"], spec["max_iterations"], spec["stream_specs_factory"](), spec["tools_factory"](),
        )
        gp, gr = await _collect_graph_stream(
            spec["input"], spec["max_iterations"], spec["stream_specs_factory"](), spec["tools_factory"](),
        )
    assert lp == gp, f"[{sid}] ProgressEvent 시퀀스 불일치:\n  loop={lp}\n  graph={gp}"
    _assert_parity(lr, gr)


# ── ContextVar 전파 실측 (Phase 1c 핵심 리스크) ───────────────────────────────
#
# tool_search 의 발견 도구 대기열(_discovered ContextVar)은 부모가 reset() 한 리스트
# 객체를 execute_tools 의 gather 자식 Task 들이 물려받아 in-place extend → 부모가 drain
# 하는 설계다. 그래프에선 reset=start 노드, drain=execute 노드로 갈라지므로, ContextVar
# 가 노드 경계를 넘어 공유되는지 실제 실행으로 확인한다(mock 으로는 안 드러남).

def _make_finder_and_discovered():
    """find_x(호출 시 discovered_api 를 _discovered 대기열에 enqueue) + discovered_api."""
    discovered = _make_tool("discovered_api", "발견결과")

    finder = MagicMock()
    finder.name = "find_x"

    async def _exec(**kw):
        # discover_tools 가 gather 자식에서 하는 일을 모사: 발견 도구를 대기열에 적재.
        from agent_arch.agents.tools.tool_search import _enqueue
        _enqueue([discovered])
        return ToolResult(success=True, output={"found": 1})

    finder.execute = AsyncMock(side_effect=_exec)
    return finder, discovered


def _discover_stream_specs():
    return [
        stream_tool_calls([ToolCall(id="f1", name="find_x", arguments={"query": "x"})], usage=_U1),
        stream_text("발견된 도구로 답변합니다.", usage=_U2),
    ]


@pytest.mark.asyncio
async def test_contextvar_discovered_tools_propagate_in_graph():
    """핵심 실측: start(reset)→execute(gather 자식 enqueue→drain) 가 _discovered 를 공유하는가."""
    finder, _ = _make_finder_and_discovered()
    tools = [finder]
    result = await run_react_graph(
        _make_llm(_discover_stream_specs()),
        "x 찾아줘", [Message(role="user", content="x")], tools,
        AsyncMock(return_value=[]), 3,
    )
    assert any(t.name == "discovered_api" for t in tools), (
        "ContextVar(_discovered)가 그래프 노드 경계를 넘지 못함 — "
        "발견 도구가 all_tools 에 머지 안 됨(멀티스텝 탐색 붕괴)"
    )
    assert result.content == "발견된 도구로 답변합니다."


@pytest.mark.asyncio
async def test_contextvar_discovered_tools_control_loop():
    """대조군: 기존 루프는 단일 컨텍스트라 항상 머지(테스트 셋업 자체 검증)."""
    finder, _ = _make_finder_and_discovered()
    tools = [finder]
    async for _ev in react_loop_events(
        _make_llm(_discover_stream_specs()),
        "x 찾아줘", [Message(role="user", content="x")], tools,
        AsyncMock(return_value=[]), 3,
    ):
        pass
    assert any(t.name == "discovered_api" for t in tools)


# ── 스텝 계측 (step_meta) ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_step_meta_populated():
    """그래프가 step_meta 를 채운다: 경로·타이밍·iterations·발견도구·finish_reason."""
    finder, _ = _make_finder_and_discovered()
    result = await run_react_graph(
        _make_llm(_discover_stream_specs()),
        "x 찾아줘", [Message(role="user", content="x")], [finder],
        AsyncMock(return_value=[]), 3,
    )
    sm = result.step_meta
    assert sm is not None, "step_meta 미채움"
    assert "call_model" in sm["node_trace"] and "execute" in sm["node_trace"]
    assert "answer" in sm["node_trace"]              # 텍스트 답변으로 종결
    assert sm["iterations"] >= 1
    assert sm["llm_invoke_count"] == 2
    assert "discovered_api" in sm["discovered_tools"]
    assert sm["finish_reason"] == "stop"
    assert isinstance(sm["node_timings"], dict) and "call_model" in sm["node_timings"]


@pytest.mark.asyncio
async def test_step_meta_retry_any_counted():
    """ANY 재시도 경로가 step_meta.retry_any 로 집계된다."""
    result = await run_react_graph(
        _make_llm([stream_done_only("stop", usage=_U1), stream_text("답변", usage=_U2)]),
        "애매", [Message(role="user", content="애매")], [_make_tool("calc")],
        AsyncMock(return_value=[]), 3,
    )
    assert result.step_meta["retry_any"] == 1
    assert "retry_any" in result.step_meta["node_trace"]


@pytest.mark.asyncio
async def test_step_meta_loop_path_is_none():
    """기존 루프(react_loop_events)는 step_meta 를 채우지 않는다(그래프 전용)."""
    res = None
    async for ev in react_loop_events(
        _make_llm([stream_text("답변", usage=_U1)]),
        "안녕", [Message(role="user", content="안녕")], [],
        AsyncMock(return_value=[]), 3,
    ):
        if isinstance(ev, LoopResult):
            res = ev
    assert res.step_meta is None


# ── 한국어 스트림 게이트 (영어 답변 보류 → 재생성본만 방출) ─────────────────────
#
# 한국어 가드는 스트림이 끝난 뒤에 도는데 토큰은 이미 프론트로 나가 버려, 화면엔 영어가
# 남고 DB 에만 한국어가 저장되던 실측 버그(2026-07-27)를 막는다. SSE 에 본문 교체
# 이벤트가 없으므로 "먼저 흘리고 나중에 고치기"가 불가능 — 앞부분을 붙들고 판정한다.

_ENGLISH_ANSWER = (
    "I have searched for recent issues related to Acme over the past 7 days. "
    "Here is the summary of the findings."
)
_KOREAN_ANSWER = "최근 7일간 회사 관련 이슈를 검색했습니다. 아래는 채널별 요약입니다."


async def _collect_with_llm(llm, input="최근 회사 이슈 조회해줘", target_language="ko"):
    progress, result = [], None
    async for ev in stream_react_graph(
        llm, input, [Message(role="user", content=input)], [],
        AsyncMock(return_value=[]), 3, target_language=target_language,
    ):
        if isinstance(ev, ProgressEvent):
            progress.append((ev.event_type, ev.content))
        elif isinstance(ev, LoopResult):
            result = ev
    return [c for t, c in progress if t == "token"], result


@pytest.mark.asyncio
async def test_english_answer_is_withheld_and_korean_emitted_once():
    """영어 답변은 한 토큰도 방출하지 않고, 재생성된 한국어만 나간다(화면=히스토리)."""
    llm = _make_llm([stream_text(_ENGLISH_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_KOREAN_ANSWER))

    tokens, result = await _collect_with_llm(llm)

    assert tokens == [_KOREAN_ANSWER], "영어 원문이 화면으로 새어 나갔다"
    assert result.content == _KOREAN_ANSWER
    assert _ENGLISH_ANSWER not in "".join(tokens)


@pytest.mark.asyncio
async def test_korean_answer_streams_through():
    """한국어 답변은 게이트를 그대로 통과한다(재생성 호출 없음)."""
    llm = _make_llm([stream_text(_KOREAN_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock()

    tokens, result = await _collect_with_llm(llm)

    assert tokens == [_KOREAN_ANSWER]
    assert result.content == _KOREAN_ANSWER
    llm.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_gate_disabled_streams_english_as_is():
    """KOREAN_ANSWER_GUARD=false 면 게이트도 비활성 — 기존 동작(그대로 스트리밍)."""
    llm = _make_llm([stream_text(_ENGLISH_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock()

    with patch.object(settings, "korean_answer_guard", False):
        tokens, result = await _collect_with_llm(llm)

    assert tokens == [_ENGLISH_ANSWER]
    assert result.content == _ENGLISH_ANSWER


@pytest.mark.asyncio
async def test_withheld_stream_falls_back_to_original_when_regeneration_fails():
    """재생성이 실패하면 보류분을 원문 그대로라도 내보낸다(빈 답변 금지)."""
    llm = _make_llm([stream_text(_ENGLISH_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock(side_effect=RuntimeError("provider down"))

    tokens, result = await _collect_with_llm(llm)

    assert tokens == [_ENGLISH_ANSWER]
    assert result.content == _ENGLISH_ANSWER


# ── malformed 복구가 빈 응답으로 끝나지 않는다 ────────────────────────────────────

def stream_raises():
    """스트림 도중 예외 — 재시도 자체가 실패하는 시나리오."""
    async def _gen(*a, **k):
        raise RuntimeError("provider 폭발")
        yield  # pragma: no cover  (async generator 로 만들기 위한 unreachable yield)
    return _gen


@pytest.mark.asyncio
async def test_parity_malformed_then_empty_retry_emits_fallback():
    """malformed → 재시도도 빈 응답 → **폴백 문구**로 종결한다.

    회귀(실측 2026-08-18): 되묻기에 "1번"/"3번" 으로 답한 턴에서 malformed 가 나고 재시도가
    텍스트를 못 내면 토큰이 하나도 방출되지 않고 content 도 빈 문자열이라, SSE 가 내용 없이
    끝나 화면이 무한 로딩으로 남았다. 형제 경로(_force_answer)에는 이 가드가 있었다.
    """
    from agent_arch.agents.react.loop import _SAFE_MSG_MALFORMED

    a, b = await _run_parity(
        input="1번",
        max_iterations=3,
        stream_specs_factory=lambda: [
            stream_done_only("malformed_function_call", usage=_U1),
            stream_done_only("stop", usage=_U2),          # 재시도가 텍스트를 못 냄
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )
    _assert_parity(a, b)
    assert b.content == _SAFE_MSG_MALFORMED      # 빈 응답이 아니다
    assert b.content.strip()


@pytest.mark.asyncio
async def test_parity_malformed_then_retry_raises_emits_fallback():
    """malformed → 재시도가 예외 → 예외를 삼키고 폴백 문구로 종결한다.

    예외를 밖으로 흘리면 스트림이 종료 이벤트 없이 끊겨 역시 무한 로딩이 된다.
    """
    from agent_arch.agents.react.loop import _SAFE_MSG_MALFORMED

    a, b = await _run_parity(
        input="3번",
        max_iterations=3,
        stream_specs_factory=lambda: [
            stream_done_only("malformed_function_call", usage=_U1),
            stream_raises(),
        ],
        tools_factory=lambda: [_make_tool("calc")],
    )
    _assert_parity(a, b)
    assert b.content == _SAFE_MSG_MALFORMED
    assert b.finish_reason == "error"


@pytest.mark.asyncio
async def test_malformed_fallback_is_streamed_to_user():
    """폴백 문구가 **토큰으로도 방출**돼야 한다 — 화면에 아무것도 안 나가는 게 원인이었다."""
    from agent_arch.agents.react.loop import _SAFE_MSG_MALFORMED

    tools = [_make_tool("calc")]
    llm = _make_llm([
        stream_done_only("malformed_function_call", usage=_U1),
        stream_done_only("stop", usage=_U2),
    ])
    tokens = []
    async for ev in stream_react_graph(
        llm, "1번", [Message(role="user", content="1번")], tools,
        AsyncMock(return_value=[]), 3,
    ):
        if isinstance(ev, ProgressEvent) and ev.event_type == "token":
            tokens.append(ev.content)
    assert "".join(tokens) == _SAFE_MSG_MALFORMED


# ── 우회 노드의 이름 가드 (2026-09-03) ────────────────────────────────────────
# _malformed_node / _force_answer_node 는 토큰을 직접 방출하고 최종 답변 노드를 거치지
# 않아, 한국어 가드도 이름 가드도 안 돌았다. 특히 _force_answer_node 는 max_iterations
# 소진 경로 — **도구를 여러 번 돈 긴 메신저 턴이 정확히 여기로 떨어진다.** 위험이 가장
# 높은 턴이 유일하게 무방비였다.

_BARE_AUTHOR_TOOL_OUT = {"posts": [{"author": "jdoe", "message": "확인 부탁드립니다"}]}
_FABRICATED = "김민수 님이 확인을 요청했습니다."


async def _collect_graph(llm, tools, max_iterations, input="안 읽은 메시지 정리해줘"):
    progress, result = [], None
    async for ev in stream_react_graph(
        llm, input, [Message(role="user", content=input)], tools,
        AsyncMock(return_value=[]), max_iterations,
    ):
        if isinstance(ev, ProgressEvent):
            progress.append((ev.event_type, ev.content))
        elif isinstance(ev, LoopResult):
            result = ev
    return [c for t, c in progress if t == "token"], result


@pytest.mark.asyncio
async def test_force_answer_node_applies_name_guard():
    """max_iterations 소진 경로도 지어낸 이름을 마스킹해야 한다.

    도구가 맨 ``jdoe`` 를 주고 강제 답변이 "김민수 님"을 쓴다. 재생성도 같은 이름을
    반복하면 마스킹으로 끝난다 — 틀린 이름은 남의 발언을 오귀속하므로 원문 유지보다 낫다.
    """
    llm = _make_llm([
        stream_tool_calls([ToolCall(id="t1", name="unread", arguments={})], usage=_U1),
        stream_text(_FABRICATED, usage=_U2),
    ])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_FABRICATED))  # 재생성도 재범

    tokens, result = await _collect_graph(
        llm, [_make_tool("unread", output=_BARE_AUTHOR_TOOL_OUT)], max_iterations=1
    )

    assert "이름미확인" in result.content, "force_answer 경로에서 이름 가드가 안 돌았다"
    assert "김민수" not in result.content
    # 화면 = 히스토리: 방출된 토큰을 이으면 저장본과 정확히 같아야 한다.
    assert "".join(tokens) == result.content


@pytest.mark.asyncio
async def test_malformed_node_applies_name_guard():
    """malformed 복구 경로도 마찬가지다.

    도구를 한 번 **실행한 뒤** malformed 가 나야 근거에 맨 username 이 들어간다.
    (첫 호출부터 malformed 면 도구 결과가 없어 위험 신호도 없다 — 그때 무장하지 않는 것이
    올바른 동작이다.)
    """
    llm = _make_llm([
        stream_tool_calls([ToolCall(id="t1", name="unread", arguments={})], usage=_U1),
        stream_done_only("malformed_function_call", usage=_U1),
        stream_text(_FABRICATED, usage=_U2),
    ])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_FABRICATED))

    tokens, result = await _collect_graph(
        llm, [_make_tool("unread", output=_BARE_AUTHOR_TOOL_OUT)], max_iterations=3
    )

    assert "이름미확인" in result.content
    assert "".join(tokens) == result.content


@pytest.mark.asyncio
async def test_force_answer_node_streams_clean_answer_unchanged():
    """위험 신호가 없으면 붙들지 않는다 — 오탐은 체감 지연으로 돌아온다."""
    clean = "최근 7일간 회사 관련 이슈를 정리했습니다."
    llm = _make_llm([
        stream_tool_calls([ToolCall(id="t1", name="unread", arguments={})], usage=_U1),
        stream_text(clean, usage=_U2),
    ])
    llm.invoke = AsyncMock(return_value=MagicMock(content=clean))

    tokens, result = await _collect_graph(
        llm, [_make_tool("unread", output={"posts": [{"author": "김민서"}]})], max_iterations=1
    )

    assert result.content == clean
    assert "".join(tokens) == clean


@pytest.mark.asyncio
async def test_name_guard_observability_lands_in_step_meta():
    """가드 결과가 step_meta 로 나가야 한다 — 없으면 재발을 알 방법이 없다.

    `armed=0` 이 "사고 없음"이 아니라 "가드가 죽음"일 수 있으므로, 무장 여부 자체가
    기록돼야 한다. 이 값이 agent.messages.metadata 와 LangSmith run 에 함께 저장된다.
    """
    llm = _make_llm([
        stream_tool_calls([ToolCall(id="t1", name="unread", arguments={})], usage=_U1),
        stream_text(_FABRICATED, usage=_U2),
    ])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_FABRICATED))
    _, result = await _collect_graph(
        llm, [_make_tool("unread", output=_BARE_AUTHOR_TOOL_OUT)], max_iterations=1
    )
    ng = result.step_meta["name_guard"]
    assert ng["armed"] is True
    assert ng["detected"] == ["김민수"]
    assert ng["regenerated"] is True
    assert ng["masked"] == ["김민수"]


@pytest.mark.asyncio
async def test_name_guard_records_disarmed_turns_too():
    """무장하지 않은 턴도 armed=False 로 남긴다 — 무장률 자체가 관측 대상이다."""
    clean = "최근 이슈를 정리했습니다."
    llm = _make_llm([
        stream_tool_calls([ToolCall(id="t1", name="unread", arguments={})], usage=_U1),
        stream_text(clean, usage=_U2),
    ])
    _, result = await _collect_graph(
        llm, [_make_tool("unread", output={"posts": [{"author": "김민서"}]})], max_iterations=1
    )
    assert result.step_meta["name_guard"] == {"armed": False}


# ── 언어 미러 게이트 (목표 언어가 한국어가 아닌 턴) ────────────────────────────
#
# 게이트·재생성은 "한국어냐"가 아니라 "이 턴의 목표 언어냐"로 판정한다. 영어로 물어본
# 사용자에게 한국어 답변을 되돌려주던 것이 원 증상이고, 중국어·일본어는 라틴 문자가 없어
# 가드가 아예 안 붙던 비대칭(실측 2026-09-09)이 나머지 절반이었다.

_CHINESE_ANSWER = "以下是最近7天与Acme相关的问题摘要。如需详细信息，请告知。"


@pytest.mark.asyncio
async def test_english_target_streams_english_through():
    """영어 목표 턴에서 영어 답변은 그대로 흐른다 — 재생성 호출 없음."""
    llm = _make_llm([stream_text(_ENGLISH_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock()

    tokens, result = await _collect_with_llm(llm, input="show me recent issues", target_language="en")

    assert tokens == [_ENGLISH_ANSWER]
    assert result.content == _ENGLISH_ANSWER
    llm.invoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_english_target_withholds_korean_and_emits_regenerated_english():
    """영어 목표인데 한국어로 답하면 보류 → 영어 재생성본만 방출(화면=히스토리)."""
    llm = _make_llm([stream_text(_KOREAN_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_ENGLISH_ANSWER))

    tokens, result = await _collect_with_llm(llm, input="show me recent issues", target_language="en")

    assert tokens == [_ENGLISH_ANSWER], "한국어 원문이 화면으로 새어 나갔다"
    assert result.content == _ENGLISH_ANSWER
    assert _KOREAN_ANSWER not in "".join(tokens)


@pytest.mark.asyncio
async def test_chinese_target_regenerates_korean_answer():
    """중국어 목표 턴 — 한자에는 라틴 문자가 없어 예전엔 판정 자체가 건너뛰어졌다."""
    llm = _make_llm([stream_text(_KOREAN_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_CHINESE_ANSWER))

    tokens, result = await _collect_with_llm(llm, input="给我看最近的问题", target_language="zh")

    assert tokens == [_CHINESE_ANSWER]
    assert result.content == _CHINESE_ANSWER


@pytest.mark.asyncio
async def test_reask_instruction_is_written_in_target_language():
    """재작성 지시문도 목표 언어로 나간다 — 한국어 지시를 모델이 미러링하며 무시했다."""
    llm = _make_llm([stream_text(_KOREAN_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_ENGLISH_ANSWER))

    await _collect_with_llm(llm, input="show me recent issues", target_language="en")

    reask = llm.invoke.await_args.args[0][-1]
    assert reask.role == "user"
    assert "rewrite it in English" in reask.content


@pytest.mark.asyncio
async def test_language_target_defaults_to_korean():
    """target_language 를 안 넘기는 호출자는 이관 전과 동일하게 동작한다."""
    llm = _make_llm([stream_text(_ENGLISH_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_KOREAN_ANSWER))

    progress, result = [], None
    async for ev in stream_react_graph(
        llm, "최근 이슈", [Message(role="user", content="최근 이슈")], [],
        AsyncMock(return_value=[]), 3,
    ):
        if isinstance(ev, ProgressEvent) and ev.event_type == "token":
            progress.append(ev.content)
        elif isinstance(ev, LoopResult):
            result = ev

    assert progress == [_KOREAN_ANSWER]
    assert result.content == _KOREAN_ANSWER


@pytest.mark.asyncio
async def test_non_streaming_run_honors_target_language():
    """비스트리밍 run_react_graph 도 같은 목표 언어로 판정·재생성한다."""
    llm = _make_llm([stream_text(_KOREAN_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock(return_value=MagicMock(content=_ENGLISH_ANSWER))

    result = await run_react_graph(
        llm, "show me recent issues", [Message(role="user", content="show me recent issues")],
        [], AsyncMock(return_value=[]), 3, target_language="en",
    )

    assert result.content == _ENGLISH_ANSWER


@pytest.mark.asyncio
async def test_guard_disabled_leaves_other_language_answer_alone():
    """KOREAN_ANSWER_GUARD=false 면 목표 언어와 무관하게 원문 그대로(응급 레버)."""
    llm = _make_llm([stream_text(_KOREAN_ANSWER, usage=_U1)])
    llm.invoke = AsyncMock()

    with patch.object(settings, "korean_answer_guard", False):
        tokens, result = await _collect_with_llm(
            llm, input="show me recent issues", target_language="en")

    assert tokens == [_KOREAN_ANSWER]
    assert result.content == _KOREAN_ANSWER
    llm.invoke.assert_not_awaited()


# ── 답변 검증 노드(react/verify) ──────────────────────────────────────────────
#
# 검증기 LLM 은 부르지 않고 run_verifier 를 갈아 끼운다. 기본(off)에서는 노드가 라우팅되지 않아
# 위 패리티가 그대로 성립한다. shadow 는 기록만, on 은 보강 루프가 한 번 더 돈다.

from agent_arch.agents.react import verify as _V


def _fake_verifier(verdicts, calls):
    async def _run(user_request, draft, state):
        calls.append({"request": user_request, "draft": draft, "tools": [c["name"] for c in state.get("collected_tool_calls") or []]})
        v = verdicts.pop(0)
        return v, _V.collect_facts(state, draft), {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    return _run


_REVISE = {"verdict": "revise", "issues": ["요청 대상과 결과가 어긋남"], "action": {"type": "search", "detail": "industrial bracket market"}}
_PASS = {"verdict": "pass", "issues": [], "action": {"type": "none", "detail": ""}}


@pytest.mark.asyncio
async def test_verify_off_by_default_no_node(monkeypatch):
    monkeypatch.setattr(settings, "answer_verify_mode", "off", raising=False)
    calls = []
    monkeypatch.setattr(_V, "run_verifier", _fake_verifier([_REVISE], calls))
    res = await run_react_graph(_make_llm([stream_text("답", usage=_U1)]), "q", [Message(role="user", content="q")],
                                [_make_tool("brave_web_search")], AsyncMock(return_value=[]), 3)
    assert res.content == "답" and res.llm_invoke_count == 1 and calls == [] and res.step_meta["verify"] == []


@pytest.mark.asyncio
async def test_verify_shadow_records_but_does_not_change_flow(monkeypatch):
    monkeypatch.setattr(settings, "answer_verify_mode", "shadow", raising=False)
    calls = []
    monkeypatch.setattr(_V, "run_verifier", _fake_verifier([_REVISE], calls))
    msgs = [Message(role="user", content="브라켓 시장 동향")]
    res = await run_react_graph(_make_llm([
        stream_tool_calls([ToolCall(id="t1", name="brave_web_search", arguments={"query": "브라켓"})], usage=_U1),
        stream_text("치과·TV·산업용 브라켓 요약", usage=_U2),
    ]), "브라켓 시장 동향", msgs, [_make_tool("brave_web_search", {"kind": "results", "results": []})], AsyncMock(return_value=[]), 4)
    assert res.content == "치과·TV·산업용 브라켓 요약"                     # 초안 그대로 나간다
    assert res.llm_invoke_count == 3                                          # 스트림 2 + 검증 1(비용 계상)
    assert calls and calls[0]["draft"] == res.content and calls[0]["tools"] == ["brave_web_search"]
    v = res.step_meta["verify"]
    assert len(v) == 1 and v[0]["mode"] == "shadow" and v[0]["verdict"] == "revise" and v[0]["action"]["type"] == "search"
    assert v[0]["facts"]["web_searches"] == 1 and v[0]["facts"]["empty_searches"] == 1
    assert not any(getattr(m, "name", None) == _V.FEEDBACK_NAME for m in msgs)  # 보강 지시는 안 붙는다
    assert res.token_usage["total_tokens"] == _U1["total_tokens"] + _U2["total_tokens"] + 2


@pytest.mark.asyncio
async def test_verify_on_revise_loops_once_then_answers(monkeypatch):
    monkeypatch.setattr(settings, "answer_verify_mode", "on", raising=False)
    monkeypatch.setattr(settings, "answer_verify_max_rounds", 1, raising=False)
    calls = []
    monkeypatch.setattr(_V, "run_verifier", _fake_verifier([_REVISE, _REVISE], calls))   # 두 번째도 revise 지만 상한 1
    msgs = [Message(role="user", content="산업용 브라켓 시장")]
    progress, res = [], None
    async for ev in stream_react_graph(_make_llm([
        stream_tool_calls([ToolCall(id="t1", name="brave_web_search", arguments={"query": "브라켓"})], usage=_U1),
        stream_text("초안: 치과 브라켓 위주", usage=_U2),
        stream_tool_calls([ToolCall(id="t2", name="brave_web_search", arguments={"query": "industrial bracket market"})], usage=_U1),
        stream_text("보강본: 산업용 브라켓 시장 요약", usage=_U2),
    ]), "산업용 브라켓 시장", msgs, [_make_tool("brave_web_search", {"kind": "results", "results": []})], AsyncMock(return_value=[]), 6):
        if isinstance(ev, ProgressEvent):
            progress.append((ev.event_type, ev.content))
        elif isinstance(ev, LoopResult):
            res = ev
    assert res.content == "보강본: 산업용 브라켓 시장 요약"
    assert len(res.tool_calls) == 2 and res.llm_invoke_count == 6            # 스트림 4 + 검증 2
    tokens = "".join(c for t, c in progress if t == "token")
    assert "초안: 치과" not in tokens and tokens == res.content             # 초안은 화면에 안 나간다(on 모드 토큰 보류)
    assert any("검증하고 보강" in (c or "") for t, c in progress if t == "thinking")
    fb = [m for m in msgs if getattr(m, "name", None) == _V.FEEDBACK_NAME]
    assert len(fb) == 1 and "industrial bracket market" in fb[0].content and fb[0].role == "user"
    i = msgs.index(fb[0]); assert msgs[i - 1].role == "assistant" and msgs[i - 1].content == "초안: 치과 브라켓 위주"
    v = res.step_meta["verify"]
    assert [x["round"] for x in v] == [1, 2] and [x["verdict"] for x in v] == ["revise", "revise"]


@pytest.mark.asyncio
async def test_verify_on_pass_and_verifier_failure_never_block(monkeypatch):
    monkeypatch.setattr(settings, "answer_verify_mode", "on", raising=False)
    calls = []
    monkeypatch.setattr(_V, "run_verifier", _fake_verifier([_PASS], calls))
    res = await run_react_graph(_make_llm([stream_text("바로 통과", usage=_U1)]), "q", [Message(role="user", content="q")],
                                [_make_tool("brave_web_search")], AsyncMock(return_value=[]), 3)
    assert res.content == "바로 통과" and res.step_meta["verify"][0]["verdict"] == "pass" and res.llm_invoke_count == 2

    async def boom(*a, **k):
        raise RuntimeError("verifier down")
    monkeypatch.setattr(_V, "run_verifier", boom)
    res = await run_react_graph(_make_llm([stream_text("그래도 답", usage=_U1)]), "q", [Message(role="user", content="q")],
                                [_make_tool("brave_web_search")], AsyncMock(return_value=[]), 3)
    assert res.content == "그래도 답" and res.llm_invoke_count == 1
    assert res.step_meta["verify"][0]["verdict"] == "pass" and "verifier down" in res.step_meta["verify"][0]["error"]


@pytest.mark.asyncio
async def test_verify_not_applied_to_internal_turn_without_tools(monkeypatch):
    monkeypatch.setattr(settings, "answer_verify_mode", "on", raising=False)
    calls = []
    monkeypatch.setattr(_V, "run_verifier", _fake_verifier([_REVISE], calls))
    res = await run_react_graph(_make_llm([stream_text("안녕하세요", usage=_U1)]), "안녕", [Message(role="user", content="안녕")],
                                [_make_tool("calendar_list")], AsyncMock(return_value=[]), 3)
    assert res.content == "안녕하세요" and calls == [] and res.step_meta["verify"] == []


# ── 턴 예산(react/budget) — 두 경로가 같은 실행기를 지나므로 패리티가 유지된다 ──────────

@pytest.mark.asyncio
async def test_parity_duplicate_web_call_blocked_once(monkeypatch):
    monkeypatch.setattr(settings, "web_max_calls_per_turn", 8, raising=False)
    tools = []
    def tools_factory():
        t = _make_tool("brave_web_search", {"kind": "results", "results": []}); tools.append(t); return [t]
    a, b = await _run_parity(
        input="Kim 관련된거 찾아봐",
        max_iterations=5,
        stream_specs_factory=lambda: [
            stream_tool_calls([ToolCall(id="t1", name="brave_web_search", arguments={"query": "Kim PSE research"})], usage=_U1),
            stream_tool_calls([ToolCall(id="t2", name="brave_web_search", arguments={"query": "kim pse  research"})], usage=_U1),   # 같은 검색 반복
            stream_text("지금 근거로 답합니다.", usage=_U2),
        ],
        tools_factory=tools_factory,
    )
    _assert_parity(a, b)
    assert len(b.tool_results) == 2 and b.tool_results[1]["content"].startswith("[ERROR] duplicate_call:")
    assert all(t.execute.await_count == 1 for t in tools)                                            # 경로마다 실제 실행 1회
    assert b.step_meta["web_budget"] == {"web_calls": 1, "duplicates": 1, "capped": 0, "cap": 8}
# ── 같은 도구 실패 반복 차단 (2026-09-14 회귀) ────────────────────────────────
#
# 운영에서 create_file 이 **13회 연속** `filename 이 필요합니다` 로 실패하며 5분을 돌다
# 반복 상한으로 끝났다. 모델이 인자를 못 만드는 상황은 더 돈다고 풀리지 않고, 그 시간
# 내내 사용자에게는 아무 신호도 없다.

class _TC:
    """execute_tools 결과의 ToolCall 자리 — name 만 쓴다."""

    def __init__(self, name: str) -> None:
        self.name = name


def test_failure_signature_groups_by_tool_and_reason():
    from agent_arch.agents.graphs.react_graph import _failure_signature

    sig = _failure_signature([(_TC("create_file"), "[ERROR] filename 이 필요합니다.")])
    assert sig and sig.startswith("create_file:[ERROR] filename")


def test_failure_signature_is_none_when_all_succeeded():
    from agent_arch.agents.graphs.react_graph import _failure_signature

    assert _failure_signature([(_TC("create_file"), '{"filename": "a.docx"}')]) is None


def test_same_failure_repeated_breaks_the_loop():
    """상한에 닿으면 iteration 이 남아 있어도 답변으로 넘어간다."""
    from agent_arch.agents.graphs.react_graph import _MAX_SAME_FAILURES, _route_loop

    state = {
        "iteration": 0, "max_iterations": 10,
        "fail_signature": "create_file:[ERROR] filename 이 필요합니다.",
        "fail_streak": _MAX_SAME_FAILURES,
    }
    assert _route_loop(state) == "force_answer"


def test_different_failures_keep_looping():
    """실패해도 이유가 달라지고 있으면 모델이 고쳐 가는 중이다 — 끊지 않는다."""
    from agent_arch.agents.graphs.react_graph import _route_loop

    state = {
        "iteration": 0, "max_iterations": 10,
        "fail_signature": "create_file:[ERROR] docx 는 본문이 필요합니다.",
        "fail_streak": 1,
    }
    assert _route_loop(state) == "call_model"


# ── thread-first: "이 스레드" 를 그 스레드로 고정 (2026-09-14 회귀) ───────────

def test_thread_root_is_filled_by_the_system_not_the_model(monkeypatch):
    """root_id 는 봇이 이미 아는 값이다 — 모델이 다른 post_id 를 넣으면 덮어쓴다.

    프롬프트로 알려줘도 모델이 엉뚱한 id 를 넣으면 다시 다른 스레드를 읽는다.
    """
    from agent_arch.agents.context import set_user_context
    from agent_arch.agents.graphs import react_graph as rg

    set_user_context({"sub": "u1", "origin": {"channel_id": "ch-1", "root_id": "root-9"}})
    try:
        assert rg._origin_root_id() == "root-9"
    finally:
        set_user_context({})


def test_origin_root_id_is_none_outside_messenger():
    """브라우저 패널에는 origin 이 없다 — 강제도 주입도 일어나지 않아야 한다."""
    from agent_arch.agents.context import set_user_context
    from agent_arch.agents.graphs import react_graph as rg

    set_user_context({"sub": "u1"})
    try:
        assert rg._origin_root_id() is None
    finally:
        set_user_context({})


# ── 0건이면 범위를 푼다 (조회 범위 고정의 안전망) ────────────────────────────

def test_empty_search_is_detected():
    from agent_arch.agents.graphs.react_graph import _is_empty_search

    assert _is_empty_search('{"results": [], "count": 0}')
    assert _is_empty_search('{"count": 0}')
    assert not _is_empty_search('{"results": [{"text": "hit"}], "count": 1}')


def test_unparseable_result_is_not_treated_as_empty():
    """판정을 못 하면 재검색하지 않는다 — 섣부른 재호출은 비용·지연만 늘린다."""
    from agent_arch.agents.graphs.react_graph import _is_empty_search

    assert not _is_empty_search("[ERROR] 권한이 없습니다")
    assert not _is_empty_search("")


@pytest.mark.asyncio
async def test_scoped_search_retries_without_channel_filter(monkeypatch):
    """현재 채널로 묶어 0건이면 범위를 풀어 한 번 더 부른다."""
    from agent_arch.agents.graphs import react_graph as rg

    class _TC:
        def __init__(self):
            self.id = "call-1"
            self.name = "mattermost_search_knowledge"
            self.arguments = {"query": "PDF", "channel_ids": ["ch-1"]}

    tc = _TC()
    calls = []

    async def _fake_execute(tool_calls, all_tools):  # noqa: ARG001
        calls.append(dict(tool_calls[0].arguments))
        return [(tool_calls[0], '{"results": [{"text": "전사에서 찾은 결과"}], "count": 1}')]

    monkeypatch.setattr(rg, "execute_tools", _fake_execute)

    out = await rg._widen_empty_scoped(
        [(tc, '{"results": [], "count": 0}')], {"call-1"}, [])

    assert "channel_ids" not in calls[0]          # 범위를 풀어서 다시 불렀다
    assert "전사에서 찾은 결과" in out[0][1]


@pytest.mark.asyncio
async def test_non_empty_scoped_search_is_left_alone(monkeypatch):
    from agent_arch.agents.graphs import react_graph as rg

    class _TC:
        id = "call-1"
        name = "mattermost_search_knowledge"
        arguments = {"channel_ids": ["ch-1"]}

    async def _boom(*a, **k):  # noqa: ARG001
        raise AssertionError("결과가 있는데 재검색하면 안 된다")

    monkeypatch.setattr(rg, "execute_tools", _boom)

    out = await rg._widen_empty_scoped(
        [(_TC(), '{"results": [{"t": 1}], "count": 1}')], {"call-1"}, [])

    assert out[0][1].startswith('{"results"')


# ── 가이드 참고 링크 각주(코드 소유) ─────────────────────────────────────────

_GUIDE_GROUPS = [{"category": "General", "category_top_score": 0.8, "chunks": [
    {"title": "고객 문의 접수 및 대응 - 회원 문의",
     "content": "회원 문의 폼( https://example.com/main/support/inquiry/write )을 이용하세요.",
     "score": 0.8, "permission_required": "system_user", "permission_blocked": False}]}]
_GUIDE_FOOTER = "\n\n🔗 참고 링크\n- https://example.com/main/support/inquiry/write"


@pytest.mark.asyncio
async def test_stream_replaces_model_reference_section_with_code_footer():
    """모델이 지어낸 '참고 링크' 섹션은 정화가 지우고, 도구 결과 청크 본문의 URL 로 코드 각주가 붙는다.
    각주는 웹 출처 각주처럼 마지막 token 으로도 나가고(화면), final_result.content 에도 붙는다(히스토리)."""
    fake = "**참고 링크**\n- [고객 문의 접수 및 대응 - 회원 문의](https://docs.example.com/652vpj8/652vpj8)\n"
    tc = ToolCall(id="t1", name="search_user_guide_kb", arguments={"query": "고객 문의"})
    llm = _make_llm([stream_tool_calls([tc], usage=_U1), stream_text("고객 문의는 회원 문의 폼으로 접수하고 대응합니다.\n\n" + fake, usage=_U2)])
    events, result = [], None
    async for ev in stream_react_graph(llm, "고객 문의 어떻게 해?", [Message(role="user", content="고객 문의 어떻게 해?")],
                                       [_make_tool("search_user_guide_kb", output=_GUIDE_GROUPS)], AsyncMock(return_value=[]), 3, target_language="ko"):
        if isinstance(ev, ProgressEvent):
            events.append(ev)
        else:
            result = ev
    tokens = [e.content for e in events if e.event_type == "token"]
    assert tokens[-1] == _GUIDE_FOOTER
    assert result.content == "고객 문의는 회원 문의 폼으로 접수하고 대응합니다." + _GUIDE_FOOTER
    assert "docs.example.com" not in result.content
    from agent_arch.utils.sanitize import sanitize_tool_names
    assert sanitize_tool_names("".join(tokens)).rstrip() == result.content       # 화면(정화 후) == 저장본


@pytest.mark.asyncio
async def test_run_appends_guide_footer_only_when_chunks_have_urls():
    tc = ToolCall(id="t1", name="search_user_guide_kb", arguments={"query": "q"})
    no_url = [{"category": "DMP", "category_top_score": 0.8, "chunks": [
        {"title": "DMP 표시 언어 가이드 - 개요", "content": "설정 > 언어에서 바꿉니다.", "score": 0.8,
         "permission_required": "system_user", "permission_blocked": False}]}]
    answer = "고객 문의 접수 대응 안내입니다.\n\n🔗 참고 링크\n- [DMP 표시 언어 가이드](None)\n"
    for groups, footer in ((_GUIDE_GROUPS, _GUIDE_FOOTER), (no_url, "")):
        llm = _make_llm([stream_tool_calls([tc], usage=_U1), stream_text(answer, usage=_U2)])
        res = await run_react_graph(llm, "q", [Message(role="user", content="q")], [_make_tool("search_user_guide_kb", output=groups)],
                                    AsyncMock(return_value=[]), 3)
        assert res.content == "고객 문의 접수 대응 안내입니다." + footer if footer else "참고 링크" not in res.content and "(None)" not in res.content


# ── 근거 없는 가이드 링크 차단(화면 = 저장본) ─────────────────────────────────

_FORM_URL = "https://example.com/main/support/inquiry/write"
_FAKE_GUIDE_URL = "https://example.com/user-guide/agent-deep-dive"   # 실측 환각 — 진짜와 같은 호스트다


def _screen_text(events):
    """orchestrator 가 하는 일을 그대로 재현한다 — link_policy 를 정화기에 먹이고 토큰을 정화."""
    from agent_arch.agents.react.link_policy import TurnLinkPolicy
    from agent_arch.utils.sanitize import StreamSanitizer

    policy = TurnLinkPolicy()
    sanitizer = StreamSanitizer(link_permits=policy.permits)
    out = []
    for ev in events:
        if ev.event_type == "link_policy":
            policy.update_from(ev.content)
        elif ev.event_type == "token":
            out.append(sanitizer.feed(ev.content))
    out.append(sanitizer.flush())
    return "".join(out)


@pytest.mark.asyncio
async def test_stream_drops_fabricated_guide_links_but_keeps_sourced_ones():
    """2026-09-21 신고 재현: 답변 말미 가이드 링크가 지어낸 주소였다.

    화면(토큰)과 저장본(result.content) **양쪽에서** 사라져야 한다 — 최종 본문만 고치면
    화면에는 그대로 찍히고 저장본만 깨끗해진다(SSE 에는 본문 교체 이벤트가 없다).
    """
    answer = (
        "에이전트 사용 방법은 다음과 같습니다.\n\n"
        f"문의는 [문의 폼]({_FORM_URL}) 에서 가능합니다.\n\n"
        f"더 자세한 내용은 아래 가이드를 참고해 주세요. [Agent 사용 심층 가이드]({_FAKE_GUIDE_URL})\n"
    )
    groups = [{"category": "General", "category_top_score": 0.8, "chunks": [
        {"title": "고객 문의 접수 및 대응", "content": f"회원 문의 폼( {_FORM_URL} )을 이용하세요.",
         "score": 0.8, "permission_required": "system_user", "permission_blocked": False}]}]
    tc = ToolCall(id="t1", name="search_user_guide_kb", arguments={"query": "에이전트 사용방법"})
    llm = _make_llm([stream_tool_calls([tc], usage=_U1), stream_text(answer, usage=_U2)])

    events, result = [], None
    async for ev in stream_react_graph(
        llm, "에이전트 사용방법 안내해줘", [Message(role="user", content="에이전트 사용방법 안내해줘")],
        [_make_tool("search_user_guide_kb", output=groups)], AsyncMock(return_value=[]), 3,
        target_language="ko",
    ):
        if isinstance(ev, ProgressEvent):
            events.append(ev)
        else:
            result = ev

    assert _FAKE_GUIDE_URL not in result.content
    assert "가이드를 참고해 주세요" not in result.content      # 링크를 잃은 유도 문장도 함께
    assert _FORM_URL in result.content                        # 도구 결과에 있던 실무 주소는 유지

    kinds = [e.event_type for e in events]
    assert "link_policy" in kinds
    assert kinds.index("link_policy") < kinds.index("token")  # 정책이 토큰보다 먼저 나간다
    assert _screen_text(events).rstrip() == result.content.rstrip()   # 화면 == 저장본
    assert _FAKE_GUIDE_URL not in _screen_text(events)


@pytest.mark.asyncio
async def test_no_guide_search_no_policy_and_links_untouched():
    """가이드 검색을 쓰지 않은 턴은 이번 변경 전과 동일하게 동작한다(정책 자체가 안 켜진다)."""
    answer = f"참고하세요: [문서]({_FAKE_GUIDE_URL})"
    tc = ToolCall(id="t1", name="mattermost_search_knowledge", arguments={"query": "q"})
    llm = _make_llm([stream_tool_calls([tc], usage=_U1), stream_text(answer, usage=_U2)])
    events, result = [], None
    async for ev in stream_react_graph(
        llm, "q", [Message(role="user", content="q")],
        [_make_tool("mattermost_search_knowledge", output={"results": []})],
        AsyncMock(return_value=[]), 3, target_language="ko",
    ):
        if isinstance(ev, ProgressEvent):
            events.append(ev)
        else:
            result = ev
    assert _FAKE_GUIDE_URL in result.content
    assert "link_policy" not in [e.event_type for e in events]


def test_name_guard_is_not_armed_by_web_results(monkeypatch):
    """운영 서버 실사용(2026-09-23): 웹 본문의 @계정이 가드를 무장시켜 단가표 헤더 '변동률'이 '이름미확인'으로 바뀌었다."""
    from agent_arch.agents.graphs import react_graph as RG
    from agent_arch.core.config import settings
    from agent_arch.llm.base import Message
    monkeypatch.setattr(settings, "name_answer_guard", True, raising=False)
    web = Message(role="tool", name="brave_llm_context", tool_call_id="c1", content='{"results":[{"content":"follow @chemanalyst"}]}')
    mm = Message(role="tool", name="mattermost_search_knowledge", tool_call_id="c2", content='{"author": "jdoe"}')
    assert RG._name_guard_needed({"messages": [web]}) is False
    assert RG._name_guard_needed({"messages": [web, mm]}) is True                  # 사내 작성자 ID 는 여전히 무장


@pytest.mark.asyncio
async def test_prefetch_calls_run_before_first_model_call():
    """웹 조사 계획: 라우터 검색어를 코드가 첫 모델 호출 전에 실행 — 모델은 결과를 쥔 채 시작한다(실사용 2026-09-22)."""
    seen_msgs = []
    def first_answer(messages, *a, **k):
        seen_msgs.extend(messages)
        return stream_text("답변")(*a, **k)
    llm = _make_llm([first_answer])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": []})
    other = _make_tool("calculator")
    result = await run_react_graph(
        llm, "유가 추이", [Message(role="user", content="유가 추이")], [ctx, other], None, 5,
        prefetch=[{"name": "brave_llm_context", "arguments": {"query": "Brent crude price 2026"}},
                  {"name": "brave_llm_context", "arguments": {"query": "브렌트유 가격 2026년 9월"}},
                  {"name": "not_bound_tool", "arguments": {}}],                    # 스코프 밖 도구는 버린다
    )
    assert ctx.execute.await_count == 2 and other.execute.await_count == 0
    assert [c["name"] for c in result.tool_calls] == ["brave_llm_context", "brave_llm_context"]
    assert any(getattr(m, "role", None) == "tool" for m in seen_msgs)              # 첫 모델 호출이 결과를 본다
    assert result.content.startswith("답변")


@pytest.mark.asyncio
async def test_no_prefetch_keeps_normal_flow():
    llm = _make_llm([stream_text("그냥 답")])
    result = await run_react_graph(llm, "안녕", [Message(role="user", content="안녕")], [], None, 5)
    assert result.content.startswith("그냥 답") and not result.tool_calls


@pytest.mark.asyncio
async def test_prefetch_then_auto_fetch_top_pages_before_model():
    """gemini 가 어떤 안내로도 web_fetch 를 안 불러 '9월 자료 없음'(2026-09-23) — 선실행 뒤 상위 페이지를 코드가 연다."""
    seen = []
    def first_answer(messages, *a, **k):
        seen.extend(m.name for m in messages if getattr(m, "role", None) == "tool")
        return stream_text("답변")(*a, **k)
    llm = _make_llm([first_answer])
    ctx_out = {"kind": "context", "results": [{"url": "https://a.com/epdm-price"}, {"url": "https://a.com/epdm-2"}]}
    ctx2_out = {"kind": "context", "results": [{"url": "https://b.com/epdm-price-index"}]}
    ctx = _make_tool("brave_llm_context")
    ctx.execute = AsyncMock(side_effect=[ToolResult(success=True, output=ctx_out), ToolResult(success=True, output=ctx2_out)])
    fetch = _make_tool("web_fetch", output={"final_url": "x", "text": "9월 표"})
    result = await run_react_graph(
        llm, "EPDM", [Message(role="user", content="EPDM")], [ctx, fetch], None, 6,
        prefetch=[{"name": "brave_llm_context", "arguments": {"query": "epdm price"}},
                  {"name": "brave_llm_context", "arguments": {"query": "epdm price index"}}],
        auto_fetch=True,
    )
    urls = [c.kwargs["url"] for c in fetch.execute.await_args_list]
    assert urls == ["https://b.com/epdm-price-index", "https://a.com/epdm-price"]   # 관련도 순, 호스트 서로 다르게, 최대 2
    assert seen.count("web_fetch") == 2 and result.content.startswith("답변")


def test_auto_fetch_urls_prefers_query_relevant_pages_distinct_hosts():
    """처음엔 검색어별 1순위를 열어 네이버 금융 종목 페이지를 열었다(2026-09-23) — 대상어가 제목·URL 에 든 페이지를 고른다."""
    import json as _j
    from agent_arch.agents.graphs.react_graph import _auto_fetch_urls
    calls = [{"id": "c1", "arguments": {"query": "폴리머 고무 EPDM 단가 2026년 8월"}},
             {"id": "c2", "arguments": {"query": "EPDM rubber price trend August 2026"}}]
    res = [{"name": "brave_llm_context", "tool_call_id": "c1", "content": _j.dumps({"results": [
               {"url": "https://finance.naver.com/item/main.naver?code=017960", "title": "금호석유 : Npay 증권"},
               {"url": "https://analystkorea.com/Pricing-data/epdm-rubber-1035", "title": "EPDM 고무 가격"}]}, ensure_ascii=False)},
           {"name": "brave_llm_context", "tool_call_id": "c2", "content": _j.dumps({"results": [
               {"url": "https://www.imarcgroup.com/epdm-rubber-pricing-report", "title": "EPDM Rubber Prices 2026"},
               {"url": "https://www.imarcgroup.com/other", "title": "EPDM rubber price other"}]})}]
    urls = _auto_fetch_urls(res, calls)
    assert urls == ["https://www.imarcgroup.com/epdm-rubber-pricing-report", "https://analystkorea.com/Pricing-data/epdm-rubber-1035"]
    assert _auto_fetch_urls([{"name": "brave_web_search", "content": "{}"}], []) == []


@pytest.mark.asyncio
async def test_auto_fetch_off_by_default():
    llm = _make_llm([stream_text("답변")])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": [{"url": "https://a.com/epdm-price"}]})
    fetch = _make_tool("web_fetch")
    await run_react_graph(llm, "EPDM", [Message(role="user", content="EPDM")], [ctx, fetch], None, 6,
                          prefetch=[{"name": "brave_llm_context", "arguments": {"query": "epdm price"}}])
    assert fetch.execute.await_count == 0                                        # 라우터가 web_open 을 안 줬으면 열지 않는다


@pytest.mark.asyncio
async def test_auto_fetch_off_by_default():
    llm = _make_llm([stream_text("답변")])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": [{"url": "https://a.com/epdm-price"}]})
    fetch = _make_tool("web_fetch")
    await run_react_graph(llm, "EPDM", [Message(role="user", content="EPDM")], [ctx, fetch], None, 6,
                          prefetch=[{"name": "brave_llm_context", "arguments": {"query": "epdm price"}}])
    assert fetch.execute.await_count == 0                                        # 라우터가 web_open 을 안 줬으면 열지 않는다


def test_auto_fetch_prefers_pages_stating_current_quote():
    """'오늘 유로 기준환율'에 ECB 그래프·Wise 이력을 열었다(2026-09-28) — '1 EUR = N KRW' 를 단언하는 변환 페이지를 우선."""
    import json as _j
    from agent_arch.agents.graphs.react_graph import _auto_fetch_urls
    calls = [{"id": "c1", "arguments": {"query": "EUR KRW exchange rate 2026-09-28"}}]
    res = [{"name": "brave_llm_context", "tool_call_id": "c1", "content": _j.dumps({"results": [
        {"url": "https://www.ecb.europa.eu/stats/policy_and_exchange_rates/euro_reference_exchange_rates/html/eurofxref-graph-krw.en.html",
         "title": "ECB euro reference exchange rate: South Korean won (KRW)", "content": "Reference rates over last four months"},
        {"url": "https://wise.com/us/currency-converter/eur-to-krw-rate/history", "title": "Euro to KRW history", "content": "Historical"},
        {"url": "https://www.xe.com/en-us/currencyconverter/convert/?Amount=1&From=EUR&To=KRW", "title": "1 EUR to KRW",
         "content": "1.00 EUR = 1,544.1762 KRW Mid-market rate"}]})}]
    assert _auto_fetch_urls(res, calls)[0].startswith("https://www.xe.com/")


@pytest.mark.asyncio
async def test_sufficiency_loop_searches_again_until_sufficient(monkeypatch):
    """유로 기준환율(2026-09-28): 첫 결과에 공식 매매기준율이 없는데 모델이 더 찾지 않았다 — 판정이 부족하면 코드가 다시 찾는다."""
    from agent_arch.agents.react import sufficiency as S
    verdicts = iter([{"sufficient": False, "missing": "공식 매매기준율", "queries": ["유로 매매기준율 서울외국환중개"], "open": []},
                     {"sufficient": True, "queries": [], "open": []}])
    async def fake_judge(q, digest, tried, *rounds, **kw):
        return next(verdicts)
    monkeypatch.setattr(S, "judge", fake_judge)
    monkeypatch.setattr(settings, "web_sufficiency_rounds", 2, raising=False)
    llm = _make_llm([stream_text("답변")])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": []})
    result = await run_react_graph(llm, "오늘 유로 기준환율", [Message(role="user", content="오늘 유로 기준환율")], [ctx], None, 6,
                                   prefetch=[{"name": "brave_llm_context", "arguments": {"query": "유로 원 환율"}}])
    qs = [c.kwargs["query"] for c in ctx.execute.await_args_list]
    assert qs == ["유로 원 환율", "유로 매매기준율 서울외국환중개"]                        # 선실행 1 + 충분성 라운드 1
    assert [m["sufficient"] for m in result.step_meta["read"]["sufficiency"]] == [False, True] if "read" in result.step_meta else True


@pytest.mark.asyncio
async def test_sufficiency_off_or_judge_error_goes_straight_to_model(monkeypatch):
    from agent_arch.agents.react import sufficiency as S
    async def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(S, "judge", boom)
    llm = _make_llm([stream_text("답변")])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": []})
    result = await run_react_graph(llm, "q", [Message(role="user", content="q")], [ctx], None, 6,
                                   prefetch=[{"name": "brave_llm_context", "arguments": {"query": "q"}}])
    assert ctx.execute.await_count == 1 and result.content.startswith("답변")


def test_plan_calls_skips_tried_queries_and_unknown_urls():
    from agent_arch.agents.react import sufficiency as S
    v = {"sufficient": False, "queries": ["유로 원 환율", "EUR KRW 고시환율"], "open": ["https://a.com/x", "https://made.up/y"]}
    calls = S.plan_calls(v, ["유로  원 환율"], ["https://www.a.com/x/"], {"brave_llm_context", "web_fetch"})
    assert [(c.name, c.arguments.get("query") or c.arguments.get("url")) for c in calls] == [
        ("brave_llm_context", "EUR KRW 고시환율"), ("web_fetch", "https://a.com/x")]
    assert all(c.id.startswith("prefetch_") for c in calls)


@pytest.mark.asyncio
async def test_last_insufficient_round_hands_gap_to_answering_model(monkeypatch):
    """판정이 '오늘 공식 매매기준율 없음'을 짚었는데 모델이 시장환율을 기준환율처럼 섞었다(2026-09-28) — 빈칸을 넘긴다."""
    from agent_arch.agents.react import sufficiency as S
    async def fake_judge(q, digest, tried, *rounds, **kw):
        return {"sufficient": False, "missing": "오늘 공식 매매기준율", "queries": [], "open": []}
    monkeypatch.setattr(S, "judge", fake_judge)
    monkeypatch.setattr(settings, "web_sufficiency_rounds", 1, raising=False)
    seen = []
    def answer(messages, *a, **k):
        seen.extend(getattr(m, "content", "") for m in messages if getattr(m, "role", None) == "user")
        return stream_text("답변")(*a, **k)
    llm = _make_llm([answer])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": []})
    await run_react_graph(llm, "유로 기준환율", [Message(role="user", content="유로 기준환율")], [ctx], None, 6,
                          prefetch=[{"name": "brave_llm_context", "arguments": {"query": "유로 원 환율"}}])
    assert any("빠졌다고 봤다: 오늘 공식 매매기준율" in c and "가장 최신 값으로 시작하라" in c for c in seen)


def test_sufficiency_digest_shows_fetched_pages_first_and_late_numbers():
    """EPDM(2026-09-28): 검색 13회 중 앞 3회분만 판정기에 보였고, 9월 표는 본문 뒤쪽이라 잘렸다."""
    import json as _j
    from agent_arch.agents.react import sufficiency as S
    calls = [{"id": f"s{i}", "name": "brave_llm_context", "arguments": {"query": f"q{i}"}} for i in range(8)]
    calls.append({"id": "f1", "name": "web_fetch", "arguments": {"url": "https://businessanalytiq.com/epdm"}})
    res = [{"tool_call_id": f"s{i}", "content": _j.dumps({"results": [{"url": f"https://h{i}.com/{k}", "title": "t",
                                                                        "content": "x"} for k in range(6)]})} for i in range(8)]
    page = "EPDM rubber price index intro. " * 20 + "\nSeptember 2026: North America 2.99 USD/kg, up 4.9%."
    res.append({"tool_call_id": "f1", "content": _j.dumps({"final_url": "https://businessanalytiq.com/epdm", "title": "EPDM", "text": page})})
    digest, tried, _ = S.build_digest(calls, res, "EPDM 9월 단가")
    assert digest.splitlines()[0].startswith("- [web_fetch]") and "2.99 USD/kg" in digest
    assert "h7.com" in digest and "https://h0.com/5" not in digest                   # 뒤 검색도 보이고, 호출당 상위 몇 건만
    assert len(tried) == 8


def test_evidence_budget_compacts_web_results_in_place(monkeypatch):
    import json as _j
    from agent_arch.agents.react import evidence_budget as E
    monkeypatch.setattr(settings, "web_evidence_max_chars_large", 10_000, raising=False)
    big = _j.dumps({"results": [{"url": f"https://a.com/{i}", "content": "가" * 3000} for i in range(5)]}, ensure_ascii=False)
    msgs = [Message(role="user", content="q"), Message(role="tool", name="brave_llm_context", tool_call_id="c1", content=big),
            Message(role="tool", name="calendar_list", tool_call_id="c2", content="x" * 20_000)]
    assert E.over_budget(msgs, factor=1) and not E.over_budget(msgs)          # 루프 멈춤은 예산 × STOP_FACTOR 에서
    assert E.compact(msgs) > 0 and E.web_chars(msgs) <= 10_000
    assert len(msgs[2].content) == 20_000 and _j.loads(msgs[1].content)["results"][0]["url"] == "https://a.com/0"


@pytest.mark.asyncio
async def test_over_budget_skips_sufficiency_loop(monkeypatch):
    from agent_arch.agents.react import sufficiency as S
    judged = []
    async def fake_judge(*a):
        judged.append(1); return {"sufficient": False, "missing": "m", "queries": ["다른 검색"], "open": []}
    monkeypatch.setattr(S, "judge", fake_judge)
    monkeypatch.setattr(settings, "web_sufficiency_rounds", 2, raising=False)
    monkeypatch.setattr(settings, "web_evidence_max_chars_large", 500, raising=False)
    llm = _make_llm([stream_text("답변")])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": [{"url": "https://a.com", "content": "나" * 3000}]})
    result = await run_react_graph(llm, "q", [Message(role="user", content="q")], [ctx], None, 6,
                                   prefetch=[{"name": "brave_llm_context", "arguments": {"query": "q"}}])
    assert not judged and ctx.execute.await_count == 1 and result.content.startswith("답변")


@pytest.mark.asyncio
async def test_empty_answers_with_web_evidence_salvage_instead_of_fallback(monkeypatch):
    """vllm EPDM(2026-09-28): 컨텍스트 초과로 빈 응답 2회 → '조회하지 못했습니다'. 모은 근거를 줄여 도구 없이 답한다."""
    monkeypatch.setattr(settings, "web_sufficiency_rounds", 0, raising=False)
    modes = []
    def empty(*a, **k):
        modes.append(k.get("tool_mode")); return stream_text("")(*a, **k)
    def answer(*a, **k):
        modes.append(k.get("tool_mode")); return stream_text("9월 북미 2.99달러")(*a, **k)
    llm = _make_llm([empty, empty, answer])
    ctx = _make_tool("brave_llm_context", output={"kind": "context", "results": [{"url": "https://a.com", "content": "2.99"}]})
    result = await run_react_graph(llm, "EPDM", [Message(role="user", content="EPDM")], [ctx], None, 6,
                                   prefetch=[{"name": "brave_llm_context", "arguments": {"query": "epdm"}}])
    assert modes == ["AUTO", "ANY", "NONE"] and result.content.startswith("9월 북미 2.99달러")


@pytest.mark.asyncio
async def test_final_round_rewrite_fixes_draft_without_tools(monkeypatch):
    """gemini 유로(2026-09-28): 2라운드 검증기가 '오늘 환율이 결과에 있는데 없다고 단정'을 짚었지만 라운드가 없어 그대로 나갔다."""
    monkeypatch.setattr(settings, "answer_verify_mode", "on", raising=False)
    monkeypatch.setattr(settings, "answer_verify_max_rounds", 1, raising=False)
    rewrite = {"verdict": "revise", "issues": ["결과에 오늘 값이 있는데 없다고 단정"],
               "action": {"type": "rewrite", "detail": "Wise 9/28 1,548.29원을 첫 문장에"}}
    calls = []
    monkeypatch.setattr(_V, "run_verifier", _fake_verifier([_REVISE, rewrite], calls))
    modes = []
    def text(content):
        def f(*a, **k):
            modes.append(k.get("tool_mode")); return stream_text(content, usage=_U2)(*a, **k)
        return f
    msgs = [Message(role="user", content="오늘 유로 기준환율")]
    progress, res = [], None
    async for ev in stream_react_graph(_make_llm([
        stream_tool_calls([ToolCall(id="t1", name="brave_web_search", arguments={"query": "유로"})], usage=_U1),
        text("초안1"),
        stream_tool_calls([ToolCall(id="t2", name="brave_web_search", arguments={"query": "industrial bracket market"})], usage=_U1),
        text("초안2: 오늘 값은 찾을 수 없습니다"),
        text("오늘 1유로 1,548.29원(Wise, 9/28)"),
    ]), "오늘 유로 기준환율", msgs, [_make_tool("brave_web_search", {"kind": "results", "results": []})], AsyncMock(return_value=[]), 6):
        if isinstance(ev, ProgressEvent):
            progress.append((ev.event_type, ev.content))
        elif isinstance(ev, LoopResult):
            res = ev
    assert res.content == "오늘 1유로 1,548.29원(Wise, 9/28)" and modes[-1] == "NONE"
    tokens = "".join(c for t, c in progress if t == "token")
    assert tokens == res.content                                              # 틀린 초안은 화면에 안 나간다
    assert len(calls) == 2                                                    # 고쳐 쓴 뒤 다시 검증하지 않는다(1회)


def test_rounded_numbers_are_not_unsupported():
    import json as _j
    res = [{"name": "web_fetch", "content": _j.dumps({"text": "1 EUR = 1,540.207344 KRW, +0.306%"})}]
    assert _V.unsupported_numbers("오늘 1,540.21원, 0.31% 상승, 1,999.50원", res) == ["1,999.50"]
    inv = [{"name": "web_fetch", "content": _j.dumps({"text": "1 KRW = 0.000645998 EUR"})}]
    assert _V.unsupported_numbers("1유로 1,548.00원, 1,600.00원", inv) == ["1,600.00"]      # 역수 환산은 근거 있음
    tz = [{"name": "web_fetch", "content": _j.dumps({"text": "CAD=955.17&diff5=-11.7"})}]
    assert _V.unsupported_numbers("캐나다 달러 955.17원(−11.70), 11.75", tz) == ["11.75"]   # 끝자리 0 차이는 같은 값


def test_evidence_budget_is_narrow_only_for_small_window_models(monkeypatch):
    """운영 서버(2026-09-28): vLLM 창에 맞춘 6만 자를 gemini 에도 걸어 근거가 18만→5만 자로 잘렸고 답이 절반으로 짧아졌다."""
    from agent_arch.agents.react import evidence_budget as E
    from agent_arch.llm.vllm import VllmLLM
    monkeypatch.setattr(settings, "web_evidence_max_chars", 60_000, raising=False)
    monkeypatch.setattr(settings, "web_evidence_max_chars_large", 400_000, raising=False)
    assert E.budget(object.__new__(VllmLLM)) == 60_000
    assert E.budget(MagicMock()) == 400_000 and E.budget() == 400_000


@pytest.mark.asyncio
async def test_first_round_rewrite_is_single_pass_without_reverify(monkeypatch):
    """운영 서버 vllm 유로(2026-09-28): rewrite → 재작성 → 재검증 rewrite → 고쳐 쓰기로 답을 세 번 써 79초. 한 번에 끝낸다."""
    monkeypatch.setattr(settings, "answer_verify_mode", "on", raising=False)
    monkeypatch.setattr(settings, "answer_verify_max_rounds", 1, raising=False)
    rewrite = {"verdict": "revise", "issues": ["어긋남"], "action": {"type": "rewrite", "detail": "오늘 값을 첫 문장에"}}
    calls = []
    monkeypatch.setattr(_V, "run_verifier", _fake_verifier([rewrite, rewrite], calls))
    modes = []
    def text(content):
        def f(*a, **k):
            modes.append(k.get("tool_mode")); return stream_text(content, usage=_U2)(*a, **k)
        return f
    res = await run_react_graph(_make_llm([
        stream_tool_calls([ToolCall(id="t1", name="brave_web_search", arguments={"query": "유로"})], usage=_U1),
        text("초안"), text("고친 답"),
    ]), "유로", [Message(role="user", content="유로")], [_make_tool("brave_web_search", {"kind": "results", "results": []})],
        AsyncMock(return_value=[]), 6)
    assert res.content == "고친 답" and modes == ["AUTO", "NONE"] and len(calls) == 1


def test_verifier_counts_previous_turn_values_as_supported():
    """앞 턴에 smbs 에서 확인한 1,538.04원을 이번 턴 답에 쓰자 검증기가 '이번 턴 결과에 없다'며 두 번 되돌렸다."""
    import json as _j
    from agent_arch.agents.graphs.react_graph import _history_evidence
    msgs = [Message(role="system", content="시스템 9,999.99"), Message(role="user", content="오늘 환율"),
            Message(role="assistant", content="유로 매매기준율 1,538.04원 [smbs](http://www.smbs.biz/ExRate/TodayExRate.jsp)"),
            Message(role="user", content="유로 기준환율")]
    hist = _history_evidence(msgs)
    assert "1,538.04" in hist and "9,999.99" not in hist and "유로 기준환율" not in hist
    state = {"collected_tool_calls": [{"id": "c1", "name": "brave_llm_context", "arguments": {"query": "유로"}}],
             "collected_tool_results": [{"tool_call_id": "c1", "name": "brave_llm_context",
                                         "content": _j.dumps({"kind": "context", "results": [{"url": "https://wise.com/x", "content": "1 EUR = 1,548.29 KRW"}]})}],
             "history_evidence": hist}
    f = _V.collect_facts(state, "오늘 1,538.04원(서울외국환중개) [smbs](http://www.smbs.biz/ExRate/TodayExRate.jsp), 시장 1,548.29원, 1,777.77원")
    assert f["numbers_not_in_results"] == ["1,777.77"] and f["answer_urls_not_in_results"] == []
