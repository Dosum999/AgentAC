"""USE_LANGGRAPH_REACT flag 라우팅 테스트 (Phase 1c).

orchestrator 가 flag 에 따라 react_loop_events(기존) ↔ react 그래프(run_react_graph)
로 정확히 분기하는지 확인한다. 두 경로의 동작 동일성은 tests/agents/graphs 의 패리티
테스트가 담당하고, 여기서는 '올바른 경로가 선택되는가'만 검증한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.core.config import settings
from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.llm.base import Message, StreamEvent
from agent_arch.agents.react.types import LoopResult


def _make_orc():
    orc = AgentOrchestrator.__new__(AgentOrchestrator)
    orc.agent_mode = "standard"
    orc.llm_provider = "vllm"
    orc.enable_pii_masking = False
    orc.llm = MagicMock(stateless=False, supports_function_calling=True, model="m")
    orc.static_tools = []
    orc.guide_tool = MagicMock()
    orc._search_dynamic_tools = AsyncMock(return_value=[])
    return orc


def _stream_text(content):
    async def _gen(*a, **k):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason="stop", usage=None)
    return _gen


@pytest.mark.asyncio
async def test_collect_react_result_flag_off_uses_loop():
    """flag OFF(기본) → react_loop_events 사용, run_react_graph 미호출."""
    orc = _make_orc()
    orc.llm.stream = _stream_text("루프 답변")
    orc.llm.bind_tools = AsyncMock()
    with patch.object(settings, "use_langgraph_react", False), \
         patch.object(settings, "max_iterations", 3), \
         patch("agent_arch.agents.orchestrator.run_react_graph", new=AsyncMock()) as spy_graph:
        res = await orc._collect_react_result("q", [Message(role="user", content="q")], [])
    spy_graph.assert_not_called()
    assert res is not None and res.content == "루프 답변"


@pytest.mark.asyncio
async def test_collect_react_result_flag_on_uses_graph():
    """flag ON → run_react_graph 사용(react_loop_events 대신)."""
    orc = _make_orc()
    orc.llm.stream = _stream_text("이 경로는 안 쓰여야 함")
    orc.llm.bind_tools = AsyncMock()
    sentinel = LoopResult(content="그래프 답변", finish_reason="stop")
    with patch.object(settings, "use_langgraph_react", True), \
         patch.object(settings, "max_iterations", 3), \
         patch("agent_arch.agents.orchestrator.run_react_graph", new=AsyncMock(return_value=sentinel)) as spy_graph:
        res = await orc._collect_react_result("q", [Message(role="user", content="q")], [])
    spy_graph.assert_awaited_once()
    assert res is sentinel


def test_save_kwargs_includes_step_meta_as_metadata():
    """result.step_meta 가 있으면 _save_kwargs 가 metadata={'react_steps': ...} 로 넘긴다(DB 저장)."""
    import time as _t
    orc = _make_orc()
    orc.llm.model = "m"
    r = LoopResult(content="a", finish_reason="stop",
                   step_meta={"iterations": 2, "node_trace": ["call_model", "answer"]})
    kw = orc._save_kwargs("sub", "q", r, _t.monotonic(), "e@x")
    assert kw["metadata"] == {"react_steps": {"iterations": 2, "node_trace": ["call_model", "answer"]}}
    assert kw["finish_reason"] == "stop"


def test_save_kwargs_no_metadata_when_no_step_meta():
    """step_meta 없으면(루프 경로) metadata 는 None."""
    import time as _t
    orc = _make_orc()
    orc.llm.model = "m"
    r = LoopResult(content="a", finish_reason="stop")  # step_meta None
    kw = orc._save_kwargs("sub", "q", r, _t.monotonic(), "e@x")
    assert kw["metadata"] is None


@pytest.mark.asyncio
async def test_run_react_supervisor_flag_routes_to_supervisor():
    """USE_LANGGRAPH_SUPERVISOR ON → _run_react 가 run_supervisor 로 라우팅."""
    orc = _make_orc()
    orc._prepare_tools = AsyncMock(return_value=[])
    orc._kickoff_eval_replay = MagicMock()
    orc._router_llm = MagicMock(return_value=MagicMock())
    orc._make_worker_llm = MagicMock(return_value=MagicMock())
    ctx = MagicMock()
    ctx.augmented_input = "q"; ctx.messages = []; ctx.profile = None
    ctx.user_sub = None; ctx.conversation_id = "c"; ctx.input = "q"
    sentinel = LoopResult(content="supervisor 답변", finish_reason="stop")
    with patch.object(settings, "use_langgraph_supervisor", True), \
         patch.object(settings, "max_iterations", 3), \
         patch.object(settings, "router_confidence_threshold", 0.6), \
         patch("agent_arch.agents.orchestrator.run_supervisor", new=AsyncMock(return_value=sentinel)) as spy:
        out = await orc._run_react(ctx)
    spy.assert_awaited_once()
    assert out["output"] == "supervisor 답변"


@pytest.mark.asyncio
async def test_run_react_supervisor_off_uses_normal_path():
    """supervisor OFF → run_supervisor 미호출, 일반 경로(_collect_react_result)."""
    orc = _make_orc()
    orc._prepare_tools = AsyncMock(return_value=[])
    orc._kickoff_eval_replay = MagicMock()
    orc._collect_react_result = AsyncMock(return_value=LoopResult(content="일반", finish_reason="stop"))
    ctx = MagicMock()
    ctx.augmented_input = "q"; ctx.messages = []; ctx.profile = None
    ctx.user_sub = None; ctx.conversation_id = "c"; ctx.input = "q"
    with patch.object(settings, "use_langgraph_supervisor", False), \
         patch("agent_arch.agents.orchestrator.run_supervisor", new=AsyncMock()) as spy:
        out = await orc._run_react(ctx)
    spy.assert_not_called()
    assert out["output"] == "일반"


@pytest.mark.asyncio
async def test_react_graph_is_the_default_path():
    """코드 기본값 = ON. 답변 가드가 그래프 경로에만 있기 때문이다.

    원래는 "명시 opt-in 전까지 기존 경로 유지"였고 기본값이 False 였다. 그런데 이름
    환각·한국어·환각 액션 가드가 셋 다 그래프 경로에만 들어갔고 loop.py 에는 없다.
    운영 `.env` 는 진작 true 로 켜 놨지만, 그 줄이 빠진 환경은 **가드가 통째로 없는 채로**
    도는데 아무 신호도 없었다(2026-09-03 확인). 기본값을 배포 현실에 맞춘다.

    supervisor 는 별개다 — 그쪽은 여전히 명시 opt-in 이다.
    """
    from agent_arch.core.config import Settings
    assert Settings.model_fields["use_langgraph_react"].default is True
    assert Settings.model_fields["use_langgraph_supervisor"].default is False
