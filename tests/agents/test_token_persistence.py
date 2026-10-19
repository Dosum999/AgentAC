"""토큰 수 누적 및 PostgreSQL 저장 검증.

_merge_usage 단위 테스트와 orchestrator.run() / stream()에서
토큰 수가 save_logged_in_turn()에 올바르게 전달되는지 검증한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.agents.react import merge_usage as _merge_usage
from agent_arch.llm.base import LLMResponse


def _make_orchestrator(
    agent_mode: str = "standard",
    llm_provider: str = "gemini",
    supports_function_calling: bool = False,
) -> AgentOrchestrator:
    """__init__을 우회하여 AgentOrchestrator 인스턴스를 생성한다."""
    orc = AgentOrchestrator.__new__(AgentOrchestrator)
    orc.agent_mode = agent_mode
    orc.llm_provider = llm_provider
    orc.enable_pii_masking = False
    orc.llm = MagicMock(
        stateless=False,
        supports_function_calling=supports_function_calling,
    )
    orc.static_tools = []
    orc.guide_tool = MagicMock()
    return orc


def _make_response(content: str = "응답", usage: dict | None = None) -> LLMResponse:
    """LLMResponse Mock을 생성한다."""
    return LLMResponse(content=content, finish_reason="stop", usage=usage)


def _stream_events_text(content: str, usage: dict | None = None):
    """텍스트 답변 회차 StreamEvent 시퀀스 (text → done(stop))를 만드는 팩토리."""
    from agent_arch.llm.base import StreamEvent

    async def _gen(*args, **kwargs):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason="stop", usage=usage)
    return _gen


def _stream_events_tool_calls(tool_calls, usage: dict | None = None):
    """도구 호출 회차 StreamEvent 시퀀스 (tool_calls → done(tool_calls))를 만드는 팩토리."""
    from agent_arch.llm.base import StreamEvent

    async def _gen(*args, **kwargs):
        yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
        yield StreamEvent(type="done", finish_reason="tool_calls", usage=usage)
    return _gen


def _stream_events_sequence(*per_iter_gens):
    """여러 iteration 분의 stream 시퀀스를 순차 호출 단위로 묶는 팩토리.

    각 호출마다 다음 generator 가 사용되도록 side_effect 로 주입한다.
    """
    iter_gens = list(per_iter_gens)

    def _next_gen(*args, **kwargs):
        gen = iter_gens.pop(0)
        return gen(*args, **kwargs)

    return _next_gen


_USER_INFO = {"sub": "u1", "email": "u@test.com"}
_USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

# run() / stream() 공통 패치 목록
_COMMON_PATCHES = [
    "agent_arch.agents.orchestrator.handle_command",
    "agent_arch.agents.orchestrator.load_history",
    "agent_arch.agents.orchestrator.build_system_prompt",
    "agent_arch.agents.orchestrator.set_user_context",
    "agent_arch.agents.orchestrator.get_current_run_tree",
    "agent_arch.agents.orchestrator.save_logged_in_turn",
]


class TestMergeUsage:
    """_merge_usage() 단위 테스트."""

    def test_accumulates_values(self):
        """usage 값이 누적 딕셔너리에 정확히 합산된다."""
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _merge_usage(acc, {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30})
        assert acc == {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    def test_multiple_accumulations_sum_correctly(self):
        """여러 번 호출하면 누적 합산된다."""
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _merge_usage(acc, {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15})
        _merge_usage(acc, {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10})
        assert acc["prompt_tokens"] == 8
        assert acc["completion_tokens"] == 17
        assert acc["total_tokens"] == 25

    def test_none_usage_is_ignored(self):
        """usage가 None이면 acc가 변경되지 않는다."""
        acc = {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}
        _merge_usage(acc, None)
        assert acc == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}

    def test_missing_keys_default_to_zero(self):
        """usage에 일부 키가 없으면 해당 값을 0으로 처리한다."""
        acc = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _merge_usage(acc, {"prompt_tokens": 5})
        assert acc["prompt_tokens"] == 5
        assert acc["completion_tokens"] == 0
        assert acc["total_tokens"] == 0

    def test_zero_usage_values_are_accumulated(self):
        """usage 값이 0이어도 합산 연산이 정상 수행된다."""
        acc = {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}
        _merge_usage(acc, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        assert acc == {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}


class TestOrchestratorRunTokens:
    """orchestrator.run()의 토큰 수 PostgreSQL 저장 검증."""

    @pytest.mark.asyncio
    async def test_direct_path_tokens_passed_to_save(self):
        """직접 응답 경로에서 invoke() usage가 save_logged_in_turn에 전달된다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response("응답", usage=_USAGE))

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))) as mock_save:

            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = False

            await orc.run("질문", user_info=_USER_INFO)

        mock_save.assert_called_once()
        kw = mock_save.call_args.kwargs
        assert kw["prompt_tokens"] == 10
        assert kw["completion_tokens"] == 20
        assert kw["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_direct_path_none_tokens_when_usage_absent(self):
        """usage가 None이면 save_logged_in_turn에 None 토큰 값이 전달된다."""
        orc = _make_orchestrator()
        orc.llm.invoke = AsyncMock(return_value=_make_response("응답", usage=None))

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))) as mock_save:

            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = False

            await orc.run("질문", user_info=_USER_INFO)

        kw = mock_save.call_args.kwargs
        assert kw["prompt_tokens"] is None
        assert kw["completion_tokens"] is None
        assert kw["total_tokens"] is None


class TestOrchestratorStreamTokens:
    """orchestrator.stream()의 토큰 수 PostgreSQL 저장 검증."""

    @pytest.mark.asyncio
    async def test_react_no_tool_calls_tokens_passed_to_save(self):
        """ReAct 루프에서 도구 호출 없이 즉시 답변 시 토큰이 save_logged_in_turn에 전달된다."""
        orc = _make_orchestrator(supports_function_calling=True)
        orc.llm.stream = _stream_events_text("답변", usage=_USAGE)
        orc.llm.bind_tools = AsyncMock()
        orc._search_dynamic_tools = AsyncMock(return_value=[])

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))) as mock_save:

            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = False
            mock_settings.max_iterations = 3
            mock_settings.use_langgraph_react = False
            mock_settings.use_langgraph_supervisor = False

            _ = [t async for t in orc.stream("질문", user_info=_USER_INFO)]

        mock_save.assert_called_once()
        kw = mock_save.call_args.kwargs
        assert kw["prompt_tokens"] == 10
        assert kw["completion_tokens"] == 20
        assert kw["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_react_multiple_invokes_accumulate_tokens(self):
        """ReAct 루프에서 여러 번 stream() 호출 시 토큰이 누적되어 저장된다."""
        orc = _make_orchestrator(supports_function_calling=True)

        from agent_arch.llm.base import ToolCall
        tool_call = ToolCall(id="tc1", name="fake_tool", arguments={})

        # 첫 stream: tool_calls 반환 → 두 번째 stream: 최종 답변
        first_usage = {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15}
        second_usage = {"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}

        orc.llm.stream = _stream_events_sequence(
            _stream_events_tool_calls([tool_call], usage=first_usage),
            _stream_events_text("최종 답변", usage=second_usage),
        )
        orc.llm.bind_tools = AsyncMock()
        orc._search_dynamic_tools = AsyncMock(return_value=[])

        fake_tool = MagicMock()
        fake_tool.name = "fake_tool"
        from agent_arch.agents.tools.base import ToolResult
        fake_tool.execute = AsyncMock(return_value=ToolResult(success=True, output="결과"))
        orc.static_tools = [fake_tool]

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))) as mock_save:

            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = False
            mock_settings.max_iterations = 3
            mock_settings.use_langgraph_react = False
            mock_settings.use_langgraph_supervisor = False

            _ = [t async for t in orc.stream("질문", user_info=_USER_INFO)]

        mock_save.assert_called_once()
        kw = mock_save.call_args.kwargs
        # 5 + 3 = 8, 10 + 7 = 17, 15 + 10 = 25
        assert kw["prompt_tokens"] == 8
        assert kw["completion_tokens"] == 17
        assert kw["total_tokens"] == 25


class TestFinishReasonPersistence:
    """마지막 LLM 응답의 finish_reason이 save_logged_in_turn에 전달되는지 검증.

    두 배선 경로를 각각 확인한다:
      - direct/demo run 경로: response.finish_reason 직접 전달
      - ReAct 경로: _save_kwargs 가 LoopResult.finish_reason 전달
    """

    @pytest.mark.asyncio
    async def test_direct_llm_finish_reason_passed_to_save(self):
        """direct LLM(run) 경로에서 response.finish_reason 이 저장에 전달된다."""
        orc = _make_orchestrator()  # supports_function_calling=False → DIRECT_LLM
        orc.llm.invoke = AsyncMock(return_value=_make_response("응답", usage=_USAGE))

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))) as mock_save:

            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = False

            await orc.run("질문", user_info=_USER_INFO)

        assert mock_save.call_args.kwargs["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_react_finish_reason_passed_to_save(self):
        """ReAct(stream) 경로에서 LoopResult.finish_reason 이 저장에 전달된다."""
        orc = _make_orchestrator(supports_function_calling=True)
        orc.llm.stream = _stream_events_text("답변", usage=_USAGE)
        orc.llm.bind_tools = AsyncMock()
        orc._search_dynamic_tools = AsyncMock(return_value=[])

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))) as mock_save:

            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = False
            mock_settings.max_iterations = 3
            mock_settings.use_langgraph_react = False
            mock_settings.use_langgraph_supervisor = False

            _ = [t async for t in orc.stream("질문", user_info=_USER_INFO)]

        assert mock_save.call_args.kwargs["finish_reason"] == "stop"
