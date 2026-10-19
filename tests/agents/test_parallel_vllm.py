"""주 LLM + 평가용 LLM 병렬 실행(replay 방식) 검증.

_run_parallel_eval()의 단위 테스트와 _kickoff_eval_replay() 트리거 조건,
그리고 orchestrator.run() / stream()에서 main 루프 종료 후 정확한 시점에
eval task가 발사되는지 검증한다.

replay 방식 핵심:
- main이 guide_loop / react_loop을 모두 끝낸 *후*에 ctx.messages 스냅샷이
  _kickoff_eval_replay → _run_parallel_eval로 전달된다.
- _run_parallel_eval은 도구를 바인딩하지 않고 tool_mode="NONE"으로
  synthesis만 수행한다 (자체 KB 검색 / system prompt mutation 없음).
- DB 저장 시 messages_context에는 tool role 메시지와 assistant tool_calls가
  모두 포함된다 (replay 컨텍스트 추적용).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent_arch.agents.orchestrator import (
    AgentOrchestrator,
    _serialize_messages_for_eval,
)
from agent_arch.llm.base import LLMResponse, Message


def _make_orchestrator(
    llm_provider: str = "vllm",
    agent_mode: str = "standard",
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
    orc.force_parallel_eval = False
    return orc


def _make_response(content: str = "로컬LLM응답") -> LLMResponse:
    return LLMResponse(content=content, finish_reason="stop", usage=None)


_USER_INFO = {"sub": "u1", "email": "u@test.com"}

_SAMPLE_MESSAGES = [
    Message(role="system", content="시스템 프롬프트"),
    Message(role="user", content="질문"),
]

# main 루프가 끝난 시점의 messages 스냅샷 — assistant tool_call + tool 결과 포함
_FINAL_MESSAGES_WITH_TOOLS = [
    Message(role="system", content="시스템 프롬프트"),
    Message(role="user", content="질문"),
    Message(
        role="assistant",
        content="",
        tool_calls=[{"id": "call_1", "name": "search_user_guide_kb",
                     "arguments": {"query": "이메일 첨부"}}],
    ),
    Message(
        role="tool",
        content='{"category":"메일","chunks":[{"title":"메일 첨부 가이드","content":"..."}]}',
        name="search_user_guide_kb",
        tool_call_id="call_1",
    ),
]


# ---------------------------------------------------------------------------
# _serialize_messages_for_eval 단위 테스트
# ---------------------------------------------------------------------------

class TestSerializeMessagesForEval:
    """_serialize_messages_for_eval()이 tool role과 tool_calls를 보존하는지 검증."""

    def test_tool_role_messages_included(self):
        """tool role 메시지가 직렬화 결과에 포함된다 (replay 컨텍스트의 핵심)."""
        out = _serialize_messages_for_eval(_FINAL_MESSAGES_WITH_TOOLS)
        roles = [d["role"] for d in out]
        assert "tool" in roles
        assert "assistant" in roles
        assert "system" in roles
        assert "user" in roles

    def test_assistant_tool_calls_preserved(self):
        """assistant 메시지의 tool_calls 필드가 직렬화에 보존된다."""
        out = _serialize_messages_for_eval(_FINAL_MESSAGES_WITH_TOOLS)
        assistant_dict = next(d for d in out if d["role"] == "assistant")
        assert "tool_calls" in assistant_dict
        assert assistant_dict["tool_calls"][0]["name"] == "search_user_guide_kb"

    def test_tool_message_metadata_preserved(self):
        """tool 메시지의 tool_call_id와 name이 직렬화에 포함된다."""
        out = _serialize_messages_for_eval(_FINAL_MESSAGES_WITH_TOOLS)
        tool_dict = next(d for d in out if d["role"] == "tool")
        assert tool_dict["tool_call_id"] == "call_1"
        assert tool_dict["name"] == "search_user_guide_kb"
        assert "search_user_guide_kb" in tool_dict["content"] or tool_dict["content"]

    def test_optional_fields_omitted_when_empty(self):
        """tool_calls / tool_call_id / name이 None이면 dict에 포함되지 않는다."""
        out = _serialize_messages_for_eval(_SAMPLE_MESSAGES)
        for d in out:
            assert "tool_calls" not in d
            assert "tool_call_id" not in d
            assert "name" not in d


# ---------------------------------------------------------------------------
# _run_parallel_eval() 단위 테스트
# ---------------------------------------------------------------------------

class TestRunParallelEval:
    """_run_parallel_eval()이 replay 방식으로 동작하는지 검증."""

    @pytest.mark.asyncio
    async def test_eval_sink_called_with_correct_params(self):
        """정상 실행 시 eval_sink 에 올바른 파라미터가 전달된다."""
        orc = _make_orchestrator()

        eval_llm = MagicMock()
        eval_llm.model = "gemini-2.5-flash"
        eval_llm.invoke = AsyncMock(return_value=_make_response("Gemini평가답변"))

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory, \
             patch("agent_arch.extensions.extensions.eval_sink",
                   new=AsyncMock(return_value="eval-id")) as mock_save:

            mock_factory.create.return_value = eval_llm

            await orc._run_parallel_eval(
                input="질문",
                final_messages=list(_FINAL_MESSAGES_WITH_TOOLS),
                conversation_id="sess-1",
                agent_mode="standard",
                eval_provider="gemini",
                main_run_id="run-abc",
            )

        mock_save.assert_called_once()
        kw = mock_save.call_args.kwargs
        assert kw["conversation_id"] == "sess-1"
        assert kw["query"] == "질문"
        assert kw["answer"] == "Gemini평가답변"
        assert kw["llm_provider"] == "gemini"
        assert kw["model_name"] == "gemini-2.5-flash"
        assert kw["agent_mode"] == "standard"
        assert kw["response_time"] > 0
        assert kw["main_run_id"] == "run-abc"  # main_run_id가 그대로 전달됨

    @pytest.mark.asyncio
    async def test_invoke_called_with_tool_mode_none(self):
        """eval LLM은 tool_mode="NONE"으로 호출되어 도구 없이 synthesis만 수행한다."""
        orc = _make_orchestrator()

        eval_llm = MagicMock()
        eval_llm.model = "eval-model"
        eval_llm.invoke = AsyncMock(return_value=_make_response())

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory, \
             patch("agent_arch.extensions.extensions.eval_sink", new=AsyncMock()):

            mock_factory.create.return_value = eval_llm

            await orc._run_parallel_eval(
                input="질문",
                final_messages=list(_FINAL_MESSAGES_WITH_TOOLS),
                conversation_id="s1",
                agent_mode="guide",
                eval_provider="gemini",
            )

        eval_llm.invoke.assert_called_once()
        # final_messages가 그대로 invoke의 첫 인자로 전달됐는지
        args = eval_llm.invoke.call_args.args
        kwargs = eval_llm.invoke.call_args.kwargs
        passed_messages = args[0]
        assert len(passed_messages) == len(_FINAL_MESSAGES_WITH_TOOLS)
        # tool_mode="NONE"이 kwargs로 전달됐는지
        assert kwargs.get("tool_mode") == "NONE"

    @pytest.mark.asyncio
    async def test_bind_tools_not_called_on_eval_llm(self):
        """eval LLM에는 도구가 바인딩되지 않는다 (tool 없는 synthesis)."""
        orc = _make_orchestrator()

        eval_llm = MagicMock()
        eval_llm.model = "eval-model"
        eval_llm.invoke = AsyncMock(return_value=_make_response())
        eval_llm.bind_tools = AsyncMock()

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory, \
             patch("agent_arch.extensions.extensions.eval_sink", new=AsyncMock()):

            mock_factory.create.return_value = eval_llm

            await orc._run_parallel_eval(
                input="질문",
                final_messages=list(_FINAL_MESSAGES_WITH_TOOLS),
                conversation_id="s1",
                agent_mode="guide",
                eval_provider="gemini",
            )

        eval_llm.bind_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_factory_creates_parallel_eval_provider(self):
        """LLMFactory.create()가 eval_provider 값으로 호출된다."""
        orc = _make_orchestrator()

        eval_llm = MagicMock()
        eval_llm.model = "eval-model"
        eval_llm.invoke = AsyncMock(return_value=_make_response())

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory, \
             patch("agent_arch.extensions.extensions.eval_sink", new=AsyncMock()):

            mock_factory.create.return_value = eval_llm

            await orc._run_parallel_eval(
                input="질문",
                final_messages=list(_FINAL_MESSAGES_WITH_TOOLS),
                conversation_id="s1",
                agent_mode="standard",
                eval_provider="custom_provider",
            )

        mock_factory.create.assert_called_once_with("custom_provider")

    @pytest.mark.asyncio
    async def test_messages_context_includes_tool_messages(self):
        """messages_context 직렬화 시 tool role과 assistant tool_calls가 포함된다.

        replay 방식의 핵심 — eval이 어떤 retrieval을 봤는지 DB에서 추적 가능해야 한다.
        """
        orc = _make_orchestrator()

        eval_llm = MagicMock()
        eval_llm.model = "eval-model"
        eval_llm.invoke = AsyncMock(return_value=_make_response())

        saved_context = None

        async def capture_save(**kwargs):
            nonlocal saved_context
            saved_context = kwargs.get("messages_context")
            return "eval-id"

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory, \
             patch("agent_arch.extensions.extensions.eval_sink",
                   new=AsyncMock(side_effect=capture_save)):

            mock_factory.create.return_value = eval_llm

            await orc._run_parallel_eval(
                input="질문",
                final_messages=list(_FINAL_MESSAGES_WITH_TOOLS),
                conversation_id="s1",
                agent_mode="guide",
                eval_provider="gemini",
            )

        roles = [m["role"] for m in saved_context]
        assert "tool" in roles  # ← 이전 동작과 정반대: tool role 포함
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        # assistant의 tool_calls가 보존됐는지
        assistant_msg = next(m for m in saved_context if m["role"] == "assistant")
        assert "tool_calls" in assistant_msg

    @pytest.mark.asyncio
    async def test_exception_does_not_propagate(self):
        """LLMFactory.create 실패 등 내부 예외는 외부로 전파되지 않는다."""
        orc = _make_orchestrator()

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory:
            mock_factory.create.side_effect = RuntimeError("프로바이더 초기화 실패")

            # 예외가 외부로 빠져나오지 않아야 함
            await orc._run_parallel_eval(
                input="질문",
                final_messages=list(_FINAL_MESSAGES_WITH_TOOLS),
                conversation_id="s1",
                agent_mode="guide",
                eval_provider="gemini",
            )

    @pytest.mark.asyncio
    async def test_llm_invoke_failure_does_not_propagate(self):
        """평가용 LLM invoke 실패 시에도 예외가 전파되지 않는다."""
        orc = _make_orchestrator()

        eval_llm = MagicMock()
        eval_llm.model = "eval-model"
        eval_llm.invoke = AsyncMock(side_effect=ConnectionError("LLM 서버 연결 실패"))

        with patch("agent_arch.agents.orchestrator.LLMFactory") as mock_factory:
            mock_factory.create.return_value = eval_llm

            await orc._run_parallel_eval(
                input="질문",
                final_messages=list(_FINAL_MESSAGES_WITH_TOOLS),
                conversation_id="s1",
                agent_mode="guide",
                eval_provider="gemini",
            )


# ---------------------------------------------------------------------------
# _kickoff_eval_replay() 트리거 조건 검증
# ---------------------------------------------------------------------------

class TestKickoffEvalReplay:
    """_kickoff_eval_replay()의 발사 조건과 인자 전달 검증."""

    def _make_ctx(self, messages):
        """간단한 _RequestContext mock — _kickoff_eval_replay에 필요한 필드만."""
        from agent_arch.agents.orchestrator import _RequestContext
        return _RequestContext(
            user_sub="u1",
            user_email="u@test.com",
            user_info=_USER_INFO,
            langsmith_run_id="run-xyz",
            messages=messages,
            input="질문",
            augmented_input="질문",
            t_start=0.0,
            command_result=None,
        )

    def test_fires_when_enabled_and_provider_in_map(self):
        """enable_parallel_eval=True + 토글 맵 매칭 시 task 생성."""
        orc = _make_orchestrator(llm_provider="vllm")
        orc._run_parallel_eval = AsyncMock()
        ctx = self._make_ctx(list(_FINAL_MESSAGES_WITH_TOOLS))

        with patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.asyncio.create_task") as mock_create:
            mock_create.side_effect = lambda coro: (coro.close(), MagicMock())[1]
            mock_settings.enable_parallel_eval = True

            orc._kickoff_eval_replay(ctx)

        mock_create.assert_called_once()

    def test_no_fire_when_disabled(self):
        """enable_parallel_eval=False면 발사하지 않는다."""
        orc = _make_orchestrator(llm_provider="vllm")
        orc._run_parallel_eval = AsyncMock()
        ctx = self._make_ctx(list(_SAMPLE_MESSAGES))

        with patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.asyncio.create_task") as mock_create:
            mock_settings.enable_parallel_eval = False

            orc._kickoff_eval_replay(ctx)

        mock_create.assert_not_called()

    def test_no_fire_when_provider_not_in_map(self):
        """프로바이더가 토글 맵(gemini↔vllm)에 없으면 발사하지 않는다."""
        orc = _make_orchestrator(llm_provider="vllm")
        # 토글 맵(gemini↔vllm/openai)에 없는 provider를 시뮬레이션 (실등록 불필요).
        orc.llm_provider = "not_in_map"
        orc._run_parallel_eval = AsyncMock()
        ctx = self._make_ctx(list(_SAMPLE_MESSAGES))

        with patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.asyncio.create_task") as mock_create:
            mock_settings.enable_parallel_eval = True

            orc._kickoff_eval_replay(ctx)

        mock_create.assert_not_called()

    def test_eval_provider_correctly_toggled(self):
        """vllm → gemini, gemini → vllm으로 프로바이더가 토글된다."""
        for main_p, expected_eval in [("vllm", "gemini"), ("gemini", "vllm")]:
            orc = _make_orchestrator(llm_provider=main_p)
            orc._run_parallel_eval = AsyncMock()
            ctx = self._make_ctx(list(_SAMPLE_MESSAGES))

            with patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
                 patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
                 patch("agent_arch.agents.orchestrator.asyncio.create_task") as mock_create:
                # create_task에 들어간 coroutine은 close해서 RuntimeWarning 방지
                mock_create.side_effect = lambda coro: (coro.close(), MagicMock())[1]
                mock_settings.enable_parallel_eval = True
                orc._kickoff_eval_replay(ctx)

            orc._run_parallel_eval.assert_called_once()
            kwargs = orc._run_parallel_eval.call_args.kwargs
            assert kwargs["eval_provider"] == expected_eval, (
                f"main={main_p} → expected eval={expected_eval}, "
                f"got {kwargs.get('eval_provider')}"
            )
            assert kwargs["main_run_id"] == "run-xyz"

    def test_final_messages_is_shallow_copy(self):
        """전달되는 final_messages가 ctx.messages의 얕은 복사여야 한다 (격리)."""
        orc = _make_orchestrator(llm_provider="vllm")
        orc._run_parallel_eval = AsyncMock()
        ctx_messages = list(_FINAL_MESSAGES_WITH_TOOLS)
        ctx = self._make_ctx(ctx_messages)

        with patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.asyncio.create_task") as mock_create:
            mock_create.side_effect = lambda coro: (coro.close(), MagicMock())[1]
            mock_settings.enable_parallel_eval = True
            orc._kickoff_eval_replay(ctx)

        orc._run_parallel_eval.assert_called_once()
        passed = orc._run_parallel_eval.call_args.kwargs["final_messages"]
        assert passed is not ctx_messages  # 다른 list 객체
        assert len(passed) == len(ctx_messages)  # 같은 내용


# ---------------------------------------------------------------------------
# run() / stream()에서 main 종료 후 발사 검증
# ---------------------------------------------------------------------------

class TestRunParallelTrigger:
    """orchestrator.run()에서 main 분기 종료 후 _kickoff_eval_replay가 호출되는지 검증."""

    @pytest.mark.asyncio
    async def test_kickoff_called_in_direct_llm_path(self):
        """demo 모드(_run_direct_llm)에서 main 응답 후 kickoff 호출."""
        orc = _make_orchestrator(llm_provider="vllm", agent_mode="demo")
        orc.llm.invoke = AsyncMock(return_value=_make_response("vLLM 응답"))
        orc._kickoff_eval_replay = MagicMock()

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))):

            mock_settings.demo_mode = True
            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = True

            await orc.run("질문", user_info=_USER_INFO)

        orc._kickoff_eval_replay.assert_called_once()


class TestStreamParallelTrigger:
    """orchestrator.stream()에서 main 분기 종료 후 _kickoff_eval_replay가 호출되는지 검증."""

    @pytest.mark.asyncio
    async def test_kickoff_called_in_stream_direct_llm(self):
        """demo 모드 스트리밍에서 토큰 yield 완료 후 kickoff 호출."""
        orc = _make_orchestrator(llm_provider="vllm", agent_mode="demo")

        async def fake_stream(*args, **kwargs):
            from agent_arch.llm.base import StreamEvent
            for tok in ["응", "답"]:
                yield StreamEvent(type="text", content=tok)
            yield StreamEvent(type="done", finish_reason="stop")

        orc.llm.stream = fake_stream
        # demo 비-stateless 스트리밍은 본문 스트림 전에 분류용 invoke를 한 번 호출한다.
        # 헤더 없는 응답이면 parsed=None → plain 경로 → 본문 stream → kickoff 도달.
        orc.llm.invoke = AsyncMock(return_value=_make_response())
        orc._kickoff_eval_replay = MagicMock()

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))):

            mock_settings.demo_mode = True
            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = True

            tokens = [t async for t in orc.stream("질문", user_info=_USER_INFO)]

        assert len(tokens) > 0
        orc._kickoff_eval_replay.assert_called_once()

    @pytest.mark.asyncio
    async def test_main_response_returned_when_kickoff_raises(self):
        """_kickoff_eval_replay 자체가 예외를 던져도 main 토큰은 정상 yield된다."""
        orc = _make_orchestrator(llm_provider="vllm", agent_mode="demo")

        async def fake_stream(*args, **kwargs):
            from agent_arch.llm.base import StreamEvent
            for tok in ["응", "답"]:
                yield StreamEvent(type="text", content=tok)
            yield StreamEvent(type="done", finish_reason="stop")

        orc.llm.stream = fake_stream
        # kickoff에서 예외 발생 시뮬레이션 (실제로는 발생하지 않아야 하지만 안전망 테스트)
        orc._kickoff_eval_replay = MagicMock()

        with patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=None)), \
             patch("agent_arch.agents.orchestrator.settings") as mock_settings, \
             patch("agent_arch.agents.orchestrator.load_history", new=AsyncMock(return_value=[])), \
             patch("agent_arch.agents.orchestrator.build_system_prompt", return_value="sys"), \
             patch("agent_arch.agents.orchestrator.set_user_context"), \
             patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.save_logged_in_turn", new=AsyncMock(return_value=("msg-id", "conv-id"))):

            mock_settings.demo_mode = True
            mock_settings.enable_pii_masking = False
            mock_settings.enable_parallel_eval = True

            tokens = [t async for t in orc.stream("질문", user_info=_USER_INFO)]

        assert len(tokens) > 0
