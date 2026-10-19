"""AgentOrchestrator.run() / stream()의 LangSmith 태그 주입 검증.

생성자 의존성을 우회하기 위해 __new__로 인스턴스를 생성하고,
handle_command를 커맨드 결과 반환으로 모킹하여 태그 코드 직후 조기 종료시킨다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agent_arch.agents.orchestrator import AgentOrchestrator


def _make_orchestrator(agent_mode: str = "standard", llm_provider: str = "gemini") -> AgentOrchestrator:
    """__init__을 우회하여 AgentOrchestrator 인스턴스를 생성한다."""
    orc = AgentOrchestrator.__new__(AgentOrchestrator)
    orc.agent_mode = agent_mode
    orc.llm_provider = llm_provider
    orc.enable_pii_masking = False
    orc.llm = MagicMock(stateless=False)
    orc.static_tools = []
    orc.guide_tool = MagicMock()
    return orc


# handle_command가 커맨드 결과를 반환하면 run()은 태그 설정 직후 종료된다.
_CMD_RESULT = {"output": "cmd", "success": True}
_USER_INFO = {"sub": "u1", "email": "u@test.com"}


class TestAgentRunTags:
    """run()의 LangSmith 태그 검증."""

    @pytest.mark.asyncio
    async def test_standard_mode_tags(self):
        """standard 모드에서 agent_mode와 llm_provider 태그가 추가된다."""
        orc = _make_orchestrator(agent_mode="standard", llm_provider="gemini")
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            await orc.run("안녕", user_info=_USER_INFO)

        mock_run.add_tags.assert_called_once_with(["agent_mode", "gemini"])

    @pytest.mark.asyncio
    async def test_guide_mode_tags(self):
        """guide 모드에서 guide 태그가 추가로 포함된다."""
        orc = _make_orchestrator(agent_mode="guide", llm_provider="gemini")
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            await orc.run("안녕", user_info=_USER_INFO)

        mock_run.add_tags.assert_called_once_with(["agent_mode", "gemini", "guide"])

    @pytest.mark.asyncio
    async def test_non_gemini_provider_tag(self):
        """llm_provider가 gemini가 아닌 경우에도 해당 프로바이더 이름이 태그에 포함된다."""
        orc = _make_orchestrator(agent_mode="standard", llm_provider="vllm")
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            await orc.run("안녕", user_info=_USER_INFO)

        mock_run.add_tags.assert_called_once_with(["agent_mode", "vllm"])

    @pytest.mark.asyncio
    async def test_no_error_when_run_tree_is_none(self):
        """활성 트레이스가 없어 run tree가 None이어도 run()은 정상 동작한다."""
        orc = _make_orchestrator()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            result = await orc.run("안녕", user_info=_USER_INFO)

        assert result is not None


class TestAgentStreamTags:
    """stream()의 LangSmith 태그 검증."""

    @pytest.mark.asyncio
    async def test_standard_mode_tags(self):
        """standard 모드에서 agent_mode와 llm_provider 태그가 추가된다."""
        orc = _make_orchestrator(agent_mode="standard", llm_provider="gemini")
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            _ = [t async for t in orc.stream("안녕", user_info=_USER_INFO)]

        mock_run.add_tags.assert_called_once_with(["agent_mode", "gemini"])

    @pytest.mark.asyncio
    async def test_guide_mode_tags(self):
        """guide 모드에서 guide 태그가 추가로 포함된다."""
        orc = _make_orchestrator(agent_mode="guide", llm_provider="gemini")
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            _ = [t async for t in orc.stream("안녕", user_info=_USER_INFO)]

        mock_run.add_tags.assert_called_once_with(["agent_mode", "gemini", "guide"])

    @pytest.mark.asyncio
    async def test_no_error_when_run_tree_is_none(self):
        """활성 트레이스가 없어 run tree가 None이어도 stream()은 정상 동작한다."""
        orc = _make_orchestrator()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            tokens = [t async for t in orc.stream("안녕", user_info=_USER_INFO)]

        assert tokens == [_CMD_RESULT["output"]]
