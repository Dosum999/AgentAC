"""AgentOrchestrator._preprocess_request()의 LangSmith 메타데이터 주입 검증.

handle_command를 커맨드 결과 반환으로 모킹하여
메타데이터 설정 코드 직후 조기 종료시킨다.
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


_CMD_RESULT = {"output": "cmd", "success": True}

_FULL_USER_INFO = {
    "sub": "user-uuid-123",
    "email": "user@example.com",
    "company": "acme",
    "session_id": "sess-abc",
}


class TestRunMetadata:
    """run()의 LangSmith 메타데이터 주입 검증."""

    @pytest.mark.asyncio
    async def test_full_user_info_adds_all_metadata_fields(self):
        """user_info가 있으면 user_id·user_email·company·session_id 4개 필드가 추가된다."""
        orc = _make_orchestrator()
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            await orc.run("안녕", user_info=_FULL_USER_INFO)

        mock_run.add_metadata.assert_called_once_with({
            "user_id":    "user-uuid-123",
            "user_email": "user@example.com",
            "company":    "acme",
            "session_id": "sess-abc",
            # 사용량 관측의 서비스 축 — user_info 에 없으면 None(=미분류)으로 남는다.
            "service": None,
        })

    @pytest.mark.asyncio
    async def test_user_info_none_skips_metadata(self):
        """user_info가 None이면 add_metadata가 호출되지 않는다."""
        orc = _make_orchestrator()
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            await orc.run("안녕", user_info=None)

        mock_run.add_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_partial_user_info_fills_missing_fields_with_none(self):
        """일부 필드만 있는 user_info는 누락 필드를 None으로 채워 메타데이터를 추가한다."""
        orc = _make_orchestrator()
        mock_run = MagicMock()
        partial_user_info = {"sub": "user-uuid-123", "email": "user@example.com"}

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            await orc.run("안녕", user_info=partial_user_info)

        mock_run.add_metadata.assert_called_once_with({
            "user_id":    "user-uuid-123",
            "user_email": "user@example.com",
            "company":    None,
            "session_id": None,
            # 사용량 관측의 서비스 축 — user_info 에 없으면 None(=미분류)으로 남는다.
            "service": None,
        })

    @pytest.mark.asyncio
    async def test_run_tree_none_skips_metadata(self):
        """활성 트레이스가 없어 run tree가 None이면 add_metadata가 호출되지 않는다."""
        orc = _make_orchestrator()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=None), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            result = await orc.run("안녕", user_info=_FULL_USER_INFO)

        assert result is not None


class TestStreamMetadata:
    """stream()의 LangSmith 메타데이터 주입 검증."""

    @pytest.mark.asyncio
    async def test_full_user_info_adds_all_metadata_fields(self):
        """user_info가 있으면 스트림에서도 4개 메타데이터 필드가 추가된다."""
        orc = _make_orchestrator()
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            _ = [t async for t in orc.stream("안녕", user_info=_FULL_USER_INFO)]

        mock_run.add_metadata.assert_called_once_with({
            "user_id":    "user-uuid-123",
            "user_email": "user@example.com",
            "company":    "acme",
            "session_id": "sess-abc",
            # 사용량 관측의 서비스 축 — user_info 에 없으면 None(=미분류)으로 남는다.
            "service": None,
        })

    @pytest.mark.asyncio
    async def test_user_info_none_skips_metadata(self):
        """user_info가 None이면 스트림에서도 add_metadata가 호출되지 않는다."""
        orc = _make_orchestrator()
        mock_run = MagicMock()

        with patch("agent_arch.agents.orchestrator.get_current_run_tree", return_value=mock_run), \
             patch("agent_arch.agents.orchestrator.handle_command", new=AsyncMock(return_value=_CMD_RESULT)):
            _ = [t async for t in orc.stream("안녕", user_info=None)]

        mock_run.add_metadata.assert_not_called()
