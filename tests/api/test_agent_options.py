"""에이전트 API 엔드포인트의 사용자 옵션 필드(`agent_mode`, `llm_provider`) 동작 검증.

핸들러가:
- 새 Form 필드 `agent_mode`를 AgentOrchestrator로 전달하는지
- AgentOrchestrator의 ValueError(잘못된 agent_mode)를 HTTP 400으로 매핑하는지
- llm_provider 별칭(`gemma`)을 그대로 전달하는지 (정규화는 orchestrator 내부 책임)
를 핸들러 레벨에서 확인한다.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from agent_arch.api.v1.agent import run_agent, run_agent_async, stream_agent


_USER_INFO = {"sub": "u1", "email": "u@test.com", "company": "Acme"}


def _make_http_request() -> MagicMock:
    """FastAPI Request 목 객체를 생성한다."""
    req = MagicMock()
    req.state.user = _USER_INFO
    req.state.company = "Acme"
    req.state.session_id = None
    req.cookies = {}
    return req


def _make_orchestrator_mock() -> MagicMock:
    """run/stream 둘 다 정상 응답하는 orchestrator 목."""
    mock_orc = MagicMock()
    mock_orc.llm_provider = "vllm"
    mock_orc.run = AsyncMock(return_value={
        "success": True,
        "output": "답변",
        "llm_provider": "vllm",
        "tool_calls": [],
        "message_id": "msg-001",
        "metadata": {},
    })

    async def _stream(*args, **kwargs):
        yield "토큰"
    mock_orc.stream = MagicMock(side_effect=_stream)
    return mock_orc


# ---------------------------------------------------------------------------
# /agent/run — agent_mode + llm_provider 전달 및 검증
# ---------------------------------------------------------------------------

class TestRunAgentOptions:
    """run_agent() 핸들러의 옵션 필드 처리 검증."""

    @pytest.mark.asyncio
    async def test_agent_mode_forwarded_to_orchestrator(self):
        """agent_mode가 AgentOrchestrator 생성자에 그대로 전달된다."""
        mock_orc = _make_orchestrator_mock()

        with patch("agent_arch.api.v1.agent.AgentOrchestrator", return_value=mock_orc) as orc_cls, \
             patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            await run_agent(
                http_request=_make_http_request(),
                input="질문",
                files=None,
                llm_provider=None,
                agent_mode="guide",
            )

        assert orc_cls.call_args.kwargs["agent_mode"] == "guide"

    @pytest.mark.asyncio
    async def test_llm_provider_alias_forwarded_raw(self):
        """핸들러는 resolve_effective_provider의 결과를 orchestrator에 전달한다.

        provider 게이트가 핸들러 레벨에서 provider를 확정하므로, 여기선
        resolve_effective_provider를 passthrough로 mock해 전달 배선만 검증한다.
        """
        mock_orc = _make_orchestrator_mock()

        with patch("agent_arch.api.v1.agent.AgentOrchestrator", return_value=mock_orc) as orc_cls, \
             patch("agent_arch.api.v1.agent.resolve_effective_provider",
                   new=AsyncMock(side_effect=lambda requested, company=None: requested)), \
             patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            await run_agent(
                http_request=_make_http_request(),
                input="질문",
                files=None,
                llm_provider="gemma",
                agent_mode=None,
            )

        assert orc_cls.call_args.kwargs["llm_provider"] == "gemma"

    @pytest.mark.asyncio
    async def test_invalid_agent_mode_returns_400(self):
        """orchestrator 생성 중 ValueError가 발생하면 핸들러는 400을 반환한다."""
        with patch(
            "agent_arch.api.v1.agent.AgentOrchestrator",
            side_effect=ValueError("Unknown agent_mode: xxx"),
        ), patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            with pytest.raises(HTTPException) as exc_info:
                await run_agent(
                    http_request=_make_http_request(),
                    input="질문",
                    files=None,
                    llm_provider=None,
                    agent_mode="invalid_mode",
                )

        assert exc_info.value.status_code == 400
        assert "Unknown agent_mode" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_options_omitted_pass_none(self):
        """옵션 미전송 시 agent_mode=None, llm_provider는 resolver 결과(passthrough mock으로 None)가 전달된다."""
        mock_orc = _make_orchestrator_mock()

        with patch("agent_arch.api.v1.agent.AgentOrchestrator", return_value=mock_orc) as orc_cls, \
             patch("agent_arch.api.v1.agent.resolve_effective_provider",
                   new=AsyncMock(side_effect=lambda requested, company=None: requested)), \
             patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            await run_agent(
                http_request=_make_http_request(),
                input="질문",
                files=None,
                llm_provider=None,
                agent_mode=None,
            )

        assert orc_cls.call_args.kwargs["agent_mode"] is None
        assert orc_cls.call_args.kwargs["llm_provider"] is None


# ---------------------------------------------------------------------------
# /agent/run/async — input_data·metadata에도 agent_mode가 기록됨
# ---------------------------------------------------------------------------

class TestRunAgentAsyncOptions:
    """run_agent_async() 핸들러의 옵션 필드 처리 검증."""

    @pytest.mark.asyncio
    async def test_agent_mode_in_job_metadata_and_input_data(self):
        """job_manager.enqueue 호출 시 input_data와 metadata 모두에 agent_mode가 기록된다."""
        mock_orc = _make_orchestrator_mock()
        mock_enqueue = AsyncMock(return_value="job-001")

        with patch("agent_arch.api.v1.agent.AgentOrchestrator", return_value=mock_orc), \
             patch("agent_arch.api.v1.agent.job_manager.enqueue", mock_enqueue), \
             patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            await run_agent_async(
                http_request=_make_http_request(),
                input="질문",
                files=None,
                llm_provider="gemma",
                agent_mode="agent",
            )

        kwargs = mock_enqueue.call_args.kwargs
        assert kwargs["input_data"]["agent_mode"] == "agent"
        assert kwargs["input_data"]["llm_provider"] == "gemma"
        assert kwargs["metadata"]["agent_mode"] == "agent"
        assert kwargs["metadata"]["llm_provider"] == "gemma"

    @pytest.mark.asyncio
    async def test_invalid_agent_mode_returns_400(self):
        """async 핸들러도 ValueError를 400으로 매핑한다."""
        with patch(
            "agent_arch.api.v1.agent.AgentOrchestrator",
            side_effect=ValueError("Unknown agent_mode: xxx"),
        ), patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            with pytest.raises(HTTPException) as exc_info:
                await run_agent_async(
                    http_request=_make_http_request(),
                    input="질문",
                    files=None,
                    llm_provider=None,
                    agent_mode="bogus",
                )

        assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# /agent/stream — agent_mode 전달 + ValueError → 400
# ---------------------------------------------------------------------------

class TestStreamAgentOptions:
    """stream_agent() 핸들러의 옵션 필드 처리 검증."""

    async def _collect_sse(self, response) -> list[dict]:
        events = []
        async for chunk in response.body_iterator:
            for line in chunk.split("\n"):
                if line.startswith("data: "):
                    try:
                        events.append(json.loads(line[6:]))
                    except json.JSONDecodeError:
                        pass
        return events

    @pytest.mark.asyncio
    async def test_agent_mode_forwarded_to_orchestrator(self):
        """stream 핸들러도 agent_mode를 orchestrator 생성자에 전달한다."""
        mock_orc = _make_orchestrator_mock()

        with patch("agent_arch.api.v1.agent.AgentOrchestrator", return_value=mock_orc) as orc_cls, \
             patch("agent_arch.api.v1.agent.resolve_effective_provider",
                   new=AsyncMock(side_effect=lambda requested, company=None: requested)), \
             patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            response = await stream_agent(
                http_request=_make_http_request(),
                input="질문",
                files=None,
                llm_provider="gemma",
                agent_mode="guide",
            )
            await self._collect_sse(response)

        assert orc_cls.call_args.kwargs["agent_mode"] == "guide"
        assert orc_cls.call_args.kwargs["llm_provider"] == "gemma"

    @pytest.mark.asyncio
    async def test_invalid_agent_mode_returns_400(self):
        """stream 핸들러도 orchestrator의 ValueError를 400으로 매핑한다.

        ValueError는 orchestrator 생성 시점에 발생하므로 SSE 스트림이 시작되기 전에
        HTTPException이 던져진다.
        """
        with patch(
            "agent_arch.api.v1.agent.AgentOrchestrator",
            side_effect=ValueError("Unknown agent_mode: xxx"),
        ), patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            with pytest.raises(HTTPException) as exc_info:
                await stream_agent(
                    http_request=_make_http_request(),
                    input="질문",
                    files=None,
                    llm_provider=None,
                    agent_mode="invalid",
                )

        assert exc_info.value.status_code == 400
