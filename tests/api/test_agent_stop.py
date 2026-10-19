"""생성 중단(stop) 기능의 API 레벨 테스트.

- /agent/stream 이 첫 SSE 이벤트로 run_started(run_id) 를 내려보내는지
- 사용자 중단 시 done 직전에 stopped 이벤트를 발사하는지
- POST /agent/runs/{run_id}/stop 의 소유권 검증(본인 성공 / 타인·미존재 404)
을 핸들러 레벨에서 검증한다.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException

from agent_arch.api.v1.agent import stream_agent, stop_run
from agent_arch.agents.run_registry import register_run, unregister_run, _RUNS


_USER_INFO = {"sub": "u1", "email": "u@test.com", "company": "Acme"}


def _make_http_request(sub: str = "u1") -> MagicMock:
    """FastAPI Request 목."""
    req = MagicMock()
    req.state.user = {**_USER_INFO, "sub": sub}
    req.state.company = "Acme"
    req.state.session_id = None
    req.cookies = {}
    return req


async def _collect_sse(response) -> list[dict]:
    events = []
    async for chunk in response.body_iterator:
        for line in chunk.split("\n"):
            if line.startswith("data: "):
                try:
                    events.append(json.loads(line[6:]))
                except json.JSONDecodeError:
                    pass
    return events


def _orchestrator_mock(stream_fn) -> MagicMock:
    mock_orc = MagicMock()
    mock_orc.llm_provider = "vllm"
    mock_orc.stream = MagicMock(side_effect=stream_fn)
    return mock_orc


# ── /agent/stream SSE 계약 ────────────────────────────────────────────────────

class TestStreamStopEvents:
    """run_started / stopped SSE 이벤트 검증."""

    @pytest.mark.asyncio
    async def test_run_started_첫이벤트로_run_id_방출(self):
        """스트림 첫 이벤트는 run_started 이고 run_id 를 동봉한다."""
        async def _stream(*_a, **_k):
            yield "토큰"

        with patch("agent_arch.api.v1.agent.AgentOrchestrator", return_value=_orchestrator_mock(_stream)), \
             patch("agent_arch.api.v1.agent.resolve_effective_provider", new=AsyncMock(return_value="vllm")), \
             patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            response = await stream_agent(
                http_request=_make_http_request(), input="질문",
                files=None, llm_provider=None, agent_mode=None,
            )
            events = await _collect_sse(response)

        assert events[0]["type"] == "run_started"
        assert events[0]["data"]["run_id"]
        # 스트림 종료 후 레지스트리는 finally 에서 정리되어야 한다
        assert events[0]["data"]["run_id"] not in _RUNS

    @pytest.mark.asyncio
    async def test_stopped_이벤트_done_직전_방출(self):
        """orchestrator 가 _out['stopped'] 를 세팅하면 done 직전에 stopped 가 나온다."""
        async def _stream(*_a, **kwargs):
            kwargs["_out"]["stopped"] = True
            yield "부분답변"

        with patch("agent_arch.api.v1.agent.AgentOrchestrator", return_value=_orchestrator_mock(_stream)), \
             patch("agent_arch.api.v1.agent.resolve_effective_provider", new=AsyncMock(return_value="vllm")), \
             patch("agent_arch.api.v1.agent.build_user_info", return_value=_USER_INFO):
            response = await stream_agent(
                http_request=_make_http_request(), input="질문",
                files=None, llm_provider=None, agent_mode=None,
            )
            events = await _collect_sse(response)

        types = [e["type"] for e in events]
        assert "stopped" in types
        assert types.index("stopped") < types.index("done"), "stopped 는 done 직전에 와야 한다"


# ── POST /agent/runs/{run_id}/stop ────────────────────────────────────────────

class TestStopEndpoint:
    """중단 엔드포인트 소유권 검증."""

    @pytest.mark.asyncio
    async def test_본인_run_중지_성공(self):
        run_id, event = register_run("u1")
        try:
            with patch("agent_arch.api.v1.agent.build_user_info", return_value={"sub": "u1"}):
                result = await stop_run(run_id, _make_http_request("u1"))
            assert result["success"] is True
            assert result["run_id"] == run_id
            assert event.is_set(), "취소 Event 가 set 되어야 한다"
        finally:
            unregister_run(run_id)

    @pytest.mark.asyncio
    async def test_타인_run_404(self):
        run_id, event = register_run("u1")
        try:
            with patch("agent_arch.api.v1.agent.build_user_info", return_value={"sub": "u2"}):
                with pytest.raises(HTTPException) as exc:
                    await stop_run(run_id, _make_http_request("u2"))
            assert exc.value.status_code == 404
            assert not event.is_set(), "타인 요청으로는 Event 가 set 되면 안 된다"
        finally:
            unregister_run(run_id)

    @pytest.mark.asyncio
    async def test_미존재_run_404(self):
        with patch("agent_arch.api.v1.agent.build_user_info", return_value={"sub": "u1"}):
            with pytest.raises(HTTPException) as exc:
                await stop_run("deadbeef", _make_http_request("u1"))
        assert exc.value.status_code == 404
