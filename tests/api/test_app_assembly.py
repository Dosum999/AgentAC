"""앱 조립부 — 외부 의존 없이 뜨고, 조립 훅이 슬롯을 채우고, HTTP 계약(인증·resume 409·첨부 질문)이 지켜지는지.

create_app() 은 STARTUP_CONNECT_EXTERNAL=false 면 Redis·PostgreSQL·Qdrant·MCP 없이 뜬다. 업무 도메인은
setup 훅이 extensions 슬롯에 꽂는다(코어는 도메인을 import 하지 않는다).
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agent_arch.core.config import settings
from agent_arch.extensions import extensions

HEADERS = {"X-User-Sub": "u-1", "X-Forwarded-User": "u1@example.com"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(settings, "startup_connect_external", False, raising=False)
    from agent_arch.main import create_app

    with TestClient(create_app()) as c:
        yield c


def test_app_starts_without_external_dependencies(client):
    r = client.get(f"{settings.api_v1_prefix}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"                     # 외부 연결 없이 떴으므로
    assert body["services"]["postgres"] == "not_connected"


def test_setup_hook_fills_extension_slots(monkeypatch):
    monkeypatch.setattr(settings, "startup_connect_external", False, raising=False)
    monkeypatch.setattr(extensions, "mode_prompts", {})
    from agent_arch.main import create_app

    seen = []

    def setup(ext):
        seen.append(ext)
        ext.mode_prompts["helpdesk"] = lambda surface: "헬프데스크 전용 프롬프트"

    create_app(setup)
    assert seen == [extensions]
    assert "helpdesk" in extensions.mode_prompts


def test_authenticated_route_requires_gateway_header(client):
    r = client.post(f"{settings.api_v1_prefix}/agent/x/resume", json={})
    assert r.status_code == 401


def test_dev_auth_mode_passes_without_header(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_mode", "dev", raising=False)
    with patch("agent_arch.api.v1.agent.get_conversation", new=AsyncMock(return_value=None)):
        r = client.post(f"{settings.api_v1_prefix}/agent/conv-1/resume", json={})
    assert r.status_code == 404                              # 인증은 통과, 대화가 없을 뿐


def test_resume_refuses_stale_card_with_409(client):
    from agent_arch.agents.hitl import StalePendingAction

    pending = {"tool_name": "mail_send_message", "arguments": {}, "pending_id": "p-new"}
    with patch("agent_arch.api.v1.agent.get_conversation", new=AsyncMock(return_value={"id": "c"})), \
         patch("agent_arch.agents.hitl.load_pending", new=AsyncMock(return_value=pending)), \
         patch("agent_arch.agents.hitl.resume_action", new=AsyncMock(side_effect=StalePendingAction("stale"))):
        r = client.post(
            f"{settings.api_v1_prefix}/agent/c/resume",
            json={"action": "approve", "values": {}, "pending_id": "p-old"},
            headers=HEADERS,
        )
    assert r.status_code == 409


def test_resume_passes_pending_id_through(client):
    pending = {"tool_name": "mail_send_message", "arguments": {}, "pending_id": "p-1"}
    resume = AsyncMock(return_value={"success": True, "view": None, "next_pending": None})
    with patch("agent_arch.api.v1.agent.get_conversation", new=AsyncMock(return_value={"id": "c"})), \
         patch("agent_arch.agents.hitl.load_pending", new=AsyncMock(return_value=pending)), \
         patch("agent_arch.agents.hitl.resume_action", new=resume), \
         patch("agent_arch.agents.hitl.followups.followup_note", new=AsyncMock(return_value="")), \
         patch("agent_arch.api.v1.agent._persist_action_outcome", new=AsyncMock()), \
         patch("agent_arch.api.v1.agent._maybe_continue", new=AsyncMock(return_value=None)):
        r = client.post(
            f"{settings.api_v1_prefix}/agent/c/resume",
            json={"action": "approve", "values": {}, "pending_id": "p-1"},
            headers=HEADERS,
        )
    assert r.status_code == 200, r.text
    assert resume.await_args.kwargs["pending_id"] == "p-1"


def test_text_attachment_is_prepended_and_question_kept_separately(client):
    """평문 첨부는 질문 앞에 붙어 모델로 가고, 사용자가 쓴 질문은 turn_question 으로 따로 간다."""
    captured = {}

    class FakeOrchestrator:
        def __init__(self, **kwargs):
            self.llm_provider = "gemini"
            self.agent_mode = kwargs.get("agent_mode")
            self.static_tools = []

        async def run(self, input, conversation_id, user_info, images):
            captured["input"] = input
            captured["question"] = getattr(self, "turn_question", None)
            return {"success": True, "output": "ok", "llm_provider": "gemini"}

    with patch("agent_arch.api.v1.agent.AgentOrchestrator", FakeOrchestrator), \
         patch("agent_arch.api.v1.agent.create_conversation", new=AsyncMock(return_value="conv-new")), \
         patch("agent_arch.api.v1.agent._apply_web_search", new=AsyncMock(return_value=False)):
        r = client.post(
            f"{settings.api_v1_prefix}/agent/run",
            data={"input": "합계 알려줘"},
            files={"files": ("rows.csv", b"name,amount\nA,100\nB,200\n", "text/csv")},
            headers=HEADERS,
        )
    assert r.status_code == 200, r.text
    assert captured["input"].endswith("합계 알려줘") and "A,100" in captured["input"]
    assert captured["question"] == "합계 알려줘"


def test_unsupported_attachment_is_400(client):
    with patch("agent_arch.api.v1.agent.create_conversation", new=AsyncMock(return_value="conv-new")):
        r = client.post(
            f"{settings.api_v1_prefix}/agent/run",
            data={"input": "봐줘"},
            files={"files": ("a.docx", b"PK..", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
            headers=HEADERS,
        )
    assert r.status_code == 400
