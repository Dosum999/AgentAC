"""예시 도메인(헬프데스크)이 코어 수정 없이 끝까지 도는지 — 관문 발견 → 초안 강제 → 승인 재개 → 후처리.

LLM 은 대본(stream 이벤트 목록)으로 대신한다. 외부 인프라(벡터 인덱스·Redis·DB) 없이 돈다.
"""
import copy

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.context import set_profile_context
from agent_arch.agents.graphs import router
from agent_arch.agents.tools import tool_search
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.examples import helpdesk
from agent_arch.extensions import extensions
from agent_arch.llm.base import Message, StreamEvent, ToolCall

PROFILE = {"is_guest": False, "permissions": {"helpdesk": ["helpdesk.access"]}}


@pytest.fixture
def installed(monkeypatch):
    """install() 을 돌리고 끝나면 전역 상태(도메인 표·슬롯·레지스트리·저장소)를 되돌린다."""
    saved_slots = {k: copy.copy(v) for k, v in vars(extensions).items()}
    saved_domains = dict(tool_search._DOMAIN_TOOLS)
    saved_prefixes = dict(tool_search.EXTRA_TOOL_PREFIXES)
    monkeypatch.setattr(helpdesk, "store", helpdesk.TicketStore.sample())
    helpdesk.install()
    set_profile_context(PROFILE)
    yield
    set_profile_context(None)
    for k, v in saved_slots.items():
        setattr(extensions, k, v)
    tool_search._DOMAIN_TOOLS.clear()
    tool_search._DOMAIN_TOOLS.update(saved_domains)
    tool_search.EXTRA_TOOL_PREFIXES.clear()
    tool_search.EXTRA_TOOL_PREFIXES.update(saved_prefixes)
    for cls in helpdesk.TOOLS:
        ToolRegistry.unregister(cls().name)
    ToolRegistry.unregister("find_helpdesk_tools")


def _llm(*streams):
    gens = list(streams)
    llm = MagicMock()
    llm.stream = lambda *a, **k: gens.pop(0)(*a, **k)
    llm.bind_tools = AsyncMock()
    return llm


def _calls(*tcs):
    async def gen(*a, **k):
        yield StreamEvent(type="tool_calls", tool_calls=list(tcs))
        yield StreamEvent(type="done", finish_reason="tool_calls")
    return gen


def _text(t):
    async def gen(*a, **k):
        yield StreamEvent(type="text", content=t)
        yield StreamEvent(type="done", finish_reason="stop")
    return gen


def test_domain_is_known_to_router_without_core_edits(installed):
    assert "helpdesk" in router.ROUTABLE_DOMAINS
    assert router.FIND_TOOL_FOR_DOMAIN.get("helpdesk") == "find_helpdesk_tools"
    assert router.domain_from_tool_names(["helpdesk_get_ticket"]) == "helpdesk"
    # 도메인 도구가 글로벌 유틸리티로 새어 모든 워커에 얹히지 않는다.
    assert not router.is_domainless_utility(ToolRegistry.get("helpdesk_get_ticket"))
    assert "- helpdesk: 헬프데스크" in router._build_router_system_prompt(None)


@pytest.mark.asyncio
async def test_gateway_returns_domain_tools_even_without_index(installed, monkeypatch):
    from agent_arch.services.api_registry import api_indexer

    monkeypatch.setattr(api_indexer, "search", AsyncMock(side_effect=RuntimeError("인덱스 없음")))
    tool_search.reset_discovered_tools()
    result = await ToolRegistry.get("find_helpdesk_tools").execute(query="티켓 닫기")
    assert result.success
    assert {t.name for t in tool_search.drain_discovered_tools()} >= {
        "helpdesk_list_tickets", "helpdesk_get_ticket", "helpdesk_close_ticket"}


@pytest.mark.asyncio
async def test_gateway_hides_tools_without_permission(installed, monkeypatch):
    from agent_arch.services.api_registry import api_indexer

    monkeypatch.setattr(api_indexer, "search", AsyncMock(return_value=[]))
    set_profile_context({"is_guest": False, "permissions": {}})
    tool_search.reset_discovered_tools()
    await ToolRegistry.get("find_helpdesk_tools").execute(query="티켓")
    assert tool_search.drain_discovered_tools() == []


@pytest.mark.asyncio
async def test_close_is_forced_to_draft_then_runs_only_on_resume(installed):
    from agent_arch.agents.graphs.react_graph import run_react_graph
    from agent_arch.agents.hitl import StalePendingAction, get_action_graph, resume_action

    tools = [ToolRegistry.get(n) for n in ("helpdesk_get_ticket", "helpdesk_close_ticket")]
    # 모델이 처음부터 confirm=True 로 실행하려 한다 — 코어가 초안으로 바꿔야 한다.
    llm = _llm(
        _calls(ToolCall(id="c1", name="helpdesk_close_ticket",
                        arguments={"ticket_id": 2, "resolution": "모니터 교체 완료", "confirm": True})),
        _text("2번 티켓 종료 초안을 만들었어요. 카드에서 승인해 주세요."),
    )
    result = await run_react_graph(llm, "2번 티켓 닫아줘", [Message(role="user", content="2번 티켓 닫아줘")],
                                   tools, AsyncMock(return_value=[]), max_iterations=4)

    assert result.finish_reason == "action_pending"
    assert helpdesk.store.tickets[2].status == "open"            # 아직 아무것도 안 바뀌었다
    pending = result.pending_queue[0]
    assert pending["tool_name"] == "helpdesk_close_ticket"
    assert pending["draft_view"]["kind"] == "draft"

    # 오케스트레이터가 하는 일: 대기 항목에 pending_id 를 붙여 HITL 그래프에 적재한다.
    cid = "helpdesk-e2e-1"
    await get_action_graph().ainvoke({"pending_action": {**pending, "pending_id": "p-1"}},
                                     {"configurable": {"thread_id": cid}})
    with pytest.raises(StalePendingAction):
        await resume_action(cid, {}, pending_id="p-old")       # 옛 카드 버튼은 실행되지 않는다
    assert helpdesk.store.tickets[2].status == "open"

    done = await resume_action(cid, {}, pending_id="p-1")
    assert done["success"] is True
    assert helpdesk.store.tickets[2].status == "closed"
    assert "종료 알림" in done["view"]["result_note"]            # 승인 후처리 훅
    assert helpdesk.store.notifications == ["이영희: #2 종료 — 모니터 교체 완료"]

    from agent_arch.agents.hitl.policy import action_done_message
    assert action_done_message("helpdesk_close_ticket") == "티켓을 종료했어요."


def test_dedicated_channel_mode_uses_domain_prompt(installed):
    from agent_arch.agents.prompts.system import build_system_prompt

    prompt = build_system_prompt({"sub": "u1"}, mode="helpdesk", include_turn_context=False)
    assert "헬프데스크 담당 에이전트" in prompt


@pytest.fixture
def example_client(installed, monkeypatch):
    from fastapi.testclient import TestClient

    from agent_arch.core.config import settings

    monkeypatch.setattr(settings, "startup_connect_external", False, raising=False)
    monkeypatch.setattr(settings, "auth_mode", "dev", raising=False)
    with TestClient(helpdesk.create_example_app()) as c:
        yield c, settings.api_v1_prefix


@pytest.mark.asyncio
async def test_example_app_resume_over_http(example_client, monkeypatch):
    """HTTP 로도 같은 계약: 옛 카드 id → 409, 맞는 id → 실행 + 완료 문구."""
    from agent_arch.agents.hitl import get_action_graph
    from agent_arch.api.v1 import agent as agent_api

    client, prefix = example_client
    monkeypatch.setattr(agent_api, "get_conversation", AsyncMock(return_value={"id": "c"}))
    monkeypatch.setattr(agent_api, "_persist_action_outcome", AsyncMock())
    monkeypatch.setattr(agent_api, "_maybe_continue", AsyncMock(return_value=None))

    cid = "helpdesk-http-1"
    draft = (await ToolRegistry.get("helpdesk_close_ticket").execute(ticket_id=1, resolution="VPN 재설정")).output
    await get_action_graph().ainvoke({"pending_action": {
        "action_id": "a1", "tool_name": "helpdesk_close_ticket", "pending_id": "p-9",
        "arguments": {"ticket_id": 1, "resolution": "VPN 재설정"}, "draft_view": draft["view"],
    }}, {"configurable": {"thread_id": cid}})

    stale = client.post(f"{prefix}/agent/{cid}/resume", json={"pending_id": "p-1"})
    assert stale.status_code == 409 and helpdesk.store.tickets[1].status == "open"

    ok = client.post(f"{prefix}/agent/{cid}/resume", json={"pending_id": "p-9"})
    assert ok.status_code == 200, ok.text
    assert helpdesk.store.tickets[1].status == "closed"
    assert "티켓을 종료했어요." in ok.text


def test_example_app_health(example_client):
    client, prefix = example_client
    assert client.get(f"{prefix}/health").status_code == 200
