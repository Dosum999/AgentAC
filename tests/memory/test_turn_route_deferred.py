"""이번 턴 라우트는 답이 저장될 때 확정 대화 id 로 남는다.

전에는 supervisor 가 답 전에 저장해서 (1) 실패해 저장되지 않은 턴의 라우트가 남았고
(2) 새 대화 첫 턴은 대화 id 가 없어 아예 남지 않았다(2턴째 후속 상속이 직전 도구 이름 폴백에 의존).
"""
from unittest.mock import AsyncMock

import pytest

from agent_arch.memory import conversation as conv


@pytest.fixture
def mocks(monkeypatch):
    pg = AsyncMock(return_value=("mid-1", "conv-new"))
    monkeypatch.setattr(conv, "pg_save_turn", pg)
    monkeypatch.setattr(conv, "append_turn", AsyncMock())
    saved = AsyncMock()
    monkeypatch.setattr(conv, "save_last_route", saved)
    monkeypatch.setattr("agent_arch.services.memory_v2.on_turn_saved", lambda *a, **k: None)
    return pg, saved


@pytest.mark.asyncio
async def test_first_turn_route_saved_with_new_conversation_id(mocks):
    _, saved = mocks
    conv.begin_turn_route()
    assert conv.record_turn_route(["web"]) is True
    await conv.save_logged_in_turn(None, "u1", "질문", "답")
    saved.assert_awaited_once_with("conv-new", ["web"])


@pytest.mark.asyncio
async def test_route_not_saved_when_turn_not_saved(mocks):
    _, saved = mocks
    conv.begin_turn_route()
    conv.record_turn_route(["mail"])
    # 답 저장이 일어나지 않으면 라우트도 남지 않는다.
    saved.assert_not_awaited()


@pytest.mark.asyncio
async def test_record_outside_turn_returns_false():
    conv._turn_route.set(None)
    assert conv.record_turn_route(["hr"]) is False


@pytest.mark.asyncio
async def test_supervisor_records_instead_of_saving_inside_turn(mocks):
    from agent_arch.agents.graphs.supervisor import _save_route

    _, saved = mocks
    conv.begin_turn_route()
    await _save_route("conv-old", ["calendar"])
    saved.assert_not_awaited()          # 답 저장 전에는 쓰지 않는다
    await conv.save_logged_in_turn("conv-old", "u1", "질문", "답")
    saved.assert_awaited_once_with("conv-new", ["calendar"])  # pg 가 돌려준 확정 id
