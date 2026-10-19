"""저장 깔때기가 답변 메타(웹 검색 사용 여부)를 요청 스코프에서 흘려보내는가.

`save_logged_in_turn` 은 저장 호출 지점이 10곳이라, 첨부·회사·서비스와 마찬가지로 이
축도 파라미터가 아니라 ContextVar 에서 읽는다. 한 경로라도 파라미터로 뚫으면 그 경로만
조용히 값이 빠진다 — 실제로 그렇게 빠져 있던 축이라 회귀 가드를 둔다.

경계: 미기록(None)과 '안 썼다'(False)가 다른 사실이라 None 을 남긴다.
False 로 접으면 그 턴들이 "웹 없이 답한 턴"으로 집계된다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.context import set_user_context, set_web_search_enabled
from agent_arch.memory import conversation as conv_module


@pytest.fixture
def save_mocks(monkeypatch):
    pg = AsyncMock(return_value=("mid-1", "conv-1"))
    monkeypatch.setattr(conv_module, "pg_save_turn", pg)
    monkeypatch.setattr(conv_module, "append_turn", AsyncMock())
    monkeypatch.setattr("agent_arch.services.memory_v2.on_turn_saved", MagicMock())
    # ContextVar 잔존값 격리 — 앞 테스트의 요청이 다음 턴으로 새지 않게 한다.
    set_user_context({})
    set_web_search_enabled(None)
    return pg


async def _save():
    await conv_module.save_logged_in_turn(
        "conv-1", "user-sub", "질문", "답변", save_to_redis=False
    )


@pytest.mark.asyncio
async def test_saves_the_resolved_web_search_choice(save_mocks):
    set_web_search_enabled(True)

    await _save()

    assert save_mocks.await_args.kwargs["web_search"] is True


@pytest.mark.asyncio
async def test_records_declined_web_search_as_false_not_missing(save_mocks):
    """웹을 쓰지 않은 턴은 False 로 남는다 — 미기록(None)과 다른 사실이다."""
    set_web_search_enabled(False)

    await _save()

    assert save_mocks.await_args.kwargs["web_search"] is False


@pytest.mark.asyncio
async def test_outside_the_panel_web_search_stays_unknown(save_mocks):
    """패널 토글을 안 거친 경로(메신저 웹훅·승인 카드 버튼 턴)는 '안 썼다'로 단정할 수 없다."""
    await _save()

    kwargs = save_mocks.await_args.kwargs
    assert "answer_mode" not in kwargs
    assert kwargs["web_search"] is None
