"""저장 깔때기가 답변 메타 두 축(답변 방식·웹 검색)을 요청 스코프에서 흘려보내는가.

`save_logged_in_turn` 은 저장 호출 지점이 10곳이라, 첨부·회사·서비스와 마찬가지로 이 두
축도 파라미터가 아니라 ContextVar 에서 읽는다. 한 경로라도 파라미터로 뚫으면 그 경로만
조용히 값이 빠진다 — 실제로 그렇게 빠져 있던 축이라 회귀 가드를 둔다.

경계는 **두 축의 기본값이 다르다**는 것이다. 답변 방식은 레인이 둘뿐이고 위탁만 ContextVar 를
세우므로, 안 세워진 턴은 기존 ReAct 경로로 답한 것 — 패널 밖 경로(메신저·버튼 턴)도 'quick'
이 사실이다. 웹 검색은 반대로 미기록(None)과 '안 썼다'(False)가 다른 사실이라 None 을 남긴다.
False 로 접으면 그 턴들이 "웹 없이 답한 턴"으로 집계된다.

6기법:
 - 동등 분할 : reasoning+웹 / quick+웹없음 / 미설정
 - 경계값   : web_search=False(기록된 '아니오') vs 미설정(None)
 - 오류 예측 : 요청한 값이 아니라 **확정된 값**이 저장되는가(회사 설정이 막은 reasoning)
 - 원인-결과 : ContextVar → pg_save_turn 인자
 - 의사결정표: (answer_mode, web_search) 조합 → 저장 인자
 - 제외     : 상태전이 — 요청 스코프라 턴마다 새로 시작한다.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.context import set_answer_mode, set_user_context, set_web_search_enabled
from agent_arch.memory import conversation as conv_module


@pytest.fixture
def save_mocks(monkeypatch):
    pg = AsyncMock(return_value=("mid-1", "conv-1"))
    monkeypatch.setattr(conv_module, "pg_save_turn", pg)
    monkeypatch.setattr(conv_module, "append_turn", AsyncMock())
    monkeypatch.setitem(
        sys.modules, "agent_arch.services.memory", types.SimpleNamespace(schedule_remember=MagicMock())
    )
    # ContextVar 잔존값 격리 — 앞 테스트의 요청이 다음 턴으로 새지 않게 한다.
    set_user_context({})
    set_answer_mode(None)
    set_web_search_enabled(None)
    return pg


async def _save():
    await conv_module.save_logged_in_turn(
        "conv-1", "user-sub", "질문", "답변", save_to_redis=False
    )


@pytest.mark.asyncio
async def test_saves_the_resolved_request_choices(save_mocks):
    set_answer_mode("reasoning")
    set_web_search_enabled(True)

    await _save()

    kwargs = save_mocks.await_args.kwargs
    assert kwargs["answer_mode"] == "reasoning"
    assert kwargs["web_search"] is True


@pytest.mark.asyncio
async def test_records_declined_web_search_as_false_not_missing(save_mocks):
    """웹을 쓰지 않은 턴은 False 로 남는다 — 미기록(None)과 다른 사실이다."""
    set_answer_mode("quick")
    set_web_search_enabled(False)

    await _save()

    kwargs = save_mocks.await_args.kwargs
    assert kwargs["answer_mode"] == "quick"
    assert kwargs["web_search"] is False


@pytest.mark.asyncio
async def test_outside_the_panel_the_lane_is_quick_but_web_search_stays_unknown(save_mocks):
    """패널 밖 경로(메신저 웹훅·승인 카드 버튼 턴)의 두 축은 서로 다르게 떨어진다.

    답변 방식은 위탁 레인만 ContextVar 를 세우므로, 안 세워졌다는 것 자체가 기존 ReAct
    경로로 답했다는 뜻이다 — 'quick' 은 지어낸 값이 아니다. 반면 웹 검색은 패널 토글을 안
    거친 것일 뿐이라 '안 썼다'로 단정할 수 없다. 여기서 둘을 같이 접으면 리뷰 화면이
    "웹 없이 답한 턴"을 실제보다 많이 세게 된다.
    """
    await _save()

    kwargs = save_mocks.await_args.kwargs
    assert kwargs["answer_mode"] == "quick"
    assert kwargs["web_search"] is None
