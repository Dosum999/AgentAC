"""save_logged_in_turn 의 POS 턴 정화 훅 테스트.

pg_save_turn / append_turn / schedule_remember 를 mock 으로 갈아끼우고
훅이 전달하는 인자(정화된 query/answer, metadata.pos_raw / pos_internal)와
internal 턴의 Redis·메모리 추출 스킵을 검증한다. 페이로드는 2026-08-28 HAR 실측 형식.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.context import set_user_context
from agent_arch.memory import conversation as conv_module
from agent_arch.utils.pos_history import _POS_RAW_MAX

QUESTION = "개발실에 몇 명 있어?"

TURN1_INPUT = (
    "[POS Agent Catalog]\r\n"
    '{"catalog_version":"a8763c782572d421"}\r\n'
    "[사용자 질문]\r\n"
    f"{QUESTION}\r\n"
    "\r\n"
    "[역할]\r\n- ...\r\n"
)
TURN1_ANSWER = '[POS_QUERY_PLAN]{"intent":"status"}[/POS_QUERY_PLAN]'

TURN2_INPUT = (
    "[POS Query Plan]\r\n"
    '{"intent":"status"}\r\n'
    "[사용자 질문]\r\n"
    f"{QUESTION}\r\n"
    "\r\n"
    "[판단 규칙]\r\n- ...\r\n"
)
TURN2_ANSWER = (
    "현재 개발실에는 10명이 있습니다.\n"
    '[POS_ACTION_PLAN]{"actions":[]}[/POS_ACTION_PLAN]'
)


@pytest.fixture
def save_mocks(monkeypatch):
    """저장 깔때기의 외부 의존 3종을 mock 으로 치환한다."""
    pg = AsyncMock(return_value=("mid-1", "conv-1"))
    redis = AsyncMock()
    monkeypatch.setattr(conv_module, "pg_save_turn", pg)
    monkeypatch.setattr(conv_module, "append_turn", redis)
    # 메모리 캡처 훅은 함수 내부 import — 모듈 속성을 스텁으로 바꾼다
    remember = MagicMock()
    monkeypatch.setattr("agent_arch.services.memory_v2.on_turn_saved", remember)
    # ContextVar 잔존값 격리
    set_user_context({})
    return pg, redis, remember


class TestPosSaveHook:
    @pytest.mark.asyncio
    async def test_decision_turn_sanitized(self, save_mocks):
        pg, redis, remember = save_mocks
        await conv_module.save_logged_in_turn("conv-1", "user-1", TURN2_INPUT, TURN2_ANSWER)

        kwargs = pg.await_args.kwargs
        assert kwargs["user_input"] == QUESTION
        assert kwargs["assistant_output"] == "현재 개발실에는 10명이 있습니다."
        assert kwargs["metadata"]["pos_raw"] == {"input": TURN2_INPUT, "answer": TURN2_ANSWER}
        assert "pos_internal" not in kwargs["metadata"]
        # 판단 턴은 Redis·메모리 추출 정상 동작 (정화본으로)
        redis.assert_awaited_once()
        assert redis.await_args.args[1] == QUESTION
        assert redis.await_args.args[2] == "현재 개발실에는 10명이 있습니다."
        remember.assert_called_once()

    @pytest.mark.asyncio
    async def test_planning_turn_internal(self, save_mocks):
        pg, redis, remember = save_mocks
        mid, cid = await conv_module.save_logged_in_turn(
            None, "user-1", TURN1_INPUT, TURN1_ANSWER
        )

        # PG 저장은 수행 — lazy-create 로 대화방이 생기고 제목이 질문에서 파생돼야 하므로
        kwargs = pg.await_args.kwargs
        assert kwargs["user_input"] == QUESTION
        assert kwargs["assistant_output"] == ""
        assert kwargs["metadata"]["pos_internal"] is True
        assert (mid, cid) == ("mid-1", "conv-1")
        # 플래닝 턴은 Redis 재주입·메모리 추출 스킵
        redis.assert_not_awaited()
        remember.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_pos_turn_unchanged(self, save_mocks):
        # 회귀 가드 — 일반 대화는 바이트 단위로 기존 동작 유지
        pg, redis, remember = save_mocks
        await conv_module.save_logged_in_turn("conv-1", "user-1", "일반 질문", "일반 답변")

        kwargs = pg.await_args.kwargs
        assert kwargs["user_input"] == "일반 질문"
        assert kwargs["assistant_output"] == "일반 답변"
        assert kwargs["metadata"] is None
        redis.assert_awaited_once()
        remember.assert_called_once()

    @pytest.mark.asyncio
    async def test_company_metadata_merge_coexists(self, save_mocks):
        # 기존 company ContextVar 병합 패턴과 pos 메타가 공존해야 한다
        pg, _, _ = save_mocks
        set_user_context({"sub": "user-1", "company": "acme"})
        await conv_module.save_logged_in_turn("conv-1", "user-1", TURN2_INPUT, TURN2_ANSWER)

        metadata = pg.await_args.kwargs["metadata"]
        assert metadata["company"] == "acme"
        assert "pos_raw" in metadata

    @pytest.mark.asyncio
    async def test_pos_raw_truncated(self, save_mocks):
        pg, _, _ = save_mocks
        # 상한 초과 페이로드 — 카탈로그 자리에 대형 JSON 을 끼워 넣는다
        big_input = TURN1_INPUT.replace(
            '{"catalog_version":"a8763c782572d421"}', "{" + "x" * (_POS_RAW_MAX + 1000) + "}"
        )
        await conv_module.save_logged_in_turn("conv-1", "user-1", big_input, TURN1_ANSWER)

        raw = pg.await_args.kwargs["metadata"]["pos_raw"]
        assert len(raw["input"]) == _POS_RAW_MAX
        assert raw["answer"] == TURN1_ANSWER  # 상한 미만은 절단 없음

    @pytest.mark.asyncio
    async def test_extraction_failure_saves_original(self, save_mocks):
        # [사용자 질문] 없는 POS 형식 — 반쪽 정화 대신 원문 그대로 (fail-open)
        pg, redis, _ = save_mocks
        broken = "[POS Agent Catalog]\r\n{...}"
        await conv_module.save_logged_in_turn("conv-1", "user-1", broken, TURN1_ANSWER)

        kwargs = pg.await_args.kwargs
        assert kwargs["user_input"] == broken
        assert kwargs["assistant_output"] == TURN1_ANSWER
        assert kwargs["metadata"] is None
        redis.assert_awaited_once()
