"""``list_conversations_for_user`` 단위 테스트.

집계 컬럼(``message_count``, ``rated_count``)을 LEFT JOIN으로 계산하는
SQL의 분기·바인딩 검증.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import conversations as conv_module
from agent_arch.db.repository.conversations import list_conversations_for_user


class _FakePool:
    def __init__(self, rows: Optional[List[dict]] = None):
        self.rows = rows or []
        self.last_sql: Optional[str] = None
        self.last_params: tuple = ()

    async def fetch(self, sql: str, *params: Any) -> list:
        self.last_sql = sql
        self.last_params = params
        return list(self.rows)


@pytest.fixture
def fake_pool(monkeypatch):
    pool = _FakePool()
    monkeypatch.setattr(conv_module.db_pool, "get_pool", lambda: pool)
    return pool


class TestListConversationsForUser:
    """집계·필터·페이지네이션 SQL 검증."""

    @pytest.mark.asyncio
    async def test_pool_none_returns_empty(self, monkeypatch):
        monkeypatch.setattr(conv_module.db_pool, "get_pool", lambda: None)
        assert await list_conversations_for_user("u1", limit=10) == []

    @pytest.mark.asyncio
    async def test_default_filters_archived_excluded(self, fake_pool):
        """기본 호출 시 archived 방 제외 + user_sub 매칭."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "c.user_sub = $1" in sql
        assert "c.archived_at IS NULL" in sql

    @pytest.mark.asyncio
    async def test_aggregate_columns_present(self, fake_pool):
        """집계 컬럼 4개가 SELECT에 정확히 포함."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "COUNT(m.id)::int" in sql
        assert "AS message_count" in sql
        assert "AS rated_count" in sql
        assert "AS positive_count" in sql
        assert "AS negative_count" in sql

    @pytest.mark.asyncio
    async def test_rated_count_excludes_empty_rating(self, fake_pool):
        """빈 문자열 rating은 미평가 — classify_rating과 같은 기준이어야 한다."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "m.rating IS NOT NULL AND m.rating <> ''" in sql
        assert "FILTER (WHERE m.rating IS NOT NULL)" not in sql

    @pytest.mark.asyncio
    async def test_positives_param_bound_before_limit(self, fake_pool):
        """positives 배열은 LIMIT 앞에 바인딩 — 기존 $n 번호가 밀리면 안 된다."""
        await list_conversations_for_user("u1", limit=20)
        assert fake_pool.last_params[-1] == 21
        assert fake_pool.last_params[-2] == ["thumbs_up"]

    @pytest.mark.asyncio
    async def test_limit_position_stable_with_all_filters(self, fake_pool):
        """cursor + since + until 을 모두 줘도 LIMIT 파라미터는 여전히 마지막."""
        await list_conversations_for_user(
            "u1",
            limit=15,
            cursor=(datetime(2026, 5, 14, 10), "conv-1"),
            since=datetime(2026, 5, 1),
            until=datetime(2026, 5, 20),
        )
        assert fake_pool.last_params[-1] == 16
        assert fake_pool.last_params[-2] == ["thumbs_up"]

    @pytest.mark.asyncio
    async def test_left_join_messages(self, fake_pool):
        """``LEFT JOIN agent.messages``로 빈 방도 결과에 포함된다."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "LEFT JOIN agent.messages m" in sql

    @pytest.mark.asyncio
    async def test_select_omits_llm_internal_columns(self, fake_pool):
        """대화방 응답에도 LLM 내부 필드는 없어야 한다."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        for col in ["llm_provider", "model_name", "tool_calls", "langsmith_run_id"]:
            assert col not in sql

    @pytest.mark.asyncio
    async def test_ascending_order_flips_sort_and_cursor(self, fake_pool):
        """과거순 — ORDER BY 와 cursor 비교 연산자가 함께 뒤집힌다.

        둘 중 하나만 뒤집으면 첫 페이지는 맞고 "더 보기"부터 엉뚱한 구간이 나온다.
        """
        await list_conversations_for_user(
            "u1", limit=10, cursor=(datetime(2026, 5, 14, 10), "conv-1"), order="asc"
        )
        sql = fake_pool.last_sql or ""
        assert "ASC NULLS LAST, c.id ASC" in sql
        assert "DESC NULLS LAST" not in sql
        assert "c.id) > ($" in sql

    @pytest.mark.asyncio
    async def test_sort_key_is_coalesced_in_both_directions(self, fake_pool):
        """정렬 키는 양방향 모두 ``COALESCE(last_message_at, created_at)`` 하나다.

        키가 방향마다 갈리면 "최신순 첫 항목"과 "과거순 마지막 항목"이 서로 달라진다.
        """
        for order in ("desc", "asc"):
            await list_conversations_for_user("u1", limit=10, order=order)
            sql = fake_pool.last_sql or ""
            assert "ORDER BY COALESCE(c.last_message_at, c.created_at)" in sql
            assert "NULLS LAST" in sql  # COALESCE 가 걷힐 때를 대비한 방어

    @pytest.mark.asyncio
    async def test_order_nulls_last(self, fake_pool):
        """last_message_at NULL인 빈 방이 정렬 끝에 오도록 ``NULLS LAST``."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "NULLS LAST" in sql
        assert "DESC NULLS LAST, c.id DESC" in sql

    @pytest.mark.asyncio
    async def test_limit_plus_one(self, fake_pool):
        """LIMIT은 limit+1 — 호출자가 has_more 판단."""
        await list_conversations_for_user("u1", limit=20)
        assert fake_pool.last_params[-1] == 21

    @pytest.mark.asyncio
    async def test_cursor_appends_tuple_compare(self, fake_pool):
        """cursor 전달 시 ``(COALESCE(last_message_at, created_at), id) < (...)`` 절."""
        cur = (datetime(2026, 5, 14, 10), "conv-1")
        await list_conversations_for_user("u1", limit=10, cursor=cur)
        sql = fake_pool.last_sql or ""
        assert "(COALESCE(c.last_message_at, c.created_at), c.id) <" in sql

    @pytest.mark.asyncio
    async def test_since_and_until_filter_use_coalesce(self, fake_pool):
        """since/until은 ``COALESCE(last_message_at, created_at)`` 기준."""
        since = datetime(2026, 1, 1)
        until = datetime(2026, 6, 1)
        await list_conversations_for_user("u1", limit=10, since=since, until=until)
        sql = fake_pool.last_sql or ""
        assert "COALESCE(c.last_message_at, c.created_at) >=" in sql
        assert "COALESCE(c.last_message_at, c.created_at) <=" in sql
        assert since in fake_pool.last_params
        assert until in fake_pool.last_params

    @pytest.mark.asyncio
    async def test_returns_rows_as_dicts(self, fake_pool):
        """fake가 돌려준 rows를 dict 리스트로 반환."""
        fake_pool.rows = [{"id": "c1", "message_count": 5, "rated_count": 2}]
        rows = await list_conversations_for_user("u1", limit=10)
        assert rows == fake_pool.rows


class TestPosInternalFilter:
    """POS 플래닝 턴 숨김 — 집계·first_query 가 메시지 목록 API 와 같은 기준이어야 한다."""

    @pytest.mark.asyncio
    async def test_join_on_clause_excludes_pos_internal(self, fake_pool):
        """술어가 LEFT JOIN ON 절에 있어야 한다 — WHERE 면 빈 방 행이 사라진다."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        join_idx = sql.index("LEFT JOIN agent.messages m")
        where_idx = sql.index("WHERE c.user_sub")
        pred_idx = sql.index("(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'")
        assert join_idx < pred_idx < where_idx

    @pytest.mark.asyncio
    async def test_first_query_subquery_excludes_pos_internal(self, fake_pool):
        """제목 폴백(first_query)도 숨김 행을 건너뛴다."""
        await list_conversations_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "(fm.metadata->>'pos_internal') IS DISTINCT FROM 'true'" in sql
