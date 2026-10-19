"""``list_rated_messages_for_user`` 단위 테스트.

대화방을 가로질러 "내가 평가한 turn"만 모아 오는 쿼리. asyncpg pool을 in-memory fake로
대체해 SQL/params 구성을 검증한다 — 소유자 격리·숨김 턴 제외·평가 판정·cursor·정렬이
의도대로 반영되는지가 핵심.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import messages as messages_module
from agent_arch.db.repository.messages import list_rated_messages_for_user


class _FakePool:
    """asyncpg pool 인터페이스 일부를 모방. 마지막 fetch 인자를 기록."""

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
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)
    return pool


def _select_clause(sql: str) -> str:
    """SELECT 절만 잘라낸다 — ``m.metadata``는 WHERE에 정당하게 등장하므로."""
    return sql[: sql.index("FROM agent.messages")]


class TestGuards:
    @pytest.mark.asyncio
    async def test_pool_none_returns_empty(self, monkeypatch):
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: None)
        assert await list_rated_messages_for_user("u1", limit=10) == []

    @pytest.mark.asyncio
    async def test_blank_user_sub_returns_empty(self, fake_pool):
        assert await list_rated_messages_for_user("", limit=10) == []
        assert fake_pool.last_sql is None


class TestSql:
    @pytest.mark.asyncio
    async def test_default_clauses(self, fake_pool):
        """소유자·아카이브·숨김 턴·평가됨 4종이 기본으로 걸린다."""
        await list_rated_messages_for_user("u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "c.user_sub = $1" in sql
        assert "c.archived_at IS NULL" in sql
        assert "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'" in sql
        assert "m.rating IS NOT NULL AND m.rating <> ''" in sql
        assert fake_pool.last_params[0] == "u1"

    @pytest.mark.asyncio
    async def test_owner_is_conversation_not_message(self, fake_pool):
        """``m.user_sub``는 v19 이후 컬럼이라 과거 행이 NULL — 소유 판정에 쓰면 안 된다."""
        await list_rated_messages_for_user("u1", limit=10)
        assert "m.user_sub" not in (fake_pool.last_sql or "")

    @pytest.mark.asyncio
    async def test_negative_filter_binds_positive_presets(self, fake_pool):
        await list_rated_messages_for_user("u1", limit=10, rating_filter="negative")
        sql = fake_pool.last_sql or ""
        assert "m.rating <> ALL($2::text[])" in sql
        assert fake_pool.last_params[1] == ["thumbs_up"]

    @pytest.mark.asyncio
    async def test_unknown_filter_falls_back_to_rated(self, fake_pool):
        """라우터가 막지만 repository도 닫는다 — 미평가 전량 유출 방지."""
        await list_rated_messages_for_user("u1", limit=10, rating_filter="any")
        sql = fake_pool.last_sql or ""
        assert "m.rating IS NOT NULL AND m.rating <> ''" in sql
        assert "m.rating IS NULL" not in sql

    @pytest.mark.asyncio
    async def test_cursor_appends_tuple_compare(self, fake_pool):
        await list_rated_messages_for_user(
            "u1", limit=10, cursor=(datetime(2026, 9, 3, 15, 33), "m-1")
        )
        sql = fake_pool.last_sql or ""
        assert "(m.created_at, m.id) < ($2::timestamp, $3::uuid)" in sql

    @pytest.mark.asyncio
    async def test_since_until_bound(self, fake_pool):
        await list_rated_messages_for_user(
            "u1", limit=10, since=datetime(2026, 9, 1), until=datetime(2026, 9, 4)
        )
        sql = fake_pool.last_sql or ""
        assert "m.created_at >= $2" in sql
        assert "m.created_at <= $3" in sql

    @pytest.mark.asyncio
    async def test_order_and_limit_plus_one(self, fake_pool):
        await list_rated_messages_for_user("u1", limit=20)
        sql = fake_pool.last_sql or ""
        assert "ORDER BY m.created_at DESC, m.id DESC" in sql
        assert fake_pool.last_params[-1] == 21

    @pytest.mark.asyncio
    async def test_ascending_order_flips_sort_and_cursor(self, fake_pool):
        """과거순 — 정렬과 cursor 비교 연산자가 함께 뒤집힌다."""
        await list_rated_messages_for_user(
            "u1", limit=10, cursor=(datetime(2026, 9, 3, 15, 33), "m-1"), order="asc"
        )
        sql = fake_pool.last_sql or ""
        assert "ORDER BY m.created_at ASC, m.id ASC" in sql
        assert "(m.created_at, m.id) > (" in sql

    @pytest.mark.asyncio
    async def test_selects_conversation_title(self, fake_pool):
        """어느 방의 턴인지 알아야 목록에서 대화방으로 점프할 수 있다."""
        await list_rated_messages_for_user("u1", limit=10)
        assert "c.title AS conversation_title" in _select_clause(fake_pool.last_sql or "")

    @pytest.mark.asyncio
    async def test_select_omits_llm_internal_columns(self, fake_pool):
        """SELECT 절에 LLM 내부 필드가 절대 없다 (노출 정책)."""
        await list_rated_messages_for_user("u1", limit=10)
        select = _select_clause(fake_pool.last_sql or "")
        for col in [
            "llm_provider",
            "model_name",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "llm_invoke_count",
            "tool_calls",
            "langsmith_run_id",
            "session_id",
            "metadata",
        ]:
            assert col not in select, f"SELECT 절에 내부 컬럼 누출: {col}"

    @pytest.mark.asyncio
    async def test_returns_plain_dicts(self, monkeypatch):
        rows = [{"id": "m1", "conversation_id": "c1", "rating": "속도가 느림"}]
        pool = _FakePool(rows=rows)
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

        out = await list_rated_messages_for_user("u1", limit=10)
        assert out == rows
        assert all(isinstance(r, dict) for r in out)
