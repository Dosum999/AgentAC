"""``search_user_conversations``(recall_conversation 도구용 pg_trgm 검색) SQL 검증.

POS 플래닝 턴은 판단 턴과 query 가 동일해 필터가 없으면 회상 결과가 중복 히트하고
answer="" 행이 섞인다 — 이력 조회 API 의 숨김 필터와 같은 기준을 강제한다.
"""
from __future__ import annotations

from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import memory_records as mr_module
from agent_arch.db.repository.memory_records import search_user_conversations


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
    monkeypatch.setattr(mr_module.db_pool, "get_pool", lambda: pool)
    return pool


class TestSearchUserConversations:
    @pytest.mark.asyncio
    async def test_owner_isolation_and_pos_filter(self, fake_pool):
        await search_user_conversations("u1", "개발실")
        sql = fake_pool.last_sql or ""
        assert "user_sub = $1" in sql
        assert "(metadata->>'pos_internal') IS DISTINCT FROM 'true'" in sql

    @pytest.mark.asyncio
    async def test_pool_none_returns_empty(self, monkeypatch):
        monkeypatch.setattr(mr_module.db_pool, "get_pool", lambda: None)
        assert await search_user_conversations("u1", "개발실") == []
