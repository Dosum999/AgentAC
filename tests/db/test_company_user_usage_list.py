"""``list_company_user_usage`` 단위 테스트.

DMP '직원별 LLM 사용량' 탭의 소스다. 회사 필터나 LLM 턴 판별이 어긋나면 다른 회사 사용량이
섞이거나 승인 카드 기록이 호출 수를 부풀린다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import messages as messages_module
from agent_arch.db.repository.messages import list_company_user_usage


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
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)
    return pool


@pytest.mark.asyncio
async def test_returns_empty_when_db_disabled(monkeypatch):
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: None)
    assert await list_company_user_usage("회사 사업 본부") == []


@pytest.mark.asyncio
async def test_filters_by_request_company_and_llm_turns(fake_pool):
    await list_company_user_usage("회사 사업 본부")
    sql = fake_pool.last_sql
    # 회사는 요청 시점의 선택 회사다 — 다중소속 사용자의 다른 회사 사용량이 섞이면 안 된다.
    assert "m.metadata->>'company' = $1" in sql
    # 승인/거절 액션 기록은 LLM 을 부르지 않는다 — 일별 원장과 같은 판별.
    assert "(m.llm_invoke_count IS NOT NULL OR m.total_tokens IS NOT NULL)" in sql
    assert "GROUP BY 1, 2, 3" in sql
    assert fake_pool.last_params == ("회사 사업 본부",)


@pytest.mark.asyncio
async def test_time_range_binds_naive_utc(fake_pool):
    kst = timezone(timedelta(hours=9))
    await list_company_user_usage(
        "(주)회사",
        start=datetime(2026, 10, 1, 9, 0, tzinfo=kst),
        end=datetime(2026, 10, 8),
    )
    sql = fake_pool.last_sql
    assert "m.created_at >= $2" in sql
    assert "m.created_at <= $3" in sql
    # created_at 은 naive UTC — aware 를 그대로 바인딩하면 asyncpg 가 죽는다.
    assert fake_pool.last_params == ("(주)회사", datetime(2026, 10, 1, 0, 0), datetime(2026, 10, 8))


@pytest.mark.asyncio
async def test_maps_rows(fake_pool):
    fake_pool.rows = [
        {
            "user_sub": "sub-A", "user_email": "a@x.kr", "provider": "gemini", "turns": 3,
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
        }
    ]
    assert await list_company_user_usage("(주)회사") == [
        {
            "user_sub": "sub-A", "user_email": "a@x.kr", "provider": "gemini", "turns": 3,
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110,
        }
    ]
