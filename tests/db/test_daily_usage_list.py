"""``list_daily_usage`` 단위 테스트.

asyncpg pool 을 in-memory fake 로 대체해 SQL/params 구성을 검증한다. 이 집계가 관측 'LLM 사용
대시보드'의 유일한 소스라, 일자 경계(KST)나 신원 보강이 어긋나면 회사별 사용량이 조용히 틀린다.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import messages as messages_module
from agent_arch.db.repository.messages import list_daily_meters, list_daily_usage


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
    assert await list_daily_usage() == []


@pytest.mark.asyncio
async def test_groups_by_kst_day_email_company_model_and_service(fake_pool):
    await list_daily_usage()
    sql = fake_pool.last_sql
    # created_at 은 naive UTC — KST 로 9시간 밀어야 대시보드 표기와 일자가 맞는다.
    assert "m.created_at + interval '9 hours'" in sql
    # 회사가 축에 빠지면 다중소속 사용자의 두 회사 사용량이 한 행으로 뭉쳐 구분이 사라진다.
    # 서비스가 빠지면 메신저·MES 사용량이 브라우저 패널과 한 행으로 뭉쳐 청구를 못 가른다.
    assert "m.service AS service" in sql
    assert "GROUP BY 1, 2, 3, 4, 5" in sql


@pytest.mark.asyncio
async def test_company_comes_from_request_metadata(fake_pool):
    # 소속이 2곳 이상인 사용자는 common_app 이 Employee 로 역추적하지 못해 '미매핑'으로 빠진다.
    # 어느 회사로 물었는지는 요청 시점에만 있고, 이 컬럼이 유일한 기록이다.
    await list_daily_usage()
    assert "m.metadata->>'company' AS company" in fake_pool.last_sql


@pytest.mark.asyncio
async def test_email_comes_from_users_join(fake_pool):
    # messages.user_email 은 스냅샷이라 비어 있는 턴이 있다. 신원이 있는데 NULL 로 집계되면
    # 그 사용량이 회사가 아니라 '미매핑'으로 새어 나간다.
    await list_daily_usage()
    assert "LEFT JOIN agent.users u ON u.user_sub = m.user_sub" in fake_pool.last_sql
    assert "COALESCE(m.user_email, u.email) AS user_email" in fake_pool.last_sql


@pytest.mark.asyncio
async def test_day_range_is_inclusive_exclusive(fake_pool):
    from datetime import date
    await list_daily_usage(since_day="2026-07-01", until_day="2026-07-31")
    sql = fake_pool.last_sql
    assert "::date >= $1" in sql
    assert "::date < $2" in sql
    # 문자열이 아니라 date 로 바인딩된다(asyncpg 가 ::date 캐스팅에서 date 를 요구).
    assert fake_pool.last_params == (date(2026, 7, 1), date(2026, 7, 31))


@pytest.mark.asyncio
async def test_no_range_has_no_date_condition(fake_pool):
    await list_daily_usage()
    # FILTER (WHERE ...) 절이 있으므로 'WHERE' 유무가 아니라 날짜 조건 유무로 본다.
    assert "::date >=" not in fake_pool.last_sql
    assert "::date <" not in fake_pool.last_sql
    assert fake_pool.last_params == ()


@pytest.mark.asyncio
async def test_counts_missing_token_rows(fake_pool):
    # 토큰이 NULL 인 턴은 요청 수에는 잡히지만 크레딧에는 안 잡힌다 — 과소 집계를 함께 올린다.
    await list_daily_usage()
    assert "count(*) FILTER (WHERE m.total_tokens IS NULL)" in fake_pool.last_sql


@pytest.mark.asyncio
async def test_counts_failed_calls_separately(fake_pool):
    """호출 실패 턴을 따로 센다 — 토큰 누락과 답변 실패는 다른 사건이다.

    연결 실패·429·503·401·400 은 추론이 시작되지 않아 **토큰 소비가 없다.** 청구 누락이
    아니므로 화면이 이를 "크레딧이 적게 잡혔다"로 읽으면 안 된다. 2026-09-03 dev 실측에서
    남은 미기록 93건 중 31건이 이 실패였다(vLLM 연결 실패·Gemini 503·OpenAI 크레딧 소진).
    """
    await list_daily_usage()
    sql = fake_pool.last_sql
    assert "m.finish_reason IN ('error', 'fallback')" in sql
    # missing_token_rows 의 부분집합이어야 한다 — 총계에서 빼지 않는다.
    assert "WHERE m.total_tokens IS NULL\n                     AND" in sql or (
        "total_tokens IS NULL" in sql and "finish_reason IN" in sql
    )


@pytest.mark.asyncio
async def test_maps_failed_turns_to_push_payload(monkeypatch):
    """계약 필드로 그대로 올라간다 — 이름이 어긋나면 common_app 이 0으로 받는다."""
    pool = _FakePool(rows=[{
        "usage_date": "2026-09-02",
        "user_email": "user@example.com",
        "company": "(주)회사",
        "model_name": "gemma-4-31b-it",
        "service": "datbi_browser",
        "request_count": 5,
        "input_tokens": 700,
        "output_tokens": 300,
        "total_tokens": 1000,
        "missing_token_rows": 3,
        "failed_turns": 2,
        "last_activity_at": datetime(2026, 9, 2, 9, 11),
    }])
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

    rows = await list_daily_usage()

    assert rows[0]["failed_turns"] == 2
    assert rows[0]["missing_token_rows"] == 3


@pytest.mark.asyncio
async def test_excludes_turns_without_llm_call(fake_pool):
    """승인/거절 액션 기록은 LLM 사용량이 아니다 — 이 원장에서 제외한다.

    그 턴은 모델을 부르지 않고 결과만 남기므로 토큰이 없는 것이 정상이다. 그런데 섞이면
    두 숫자가 동시에 틀어진다: 요청 수가 부풀고, ``missing_token_rows`` 가 '토큰 누락'으로
    잡혀 화면이 "크레딧이 실제보다 적게 잡혔다"고 과장한다. 2026-09-03 dev 실측에서
    미기록 367건 중 274건(75%)이 이 액션 기록이었다.
    """
    await list_daily_usage()
    assert (
        "(m.llm_invoke_count IS NOT NULL OR m.total_tokens IS NOT NULL)"
        in fake_pool.last_sql
    )


@pytest.mark.asyncio
async def test_llm_guard_applies_even_without_date_range(fake_pool):
    """경계값: 기간을 안 걸어도 가드는 붙는다(WHERE 절이 조건 유무로 갈리지 않는다)."""
    await list_daily_usage()
    assert "WHERE" in fake_pool.last_sql
    assert fake_pool.last_params == ()


@pytest.mark.asyncio
async def test_llm_guard_keeps_legacy_turns_that_have_tokens(fake_pool):
    """오류예측: llm_invoke_count 컬럼이 없던 시절의 턴을 토큰으로 살린다.

    가드를 ``llm_invoke_count IS NOT NULL`` 단독으로 걸면 그 옛 턴의 사용량이 조용히
    사라진다 — 이미 청구 화면에 보여준 과거 구간이 줄어든다.
    """
    await list_daily_usage()
    assert "OR m.total_tokens IS NOT NULL" in fake_pool.last_sql


@pytest.mark.asyncio
async def test_maps_row_to_push_payload(monkeypatch):
    pool = _FakePool(rows=[{
        "usage_date": "2026-07-30",
        "user_email": "u@slbmetal.kr",
        "company": "(주)에스엘비메탈",
        "model_name": "gemini-2.5-flash",
        "service": "datbi_browser",
        "request_count": 3,
        "input_tokens": 100,
        "output_tokens": 40,
        "total_tokens": 140,
        "missing_token_rows": 1,
        "failed_turns": 0,
        "last_activity_at": datetime(2026, 7, 30, 5, 0),
    }])
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

    (row,) = await list_daily_usage()

    assert row["usage_date"] == "2026-07-30"
    assert row["user_email"] == "u@slbmetal.kr"
    assert row["company"] == "(주)에스엘비메탈"
    assert row["total_tokens"] == 140
    assert row["missing_token_rows"] == 1
    assert row["service"] == "datbi_browser"


@pytest.mark.asyncio
async def test_company_stays_null_for_pre_metadata_turns(monkeypatch):
    # 2026-07-30 이전 저장분엔 metadata.company 가 없다 — NULL 그대로 보내고
    # common_app 이 이메일→Employee 로 폴백한다(임의 값으로 채우면 조용히 틀린 태깅).
    pool = _FakePool(rows=[{
        "usage_date": "2026-07-20",
        "user_email": "u@slbmetal.kr",
        "company": None,
        "model_name": "gemini-2.5-flash",
        # 축 도입 이전 턴은 service 가 NULL 이다 — 그대로 흘려보내 '미분류'로 합산된다.
        "service": None,
        "request_count": 1,
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
        "missing_token_rows": 0,
        "failed_turns": 0,
        "last_activity_at": datetime(2026, 7, 20, 5, 0),
    }])
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

    (row,) = await list_daily_usage()

    assert row["company"] is None
    assert row["service"] is None


@pytest.mark.asyncio
async def test_day_strings_are_bound_as_date_objects(fake_pool):
    # SQL 이 ::date 로 캐스팅하면 asyncpg 가 파라미터 타입을 date 로 추론해 **문자열을 거부**한다
    # ("'str' object has no attribute 'toordinal'" — 2026-07-31 dev 첫 주기에서 실제로 발생).
    from datetime import date
    await list_daily_usage(since_day="2026-07-01", until_day="2026-07-31")
    assert fake_pool.last_params == (date(2026, 7, 1), date(2026, 7, 31))


@pytest.mark.asyncio
async def test_date_objects_pass_through(fake_pool):
    from datetime import date
    await list_daily_usage(since_day=date(2026, 7, 1))
    assert fake_pool.last_params == (date(2026, 7, 1),)


# ─── list_daily_meters: 캐시 열 · 실패 구분 (이슈) ────────────────────────

def _meter_db_row(**kw):
    base = {"usage_date": "2026-10-01", "user_email": "a@d.kr", "company": "(주)회사",
            "model_name": "gpt-5.6-luna", "service": "datbi_browser", "llm_calls": 2, "llm_missing": 0,
            "llm_input_tokens": 1000, "llm_output_tokens": 40, "llm_cached_input_tokens": 600,
            "llm_cache_write_tokens": 100, "brave_search_calls": 0, "openai_search_calls": 0,
            "stt_seconds": 0, "last_activity_at": datetime(2026, 10, 1, 1, 0)}
    base.update(kw)
    return base


@pytest.mark.asyncio
async def test_meters_sum_cache_columns_for_llm_tokens_only(fake_pool):
    fake_pool.rows = [_meter_db_row()]
    out = await list_daily_meters()
    sql = fake_pool.last_sql
    assert ("COALESCE(sum(cached_input_tokens) FILTER (WHERE meter = 'llm_tokens'), 0) "
            "AS llm_cached_input_tokens") in sql
    assert ("COALESCE(sum(cache_write_tokens) FILTER (WHERE meter = 'llm_tokens'), 0) "
            "AS llm_cache_write_tokens") in sql
    assert (out[0]["llm_cached_input_tokens"], out[0]["llm_cache_write_tokens"]) == (600, 100)


@pytest.mark.asyncio
async def test_meters_null_cache_sums_become_zero(fake_pool):
    fake_pool.rows = [_meter_db_row(llm_cached_input_tokens=None, llm_cache_write_tokens=None)]
    out = await list_daily_meters()
    assert (out[0]["llm_cached_input_tokens"], out[0]["llm_cache_write_tokens"]) == (0, 0)


@pytest.mark.asyncio
async def test_meters_missing_table_is_empty(monkeypatch):
    """migration_v42 미적용 인스턴스 — 계측 없이 턴 사용량만 푼다."""
    import asyncpg

    class _NoTable(_FakePool):
        async def fetch(self, sql, *params):
            raise asyncpg.exceptions.UndefinedTableError('relation "agent.usage_meters" does not exist')

    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: _NoTable())
    assert await list_daily_meters() == []


@pytest.mark.asyncio
async def test_meters_other_failures_are_raised_not_swallowed(monkeypatch):
    """열 없음(v45 미적용)·연결 끊김은 '0건'이 아니다 — [] 로 삼키면 그 날이 0 으로 푸시된다."""
    import asyncpg

    class _Broken(_FakePool):
        def __init__(self, exc):
            super().__init__()
            self.exc = exc

        async def fetch(self, sql, *params):
            raise self.exc

    for exc in (asyncpg.exceptions.UndefinedColumnError('column "cached_input_tokens" does not exist'),
                ConnectionError("down")):
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda e=exc: _Broken(e))
        with pytest.raises(type(exc)):
            await list_daily_meters()
