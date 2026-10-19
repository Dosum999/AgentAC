"""``list_review_turns`` 단위 테스트.

asyncpg pool을 in-memory fake로 대체해 SQL/params 구성을 검증한다. 이 쿼리는 회사 리뷰
기록 푸시의 소스라, 경계 조건(since 포함 / until 제외)과 정렬·페이징이 어긋나면 푸시가
구간을 건너뛰거나 같은 구간을 무한히 다시 보낸다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import messages as messages_module
from agent_arch.db.repository.messages import list_review_turns


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


_SINCE = datetime(2026, 7, 1, tzinfo=timezone.utc)
_UNTIL = datetime(2026, 7, 30, tzinfo=timezone.utc)


class TestParams:
    @pytest.mark.asyncio
    async def test_no_range_has_no_time_condition(self, fake_pool):
        await list_review_turns(limit=100, offset=0)
        assert "m.created_at >=" not in fake_pool.last_sql
        assert "m.created_at <" not in fake_pool.last_sql
        assert fake_pool.last_params == (100, 0)

    @pytest.mark.asyncio
    async def test_since_is_inclusive_until_is_exclusive(self, fake_pool):
        await list_review_turns(since=_SINCE, until=_UNTIL, limit=10, offset=20)
        assert "m.created_at >= $1" in fake_pool.last_sql
        assert "m.created_at < $2" in fake_pool.last_sql
        # 컬럼이 naive(timestamp without time zone)라 경계에서 naive UTC 로 바뀐다.
        assert fake_pool.last_params == (_SINCE.replace(tzinfo=None), _UNTIL.replace(tzinfo=None), 10, 20)

    @pytest.mark.asyncio
    async def test_only_until_shifts_placeholders(self, fake_pool):
        await list_review_turns(until=_UNTIL, limit=5, offset=0)
        assert "m.created_at < $1" in fake_pool.last_sql
        assert fake_pool.last_params == (_UNTIL.replace(tzinfo=None), 5, 0)

    @pytest.mark.asyncio
    async def test_orders_oldest_first_with_id_tiebreak(self, fake_pool):
        # 오래된 것부터 밀어야 중간이 비지 않고, 같은 시각 다건에서 offset 경계가 흔들리지
        # 않도록 id 를 2차 키로 둔다.
        await list_review_turns()
        assert "ORDER BY m.created_at ASC, m.id ASC" in fake_pool.last_sql

    @pytest.mark.asyncio
    async def test_selects_company_from_metadata(self, fake_pool):
        # agent.messages 에 company 컬럼이 없어 metadata 에서 읽는다.
        await list_review_turns()
        assert "m.metadata->>'company' AS company" in fake_pool.last_sql

    @pytest.mark.asyncio
    async def test_email_comes_from_users_join(self, fake_pool):
        # messages.user_email 은 저장 시점 스냅샷이라 비어 있는 턴이 많다(카드 승인 실행 턴 등).
        # 그 턴들은 langsmith_run_id 도 없어, 여기서 놓치면 리뷰 화면에 들어올 경로가 없다.
        await list_review_turns()
        assert "LEFT JOIN agent.users u ON u.user_sub = m.user_sub" in fake_pool.last_sql
        assert "COALESCE(m.user_email, u.email) AS user_email" in fake_pool.last_sql

    @pytest.mark.asyncio
    async def test_require_email_filters_guests_by_default(self, fake_pool):
        await list_review_turns()
        assert "COALESCE(m.user_email, u.email) IS NOT NULL" in fake_pool.last_sql
        assert "u.is_guest IS NOT TRUE" in fake_pool.last_sql

    @pytest.mark.asyncio
    async def test_require_email_off_drops_identity_filter(self, fake_pool):
        # 전수 점검(백필 --include-anonymous)용 탈출구.
        await list_review_turns(require_email=False)
        assert "u.is_guest IS NOT TRUE" not in fake_pool.last_sql


class TestRows:
    @pytest.mark.asyncio
    async def test_maps_row_to_push_payload(self, monkeypatch):
        created = datetime(2026, 7, 1, 2, 30, tzinfo=timezone.utc)
        pool = _FakePool(rows=[{
            "id": "0f9c1e64-0000-4000-8000-000000000001",
            "created_at": created,
            "user_email": "u@example.com",
            "query": "q",
            "answer": "a",
            "rating": "thumbs_up",
            "langsmith_run_id": None,
            "company": "테스트메탈",
            "service": "messenger",
            "model_name": "gpt-5.6-luna",
            "response_time": 12.4,
            "web_search": True,
            "finish_reason": None,
            "status_code": None,
            "capture_skip": None,
            "llm_provider": None,
        }])
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

        rows = await list_review_turns()

        assert rows == [{
            "message_id": "0f9c1e64-0000-4000-8000-000000000001",
            "created_at": created,
            "user_email": "u@example.com",
            "company": "테스트메탈",
            "query": "q",
            "answer": "a",
            "rating": "thumbs_up",
            "langsmith_run_id": None,
            "service": "messenger",
            "model_name": "gpt-5.6-luna",
            "response_time": 12.4,
            "web_search": True,
            "finish_reason": None,
            "status_code": None,
            "capture_skip": None,
            "llm_provider": None,
        }]

    @pytest.mark.asyncio
    async def test_selects_service_axis_like_usage_aggregation(self, fake_pool):
        """리뷰 푸시도 사용량 집계와 같은 컬럼을 고른다.

        두 쿼리는 같은 테이블을 같은 JOIN 으로 읽는다. 여기서 m.service 를 빼면 리뷰 화면에서
        메신저 에이전트 대화와 브라우저 패널 대화가 한 목록에 섞여 구분할 방법이 없어진다.
        """
        await list_review_turns()
        assert "m.service" in fake_pool.last_sql

    @pytest.mark.asyncio
    async def test_service_is_none_for_rows_predating_the_axis(self, monkeypatch):
        """축 도입 이전 턴은 None 으로 나간다 — 기본값을 지어내지 않는다."""
        pool = _FakePool(rows=[{
            "id": "0f9c1e64-0000-4000-8000-000000000002",
            "created_at": datetime(2026, 7, 1, 2, 30, tzinfo=timezone.utc),
            "user_email": "u@example.com",
            "query": "q",
            "answer": "a",
            "rating": None,
            "langsmith_run_id": None,
            "company": None,
            "service": None,
            "model_name": None,
            "response_time": None,
            "web_search": None,
            "finish_reason": None,
            "status_code": None,
            "capture_skip": None,
            "llm_provider": None,
        }])
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

        assert (await list_review_turns())[0]["service"] is None

    @pytest.mark.asyncio
    async def test_selects_the_answer_meta_axes(self, fake_pool):
        """답변이 만들어진 조건 세 축도 같은 쿼리에서 고른다.

        모델·소요 시간은 v7 부터 이 표에 있었지만 이 SELECT 에 없어 푸시에 실리지 않았고,
        그래서 PA 리뷰 화면에서 "어떤 조건에서 나온 답인가"를 되짚을 수 없었다.
        """
        await list_review_turns()
        for column in ("m.model_name", "m.response_time", "m.web_search"):
            assert column in fake_pool.last_sql, column

    @pytest.mark.asyncio
    async def test_answer_meta_is_none_for_rows_predating_the_axes(self, monkeypatch):
        """축 도입 이전 턴은 세 값 모두 None 이다 — False 를 지어내지 않는다.

        지어내면 그 순간 과거 전체가 "웹 없이 답한 턴"이 된다.
        """
        pool = _FakePool(rows=[{
            "id": "0f9c1e64-0000-4000-8000-000000000003",
            "created_at": datetime(2026, 7, 1, 2, 30, tzinfo=timezone.utc),
            "user_email": "u@example.com",
            "query": "q",
            "answer": "a",
            "rating": None,
            "langsmith_run_id": None,
            "company": None,
            "service": None,
            "model_name": None,
            "response_time": None,
            "web_search": None,
            "finish_reason": None,
            "status_code": None,
            "capture_skip": None,
            "llm_provider": None,
        }])
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

        row = (await list_review_turns())[0]
        assert [row["model_name"], row["response_time"], row["web_search"]] == [None, None, None]

    @pytest.mark.asyncio
    async def test_returns_empty_when_db_disabled(self, monkeypatch):
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: None)
        assert await list_review_turns() == []


class TestParamBinding:
    """asyncpg 바인딩 방어 — 가짜 pool 은 타입을 검사하지 않아 여기서 명시적으로 고정한다.

    created_at 은 timestamp without time zone(naive UTC)이다. aware datetime 을 그대로
    바인딩하면 실서버에서 'can't subtract offset-naive and offset-aware datetimes' 로 죽는다
    (2026-07-31 dev 첫 주기에서 실제로 발생).
    """

    @pytest.mark.asyncio
    async def test_aware_datetimes_are_converted_to_naive_utc(self, fake_pool):
        await list_review_turns(since=_SINCE, until=_UNTIL)
        since_arg, until_arg = fake_pool.last_params[0], fake_pool.last_params[1]
        assert since_arg.tzinfo is None and until_arg.tzinfo is None
        assert since_arg == datetime(2026, 7, 1)      # UTC 자정 그대로
        assert until_arg == datetime(2026, 7, 30)

    @pytest.mark.asyncio
    async def test_kst_aware_datetime_shifts_to_utc(self, fake_pool):
        kst = timezone(timedelta(hours=9))
        await list_review_turns(since=datetime(2026, 7, 1, 9, 0, tzinfo=kst))
        assert fake_pool.last_params[0] == datetime(2026, 7, 1, 0, 0)  # KST 09:00 = UTC 00:00

    @pytest.mark.asyncio
    async def test_naive_datetime_passes_through(self, fake_pool):
        await list_review_turns(since=datetime(2026, 7, 1, 5, 30))
        assert fake_pool.last_params[0] == datetime(2026, 7, 1, 5, 30)


class TestFailureFields:
    """턴 실패 분류의 원천 네 축(계약 1장) — common_app 이 failure_category 를 파생한다."""

    @pytest.mark.asyncio
    async def test_selects_the_failure_axes(self, fake_pool):
        await list_review_turns()
        sql = fake_pool.last_sql
        assert "m.finish_reason AS finish_reason" in sql
        assert "m.status_code AS status_code" in sql
        # capture_skip 은 컬럼이 아니라 metadata 에 있다(메모리 캡처 스킵 사유).
        assert "m.metadata->>'capture_skip' AS capture_skip" in sql
        assert "m.llm_provider AS llm_provider" in sql

    @pytest.mark.asyncio
    async def test_maps_failure_axes_through(self, monkeypatch):
        pool = _FakePool(rows=[{
            "id": "0f9c1e64-0000-4000-8000-000000000009",
            "created_at": datetime(2026, 7, 1, 2, 30, tzinfo=timezone.utc),
            "user_email": "u@example.com",
            "query": "q",
            "answer": "a",
            "rating": None,
            "langsmith_run_id": None,
            "company": None,
            "service": None,
            "model_name": None,
            "response_time": None,
            "web_search": None,
            "finish_reason": "error",
            "status_code": 500,
            "capture_skip": "tool_error",
            "llm_provider": "gemini",
        }])
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

        row = (await list_review_turns())[0]
        assert (row["finish_reason"], row["status_code"], row["capture_skip"], row["llm_provider"]) == (
            "error", 500, "tool_error", "gemini",
        )
