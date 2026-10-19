"""``list_messages_in_conversation`` / ``get_message_in_conversation`` 단위 테스트.

asyncpg pool을 in-memory fake로 대체해 SQL/params 구성을 검증한다.
실제 PG 연결 없이 분기 로직(필터·cursor·order·since/until)이 의도대로 SQL과 params에
반영되는지가 핵심.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import messages as messages_module
from agent_arch.db.repository.messages import (
    get_message_in_conversation,
    list_messages_in_conversation,
)


class _FakePool:
    """asyncpg pool 인터페이스 일부를 모방. 마지막 fetch/fetchrow 인자를 기록."""

    def __init__(self, rows: Optional[List[dict]] = None, row: Optional[dict] = None):
        self.rows = rows or []
        self.row = row
        self.last_sql: Optional[str] = None
        self.last_params: tuple = ()

    async def fetch(self, sql: str, *params: Any) -> list:
        self.last_sql = sql
        self.last_params = params
        return list(self.rows)

    async def fetchrow(self, sql: str, *params: Any) -> Optional[dict]:
        self.last_sql = sql
        self.last_params = params
        return self.row


@pytest.fixture
def fake_pool(monkeypatch):
    """``db_pool.get_pool()``이 ``_FakePool``을 반환하도록 한다."""
    pool = _FakePool()
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)
    return pool


# ---------------------------------------------------------------------------
# list_messages_in_conversation
# ---------------------------------------------------------------------------


class TestListMessagesParams:
    """SQL 파라미터 바인딩과 필터 절 분기 검증."""

    @pytest.mark.asyncio
    async def test_pool_none_returns_empty(self, monkeypatch):
        """DB 풀 비활성화 시 빈 리스트."""
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: None)
        result = await list_messages_in_conversation("c1", "u1", limit=10)
        assert result == []

    @pytest.mark.asyncio
    async def test_default_query_binds_conv_and_user(self, fake_pool):
        """기본 호출 시 SQL에 conversation_id·user_sub·archived 절이 포함된다."""
        await list_messages_in_conversation("c1", "u1", limit=10)

        sql = fake_pool.last_sql or ""
        assert "m.conversation_id = $1::uuid" in sql
        assert "c.user_sub = $2" in sql
        assert "c.archived_at IS NULL" in sql
        # 기본 정렬: desc
        assert "m.created_at DESC, m.id DESC" in sql
        # LIMIT은 limit+1
        assert fake_pool.last_params[-1] == 11

    @pytest.mark.asyncio
    async def test_select_omits_llm_internal_columns(self, fake_pool):
        """SELECT 컬럼에 LLM 내부 필드가 절대 포함되지 않는다 (노출 정책)."""
        await list_messages_in_conversation("c1", "u1", limit=10)
        sql = fake_pool.last_sql or ""
        forbidden = [
            "llm_provider", "model_name",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "llm_invoke_count", "tool_calls", "langsmith_run_id",
        ]
        for col in forbidden:
            assert col not in sql, f"SELECT에 금지 컬럼 노출: {col}"

    @pytest.mark.asyncio
    async def test_order_asc_reverses_cursor_op(self, fake_pool):
        """order=asc이면 cursor 비교 연산자가 ``>``로 반전된다."""
        cur = (datetime(2026, 5, 14, 10), "msg-1")
        await list_messages_in_conversation("c1", "u1", limit=10, cursor=cur, order="asc")
        sql = fake_pool.last_sql or ""
        assert "(m.created_at, m.id) >" in sql
        assert "m.created_at ASC, m.id ASC" in sql

    @pytest.mark.asyncio
    async def test_order_desc_uses_lt_for_cursor(self, fake_pool):
        """order=desc이면 cursor 비교 연산자가 ``<``로 유지된다."""
        cur = (datetime(2026, 5, 14, 10), "msg-1")
        await list_messages_in_conversation("c1", "u1", limit=10, cursor=cur, order="desc")
        assert "(m.created_at, m.id) <" in fake_pool.last_sql

    @pytest.mark.asyncio
    async def test_rating_positive_uses_any_array(self, fake_pool):
        """rating=positive → m.rating = ANY(text[]) 절 + _POSITIVE_PRESETS 바인딩."""
        from agent_arch.utils.rating import _POSITIVE_PRESETS

        await list_messages_in_conversation(
            "c1", "u1", limit=10, rating_filter="positive"
        )
        sql = fake_pool.last_sql or ""
        assert "m.rating = ANY(" in sql
        # _POSITIVE_PRESETS의 모든 코드값이 바인딩 params에 포함됨
        bound = [p for p in fake_pool.last_params if isinstance(p, list)]
        assert bound, "positive 필터에 list 파라미터가 바인딩돼야 함"
        assert set(bound[0]) == set(_POSITIVE_PRESETS)

    @pytest.mark.asyncio
    async def test_rating_negative_uses_not_any(self, fake_pool):
        """rating=negative → NOT NULL AND <> ALL(_POSITIVE_PRESETS) 절."""
        await list_messages_in_conversation(
            "c1", "u1", limit=10, rating_filter="negative"
        )
        sql = fake_pool.last_sql or ""
        assert "m.rating IS NOT NULL" in sql
        assert "<> ALL(" in sql

    @pytest.mark.asyncio
    async def test_rating_rated_filter(self, fake_pool):
        """rating=rated → m.rating IS NOT NULL."""
        await list_messages_in_conversation(
            "c1", "u1", limit=10, rating_filter="rated"
        )
        assert "m.rating IS NOT NULL" in (fake_pool.last_sql or "")

    @pytest.mark.asyncio
    async def test_rating_unrated_filter(self, fake_pool):
        """rating=unrated → m.rating IS NULL."""
        await list_messages_in_conversation(
            "c1", "u1", limit=10, rating_filter="unrated"
        )
        assert "m.rating IS NULL" in (fake_pool.last_sql or "")

    @pytest.mark.asyncio
    async def test_rating_none_no_where_clause(self, fake_pool):
        """rating=None → WHERE 절에 rating 필터가 추가되지 않음.

        Note:
            SELECT 컬럼에는 ``m.rating``이 항상 포함되므로 SQL 전체에 ``m.rating``
            부재로 검증하면 안 된다. WHERE 절에 들어갈 4가지 필터 마커
            (``IS NULL`` / ``IS NOT NULL`` / ``= ANY`` / ``<> ALL``) 모두 부재해야
            ``rating_filter=None`` 분기가 정상.
        """
        await list_messages_in_conversation("c1", "u1", limit=10, rating_filter=None)
        sql = fake_pool.last_sql or ""
        assert "m.rating IS NULL" not in sql
        assert "m.rating IS NOT NULL" not in sql
        assert "m.rating = ANY" not in sql
        assert "m.rating <> ALL" not in sql
        # SELECT 컬럼은 항상 포함됨
        assert "m.rating," in sql

    @pytest.mark.asyncio
    async def test_since_and_until_bound(self, fake_pool):
        """since/until은 params로 바인딩되고 SQL에 절이 추가된다."""
        since = datetime(2026, 1, 1)
        until = datetime(2026, 6, 1)
        await list_messages_in_conversation(
            "c1", "u1", limit=10, since=since, until=until
        )
        sql = fake_pool.last_sql or ""
        assert "m.created_at >=" in sql
        assert "m.created_at <=" in sql
        assert since in fake_pool.last_params
        assert until in fake_pool.last_params


class TestListMessagesCatalogSoT:
    """``_POSITIVE_PRESETS`` 카탈로그 확장이 필터 SQL에 즉시 반영되는지 회귀 보호."""

    @pytest.mark.asyncio
    async def test_positive_catalog_extension_reaches_sql(self, fake_pool, monkeypatch):
        """카탈로그에 새 코드값 추가 → positive 필터 호출 시 그 값이 ANY 배열에 포함."""
        from agent_arch.utils import rating as rating_module

        extended = rating_module._POSITIVE_PRESETS | {"great_answer"}
        monkeypatch.setattr(rating_module, "_POSITIVE_PRESETS", extended)

        await list_messages_in_conversation(
            "c1", "u1", limit=5, rating_filter="positive"
        )
        bound = [p for p in fake_pool.last_params if isinstance(p, list)]
        assert "great_answer" in bound[0]


class TestListMessagesPagination:
    """limit+1 SELECT가 정상적으로 SQL에 반영됨을 검증."""

    @pytest.mark.asyncio
    async def test_returns_rows_as_is(self, fake_pool):
        """fake가 돌려준 모든 row를 그대로 반환 (호출자가 trim 책임)."""
        fake_pool.rows = [{"id": f"m{i}"} for i in range(11)]
        rows = await list_messages_in_conversation("c1", "u1", limit=10)
        assert len(rows) == 11  # limit+1 SELECT — 라우터에서 trim


# ---------------------------------------------------------------------------
# get_message_in_conversation
# ---------------------------------------------------------------------------


class TestGetMessageInConversation:
    """단건 조회 — 소유자·소속 검증 한 번에."""

    @pytest.mark.asyncio
    async def test_pool_none_returns_none(self, monkeypatch):
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: None)
        assert await get_message_in_conversation("c1", "m1", "u1") is None

    @pytest.mark.asyncio
    async def test_row_returns_dict(self, fake_pool):
        """row 반환 시 dict로 변환."""
        fake_pool.row = {"id": "m1", "conversation_id": "c1", "query": "q", "answer": "a"}
        result = await get_message_in_conversation("c1", "m1", "u1")
        assert result == fake_pool.row

    @pytest.mark.asyncio
    async def test_no_row_returns_none(self, fake_pool):
        """row 없으면 None — 호출자(라우터)가 404로 변환."""
        fake_pool.row = None
        assert await get_message_in_conversation("c1", "m1", "u1") is None

    @pytest.mark.asyncio
    async def test_select_omits_llm_internal_columns(self, fake_pool):
        """단건 응답도 LLM 내부 컬럼 미포함."""
        fake_pool.row = {"id": "m1"}
        await get_message_in_conversation("c1", "m1", "u1")
        sql = fake_pool.last_sql or ""
        for col in ["llm_provider", "model_name", "tool_calls", "langsmith_run_id", "prompt_tokens"]:
            assert col not in sql

    @pytest.mark.asyncio
    async def test_params_order_matches_placeholders(self, fake_pool):
        """fetchrow params는 (message_id, conversation_id, user_sub) 순서."""
        fake_pool.row = {"id": "m1"}
        await get_message_in_conversation("c1", "m1", "u1")
        assert fake_pool.last_params == ("m1", "c1", "u1")


# ---------------------------------------------------------------------------
# POS 플래닝 턴 숨김 필터 — 계약: memory/conversation.save_logged_in_turn 의 pos_internal
# ---------------------------------------------------------------------------


class TestPosInternalFilter:
    """POS 플래닝 턴(metadata.pos_internal=true) 숨김 술어 검증.

    기존 행은 키 부재 → NULL → ``IS DISTINCT FROM 'true'`` 통과 (하위호환).
    """

    @pytest.mark.asyncio
    async def test_list_excludes_pos_internal(self, fake_pool):
        await list_messages_in_conversation("c1", "u1", limit=10)
        sql = fake_pool.last_sql or ""
        assert "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'" in sql

    @pytest.mark.asyncio
    async def test_get_excludes_pos_internal(self, fake_pool):
        """목록에서 숨긴 행이 단건 GET 으로만 200이 되는 비일관 방지."""
        fake_pool.row = {"id": "m1"}
        await get_message_in_conversation("c1", "m1", "u1")
        sql = fake_pool.last_sql or ""
        assert "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'" in sql


class TestFetchRecentTurns:
    """이력 폴백용 조회 — 소유권·숨김 턴·정렬·컬럼 범위가 계약이다."""

    def _pool(self, monkeypatch, rows=None):
        pool = _FakePool(rows=rows or [])
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)
        return pool

    @pytest.mark.asyncio
    async def test_owner_join_and_hidden_turn_filter(self, monkeypatch):
        from agent_arch.db.repository.messages import fetch_recent_turns

        pool = self._pool(monkeypatch)
        await fetch_recent_turns("conv-1", "u1", 10)

        assert "JOIN agent.conversations c ON c.id = m.conversation_id" in pool.last_sql
        assert "c.user_sub = $2" in pool.last_sql
        assert "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'" in pool.last_sql
        assert pool.last_params == ("conv-1", "u1", 10)

    @pytest.mark.asyncio
    async def test_cancelled_turns_are_excluded(self, monkeypatch):
        """중단된 턴은 부분답변을 PG 에만 남기고 Redis 에는 일부러 안 넣는다(save_to_redis=False).

        되살릴 때 넣으면 "재시도/다음 턴이 중단된 부분답변에 오염되지 않게" 라는 그 조치가
        정확히 무효가 된다.
        """
        from agent_arch.db.repository.messages import fetch_recent_turns

        pool = self._pool(monkeypatch)
        await fetch_recent_turns("conv-1", "u1", 10)

        assert "m.status_code IS NULL OR m.status_code <> 499" in pool.last_sql

    @pytest.mark.asyncio
    async def test_archived_is_not_filtered(self, monkeypatch):
        """보관 처리는 Redis 키를 지우지 않는다 — 여기서 막으면 캐시보다 인색해진다."""
        from agent_arch.db.repository.messages import fetch_recent_turns

        pool = self._pool(monkeypatch)
        await fetch_recent_turns("conv-1", "u1", 10)

        assert "archived_at" not in pool.last_sql

    @pytest.mark.asyncio
    async def test_since_watermark_is_bound(self, monkeypatch):
        from agent_arch.db.repository.messages import fetch_recent_turns

        pool = self._pool(monkeypatch)
        when = datetime(2026, 9, 22, 10, 0)
        await fetch_recent_turns("conv-1", "u1", 10, since=when)

        assert "m.created_at > $4" in pool.last_sql
        assert pool.last_params == ("conv-1", "u1", 10, when)

    @pytest.mark.asyncio
    async def test_newest_are_taken_then_returned_oldest_first(self, monkeypatch):
        from agent_arch.db.repository.messages import fetch_recent_turns

        pool = self._pool(monkeypatch)
        await fetch_recent_turns("conv-1", "u1", 10)

        inner, outer = pool.last_sql.split("ORDER BY")[1], pool.last_sql.split("ORDER BY")[2]
        assert "DESC" in inner        # 최근 N턴을 고르고
        assert "ASC" in outer         # 오래된 순으로 돌려준다

    @pytest.mark.asyncio
    async def test_llm_internal_columns_are_not_selected(self, monkeypatch):
        """이 모듈이 LLM 내부 컬럼을 응답 경로로 흘리지 않는다는 규약을 새 함수도 지킨다."""
        from agent_arch.db.repository.messages import fetch_recent_turns

        pool = self._pool(monkeypatch)
        await fetch_recent_turns("conv-1", "u1", 10)

        for col in ("tool_calls", "llm_provider", "model_name", "total_tokens", "langsmith_run_id"):
            assert col not in pool.last_sql

    @pytest.mark.asyncio
    async def test_missing_inputs_short_circuit(self, monkeypatch):
        from agent_arch.db.repository.messages import fetch_recent_turns

        pool = self._pool(monkeypatch)
        assert await fetch_recent_turns("", "u1", 10) == []
        assert await fetch_recent_turns("conv-1", "", 10) == []
        assert pool.last_sql is None
