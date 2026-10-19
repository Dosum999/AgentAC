"""``search_messages_for_user`` 단위 테스트.

asyncpg pool을 in-memory fake로 대체해 SQL/params 구성을 검증한다.
실제 PG 연결 없이 소유자 격리·아카이브 제외·페이지네이션 바인딩이 의도대로
SQL에 반영되는지가 핵심.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional

import pytest

from agent_arch.db.repository import messages as messages_module
from agent_arch.db.repository.messages import _like_pattern, search_messages_for_user


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
    """``db_pool.get_pool()``이 ``_FakePool``을 반환하도록 한다."""
    pool = _FakePool()
    monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)
    return pool


# LIKE 메타문자 이스케이프 절. 백슬래시가 들어가 인라인으로 쓰면 읽기 어려워 상수로 뺀다.
ESCAPE_CLAUSE = "ESCAPE '" + chr(92) + "'"


class TestSearchMessagesGuards:
    """DB 접근 전 조기 반환 분기."""

    @pytest.mark.asyncio
    async def test_pool_none_returns_empty(self, monkeypatch):
        """DB 풀 비활성화 시 빈 리스트."""
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: None)
        assert await search_messages_for_user("u1", "배포", limit=10) == []

    @pytest.mark.asyncio
    async def test_blank_query_returns_empty_without_db_call(self, fake_pool):
        """공백뿐인 검색어는 쿼리하지 않는다."""
        assert await search_messages_for_user("u1", "   ", limit=10) == []
        assert fake_pool.last_sql is None

    @pytest.mark.asyncio
    async def test_missing_user_sub_returns_empty(self, fake_pool):
        """user_sub 없으면 전역 검색으로 새지 않고 빈 리스트."""
        assert await search_messages_for_user("", "배포", limit=10) == []
        assert fake_pool.last_sql is None


class TestSearchMessagesSql:
    """생성된 SQL과 파라미터 바인딩 검증."""

    @pytest.mark.asyncio
    async def test_owner_isolation_and_archive_filter(self, fake_pool):
        """소유자는 conversations.user_sub로 강제하고 아카이브 대화방은 제외한다."""
        await search_messages_for_user("u1", "배포", limit=10)

        sql = fake_pool.last_sql or ""
        assert "JOIN agent.conversations c ON c.id = m.conversation_id" in sql
        assert "c.user_sub = $1" in sql
        assert "c.archived_at IS NULL" in sql
        # m.user_sub는 v19에서 뒤늦게 추가돼 NULL 행이 있을 수 있어 기준으로 쓰지 않는다.
        assert "m.user_sub" not in sql

    @pytest.mark.asyncio
    async def test_uses_pg_trgm_word_similarity(self, fake_pool):
        """query/answer 양쪽에 trgm 매칭 + 유사도 정렬."""
        await search_messages_for_user("u1", "배포", limit=10)

        sql = fake_pool.last_sql or ""
        assert "$2 <% m.query" in sql
        assert "$2 <% m.answer" in sql
        assert "word_similarity" in sql
        assert "ORDER BY score DESC, m.created_at DESC, m.id DESC" in sql

    @pytest.mark.asyncio
    async def test_ilike_branch_catches_mid_word(self, fake_pool):
        """어절 중간 매칭용 ILIKE 갈래가 query/answer 양쪽에 OR로 붙는다.

        word_similarity 는 어절 시작 경계부터만 매칭해 '보라'로 '김보라님에게'를
        못 잡는다. 성 뺀 이름 검색이 실사용에서 걸려 추가한 갈래 — 지우면 회귀한다.
        """
        await search_messages_for_user("u1", "보라", limit=10)

        sql = fake_pool.last_sql or ""
        assert "m.query ILIKE $3 ESCAPE" in sql
        assert "m.answer ILIKE $3 ESCAPE" in sql

    @pytest.mark.asyncio
    async def test_score_has_substring_floor(self, fake_pool):
        """부분문자열로만 걸린 건이 바닥 점수로 뭉치지 않도록 하한을 준다."""
        await search_messages_for_user("u1", "보라", limit=10)

        sql = fake_pool.last_sql or ""
        assert "CASE WHEN m.query  ILIKE $3 ESCAPE" in sql
        assert "CASE WHEN m.answer ILIKE $3 ESCAPE" in sql

    @pytest.mark.asyncio
    async def test_excludes_pos_internal_turns(self, fake_pool):
        """POS 플래닝 턴은 검색에서도 제외한다 — 이력 조회 API 와 같은 기준.

        정화 후에도 query 에는 사용자 질문이 그대로 남아 검색에는 걸리는데, 목록/단건
        에서는 숨겨져 있어 결과를 눌러도 404 가 된다(판단 턴과 query 가 동일해 같은
        질문이 두 번 뜨는 문제도 있다). 계약은 memory/conversation.save_logged_in_turn 참조.
        기존 행은 키 부재 → NULL → ``IS DISTINCT FROM 'true'`` 통과(하위호환).
        """
        await search_messages_for_user("u1", "배포", limit=10)
        sql = fake_pool.last_sql or ""
        assert "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'" in sql

    @pytest.mark.asyncio
    async def test_answer_is_not_wrapped_in_coalesce(self, fake_pool):
        """answer를 COALESCE로 감싸면 GIN trgm 인덱스를 못 탄다 — 회귀 방지.

        인덱스는 answer 컬럼 자체에 걸려 있어 표현식이 달라지는 순간 Seq Scan으로
        고정된다(EXPLAIN으로 확인). answer는 migration_v3에서 NOT NULL이라 COALESCE
        자체가 불필요하다.
        """
        await search_messages_for_user("u1", "배포", limit=10)
        assert "COALESCE(m.answer" not in (fake_pool.last_sql or "")

    @pytest.mark.asyncio
    async def test_binds_pattern_limit_plus_one_and_offset(self, fake_pool):
        """has_more 판정을 위해 limit+1을 조회하고, ILIKE 패턴이 $3으로 들어간다."""
        await search_messages_for_user("u1", "배포", limit=20, offset=40)

        sql = fake_pool.last_sql or ""
        assert "LIMIT $4 OFFSET $5" in sql
        assert fake_pool.last_params == ("u1", "배포", "%배포%", 21, 40)

    @pytest.mark.asyncio
    async def test_select_omits_llm_internal_columns(self, fake_pool):
        """SELECT 컬럼에 LLM 내부 필드가 절대 포함되지 않는다 (노출 정책).

        검사 범위는 SELECT 절뿐이다 — WHERE 는 m.metadata 를 pos_internal 숨김 술어로
        정당하게 참조하므로, SQL 전문을 훑으면 그 술어에 걸려 오탐이 난다.
        """
        await search_messages_for_user("u1", "배포", limit=10)

        full_sql = fake_pool.last_sql or ""
        sql = full_sql[: full_sql.index("FROM agent.messages")]
        for forbidden in (
            "llm_provider",
            "model_name",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "llm_invoke_count",
            "tool_calls",
            "langsmith_run_id",
            "m.metadata",
        ):
            assert forbidden not in sql, f"SELECT에 내부 컬럼 누출: {forbidden}"

    @pytest.mark.asyncio
    async def test_selects_conversation_title(self, fake_pool):
        """결과 카드에 붙일 대화방 제목을 함께 내려보낸다."""
        sql_holder = fake_pool
        await search_messages_for_user("u1", "배포", limit=10)
        assert "c.title AS conversation_title" in (sql_holder.last_sql or "")

    @pytest.mark.asyncio
    async def test_returns_plain_dicts(self, monkeypatch):
        """asyncpg Record를 dict로 정규화해 반환한다."""
        row = {
            "id": "m1",
            "conversation_id": "c1",
            "conversation_title": "배포 논의",
            "query": "배포 언제?",
            "answer": "다음 주",
            "created_at": datetime(2026, 5, 14),
            "score": 0.8,
        }
        pool = _FakePool(rows=[row])
        monkeypatch.setattr(messages_module.db_pool, "get_pool", lambda: pool)

        result = await search_messages_for_user("u1", "배포", limit=10)
        assert result == [row]
        assert isinstance(result[0], dict)


class TestSearchMessagesFilters:
    """기간·평가 필터. 고정 위치($1~$3)를 유지한 채 뒤에만 붙는지가 핵심."""

    @pytest.mark.asyncio
    async def test_no_rating_clause_by_default(self, fake_pool):
        """필터를 안 주면 rating 조건이 SQL에 아예 없어야 한다 (기존 동작 회귀)."""
        await search_messages_for_user("u1", "배포", limit=20)
        assert "m.rating" not in (fake_pool.last_sql or "")

    @pytest.mark.asyncio
    async def test_since_until_bound_after_fixed_params(self, fake_pool):
        """since/until 은 $4/$5 로 붙고 LIMIT·OFFSET 이 뒤로 밀린다."""
        await search_messages_for_user(
            "u1",
            "배포",
            limit=20,
            offset=10,
            since=datetime(2026, 9, 1),
            until=datetime(2026, 9, 4),
        )
        sql = fake_pool.last_sql or ""
        assert "m.created_at >= $4" in sql
        assert "m.created_at <= $5" in sql
        assert "LIMIT $6 OFFSET $7" in sql
        assert fake_pool.last_params[-2:] == (21, 10)

    @pytest.mark.asyncio
    async def test_negative_filter_excludes_empty_rating(self, fake_pool):
        """'별로예요'는 평가됨 AND 긍정 프리셋 아님. 빈 문자열은 미평가라 빠진다."""
        await search_messages_for_user(
            "u1", "배포", limit=20, rating_filter="negative"
        )
        sql = fake_pool.last_sql or ""
        assert "m.rating IS NOT NULL AND m.rating <> ''" in sql
        assert "m.rating <> ALL($4::text[])" in sql
        assert ["thumbs_up"] in list(fake_pool.last_params)

    @pytest.mark.asyncio
    async def test_rated_filter_excludes_empty_rating(self, fake_pool):
        await search_messages_for_user("u1", "배포", limit=20, rating_filter="rated")
        assert "m.rating IS NOT NULL AND m.rating <> ''" in (fake_pool.last_sql or "")

    @pytest.mark.asyncio
    async def test_positive_filter_uses_preset_catalog(self, fake_pool):
        await search_messages_for_user("u1", "배포", limit=20, rating_filter="positive")
        assert "m.rating = ANY($4::text[])" in (fake_pool.last_sql or "")


class TestSearchMessagesRefactorGuards:
    """동적 clause 조립으로 바꾸면서 깨지기 쉬운 3대 지점 회귀 방지."""

    @pytest.mark.asyncio
    async def test_escape_clause_survives(self, fake_pool):
        """ESCAPE 절이 유실되면 사용자가 친 %/_ 가 와일드카드로 해석된다."""
        await search_messages_for_user("u1", "50%", limit=20)
        sql = fake_pool.last_sql or ""
        # WHERE 2회 + SELECT score CASE 2회
        assert sql.count(ESCAPE_CLAUSE) == 4

    @pytest.mark.asyncio
    async def test_trgm_operator_keeps_fixed_placeholders(self, fake_pool):
        """$2/$3 은 SELECT score 식과 공유하므로 번호가 밀리면 안 된다."""
        await search_messages_for_user(
            "u1", "배포", limit=20, rating_filter="negative", since=datetime(2026, 9, 1)
        )
        sql = fake_pool.last_sql or ""
        assert "$2 <% m.query" in sql
        assert "$2 <% m.answer" in sql
        assert "word_similarity($2, m.query)" in sql
        assert fake_pool.last_params[1] == "배포"
        assert fake_pool.last_params[2] == "%배포%"

    @pytest.mark.asyncio
    async def test_answer_still_not_wrapped_in_coalesce(self, fake_pool):
        await search_messages_for_user("u1", "배포", limit=20, since=datetime(2026, 9, 1))
        assert "COALESCE(m.answer" not in (fake_pool.last_sql or "")


class TestLikePattern:
    """``_like_pattern`` — LIKE 메타문자 이스케이프."""

    def test_plain_keyword(self):
        assert _like_pattern("보라") == "%보라%"

    def test_percent_is_escaped(self):
        """'50%' 를 쳤을 때 % 가 와일드카드로 새지 않는다."""
        assert _like_pattern("50%") == r"%50\%%"

    def test_underscore_is_escaped(self):
        """_ 는 LIKE 에서 한 글자 와일드카드라 반드시 막아야 한다."""
        assert _like_pattern("_test") == r"%\_test%"

    def test_backslash_escaped_first(self):
        """백슬래시를 먼저 치환하지 않으면 뒤 이스케이프가 이중으로 깨진다."""
        assert _like_pattern("a\\b") == r"%a\\b%"
