"""memory_v2 repository — SQL 계약(격리·커서 튜플·pos_internal 필터·단조 전진) 검증(FakePool)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pytest

from agent_arch.db.repository import memory_v2 as repo


class _FakePool:
    def __init__(self, row: Optional[dict] = None, rows: Optional[list] = None):
        self.row, self.rows = row, rows or []
        self.sql: list[str] = []
        self.params: list[tuple] = []

    async def fetchrow(self, sql: str, *params: Any):
        self.sql.append(sql); self.params.append(params)
        return self.row

    async def fetch(self, sql: str, *params: Any):
        self.sql.append(sql); self.params.append(params)
        return list(self.rows)

    async def execute(self, sql: str, *params: Any):
        self.sql.append(sql); self.params.append(params)
        return "UPDATE 1"

    async def fetchval(self, sql: str, *params: Any):
        self.sql.append(sql); self.params.append(params)
        return 0


@pytest.fixture
def pool(monkeypatch):
    p = _FakePool()
    monkeypatch.setattr(repo.db_pool, "get_pool", lambda: p)
    return p


@pytest.mark.asyncio
async def test_bump_pending_upsert_contract(pool):
    pool.row = {"user_sub": "u1", "pending_count": 3, "warmup_threshold": 2, "next_due_at": None}
    out = await repo.bump_pending("u1", "mid", immediate=True, idle_seconds=600, default_threshold=5)
    sql, params = pool.sql[0], pool.params[0]
    assert "ON CONFLICT (user_sub)" in sql and "pending_count + 1" in sql
    assert "warmup_threshold > 0" in sql                       # 웜업 임계 vs 기본 N
    assert "SELECT created_at FROM agent.messages WHERE id = $2::uuid" in sql   # 첫 커서 = 이 메시지
    assert params[0] == "u1" and params[1] == "mid" and params[2] is True
    assert isinstance(params[3], float) and params[4] == 5
    assert out["pending_count"] == 3


@pytest.mark.asyncio
async def test_fetch_messages_after_filters_and_orders(pool):
    await repo.fetch_messages_after("u1", datetime(2026, 9, 4), "cid", 11)
    sql, params = pool.sql[0], pool.params[0]
    assert "user_sub = $1" in sql
    assert "(created_at, id) > ($2::timestamp, $3::uuid)" in sql
    assert "pos_internal" in sql and "capture_skip" in sql and "ORDER BY created_at, id" in sql
    assert params == ("u1", datetime(2026, 9, 4), "cid", 11)


@pytest.mark.asyncio
async def test_l0_filter_applied_everywhere(pool):
    pool.rows = []
    await repo.fetch_messages_before("u1", datetime(2026, 9, 4), "cid", 5)
    await repo.count_messages_after("u1", None, None)
    await repo.list_users_with_messages(5)
    assert all("capture_skip" in q and "pos_internal" in q for q in pool.sql)


@pytest.mark.asyncio
async def test_fetch_messages_after_none_cursor_means_epoch(pool):
    await repo.fetch_messages_after("u1", None, None, 5)
    params = pool.params[0]
    assert params[1] == datetime(1970, 1, 1) and params[2] == repo._NIL_UUID


@pytest.mark.asyncio
async def test_fetch_messages_before_reverses_to_chronological(pool):
    pool.rows = [{"id": "b", "conversation_id": None, "user_sub": "u1", "query": "2", "answer": "", "tool_calls": None, "metadata": None, "created_at": 2},
                 {"id": "a", "conversation_id": None, "user_sub": "u1", "query": "1", "answer": "", "tool_calls": None, "metadata": None, "created_at": 1}]
    out = await repo.fetch_messages_before("u1", datetime(2026, 9, 4), "cid", 5)
    assert [m["query"] for m in out] == ["1", "2"]
    assert "ORDER BY created_at DESC, id DESC" in pool.sql[0]
    assert await repo.fetch_messages_before("u1", None, None, 5) == []


@pytest.mark.asyncio
async def test_mark_job_success_is_monotonic_and_warms_up(pool):
    await repo.mark_job_success("u1", cursor_at=datetime(2026, 9, 4), cursor_id="cid", processed=3, has_more=True,
                                scene_name="s", atoms_added=2, default_threshold=5, idle_seconds=600)
    sql, params = pool.sql[0], pool.params[0]
    assert "(last_cursor_at, last_cursor_id) < ($2::timestamp, $3::uuid)" in sql      # 단조 전진
    assert "GREATEST(0, pending_count - $4)" in sql
    assert "warmup_threshold * 2 >= $7" in sql and "ELSE warmup_threshold * 2" in sql
    assert "retry_count = 0" in sql
    assert params[0] == "u1" and params[3] == 3 and params[4] is True and params[8] == 2


@pytest.mark.asyncio
async def test_mark_job_failure_gives_up_after_max(pool):
    pool.row = {"retry_count": 5}
    n = await repo.mark_job_failure("u1", "boom", retry_delay_seconds=30, max_retries=5)
    sql = pool.sql[0]
    assert n == 5 and "retry_count + 1 >= $4" in sql and "pending_count = CASE" in sql
    assert "last_cursor" not in sql                             # 실패는 커서를 건드리지 않는다


@pytest.mark.asyncio
async def test_insert_atom_params(pool):
    pool.row = {"id": "new-id"}
    out = await repo.insert_atom("u1", atom_type="alias", content="c", priority=70, scene_name="s",
                                 payload={"aliases": ["w"]}, source_message_ids=["m1"], timestamps=[datetime(2026, 9, 4)],
                                 version=2, generation_id="g1")
    sql, params = pool.sql[0], pool.params[0]
    assert out == "new-id" and "$8::uuid[]" in sql and "$9::timestamp[]" in sql and "$7::jsonb" in sql
    assert params[6] == '{"aliases": ["w"]}' and params[7] == ["m1"] and params[9] == 2 and params[10] == "g1"


@pytest.mark.asyncio
async def test_owner_isolation_in_mutations(pool):
    await repo.get_active_atoms_by_ids("u1", ["a", "a", "b"])
    assert "user_sub = $1" in pool.sql[0] and pool.params[0][1] == ["a", "b"]
    await repo.supersede_atoms("u1", ["a"], "n")
    assert "user_sub = $1" in pool.sql[1] and "status = 'superseded'" in pool.sql[1]
    await repo.touch_atoms("u1", ["a"])
    assert "user_sub = $1" in pool.sql[2] and "observed_count + 1" in pool.sql[2]
    await repo.search_atoms_keyword("u1", "표", limit=3, atom_type="instruction")
    assert "user_sub = $1" in pool.sql[3] and "$2 <% content" in pool.sql[3] and "atom_type = $3" in pool.sql[3]
    assert pool.params[3] == ("u1", "표", "instruction", 3)


@pytest.mark.asyncio
async def test_search_atoms_keyword_empty_query_short_circuits(pool):
    assert await repo.search_atoms_keyword("u1", "   ") == []
    assert pool.sql == []


@pytest.mark.asyncio
async def test_insert_generation_log_truncates_raw(pool):
    pool.row = {"id": "log-id"}
    out = await repo.insert_generation_log(user_sub="u1", layer="l1", status="failed", model="m", prompt_sha="s",
                                           input_message_ids=["a"], output_atom_ids=[], decision_counts={"store": 1},
                                           latency_ms=3, error="e", raw_output="x" * 5000, log_id="pre")
    params = pool.params[0]
    assert out == "log-id" and params[0] == "pre" and len(params[11]) == 4000 and params[8] == '{"store": 1}'


@pytest.mark.asyncio
async def test_list_alias_hints_formats(pool):
    pool.rows = [{"payload": '{"aliases": ["watchdog", "왓치독"], "canonical_name": "ACME Heros"}'},
                 {"payload": '{"aliases": [], "canonical_name": "x"}'}]
    assert await repo.list_alias_hints("u1") == ["watchdog/왓치독 → ACME Heros (id 없음)"]
