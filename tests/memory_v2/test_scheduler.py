"""스케줄러 — due 유저 처리, 동시성 상한, 주기 인덱스 동기화, 실패 격리."""
from __future__ import annotations

import pytest

from agent_arch.services.memory_v2 import scheduler as S
from agent_arch.services.memory_v2.models import L1JobResult


@pytest.mark.asyncio
async def test_tick_runs_due_users_and_syncs_periodically(monkeypatch):
    ran, synced = [], []

    async def due(limit):
        return ["u1", "u2"]

    async def job(user_sub):
        ran.append(user_sub)
        return L1JobResult(user_sub, "done")

    class _Idx:
        async def sync_pending(self, limit):
            synced.append(limit)
            return 3

    async def get_idx():
        return _Idx()

    monkeypatch.setattr(S, "list_due_users", due)
    monkeypatch.setattr(S, "run_l1_job", job)
    import agent_arch.services.memory_v2.indexer as idx
    monkeypatch.setattr(idx, "get_v2_indexer", get_idx)

    sch = S.MemoryV2Scheduler(poll_seconds=0)
    for _ in range(S.SYNC_EVERY_TICKS):
        assert await sch.tick() == 2
    assert ran.count("u1") == S.SYNC_EVERY_TICKS and synced == [200]     # 10번째 tick 에서 1회 동기화


@pytest.mark.asyncio
async def test_tick_without_due_users(monkeypatch):
    async def due(limit):
        return []

    monkeypatch.setattr(S, "list_due_users", due)
    assert await S.MemoryV2Scheduler().tick() == 0


@pytest.mark.asyncio
async def test_sync_failure_is_isolated(monkeypatch):
    async def broken():
        raise ConnectionError("qdrant")

    import agent_arch.services.memory_v2.indexer as idx
    monkeypatch.setattr(idx, "get_v2_indexer", broken)
    assert await S.MemoryV2Scheduler()._sync_index() == 0


@pytest.mark.asyncio
async def test_job_exception_does_not_break_tick(monkeypatch):
    async def due(limit):
        return ["u1", "u2"]

    async def job(user_sub):
        if user_sub == "u1":
            raise RuntimeError("boom")
        return L1JobResult(user_sub, "done")

    monkeypatch.setattr(S, "list_due_users", due)
    monkeypatch.setattr(S, "run_l1_job", job)
    assert await S.MemoryV2Scheduler().tick() == 2
