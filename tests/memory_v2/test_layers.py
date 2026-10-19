"""상위층 규칙 — due 판정, 락, 강제/fresh 전달, 예약 트리거(pipeline/explicit 연결)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

import pytest

from agent_arch.services.memory_v2 import layers as L
from agent_arch.services.memory_v2 import persona as P
from agent_arch.services.memory_v2.scenes import RepairResult, ScenesResult
from tests.memory_v2.conftest import ORIGINAL_SCHEDULE


def test_scenes_due():
    now = datetime(2026, 9, 7, 12, 0)
    assert L.scenes_due(None, now)
    assert not L.scenes_due(now - timedelta(minutes=5), now)
    assert L.scenes_due(now - timedelta(minutes=15), now)


def test_persona_due():
    assert not L.persona_due({}, atom_count=P.PERSONA_MIN_ATOMS - 1, has_persona=False)
    assert L.persona_due({}, atom_count=P.PERSONA_MIN_ATOMS, has_persona=False)
    assert not L.persona_due({"atoms_since_persona": 10}, atom_count=20, has_persona=True)
    assert L.persona_due({"atoms_since_persona": P.PERSONA_REBUILD_EVERY}, atom_count=20, has_persona=True)


class _Env:
    def __init__(self, cp, atom_count=10, has_persona=True, lock=True):
        self.cp, self.atom_count, self.has_persona, self.lock = cp, atom_count, has_persona, lock
        self.calls: list[tuple] = []
        self.store: dict = {}

    async def get_checkpoint(self, user_sub):
        return self.cp

    async def count_active_atoms(self, user_sub):
        return self.atom_count

    async def get_active_persona(self, user_sub, company_scope=None):
        return {"content": "p"} if self.has_persona else None

    async def mark_l2_done(self, user_sub):
        self.calls.append(("l2_done",))

    async def rebuild_scenes(self, user_sub, *, since, company_scope=None):
        self.calls.append(("scenes", since))
        return ScenesResult(created=1)

    async def rebuild_persona(self, user_sub, *, fresh=False, company_scope=None):
        self.calls.append(("persona", fresh) if company_scope is None else ("persona", fresh, company_scope))
        return "persona-text"

    async def list_active_persona_scopes(self, user_sub):
        return list(getattr(self, "scopes", [None]))

    async def repair_scenes(self, user_sub):
        self.calls.append(("repair",))
        return RepairResult()

    async def set_nx(self, key, value, expire=None):
        return self.lock

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, expire=None):
        self.store[key] = value
        return True

    async def getdel(self, key):
        return self.store.pop(key, None)

    async def delete(self, key):
        self.calls.append(("unlock",))
        return True


@pytest.fixture
def wire(monkeypatch):
    def _wire(env: _Env):
        for name in ("get_checkpoint", "count_active_atoms", "get_active_persona", "mark_l2_done", "list_active_persona_scopes"):
            monkeypatch.setattr(L.repo, name, getattr(env, name))
        monkeypatch.setattr(L.scenes_mod, "rebuild_scenes", env.rebuild_scenes)
        monkeypatch.setattr(L.scenes_mod, "repair_scenes", env.repair_scenes)
        monkeypatch.setattr(L.persona_mod, "rebuild_persona", env.rebuild_persona)
        monkeypatch.setattr(L, "redis_store", env)
    return _wire


@pytest.mark.asyncio
async def test_locked_skips(wire):
    env = _Env({}, lock=False)
    wire(env)
    assert await L.run_upper_layers("u1") == {"scenes": None, "persona": "locked"} and env.calls == []


@pytest.mark.asyncio
async def test_locked_request_is_rerun_by_holder(wire):
    """락에 걸린 요청(예: 같은 턴의 두 번째 forget)은 버리지 않고, 락 보유자가 끝난 뒤 한 번 더 돈다."""
    env = _Env({"last_l2_at": datetime.now(), "atoms_since_persona": 0}, atom_count=10, has_persona=True, lock=False)
    wire(env)
    assert (await L.run_upper_layers("u1", force_persona=True, fresh_persona=True))["persona"] == "locked"
    assert env.store[L._rerun_key("u1")]["fresh_persona"] is True
    env.lock = True
    await L.run_upper_layers("u1", force_scenes=True)          # 락 보유자: 자기 몫 + 표시된 재실행
    assert env.calls.count(("repair",)) == 2
    assert ("persona", True) in env.calls                       # 재실행이 fresh 페르소나 요구를 이어받았다
    assert L._rerun_key("u1") not in env.store


@pytest.mark.asyncio
async def test_fresh_rebuilds_every_persona_scope(wire):
    """기억을 지운 뒤(fresh) 는 범위 없는 본뿐 아니라 회사 범위 페르소나까지 전부 다시 쓴다."""
    env = _Env({"last_l2_at": datetime.now(), "atoms_since_persona": 0}, atom_count=10, has_persona=True)
    env.scopes = [None, "테스트 고객사"]
    wire(env)
    await L.run_upper_layers("u1", force_persona=True, fresh_persona=True)
    assert ("persona", True) in env.calls and ("persona", True, "테스트 고객사") in env.calls


@pytest.mark.asyncio
async def test_repair_runs_before_persona(wire):
    env = _Env({"last_l2_at": datetime.now(), "atoms_since_persona": 0}, atom_count=10, has_persona=True)
    wire(env)
    await L.run_upper_layers("u1", force_persona=True, fresh_persona=True)
    assert env.calls.index(("repair",)) < env.calls.index(("persona", True))


@pytest.mark.asyncio
async def test_scenes_and_persona_due_paths(wire):
    env = _Env({"last_l2_at": None, "atoms_since_persona": 0}, atom_count=10, has_persona=False)
    wire(env)
    out = await L.run_upper_layers("u1")
    assert out["scenes"].created == 1 and out["persona"] == "persona-text"
    assert env.calls == [("repair",), ("scenes", None), ("l2_done",), ("persona", False), ("unlock",)]


@pytest.mark.asyncio
async def test_not_due_skips_both(wire):
    env = _Env({"last_l2_at": datetime.now(), "atoms_since_persona": 3}, atom_count=10, has_persona=True)
    wire(env)
    out = await L.run_upper_layers("u1")
    assert out["scenes"] is None and out["persona"] == "skipped" and env.calls == [("repair",), ("unlock",)]


@pytest.mark.asyncio
async def test_force_and_fresh(wire):
    env = _Env({"last_l2_at": datetime.now(), "atoms_since_persona": 0}, atom_count=10, has_persona=True)
    wire(env)
    out = await L.run_upper_layers("u1", force_scenes=True, force_persona=True, fresh_persona=True)
    assert out["persona"] == "persona-text"
    assert ("scenes", env.cp["last_l2_at"]) in env.calls and ("persona", True) in env.calls


@pytest.mark.asyncio
async def test_schedule_runs_later(wire, monkeypatch):
    env = _Env({"last_l2_at": None, "atoms_since_persona": 0}, atom_count=10, has_persona=False)
    wire(env)
    task = ORIGINAL_SCHEDULE("u1", delay=0.01)
    assert task is not None
    await task
    assert ("scenes", None) in env.calls and ("persona", False) in env.calls


@pytest.mark.asyncio
async def test_pipeline_and_explicit_trigger_upper_layers(upper_layers_spy, monkeypatch):
    """pipeline 은 atom 이 쓰였을 때, explicit 은 store/update 뒤·forget 뒤 상위층을 예약한다."""
    from agent_arch.services.memory_v2 import explicit
    from agent_arch.services.memory_v2.models import DedupDecision, JudgeResult, WriteStats

    async def judge(user_sub, drafts):
        return JudgeResult(decisions=[DedupDecision(index=0, action="update", target_ids=["t"])])

    async def apply(user_sub, drafts, decisions, **kw):
        return WriteStats(updated=1, output_atom_ids=["n1"])

    async def log(**kw):
        return "log"

    async def soft_delete(user_sub, atom_id):
        return {"id": atom_id, "atom_type": "persona", "content": "x"}

    async def broken_indexer():
        raise ConnectionError("no qdrant")

    monkeypatch.setattr(explicit.dedup, "judge_batch", judge)
    monkeypatch.setattr(explicit.writer, "apply_decisions", apply)
    monkeypatch.setattr(explicit.repo, "insert_generation_log", log)
    monkeypatch.setattr(explicit.repo, "soft_delete_atom", soft_delete)
    import agent_arch.services.memory_v2.indexer as idx
    monkeypatch.setattr(idx, "get_v2_indexer", broken_indexer)

    await explicit.save_explicit_atom("u1", "앞으로 표로 정리한다")
    await explicit.forget_atom("u1", "11111111-1111-4111-8111-111111111111")
    assert len(upper_layers_spy) == 2
    assert upper_layers_spy[0]["force_persona"] is True and upper_layers_spy[0]["fresh_persona"] is True   # update → fresh
    assert upper_layers_spy[1]["fresh_persona"] is True and upper_layers_spy[1]["delay"] == 0
