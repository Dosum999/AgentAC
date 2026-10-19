"""관리 기능 — export 형태, reset 3층+벡터, v1 별칭 시드, 백필 루프."""
from __future__ import annotations

from datetime import datetime

import pytest

from agent_arch.services.memory_v2 import admin
from agent_arch.services.memory_v2.models import DedupDecision, JudgeResult, L1JobResult, WriteStats
from agent_arch.services.memory_v2.scenes import ScenesResult


@pytest.mark.asyncio
async def test_export_shape(monkeypatch):
    async def atoms(user_sub, *, atom_type=None, limit=50):
        return [{"id": "a1", "atom_type": "alias", "content": "c", "priority": 70, "scene_name": None,
                 "payload": {"aliases": ["w"]}, "version": 1, "created_at": datetime(2026, 9, 7), "last_seen_at": None}]

    async def scenes(user_sub, *, limit=8, with_content=False):
        return [{"name": "n", "title": "t", "summary": "s", "content": "c", "heat": 2, "version": 1, "updated_at": None}]

    async def persona(user_sub, company_scope=None):
        return {"content": "p", "version": 3, "updated_at": datetime(2026, 9, 7, 1)}

    async def cp(user_sub):
        return {"pending_count": 2, "next_due_at": None, "last_l1_at": datetime(2026, 9, 7), "last_l2_at": None, "last_l3_at": None}

    async def logs(user_sub, limit=20):
        return [{"layer": "l1", "status": "succeeded", "model": "m", "decision_counts": {"store": 1}, "latency_ms": 5, "error": None, "created_at": None}]

    for name, fn in (("list_active_atoms", atoms), ("list_scenes", scenes), ("get_active_persona", persona),
                     ("get_checkpoint", cp), ("list_generation_logs", logs)):
        monkeypatch.setattr(admin.repo, name, fn)
    out = await admin.export_user_memory("u1")
    assert out["atoms"][0] == {"id": "a1", "type": "alias", "content": "c", "priority": 70, "scene": None,
                               "payload": {"aliases": ["w"]}, "version": 1, "created_at": "2026-09-07T00:00:00", "last_seen_at": None}
    assert out["scenes"][0]["heat"] == 2.0 and out["persona"]["version"] == 3
    assert out["checkpoint"]["pending_count"] == 2 and out["recent_generations"][0]["counts"] == {"store": 1}


@pytest.mark.asyncio
async def test_reset_counts_and_tolerates_qdrant(monkeypatch):
    calls = []

    async def del_atoms(user_sub):
        calls.append("atoms"); return 7

    async def del_scenes(user_sub):
        calls.append("scenes"); return 2

    async def del_persona(user_sub, company_scope=None):
        calls.append("persona"); return True

    async def del_cp(user_sub):
        calls.append("cp")

    async def broken():
        raise ConnectionError("qdrant")

    monkeypatch.setattr(admin.repo, "soft_delete_all_atoms", del_atoms)
    monkeypatch.setattr(admin.repo, "soft_delete_all_scenes", del_scenes)
    monkeypatch.setattr(admin.repo, "delete_persona", del_persona)
    monkeypatch.setattr(admin.repo, "delete_checkpoint", del_cp)
    import agent_arch.services.memory_v2.indexer as idx
    monkeypatch.setattr(idx, "get_v2_indexer", broken)
    out = await admin.reset_user_memory("u1")
    assert out == {"atoms": 7, "scenes": 2, "persona": True, "qdrant": False}
    assert calls == ["atoms", "scenes", "persona", "cp"]


def test_alias_draft_from_v1():
    rec = {"payload": {"aliases": ["watchdog", "왓치독"], "canonical_name": "ACME Heros", "canonical_kind": "mattermost_channel", "canonical_id": "ch1"},
           "support_count": 71, "source_message_ids": ["m1"]}
    d = admin.alias_draft_from_v1(rec)
    assert d.atom_type == "alias" and d.priority == 85 and d.payload["canonical_id"] == "ch1"
    assert "watchdog" in d.content and "ACME Heros" in d.content and d.source_message_ids == ["m1"]
    assert admin.alias_draft_from_v1({"payload": {"aliases": [], "canonical_name": "x"}}) is None
    assert admin.alias_draft_from_v1({"payload": {"aliases": ["a"], "canonical_name": ""}}) is None
    assert admin.alias_draft_from_v1({"payload": {"aliases": ["a"], "canonical_name": "n"}, "support_count": 1}).priority == 43
    # v1 데이터 품질 필터: 무정보(별칭=정식이름) · 봇 이름 오염은 시드하지 않는다
    assert admin.alias_draft_from_v1({"payload": {"aliases": ["watchdog"], "canonical_name": "watchdog"}}) is None
    assert admin.alias_draft_from_v1({"payload": {"aliases": ["에이전트"], "canonical_name": "신희섬"}}) is None
    mixed = admin.alias_draft_from_v1({"payload": {"aliases": ["개발팀", "dev-team"], "canonical_name": "개발팀"}})
    assert mixed.payload["aliases"] == ["dev-team"]            # 정보 있는 별칭만 남긴다
    with_id = admin.alias_draft_from_v1({"payload": {"aliases": ["watchdog"], "canonical_name": "watchdog", "canonical_id": "ch1"}})
    assert with_id is not None and with_id.payload["canonical_id"] == "ch1"   # 이름은 같아도 id 가 새 정보


@pytest.mark.asyncio
async def test_seed_aliases_goes_through_judge(monkeypatch):
    async def v1_list(user_sub, *, record_type=None, company_scope=None, domain_scope=None, limit=50):
        assert record_type == "entity_alias"
        return [{"payload": {"aliases": ["개발팀"], "canonical_name": "dev"}, "support_count": 39},
                {"payload": {"aliases": [], "canonical_name": "x"}}]

    async def judge(user_sub, drafts):
        assert len(drafts) == 1
        return JudgeResult(decisions=[DedupDecision(index=0, action="store")])

    async def apply(user_sub, drafts, decisions, **kw):
        return WriteStats(stored=1, output_atom_ids=["n1"])

    import agent_arch.db.repository.memory_records as v1
    monkeypatch.setattr(v1, "list_active_records", v1_list)
    monkeypatch.setattr(admin.dedup, "judge_batch", judge)
    monkeypatch.setattr(admin.writer, "apply_decisions", apply)
    assert await admin.seed_aliases_from_v1("u1") == 1


@pytest.mark.asyncio
async def test_backfill_loops_until_no_more_and_runs_upper(monkeypatch):
    seq = iter([
        L1JobResult("u1", "done", processed=10, stats=WriteStats(stored=2), has_more=True),
        L1JobResult("u1", "done", processed=10, stats=WriteStats(merged=1, skipped=3), has_more=True),
        L1JobResult("u1", "empty", processed=4, stats=WriteStats(), has_more=False),
    ])
    calls = []

    async def run_job(user_sub):
        return next(seq)

    async def seed(user_sub, cursor_at, cursor_id, pending):
        calls.append(("seed", cursor_at, pending))

    async def count(user_sub, cursor_at, cursor_id):
        return 24

    async def upper(user_sub, **kw):
        calls.append(("upper", kw))
        return {"scenes": ScenesResult(created=1), "persona": "페르소나 본문"}

    monkeypatch.setattr(admin.pipeline, "run_l1_job", run_job)
    monkeypatch.setattr(admin.repo, "seed_checkpoint", seed)
    monkeypatch.setattr(admin.repo, "count_messages_after", count)
    monkeypatch.setattr(admin.layers, "run_upper_layers", upper)
    out = await admin.backfill_user("u1", since=datetime(2026, 7, 1))
    assert (out["rounds"], out["processed"], out["stored"], out["merged"], out["skipped"]) == (3, 24, 2, 1, 3)
    assert calls[0] == ("seed", datetime(2026, 7, 1), 24)
    assert calls[1][1] == {"force_scenes": True, "force_persona": True, "fresh_persona": True}
    assert out["upper"] == {"scenes": (1, 0, 0, None), "persona": "7 chars"}


@pytest.mark.asyncio
async def test_backfill_waits_for_upper_lock(monkeypatch):
    outs = iter([{"scenes": None, "persona": "locked"}, {"scenes": ScenesResult(), "persona": None}])
    slept = []

    async def run_job(user_sub):
        return L1JobResult("u1", "no_new")

    async def seed(*a, **kw):
        pass

    async def count(*a, **kw):
        return 0

    async def upper(user_sub, **kw):
        return next(outs)

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(admin.pipeline, "run_l1_job", run_job)
    monkeypatch.setattr(admin.repo, "seed_checkpoint", seed)
    monkeypatch.setattr(admin.repo, "count_messages_after", count)
    monkeypatch.setattr(admin.layers, "run_upper_layers", upper)
    monkeypatch.setattr(admin.asyncio, "sleep", fake_sleep)
    out = await admin.backfill_user("u1")
    assert slept == [admin.UPPER_LOCK_WAIT_SECONDS] and out["upper"]["persona"] is None and out["upper"]["scenes"] == (0, 0, 0, None)


@pytest.mark.asyncio
async def test_backfill_resume_keeps_cursor(monkeypatch):
    calls = []

    async def run_job(user_sub):
        return L1JobResult("u1", "no_new")

    async def seed(user_sub, cursor_at, cursor_id, pending):
        calls.append((cursor_at, cursor_id, pending))

    async def count(user_sub, cursor_at, cursor_id):
        return 3

    async def cp(user_sub):
        return {"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "cid"}

    monkeypatch.setattr(admin.pipeline, "run_l1_job", run_job)
    monkeypatch.setattr(admin.repo, "seed_checkpoint", seed)
    monkeypatch.setattr(admin.repo, "count_messages_after", count)
    monkeypatch.setattr(admin.repo, "get_checkpoint", cp)
    out = await admin.backfill_user("u1", since=datetime(2026, 7, 1), resume=True, upper_layers=False)
    assert calls == [(datetime(2026, 9, 4), "cid", 3)] and out["rounds"] == 1


@pytest.mark.asyncio
async def test_backfill_stops_on_llm_failure(monkeypatch):
    async def run_job(user_sub):
        return L1JobResult("u1", "failed", processed=0, error="TimeoutError")

    async def seed(*a, **kw):
        pass

    async def count(*a, **kw):
        return 5

    async def upper(*a, **kw):
        raise AssertionError("상위층은 돌지 않아야")

    monkeypatch.setattr(admin.pipeline, "run_l1_job", run_job)
    monkeypatch.setattr(admin.repo, "seed_checkpoint", seed)
    monkeypatch.setattr(admin.repo, "count_messages_after", count)
    monkeypatch.setattr(admin.layers, "run_upper_layers", upper)
    out = await admin.backfill_user("u1")
    assert out["rounds"] == 1 and out["failed"] == 1 and out["upper"] is None
