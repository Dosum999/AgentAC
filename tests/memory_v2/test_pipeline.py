"""L1 잡 — 락·커서·재시도·생성 로그 규약(전부 fake, LLM/DB 없음)."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from agent_arch.services.memory_v2 import pipeline
from agent_arch.services.memory_v2.models import AtomDraft, DedupDecision, ExtractionResult, JudgeResult, WriteStats
from agent_arch.services.memory_v2.pipeline import run_l1_job


def _msg(i: int):
    return {"id": f"00000000-0000-0000-0000-0000000000{i:02d}", "conversation_id": "c", "query": f"q{i}",
            "answer": f"a{i}", "metadata": {}, "created_at": datetime(2026, 9, 4, 10, 0) + timedelta(minutes=i)}


class _Repo:
    def __init__(self, checkpoint=None, messages=None):
        self.checkpoint = checkpoint
        self.messages = messages or []
        self.calls: list[tuple] = []

    async def get_checkpoint(self, user_sub):
        self.calls.append(("get_checkpoint", user_sub))
        return self.checkpoint

    async def fetch_messages_after(self, user_sub, cursor_at, cursor_id, limit):
        self.calls.append(("fetch_after", limit))
        return self.messages[:limit]

    async def fetch_messages_before(self, user_sub, cursor_at, cursor_id, limit):
        return []

    async def list_alias_hints(self, user_sub):
        return ["watchdog → ACME Heros"]

    async def clear_pending(self, user_sub):
        self.calls.append(("clear_pending",))

    async def mark_job_failure(self, user_sub, error, **kw):
        self.calls.append(("mark_job_failure", error))
        return 1

    async def mark_job_success(self, user_sub, **kw):
        self.calls.append(("mark_job_success", kw))

    async def insert_generation_log(self, **kw):
        self.calls.append(("insert_generation_log", kw))
        return kw.get("log_id") or "log"


class _Redis:
    def __init__(self, acquire=True):
        self.acquire = acquire
        self.deleted: list[str] = []

    async def set_nx(self, key, value, expire=None):
        return self.acquire

    async def delete(self, key):
        self.deleted.append(key)
        return True


def _extraction(drafts, *, error=None, raw='{"atoms":[]}', scene="장면"):
    return ExtractionResult(scene_name=scene, drafts=drafts, raw=raw, model="m", prompt_sha="sha",
                            latency_ms=5, input_message_ids=["x"], error=error)


@pytest.fixture
def wire(monkeypatch):
    def _wire(repo: _Repo, *, redis=None, extraction=None, judge=None, stats=None):
        for name in ("get_checkpoint", "fetch_messages_after", "fetch_messages_before", "list_alias_hints",
                     "clear_pending", "mark_job_failure", "mark_job_success", "insert_generation_log"):
            monkeypatch.setattr(pipeline.repo, name, getattr(repo, name))
        r = redis or _Redis()
        monkeypatch.setattr(pipeline, "redis_store", r)

        async def fake_extract(new_msgs, background, **kw):
            return extraction

        async def fake_judge(user_sub, drafts):
            return judge or JudgeResult(decisions=[DedupDecision(index=i, action="store") for i in range(len(drafts))])

        async def fake_apply(user_sub, drafts, decisions, **kw):
            return stats or WriteStats(stored=len(drafts), output_atom_ids=[f"atom{i}" for i in range(len(drafts))])

        monkeypatch.setattr(pipeline.extractor, "extract_atoms", fake_extract)
        monkeypatch.setattr(pipeline.dedup, "judge_batch", fake_judge)
        monkeypatch.setattr(pipeline.writer, "apply_decisions", fake_apply)
        return r
    return _wire


def _names(repo):
    return [c[0] for c in repo.calls]


@pytest.mark.asyncio
async def test_locked_user_is_skipped(wire):
    repo = _Repo(checkpoint={"last_cursor_at": None, "last_cursor_id": None})
    wire(repo, redis=_Redis(acquire=False), extraction=_extraction([]))
    res = await run_l1_job("u1")
    assert res.status == "skipped_locked" and repo.calls == []


@pytest.mark.asyncio
async def test_no_checkpoint(wire):
    repo = _Repo(checkpoint=None)
    r = wire(repo, extraction=_extraction([]))
    res = await run_l1_job("u1")
    assert res.status == "no_checkpoint" and r.deleted == ["memory2:lock:u1"]


@pytest.mark.asyncio
async def test_no_new_messages_clears_pending(wire):
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x"}, messages=[])
    wire(repo, extraction=_extraction([]))
    res = await run_l1_job("u1")
    assert res.status == "no_new" and "clear_pending" in _names(repo) and "mark_job_success" not in _names(repo)


@pytest.mark.asyncio
async def test_llm_failure_keeps_cursor_and_schedules_retry(wire):
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x"}, messages=[_msg(1), _msg(2)])
    wire(repo, extraction=_extraction([], error="TimeoutError: down", raw=""))
    res = await run_l1_job("u1")
    assert res.status == "failed"
    assert "mark_job_failure" in _names(repo) and "mark_job_success" not in _names(repo)
    log = next(c[1] for c in repo.calls if c[0] == "insert_generation_log")
    assert log["status"] == "failed" and "retry 1/5" in log["error"]


@pytest.mark.asyncio
async def test_parse_failure_advances_cursor_and_logs_raw(wire):
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x"}, messages=[_msg(1), _msg(2)])
    wire(repo, extraction=_extraction([], error="parse_failed", raw="죄송합니다"))
    res = await run_l1_job("u1")
    assert res.status == "failed" and res.processed == 2
    success = next(c[1] for c in repo.calls if c[0] == "mark_job_success")
    assert success["cursor_id"] == _msg(2)["id"] and success["processed"] == 2 and success["has_more"] is False
    log = next(c[1] for c in repo.calls if c[0] == "insert_generation_log")
    assert log["status"] == "failed" and log["raw_output"] == "죄송합니다" and "mark_job_failure" not in _names(repo)


@pytest.mark.asyncio
async def test_happy_path_schedules_upper_layers_with_fresh_on_update(wire, upper_layers_spy):
    msgs = [_msg(1)]
    drafts = [AtomDraft(atom_type="instruction", content="x", source_message_ids=[msgs[0]["id"]])]
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x"}, messages=msgs)
    wire(repo, extraction=_extraction(drafts), stats=WriteStats(updated=1, output_atom_ids=["a"]))
    await run_l1_job("u1")
    assert upper_layers_spy == [{"user_sub": "u1", "delay": 10.0, "force_persona": True, "fresh_persona": True}]


@pytest.mark.asyncio
async def test_no_atoms_written_does_not_schedule_upper_layers(wire, upper_layers_spy):
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x"}, messages=[_msg(1)])
    wire(repo, extraction=_extraction([]))
    await run_l1_job("u1")
    assert upper_layers_spy == []


@pytest.mark.asyncio
async def test_happy_path_writes_log_and_advances_cursor(wire):
    msgs = [_msg(1), _msg(2), _msg(3)]
    drafts = [AtomDraft(atom_type="episodic", content="x", source_message_ids=[msgs[0]["id"]]),
              AtomDraft(atom_type="persona", content="y", source_message_ids=[msgs[2]["id"]])]
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x", "last_scene_name": "이전"}, messages=msgs)
    judge = JudgeResult(decisions=[DedupDecision(index=0, action="store"), DedupDecision(index=1, action="skip", target_ids=["t"])],
                        pool_size=1, raw="[…]", model="m", latency_ms=4, llm_called=True)
    wire(repo, extraction=_extraction(drafts, scene="메일 루틴"), judge=judge,
         stats=WriteStats(stored=1, skipped=1, output_atom_ids=["atom0"]))
    res = await run_l1_job("u1")
    assert res.status == "done" and res.processed == 3 and res.scene_name == "메일 루틴" and not res.has_more
    success = next(c[1] for c in repo.calls if c[0] == "mark_job_success")
    assert success["cursor_at"] == msgs[-1]["created_at"] and success["cursor_id"] == msgs[-1]["id"]
    assert success["atoms_added"] == 1 and success["scene_name"] == "메일 루틴"
    log = next(c[1] for c in repo.calls if c[0] == "insert_generation_log")
    assert log["status"] == "succeeded" and log["output_atom_ids"] == ["atom0"]
    assert log["decision_counts"]["store"] == 1 and log["decision_counts"]["skip"] == 1 and log["decision_counts"]["drafts"] == 2
    assert log["decision_counts"]["pool"] == 1
    assert log["latency_ms"] == 9 and "--- dedup ---" in log["raw_output"]
    assert log["log_id"] is not None


@pytest.mark.asyncio
async def test_backlog_sets_has_more(wire):
    msgs = [_msg(i) for i in range(1, 13)]   # 12 > NEW_BATCH(10)
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x"}, messages=msgs)
    wire(repo, extraction=_extraction([]))
    res = await run_l1_job("u1")
    assert res.status == "empty" and res.processed == 10 and res.has_more is True
    fetch = next(c for c in repo.calls if c[0] == "fetch_after")
    assert fetch[1] == pipeline.NEW_BATCH + 1
    success = next(c[1] for c in repo.calls if c[0] == "mark_job_success")
    assert success["cursor_id"] == msgs[9]["id"] and success["has_more"] is True


@pytest.mark.asyncio
async def test_unexpected_exception_marks_failure_and_releases_lock(wire, monkeypatch):
    repo = _Repo(checkpoint={"last_cursor_at": datetime(2026, 9, 4), "last_cursor_id": "x"}, messages=[_msg(1)])
    r = wire(repo, extraction=_extraction([]))

    async def boom(*a, **kw):
        raise ValueError("unexpected")

    monkeypatch.setattr(pipeline.repo, "fetch_messages_before", boom)
    res = await run_l1_job("u1")
    assert res.status == "failed" and "ValueError" in res.error
    assert "mark_job_failure" in _names(repo) and r.deleted == ["memory2:lock:u1"]
