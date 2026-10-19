"""L2 장면 — 이름 정규화, ops 검증, 상한, 적용(fake repo/LLM)."""
from __future__ import annotations

from datetime import datetime

import pytest

from agent_arch.services.memory_v2 import scenes as S


def test_normalize_name():
    assert S.normalize_name(" 메일 공유 루틴 ") == "메일-공유-루틴"
    assert S.normalize_name("PROC_견적/조회!!") == "proc-견적-조회"
    assert S.normalize_name("BATCH_2026") is None
    assert S.normalize_name("주간 요약") is None
    assert S.normalize_name("") is None
    assert len(S.normalize_name("가" * 100)) <= S.MAX_NAME_CHARS


def test_parse_ops_validation_and_mapping():
    tag_map = {"a1": "id-1", "a2": "id-2"}
    data = {"ops": [
        {"op": "create", "name": "메일 공유 루틴", "title": "메일 루틴", "summary": "요약\n줄", "content": "서사…", "atoms": ["a1", "zz"]},
        {"op": "update", "name": "없는-장면", "title": "t", "content": "c"},            # 없는 장면 update → create
        {"op": "create", "name": "기존", "title": "t2", "content": "c2", "atoms": "a2"},  # 있는 장면 create → update
        {"op": "delete", "name": "기존2"},
        {"op": "delete", "name": "없는-삭제"},                                          # 무시
        {"op": "create", "name": "제목없음", "content": "c"},                            # title 없음 → 무시
        {"op": "create", "name": "summary-only", "title": "t", "content": ""},         # content 없음 → 무시
        {"op": "rename", "name": "x", "title": "t", "content": "c"},                    # 모르는 op
        {"op": "create", "name": "메일 공유 루틴", "title": "dup", "content": "c"},       # 중복 이름 무시
    ]}
    ops = S.parse_ops(data, tag_map, existing_names={"기존", "기존2"})
    assert [(o.op, o.name) for o in ops] == [
        ("create", "메일-공유-루틴"), ("create", "없는-장면"), ("update", "기존"), ("delete", "기존2"),
    ]
    assert ops[0].atom_ids == ["id-1"] and ops[0].summary == "요약 줄"
    assert ops[2].atom_ids == ["id-2"]


def test_parse_ops_normalizes_latex_arrows():
    ops = S.parse_ops({"ops": [{"op": "create", "name": "proc", "title": "견적 $\\rightarrow$ 발주", "summary": "A $\\rightarrow$ B", "content": "받은 견적요청 $\\rightarrow$ bidding"}]}, {}, set())
    assert ops[0].title == "견적 → 발주" and ops[0].summary == "A → B" and "→" in ops[0].content and "$" not in ops[0].content


def test_parse_ops_accepts_bare_list_and_rejects_garbage():
    assert S.parse_ops([{"op": "delete", "name": "a"}], {}, {"a"})[0].op == "delete"
    assert S.parse_ops({"ops": "nope"}, {}, set()) == []
    assert S.parse_ops(None, {}, set()) == []


def test_enforce_cap_drops_trailing_creates():
    ops = [S.SceneOp("create", f"n{i}", "t", "", "c") for i in range(5)] + [S.SceneOp("delete", "old"), S.SceneOp("update", "u", "t", "", "c")]
    kept = S.enforce_cap(ops, existing_count=13, cap=15)      # 13 - 1 delete = 12 → room 3
    assert [o.name for o in kept if o.op == "create"] == ["n0", "n1", "n2"]
    assert any(o.op == "delete" for o in kept) and any(o.op == "update" for o in kept)
    assert S.enforce_cap(ops[:2], existing_count=0) == ops[:2]


def test_format_scene_input_tags():
    scenes = [{"name": "메일", "title": "메일 루틴", "heat": 2.5, "summary": "s", "content": "본문\n둘째줄"}]
    atoms = [{"id": "x1", "atom_type": "episodic", "scene_name": "메일", "priority": 30, "content": "9/4 메일 확인"}]
    text, tag_map = S.format_scene_input(scenes, atoms)
    assert "name=메일 | title=메일 루틴 | heat=2.5" in text and "content: 본문 둘째줄" in text
    assert "a1 [episodic|메일|p30] 9/4 메일 확인" in text and tag_map == {"a1": "x1"}


class _Repo:
    def __init__(self, atoms, scenes):
        self.atoms, self.scenes = atoms, scenes
        self.calls: list[tuple] = []

    async def list_atoms_since(self, user_sub, since, limit=120):
        return self.atoms

    async def list_scenes(self, user_sub, *, limit=8, with_content=False):
        return self.scenes

    async def decay_scene_heat(self, user_sub, factor=0.9):
        self.calls.append(("decay", factor))

    async def upsert_scene(self, user_sub, **kw):
        self.calls.append(("upsert", kw))
        return "sid"

    async def delete_scene(self, user_sub, name):
        self.calls.append(("delete", name))
        return True

    async def insert_generation_log(self, **kw):
        self.calls.append(("log", kw))
        return "log"


@pytest.fixture
def wire(monkeypatch):
    def _wire(repo: _Repo, raw: str | Exception):
        for name in ("list_atoms_since", "list_scenes", "decay_scene_heat", "upsert_scene", "delete_scene", "insert_generation_log"):
            monkeypatch.setattr(S.repo, name, getattr(repo, name))

        async def fake_invoke(messages):
            if isinstance(raw, Exception):
                raise raw
            return raw, "m", 3

        monkeypatch.setattr(S, "invoke_text", fake_invoke)
    return _wire


@pytest.mark.asyncio
async def test_rebuild_scenes_noop_without_atoms(wire):
    repo = _Repo([], [])
    wire(repo, '{"ops": []}')
    res = await S.rebuild_scenes("u1", since=None)
    assert res.changed == 0 and repo.calls == []


@pytest.mark.asyncio
async def test_rebuild_scenes_applies_ops_with_heat(wire):
    atoms = [{"id": "x1", "atom_type": "episodic", "scene_name": "메일", "priority": 30, "content": "a"},
             {"id": "x2", "atom_type": "persona", "scene_name": None, "priority": 50, "content": "b"}]
    scenes = [{"name": "옛장면", "title": "t", "heat": 1.0, "summary": "", "content": "c", "id": "s0"}]
    repo = _Repo(atoms, scenes)
    wire(repo, '```json\n{"ops":[{"op":"create","name":"메일 루틴","title":"메일","summary":"s","content":"서사","atoms":["a1","a2"]},{"op":"delete","name":"옛장면"}]}\n```')
    res = await S.rebuild_scenes("u1", since=datetime(2026, 9, 1))
    assert (res.created, res.updated, res.deleted, res.input_atoms) == (1, 0, 1, 2)
    kinds = [c[0] for c in repo.calls]
    assert kinds[0] == "decay" and "upsert" in kinds and "delete" in kinds and kinds[-1] == "log"
    up = next(c[1] for c in repo.calls if c[0] == "upsert")
    assert up["name"] == "메일-루틴" and up["heat_delta"] == 2.0 and up["evidence_atom_ids"] == ["x1", "x2"]
    log = next(c[1] for c in repo.calls if c[0] == "log")
    assert log["layer"] == "l2" and log["status"] == "succeeded" and log["decision_counts"]["created"] == 1


@pytest.mark.asyncio
async def test_rebuild_scenes_llm_failure_logs_failed(wire):
    repo = _Repo([{"id": "x1", "atom_type": "episodic", "scene_name": None, "priority": 30, "content": "a"}], [])
    wire(repo, RuntimeError("down"))
    res = await S.rebuild_scenes("u1", since=None)
    assert res.error.startswith("RuntimeError") and res.changed == 0
    assert [c[0] for c in repo.calls] == ["log"] and repo.calls[0][1]["status"] == "failed"


@pytest.mark.asyncio
async def test_rebuild_scenes_parse_failure(wire):
    repo = _Repo([{"id": "x1", "atom_type": "episodic", "scene_name": None, "priority": 30, "content": "a"}], [])
    wire(repo, "장면을 만들 수 없습니다")
    res = await S.rebuild_scenes("u1", since=None)
    assert res.error == "parse_failed" and repo.calls[0][1]["status"] == "failed"



class _RepairRepo:
    """repair_scenes 용 가짜 저장소 — 근거 해석 표(live_map)와 살아 있는 atom 을 준다."""

    def __init__(self, scenes, live_map, atoms):
        self.scenes, self.live_map, self.atoms = scenes, live_map, atoms
        self.calls: list[tuple] = []

    async def list_scenes(self, user_sub, *, limit=8, with_content=False):
        return self.scenes

    async def resolve_evidence(self, user_sub, ids):
        return {i: self.live_map.get(i) for i in ids}

    async def get_active_atoms_by_ids(self, user_sub, ids):
        return [self.atoms[i] for i in ids if i in self.atoms]

    async def rewrite_scene(self, user_sub, scene_id, **kw):
        self.calls.append(("rewrite", scene_id, kw))
        return True

    async def delete_scene_by_id(self, user_sub, scene_id):
        self.calls.append(("delete", scene_id))
        return True


@pytest.fixture
def repair_wire(monkeypatch):
    def _wire(repo, raw):
        for name in ("list_scenes", "resolve_evidence", "get_active_atoms_by_ids", "rewrite_scene", "delete_scene_by_id"):
            monkeypatch.setattr(S.repo, name, getattr(repo, name))

        async def fake_invoke(messages):
            fake_invoke.seen = messages
            if isinstance(raw, Exception):
                raise raw
            return raw, "m", 3
        monkeypatch.setattr(S, "invoke_text", fake_invoke)
        return fake_invoke
    return _wire


@pytest.mark.asyncio
async def test_repair_rewrites_from_live_evidence_and_deletes_dead_scenes(repair_wire):
    """지운 기억을 근거로 쓴 장면: 남은 근거가 있으면 그것만으로 다시 쓰고, 없으면 지운다. 멀쩡한 장면은 그대로."""
    scenes = [
        {"id": "s-dead", "name": "업무-경비-청구", "title": "경비", "summary": "", "content": "닭갈비 28,300원 …", "evidence_atom_ids": ["d1", "d2"]},
        {"id": "s-part", "name": "proc", "title": "PROC", "summary": "", "content": "입고 … 점심 …", "evidence_atom_ids": ["k1", "d3"]},
        {"id": "s-ok", "name": "메일", "title": "메일", "summary": "", "content": "…", "evidence_atom_ids": ["old"]},
    ]
    live = {"k1": "k1", "old": "new"}                       # old 는 대체돼 new 로 살아 있다(삭제 아님)
    atoms = {"k1": {"id": "k1", "atom_type": "episodic", "priority": 50, "content": "PROC 입고 검사를 문의했다"}}
    repo = _RepairRepo(scenes, live, atoms)
    invoke = repair_wire(repo, '{"title": "PROC 입고", "summary": "입고 검사 문의", "content": "PROC 입고 검사를 문의했다."}')
    res = await S.repair_scenes("u1")
    assert ("delete", "s-dead") in repo.calls
    rewrite = next(c for c in repo.calls if c[0] == "rewrite")
    assert rewrite[1] == "s-part" and rewrite[2]["evidence_atom_ids"] == ["k1"]   # 근거는 교체
    assert "점심" not in rewrite[2]["content"]
    assert "PROC 입고 검사를 문의했다" in invoke.seen[1].content                  # 남은 atom 만 재료로
    assert not any(c[1] == "s-ok" for c in repo.calls)
    assert (res.rewritten, res.deleted, res.failed) == (1, 1, 0)


@pytest.mark.asyncio
async def test_repair_llm_failure_deletes_scene(repair_wire):
    """다시 쓰기에 실패하면 장면을 지운다 — 지운 기억이 남는 것보다 낫다."""
    scenes = [{"id": "s1", "name": "x", "title": "x", "summary": "", "content": "…", "evidence_atom_ids": ["k1", "d1"]}]
    repo = _RepairRepo(scenes, {"k1": "k1"}, {"k1": {"id": "k1", "atom_type": "episodic", "priority": 50, "content": "c"}})
    repair_wire(repo, RuntimeError("llm down"))
    res = await S.repair_scenes("u1")
    assert repo.calls == [("delete", "s1")] and res.failed == 1


@pytest.mark.asyncio
async def test_repair_noop_when_all_evidence_alive(repair_wire):
    scenes = [{"id": "s1", "name": "x", "title": "x", "summary": "", "content": "…", "evidence_atom_ids": ["k1"]}]
    repo = _RepairRepo(scenes, {"k1": "k1"}, {})
    invoke = repair_wire(repo, "{}")
    res = await S.repair_scenes("u1")
    assert repo.calls == [] and res.changed == 0 and not hasattr(invoke, "seen")
