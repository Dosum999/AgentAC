"""판정 적용 — store/update/merge/skip 쓰기 의미론(버전·승계·priority·payload 합집합)."""
from __future__ import annotations

from datetime import datetime

import pytest

from agent_arch.services.memory_v2 import writer
from agent_arch.services.memory_v2.models import AtomDraft, DedupDecision
from agent_arch.services.memory_v2.writer import apply_decisions, merge_payload, resolve_atom_type, resolve_priority

T1 = "aaaaaaaa-0000-0000-0000-000000000001"
T2 = "aaaaaaaa-0000-0000-0000-000000000002"


class _Fake:
    def __init__(self, atoms_by_id=None):
        self.atoms = atoms_by_id or {}
        self.inserted: list[dict] = []
        self.superseded: list[tuple] = []
        self.touched: list[list[str]] = []
        self.deleted_points: list[list[str]] = []
        self.indexed: list[dict] = []
        self._n = 0

    async def insert_atom(self, user_sub, **kw):
        self._n += 1
        new_id = f"bbbbbbbb-0000-0000-0000-00000000000{self._n}"
        self.inserted.append({"id": new_id, "user_sub": user_sub, **kw})
        return new_id

    async def get_active_atoms_by_ids(self, user_sub, ids):
        return [self.atoms[i] for i in ids if i in self.atoms]

    async def supersede_atoms(self, user_sub, ids, new_id):
        self.superseded.append((tuple(ids), new_id))
        return list(ids)

    async def touch_atoms(self, user_sub, ids):
        self.touched.append(list(ids))

    async def _delete_points(self, ids):
        self.deleted_points.append(list(ids))

    async def _index_atom(self, atom):
        self.indexed.append(atom)


@pytest.fixture
def fake(monkeypatch):
    f = _Fake({
        T1: {"id": T1, "atom_type": "instruction", "content": "표 금지", "priority": 80, "version": 3,
             "source_message_ids": ["m-old"], "timestamps": [datetime(2026, 7, 1)], "payload": {}, "scene_name": "형식"},
        T2: {"id": T2, "atom_type": "alias", "content": "watchdog=ACME Heros", "priority": 60, "version": 1,
             "source_message_ids": [], "timestamps": [], "scene_name": None,
             "payload": {"aliases": ["watchdog"], "canonical_name": "ACME Heros", "canonical_kind": "mattermost_channel"}},
    })
    for name in ("insert_atom", "get_active_atoms_by_ids", "supersede_atoms", "touch_atoms", "_delete_points", "_index_atom"):
        monkeypatch.setattr(writer, name, getattr(f, name))
    return f


@pytest.mark.asyncio
async def test_store_inserts_version_1(fake):
    d = AtomDraft(atom_type="episodic", content="2026-09-04 메일 확인", priority=40, source_message_ids=["m1"])
    stats = await apply_decisions("u1", [d], [DedupDecision(index=0, action="store")], generation_id="g1", scene_name="메일",
                                  timestamps_for={0: [datetime(2026, 9, 4)]})
    assert stats.stored == 1 and stats.written == 1 and len(stats.output_atom_ids) == 1
    row = fake.inserted[0]
    assert row["version"] == 1 and row["priority"] == 40 and row["scene_name"] == "메일"
    assert row["generation_id"] == "g1" and row["timestamps"] == [datetime(2026, 9, 4)]
    assert fake.superseded == [] and len(fake.indexed) == 1


@pytest.mark.asyncio
async def test_update_supersedes_target_and_bumps_version(fake):
    d = AtomDraft(atom_type="instruction", content="표로 정리", priority=90, source_message_ids=["m-new"])
    dec = DedupDecision(index=0, action="update", target_ids=[T1], content="사용자는 표로 정리하기를 원한다")
    stats = await apply_decisions("u1", [d], [dec], generation_id="g", scene_name=None)
    assert stats.updated == 1
    row = fake.inserted[0]
    assert row["version"] == 4 and row["content"] == "사용자는 표로 정리하기를 원한다"
    assert row["source_message_ids"] == ["m-old", "m-new"]
    assert row["scene_name"] == "형식"                          # 대상의 장면을 이어받음
    assert fake.superseded == [((T1,), row["id"])]
    assert fake.deleted_points == [[T1]]


@pytest.mark.asyncio
async def test_merge_alias_unions_payload_and_raises_priority(fake):
    d = AtomDraft(atom_type="alias", content="watchdog 채널", priority=50,
                  payload={"aliases": ["watchdog", "왓치독"], "canonical_name": "ACME Heros", "canonical_id": "ch1"})
    dec = DedupDecision(index=0, action="merge", target_ids=[T2])
    stats = await apply_decisions("u1", [d], [dec], generation_id="g", scene_name=None)
    assert stats.merged == 1
    row = fake.inserted[0]
    assert row["payload"]["aliases"] == ["watchdog", "왓치독"]
    assert row["payload"]["canonical_id"] == "ch1" and row["payload"]["canonical_kind"] == "mattermost_channel"
    assert row["priority"] == 65                                # max(50,60)+5 (상한 70 이내)


@pytest.mark.asyncio
async def test_update_missing_target_becomes_store(fake):
    d = AtomDraft(atom_type="persona", content="x", priority=50)
    dec = DedupDecision(index=0, action="update", target_ids=["cccccccc-0000-0000-0000-000000000000"])
    stats = await apply_decisions("u1", [d], [dec], generation_id="g", scene_name=None)
    assert stats.stored == 1 and stats.updated == 0 and fake.superseded == []


@pytest.mark.asyncio
async def test_skip_touches_targets_only(fake):
    d = AtomDraft(atom_type="persona", content="x", priority=50)
    stats = await apply_decisions("u1", [d], [DedupDecision(index=0, action="skip", target_ids=[T1])], generation_id="g", scene_name=None)
    assert stats.skipped == 1 and fake.inserted == [] and fake.touched == [[T1]]


@pytest.mark.asyncio
async def test_missing_decision_defaults_to_store(fake):
    d = AtomDraft(atom_type="persona", content="x", priority=50)
    stats = await apply_decisions("u1", [d], [], generation_id="g", scene_name=None)
    assert stats.stored == 1


def test_resolve_priority_rules():
    d = AtomDraft(atom_type="instruction", content="x", priority=70)
    assert resolve_priority(d, DedupDecision(index=0, action="store"), [], "store") == 70
    assert resolve_priority(d, DedupDecision(index=0, action="store", priority=95), [], "store") == 95
    low = AtomDraft(atom_type="episodic", content="x", priority=50)
    assert resolve_priority(low, DedupDecision(index=0, action="merge"), [{"priority": 60}], "merge") == 65
    assert resolve_priority(low, DedupDecision(index=0, action="merge"), [{"priority": 68}], "merge") == 70  # 자동 merge 상한
    assert resolve_priority(d, DedupDecision(index=0, action="merge"), [{"priority": 60}], "merge") == 70    # 이미 상한 이상이면 그대로
    assert resolve_priority(d, DedupDecision(index=0, action="merge"), [{"priority": 80}], "merge") == 80    # 이미 높은 건 유지
    assert resolve_priority(d, DedupDecision(index=0, action="merge", priority=95), [{"priority": 80}], "merge") == 95  # 명시 지정은 상한 무관
    # 강제(-1) 전파: 대상이나 초안 중 하나라도 -1 이면 update/merge 결과도 -1
    assert resolve_priority(d, DedupDecision(index=0, action="update", priority=90), [{"priority": -1}], "update") == -1
    strict = AtomDraft(atom_type="instruction", content="x", priority=-1)
    assert resolve_priority(strict, DedupDecision(index=0, action="store"), [], "store") == -1


def test_resolve_atom_type_keeps_instruction_over_persona():
    """지시를 대체·병합한 persona 는 지시로 남는다 — 규칙이 참고 정보로 강등되지 않게(실측 2026-09-28)."""
    pref = AtomDraft(atom_type="persona", content="점심 상신 시 기존 결재선을 사용한다", priority=80)
    rule = [{"atom_type": "instruction", "priority": 80}]
    assert resolve_atom_type(pref, DedupDecision(index=0, action="update"), rule, "update") == "instruction"
    assert resolve_atom_type(pref, DedupDecision(index=0, action="merge"), rule, "merge") == "instruction"
    # 판정기가 persona 로 지정해도 대상이 지시면 지시
    assert resolve_atom_type(pref, DedupDecision(index=0, action="update", atom_type="persona"), rule, "update") == "instruction"
    # 대상에 지시가 없거나 store 면 그대로
    assert resolve_atom_type(pref, DedupDecision(index=0, action="update"), [{"atom_type": "persona"}], "update") == "persona"
    assert resolve_atom_type(pref, DedupDecision(index=0, action="store"), [], "store") == "persona"
    # 사건·별칭은 이 규칙 대상이 아니다
    ep = AtomDraft(atom_type="episodic", content="9/22 상신", priority=50)
    assert resolve_atom_type(ep, DedupDecision(index=0, action="merge"), rule, "merge") == "episodic"


def test_merge_payload_non_alias_prefers_draft():
    d = AtomDraft(atom_type="persona", content="x", payload={"k": "new"})
    assert merge_payload(d, [{"payload": {"k": "old", "z": 1}}], "persona") == {"k": "new", "z": 1}


@pytest.mark.asyncio
async def test_skip_promoted_to_update_when_alias_gains_canonical_id(fake):
    """판정기가 문장만 보고 skip 을 내려도, 신규 alias 가 정식 id 를 새로 확보했으면 payload 를 채운 update 로 올린다."""
    d = AtomDraft(atom_type="alias", content="watchdog 채널을 watchdog 이라 부른다", priority=60,
                  payload={"aliases": ["watchdog"], "canonical_name": "watchdog", "canonical_kind": "mattermost_channel",
                           "canonical_id": "4b7c"})
    stats = await apply_decisions("u1", [d], [DedupDecision(index=0, action="skip", target_ids=[T2])], generation_id="g", scene_name=None)
    assert stats.updated == 1 and stats.id_grounded == 1 and stats.skipped == 0
    new = fake.inserted[-1]
    assert new["payload"]["canonical_id"] == "4b7c" and "watchdog" in new["payload"]["aliases"]
    assert new["content"] == "watchdog=ACME Heros"            # 문장은 기존 것을 유지(중복 판정은 존중)
    assert new["priority"] == 60 and new["version"] == 2
    assert fake.superseded == [((T2,), new["id"])] and fake.touched == []


@pytest.mark.asyncio
async def test_skip_stays_skip_without_new_id_or_for_other_kinds(fake):
    fake.atoms[T2]["payload"]["canonical_id"] = "already"
    d = AtomDraft(atom_type="alias", content="x", priority=60,
                  payload={"aliases": ["watchdog"], "canonical_name": "watchdog", "canonical_kind": "mattermost_channel", "canonical_id": "new"})
    stats = await apply_decisions("u1", [d], [DedupDecision(index=0, action="skip", target_ids=[T2])], generation_id="g", scene_name=None)
    assert stats.skipped == 1 and fake.inserted == []            # 대상에 id 가 이미 있으면 그대로 skip
    del fake.atoms[T2]["payload"]["canonical_id"]
    d2 = AtomDraft(atom_type="alias", content="x", priority=60,
                   payload={"aliases": ["watchdog"], "canonical_name": "watchdog", "canonical_kind": "drive_file", "canonical_id": "f1"})
    stats = await apply_decisions("u1", [d2], [DedupDecision(index=0, action="skip", target_ids=[T2])], generation_id="g", scene_name=None)
    assert stats.skipped == 1 and fake.inserted == []            # 종류가 다르면 남의 id 를 붙이지 않는다
    d3 = AtomDraft(atom_type="alias", content="x", priority=60,
                   payload={"aliases": ["watchdog"], "canonical_name": "watchdog", "canonical_kind": "mattermost_channel"})
    stats = await apply_decisions("u1", [d3], [DedupDecision(index=0, action="skip", target_ids=[T2])], generation_id="g", scene_name=None)
    assert stats.skipped == 1 and fake.inserted == []            # 신규에 id 가 없으면 원래대로 skip


T3 = "aaaaaaaa-0000-0000-0000-000000000003"


@pytest.mark.asyncio
async def test_skip_promoted_to_global_when_same_rule_is_pinned_always(fake):
    """같은 규칙을 '항상'(전역)으로 다시 저장 — 판정기가 문장만 보고 skip 해도 등급을 전역(-1)으로 올린다(로컬 2026-09-29)."""
    fake.atoms[T3] = {"id": T3, "atom_type": "instruction", "content": "답변에 표를 쓰지 않는다", "priority": 90,
                      "version": 1, "payload": {}}
    d = AtomDraft(atom_type="instruction", content="표 쓰지 말 것", priority=-1)
    stats = await apply_decisions("u1", [d], [DedupDecision(index=0, action="skip", target_ids=[T3])], generation_id="g", scene_name=None)
    new = fake.inserted[-1]
    assert stats.updated == 1 and stats.skipped == 0 and new["priority"] == -1
    assert new["content"] == "답변에 표를 쓰지 않는다" and new["atom_type"] == "instruction"   # 문장은 기존 것
    fake.atoms[T3]["priority"] = -1                                                              # 이미 전역이면 그대로 skip
    stats = await apply_decisions("u1", [d], [DedupDecision(index=0, action="skip", target_ids=[T3])], generation_id="g", scene_name=None)
    assert stats.skipped == 1


def test_judge_cannot_demote_new_global_rule():
    d = AtomDraft(atom_type="instruction", content="답변 끝에 서명을 붙인다", priority=-1)
    assert resolve_priority(d, DedupDecision(index=0, action="store", priority=80), [], "store") == -1


def test_global_instruction_stays_instruction_when_judge_merges_into_note():
    """지운 규칙을 다시 전역으로 저장 — 판정기가 '규칙을 삭제했다' 기록(persona)에 합쳐도 지시로 남는다."""
    d = AtomDraft(atom_type="instruction", content="답변 끝에 서명을 붙인다", priority=-1)
    note = [{"id": T1, "atom_type": "persona", "content": "사용자는 서명 규칙을 삭제했다", "priority": 60}]
    dec = DedupDecision(index=0, action="update", target_ids=[T1], atom_type="persona")
    assert resolve_atom_type(d, dec, note, "update") == "instruction" and resolve_priority(d, dec, note, "update") == -1
