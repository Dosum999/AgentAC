"""배치 중복 판정 — 결정 파싱 규약과 fail-open(모두 store)."""
from __future__ import annotations

import pytest

from agent_arch.services.memory_v2 import dedup
from agent_arch.services.memory_v2.dedup import judge_batch, parse_decisions
from agent_arch.services.memory_v2.models import AtomDraft

E1, E2 = "aaaaaaaa-0000-0000-0000-000000000001", "aaaaaaaa-0000-0000-0000-000000000002"
N_TAGS = {"n1": 0, "n2": 1, "n3": 2}
E_TAGS = {"e1": E1, "e2": E2}


def _drafts(n=3):
    return [AtomDraft(atom_type="persona", content=f"사실 {i}", priority=50) for i in range(n)]


def test_parse_decisions_resolves_tags_and_defaults():
    arr = [
        {"new": "n1", "action": "update", "targets": ["e1"], "content": "최종 문장", "priority": 90},
        {"new": "n2", "action": "merge", "targets": []},                         # 대상 없는 merge → store
        {"new": "n1", "action": "skip", "targets": ["e2"]},                       # 중복 new 태그 무시
        {"new": "n9", "action": "skip"},                                          # 모르는 태그 무시
        "garbage",
    ]
    out = parse_decisions(arr, _drafts(), N_TAGS, E_TAGS)
    assert [d.action for d in out] == ["update", "store", "store"]              # n3 누락 → store
    assert out[0].target_ids == [E1] and out[0].content == "최종 문장" and out[0].priority == 90


def test_parse_decisions_intra_batch_duplicate_becomes_skip():
    arr = [{"new": "n2", "action": "merge", "targets": ["n1"]}]
    out = parse_decisions(arr, _drafts(), N_TAGS, E_TAGS)
    assert out[1].action == "skip" and out[1].target_ids == []


def test_parse_decisions_unfounded_skip_becomes_store():
    arr = [{"new": "n1", "action": "skip"}, {"new": "n2", "action": "skip", "targets": ["e2"]}]
    out = parse_decisions(arr, _drafts(), N_TAGS, E_TAGS)
    assert out[0].action == "store"
    assert out[1].action == "skip" and out[1].target_ids == [E2]


def test_parse_decisions_targetless_skip_honored_for_intra_batch_near_duplicate():
    drafts = [AtomDraft(atom_type="episodic", content="사용자는 2026-09-04에 안 읽은 메일과 채널 메시지를 확인하여 정리 요청하였다", priority=40),
              AtomDraft(atom_type="episodic", content="사용자는 2026-09-04에 안 읽은 메일과 채널 메시지를 확인하여 정리 요청했다.", priority=40),
              AtomDraft(atom_type="episodic", content="사용자는 2026-08-27에 PO-2026-0063 상세를 확인했다", priority=30)]
    arr = [{"new": "n2", "action": "skip"}, {"new": "n3", "action": "skip"}]
    out = parse_decisions(arr, drafts, N_TAGS, E_TAGS)
    assert out[0].action == "store"                       # 첫 번째는 살아남는다
    assert out[1].action == "skip"                        # 앞선 것과 거의 같은 문장 → skip 인정
    assert out[2].action == "store"                       # 닮은 게 없으면 근거 없는 skip → store


def test_parse_decisions_unknown_action_and_type_normalization():
    arr = [{"new": "n1", "action": "delete", "targets": ["e1"]},
           {"new": "n2", "action": "update", "targets": ["e1"], "atom_type": "directive", "content": "x"}]
    out = parse_decisions(arr, _drafts(), N_TAGS, E_TAGS)
    assert out[0].action == "store"
    assert out[1].atom_type == "instruction"


@pytest.mark.asyncio
async def test_judge_batch_empty():
    res = await judge_batch("u1", [])
    assert res.decisions == [] and not res.llm_called


@pytest.mark.asyncio
async def test_judge_batch_single_draft_without_pool_skips_llm(monkeypatch):
    called = {"llm": False}

    async def fake_recall(user_sub, drafts):
        return [], [[]]

    async def fake_invoke(messages):
        called["llm"] = True
        return "[]", "m", 1

    monkeypatch.setattr(dedup, "recall_candidates", fake_recall)
    monkeypatch.setattr(dedup, "invoke_text", fake_invoke)
    res = await judge_batch("u1", _drafts(1))
    assert res.decisions[0].action == "store" and not called["llm"]


@pytest.mark.asyncio
async def test_judge_batch_llm_failure_is_all_store(monkeypatch):
    async def fake_recall(user_sub, drafts):
        return [{"id": E1, "atom_type": "persona", "priority": 50, "scene_name": None, "content": "x"}], [[E1], []]

    async def fake_invoke(messages):
        raise RuntimeError("boom")

    monkeypatch.setattr(dedup, "recall_candidates", fake_recall)
    monkeypatch.setattr(dedup, "invoke_text", fake_invoke)
    res = await judge_batch("u1", _drafts(2))
    assert [d.action for d in res.decisions] == ["store", "store"]
    assert res.error.startswith("RuntimeError") and res.llm_called and res.pool_size == 1


@pytest.mark.asyncio
async def test_judge_batch_parse_failure_is_all_store(monkeypatch):
    async def fake_recall(user_sub, drafts):
        return [{"id": E1, "atom_type": "persona", "priority": 50, "scene_name": None, "content": "x"}], [[E1], []]

    async def fake_invoke(messages):
        return "판정할 수 없습니다", "m", 1

    monkeypatch.setattr(dedup, "recall_candidates", fake_recall)
    monkeypatch.setattr(dedup, "invoke_text", fake_invoke)
    res = await judge_batch("u1", _drafts(2))
    assert [d.action for d in res.decisions] == ["store", "store"] and res.error == "parse_failed"


@pytest.mark.asyncio
async def test_judge_batch_recall_failure_is_all_store(monkeypatch):
    async def fake_recall(user_sub, drafts):
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(dedup, "recall_candidates", fake_recall)
    res = await judge_batch("u1", _drafts(2))
    assert [d.action for d in res.decisions] == ["store", "store"] and "recall" in res.error


@pytest.mark.asyncio
async def test_judge_batch_happy_path_resolves_targets(monkeypatch):
    pool = [{"id": E1, "atom_type": "instruction", "priority": 80, "scene_name": "형식", "content": "표 금지"}]

    async def fake_recall(user_sub, drafts):
        return pool, [[E1], []]

    async def fake_invoke(messages):
        assert "e1 [instruction|p80|형식] 표 금지" in messages[1].content
        return '[{"new":"n1","action":"update","targets":["e1"],"content":"사용자는 표로 정리하기를 원한다","priority":90},{"new":"n2","action":"store"}]', "m", 3

    monkeypatch.setattr(dedup, "recall_candidates", fake_recall)
    monkeypatch.setattr(dedup, "invoke_text", fake_invoke)
    res = await judge_batch("u1", _drafts(2))
    assert res.decisions[0].action == "update" and res.decisions[0].target_ids == [E1]
    assert res.decisions[0].content == "사용자는 표로 정리하기를 원한다" and res.decisions[0].priority == 90
    assert res.decisions[1].action == "store" and res.error is None and res.pool_size == 1
