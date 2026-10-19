"""명시 저장·잊기(v2) — SoR 거부, 판정기 경유, 전역 강제 승격, 멱등 forget."""
from __future__ import annotations

import pytest

from agent_arch.services.memory_v2 import explicit
from agent_arch.services.memory_v2.models import DedupDecision, JudgeResult, WriteStats


@pytest.fixture
def env(monkeypatch):
    calls = {"judge": [], "apply": [], "log": []}

    async def judge(user_sub, drafts):
        calls["judge"].append(drafts)
        return JudgeResult(decisions=[DedupDecision(index=0, action="store")], model="m")

    async def apply(user_sub, drafts, decisions, **kw):
        calls["apply"].append((drafts, decisions, kw))
        return WriteStats(stored=1, output_atom_ids=["atom-1"])

    async def log(**kw):
        calls["log"].append(kw)
        return "log"

    monkeypatch.setattr(explicit.dedup, "judge_batch", judge)
    monkeypatch.setattr(explicit.writer, "apply_decisions", apply)
    monkeypatch.setattr(explicit.repo, "insert_generation_log", log)
    return calls


@pytest.mark.asyncio
async def test_save_refuses_sor_values(env):
    assert await explicit.save_explicit_atom("u1", "내 연차는 12일 남았다") is None
    assert env["judge"] == []


@pytest.mark.asyncio
async def test_save_default_instruction_priority_80(env):
    out = await explicit.save_explicit_atom("u1", "답변에 표를 쓰지 않는다", company_scope="c")
    assert out == {"id": "atom-1", "action": "store", "atom_type": "instruction", "content": "답변에 표를 쓰지 않는다", "priority": 80}
    draft = env["judge"][0][0]
    assert draft.priority == 80 and draft.atom_type == "instruction" and draft.company_scope == "c"
    assert env["apply"][0][2]["company_scope"] == "c"
    assert env["log"][0]["status"] == "succeeded" and env["log"][0]["model"] == "explicit/m"


@pytest.mark.asyncio
async def test_save_strict_marker_promotes_to_global(env):
    await explicit.save_explicit_atom("u1", "앞으로 항상 존댓말로 답한다")
    assert env["judge"][0][0].priority == -1


@pytest.mark.asyncio
async def test_save_type_normalization(env):
    await explicit.save_explicit_atom("u1", "녹차를 선호한다", atom_type="preference")
    assert env["judge"][0][0].atom_type == "persona"
    await explicit.save_explicit_atom("u1", "x", atom_type="unknown")
    assert env["judge"][1][0].atom_type == "instruction"


@pytest.mark.asyncio
async def test_save_reports_written_type_after_update(monkeypatch, env):
    """쓰기 단계가 종류를 지시로 유지하면 도구 응답도 실제 저장된 종류를 말한다."""
    async def judge_update(user_sub, drafts):
        return JudgeResult(decisions=[DedupDecision(index=0, action="update", target_ids=["t"])])

    async def apply_update(user_sub, drafts, decisions, **kw):
        return WriteStats(updated=1, output_atom_ids=["atom-2"])

    async def by_ids(user_sub, ids):
        return [{"id": "atom-2", "atom_type": "instruction"}]

    async def noop(*a, **k):
        return None

    from agent_arch.services.memory_v2 import layers, recall
    monkeypatch.setattr(explicit.dedup, "judge_batch", judge_update)
    monkeypatch.setattr(explicit.writer, "apply_decisions", apply_update)
    monkeypatch.setattr(explicit.repo, "get_active_atoms_by_ids", by_ids)
    monkeypatch.setattr(recall, "invalidate_stable_block", noop)
    monkeypatch.setattr(layers, "schedule_upper_layers", lambda *a, **k: None)
    out = await explicit.save_explicit_atom("u1", "점심 상신 시 기존 결재선을 사용한다", atom_type="preference")
    assert out["action"] == "update" and out["atom_type"] == "instruction" and out["id"] == "atom-2"


@pytest.mark.asyncio
async def test_save_skip_reports_action(monkeypatch, env):
    async def judge_skip(user_sub, drafts):
        return JudgeResult(decisions=[DedupDecision(index=0, action="skip", target_ids=["t"])])

    async def apply_skip(user_sub, drafts, decisions, **kw):
        return WriteStats(skipped=1)

    monkeypatch.setattr(explicit.dedup, "judge_batch", judge_skip)
    monkeypatch.setattr(explicit.writer, "apply_decisions", apply_skip)
    out = await explicit.save_explicit_atom("u1", "답변에 표를 쓰지 않는다")
    assert out["action"] == "skip" and out["id"] is None


@pytest.mark.asyncio
async def test_forget_atom_idempotent_and_tolerates_qdrant(monkeypatch):
    rows = {"11111111-1111-4111-8111-111111111111": {"id": "11111111-1111-4111-8111-111111111111", "atom_type": "instruction", "content": "x", "scene_name": None}}

    async def soft_delete(user_sub, atom_id):
        return rows.pop(atom_id, None)

    async def broken_indexer():
        raise ConnectionError("qdrant down")

    monkeypatch.setattr(explicit.repo, "soft_delete_atom", soft_delete)
    import agent_arch.services.memory_v2.indexer as idx
    monkeypatch.setattr(idx, "get_v2_indexer", broken_indexer)
    assert (await explicit.forget_atom("u1", "11111111-1111-4111-8111-111111111111"))["atom_type"] == "instruction"
    assert await explicit.forget_atom("u1", "11111111-1111-4111-8111-111111111111") is None
    assert await explicit.forget_atom("u1", "업무-경비-청구") is None     # id 형식이 아니면 DB 에 넣지 않는다


@pytest.mark.asyncio
async def test_save_binds_to_the_current_window(env, monkeypatch):
    """분리된 창(MES 앱)에서 "기억해둬" 한 규칙은 그 창에 매인다.

    자동 추출은 근거 턴 metadata 로 장소를 찍는데(extractor.stamp_places) 명시 저장만
    빠져 있었다 — 그 경로로 저장된 규칙은 브라우저 패널까지 따라다녔다.
    """
    from agent_arch.agents import context as C

    C.set_user_context({"sub": "u1", "memory_surface": "mes"})
    try:
        await explicit.save_explicit_atom("u1", "앞으로 수량은 천 단위로 끊어 보여줘")
    finally:
        C.set_user_context({})

    assert env["judge"][0][0].places == ["agent:mes"]


@pytest.mark.asyncio
async def test_save_from_panel_stays_neutral(env):
    """패널·DM 은 자리가 없다 — 그 사람 자체의 맥락이라 어디서나 적용된다(종전 동작)."""
    from agent_arch.agents import context as C

    C.set_user_context({"sub": "u1"})
    try:
        await explicit.save_explicit_atom("u1", "답변은 짧게 해줘")
    finally:
        C.set_user_context({})

    assert env["judge"][0][0].places == []



def test_is_atom_id():
    assert explicit.is_atom_id("11111111-1111-4111-8111-111111111111") and explicit.is_atom_id("11111111111141118111111111111111")
    assert not explicit.is_atom_id("업무-경비-청구") and not explicit.is_atom_id("") and not explicit.is_atom_id("a1")


@pytest.mark.asyncio
async def test_forget_scene_by_name(monkeypatch):
    scenes = {"업무-경비-청구": {"id": "s1", "name": "업무-경비-청구", "title": "경비 청구", "summary": "…"}}
    deleted, spy = [], []

    async def get_scene(user_sub, name):
        return scenes.get(name)

    async def delete_scene(user_sub, name):
        deleted.append(name)
        return True

    async def noop(*a, **k):
        return None

    from agent_arch.services.memory_v2 import layers, recall
    monkeypatch.setattr(explicit.repo, "get_scene", get_scene)
    monkeypatch.setattr(explicit.repo, "delete_scene", delete_scene)
    monkeypatch.setattr(recall, "invalidate_stable_block", noop)
    monkeypatch.setattr(layers, "schedule_upper_layers", lambda *a, **k: spy.append(k))
    out = await explicit.forget_scene("u1", "업무-경비-청구")
    assert out["title"] == "경비 청구" and deleted == ["업무-경비-청구"]
    assert spy and spy[0]["fresh_persona"] is True                  # 페르소나는 장면 본문을 읽는다
    assert await explicit.forget_scene("u1", "없는-장면") is None
