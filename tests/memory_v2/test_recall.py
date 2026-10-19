"""회상 v2 — 별칭 어휘 매칭, 하이브리드 RRF, 분할 블록 포매터, 타임아웃 fail-open."""
from __future__ import annotations

import asyncio

import pytest

from agent_arch.services.memory_v2 import recall as R


def _atom(i, atype="persona", content="c", priority=50, scene=None, payload=None):
    return {"id": f"a{i}", "atom_type": atype, "content": content, "priority": priority,
            "scene_name": scene, "payload": payload or {}}


# ─── 별칭 ───

def test_alias_hit_boundaries_and_josa():
    assert R.alias_hit("회사", "회사를 알려줘")
    assert R.alias_hit("회사", "어제 회사에서 온 메일")
    assert not R.alias_hit("회사", "회사미디어 매출")
    assert not R.alias_hit("scrum", "scrummage rules")
    assert R.alias_hit("watchdog", "watchdog에 올라온 거 요약")
    assert R.alias_hit("개발팀", "개발팀 채널 봐줘")


def test_match_aliases_uses_payload_aliases():
    pool = [_atom(1, "alias", payload={"aliases": ["watchdog"], "canonical_name": "ACME Heros"}),
            _atom(2, "alias", payload={"aliases": ["개발팀"], "canonical_name": "dev"})]
    hits = R.match_aliases("watchdog 요약해줘", pool)
    assert [h["id"] for h in hits] == ["a1"]
    assert R.match_aliases("", pool) == []


def test_format_alias():
    a = _atom(1, "alias", payload={"aliases": ["watchdog", "왓치독"], "canonical_name": "ACME Heros",
                                  "canonical_kind": "mattermost_channel", "canonical_id": "ch1"})
    line = R.format_alias(a)
    assert line.startswith('"watchdog", "왓치독" → mattermost_channel \'ACME Heros\' (id: ch1)')
    assert "mattermost_skill_get_channel_messages(channel_id=ch1)" in line       # id 를 받는 도구까지 잇는다
    plain = R.format_alias(_atom(2, "alias", payload={"aliases": ["클레임"], "canonical_name": "CAPA", "canonical_kind": "business_concept"}))
    assert plain == '"클레임" → business_concept \'CAPA\''


# ─── 포매터 ───

def test_turn_block_empty_when_nothing():
    assert R.format_turn_block([], [], []) == ""


def test_turn_block_sections_and_budget():
    strict = [_atom(1, "instruction", "답변에 표를 쓰지 않는다", -1)]
    aliases = [_atom(2, "alias", payload={"aliases": ["watchdog"], "canonical_name": "ACME Heros"})]
    atoms = [_atom(3, "episodic", "x" * 300, scene="메일"), _atom(4, "episodic", "y" * 300), _atom(5, "persona", "z" * 300)]
    block = R.format_turn_block(strict, aliases, atoms, max_chars=len(R.TURN_HEADER) + 460)
    assert block.startswith(R.TURN_HEADER)
    assert "지시" not in block                                    # 전역 지시는 system 안정 블록에만
    sys_block = R.format_system_block(None, [], strict)
    assert "● 전역 지시" in sys_block and "  - 답변에 표를 쓰지 않는다" in sys_block and "(id: a1)" not in sys_block
    assert "● 별칭" in block and "ACME Heros" in block
    assert "[episodic|메일] " + "x" * 300 in block
    assert "y" * 300 not in block                    # 예산 초과분은 잘린다


def test_turn_block_separates_user_rules_from_records():
    """사용자가 정한 규칙은 '참고용 과거 기록'과 다른 갈래로, 먼저 싣는다(예산이 모자라면 기록이 잘린다)."""
    atoms = [_atom(3, "episodic", "9월 22일에 김부장 결재자로 상신했다"),
             _atom(4, "instruction", "점심 상신 시 결재자는 김부장, 결제수단은 법인카드(2857)로 한다")]
    block = R.format_turn_block([], [], atoms)
    rules_at = block.index("● 사용자가 정한 규칙")
    records_at = block.index("● 관련 기억(과거 기록")
    assert rules_at < block.index("법인카드(2857)로 한다") < records_at < block.index("9월 22일")
    assert "참고용" not in block[:records_at]            # 규칙 갈래엔 '참고용' 프레이밍이 없다
    only_records = R.format_turn_block([], [], atoms[:1])
    assert "● 사용자가 정한 규칙" not in only_records and "● 관련 기억(과거 기록" in only_records
    tight = R.format_turn_block([], [], [atoms[0], _atom(5, "instruction", "r" * 100)],
                                max_chars=len(R.TURN_HEADER) + 200)
    assert "r" * 100 in tight and "9월 22일" not in tight


def test_system_block_always_has_guide():
    assert R.format_system_block(None, []) == R.TOOLS_GUIDE
    block = R.format_system_block("## 에이전트\n- 담당: MES", [{"name": "mail-routine", "title": "메일 루틴", "summary": "요약 후 DM"}])
    assert "● 사용자 페르소나" in block and "  - 담당: MES" in block
    assert "  - mail-routine: 메일 루틴 — 요약 후 DM" in block
    assert "memory_read_scene" in block


def test_strip_tool_guidance_keeps_content_drops_tool_pointers():
    """메모리 도구를 안 물리는 채널(가이드·MES 앱/PROC)용 — 도구 지시만 걷어낸다.

    남겨 두면 부를 수 없는 도구를 가리키는 죽은 지시가 된다. 페르소나·전역 지시·장면
    요약은 도구 없이도 유효하므로 살아 있어야 한다.
    """
    block = R.format_system_block(
        "## 에이전트\n- 담당: MES",
        [{"name": "mail-routine", "title": "메일 루틴", "summary": "요약 후 DM"}],
        [{"content": "답변에 표를 쓰지 않는다"}],
    )

    stripped = R.strip_tool_guidance(block)

    # 도구를 가리키는 문구는 전부 사라진다
    assert R.TOOLS_GUIDE not in stripped
    assert "memory_search" not in stripped
    assert "memory_read_scene" not in stripped
    assert "find_memory_tools" not in stripped
    # 내용은 남는다
    assert "  - 답변에 표를 쓰지 않는다" in stripped
    assert "  - 담당: MES" in stripped
    assert "  - mail-routine: 메일 루틴 — 요약 후 DM" in stripped


def test_strip_tool_guidance_on_empty_block():
    """회상이 비었을 때(타임아웃·실패 폴백 포함) 그대로 통과시킨다."""
    assert R.strip_tool_guidance("") == ""
    assert R.strip_tool_guidance(R.TOOLS_GUIDE) == ""


# ─── 하이브리드 ───

@pytest.fixture
def hybrid_env(monkeypatch):
    state = {"kw": [], "vec": [], "hydrated": []}

    async def kw(user_sub, q, *, limit, atom_type=None):
        return state["kw"]

    async def vec(user_sub, q, k, atom_type):
        return state["vec"]

    async def hydrate(user_sub, ids):
        return [a for a in state["hydrated"] if a["id"] in ids]

    monkeypatch.setattr(R.repo, "search_atoms_keyword", kw)
    monkeypatch.setattr(R, "_vector_search", vec)
    monkeypatch.setattr(R.repo, "get_active_atoms_by_ids", hydrate)
    return state


@pytest.mark.asyncio
async def test_hybrid_rrf_prefers_items_in_both_lists(hybrid_env):
    hybrid_env["kw"] = [{**_atom(1, content="A"), "score": 0.6}, {**_atom(2, content="B"), "score": 0.5}]
    hybrid_env["vec"] = [{"id": "a3", "score": 0.9}, {"id": "a2", "score": 0.8}]
    hybrid_env["hydrated"] = [_atom(3, content="C")]
    out = await R.hybrid_search("u", "q", limit=5)
    assert [o["id"] for o in out][0] == "a2"           # 양쪽에 있는 a2 가 1위
    assert {o["id"] for o in out} == {"a1", "a2", "a3"}
    assert out[0]["content"] == "B" and out[-1]["score"] > 0


@pytest.mark.asyncio
async def test_hybrid_floors_and_exclude(hybrid_env):
    hybrid_env["kw"] = [{**_atom(1), "score": 0.1}]                       # 하한 미달
    hybrid_env["vec"] = [{"id": "a2", "score": 0.95}, {"id": "a3", "score": 0.2}]  # a3 하한 미달
    hybrid_env["hydrated"] = [_atom(2, "alias")]
    assert await R.hybrid_search("u", "q", limit=5, exclude_types=("alias",)) == []
    assert [o["id"] for o in await R.hybrid_search("u", "q", limit=5)] == ["a2"]
    assert await R.hybrid_search("u", "   ", limit=5) == []


@pytest.mark.asyncio
async def test_hybrid_survives_source_failure(monkeypatch):
    async def kw_fail(*a, **kw):
        raise RuntimeError("pg down")

    async def vec(user_sub, q, k, atom_type):
        return [{"id": "a1", "score": 0.9}]

    async def hydrate(user_sub, ids):
        return [_atom(1)]

    monkeypatch.setattr(R.repo, "search_atoms_keyword", kw_fail)
    monkeypatch.setattr(R, "_vector_search", vec)
    monkeypatch.setattr(R.repo, "get_active_atoms_by_ids", hydrate)
    assert [o["id"] for o in await R.hybrid_search("u", "q", limit=3)] == ["a1"]


# ─── 진입점 ───

@pytest.mark.asyncio
async def test_recall_for_turn_timeout_is_fail_open_and_does_not_cancel(monkeypatch):
    finished = {"done": False}

    async def slow(user_sub, query, company_scope, cached=None, here=None):
        await asyncio.sleep(0.15)
        finished["done"] = True
        return R.RecallResult()

    monkeypatch.setattr(R, "_recall", slow)
    res = await R.recall_for_turn("u1", "q", timeout=0.05)
    assert res.timed_out and res.system_block == R.TOOLS_GUIDE and res.turn_block == ""
    assert not finished["done"]
    await asyncio.sleep(0.2)
    assert finished["done"]                      # shield: 늦은 태스크는 취소되지 않고 완주한다(DB 연결 보호)


def test_floors_follow_measured_boundary():
    # dev 실측: 관련 0.70~0.78 / 무관 ≤0.675 → 주입 하한은 그 사이, 도구 검색은 더 관대.
    assert 0.675 < R.VECTOR_MIN_SCORE <= 0.70
    assert R.TOOL_VECTOR_MIN_SCORE < R.VECTOR_MIN_SCORE


@pytest.mark.asyncio
async def test_recall_for_turn_exception_is_fail_open(monkeypatch):
    async def boom(user_sub, query, company_scope, cached=None, here=None):
        raise RuntimeError("x")

    monkeypatch.setattr(R, "_recall", boom)
    res = await R.recall_for_turn("u1", "q")
    assert res.error.startswith("RuntimeError") and res.system_block == R.TOOLS_GUIDE


@pytest.mark.asyncio
async def test_recall_for_turn_without_user_is_empty():
    res = await R.recall_for_turn(None, "q")
    assert res.system_block == "" and res.turn_block == ""


@pytest.mark.asyncio
async def test_recall_assembles_blocks_and_tolerates_one_failure(monkeypatch):
    async def strict(user_sub, limit):
        return [_atom(1, "instruction", "표를 쓰지 않는다", -1)]

    async def aliases(user_sub, *, atom_type=None, limit=50):
        return [_atom(2, "alias", payload={"aliases": ["watchdog"], "canonical_name": "ACME Heros"})]

    async def persona(user_sub, company_scope=None):
        return {"content": "## 에이전트\n- 담당: MES"}

    async def scenes_fail(user_sub, *, limit=8, with_content=False):
        raise RuntimeError("scenes down")

    async def hybrid(user_sub, query, *, limit, exclude_types=(), place=None):
        return [_atom(1, "instruction", "표를 쓰지 않는다", -1), _atom(3, "episodic", "9/4 메일 확인", scene="메일")]

    monkeypatch.setattr(R.repo, "list_strict_instructions", strict)
    monkeypatch.setattr(R.repo, "list_active_atoms", aliases)
    monkeypatch.setattr(R.repo, "get_active_persona", persona)
    monkeypatch.setattr(R.repo, "list_scenes", scenes_fail)
    monkeypatch.setattr(R, "hybrid_search", hybrid)
    res = await R.recall_for_turn("u1", "watchdog 요약해줘", company_scope="c")
    assert not res.timed_out and res.error is None
    assert "● 사용자 페르소나" in res.system_block and "기억 장면" not in res.system_block   # 장면 소스 실패는 그 소스만 비움
    assert "● 전역 지시" in res.system_block and "표를 쓰지 않는다" in res.system_block
    assert "표를 쓰지 않는다" not in res.turn_block                # 두 곳에 중복 주입하지 않는다
    assert "ACME Heros" in res.turn_block
    assert [a["id"] for a in res.atoms] == ["a3"]                # 지시와 겹치는 a1 은 관련 기억에서 제외
    assert res.system_block.count("표를 쓰지 않는다") == 1


# ─── 안정 블록 캐시(세션 초기화형) · 턴별 회상 플래그 ───

class _FakeRedis:
    def __init__(self):
        self.store: dict = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, expire=None):
        self.store[key] = value

    async def delete(self, key):
        return self.store.pop(key, None) is not None


@pytest.fixture
def cache_env(monkeypatch, stable_cache_spy):
    calls = {"strict": 0, "persona": 0, "scenes": 0, "hybrid": 0}
    redis = _FakeRedis()
    monkeypatch.setattr(R, "redis_store", redis)
    # conftest 가 무효화를 스파이로 바꿔 두므로 여기서는 진짜 구현(캐시 삭제)을 되살린다
    async def real_invalidate(user_sub):
        await redis.delete(R._stable_key(user_sub))
    monkeypatch.setattr(R, "invalidate_stable_block", real_invalidate)

    async def strict(user_sub, limit):
        calls["strict"] += 1
        return [_atom(1, "instruction", "표를 쓰지 않는다", -1)]

    async def aliases(user_sub, *, atom_type=None, limit=50):
        return []

    async def persona(user_sub, company_scope=None):
        calls["persona"] += 1
        return {"content": "## 사용자"}

    async def scenes(user_sub, *, limit=8, with_content=False):
        calls["scenes"] += 1
        return []

    async def hybrid(user_sub, query, *, limit, exclude_types=(), place=None):
        calls["hybrid"] += 1
        return [_atom(1, "instruction", "표를 쓰지 않는다", -1), _atom(3, "episodic", "9/4 메일 확인")]

    monkeypatch.setattr(R.repo, "list_strict_instructions", strict)
    monkeypatch.setattr(R.repo, "list_active_atoms", aliases)
    monkeypatch.setattr(R.repo, "get_active_persona", persona)
    monkeypatch.setattr(R.repo, "list_scenes", scenes)
    monkeypatch.setattr(R, "hybrid_search", hybrid)
    monkeypatch.setattr(R.settings, "memory_v2_turn_recall", True, raising=False)
    return calls, redis


@pytest.mark.asyncio
async def test_stable_block_cached_per_user_and_invalidated(cache_env):
    calls, redis = cache_env
    r1 = await R.recall_for_turn("u1", "q1", company_scope="c")
    r2 = await R.recall_for_turn("u1", "q2", company_scope="c")
    assert r1.system_block == r2.system_block and "● 전역 지시" in r2.system_block
    assert calls["strict"] == 1 and calls["persona"] == 1 and calls["scenes"] == 1     # 두 번째 턴은 캐시
    assert calls["hybrid"] == 2                                                        # 턴별 회상은 매 턴
    assert [a["id"] for a in r2.atoms] == ["a3"]                                       # 캐시된 strict_ids 로도 중복 제외
    await R.invalidate_stable_block("u1")
    await R.recall_for_turn("u1", "q3", company_scope="c")
    assert calls["strict"] == 2                                                        # 무효화 뒤 재계산
    # 회사 스코프가 다르면 캐시를 쓰지 않는다
    await R.recall_for_turn("u1", "q4", company_scope="other")
    assert calls["strict"] == 3


@pytest.mark.asyncio
async def test_cache_hit_keeps_global_rules_for_turn_block_and_reminder(cache_env):
    """로컬(2026-09-29): 캐시 미스 턴은 지시=1 로 서명 규칙이 적용되고, 바로 뒤 적중 턴은 strict 가 비어 규칙 리마인더에서 빠졌다."""
    calls, _ = cache_env
    await R.recall_for_turn("u1", "q1", company_scope="c")
    r2 = await R.recall_for_turn("u1", "q2", company_scope="c")
    assert calls["strict"] == 1                                                        # 적중(재조회 없음)
    assert [a["content"] for a in r2.strict] == ["표를 쓰지 않는다"]
    assert "(모든 답변에 적용) 표를 쓰지 않는다" in R.rule_reminder(r2)                 # 전역 지시는 항상 적용이라고 밝힌다
    assert "표를 쓰지 않는다" in R.rule_reminder(r2) and "표를 쓰지 않는다" in r2.system_block   # 턴 블록엔 원래 안 싣는다(시스템 블록 몫)


@pytest.mark.asyncio
async def test_upper_stale_omits_persona_and_scenes_until_rebuilt(cache_env):
    """로컬(2026-09-29): '[요약]' 규칙을 지운 8초 뒤 답에 [요약]이 또 붙었다 — 재작성 전 옛 페르소나를 읽었다."""
    calls, redis = cache_env
    await R.mark_upper_stale("u1")
    r1 = await R.recall_for_turn("u1", "q1", company_scope="c")
    assert "## 사용자" not in r1.system_block and "● 전역 지시" in r1.system_block          # 페르소나는 빼고 전역 지시는 싣는다
    assert R._stable_key("u1") not in redis.store                                          # 빠진 블록은 캐시하지 않는다
    await R.clear_upper_stale("u1")
    r2 = await R.recall_for_turn("u1", "q2", company_scope="c")
    assert "## 사용자" in r2.system_block


@pytest.mark.asyncio
async def test_turn_recall_flag_off_skips_hybrid(cache_env, monkeypatch):
    calls, _ = cache_env
    monkeypatch.setattr(R.settings, "memory_v2_turn_recall", False, raising=False)
    res = await R.recall_for_turn("u1", "watchdog 요약")
    assert calls["hybrid"] == 0 and res.atoms == [] and "● 전역 지시" in res.system_block


def test_recalled_user_rules_only_instructions_deduped():
    rec = R.RecallResult(
        strict=[_atom(1, "instruction", "답변에 표를 쓰지 않는다", -1)],
        atoms=[_atom(2, "instruction", "점심 상신은 법인카드로 한다"), _atom(3, "episodic", "9/22 상신"),
               _atom(4, "persona", "녹차를 좋아한다"), _atom(5, "instruction", "답변에 표를 쓰지 않는다")],
    )
    assert R.recalled_user_rules(rec) == ["답변에 표를 쓰지 않는다", "점심 상신은 법인카드로 한다"]
    assert R.recalled_user_rules(None) == []


def test_rule_reminder_empty_without_rules_and_capped():
    assert R.rule_reminder(None) == ""
    assert R.rule_reminder(R.RecallResult(atoms=[_atom(1, "episodic", "9/22 상신")])) == ""
    text = R.rule_reminder(R.RecallResult(atoms=[_atom(i, "instruction", f"규칙{i} " + "가" * 200) for i in range(5)]))
    assert text.startswith("[사용자가 정한 규칙") and "규칙0" in text and "규칙4" not in text   # 상한
    assert "다시 묻지 말고" in text
