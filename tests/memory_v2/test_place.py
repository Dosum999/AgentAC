"""기억의 장소·출처 등급 — 코드가 정하고, 회상 순위·상시 블록 자격·강제 지시 천장에 쓰인다."""
from __future__ import annotations

import json

import pytest

from agent_arch.services.memory_v2 import place as P
from agent_arch.services.memory_v2.models import AtomDraft
from agent_arch.services.memory_v2.writer import _strict_allowed, resolve_priority, stamp_place
from agent_arch.services.memory_v2.models import DedupDecision

_CH = {"provider": "mattermost", "channel_id": "ch-1", "is_dm": False, "channel_name": "개발팀"}
_DM = {"provider": "mattermost", "channel_id": "ch-dm", "is_dm": True}


def _turn(place=None, excerpt=False, as_json=False):
    meta = {}
    if place:
        meta["place"] = place
    if excerpt:
        meta["tool_digest"] = [{"tool": "mattermost_skill_get_channel_messages",
                                "messages": [{"text": "네 제가 할게요", "author": "이대리"}]}]
    return {"id": "m1", "metadata": json.dumps(meta) if as_json else meta}


class TestKeys:
    def test_channel_makes_a_place_but_dm_does_not(self):
        """DM 은 특정 장소가 아니라 그 사람 자체의 맥락이다 — 장소로 묶으면 다른 자리서 못 꺼낸다."""
        assert P.key_of(_CH) == "mattermost:ch-1"
        assert P.key_of(_DM) is None
        assert P.key_of({"channel_id": ""}) is None
        assert P.key_of(None) is None

    def test_places_of_collects_channels_only_once(self):
        assert P.places_of([_turn(_CH), _turn(_CH), _turn(_DM)]) == ["mattermost:ch-1"]

    def test_metadata_may_arrive_as_json_text(self):
        assert P.places_of([_turn(_CH, as_json=True)]) == ["mattermost:ch-1"]

    def test_panel_turn_has_no_place(self):
        assert P.places_of([_turn()]) == []
        assert P.places_of([]) == []


class TestRankingAndEligibility:
    def _atom(self, places=(), origin=P.ORIGIN_USER):
        return {"id": "a", "payload": {P.PAYLOAD_PLACES: list(places), P.PAYLOAD_ORIGIN: origin}}

    def test_place_factor(self):
        here = "mattermost:ch-1"
        assert P.rank_factor(self._atom([here]), here) == P.FACTOR_SAME
        assert P.rank_factor(self._atom(["mattermost:ch-2"]), here) == P.FACTOR_OTHER
        assert P.rank_factor(self._atom(), here) == P.FACTOR_NEUTRAL      # 장소 없는 기억은 중립
        # 장소 밖(패널·DM)에서는 아무것도 깎지 않는다 — 거기는 "다른 채널"이 아니다
        assert P.rank_factor(self._atom([here]), None) == P.FACTOR_NEUTRAL

    def test_eligibility_only_binds_placed_memories(self):
        here = "mattermost:ch-1"
        assert P.eligible(self._atom(), here) is True                     # 장소 없으면 언제나
        assert P.eligible(self._atom(), None) is True
        assert P.eligible(self._atom([here]), here) is True
        assert P.eligible(self._atom([here]), "mattermost:ch-2") is False
        # 특정 채널에서 못 박은 규칙은 장소 밖(패널·DM)에서도 상시 주입되지 않는다.
        # 검색으로는 여전히 올라온다 — 막는 것은 "매번 강제로 읽히는 자리" 하나뿐이다.
        assert P.eligible(self._atom([here]), None) is False
        assert P.rank_factor(self._atom([here]), None) == P.FACTOR_NEUTRAL


class TestWriterContract:
    def test_places_union_and_mixed_wins(self):
        draft = AtomDraft(atom_type="episodic", content="x",
                          places=["mattermost:ch-1"], origin=P.ORIGIN_USER)
        targets = [{"payload": {P.PAYLOAD_PLACES: ["mattermost:ch-2"], P.PAYLOAD_ORIGIN: P.ORIGIN_MIXED}}]
        out = stamp_place({}, draft, targets)

        assert out[P.PAYLOAD_PLACES] == ["mattermost:ch-1", "mattermost:ch-2"]
        assert out[P.PAYLOAD_ORIGIN] == P.ORIGIN_MIXED          # 한 번 섞이면 계속 섞인 것

    def test_placeless_user_memory_keeps_its_payload_unchanged(self):
        """패널에서 오던 기억은 저장 모양이 이 기능 이전과 같아야 한다 — 키가 하나도 안 는다."""
        out = stamp_place({"aliases": ["x"]}, AtomDraft(atom_type="alias", content="x"), [])
        assert out == {"aliases": ["x"]}

    def test_mixed_is_written_even_without_a_place(self):
        out = stamp_place({}, AtomDraft(atom_type="episodic", content="x", origin=P.ORIGIN_MIXED), [])
        assert out == {P.PAYLOAD_ORIGIN: P.ORIGIN_MIXED}

    def test_mixed_memory_cannot_become_an_always_injected_rule(self):
        """남이 채널에서 한 말이 에이전트의 상시 규칙이 되면 안 된다 — 감점이 아니라 선행 조건."""
        mixed = AtomDraft(atom_type="instruction", content="항상 표로", priority=-1,
                          origin=P.ORIGIN_MIXED)
        assert _strict_allowed(mixed, []) is False
        assert resolve_priority(mixed, DedupDecision(index=0, action="store"), [], "store") == 80

        owned = AtomDraft(atom_type="instruction", content="항상 표로", priority=-1)
        assert _strict_allowed(owned, []) is True
        assert resolve_priority(owned, DedupDecision(index=0, action="store"), [], "store") == -1

    def test_mixed_target_also_blocks_promotion(self):
        owned = AtomDraft(atom_type="instruction", content="항상 표로", priority=-1)
        targets = [{"priority": -1, "payload": {P.PAYLOAD_ORIGIN: P.ORIGIN_MIXED}}]
        assert _strict_allowed(owned, targets) is False
        assert resolve_priority(owned, DedupDecision(index=0, action="merge"), targets, "merge") != -1


class TestPipelineStamping:
    def test_stamp_places_marks_each_draft_from_its_evidence(self):
        """장소만 찍는다 — 등급은 build_drafts 가 인용된 근거로 이미 정했으므로 건드리지 않는다."""
        from agent_arch.services.memory_v2 import extractor

        msgs = [{"id": "m1", "metadata": {"place": _CH}}, {"id": "m2", "metadata": {}}]
        d1 = AtomDraft(atom_type="episodic", content="a", source_message_ids=["m1"])
        d2 = AtomDraft(atom_type="episodic", content="b", source_message_ids=["m2"],
                       origin=P.ORIGIN_MIXED)
        d3 = AtomDraft(atom_type="episodic", content="c", source_message_ids=["없는id"])
        extractor.stamp_places(msgs, [d1, d2, d3])

        assert (d1.places, d1.origin) == (["mattermost:ch-1"], P.ORIGIN_USER)
        assert (d2.places, d2.origin) == ([], P.ORIGIN_MIXED)     # 등급 보존
        assert (d3.places, d3.origin) == ([], P.ORIGIN_USER)


class TestRecallWiring:
    def test_flag_off_means_no_place_at_all(self, monkeypatch):
        """끄면 회상은 이 기능 이전과 같다 — 장소를 아예 안 본다."""
        from agent_arch.agents.context import set_user_context
        from agent_arch.core.config import settings
        from agent_arch.services.memory_v2 import recall

        set_user_context({"sub": "u1", "origin": {"channel_id": "ch-1", "is_direct": False}})
        monkeypatch.setattr(settings, "memory_v2_place_scope", False, raising=False)
        assert recall._current_place() is None

        monkeypatch.setattr(settings, "memory_v2_place_scope", True, raising=False)
        assert recall._current_place() == "mattermost:ch-1"
        set_user_context({})
        assert recall._current_place() is None


@pytest.mark.asyncio
async def test_same_place_memory_outranks_the_other_channel(monkeypatch):
    """같은 채널 기억이 올라오고 다른 채널 기억은 살짝 내려간다 — 떨어뜨리지는 않는다."""
    from agent_arch.db.repository import memory_v2 as repo
    from agent_arch.services.memory_v2 import recall

    here, there = {P.PAYLOAD_PLACES: ["mattermost:ch-1"]}, {P.PAYLOAD_PLACES: ["mattermost:ch-9"]}
    rows = [{"id": "other", "atom_type": "episodic", "content": "다른 채널", "payload": there, "score": 0.9},
            {"id": "same", "atom_type": "episodic", "content": "같은 채널", "payload": here, "score": 0.8}]

    async def _kw(user_sub, q, *, limit, atom_type=None):
        return rows

    monkeypatch.setattr(repo, "search_atoms_keyword", _kw)
    monkeypatch.setattr(recall, "_vector_search", lambda *a, **k: _none())

    async def _none():
        return []

    without = await recall.hybrid_search("u1", "질문", limit=5)
    with_place = await recall.hybrid_search("u1", "질문", limit=5, place="mattermost:ch-1")

    assert [a["id"] for a in without] == ["other", "same"]
    assert [a["id"] for a in with_place] == ["same", "other"]    # 순서만 바뀌고 둘 다 남는다


class TestStableCacheIsPlaceAware:
    """30분 캐시가 장소를 모르면, 한 채널에서 만든 상시 블록이 다른 자리에 그대로 재사용된다.

    자격 필터는 콜드 경로에서만 도는데 캐시 히트 경로는 strict 를 비운 채 블록을 그대로 쓴다 —
    그래서 채널 전용 규칙이 30분 동안 패널에도 따라붙었다(2026-09-22 검증 중 발견).
    """

    def _cached(self, bound):
        return {"company": None, "block": "b", "strict_ids": [], "place_bound": bound}

    @pytest.mark.asyncio
    async def test_block_bound_to_a_channel_is_not_reused_elsewhere(self, monkeypatch):
        from agent_arch.memory.redis_store import redis_store
        from agent_arch.services.memory_v2 import recall

        stored = {}
        monkeypatch.setattr(redis_store, "get", lambda k: _async(stored.get(k)))
        monkeypatch.setattr(redis_store, "set", lambda k, v, expire=None: _async(stored.__setitem__(k, v)))

        stored[recall._stable_key("u1")] = self._cached("mattermost:ch-1")
        assert await recall._get_stable_cache("u1", None, "mattermost:ch-1") is not None
        assert await recall._get_stable_cache("u1", None, "mattermost:ch-2") is None
        assert await recall._get_stable_cache("u1", None, None) is None

    @pytest.mark.asyncio
    async def test_block_without_placed_rules_stays_warm_everywhere(self, monkeypatch):
        """장소에 매인 지시가 없으면 블록은 어디서나 같다 — 채널을 옮겨도 캐시가 살아야 한다."""
        from agent_arch.memory.redis_store import redis_store
        from agent_arch.services.memory_v2 import recall

        stored = {recall._stable_key("u1"): self._cached(None)}
        monkeypatch.setattr(redis_store, "get", lambda k: _async(stored.get(k)))

        for here in ("mattermost:ch-1", "mattermost:ch-9", None):
            assert await recall._get_stable_cache("u1", None, here) is not None


async def _async(value):
    return value


class TestOriginIsGradedPerEvidence:
    """스레드에서 이야기하다 사용자가 직접 지시하는 것이 메신저의 자연스러운 흐름이다.

    턴 단위로 등급을 매기면 그 턴이 채널을 읽었다는 이유로 본인 지시까지 남의 말로 강등돼
    상시 규칙이 되지 못했다(사용자 지적 2026-09-22). 인용된 근거 단위로 가른다.
    """

    def _row(self, excerpt=True):
        meta = {"place": _CH}
        if excerpt:
            meta["tool_digest"] = [{"tool": "mattermost_skill_get_channel_messages",
                                    "messages": [{"text": "이번주까지 받기로", "author": "김과장"}]}]
        return {"id": "11111111-1111-1111-1111-111111111111", "conversation_id": "c",
                "query": "앞으로 이 채널에선 무조건 표로 줘", "answer": "네", "metadata": meta}

    def _drafts(self, source_ids):
        from agent_arch.services.memory_v2.extractor import build_drafts
        from agent_arch.services.memory_v2.prompts import format_extraction_input

        row = self._row()
        _, id_map = format_extraction_input([row], [], last_scene_name=None)
        data = {"atoms": [{"atom_type": "instruction", "content": "사용자는 표로 받기를 원한다",
                           "priority": -1, "source_ids": source_ids}]}
        return build_drafts(data, id_map, None)

    def test_excerpt_is_citable_as_its_own_evidence(self):
        from agent_arch.services.memory_v2.prompts import format_extraction_input

        text, id_map = format_extraction_input([self._row()], [], last_scene_name=None)
        assert "m1" in id_map and "m1x" in id_map           # 발췌가 별개 근거로 인용 가능
        assert id_map["m1x"] == id_map["m1"]                # 같은 L0 를 가리킨다
        assert "[m1x] 다른 사람들의 대화" in text

    def test_user_own_instruction_in_a_thread_stays_promotable(self):
        d = self._drafts(["m1"])[0]
        assert d.origin == P.ORIGIN_USER
        from agent_arch.services.memory_v2.models import DedupDecision
        from agent_arch.services.memory_v2.writer import resolve_priority
        assert resolve_priority(d, DedupDecision(index=0, action="store"), [], "store") == -1

    def test_rule_taken_from_someone_elses_words_cannot_be_promoted(self):
        d = self._drafts(["m1x"])[0]
        assert d.origin == P.ORIGIN_MIXED
        from agent_arch.services.memory_v2.models import DedupDecision
        from agent_arch.services.memory_v2.writer import resolve_priority
        assert resolve_priority(d, DedupDecision(index=0, action="store"), [], "store") == 80

    def test_turn_without_an_excerpt_has_no_excerpt_tag(self):
        from agent_arch.services.memory_v2.prompts import format_extraction_input

        _, id_map = format_extraction_input([self._row(excerpt=False)], [], last_scene_name=None)
        assert list(id_map) == ["m1"]


def test_user_facing_export_hides_internal_grading_keys():
    """내 기억 화면에 `mattermost:6uwx…` 같은 불투명한 키가 뜨면 뜻 없는 글자만 늘어난다."""
    from agent_arch.services.memory_v2.admin import _public_payload

    assert _public_payload({"aliases": ["개발팀"], P.PAYLOAD_PLACES: ["mattermost:ch-1"],
                            P.PAYLOAD_ORIGIN: P.ORIGIN_MIXED}) == {"aliases": ["개발팀"]}
    assert _public_payload(None) == {}


def test_invented_excerpt_tag_does_not_downgrade_a_user_rule():
    """모델이 없는 발췌 태그를 지어내도 본인 지시가 남의 말로 강등되면 안 된다."""
    from agent_arch.services.memory_v2.extractor import _cites_excerpt

    id_map = {"m1": "u-1", "m1x": "u-1"}
    assert _cites_excerpt({"source_ids": ["m1x"]}, id_map) is True
    assert _cites_excerpt({"source_ids": ["m2x"]}, id_map) is False   # 없는 태그
    assert _cites_excerpt({"source_ids": ["m1"]}, id_map) is False
    assert _cites_excerpt({"source_ids": "m1x"}, id_map) is True      # 문자열 하나로 와도


class TestAppSurfacePlace:
    """앱 전용 창(MES 앱 채팅·음성) — 채널은 아니지만 일반 채팅과 분리된 자리."""

    def test_surface_makes_a_place(self):
        assert P.surface_place("mes") == {"provider": "agent", "channel_id": "mes", "is_dm": False}
        assert P.key_of(P.surface_place("mes")) == "agent:mes"

    def test_blank_surface_is_no_place(self):
        """값이 없으면 장소 밖 — 패널 턴과 같다(중립)."""
        assert P.surface_place(None) is None
        assert P.surface_place("  ") is None

    def test_current_key_prefers_surface_then_origin(self):
        assert P.current_key({"memory_surface": "mes"}) == "agent:mes"
        assert P.current_key({"origin": _CH}) == "mattermost:ch-1"
        assert P.current_key({}) is None

    def test_mes_rule_is_injected_only_in_that_window(self):
        """MES 창에서 못 박은 규칙은 그 창에서만 상시 주입된다 — 패널로 따라오지 않는다."""
        atom = {"payload": {P.PAYLOAD_PLACES: ["agent:mes"]}}

        assert P.eligible(atom, "agent:mes") is True
        assert P.eligible(atom, None) is False                    # 브라우저 패널
        assert P.eligible(atom, "mattermost:ch-1") is False       # 메신저 채널

    def test_panel_rule_still_applies_inside_mes(self):
        """장소 없는 기억(패널·DM)은 중립 — 그 사람 자체의 맥락이라 MES 창에도 적용된다."""
        atom = {"payload": {}}

        assert P.eligible(atom, "agent:mes") is True
        assert P.rank_factor(atom, "agent:mes") == P.FACTOR_NEUTRAL

    def test_rank_prefers_same_window(self):
        atom = {"payload": {P.PAYLOAD_PLACES: ["agent:mes"]}}

        assert P.rank_factor(atom, "agent:mes") == P.FACTOR_SAME
        assert P.rank_factor(atom, None) == P.FACTOR_NEUTRAL      # 장소 밖에선 깎지 않는다
        assert P.rank_factor(atom, "mattermost:ch-1") == P.FACTOR_OTHER
