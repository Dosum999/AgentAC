"""L1 추출 — 파싱·정규화·증거 규칙의 코드 방어(무근거 persona, SoR, 봇 자칭)."""
from __future__ import annotations

import json
from datetime import datetime

import pytest

from agent_arch.services.memory_v2 import extractor
from agent_arch.services.memory_v2.extractor import build_drafts, extract_atoms, message_timestamps

M1 = "00000000-0000-0000-0000-000000000001"
M2 = "00000000-0000-0000-0000-000000000002"
ID_MAP = {"m1": M1, "m2": M2}


def _msgs():
    return [
        {"id": M1, "conversation_id": "c", "query": "앞으로 답변에 표 쓰지 마", "answer": "네", "metadata": {}, "created_at": datetime(2026, 9, 4, 10, 0)},
        {"id": M2, "conversation_id": "c", "query": "watchdog 채널 요약해줘", "answer": "요약…", "metadata": {}, "created_at": datetime(2026, 9, 4, 10, 5)},
    ]


def test_build_drafts_happy_path_maps_tags_and_types():
    data = {"scene_name": "형식 규칙", "atoms": [
        {"atom_type": "instruction", "content": "사용자는 답변에 표를 쓰지 않기를 원한다", "priority": -1, "source_ids": ["m1"]},
        {"atom_type": "preference", "content": "사용자는 간결한 답을 선호한다", "priority": 60, "source_ids": ["m1", "bogus"]},
        {"atom_type": "alias", "content": "watchdog 은 ACME Heros 채널", "priority": 70, "source_ids": ["m2"],
         "payload": {"aliases": ["watchdog"], "canonical_name": "ACME Heros", "canonical_kind": "mattermost_channel", "canonical_id": "ch1"}},
        {"atom_type": "episodic", "content": "2026-09-04 watchdog 채널 요약을 요청했다", "priority": 40, "source_ids": ["m2"]},
    ]}
    drafts = build_drafts(data, ID_MAP, "형식 규칙")
    assert [d.atom_type for d in drafts] == ["instruction", "persona", "alias", "episodic"]
    assert drafts[0].priority == -1 and drafts[0].source_message_ids == [M1]
    assert drafts[1].source_message_ids == [M1]                 # 모르는 태그는 버림
    assert drafts[2].payload["canonical_id"] == "ch1" and drafts[2].payload["aliases"] == ["watchdog"]
    assert all(d.scene_name == "형식 규칙" for d in drafts)


def test_build_drafts_drops_rule_violations():
    data = {"atoms": [
        {"atom_type": "persona", "content": "사용자는 녹차를 선호한다", "priority": 50},                 # 무근거 persona
        {"atom_type": "instruction", "content": "표 금지", "priority": 90, "source_ids": ["zzz"]},       # 근거 태그 무효
        {"atom_type": "persona", "content": "사용자의 연차는 12일 남았다", "source_ids": ["m1"]},         # SoR
        {"atom_type": "alias", "content": "사용자는 자신을 에이전트라고 부른다", "source_ids": ["m1"],
         "payload": {"aliases": ["에이전트"], "canonical_name": "신희섬"}},                              # 봇 자칭
        {"atom_type": "alias", "content": "별칭인데 payload 없음", "source_ids": ["m1"]},               # payload 불완전
        {"atom_type": "unknown_type", "content": "x", "source_ids": ["m1"]},
        {"atom_type": "episodic", "content": "   ", "source_ids": ["m1"]},
        "not a dict",
    ]}
    assert build_drafts(data, ID_MAP, None) == []


def test_sor_filter_keeps_episodic_acts_but_drops_values():
    from agent_arch.services.memory_v2.extractor import looks_like_sor
    assert looks_like_sor("사용자의 연차는 12일 남았다", "persona")
    assert looks_like_sor("사용자는 연차를 자주 확인한다", "persona")            # 상태형은 마커만으로 드롭
    assert not looks_like_sor("사용자는 2026-09-04 잔여 연차 현황을 확인해 개발팀 채널에 공유하도록 요청했다", "episodic")
    assert looks_like_sor("사용자는 2026-09-04 잔여 연차가 12일임을 확인했다", "episodic")   # 값이 박히면 드롭
    assert not looks_like_sor("사용자는 메일을 확인했다", "episodic")
    assert not looks_like_sor("사용자는 9월 4일 14:30에 휴가 신청 현황을 물었다", "episodic")   # 날짜·시각은 값이 아니다


def test_build_drafts_skips_persona_from_explicit_save_turns():
    data = {"atoms": [
        {"atom_type": "instruction", "content": "사용자는 표를 쓰지 않기를 원한다", "priority": 90, "source_ids": ["m1"]},
        {"atom_type": "episodic", "content": "2026-09-07 표 지시를 저장했다", "priority": 30, "source_ids": ["m1"]},
        {"atom_type": "instruction", "content": "존댓말로 답한다", "priority": 80, "source_ids": ["m1", "m2"]},
    ]}
    drafts = build_drafts(data, ID_MAP, None, explicit_message_ids={M1})
    assert [d.atom_type for d in drafts] == ["episodic", "instruction"]      # m1 만 근거인 지시는 이미 memory_remember 로 저장됨
    assert drafts[1].content == "존댓말로 답한다"


@pytest.mark.asyncio
async def test_extract_atoms_detects_explicit_save_turns(monkeypatch):
    msgs = _msgs()
    msgs[0]["tool_calls"] = [{"name": "memory_remember", "arguments": {"content": "표 금지"}}]
    raw = json.dumps({"scene_name": "형식", "atoms": [
        {"atom_type": "instruction", "content": "사용자는 표를 쓰지 않기를 원한다", "priority": 90, "source_ids": ["m1"]}]}, ensure_ascii=False)

    async def fake_invoke(messages):
        return raw, "m", 1

    monkeypatch.setattr(extractor, "invoke_text", fake_invoke)
    res = await extract_atoms(msgs, [], last_scene_name=None)
    assert res.drafts == [] and res.error is None


def test_build_drafts_accepts_bare_array():
    arr = [{"type": "episodic", "content": "2026-09-04 요약 요청", "source_ids": ["m2"]}]
    drafts = build_drafts(arr, ID_MAP, None)
    assert len(drafts) == 1 and drafts[0].atom_type == "episodic"


@pytest.mark.asyncio
async def test_extract_atoms_parses_object_response(monkeypatch):
    raw = json.dumps({"scene_name": "형식 규칙", "atoms": [
        {"atom_type": "instruction", "content": "사용자는 표를 쓰지 않기를 원한다", "priority": -1, "source_ids": ["m1"]}]}, ensure_ascii=False)

    async def fake_invoke(messages):
        assert messages[0].role == "system" and "[신규 대화]" in messages[1].content
        return raw, "fake-model", 7

    monkeypatch.setattr(extractor, "invoke_text", fake_invoke)
    res = await extract_atoms(_msgs(), [], last_scene_name=None, known_aliases=[])
    assert res.error is None and res.scene_name == "형식 규칙" and res.model == "fake-model"
    assert len(res.drafts) == 1 and res.input_message_ids == [M1, M2]
    assert len(res.prompt_sha) == 16


@pytest.mark.asyncio
async def test_extract_atoms_accepts_array_only_response(monkeypatch):
    async def fake_invoke(messages):
        return '[{"atom_type":"episodic","content":"2026-09-04 요약 요청","source_ids":["m2"]}]', "m", 1

    monkeypatch.setattr(extractor, "invoke_text", fake_invoke)
    res = await extract_atoms(_msgs(), [], last_scene_name=None)
    assert res.error is None and res.scene_name is None and len(res.drafts) == 1


@pytest.mark.asyncio
async def test_extract_atoms_parse_failure_keeps_raw(monkeypatch):
    async def fake_invoke(messages):
        return "죄송합니다, 추출할 수 없습니다.", "m", 1

    monkeypatch.setattr(extractor, "invoke_text", fake_invoke)
    res = await extract_atoms(_msgs(), [], last_scene_name=None)
    assert res.error == "parse_failed" and res.raw.startswith("죄송") and res.drafts == []


@pytest.mark.asyncio
async def test_extract_atoms_llm_failure_has_empty_raw(monkeypatch):
    async def fake_invoke(messages):
        raise TimeoutError("vllm down")

    monkeypatch.setattr(extractor, "invoke_text", fake_invoke)
    res = await extract_atoms(_msgs(), [], last_scene_name=None)
    assert res.raw == "" and res.error.startswith("TimeoutError") and res.drafts == []


def test_message_timestamps():
    ts = message_timestamps(_msgs(), [M2, "nope"])
    assert ts == [datetime(2026, 9, 4, 10, 5)]


@pytest.mark.asyncio
async def test_excerpt_rule_rides_along_only_when_the_batch_has_one(monkeypatch):
    """발췌 없는 배치는 프롬프트도 버전 해시도 이전과 같다 — 패널 경로를 건드리지 않는 지점."""
    seen: list[str] = []

    async def fake_invoke(messages):
        seen.append(messages[0].content)
        return json.dumps({"scene_name": "s", "atoms": []}, ensure_ascii=False), "m", 1

    monkeypatch.setattr(extractor, "invoke_text", fake_invoke)

    plain = await extract_atoms(_msgs(), [], last_scene_name=None, known_aliases=[])

    with_excerpt = _msgs()
    with_excerpt[1]["metadata"] = {"tool_digest": [{
        "tool": "mattermost_skill_get_channel_messages",
        "messages": [{"text": "제가 정리해서 올릴게요", "author": "이대리"}]}]}
    withx = await extract_atoms(with_excerpt, [], last_scene_name=None, known_aliases=[])

    assert "남의 말" not in seen[0] and "남의 말" in seen[1]
    assert plain.prompt_sha != withx.prompt_sha
