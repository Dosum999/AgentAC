"""이름 가드의 **근거 범위** 계약 — 시스템 프롬프트가 이름을 화이트리스트하면 안 된다.

실측 사고(2026-09-03 확인): `_evidence_text` 가 `messages[0]`(시스템 프롬프트)까지 근거로
넣는데, 프롬프트의 이름 규칙 조항 반례에 실명이 문자열로 박혀 있었다. 그래서 8/31 사고를
낸 바로 그 이름은 **영구히 환각으로 판정될 수 없었다.** 오탐을 줄이려 근거를 넓게 잡은
설계가 정확히 역효과를 냈다.

수정 방향은 "프롬프트를 통째로 뺀다"가 아니다. 요청자 실명은 `[요청자 정보]` 블록에만
있어서, 통째로 빼면 **사용자 본인 이름이 환각으로 오탐된다.** 그 블록만 되살린다.

이 파일이 못박는 계약은 셋이다.
    1. 프롬프트 본문의 이름은 근거가 아니다 (화이트리스트 금지)
    2. `[요청자 정보]` 의 이름은 근거다 (오탐 금지)
    3. 압축 요약은 role=system 이지만 지난 대화 내용이므로 근거다
"""
from __future__ import annotations

from agent_arch.agents.graphs.react_graph import _evidence_text
from agent_arch.agents.hitl.answer_guard import find_unsourced_person_names, requester_context
from agent_arch.agents.prompts.system import build_system_prompt
from agent_arch.llm.base import Message
from agent_arch.memory.conversation import COMPACT_SUMMARY_MARKER


_TOOL_RESULT = '{"posts": [{"author": "@jdoe", "message": "확인 부탁드립니다"}]}'


def _state(messages):
    return {"messages": messages}


# ── 1. 프롬프트 본문은 근거가 아니다 ────────────────────────────────────────

def test_system_prompt_body_is_not_evidence():
    """프롬프트에 어떤 이름이 예시로 적혀 있든 근거가 되면 안 된다.

    반례에 실명을 쓰지 않도록 프롬프트도 고쳤지만, 그건 심층방어다. 근거 범위 자체가
    막혀 있어야 다음에 누가 반례에 실명을 넣어도 사고가 안 난다.
    """
    prompt = build_system_prompt(mode="default")
    state = _state([
        Message(role="system", content=prompt),
        Message(role="user", content="메신저에서 답변 안 한 사람 알려줘"),
        Message(role="tool", content=_TOOL_RESULT, name="mattermost_skill_get_unread_messages"),
    ])
    evidence = _evidence_text(state)

    # 프롬프트 본문의 특징적인 문구가 근거에서 빠져 있어야 한다.
    assert "영문 ID를 발음대로" not in evidence
    # 도구 결과와 사용자 질문은 그대로 남아 있어야 한다.
    assert "@jdoe" in evidence
    assert "답변 안 한 사람" in evidence


def test_name_in_prompt_does_not_whitelist_itself():
    """프롬프트에 실명이 박혀 있어도 그 이름은 여전히 환각으로 잡혀야 한다.

    회귀 재현: 프롬프트에 "김민수" 를 직접 넣고, 답변이 그 이름을 쓰게 한다.
    근거 범위 수정 전에는 `[]` 가 나왔다(= 못 잡음).
    """
    poisoned = build_system_prompt(mode="default") + '\n\n예시: `jdoe` → "김민수" 금지'
    state = _state([
        Message(role="system", content=poisoned),
        Message(role="tool", content=_TOOL_RESULT, name="mattermost_skill_get_unread_messages"),
    ])
    found = find_unsourced_person_names("김민수 님이 확인을 요청했습니다.", _evidence_text(state))
    assert found == ["김민수"]


# ── 2. 요청자 이름은 근거다 (오탐 방지 — 이 수정의 진짜 계약) ───────────────

def test_requester_name_is_not_flagged():
    """요청자 실명은 시스템 프롬프트에만 있다. 통째로 빼면 본인 이름이 오탐된다."""
    prompt = build_system_prompt(mode="default", user_info={"name": "백지선"})
    state = _state([
        Message(role="system", content=prompt),
        Message(role="tool", content=_TOOL_RESULT, name="mattermost_skill_get_unread_messages"),
    ])
    found = find_unsourced_person_names("백지선 님이 요청한 내용입니다.", _evidence_text(state))
    assert found == []


def test_requester_context_extracts_only_the_block():
    """`[요청자 정보]` 블록만 잘라야 한다 — 다음 섹션이 딸려오면 화이트리스트가 넓어진다."""
    prompt = build_system_prompt(mode="default", user_info={"name": "백지선"})
    block = requester_context(prompt)
    assert block.startswith("[요청자 정보]")
    assert "백지선" in block
    assert "\n\n" not in block


def test_requester_context_is_empty_without_user_info():
    """사용자 정보가 없으면 빈 문자열 — 근거가 넓어지지 않는다."""
    assert requester_context(build_system_prompt(mode="default")) == ""


# ── 3. 압축 요약은 근거다 ───────────────────────────────────────────────────

def test_compact_summary_is_evidence():
    """압축 요약은 role=system 이지만 **지난 대화 내용 그 자체**라 정당한 근거다."""
    state = _state([
        Message(role="system", content=build_system_prompt(mode="default")),
        Message(
            role="system",
            name=COMPACT_SUMMARY_MARKER,
            content="지난 대화 요약: 김민서 님이 원스글로벌 서버 건을 공유했다.",
        ),
        Message(role="tool", content=_TOOL_RESULT, name="mattermost_skill_get_unread_messages"),
    ])
    found = find_unsourced_person_names("김민서 님이 공유한 내용입니다.", _evidence_text(state))
    assert found == []
