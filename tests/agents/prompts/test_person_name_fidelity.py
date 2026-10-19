"""사람 이름은 도구가 준 값 그대로 — 영문 ID로 한글 실명을 지어내지 않는다.

메신저 조회 답변에서 이름이 가끔 실제와 어긋났다. 도구(`mattermost_skill_get_unread_messages`)가
작성자를 영문 username(`jdoe`·`bcha`)으로만 내보내는데 답변은 한국어라, 모델이 영문 ID를
한글 실명으로 **역추정**한 것이다. `jdoe→김민수`는 우연히 맞고 `bcha→차병준`은 틀린다.

도구 쪽은 mattermost-mcp 에서 실명을 내도록 고쳤고(`User.display` 로 일원화), 이 프롬프트
조항은 **실명을 못 구한 계정**에 대해 모델이 빈칸을 상상으로 채우지 못하게 막는 두 번째 방어선이다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.prompts import system as sys_mod


@pytest.fixture
def production_prompt() -> str:
    return sys_mod.build_system_prompt(mode="default")


@pytest.fixture
def block(production_prompt: str) -> str:
    """규칙 본문만, 줄바꿈을 접어서 — 문구를 지키자는 것이지 줄바꿈이 아니다."""
    assert "사람 이름은 도구가 준 값 그대로" in production_prompt
    tail = production_prompt.split("사람 이름은 도구가 준 값 그대로")[1]
    return " ".join(tail.split("- **API 엔드포인트")[0].split())


def test_english_id_is_kept_as_is(block: str):
    assert "영문 ID뿐이면 그 값을 그대로" in block


def test_transliteration_to_korean_is_banned(block: str):
    # 이번 버그의 정확한 형태 — 발음으로 한글 이름을 만드는 것.
    assert "발음대로 한글 이름으로 옮겨 적지 마세요" in block


def test_inventing_parenthetical_id_is_banned(block: str):
    assert '"한글이름(jdoe)" 형태 금지' in block


def test_cross_tool_pairing_is_banned(block: str):
    """지식검색(한글 이름) ↔ 안읽은메시지(영문 ID)를 짝짓지 말라는 조항."""
    assert "같은 도구 결과의 같은 항목" in block
    assert "짝지어 추측하지 마세요" in block


def test_username_still_allowed_as_readable_name(production_prompt: str):
    """실명 미확보 계정은 username 이 유일한 이름 — 허용은 유지돼야 한다."""
    assert "display_name, username, team_name 등" in production_prompt


# ── 반례에 실명을 쓰지 않는다 ────────────────────────────────────────────────

# 사고 이력에 등장한 실명·오답 이름. 프롬프트에 이 중 무엇이든 들어가면 안 된다.
_INCIDENT_NAMES = ("김민수", "김민서", "박지운", "박지훈", "홍길동", "차병준", "정민김")


@pytest.mark.parametrize("name", _INCIDENT_NAMES)
def test_prompt_contains_no_incident_name(production_prompt: str, name: str):
    """프롬프트의 ✗ 예시는 구조적으로 **화이트리스트 항목**이다.

    이름 가드는 "답변의 이름이 이 턴 입력 어디에도 없으면 환각"으로 판정한다. 근거에
    시스템 프롬프트가 섞이면 프롬프트에 적힌 이름은 영원히 무죄가 된다 — 실제로 8/31
    사고를 낸 `김민수` 가 반례로 박혀 있어 그 이름만 탐지 불가였다(2026-09-03 확인).

    근거 범위 자체는 `_evidence_text` 에서 막았지만(`test_name_guard_evidence_scope.py`),
    반례에 실명을 두면 **모델이 그 이름을 실제로 출력하는** 2차 위험이 남는다. 그래서
    가짜 이름조차 쓰지 않고 `한글이름` 같은 자리표시자를 쓴다.
    """
    assert name not in production_prompt
