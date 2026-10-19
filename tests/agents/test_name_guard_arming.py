"""가드 **무장 조건** — 사고를 낸 그 입력 형태에서 실제로 발동하는가.

가드는 위험한 턴에만 무장한다(스트림을 붙들고 답변을 검증한다). 그런데 무장 조건이
``@handle`` 문자열 하나였다. 신고된 화면의 도구 결과는 맨 ``{"author": "jdoe"}`` 였고
``@`` 가 어디에도 없었다 — **가드가 아예 안 돌았다.**

실측이 이를 뒷받침한다(2026-09-03, stg DB): 30일간 메신저 도구를 쓴 턴이 791건인데
``이름미확인`` 마스킹은 **생애 통틀어 1건**이었다. 기계는 동작하는데 거의 무장하지 않았다.

그래서 무장 신호를 셋으로 넓혔다. 이 파일은 그 셋이 실제 MCP 출력 형태에서 켜지는지,
그리고 **켜지면 안 되는 것**(이메일·내부 ID·영문 채널명)에서 안 켜지는지를 못박는다.
오탐은 정확성이 아니라 체감 지연을 해치므로, 양쪽 다 계약이다.
"""
from __future__ import annotations

import json

import pytest

from agent_arch.agents.hitl.answer_guard import (
    extract_id_tokens,
    find_unsourced_person_names,
    has_unnamed_author,
)


# ── 무장해야 하는 것 ────────────────────────────────────────────────────────

def test_bare_username_arms_guard():
    """신고된 화면의 정확한 형태 — 맨 username, `@` 없음. 예전엔 무장조차 안 했다."""
    evidence = '{"posts": [{"author": "jdoe", "message": "확인 부탁드립니다"}]}'
    assert has_unnamed_author(evidence)
    assert "jdoe" in extract_id_tokens(evidence)


def test_group_dm_label_arms_guard():
    """그룹 DM 제목이 username 나열인 경우 — 콤마로 쪼개 각 조각을 잡아야 한다."""
    evidence = '{"channel_name": "gnoopy, hyunho_lim, miim, yeonju"}'
    assert has_unnamed_author(evidence)
    assert {"gnoopy", "hyunho_lim", "miim", "yeonju"} <= extract_id_tokens(evidence)


def test_handle_still_arms_guard():
    """기존 신호(@handle)는 그대로 동작해야 한다 — 넓히면서 잃으면 안 된다."""
    assert has_unnamed_author('{"author": "@jdoe"}')
    assert extract_id_tokens('{"author": "@jdoe"}') == {"jdoe"}


def test_korean_name_with_parenthetical_id_arms_guard():
    """``김민수(jdoe)`` — 모델이 이름↔ID 결속을 스스로 주장한 형태.

    JSON 키가 없는 supervisor 합성 경로(워커 산문만 근거)에서도 이게 걸려야 한다.
    """
    assert has_unnamed_author("김민수(jdoe) 님이 공유했습니다.")
    assert "jdoe" in extract_id_tokens("김민수(jdoe) 님이 공유했습니다.")


def test_real_mcp_shape_arms_guard():
    """실제 MCP 출력 형태 그대로 — 표시이름과 핸들이 같은 행에 있는 경우."""
    payload = json.dumps({
        "results": [
            {"author": "김민서", "author_handle": "@jdoe", "channel_name": "watchdog"},
            {"author": "@parkjungbin", "author_handle": "@parkjungbin"},
        ]
    }, ensure_ascii=False)
    assert has_unnamed_author(payload)


# ── 무장하면 안 되는 것 (오탐 = 체감 지연) ──────────────────────────────────

def test_email_does_not_arm_guard():
    """이메일의 ``@example.com`` 은 핸들이 아니다 — 예전엔 이것만 있어도 무장했다."""
    evidence = '{"author": "김민서", "email": "jdoe@example.com"}'
    assert not has_unnamed_author(evidence)


def test_mattermost_internal_id_does_not_arm_guard():
    """26자 내부 ID 는 사람 이름으로 안 읽힌다 — 음차 재료가 아니다."""
    evidence = '{"author": "김민서", "channel_id": "s4pib15by7dx9pfzgcjk3g5gia"}'
    assert not has_unnamed_author(evidence)


def test_english_channel_name_alone_does_not_arm_guard():
    """영문 채널명 하나로 무장하면 거의 모든 메신저 턴이 답변 전체를 버퍼링한다.

    ``channel_name`` 은 사람 나열(그룹 DM 라벨)일 수도, 그냥 영문 채널명일 수도 있다.
    누출의 형태는 **콤마 나열**이므로 그 모양일 때만 신호로 친다. 오탐 비용은 정확성이
    아니라 체감 지연이지만, 그게 커지면 가드를 끄고 싶어진다.
    """
    evidence = '{"channel_name": "watchdog", "author": "김민서"}'
    assert not has_unnamed_author(evidence)


def test_channel_name_still_arms_when_author_is_bare():
    """채널명이 영문이어도 작성자가 맨 ID 면 당연히 무장한다."""
    assert has_unnamed_author('{"channel_name": "watchdog", "author": "jdoe"}')


def test_group_dm_with_real_names_does_not_arm_guard():
    """MCP 수정 후의 그룹 DM 라벨(실명 나열)은 음차 재료가 아니다."""
    assert not has_unnamed_author('{"channel_name": "구정현, 홍길동, 이영희"}')


def test_plain_korean_answer_does_not_arm_guard():
    """사람이 전부 실명으로 나오는 턴은 음차할 재료가 없다."""
    evidence = '{"posts": [{"author": "김민서", "message": "확인 부탁드립니다"}]}'
    assert not has_unnamed_author(evidence)


def test_empty_inputs_are_safe():
    assert not has_unnamed_author("")
    assert not has_unnamed_author(None)  # type: ignore[arg-type]
    assert extract_id_tokens("") == set()


# ── 무장 + 검출이 함께 도는지 (통합) ────────────────────────────────────────

@pytest.mark.parametrize("evidence", [
    '{"posts": [{"author": "jdoe"}]}',
    '{"posts": [{"author": "@jdoe"}]}',
    '{"channel_name": "gnoopy, hyunho_lim, miim"}',
])
def test_armed_turn_catches_fabricated_name(evidence):
    """무장한 턴에서 근거에 없는 이름이 실제로 잡혀야 한다.

    무장 조건만 넓히고 검출이 안 따라오면 "붙들었는데 못 잡는" 턴이 생긴다.
    """
    assert has_unnamed_author(evidence)
    assert find_unsourced_person_names("김민수 님이 확인을 요청했습니다.", evidence) == ["김민수"]
