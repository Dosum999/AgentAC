"""탐지기 확장 — ``님`` 밖의 형태를 잡되, 오탐은 묶어 둔다.

기존 탐지기는 ``이름 [(id)] [직급] 님`` 만 잡았다. 신고된 화면은 마침 ``김민수(jdoe) 님:``
형태라 그 조항이 이번 건의 차단 원인은 아니었지만, 일반적으로는 큰 사각지대다 —
``김민수(jdoe)가 공유했습니다``·표 셀·목록 선두는 전부 안 보였다.

한글 2~4음절은 어디에나 있어서 "이름 자리에 있으면 이름"으로는 오탐이 폭발한다. 그래서
단서마다 관문을 다르게 건다.

    A. ``… 님``        — 검증된 경로. 관문 없음(손대지 않는다)
    B. ``이름(id)``    — 모델이 결속을 스스로 주장. 괄호 토큰이 근거의 ID 후보집합에 있을 때만
    C. 이름 자리       — 스톱워드 + **성씨 선두** 두 관문을 모두 통과해야 채택

성씨 필터가 재현율을 거의 안 깎는 이유: **음차 환각은 언제나 "성+이름" 형태로 나온다.**
모델이 로마자를 한국 이름으로 되짚기 때문이다.

이 파일의 절반은 오탐 배터리다. 오탐 비용은 정확성이 아니라 **멀쩡한 단어가 `이름미확인`
으로 가려지는 것**이라, 검출력만큼이나 중요한 계약이다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.hitl.answer_guard import (
    find_unsourced_person_names,
    redact_unsourced_names,
)


# 도구가 맨 username 만 준 상황 — 이 턴의 모든 한글 이름은 근거가 없다.
_EVIDENCE = '{"posts": [{"author": "jdoe", "message": "원스글로벌 서버 건 공유"}]}'


# ── 잡아야 하는 형태 ────────────────────────────────────────────────────────

def test_honorific_shape_still_caught():
    """단서 A — 기존 경로는 그대로 동작해야 한다."""
    assert find_unsourced_person_names("김민수 님이 공유했습니다.", _EVIDENCE) == ["김민수"]


def test_name_with_id_without_honorific_is_caught():
    """단서 B — ``김민수(jdoe)가 …``. ``님`` 이 없어 예전엔 안 보였다."""
    answer = "김민수(jdoe)가 원스글로벌 서버 건을 공유했습니다."
    assert find_unsourced_person_names(answer, _EVIDENCE) == ["김민수"]


def test_name_with_unknown_id_is_not_caught():
    """괄호 토큰이 근거의 ID 가 아니면 결속 주장이 아니다 — 단서 B 를 태우지 않는다."""
    answer = "홍보팀(마케팅)이 자료를 공유했습니다."
    assert find_unsourced_person_names(answer, _EVIDENCE) == []


def test_bullet_lead_name_is_caught():
    """단서 C — 목록 선두. 신고 화면이 정확히 이 렌더링이었다."""
    answer = "- 김민수: 8월 28일 원스글로벌 서버 건 진행 상황 공유"
    assert find_unsourced_person_names(answer, _EVIDENCE) == ["김민수"]


def test_bold_bullet_lead_name_is_caught():
    """마크다운 볼드가 끼어도 잡아야 한다 — 실제 답변은 늘 볼드로 나온다."""
    answer = "* **김민수**: 견적 수신 현황 공유"
    assert find_unsourced_person_names(answer, _EVIDENCE) == ["김민수"]


def test_table_cell_name_is_caught():
    answer = "| 채널 | 작성자 |\n|---|---|\n| watchdog | 김민수 |"
    assert find_unsourced_person_names(answer, _EVIDENCE) == ["김민수"]


def test_title_without_honorific_is_caught():
    """``김민수 과장이`` — 직급은 붙었는데 ``님`` 이 없는 형태."""
    assert find_unsourced_person_names("김민수 과장이 공유했습니다.", _EVIDENCE) == ["김민수"]


def test_multiple_shapes_dedupe_in_order():
    answer = "- 김민수: 공유\n박지운(jhpark2) 님이 확인했습니다.\n김민수 과장 참고."
    assert find_unsourced_person_names(answer, _EVIDENCE) == ["박지운", "김민수"]


# ── 잡으면 안 되는 것 (오탐 배터리) ─────────────────────────────────────────

@pytest.mark.parametrize("answer", [
    "- 회의 일정: 8월 28일 오후 5시",
    "- 진행 상황을 공유했습니다.",
    "- 미정: 담당자 배정 대기",
    "- 확인 요청: 연봉 관련 내용",
    "- 공지 사항: 근무 변경 신청 안내",
    "- 첨부 파일: 견적서",
    "| 미정 |",
    "| 상태 | 비고 |",
    "- 결과: 완료",
    "- 오늘 일정 정리",
])
def test_non_person_words_in_name_position_are_not_flagged(answer):
    """목록·표의 첫 칸에 실제로 자주 오는 비인명 — 스톱워드/성씨 관문이 걸러야 한다."""
    assert find_unsourced_person_names(answer, _EVIDENCE) == []


@pytest.mark.parametrize("answer", [
    "- 발주 담당: 확인 필요",          # '발주' 는 성씨가 아니다
    "- 물류 창고 재고 정리",
    "- 서버 상태 점검 결과",
    "- 예산 집행 내역",
])
def test_non_surname_head_in_name_position_is_not_flagged(answer):
    """성씨로 시작하지 않는 한글 어절은 이름 후보가 아니다."""
    assert find_unsourced_person_names(answer, _EVIDENCE) == []


def test_name_present_in_evidence_is_not_flagged():
    """근거에 있는 이름은 어떤 형태로 나와도 무탐지 — 이게 가드의 본질이다."""
    evidence = '{"posts": [{"author": "김민서", "message": "확인 부탁"}]}'
    answer = "- 김민서: 확인 부탁\n김민서(jdoe) 님\n| 김민서 |"
    assert find_unsourced_person_names(answer, evidence) == []


def test_role_words_with_honorific_are_not_flagged():
    """단서 A 의 기존 스톱워드 동작을 깨뜨리지 않았는지."""
    assert find_unsourced_person_names("담당자 님께 문의하세요.", _EVIDENCE) == []


# ── 마스킹이 새 형태를 실제로 덮는가 (동반 수정 계약) ───────────────────────

def test_masking_covers_shapes_without_honorific():
    """탐지만 넓히고 마스킹이 안 따라오면 로그만 남고 사용자는 틀린 이름을 본다.

    예전 마스킹은 ``… 님`` 문맥에서만 치환해서, 단서 B/C 로 잡은 이름은 그대로 나갔다.
    """
    answer = "- 김민수: 공유했습니다.\n김민수(jdoe)가 확인했습니다."
    out = redact_unsourced_names(answer, ["김민수"])
    assert "김민수" not in out
    assert out.count("이름미확인") == 2


def test_masking_keeps_the_handle():
    """괄호 안 핸들은 사실이므로 남는다 — 사용자가 누구인지 추적할 수 있어야 한다."""
    out = redact_unsourced_names("김민수(jdoe) 님이 공유했습니다.", ["김민수"])
    assert out == "이름미확인(jdoe) 님이 공유했습니다."


def test_masking_does_not_touch_substrings_of_other_words():
    """이름이 다른 한글 어절의 일부일 때는 건드리지 않는다."""
    out = redact_unsourced_names("김민수승 이라는 채널", ["김민수"])
    assert out == "김민수승 이라는 채널"


# ── 운영 답변 실측에서 나온 오탐 (2026-09-03, 40건 감사) ────────────────────
# 아래는 전부 **실제 운영 답변에서 이름으로 잡혔던** 문자열이다. 감사 없이 규칙만 넓혔다면
# 멀쩡한 단어가 `이름미확인` 으로 가려졌을 것이다.

@pytest.mark.parametrize("answer,culprit", [
    ("부스 연출 관련 대표님 요청사항을 확인했습니다.", "관련"),
    ("대화 이력 검색 기능 고도화에 대한 이사님의 지시사항입니다.", "대한"),
    ("통화 내용과 차이가 있어 이사님께 다시 확인하겠습니다.", "있어"),
    ("AI팀 소통방 채널에서 이사님께서 공유하셨습니다.", "채널에서"),
    ("백지선님과 신희섬님 간의 대화에서 이사님이 언급되었습니다.", "대화에서"),
    ("*   **채널명**: 에이전트 QA", "채널명"),
    ("*   **문제점**: 홀 치수 미달", "문제점"),
])
def test_real_world_false_positives_stay_clear(answer, culprit):
    """직급이 낀 형태와 목록 선두에서 평범한 단어가 이름으로 잡히던 실측 오탐.

    직급 앞 어절은 성씨 관문을, 조사로 끝나는 어절은 꼬리 관문을 통과해야 한다.
    """
    assert culprit not in find_unsourced_person_names(answer, _EVIDENCE)


def test_title_infix_with_real_name_still_caught():
    """오탐을 막느라 진짜 케이스를 잃지 않았는지 — 직급이 껴도 실명은 잡아야 한다."""
    assert find_unsourced_person_names("김민수 과장님이 확인했습니다.", _EVIDENCE) == ["김민수"]


# ── 운영 답변 실측 오탐 (2026-10-01): 성씨 글자로 시작하는 보통 명사가 목록 머리·표 칸에 왔다 ─────────────
@pytest.mark.parametrize("answer,culprit", [
    ("| 구분 | 차별점 |\n|---|---|\n| 에이전트 | 차별점 |", "차별점"),
    ("| 항목 | 내용 |\n|---|---|\n| 차별점 | 사내 데이터 연동 |", "차별점"),
    ("- 선택: 아래 두 방식 중 하나를 고르세요", "선택"),
    ("* **선택**: 견적 비교표 첨부 여부", "선택"),
    ("- 원하시면 — 바로 이어서 정리해 드립니다", "원하시면"),
])
def test_common_nouns_in_list_or_table_are_not_names(answer, culprit):
    """목록 머리·표 칸은 '사람 자리' 일 때만 이름 후보다 — 줄에 사람이 한 일·호칭이 있거나 표 머리글이 사람 칸."""
    assert culprit not in find_unsourced_person_names(answer, _EVIDENCE)


def test_person_slots_are_still_checked():
    """사람 자리면 지금처럼 잡는다 — 작성자 칸, 공유·요청 같은 행위가 있는 목록 줄."""
    assert find_unsourced_person_names("| 채널 | 담당자 |\n|---|---|\n| 서버 | 김민수 |", _EVIDENCE) == ["김민수"]
    assert find_unsourced_person_names("- 김민수: 견적서 요청드림", _EVIDENCE) == ["김민수"]


@pytest.mark.parametrize("line", [
    "*   **조건:** 필수값(프로젝트명, 담당자, 납기일, 품목 1건 이상)",
    "*   **선택:** 필수값(프로젝트명, 담당자, 납기일)",
    "- **임시저장(초안) 활용**: 보내고 싶은 시점에 확인 후 전송",
])
def test_instructional_list_head_with_noun_cue_is_not_a_person(line):
    """'담당자'·'확인 후' 같은 명사·절차 표현은 안내문에 흔하다 — 사람이 한 일의 동사 꼴만 사람 자리로 본다."""
    assert find_unsourced_person_names(line, "작성자 @jdoe 가 올린 글") == []


@pytest.mark.parametrize("line", [
    "- 김민수: 견적서 요청드림",
    "- 김민수: 회의록 작성함",
    "- 김민수: 서버 이전 일정 확인했습니다",
    "* **김민수**: 견적 수신 현황 공유",
])
def test_person_activity_list_head_still_flagged(line):
    assert find_unsourced_person_names(line, "작성자 @jdoe 가 올린 글") == ["김민수"]
