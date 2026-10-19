"""웹메일 스팸 신고 안내 — 고정 사실이 프롬프트에 살아 있는지 검증.

회귀 (실측 2026-08-10~11, "웹메일 스팸 신고 기능 알려주세요"): 가이드 KB 는 guide 모드
전용이라 일반 에이전트(ReAct) 경로엔 근거 소스가 없고, mail 도구 5종에도 스팸 기능이
없다. 모델이 일반 웹메일 지식으로 "[스팸 신고]"·"[스팸 해제]" 버튼과 "동일 발신자
자동 분류" 같은 존재하지 않는 UI·동작을 지어냈다(LangSmith 피드백 score 0.0:
"DMP 웹메일에 스팸 신고 기능이 있는데 답변을 못함"). 실제 UI 는 메일 미리보기 화면
상단의 "스팸 편지함"(신고) / "정상"(해제) 버튼이다.

프롬프트에 박은 고정 사실이라 도구 호출 없이도 성립해야 하고, 리팩터링 중 조용히
사라지면 곧바로 같은 환각으로 되돌아간다. 그래서 문자열 계약으로 잠근다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.prompts import system as sys_mod

_SECTION = "### 웹메일 스팸 신고 안내"


@pytest.fixture
def production_prompt() -> str:
    """실사용 경로(default 모드)에서 조립되는 시스템 프롬프트."""
    return sys_mod.build_system_prompt(mode="default")


@pytest.fixture
def section(production_prompt: str) -> str:
    """스팸 신고 안내 섹션 본문만 잘라낸다(다음 섹션 헤더 전까지)."""
    assert _SECTION in production_prompt
    return production_prompt.split(_SECTION)[1].split("### 사내 메신저 앱 설치 안내")[0]


def test_report_procedure_is_present(section: str):
    """신고 절차의 핵심 사실 — 미리보기 상단 '스팸 편지함' 버튼."""
    assert "스팸 편지함** 버튼" in section
    assert "미리보기 화면 상단" in section


def test_report_success_message_is_present(section: str):
    """완료 판정 기준인 성공 문구가 그대로 있어야 한다."""
    assert "스팸 메일로 성공적으로 보고함" in section


def test_reported_mail_destination_is_present(section: str):
    """신고된 메일이 스팸 편지함으로 이동한다는 사실이 있어야 한다."""
    assert "스팸 편지함으로 이동" in section


def test_unreport_procedure_is_present(section: str):
    """해제 절차의 핵심 사실 — 스팸 편지함에서 '정상' 버튼, 받은 편지함 복귀 + 학습 취소."""
    assert "정상** 버튼" in section
    assert "학습도 취소" in section


def test_hallucinated_button_names_are_forbidden(section: str):
    """가짜 버튼명이 금지 조항에 명시돼야 한다.

    [스팸 신고]/[스팸 해제]는 회귀 답변(2026-08-10)의 환각, [스팸 처리]는 안내문
    초안에 들어갔다가 실제 UI 스크린샷 대조로 오기로 판명된 이름(2026-08-11).
    """
    assert "금지" in section
    assert "[스팸 신고]" in section
    assert "[스팸 해제]" in section
    assert "[스팸 처리]" in section


def test_spam_learning_fact_is_present(section: str):
    """서버 학습 사실이 있어야 한다.

    최초 블록(2026-08-11)에서는 미확인 후속 동작으로 '금지'했으나, 실제 안내문·UI
    확인으로 사실로 확정돼 뒤집었다 — 신고하면 서버가 학습해 비슷한 메일을 거른다.
    """
    assert "학습" in section
    assert "자동으로 걸러" in section


def test_scope_facts_are_present(section: str):
    """적용 범위 — 본인 메일함 한정, 공유 수신 메일은 각자 신고."""
    assert "본인 메일함에만 적용" in section
    assert "각자 신고" in section


def test_non_webmail_client_guidance_is_present(section: str):
    """Outlook·모바일 앱은 버튼 없이 스팸 편지함 폴더 이동으로 동일 학습."""
    assert "폴더로 옮기면" in section


def test_filter_recommendation_is_present(section: str):
    """스팸 아닌 불필요 메일은 신고 대신 메일 필터 분류 권장."""
    assert "메일 필터" in section


def test_guidance_survives_tool_section_render(production_prompt: str):
    """도구 섹션 자동 렌더(<<...>>) 치환 뒤에도 섹션이 남아 있어야 한다."""
    assert _SECTION in production_prompt
    assert "<<" not in production_prompt
