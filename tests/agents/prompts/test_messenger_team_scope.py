"""메신저 팀 스코프 라우팅 — 회사명은 잡음이 아니라 팀 이름이다.

회귀 (실측 2026-08-26): "CustomerA 개발팀 어제 올라온 견적 조회해줘" 에 에이전트가
**다른 팀**("고객사 A/회사 경영관리 협업") 채널 내용을 답했다. 원인 중 하나가 프롬프트
규칙이었다 — "회사·조직명은 팀도 채널도 아니다 … 팀 인자에 절대 넣지 마라"(daf2fcd).
그 규칙은 별칭 해소가 없던 시절 `mattermost_skill_*` 의 "팀 없음" 오류를 막으려던
우회책인데, 이 메신저는 멀티테넌트라(팀 30여 개, `개발팀` 5개) 회사명을 버리면 어느
회사인지 특정할 근거가 통째로 사라진다.

지금은 `mattermost_search_knowledge` 가 별칭 표로 팀을 해석하므로 회사명을
`team_name_keyword` 로 넘긴다. 다만 **`mattermost_skill_*` 쪽 금지는 그대로 유효하다**
— 그쪽은 여전히 별칭 해석을 거치지 않는다. 이 파일은 그 양쪽을 함께 잠근다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.prompts import system as sys_mod

_SECTION = "### 메신저 도구 사용 원칙"
_NEXT = "### 초안(승인 대기) 안내 원칙"


@pytest.fixture
def production_prompt() -> str:
    return sys_mod.build_system_prompt(mode="default")


@pytest.fixture
def section(production_prompt: str) -> str:
    assert _SECTION in production_prompt
    return production_prompt.split(_SECTION)[1].split(_NEXT)[0]


def test_old_wrong_rule_is_gone(production_prompt: str):
    """회사명을 버리라던 두 문장이 살아 있으면 같은 오답으로 되돌아간다."""
    assert "회사명은 잡음이므로 떼어내고" not in production_prompt
    assert "해당 이름의 팀은 존재하지 않아 조회가 실패한다" not in production_prompt


def test_company_name_is_team_scope_signal(section: str):
    assert "팀 스코프 신호" in section
    assert "멀티테넌트" in section
    assert "team_name_keyword" in section
    assert "channel_keyword" in section


def test_worked_example_present(section: str):
    """모델이 슬롯을 맞추려면 실제 사고 문장이 예시로 있어야 한다."""
    assert 'team_name_keyword="CustomerA"' in section
    assert 'channel_keyword="개발팀"' in section
    assert 'query="견적"' in section
    assert 'period="yesterday"' in section


def test_scope_words_must_not_go_into_query(section: str):
    assert "`query`에 남기지 마라" in section


def test_skill_tools_also_take_team_keyword(section: str):
    """skill 도구도 이제 별칭을 본다(Mattermost-MCP 1d95847) — 회사명을 넘겨야 한다.

    예전엔 반대로 "skill 엔 회사명 절대 넣지 마라" 였다(daf2fcd). 그건 별칭 부재를
    감추려던 우회책이었고, 실제로 stg e2e 에서 모델이 그 예외를 어겨
    `find_channels(team_name_keyword="고객사 A놀로지")` 로 죽었다. 비대칭을 없앤 뒤로는
    회사명을 넘기는 쪽이 맞다 — 빼면 동명 채널 중 엉뚱한 회사 것이 잡힌다.
    """
    flat = " ".join(section.split())
    assert "mattermost_skill_find_channels" in flat
    assert "`team_name_keyword`" in flat
    assert "엉뚱한 회사" in flat


def test_obsolete_skill_ban_is_gone(production_prompt: str):
    """이제 틀린 규칙이라 살아 있으면 모델이 회사명을 버린다."""
    assert "skill 도구에는" not in production_prompt
    assert "회사·조직명을 **절대 넣지 마라**" not in production_prompt


def test_model_must_not_normalize_team_names(section: str):
    """별칭은 DB 소관이다 — 모델이 정식명칭을 추측해 바꾸면 DB 표와 어긋난다."""
    assert "정식명칭을 추측해" in section
    assert "별칭" in section


def test_no_team_name_invention(section: str):
    assert "지어내 채우지 마라" in section


def test_routing_tiebreaker_prefers_knowledge_search(production_prompt: str):
    """회사명이 붙으면 find_channels 로 가면 안 된다 — 팀을 구분하지 못한다."""
    assert "가름말" in production_prompt
    assert "find_channels는 팀을 구분하지 못해" in production_prompt
