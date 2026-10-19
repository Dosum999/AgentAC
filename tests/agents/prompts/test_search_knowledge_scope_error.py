"""범위 오류(`scope_error`)는 빈 결과가 아니다 — 넓히지 말고 되물어야 한다.

`mattermost_search_knowledge` 는 팀·채널 이름을 해석하지 못하면 **검색을 아예 하지 않고**
`scope_error` + 후보를 돌려준다(fail-closed). 이때 기존 "빈 결과 → 기간 7일 확장 →
find_channels" 사다리를 타면 **범위가 틀린 채로 넓히는** 셈이라, 2026-08-26 사고("CustomerA
개발팀 견적" → 다른 회사 채널 답변)를 그대로 재현한다.

문구 존재만이 아니라 **순서**까지 잠근다 — 범위 오류 규칙이 사다리 아래로 밀리면
모델이 사다리를 먼저 읽고 확장해 버린다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.prompts import system as sys_mod

_SECTION = "### 사내 지식 검색 (반드시 준수)"
_NEXT = "### 사내 데이터성 수치 질의"


@pytest.fixture
def production_prompt() -> str:
    return sys_mod.build_system_prompt(mode="default")


@pytest.fixture
def section(production_prompt: str) -> str:
    assert _SECTION in production_prompt
    return production_prompt.split(_SECTION)[1].split(_NEXT)[0]


@pytest.fixture
def scope_block(section: str) -> str:
    """범위 오류 규칙 본문만(빈 결과 사다리 시작 전까지).

    프롬프트는 가독성 때문에 줄바꿈이 들어가 있어, 문구가 두 줄에 걸치면 단순 substring
    검사가 깨진다. 여기서 공백을 접어 **줄바꿈 위치에 시험이 묶이지 않게** 한다
    (문구를 지키자는 것이지 줄바꿈을 지키자는 게 아니다).
    """
    assert "범위 오류(팀·채널 이름 미해석) 처리" in section
    block = section.split("범위 오류(팀·채널 이름 미해석) 처리")[1].split("빈 결과(`count:0`) 처리")[0]
    return " ".join(block.split())


def test_scope_error_branch_exists(section: str):
    assert "`scope_error`" in section
    assert "범위 오류" in section


def test_scope_error_precedes_widen_ladder(section: str):
    """순서 잠금 — 사다리보다 앞에 있어야 모델이 먼저 읽는다."""
    assert section.index("범위 오류") < section.index("자동 기간 확장")


def test_scope_error_forbids_both_fallbacks(scope_block: str):
    """기간 확장과 채널 재검색 **둘 다** 이름을 대며 금지해야 한다."""
    assert "사다리를 타지 마라" in scope_block
    assert 'period="7d"' in scope_block
    assert "find_channels" in scope_block


def test_scope_error_forbids_dropping_scope_args(scope_block: str):
    """인자를 빼고 재호출하면 그게 바로 '조용한 전체 검색' 이다."""
    assert "team_name_keyword" in scope_block
    assert "둘 다 금지" in scope_block


def test_askback_uses_team_qualified_candidates(scope_block: str):
    """되묻기 후보도 팀명을 함께 적되, 형식 예시에 실제 회사명을 박지 않는다."""
    assert "candidates" in scope_block
    assert "<team_name>/<channel_name>" in scope_block
    assert "되물어라" in scope_block


def test_askback_is_carved_out_of_no_askback_rule(scope_block: str):
    """'되묻기 남발 금지' 규칙과 충돌하지 않도록 예외임을 명시해야 한다."""
    assert "예외" in scope_block


def test_widen_ladder_header_defers(section: str):
    assert "그 규칙이 우선이다" in section


def test_global_empty_rule_defers(production_prompt: str):
    """전역 빈 결과 규칙(§멀티스텝)도 범위 오류엔 양보해야 한다."""
    block = production_prompt.split("빈 검색 결과 처리")[1].split("### 부정 신호 처리")[0]
    assert "scope_error" in block


def test_revenue_ladder_defers(production_prompt: str):
    """실적 90d 확장도 마찬가지 — 회사가 특정된 질문에서 넓히면 남의 회사 실적을 답한다."""
    block = production_prompt.split(_NEXT)[1].split("### 메신저 도구 사용 원칙")[0]
    assert "scope_error" in block
    assert "확장하지 마라" in block


def test_results_are_team_qualified(section: str):
    """동명 채널이 5개라 팀명 없이는 어느 회사인지 구분이 안 된다."""
    assert "<team_name>/<channel_name>" in section
    assert "팀명을 지어내지 마라" in section


def test_team_name_is_read_from_the_hit(section: str):
    """팀명 출처 잠금 — 이제 hit 이 `team_name` 을 싣는다.

    MCP 가 검색 결과에 채널 디렉토리(`agent.channel_directory`)의 팀을 조인해 붙인다
    (재색인 없이). 예전 규칙은 `scope.teams[0]` 을 보게 했는데, `scope` 는 회사·채널
    키워드가 있는 턴에만 계산된다 — 회사명이 없는 질문에서는 팀을 알 방법이 없는데
    "팀명을 생략하지 마라" 만 남아, 모델이 **프롬프트 예시의 회사명을 베껴** 세 채널
    전부에 같은 팀명을 붙였다(실측 2026-09-18 운영).

    그래서 ① 출처를 hit 으로 옮기고 ② 값이 없으면 채널명만 쓰게 하고 ③ 프롬프트에서
    실제 회사명 예시를 걷어냈다(예시가 곧 환각의 재료였다).
    """
    flat = " ".join(section.split())
    assert "그 hit의 `team_name` 값만 쓴다" in flat
    assert "`team_name`이 없는 hit에는 채널명만 쓴다" in flat
    # 다른 hit·요청자 회사·프롬프트 예시를 가져다 쓰지 말라는 금지가 함께 있어야 한다.
    assert "가져다 쓰지 마라" in flat


def test_no_real_company_name_is_used_as_the_format_example(section: str):
    """예시에 실제 회사명을 박으면 그게 곧 환각의 재료가 된다(이름 가드 주석과 같은 교훈)."""
    assert "고객사 A놀로지/개발팀" not in section


def test_scope_ids_are_not_exposed(section: str):
    """scope 에는 team_id·channel_id 도 들어 있다 — ID 금지 규칙이 여기에도 걸려야 한다."""
    flat = " ".join(section.split())
    assert "`team_id`·`channel_id`는 도구 인자용" in flat


def test_drilldown_keeps_team_scope(section: str):
    assert "그대로 유지" in section


def test_channel_id_ban_intact(production_prompt: str):
    """팀명을 노출하게 됐다고 해서 ID 금지가 느슨해지면 안 된다."""
    assert "채널 ID는 어떤 경우에도 사용자에게 노출하지 마라" in production_prompt
    assert "display_name, username, team_name 등" in production_prompt
