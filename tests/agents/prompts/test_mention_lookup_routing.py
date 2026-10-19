"""나를 태그한 메시지 조회는 의미검색이 아니라 전용 도구로 간다.

회귀 (실측 2026-09-02): "지난 일주일간 나 태그한 메시지 알려줘" 에 에이전트가
`AI팀 소통방` 1건만 답했고, "다른 채널에서 언급한 건?" 에는 "확인되지 않습니다" 라고
했다. 실제로는 8개 채널 17건이 있었고, 그중 하나는 이사님이 직접 태그한 글이었다.

원인은 랭킹이 아니라 **원리**다. 색인 본문의 멘션은 Mattermost 원문 그대로 `@jsbaek`
으로 남는데(작성자 이름만 한글 실명으로 변환된다) 질문은 "백지선" 으로 온다. BM25
토큰(`백지`/`지선` vs `jsbaek`)은 교집합이 0이고 임베딩도 둘을 잇지 못한다. 그래서
`mattermost_search_knowledge` 가 실제로 고르는 것은 "그 사람을 태그한 글" 이 아니라
"그 사람이 자주 말하는 채널" 이다.

그래서 프롬프트가 잠가야 할 것은 두 가지다 — (1) 이 질의형은 전용 도구로 보낼 것,
(2) **왜** search_knowledge 로 가면 안 되는지. 이유가 없으면 모델이 곧 되돌아간다.
"""
from __future__ import annotations

import pytest

from agent_arch.agents.prompts import system as sys_mod

_SECTION = "### 메신저 도구 사용 원칙"
_NEXT = "### 초안(승인 대기) 안내 원칙"


@pytest.fixture
def section() -> str:
    prompt = sys_mod.build_system_prompt(mode="default")
    assert _SECTION in prompt
    return prompt.split(_SECTION)[1].split(_NEXT)[0]


def test_mention_query_routes_to_dedicated_tool(section: str):
    assert "mattermost_skill_find_my_mentions" in section
    for phrase in ("태그", "멘션", "언급"):
        assert phrase in section


def test_search_knowledge_is_explicitly_forbidden_with_reason(section: str):
    """금지만 적으면 모델이 어긴다 — 근거(@영문ID vs 한글 이름)를 함께 잠근다."""
    assert "mattermost_search_knowledge" in section
    assert "@영문ID" in section
    assert "@jsbaek" in section


def test_model_is_told_not_to_invent_its_own_handle(section: str):
    """`[요청자 정보] 사용자명`은 메신저 핸들이 아니라 이메일이다."""
    assert "이메일" in section
    assert "핸들" in section


def test_followup_uses_channel_distribution_not_research(section: str):
    """'다른 채널에선?' 은 재검색이 아니라 직전 결과의 channels 로 답한다."""
    assert "channels" in section
    assert "재검색" in section


def test_zero_result_does_not_take_the_widening_ladder(section: str):
    """이 도구의 0건은 '정말 없음' — 기간 확장·되묻기 사다리를 타면 안 된다."""
    assert "count: 0" in section or "`count: 0`" in section
    assert "find_channels" in section


def test_period_is_passed_as_argument(section: str):
    """상대 기간을 모델이 날짜로 환산하면 off-by-one 이 난다."""
    assert 'period="7d"' in section
