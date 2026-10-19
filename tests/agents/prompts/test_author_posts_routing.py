"""'누가 공유한 ~' 은 의미검색이 아니라 작성자 검색 도구로 간다.

회귀 (실측 2026-09-15, 이슈): 같은 사고가 **두 번** 났다.

1. "내가 AI팀에 에이전트 정책 문서라고 공유한 링크 찾아줘" → 남이 만든 요금제 스프레드시트
2. 도구를 넣고 배포한 뒤, 실제 질문은 3인칭으로 왔다 —
   "이영희 이사님이 에이전트 정책 문서라고 공유한 링크 찾아줘" → 또 같은 요금제 시트

원인은 랭킹이 아니라 **표현력**이다. `mattermost_search_knowledge` 의 색인·필터 어디에도
작성자 조건이 없다. 사람 이름을 떼면 주제어만 남고, 같은 주제의 다른 사람 글이 이긴다.
두 번째 실패는 거기에 더해 도구 이름이 `find_my_posts` 였던 탓도 크다 — 3인칭 질문에
"my" 도구를 고를 이유가 모델에게 없다.

프롬프트가 잠가야 할 것은 넷이다 — (1) 이 질의형은 전용 도구로, (2) **3인칭·1인칭 둘 다**,
(3) **왜** 지식검색이면 안 되는지(이유가 없으면 모델이 곧 되돌아간다), (4) 'AI팀에' 같은
수신자 표현을 범위 인자에 넣지 말 것(첫 턴을 통째로 버리게 만든 오해).
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


@pytest.fixture
def flat(section: str) -> str:
    """줄바꿈·들여쓰기를 지운 본문 — 문구가 줄 경계에 걸려도 검사가 깨지지 않게."""
    return " ".join(section.split())


def test_author_query_routes_to_dedicated_tool(flat: str):
    assert "mattermost_skill_find_posts_by_author" in flat


def test_third_person_is_covered_with_an_example(flat: str):
    """3인칭이 실제로 들어온 형태다 — 예시가 없으면 모델이 지식검색으로 되돌아간다."""
    assert "이영희 이사님이 공유한 링크" in flat
    assert 'author="이영희"' in flat


def test_first_person_leaves_author_empty(flat: str):
    """본인 핸들을 지어내면 틀린다 — `[요청자 정보]` 의 사용자명은 이메일이다."""
    assert "내가 공유한 링크" in flat
    assert "author`를 비워 둬라" in flat or "author`를 비워둬라" in flat


def test_search_knowledge_is_forbidden_with_reason(flat: str):
    assert "작성자 조건을 표현할 수 없다" in flat
    assert "다른 사람 글" in flat


def test_person_name_does_not_belong_in_query(flat: str):
    assert "사람 이름은 `query`에 남기지 마라" in flat


def test_link_queries_use_the_link_filter(flat: str):
    assert "must_contain_link" in flat


def test_period_is_left_empty_by_default(flat: str):
    """기본이 전체 기간이다 — 임의로 최근으로 좁히면 옛 글이 곧바로 0건이 된다."""
    assert "전체 기간" in flat


def test_homonym_candidates_are_asked_back(flat: str):
    """동명이인을 임의로 고르면 엉뚱한 사람 글이 '그 사람 글'로 나간다."""
    assert "candidates" in flat
    assert "되물어라" in flat


def test_messenger_permalink_is_required_for_link_questions(flat: str):
    """'메신저에서 공유한 링크' 는 그 **메시지 주소**를 원한 것이다.

    실측 2026-09-15: 정답 글을 찾아 채널·본문까지 맞게 답했는데 링크는 글 안의 Confluence
    URL 만 줬다. 사용자가 원한 건 회사 scrum 의 그 글로 가는 메신저 링크였다.
    """
    assert "permalink" in flat
    assert "메시지 자체" in flat


def test_audience_words_are_not_scope(flat: str):
    """'AI팀에' 는 받는 사람이지 팀·채널이 아니다(첫 턴을 버린 원인)."""
    assert "AI팀" in flat
    assert "받는 사람" in flat
