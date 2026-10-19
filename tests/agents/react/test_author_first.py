"""작성자 질의 판정기 계약 (`app.agents.react.author_first`).

왜 있는가 — 2026-09-16 운영: "홍길동 이사님이 최근 언급한 에이전트의 역할과 권한 문서라고
공유한 링크 찾아줘" 에 에이전트가 **작성자 검색을 부르지 않고** 일반 지식검색으로 가서,
AI팀 채널에 업로드된 첨부파일(TD4 화면설계서.hwp)을 답으로 내놨다. 정답은 홍길동가 올린
메신저 게시글이다. 같은 질문을 짧게("…공유한거 찾아줘") 물으면 잘 찾았다.

`system.py` 에는 이미 「"누가 ~한 글/링크/문서"는 **반드시** 작성자 검색」 이라고 적혀
있었다. **프롬프트만으로는 문장이 길어질 때 샌다** — 그래서 코드로 고정한다.

이 판정기는 **두 실행 경로 모두**에서 쓰인다 — `graphs/react_graph`(LangGraph)와
`react/loop`(구 직접 구현). 한쪽만 걸면 `USE_LANGGRAPH_REACT` 가 꺼진 환경에서 가드가
조용히 사라진다(`agent-name-guard` 의 "레거시 경로가 코드 기본값" 함정).
"""
from __future__ import annotations

import pytest

from agent_arch.agents.react.author_first import (
    AUTHOR_TOOL_NAME,
    is_author_question,
    pick_author_tool,
)


# ── 판정기 — 양성/음성 쌍 ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        # 이번 사고를 일으킨 실제 문장
        "홍길동 이사님이 최근 언급한 에이전트의 역할과 권한 문서라고 공유한 링크 찾아줘",
        # 그 전 사고(2026-09-15)의 1인칭 문장
        "내가 AI팀에 에이전트 정책 문서라고 공유한 링크 찾아줘",
        "이영희 이사님이 공유한 링크 찾아줘",
        "대표님이 올린 공지 찾아줘",
        "백지선님이 남긴 글 보여줘",
        "제가 어디에 그렇게 썼더라",
    ],
)
def test_author_questions_are_detected(text):
    assert is_author_question(text) is True


@pytest.mark.parametrize(
    "text",
    [
        # 묶으면 죽는 실제 운영 문장 — 전사 검색이 정답이다
        "지난주 품질 이슈 찾아줘",
        "이 스레드 정리해줘",
        "업무협약서.pdf 어디 있지",
        # 지목만 있고 글 쓰는 행위가 없다
        "이영희 이사님 연락처 알려줘",
        # 행위만 있고 지목이 없다
        "공유해줘",
        # 주격조사가 붙은 평범한 어절 — 성씨/존칭 관문이 없으면 여기서 오검출한다
        "품질이 왜 이렇게 나빠졌지",
        "보고서 어떻게 써야 하지",
    ],
)
def test_ordinary_questions_are_left_alone(text):
    assert is_author_question(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "이영희 이사님께 보낸 메일 찾아줘",     # 여격 — 받는 사람이지 작성자가 아니다
        "홍길동 대표님과 미팅 잡아줘",          # 일정
        "대표님이 올린 휴가 신청 결재해줘",      # 결재
    ],
)
def test_other_domains_win_over_author_deixis(text):
    """작성자 지목이 있어도 다른 도메인이 정답이면 강제하지 않는다.

    "이영희 이사님이 보낸 메일" 을 메신저 글 검색으로 묶으면 0건이 된다.
    """
    assert is_author_question(text) is False


# ── pick_author_tool — 2조건을 한 조건씩 ─────────────────────────────────


def test_forces_author_tool_when_bound():
    assert pick_author_tool("대표님이 올린 공지 찾아줘", [AUTHOR_TOOL_NAME]) == AUTHOR_TOOL_NAME


def test_no_force_when_tool_is_not_bound():
    """도구가 안 붙은 경로(MCP 미배포 등)에서 강제하면 tool NOT FOUND 가 난다."""
    assert pick_author_tool("대표님이 올린 공지 찾아줘", ["search"]) is None


def test_no_force_for_ordinary_questions():
    assert pick_author_tool("지난주 품질 이슈 찾아줘", [AUTHOR_TOOL_NAME]) is None


def test_empty_tool_list_is_safe():
    assert pick_author_tool("대표님이 올린 공지", []) is None

# ── 수신자가 나인 질문은 멘션 조회로 양보한다 ─────────────────────────────


@pytest.mark.parametrize(
    "text",
    [
        # 2026-09-17 실사용에서 잘못 채간 문장
        "이사님께서 최근에 나를 언급한 링크 찾아줘",
        "이사님이 나를 태그한 글 찾아줘",
        "이사님이 저를 언급하신 메시지",
        "누가 날 멘션했어",
    ],
)
def test_mention_target_yields_to_the_mentions_tool(text):
    """작성자까지 지목돼도 **멘션 조회**가 정답이다.

    메신저는 사람을 태그하면 본문에 영문 핸들(`@jsbaek`)로 저장한다. 작성자 검색으로 가면
    모델이 한글 이름("백지선")을 `query` 에 넣어 **이름을 손으로 타이핑한 글**만 걸린다.

    실측(2026-09-17, 운영 네이티브 검색):
        from:miim @jsbaek  → 103건, 최신 2026-09-17
        from:miim 백지선    →   8건, 최신 2026-08-26   ← 에이전트가 이걸 보고 "최근 것 없다"고 답했다
    """
    assert is_author_question(text) is False
    assert pick_author_tool(text, [AUTHOR_TOOL_NAME]) is None


@pytest.mark.parametrize(
    "text",
    [
        "이사님께서 에이전트 정책 문서 공유한 링크 찾아줘",
        "내가 공유한 링크 찾아줘",
        "대표님이 올린 공지",
    ],
)
def test_ordinary_author_questions_still_fire(text):
    """수신자 규칙이 작성자 검색 본래 용도를 죽이면 안 된다."""
    assert is_author_question(text) is True


# ── '드라이브'가 주제어일 때 (2026-10-01 실사용) ──────────────────────────────
# 맨 단어 '드라이브'로 막았더니 "DMP 드라이브로 전환 관련해 공유한 글" 처럼 드라이브가
# **논의 주제**인 메신저 질문까지 작성자 검색에서 빠졌다. 위치로 쓰였을 때만 막는다.

def test_drive_as_topic_does_not_block_author_search():
    assert is_author_question("홍길동 이사님이 DMP 드라이브로 전환하자고 공유한 글 찾아줘") is True


@pytest.mark.parametrize("text", [
    "내가 드라이브에 올린 파일 찾아줘",
    "이영희 이사님이 공유 폴더 안에 올린 문서 찾아줘",
])
def test_drive_as_place_still_wins(text):
    assert is_author_question(text) is False


def test_talk_about_is_not_an_author_verb():
    """'내가 ~ 이야기한 것'은 작성자 검색으로 고정하지 않는다.

    작성자 검색은 **내 글만** 돌려준다. 원 질문에서 사용자가 원한 결론은 같은 논의에서
    **남이 단 답글**(홍길동 이사님의 제안)이었다 — 대화 검색이 그 논의 전체를 보여줘야 한다.
    """
    assert is_author_question(
        "내가 이전에 confluence 관련해서 DMP 드라이브 로 전환에 관련해서 이야기 한 것이 있으면 찾아줘"
    ) is False
