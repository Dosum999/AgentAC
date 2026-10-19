"""직급 어휘·호칭 제거 계약 (`app.utils.honorifics`).

왜 있는가 — 2026-09-09 실사용에서 에이전트가 `"대표님"`·`"○○ 대표님"` 을 **그대로** 인명부
검색에 넣어 0건을 받고 *"정보를 찾을 수 없습니다"* 라고 답했다. 서버 SQL 은 이미
`designation_name LIKE` 로 직급을 검색 조건에 갖고 있었으므로, 호칭만 떼면 곧바로 맞는다.

이 모듈은 흩어져 있던 세 어휘 목록의 단일 SOT 다. 새 직급이 생기면 여기만 고친다.
"""
from __future__ import annotations

import pytest

from agent_arch.utils.honorifics import TITLES, looks_like_title_only, strip_honorifics


@pytest.mark.parametrize(
    "asked, expected",
    [
        ("한글이름 대표님", "한글이름"),
        ("한글이름 이사님", "한글이름"),
        ("한글이름 부사장님", "한글이름"),
        ("한글이름 수석", "한글이름"),
        ("한글이름님", "한글이름"),
        ("@handle", "handle"),
        # 직급만 남으면 **그것을 검색어로 쓴다** — 지우면 찾을 방법이 없어진다.
        ("대표님", "대표"),
        ("팀장님", "팀장"),
        ("센터장님", "센터장"),
        # 사람이 아닌 말은 건드리지 않는다
        ("성호그룹", "성호그룹"),
        ("", ""),
    ],
)
def test_honorifics_are_stripped(asked, expected):
    assert strip_honorifics(asked) == expected


def test_longer_titles_win_over_their_own_suffix():
    """`본부장`.endswith(`부장`) 이라, 짧은 직급이 뒤에서 다시 걸리면 "본" 이 된다."""
    assert strip_honorifics("본부장님") == "본부장"
    assert strip_honorifics("한글이름 본부장님") == "한글이름"


@pytest.mark.parametrize("text, expected", [("대표님", True), ("본부장", True),
                                            ("한글이름 대표님", False), ("성호그룹", False)])
def test_title_only_detection(text, expected):
    assert looks_like_title_only(text) is expected


def test_vocabulary_is_the_union_of_the_two_old_lists():
    """흩어져 있던 두 목록을 합친 것이 이 SOT 의 존재 이유다 — 한쪽만 고치면 되살아난다."""
    # `answer_guard._TITLE_ALT` 에만 있던 것들
    for only_in_guard in ("수석", "매니저", "반장", "소장", "국장", "센터장", "파트장"):
        assert only_in_guard in TITLES
    # Mattermost `_HONORIFICS` 에만 있던 것
    assert "부사장" in TITLES


def test_titles_are_ordered_longest_first():
    """`strip_honorifics` 가 긴 것부터 떼려면 이 순서가 유지돼야 한다."""
    assert list(TITLES) == sorted(TITLES, key=len, reverse=True)


# ── 뗀 직급을 버리지 않는다 (2026-09-17) ──────────────────────────────────
#
# dev 실측: "결재자 강태욱주임님" → "강태욱" 으로 떼고 검색 → 실계정 둘(사원·주임) →
# "주임" 을 이미 알고 있었음에도 어느 분이냐고 되물었다. 직급은 검색어엔 못 넣지만
# (`LIKE '%강태욱 주임%'` 은 0건) 결과를 좁히는 필터로는 그대로 쓸 수 있다.

from agent_arch.utils.honorifics import narrow_rows_by_title, split_honorifics  # noqa: E402


@pytest.mark.parametrize(
    "asked, expected",
    [
        ("강태욱주임님", ("강태욱", "주임")),        # 띄어쓰기 없이 붙여 써도 갈린다
        ("강태욱 대리님", ("강태욱", "대리")),
        ("홍길동 대표이사님", ("홍길동", "대표이사")),
        ("한글이름 본부장님", ("한글이름", "본부장")),  # "부장" 이 아니다
        ("대표님", ("", "대표")),                     # 직급만 — 이름은 비고 직급이 검색어가 된다
        ("본부장", ("", "본부장")),
        ("한글이름님", ("한글이름", "")),              # 존칭만 — 직급 없음
        ("성호그룹", ("성호그룹", "")),
        ("", ("", "")),
    ],
)
def test_split_keeps_the_title(asked, expected):
    assert split_honorifics(asked) == expected


def test_strip_is_split_collapsed_to_one_search_term():
    """strip 은 split 의 편의형이다 — 둘이 어긋나면 도구와 경비 경로가 다른 검색어를 보낸다."""
    for asked in ("강태욱 주임님", "대표님", "성호그룹", "한글이름님"):
        bare, title = split_honorifics(asked)
        assert strip_honorifics(asked) == (bare or title or asked)


# dev (주)회사 강태욱 3명 실측(2026-09-17) 그대로 — 직급·직책이 서로 다르다.
_ROWS = [
    {"designation": "사원", "position": "팀원"},
    {"designation": "주임", "position": None},
    {"designation": "대리", "position": "팀장"},
    {"designation": "대표이사", "position": None},
]


def test_narrow_prefers_exact_designation_or_position():
    assert [r["designation"] for r in narrow_rows_by_title(_ROWS, "주임")] == ["주임"]
    assert [r["designation"] for r in narrow_rows_by_title(_ROWS, "팀장")] == ["대리"]   # 직책으로도


def test_narrow_falls_back_to_substring_when_no_exact_match():
    """"이사" 가 "대표이사" 를 삼키지 않게 정확 일치를 먼저 보고, 없을 때만 포함으로 본다."""
    assert [r["designation"] for r in narrow_rows_by_title(_ROWS, "대표")] == ["대표이사"]


def test_narrow_returns_original_when_nothing_matches():
    """직급을 틀리게 기억하는 일이 흔하다 — 0건을 확정하면 "그런 사람 없다" 옛 버그가 되살아난다."""
    assert narrow_rows_by_title(_ROWS, "부장") == _ROWS
    assert narrow_rows_by_title(_ROWS, "") == _ROWS
    assert narrow_rows_by_title([], "주임") == []
