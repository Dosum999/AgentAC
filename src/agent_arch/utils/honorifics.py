"""직급·직책 어휘와 호칭 제거 (단일 SOT).

**왜 한곳으로 모으는가.** 같은 어휘 목록이 지금 세 군데에 흩어져 있고 서로 다르다 —
`app/agents/hitl/answer_guard.py` 의 `_TITLE_ALT`(출력 환각 탐지용, 23개),
`Mattermost-MCP` 의 `_HONORIFICS`(메신저 작성자 검색용, `부사장` 포함),
그리고 프롬프트 문장들. 한쪽만 고치면 다른 쪽에서 같은 버그가 되살아난다.

⚠️ `answer_guard` 의 **정규식 자체는 여기로 옮기지 않았다.** 그쪽은 "답변에 지어낸 사람
이름이 있는가"를 잡는 탐지기라 캡처 그룹 구조와 성씨 관문이 목적에 맞춰져 있다. 공유할
가치가 있는 것은 **어휘 목록**이다.

## 무엇을 고치는가

2026-09-09 실사용: "10시에 **대표님**과 성호그룹 협의 관련 미팅을 잡아 주세요" →
에이전트가 *"대표님 정보를 찾을 수 없습니다"*. "**홍길동 대표님**과 미팅을 할 예정입니다"
도 같은 답이었다. 그런데 "홍길동" 로만 물으면 두 명이 나왔다.

원인은 인명부가 직급을 모르는 게 아니다 — 서버 SQL 은 이미
`designation_name LIKE '%검색어%'` 로 **직급을 검색 조건에 넣고 있다**. 문제는
`"대표님"`·`"홍길동 대표님"` 이 **그대로** LIKE 에 들어가 0건이 된 것이다. 즉 호칭
접미만 떼면 `"대표"` 는 직급으로, `"홍길동"` 는 이름으로 곧장 맞는다.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

__all__ = [
    "TITLES", "TITLE_ALT", "strip_honorifics", "split_honorifics",
    "narrow_rows_by_title", "looks_like_title_only",
]

#: 사내에서 쓰는 직급·직책. 길이 내림차순으로 두어 `strip_honorifics` 가
#: **긴 것부터** 떼도록 한다("본부장" 을 "부장" 으로 잘못 자르지 않게).
#:
#: 출처 두 곳의 합집합이다 — `answer_guard._TITLE_ALT`(수석·매니저·반장·소장·국장·
#: 센터장·파트장이 여기에만 있었다) ∪ `Mattermost-MCP._HONORIFICS`(부사장이 여기에만
#: 있었다). 새 직급이 생기면 **이 목록만** 고친다.
TITLES: Tuple[str, ...] = tuple(
    sorted(
        (
            "대표이사", "부사장", "본부장", "센터장", "파트장",
            "사장", "전무", "상무", "이사", "실장", "부장", "차장", "과장", "팀장",
            "대리", "주임", "사원", "선임", "책임", "수석", "매니저", "반장", "소장",
            "국장", "대표", "지사장", "그룹장", "총괄",
        ),
        key=len,
        reverse=True,
    )
)

#: 정규식 대안 문자열(`(?:이사|부장|…)`). 긴 것이 앞에 와야 교대 매칭이 옳다.
TITLE_ALT = "|".join(TITLES)

#: 이름 뒤에 붙는 존칭. 직급과 결합형("이사님")은 아래 로직이 반복 제거로 처리하므로
#: 여기엔 홑 존칭만 둔다.
_SUFFIXES: Tuple[str, ...] = ("님", "씨", "분")

_TITLE_ONLY_RE = re.compile(rf"^(?:{TITLE_ALT})$")


def split_honorifics(name: str) -> Tuple[str, str]:
    """`"강태욱 주임님"` → `("강태욱", "주임")`, `"대표님"` → `("", "대표")`, `"성호그룹"` → `("성호그룹", "")`.

    **뗀 직급을 버리지 않는다.** 2026-09-17 dev 실측: `"강태욱주임님"` 을 `"강태욱"` 으로
    떼고 검색해 실계정 둘(사원·주임)을 받아 놓고는, "주임" 을 이미 알고 있었음에도 어느
    분이냐고 되물었다. 직급은 검색어에 넣으면 0건을 만들지만(`full_name LIKE '%강태욱 주임%'`),
    **결과를 좁히는 필터**로는 그대로 쓸 수 있다 — 서버가 `designation`·`position` 을 돌려준다.

    직급이 여럿 붙어 있으면(`"홍길동 대표이사 이사님"` 같은 비정상 입력) **끝에서 가장 먼저
    만난 것**을 직급으로 보고 나머지는 떼기만 한다.

    Args:
        name: 사용자가 말한 대로의 이름/호칭.

    Returns:
        (직급을 뗀 이름, 뗀 직급). 직급만 있었으면 이름은 빈 문자열, 직급이 없었으면 직급이
        빈 문자열이다. 둘 다 빈 문자열이면 입력이 비어 있었다는 뜻이다.
    """
    raw = (name or "").strip().lstrip("@").strip()
    if not raw:
        return "", ""
    title = ""
    changed = True
    while changed and raw:
        changed = False
        for token in (*_SUFFIXES, *TITLES):
            if not raw.endswith(token):
                continue
            is_title = token in TITLES
            # 남은 것이 **직급 그 자체**면 거기서 멈춘다. 계속 돌면 더 짧은 직급이
            # 뒤에서 다시 걸려 "본부장" 이 "본" 이 된다(`본부장`.endswith(`부장`)).
            if len(raw) == len(token):
                if is_title:
                    return "", (title or raw)
                return raw, title
            head = raw[: -len(token)].strip()
            if not head:
                continue
            if is_title and not title:
                title = token
            raw = head
            changed = True
            break
    return raw, title


def strip_honorifics(name: str) -> str:
    """`"홍길동 대표님"` → `"홍길동"`, `"대표님"` → `"대표"`.

    직함이 이름에 붙어 오는 건 한국어 질문에서 기본값에 가깝다. 그대로 검색에 넣으면
    0건이 나오고, 모델은 *"그런 사람이 없다"* 로 잘못 옮긴다.

    **직급만 남는 경우는 직급을 지우지 않는다** — `"대표님"` 에서 `"대표"` 까지 떼면
    빈 문자열이 되어 검색이 아예 불가능해지는데, 서버는 직급으로도 찾을 수 있다.
    존칭(`님`)만 떼고 `"대표"` 를 남기는 것이 정답이다.

    검색어 하나만 필요할 때의 편의 함수다. 뗀 직급까지 필요하면 `split_honorifics` 를 쓴다.

    Args:
        name: 사용자가 말한 대로의 이름/호칭.

    Returns:
        검색어로 쓸 수 있게 다듬은 문자열. 전부 떨어져 나가면 원문을 돌려준다.
    """
    bare, title = split_honorifics(name)
    return bare or title or (name or "").strip()


def narrow_rows_by_title(rows: Iterable[dict], title: str) -> List[dict]:
    """직급/직책으로 후보를 **좁힌다** — 없으면 원본을 돌려주는 3계약(좁혀서 남으면 그것, 0이면 원본).

    `"강태욱 주임님"` 에서 뗀 `"주임"` 을 결과의 `designation`·`position` 에 댄다. 정확히 같은
    것을 먼저 찾고(`"이사"` 가 `"대표이사"` 를 삼키지 않게), 없으면 포함으로 한 번 더 본다
    (`"대표"` → `"대표이사"`). 그래도 없으면 **원본을 돌려준다** — 사용자가 직급을 틀리게
    기억하는 일이 흔하므로, 여기서 0건을 확정하면 "그런 사람 없다" 로 새는 옛 버그가 되살아난다.

    가상 여부와 **무관하게** 좁힌다. `"강태욱 대리님"` 의 대리가 가상 계정 하나뿐이면 그
    사람만 남기는 것이 맞다 — 가상 우선순위(§4)는 이 다음 단계가 판단한다.

    Args:
        rows: 디렉터리 행들(`designation`·`position` 키를 읽는다).
        title: 뗀 직급. 비어 있으면 원본 그대로.

    Returns:
        좁힌 리스트, 또는 원본 리스트(복사).
    """
    rows = list(rows or [])
    want = (title or "").strip().casefold()
    if not want or not rows:
        return rows

    def _fields(row: dict) -> Tuple[str, str]:
        return (
            str(row.get("designation") or "").strip().casefold(),
            str(row.get("position") or "").strip().casefold(),
        )

    exact = [r for r in rows if want in _fields(r)]
    if exact:
        return exact
    loose = [r for r in rows if any(want in f for f in _fields(r) if f)]
    return loose or rows


def looks_like_title_only(text: str) -> bool:
    """호칭을 떼고 나니 **직급 하나만** 남았는지(`"대표님"` → `"대표"`).

    이 경우 이름이 아니라 직급으로 찾아야 하므로, 호출부가 되묻기 대신 직급 검색을
    시도할 수 있다.
    """
    return bool(_TITLE_ONLY_RE.match(strip_honorifics(text)))
