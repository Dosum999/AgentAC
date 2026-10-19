"""rating 값 분류 단일 출처.

DB에 저장된 rating은 두 종류가 섞여 있다:
- preset 코드값 (예: ``'thumbs_up'``) — 프론트 버튼 클릭
- freeform 자유 텍스트 (예: ``'표현이 어색하고 이해하기 어려움'``) — 사용자 직접 입력

본 모듈은 raw 값 → ``(polarity, kind)`` 분류 + 필터 SQL용 카탈로그를 한 곳에 둔다.
카탈로그가 늘어나면 본 모듈의 set만 갱신하면 응답 분류와 필터 SQL이 자동 정합한다.

향후 부정 평가에 코드값(예: ``inaccurate``, ``hallucination``)이 도입되면
``_NEGATIVE_PRESETS``에 추가하는 것만으로 분류와 필터 SQL이 동시에 반영된다.
타 파일에서 직접 ``{"thumbs_up"}`` 같은 set 리터럴을 쓰지 말 것.
"""
from __future__ import annotations

from typing import Optional


# preset 카탈로그 — 프론트가 버튼으로 보낼 수 있는 고정 코드값 집합.
# 카탈로그에 없는 NULL이 아닌 값(자유 텍스트 등)은 rating_kind == "freeform"으로 폴백된다.
_POSITIVE_PRESETS: frozenset[str] = frozenset({"thumbs_up"})
_NEGATIVE_PRESETS: frozenset[str] = frozenset()

_ALL_PRESETS: frozenset[str] = _POSITIVE_PRESETS | _NEGATIVE_PRESETS


def rated_sql(column: str) -> str:
    """"평가됨" 판정 SQL 술어를 만든다.

    ``classify_rating``이 빈 문자열을 **미평가**로 보므로 SQL 술어도 같은 기준이어야 한다.
    ``IS NOT NULL``만 쓰면 ``rating=''``인 턴이 목록 집계에는 잡히는데 분류는 미평가라
    "사유도 배지도 없는 빈 줄"이 화면에 뜬다.

    Args:
        column: 판정할 컬럼 표현식 (예: ``"m.rating"``).

    Returns:
        괄호로 감싼 SQL 조각.
    """
    return f"({column} IS NOT NULL AND {column} <> '')"


def unrated_sql(column: str) -> str:
    """"미평가" 판정 SQL 술어. ``rated_sql``의 부정형이며 같은 기준을 쓴다."""
    return f"({column} IS NULL OR {column} = '')"


def classify_rating(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """raw rating 텍스트를 ``(polarity, kind)``로 분류한다.

    Args:
        raw: ``agent.messages.rating``에 저장된 raw TEXT. None 또는 빈 문자열이면 미평가.

    Returns:
        ``(polarity, kind)``:
            polarity: ``"positive"`` | ``"negative"`` | ``None`` (미평가)
            kind:     ``"preset"``   | ``"freeform"`` | ``None`` (미평가)

    Examples:
        >>> classify_rating(None)
        (None, None)
        >>> classify_rating("")
        (None, None)
        >>> classify_rating("thumbs_up")
        ('positive', 'preset')
        >>> classify_rating("표현이 어색하고 이해하기 어려움")
        ('negative', 'freeform')
    """
    if raw is None or raw == "":
        return None, None
    if raw in _POSITIVE_PRESETS:
        return "positive", "preset"
    if raw in _NEGATIVE_PRESETS:
        return "negative", "preset"
    # preset 카탈로그 외 = 자유 텍스트 부정 의견
    return "negative", "freeform"
