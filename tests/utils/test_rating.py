"""``app/utils/rating.py`` 분류 함수 단위 테스트.

핵심 점검:
- ``None``/``""`` → ``(None, None)`` 미평가
- preset 카탈로그 매칭 → ``("positive"|"negative", "preset")``
- 카탈로그 외 비어있지 않은 값 → ``("negative", "freeform")``
- ``_NEGATIVE_PRESETS`` 확장 시 자동 반영 (단일 출처 원칙 회귀 보호)
"""
from __future__ import annotations

import pytest

from agent_arch.utils import rating as rating_module
from agent_arch.utils.rating import classify_rating


class TestClassifyRating:
    """``classify_rating`` 분기별 케이스."""

    def test_none_is_unrated(self):
        """None은 미평가."""
        assert classify_rating(None) == (None, None)

    def test_empty_string_is_unrated(self):
        """빈 문자열도 미평가로 처리."""
        assert classify_rating("") == (None, None)

    def test_positive_preset_thumbs_up(self):
        """``thumbs_up``은 긍정 preset."""
        assert classify_rating("thumbs_up") == ("positive", "preset")

    def test_freeform_korean_is_negative_freeform(self):
        """자유 한글 텍스트는 부정 freeform."""
        assert classify_rating("표현이 어색하고 이해하기 어려움") == ("negative", "freeform")

    def test_unknown_short_token_is_negative_freeform(self):
        """preset 카탈로그 외 짧은 토큰도 freeform 폴백."""
        assert classify_rating("inaccurate") == ("negative", "freeform")

    def test_whitespace_only_is_freeform(self):
        """공백만 있는 문자열은 빈 문자열이 아니므로 freeform.

        Note:
            의도된 동작. 라우터/저장 단에서 trim 정책을 따로 두지 않는다.
        """
        assert classify_rating(" ") == ("negative", "freeform")


class TestCatalogExtensionSoT:
    """카탈로그(set) 확장 시 분류 함수가 자동 반영되는지 회귀 보호.

    rating 분류는 ``app/utils/rating.py`` 한 곳만 사용하는 단일 출처 원칙을 따른다.
    ``_NEGATIVE_PRESETS``에 신규 코드값을 추가하면 ``classify_rating``이 즉시 인식해야 한다.
    """

    def test_negative_preset_added_via_monkeypatch(self, monkeypatch):
        """``_NEGATIVE_PRESETS``에 코드값을 monkeypatch로 추가 → ``("negative", "preset")``."""
        extended = rating_module._NEGATIVE_PRESETS | {"inaccurate"}
        monkeypatch.setattr(rating_module, "_NEGATIVE_PRESETS", extended)
        # ``_ALL_PRESETS``는 모듈 로드 시점에 합쳐진 값이므로 본 테스트에서는 검증 대상 외.
        # classify_rating은 호출 시점에 ``_NEGATIVE_PRESETS``를 다시 참조하므로 동기화된다.

        assert classify_rating("inaccurate") == ("negative", "preset")

    def test_positive_preset_added_via_monkeypatch(self, monkeypatch):
        """``_POSITIVE_PRESETS``에 신규 긍정 코드 추가 → ``("positive", "preset")``."""
        extended = rating_module._POSITIVE_PRESETS | {"great_answer"}
        monkeypatch.setattr(rating_module, "_POSITIVE_PRESETS", extended)

        assert classify_rating("great_answer") == ("positive", "preset")

    def test_default_negative_presets_empty(self):
        """기본 카탈로그에 ``_NEGATIVE_PRESETS``는 비어 있음을 명시.

        Why:
            현 시점 부정 평가는 모두 자유 텍스트로만 들어옴. 카탈로그가 도입되면 본 케이스가
            깨지면서 명시적으로 갱신을 강제한다.
        """
        assert rating_module._NEGATIVE_PRESETS == frozenset()

    def test_positive_presets_contains_thumbs_up(self):
        """``thumbs_up``은 긍정 카탈로그에 포함."""
        assert "thumbs_up" in rating_module._POSITIVE_PRESETS


def test_doctest_runs():
    """모듈 docstring에 포함된 doctest가 통과하는지 확인."""
    import doctest

    results = doctest.testmod(rating_module, verbose=False)
    assert results.failed == 0
