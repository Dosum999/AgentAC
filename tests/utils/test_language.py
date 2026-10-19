"""답변 언어 감지·판정 테스트 (app/utils/language.py).

핵심은 두 가지다:
  1. 목표 언어를 넘겨 판정하는 ``looks_off_target`` 이 **한국어 목표에서 기존
     ``looks_non_korean`` 과 같은 결론**을 낸다(이관 회귀 방지).
  2. 기존 판정이 못 잡던 구멍 — 한자·가나 답변 — 이 닫혔다.
"""
import pytest

from agent_arch.agents.hitl.answer_guard import looks_non_korean
from agent_arch.utils.language import (
    detect_language,
    detect_language_with_history,
    looks_off_target,
    normalize_language,
    script_counts,
)

_EN = ("Hello! Here are your three unread emails from this morning. "
       "Let me know if you want the details.")
_KO = "안녕하세요. 오늘 오전에 도착한 읽지 않은 메일은 3건입니다. 자세한 내용도 알려드릴까요?"
_ZH = "您好！这是您今天上午收到的三封未读邮件。如果需要详细信息，请告诉我。"
_JA = "こんにちは。今朝届いた未読メールは3件です。詳細が必要ならお知らせください。"


# ── 감지 ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    (_KO, "ko"),
    (_EN, "en"),
    (_ZH, "zh"),
    (_JA, "ja"),
    ("결재(決裁) 문서 3건이 대기 중입니다.", "ko"),   # 한자 병기는 한국어
    ("Mattermost 채널을 열어줘", "ko"),               # 영문 고유명사 혼입도 한국어
    ("请查看 Agent 系统的未读消息", "zh"),             # 영문 제품명 혼입도 중국어
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_detect_language_short_input_is_undecided():
    """짧은 확인 발화는 언어를 단정하지 않는다 — 턴마다 언어가 튀면 그게 더 큰 고장이다."""
    for text in ("ok", "3", "?", "", "👍"):
        assert detect_language(text, fallback=None) is None


def test_detect_language_falls_back_when_undecided():
    assert detect_language("ok") == "ko"          # 기본값
    assert detect_language("ok", fallback="en") == "en"


def test_detect_language_with_history_inherits_previous_turn():
    """판정 불가 턴은 직전 발화에서 언어를 물려받는다."""
    assert detect_language_with_history("ok", [_EN]) == "en"
    assert detect_language_with_history("3", [_ZH, _EN]) == "zh"   # 최근 것이 먼저
    assert detect_language_with_history("ok", []) == "ko"          # 물려받을 것 없음
    # 현재 발화가 판정되면 히스토리는 보지 않는다.
    assert detect_language_with_history(_EN, [_KO]) == "en"


@pytest.mark.parametrize("raw,expected", [
    ("ko", "ko"), ("KO", "ko"), ("ko-KR", "ko"), ("zh_CN", "zh"), ("zh-Hans", "zh"),
    ("jp", "ja"), ("eng", "en"), ("EN-US", "en"),
    (None, None), ("", None), ("de", None), ("klingon", None),
])
def test_normalize_language(raw, expected):
    assert normalize_language(raw) == expected


# ── 판정(가드) ────────────────────────────────────────────────────────

def test_hangul_counts_include_jamo():
    assert script_counts("ㅇㅋ 알겠어요")["hangul"] == 6


@pytest.mark.parametrize("text,expected", [
    (_EN, True),                                    # 영어 답변 → 목표(ko) 아님
    (_KO, False),
    ("Here is the summary of your unread mail: three messages. 확인해 주세요.", True),
])
def test_korean_target_matches_legacy_guard(text, expected):
    """한국어 목표 판정은 이관 전 ``looks_non_korean`` 과 같은 결론이어야 한다."""
    assert looks_off_target(text, "ko") is expected
    assert looks_non_korean(text) is expected


def test_cjk_answers_no_longer_slip_through_korean_guard():
    """기존 구멍: 한자·가나는 라틴 문자가 없어 판정 자체를 건너뛰었다."""
    assert looks_off_target(_ZH, "ko") is True
    assert looks_off_target(_JA, "ko") is True


@pytest.mark.parametrize("language,text,expected", [
    ("en", _EN, False),
    ("en", _KO, True),        # 영어 목표인데 한국어로 답함 → 재생성 대상
    ("zh", _ZH, False),
    ("zh", _KO, True),
    ("ja", _JA, False),
    ("ja", _KO, True),
])
def test_off_target_mirrors_each_language(language, text, expected):
    assert looks_off_target(text, language) is expected


def test_short_or_identifier_only_text_is_not_judged():
    """이질 문자가 적으면(고유명사·약어 수준) 판정하지 않는다 — 오탐 방지."""
    assert looks_off_target("OK", "ko") is False
    assert looks_off_target("Agent", "ko") is False
    assert looks_off_target("", "ko") is False
    assert looks_off_target("에이전트", "en") is False        # 이질(한글) 20자 미달


def test_unknown_language_falls_back_to_korean_judgement():
    """지시문이 없는 언어는 기본값(ko) 기준으로 판정한다 — 판정 자체를 끄지는 않는다."""
    assert looks_off_target(_EN, "de") is True
    assert looks_off_target(_KO, "de") is False
