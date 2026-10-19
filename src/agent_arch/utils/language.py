"""답변 언어 결정·판정 (한국어 전용 강제의 일반화).

실사용 발화가 한국어·영어·중국어 세 가지라 **입력** 쪽은 진작 언어를 못 박지 않는다
(``settings.stt_language`` 기본 빈 값 = 모델 자동 감지. ``language=ko`` 로 고정하면 영어
mp3 를 한국어로 **번역**해 낸다 — 2026-09-03 실측). 그런데 **답변** 쪽만 한국어로 고정돼
있었다: 프롬프트가 "항상 한국어"를 지시하고, 그걸 어긴 답변은 가드가 한국어로 다시 쓰게
했다. 즉 "입력 언어는 자유, 출력은 한국어 고정" 이 상태였다.

이 모듈은 강제 **대상**을 "한국어"에서 **턴의 목표 언어**로 일반화한다. 가드를 없애지
않는다 — 프롬프트만으로는 모델이 영어 입력에서 지시를 미러링하며 무시한다는 것이 애초에
가드가 생긴 이유다(``settings.korean_answer_guard`` 주석 참조). 판정 기준만 바꾼다.

판정은 **문자 스크립트 비율**만 본다. LLM 없이 결정적이라 프로바이더(gemini/gemma/openai)
무관이고, 기존 ``looks_non_korean`` 과 동일한 계약이다.

기존 판정의 구멍(2026-09-09 실측): 한글:라틴 비율만 봤기 때문에 한자·가나에는 라틴 문자가
없어 ``_MIN_FOREIGN`` 미달로 **판정 자체를 건너뛰었다**. 그래서 중국어·일본어 답변에는
가드가 붙지 않았고(프롬프트 순종도에 따라 턴마다 결과가 흔들림), 영어 답변만 확정적으로
한국어로 교체됐다. 이질 스크립트를 모두 세어 그 비대칭을 닫는다.
"""
from __future__ import annotations

import re
from typing import Dict, Optional

# 목표 언어를 못 정했을 때의 기본값. 사용자 대다수가 한국어라 기존 동작을 그대로 유지한다.
DEFAULT_LANGUAGE = "ko"

# 답변 언어로 지원하는 코드. 프롬프트 지시문·재작성 지시문이 이 코드마다 준비돼 있어야 한다.
SUPPORTED_LANGUAGES = ("ko", "en", "zh", "ja")

# ── 스크립트 문자 클래스 ──────────────────────────────────────────────
# 한글은 음절뿐 아니라 자모(ㄱ-ㆎ)까지 센다 — "ㅇㅋ", "ㄴㄴ" 같은 입력도 한국어다.
_SCRIPT_RE: Dict[str, re.Pattern] = {
    "hangul": re.compile(r"[가-힣ㄱ-ㆎ]"),
    "kana":   re.compile(r"[ぁ-ゖ゛-ヿｦ-ﾟ]"),
    "han":    re.compile(r"[一-鿿㐀-䶿]"),
    "latin":  re.compile(r"[A-Za-z]"),
}

# 언어 → 그 언어의 본문을 이루는 스크립트. 나머지는 모두 "이질 스크립트"로 센다.
# 일본어만 둘인 것은 가나·한자 혼용이 정상 표기이기 때문이다.
_TARGET_SCRIPTS: Dict[str, tuple] = {
    "ko": ("hangul",),
    "en": ("latin",),
    "zh": ("han",),
    "ja": ("kana", "han"),
}

# 프롬프트(한국어로 쓰인 지시문)에 끼워 넣는 언어명. ko 로 렌더하면 기존 문구와 바이트 동일.
# 네 값 모두 "…어" 로 끝나 조사 "로" 가 그대로 붙는다.
LANGUAGE_NAME_KO: Dict[str, str] = {
    "ko": "한국어",
    "en": "영어",
    "zh": "중국어",
    "ja": "일본어",
}

# 요청이 보내는 흔한 표기를 지원 코드로 흡수한다. 지역 서브태그(ko-KR, zh-Hans)는
# 앞자리만 보므로 여기 넣지 않는다.
_LANGUAGE_ALIAS: Dict[str, str] = {
    "kr": "ko",
    "kor": "ko",
    "cn": "zh",
    "chs": "zh",
    "cht": "zh",
    "jp": "ja",
    "jpn": "ja",
    "eng": "en",
}

# ── 판정 상수 ────────────────────────────────────────────────────────
# 이질 문자가 이보다 적으면 판정하지 않는다(고유명사·코드·식별자 수준의 혼입 → 오탐 방지).
# 기존 ``answer_guard._MIN_LATIN`` 과 같은 값·같은 역할이다.
_MIN_FOREIGN = 20
# 목표 스크립트 비율이 이보다 낮으면 "다른 언어로 답했다"고 판정한다.
_TARGET_RATIO_FLOOR = 0.15
# 감지용 라틴 최소치. 이보다 짧은 라틴 입력("ok", "3")은 영어로 단정하지 않는다 —
# 짧은 확인 발화에서 언어가 튀면 대화 중간에 답변 언어가 바뀐다.
_MIN_LATIN_DETECT = 4


def script_counts(text: str) -> Dict[str, int]:
    """스크립트별 문자 수를 센다.

    Args:
        text: 대상 텍스트.

    Returns:
        ``{"hangul": n, "kana": n, "han": n, "latin": n}``
    """
    return {name: len(rx.findall(text)) for name, rx in _SCRIPT_RE.items()}


def normalize_language(value: Optional[str]) -> Optional[str]:
    """언어 코드를 지원 코드로 정규화한다. 모르는 값은 ``None``.

    ``"ko-KR"``, ``"zh_CN"``, ``"EN"``, ``"jp"`` 같은 표기를 흡수한다. 지원하지 않는
    언어는 ``None`` 을 돌려주므로 호출자가 기본값으로 떨어뜨릴 수 있다 — 알 수 없는
    코드를 그대로 통과시키면 지시문 없는 언어로 답하라는 프롬프트가 만들어진다.

    Args:
        value: 요청·설정이 준 원본 문자열.

    Returns:
        지원 코드(``"ko"``/``"en"``/``"zh"``/``"ja"``) 또는 ``None``.
    """
    if not value:
        return None
    v = value.strip().lower().replace("_", "-")
    if not v:
        return None
    if v in _LANGUAGE_ALIAS:
        return _LANGUAGE_ALIAS[v]
    base = v.split("-")[0]
    if base in SUPPORTED_LANGUAGES:
        return base
    return _LANGUAGE_ALIAS.get(base)


def detect_language(text: str, *, fallback: Optional[str] = DEFAULT_LANGUAGE) -> Optional[str]:
    """입력 문장의 언어를 스크립트 비율로 추정한다.

    선명한 신호부터 본다:
      1. **가나**가 있으면 일본어 — 한국어·중국어 문장에 가나가 섞이는 일은 거의 없다.
      2. **한글**이 한자보다 많으면 한국어 — 한자 병기("결재(決裁)")를 흡수한다.
      3. **한자**가 있으면 중국어(가나가 없으므로 일본어는 이미 걸러졌다).
      4. **라틴**이 ``_MIN_LATIN_DETECT`` 이상이면 영어.
      5. 그 외(숫자·이모지·아주 짧은 입력)는 판정 불가 → ``fallback``.

    5번을 억지로 판정하지 않는 것이 중요하다. "ok", "3번" 같은 확인 발화에서 언어가
    튀면 대화 중간에 답변 언어가 바뀐다. 호출자는 ``fallback=None`` 으로 받아 직전 턴
    언어를 물려받는 것을 권한다(orchestrator ``_resolve_turn_language`` 가 그렇게 쓴다).

    Args:
        text: 사용자 입력.
        fallback: 판정 불가일 때 돌려줄 값. ``None`` 이면 판정 불가를 그대로 알린다.

    Returns:
        지원 언어 코드 또는 ``fallback``.
    """
    if not text:
        return fallback
    c = script_counts(text)
    if c["kana"]:
        return "ja"
    if c["hangul"] and c["hangul"] >= c["han"]:
        return "ko"
    if c["han"]:
        return "zh"
    if c["latin"] >= _MIN_LATIN_DETECT:
        return "en"
    return fallback


def looks_off_target(text: str, language: str = DEFAULT_LANGUAGE) -> bool:
    """답변이 목표 언어가 아닌지(다른 스크립트 위주인지) 결정적으로 판별한다.

    목표 스크립트 대 이질 스크립트 비율로 본다. 이질 문자가 적으면(고유명사·코드
    수준) 판정하지 않고, 목표 스크립트가 충분하면 통과다. ``language="ko"`` 일 때
    기존 ``looks_non_korean`` 과 같은 결론을 낸다(한자·가나까지 이질로 세는 것만 확장).

    Args:
        text: 최종 답변 텍스트.
        language: 이 턴의 목표 언어 코드.

    Returns:
        목표 언어가 아닌 답변으로 판단되면 True.
    """
    if not text:
        return False
    lang = normalize_language(language) or DEFAULT_LANGUAGE
    counts = script_counts(text)
    targets = _TARGET_SCRIPTS.get(lang, _TARGET_SCRIPTS[DEFAULT_LANGUAGE])
    target = sum(counts[s] for s in targets)
    foreign = sum(n for s, n in counts.items() if s not in targets)
    if foreign < _MIN_FOREIGN:
        return False  # 이질 문자가 적음 → 목표 언어(또는 판단 불가)로 간주
    total = target + foreign
    return total > 0 and (target / total) < _TARGET_RATIO_FLOOR


def detect_language_with_history(
    text: str,
    history_texts=(),
    *,
    fallback: str = DEFAULT_LANGUAGE,
) -> str:
    """현재 발화로 언어를 정하고, 판정 불가면 직전 발화들에서 물려받는다.

    "ok", "3번", "그럼 그거로" 같은 짧은 확인 발화가 대화 중간에 답변 언어를 바꾸는
    것을 막는다 — 영어로 대화하다 "ok" 한 마디에 한국어로 답이 돌아오면 그게 더 큰
    고장이다. 히스토리에서도 판정이 안 되면 ``fallback``.

    Args:
        text: 이번 턴의 사용자 발화.
        history_texts: 과거 사용자 발화들. **최근 것부터** 순회하므로 호출자가
            역순으로 넘기거나 역순 이터러블을 넘겨야 한다.
        fallback: 아무것도 판정되지 않을 때의 기본 언어.

    Returns:
        지원 언어 코드.
    """
    lang = detect_language(text, fallback=None)
    if lang:
        return lang
    for prev in history_texts:
        lang = detect_language(prev or "", fallback=None)
        if lang:
            return lang
    return fallback
