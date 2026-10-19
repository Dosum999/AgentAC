"""플랫폼 용어 사전 조회와 표기 교정.

플랫폼 관리자가 플랫폼 매니저(시스템 > 용어 사전)에서 용어의 한국어·영어·중국어 확정 표기와
**금지 오표기**를 관리한다. 에이전트는 그 값을 두 곳에 쓴다.

1. 프롬프트 주입 — **질문에 나온 용어만** "이 용어는 이렇게 적는다"를 알려 유도한다.
2. 결정적 교정 — 최종 답변에서 오표기를 확정 표기로 되돌린다(사전 전체 기준).
3. 번역 요청 검증 — "에이전트를 중국어로 번역하면?"처럼 질문이 용어의 다른 언어 표기를 물으면,
   답변에 그 표기가 실제로 들어 있는지 코드로 확인한다(없으면 재생성은 그래프가 한다).

1번을 질문 매칭으로 좁힌 이유: 사전 전체를 매 턴 싣으면 용어 수만큼 모든 턴의 프롬프트가
길어진다. 질문에 없는 용어가 답변에 나오는 경우는 2번이 사전 전체로 잡는다 — 사전에 적지
않은 엉뚱한 표기는 오표기 칸에 등록하는 것이 해법이다(2026-09-23 합의).

2번이 필요한 이유: 프롬프트에는 이미 "고유명사·제품명은 원어를 유지하라"가 세 언어 모두에
들어 있는데도 실측에서 "에이전트"가 영어로 ``Dotbee``, 중국어로 ``DotB`` 로 나왔다. 확률적
지시로는 새므로 마지막에 코드로 고정한다(name_answer_guard 와 같은 계통).

이 module이 소유하지 않는 것: 표기 정본(출처 시스템이 소유, extensions.glossary_source), 답변 언어 결정(orchestrator),
번역 자체. 여기는 조회·캐시·치환만 한다.

**조회 실패가 답변을 죽이지 않는다.** 미설정·타임아웃·오류는 전부 빈 사전으로 떨어지고
경고만 남는다 — 용어가 안 고쳐지는 것이 답변이 안 나오는 것보다 낫다.
"""
import logging
import re
from typing import Optional

from agent_arch.core.config import settings
from agent_arch.memory.redis_store import redis_store

logger = logging.getLogger(__name__)

_CACHE_KEY = "glossary_terms:v1"

# 코드 블록·인라인 코드·URL 은 교정 대상에서 제외한다. 코드 안의 식별자를 제품명으로
# 바꾸면 복사해 쓴 사람이 깨진 코드를 받는다.
_PROTECTED = re.compile(r"```.*?```|`[^`\n]*`|https?://\S+", re.DOTALL)
_ASCII_ONLY = re.compile(r"^[\x00-\x7F]+$")


def _spelling(term: dict, language: str) -> str:
    """목표 언어의 확정 표기. **비어 있으면 한국어 표기로 떨어진다.**

    이 폴백이 "어떤 언어에서도 원문 유지"를 겸한다 — 그 언어 칸에 한국어를 적거나 비워
    두면 된다. 예전에는 같은 일을 `do_not_translate` 스위치로도 할 수 있었는데, 켜면
    언어별 칸이 통째로 무시되는 것을 화면에서 알 수 없어 오해만 만들었다(2026-09-22).

    ``ja`` 는 사전에 열이 없어 영어 표기를 쓴다 — 로마자 제품명은 일본어 문장에서도 그대로 쓴다.
    """
    ko = str(term.get("term_ko") or "")
    if language == "ko":
        return ko
    if language == "zh":
        return str(term.get("term_zh") or "") or ko
    return str(term.get("term_en") or "") or ko


async def _fetch_terms() -> list[dict]:
    """등록된 용어 출처(extensions.glossary_source)에서 용어를 읽는다. 없거나 실패하면 빈 목록(경고만).

    출처가 돌려주는 항목 모양: ``{"term_ko", "term_en", "term_zh", "wrong_*": [...]}``.
    관리 화면과 **같은 목록**을 봐야 "화면엔 있는데 답변엔 안 먹는" 상태가 생기지 않으므로 필터하지 않는다.
    """
    from agent_arch.extensions import extensions

    if extensions.glossary_source is None:
        return []
    try:
        items = await extensions.glossary_source()
    except Exception:
        logger.warning("용어 사전 조회 실패 — 용어 없이 진행", exc_info=True)
        return []
    return [item for item in items or [] if isinstance(item, dict) and item.get("term_ko")]


async def get_glossary_terms() -> list[dict]:
    """용어 목록 (Redis 캐시).

    PA가 용어를 고치면 **캐시 TTL 만큼 뒤에** 반영된다. 답변마다 HTTP를 치지 않기 위한
    의도된 지연이다. PA 화면이 저장 후 그 시간만큼 `반영 중` 을 표시하므로, 두 값
    (`GLOSSARY_CACHE_TTL` ↔ 화면의 `GLOSSARY_SYNC_SECONDS`)은 같이 움직여야 한다.
    """
    cached = await redis_store.get(_CACHE_KEY)
    if isinstance(cached, list):
        return cached

    terms = await _fetch_terms()
    await redis_store.set(_CACHE_KEY, terms, expire=settings.glossary_cache_ttl)
    return terms


def _written(term: dict, column: str) -> str:
    """그 언어 칸에 **따로 적어 둔** 표기. 비었거나 한국어와 같으면 빈 문자열("원문 유지")."""
    ko = str(term.get("term_ko") or "").strip()
    value = str(term.get(f"term_{column}") or "").strip()
    return value if value and value != ko else ""


def match_terms(text: str, terms: list[dict]) -> list[dict]:
    """텍스트에 나온 용어만 고른다 — 한국어·영어·중국어 표기와 오표기 어느 것이든."""
    if not text or not terms:
        return []
    matched = []
    for term in terms:
        for spelling in _known_spellings(term):
            pattern = _alias_pattern(spelling)
            if pattern is not None and pattern.search(text):
                matched.append(term)
                break
    return matched


# 줄의 언어 이름. 답변 언어로 쓴다 — 지시 절 전체가 그 언어로 렌더되기 때문이다.
_LANGUAGE_LABELS = {
    "ko": {"en": "영어", "zh": "중국어"},
    "en": {"en": "English", "zh": "Chinese"},
    "zh": {"en": "英文", "zh": "中文"},
    "ja": {"en": "英語", "zh": "中国語"},
}


# 답변 언어 → 사전 열. 일본어는 열이 없어 영어 표기를 쓴다(`_spelling` 과 같은 규칙).
_ANSWER_COLUMN = {"en": "en", "ja": "en", "zh": "zh"}


def _columns_for(language: str, question: str) -> list[str]:
    """이 턴에 알려 줄 사전 열 — 질문이 물은 언어 + 답변 언어. 한국어 답변의 일반 질문은 없음.

    두 열을 늘 다 실었더니 "에이전트를 영어로 번역하면?"에 모델이 "참고로 중국어 표기는 …"을
    덧붙였다(2026-09-23 화면). 묻지 않은 언어는 보여 주지 않는 것이 가장 확실하다.
    """
    wanted = {c for c, pattern in _TRANSLATION_ASKS.items() if pattern.search(question or "")}
    if language in _ANSWER_COLUMN:
        wanted.add(_ANSWER_COLUMN[language])
    return [c for c in ("en", "zh") if c in wanted]


def render_glossary_lines(terms: list[dict], language: str, question: str = "") -> list[str]:
    """프롬프트에 넣을 "한국어 — 언어별 표기" 줄. 실을 열은 `_columns_for` 가 고른다.

    답변 언어 열만 싣던 때는 한국어로 "에이전트를 중국어로 번역하면?"을 물으면(답변은 한국어)
    한 줄도 안 실려 모델이 임의로 음역했다. 그래서 질문이 가리킨 언어 열을 더한다.
    호출자는 `match_terms` 로 좁힌 목록을 넘긴다.
    """
    labels = _LANGUAGE_LABELS.get(language, _LANGUAGE_LABELS["en"])
    columns = _columns_for(language, question)
    lines: list[str] = []
    for term in terms:
        ko = str(term.get("term_ko") or "").strip()
        pairs = [f"{labels[c]}: {_written(term, c)}" for c in columns if _written(term, c)]
        if ko and pairs:
            lines.append(f"- {ko} — " + " / ".join(pairs))
    return lines


# 질문이 "그 언어로 옮겨 달라"는 뜻인지 가르는 표현. 언어 이름만으로는 부족하다 —
# "에이전트 영어 지원돼?"는 번역 요청이 아니다. 조사("로")·번역 동사·"in/into" 가 붙어야 센다.
# "영어랑 중국어로"처럼 두 언어를 묶어 물으면 뒤 언어의 조사가 앞 언어까지 걸린다.
_KO_JOIN = r"(?:\s*(?:이랑|랑|와|과|나|하고|및|,|/)\s*(?:영어|영문|중국어|중문|한자))?"
_TRANSLATION_ASKS = {
    "en": re.compile(
        rf"(?:영어|영문){_KO_JOIN}\s*(?:으로|로|번역|표기|이름|명칭)"
        r"|\b(?:in|into|to)\s+English\b|\bEnglish\s+(?:name|spelling|translation)\b"
        r"|(?:翻译|译|翻譯|譯)成?\s*(?:英文|英语|英語)|用(?:英文|英语)|(?:英文|英语|英語)(?:名|怎么|怎麼)",
        re.IGNORECASE,
    ),
    "zh": re.compile(
        rf"(?:중국어|중문|한자){_KO_JOIN}\s*(?:으로|로|번역|표기|이름|명칭)"
        r"|\b(?:in|into|to)\s+Chinese\b|\bChinese\s+(?:name|spelling|translation)\b"
        r"|(?:翻译|译|翻譯|譯)成?\s*(?:中文|汉语|漢語)|用(?:中文|汉语)|(?:中文|汉语|漢語)(?:名|怎么|怎麼)",
        re.IGNORECASE,
    ),
}


def translation_targets(question: str, terms: list[dict]) -> list[tuple[str, str]]:
    """질문이 용어의 다른 언어 표기를 묻는다면, 답변에 반드시 들어가야 할 (한국어, 표기) 목록.

    질문에 나온 용어 x 질문이 가리킨 목표 언어. 그 언어 칸이 비어 있으면("원문 유지")
    검증할 표기가 없으므로 뺀다.
    """
    matched = match_terms(question, terms)
    if not matched:
        return []
    targets: list[tuple[str, str]] = []
    for column, pattern in _TRANSLATION_ASKS.items():
        if not pattern.search(question):
            continue
        for term in matched:
            spelling = _written(term, column)
            if spelling and (term["term_ko"], spelling) not in targets:
                targets.append((str(term["term_ko"]), spelling))
    return targets


def missing_spellings(answer: str, targets: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """답변에 빠진 목표 표기. 대소문자와 괄호 앞뒤 공백 차이는 같은 표기로 본다."""
    squashed = re.sub(r"\s+", "", answer or "").casefold()
    return [(ko, s) for ko, s in targets if re.sub(r"\s+", "", s).casefold() not in squashed]


async def question_translation_targets(question: str) -> list[tuple[str, str]]:
    """`translation_targets` 의 조회 포함판. 실패하면 빈 목록 — 검증을 못 해도 답변은 나간다."""
    if not question:
        return []
    try:
        return translation_targets(question, await get_glossary_terms())
    except Exception:
        logger.warning("[glossary] 번역 요청 판정 실패 — 검증 없이 진행", exc_info=True)
        return []


def _alias_pattern(alias: str) -> Optional[re.Pattern]:
    """오표기 하나를 찾을 정규식.

    영문 오표기는 단어 경계를 지킨다 — 경계를 안 보면 ``DotB`` 가 ``DotBridge`` 안까지
    바꾼다. 한글·한자는 단어 경계 개념이 없어(`\b` 가 뜻대로 동작하지 않는다) 그대로 찾는다.
    """
    alias = alias.strip()
    if not alias:
        return None
    escaped = re.escape(alias)
    if _ASCII_ONLY.match(alias):
        escaped = rf"\b{escaped}\b"
    try:
        return re.compile(escaped, re.IGNORECASE)
    except re.error:
        logger.warning("오표기 정규식 생성 실패: %r", alias)
        return None


def _known_spellings(term: dict) -> list[str]:
    """이 용어를 가리키는 모든 표기 — 언어별 확정 표기와 금지 오표기 전부."""
    out: list[str] = []
    for key in ("term_ko", "term_en", "term_zh"):
        value = str(term.get(key) or "").strip()
        if value and value not in out:
            out.append(value)
    for alias in term.get("aliases") or []:
        value = str(alias).strip()
        if value and value not in out:
            out.append(value)
    return out


def _rules_for(term: dict, language: str) -> list[tuple[re.Pattern, str]]:
    """한 용어를 확정 표기로 되돌릴 규칙을 **적용 순서대로** 만든다.

    순서가 중요하다. 실측에서 모델은 "원문 유지" 지시와 용어표를 둘 다 만족시키려고
    ``에이전트(Agent)`` 처럼 **병기**했다. 병기를 먼저 통째로 걷어내지 않고 한국어 표기만
    바꾸면 ``Agent(达特维)(Agent)`` 가 된다.

    1. 병기 — ``에이전트(Agent)`` · ``Agent(에이전트)``
    2. 금지 오표기 — ``Dotbee``
    3. 한국어 표기 단독 — ``에이전트``
    """
    ko = str(term.get("term_ko") or "").strip()
    target = _spelling(term, language)
    if not ko or not target:
        return []

    rules: list[tuple[re.Pattern, str]] = []
    others = [spelling for spelling in _known_spellings(term) if spelling != ko]
    if others:
        # 긴 표기를 먼저 시도해야 ``Agent(达特维)`` 가 ``Agent`` 에 먼저 걸려 잘리지 않는다.
        alt = "|".join(re.escape(s) for s in sorted(others, key=len, reverse=True))
        ko_escaped = re.escape(ko)
        rules.append(
            (re.compile(rf"{ko_escaped}\s*[(（]\s*(?:{alt})\s*[)）]", re.IGNORECASE), target)
        )
        rules.append(
            (re.compile(rf"(?:{alt})\s*[(（]\s*{ko_escaped}\s*[)）]", re.IGNORECASE), target)
        )

    for alias in term.get("aliases") or []:
        pattern = _alias_pattern(str(alias))
        if pattern is not None:
            rules.append((pattern, target))

    # 한국어 표기 단독. 목표 표기가 한국어 그대로거나(번역 금지·한국어 답변) 목표 안에
    # 한국어가 들어 있으면 바꿀 것이 없다.
    if target != ko and ko not in target:
        rules.append((re.compile(re.escape(ko)), target))
    return rules


def correct_spellings(text: str, terms: list[dict], language: str) -> str:
    """최종 답변의 표기를 사전의 확정 표기로 되돌린다.

    두 가지를 잡는다.

    - **한국어 원문을 남긴 경우** (``我是에이전트``, ``에이전트(Agent)``) — 생성 경로의 실제 실패
      유형이다. 프롬프트가 "제품명은 원문 유지"를 함께 시키고 있어 모델이 절충한다.
    - **엉뚱하게 옮긴 경우** (``Dotbee``) — 사전에 적어 둔 오표기만 바꾼다. "닮은 표기"를
      추측해 넓히지 않는다.

    코드 블록·인라인 코드·URL 안은 건드리지 않는다.
    """
    if not text or not terms:
        return text

    rules: list[tuple[re.Pattern, str]] = []
    for term in terms:
        rules.extend(_rules_for(term, language))
    if not rules:
        return text

    def _fix(segment: str) -> str:
        for pattern, target in rules:
            segment = pattern.sub(target, segment)
        return segment

    out: list[str] = []
    cursor = 0
    for match in _PROTECTED.finditer(text):
        out.append(_fix(text[cursor : match.start()]))
        out.append(match.group(0))
        cursor = match.end()
    out.append(_fix(text[cursor:]))
    return "".join(out)


async def enforce_glossary(content: str, language: str) -> str:
    """최종 답변에 사전을 적용한다 — 답변을 내보내는 모든 경로가 이 함수를 부른다.

    한국어 답변에는 적용하지 않는다. 한국어 표기가 곧 확정 표기라 바꿀 것이 없고,
    `에이전트(Agent)` 같은 병기도 한국어 문장에서는 자연스럽다. 그래야 한국어 턴은 스트리밍을
    붙들 이유가 없어진다(`hold_stream_for_glossary` 참조).

    조회·치환 실패는 원문을 그대로 둔다. 용어가 안 고쳐지는 것이 답변이 안 나오는 것보다 낫다.
    """
    if not content or language == "ko":
        return content
    try:
        return correct_spellings(content, await get_glossary_terms(), language)
    except Exception:
        logger.warning("[glossary] 표기 교정 실패 → 원문 유지", exc_info=True)
        return content


async def hold_stream_for_glossary(language: str, question: str = "") -> bool:
    """이 턴은 토큰 스트리밍을 붙들어야 하는가.

    교정은 답변이 끝난 뒤에 돈다. 그런데 SSE 에는 본문 교체 이벤트가 없어서, 토큰을 그대로
    흘려보내면 **저장본만** 고쳐지고 화면에는 원문이 남는다(2026-09-22 실측: 화면
    `에이전트 (Agent)`, 저장본 `Agent(达特维)`).

    스트리밍 중에 조금씩 고치는 방법도 시도했지만, 용어가 청크 경계에 걸릴 때 단어 경계·코드
    구간·URL 판정이 전체 문맥과 달라져 화면과 저장본이 미묘하게 갈렸다. 그래서 이름 환각
    가드가 쓰는 검증된 경로(`hold_all`)를 그대로 쓴다 — 붙들었다가 교정본을 한 번에 내보내면
    화면과 저장본이 **같을 수밖에 없다**.

    붙드는 대가는 "답변이 한 번에 뜬다"이므로 필요한 턴에만 켠다. 한국어 답변은 교정이
    no-op 이고, 사전이 비어 있으면 고칠 것 자체가 없다. 단 한국어 답변이라도 **번역 요청**
    턴은 붙든다 — 표기가 빠지면 재생성해야 하므로 먼저 흘려보내면 안 된다.
    """
    if language == "ko":
        return bool(await question_translation_targets(question))
    try:
        return bool(await get_glossary_terms())
    except Exception:
        logger.warning("[glossary] 사전 조회 실패 — 스트리밍 유지", exc_info=True)
        return False
