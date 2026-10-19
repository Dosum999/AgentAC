"""LLM 응답을 사용자 표시용으로 정화하는 유틸리티.

세 가지를 결정적으로 처리한다 (모두 `sanitize_tool_names` 한 곳을 거치므로
스트리밍/최종 경로 모두 자동 적용):

1) **내부 도구 이름 마스킹** — `mespoc_service_list_defective_*_53aeef98`
   같은 내부 함수명이 본문에 누출되는 것을 차단.
    1차: 화이트리스트 alternation — 실제 도구명(동적 API + 정적 스킬 + MCP +
         tools_config.yaml)을 escape하여 OR 결합한 정규식. 100% 정확.
    2차: 안전망 패턴 — `*_(service|skill)_*` 형태의 snake_case 식별자를
         포착. 신규 도구·미캐시 케이스 대응.

2) **LaTeX/수식 → 유니코드 정규화**(`normalize_math_markup`) — 프론트 마크다운
   렌더러가 수식을 못 그려 `$\\rightarrow$` 가 생으로 노출되던 문제를 백엔드에서
   결정적으로 차단. `\\rightarrow`→`→`, `\\times`→`×` 등으로 치환하고 `$...$`·
   `\\(...\\)`·`$$...$$` 래퍼를 벗긴다. 통화 오탐 방지를 위해 인라인 `$...$`는
   내부에 수식 표시(`\\`·`^`·`_`)가 있을 때만 변환한다(`$100 와 $200`은 불변).
   (프롬프트 OUTPUT_FORMAT_RULES 가 1차 억제, 이 가드가 최종 보장.)

3) **HTML 줄바꿈 → 공백 정규화** — 렌더러가 그대로 노출하는 `<br>` 계열을
   공백으로 바꾼다. 마크다운 표 안에서도 행·링크 구조가 유지된다.

4) **가이드 위키 주소 차단**(`strip_wiki_links`) — 가이드 KB 원문(Confluence,
   `atlassian.net`)은 일반 사용자가 못 여는 내부 주소다. `[라벨](위키 URL)` 은 글자만
   남기고 생 URL 은 지운다. 도구 직렬화(guide_kb)·인제스션(html_normalizer)도 같은
   판정(`is_wiki_url`)을 쓰며, 여기는 그 둘을 뚫고 나온 것까지 화면 직전에 막는 안전망.

스트리밍 모드에서는 `StreamSanitizer`를 사용해 lookbehind 버퍼링으로
토큰 경계 분할 누락을 방지한다.
"""
from __future__ import annotations

import logging
import re
from typing import Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# 치환 문자열: LLM 문장 흐름을 깨지 않도록 자연스러운 한국어 명사구로 치환
_REPLACEMENT = "내부 도구"

# 가이드 모드 전용 내부 마커. 가이드 LLM이 "실제 데이터 조회/작업이 필요해 가이드로는
# 답할 수 없는 요청"이라고 판단하면 응답 맨 끝에 이 마커를 그대로 출력한다.
# guide_loop 가 이를 감지해 suggested_mode="agent"(AI 에이전트 모드 전환 제안) 신호로
# 승격하고, 아래 sanitize_tool_names 가 사용자 표시 텍스트에서 제거한다. 스트리밍 경로는
# 마커가 항상 응답 끝(마지막 _LOOKBEHIND 버퍼)에 있어 flush 단계에서 제거되므로 노출되지
# 않는다. 프롬프트(GUIDE_SYSTEM_PROMPT)·guide_loop 가 이 상수를 단일 출처로 공유한다.
GUIDE_AGENT_HINT = "⟦NEED_AGENT⟧"

# 화이트리스트 alternation (런타임 빌드, lazy)
_WHITELIST_RE: Optional[re.Pattern] = None

# 화이트리스트에서 **반드시 제외**할 이름의 탈출구.
# 1차 필터(언더스코어 없는 이름 배제)를 통과하지만 일반 표현과 충돌하는 도구명을 여기 넣는다.
# 2026-08 실측 기준 확장 소스(동적 16 + 레지스트리 35 + yaml 82 → 중복 제거 98)에서 언더스코어
# 없는 이름은 search / calculator / database 3개뿐이고 1차 필터가 모두 걸러내므로 지금은 비어 있다.
# 참고: 동적 소스가 16개인 것은 tool_loader 가 llm_tools.json 233개 중 enabled+is_external 만
# 캐시하기 때문이다. 확장 전 화이트리스트가 딱 이 16개였다 — 구멍이 그만큼 컸다.
# MCP 서버가 앞으로 `list_all` 같은 이름을 등록하면 정규식을 고치지 말고 여기 한 줄 추가할 것.
_NEVER_MASK: frozenset = frozenset()

# 도구명을 구성하는 문자 — 화이트리스트 정규식의 경계 클래스 [A-Za-z0-9_] 와 동일해야 한다.
# StreamSanitizer 가 emit 경계를 이 런 중간에 두지 않도록 판정하는 데 쓴다.
_IDENT_CHAR = re.compile(r"[A-Za-z0-9_]")

# ── 치환 잔재 흡수 ────────────────────────────────────────────────────────────
# 치환은 이름 단위로만 일어나는데 모델은 "식별자 + 분류어" 구조로 문장을 쓴다. 그래서
# `mail_read_message` 도구를 → `내부 도구` 도구를 처럼 치환어가 이미 품은 "도구"가
# 원문의 "도구"와 겹치고, 감싼 백틱도 남는다. 아래 규칙들이 **치환이 실제로 일어난 자리
# 인접**에서만 그 잔재를 흡수한다. 전역 정규식으로 돌리면 정상 문장의 "도구"·괄호까지
# 먹으므로, 전부 _REPLACEMENT 리터럴에 앵커링한다.
_R = re.escape(_REPLACEMENT)

# 분류어 뒤에 붙을 수 있는 조사. 긴 것부터 시도해야 "으로"가 "로"에 가려지지 않는다.
# 이 목록에 없는 것(예: "들")이 오면 분류어가 아니라 다른 단어의 일부이므로 흡수하지 않는다
# ("내부 도구 도구들을" 은 건드리지 않는다).
_PARTICLES = (
    "으로", "에서", "에게", "부터", "까지", "처럼", "마다", "조차", "밖에",
    "를", "을", "로", "가", "이", "는", "은", "와", "과", "의", "도", "만", "에", "나", "랑",
)
# 치환어를 감싼 백틱 — 한국어 명사구를 코드 스팬에 둘 이유가 없다.
_ABSORB_BACKTICK = re.compile(r"`" + _R + r"`")
# 뒤따르는 중복 분류어. _REPLACEMENT 가 "도구"로 끝나는 것에 의존하므로 치환어를 바꿀 때
# 이 규칙의 유효성을 함께 재검토할 것.
_ABSORB_CLASSIFIER = re.compile(
    _R + r"[ \t]*(?:도구|툴|함수|API)"
    r"(?=(?:" + "|".join(_PARTICLES) + r")?(?![가-힣]))"
)
# 구분자로 나란히 놓인 같은 치환어 (`A 또는 B` 가 둘 다 마스킹돼 같은 말이 된 경우)
_ABSORB_ADJACENT = re.compile(r"" + _R + r"(?:[ \t]*(?:또는|및|,|/)[ \t]*" + _R + r")+")
# 괄호 안이 치환어(+구분자)뿐이면 괄호째 버린다. 산문이 한 글자라도 섞이면 매칭되지 않는다.
# 앞의 공백 한 칸까지 흡수해 "도구 (내부 도구)를" → "도구를" 가 되게 한다.
_ABSORB_PAREN = re.compile(
    r"[ \t]?[(（][ \t]*" + _R + r"(?:[ \t]*(?:또는|및|,|/)[ \t]*" + _R + r")*[ \t]*[)）]"
)

# 흡수 규칙이 봐야 하는 "매치 뒤쪽" 문맥 창(자). 백틱 + 공백 + 분류어 + 조사 + 닫는 괄호를
# 덮는다. StreamSanitizer 가 이 창을 쪼개면 흡수가 발동하지 못한다(_safe_emit_len 참조).
_ABSORB_WINDOW = 24


def _absorb_redundancy(text: str) -> str:
    """치환으로 생긴 중복 표현을 흡수한다.

    적용 순서가 중요하다: 백틱을 먼저 벗겨야 괄호·중복 규칙이 치환어를 알아보고,
    괄호를 먼저 버려야 그 안의 중복을 따로 손볼 필요가 없다.

    Args:
        text: 도구명 치환이 끝난 텍스트.

    Returns:
        중복 분류어·백틱·괄호가 정리된 텍스트.
    """
    text = _ABSORB_BACKTICK.sub(_REPLACEMENT, text)
    text = _ABSORB_PAREN.sub("", text)
    text = _ABSORB_ADJACENT.sub(_REPLACEMENT, text)
    text = _ABSORB_CLASSIFIER.sub(_REPLACEMENT, text)
    return text

# 안전망 패턴: ASCII 단어 경계 안에서 `*_(service|skill)_*` 형태를 잡는다.
# - 한국어 옆에 박혀도 ASCII 단어 경계가 적용되어 정확히 매칭됨
# - hex 접미사는 옵션화 (hex 없는 `hr_service_attendance_read` 류도 커버)
_PATTERN_FALLBACK = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"[a-z][a-z0-9]*_(?:service|skill)_[a-z0-9_]+"
    r"(?:_[0-9a-f]{8})?"
    r"(?![A-Za-z0-9_])"
)

# ── LaTeX/수식 → 유니코드 정규화 ──────────────────────────────────────────────
# LaTeX 명령(letter) → 유니코드. 긴 이름부터 매칭해야 `\leq`가 `\le`보다,
# `\rightarrow`가 `\right`보다 먼저 잡힌다 (아래 정렬·치환 순서로 보장).
_LATEX_TO_UNICODE = {
    "longleftrightarrow": "↔", "leftrightarrow": "↔", "Leftrightarrow": "⇔",
    "longrightarrow": "→", "longleftarrow": "←",
    "rightarrow": "→", "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐",
    "uparrow": "↑", "downarrow": "↓", "to": "→", "gets": "←", "mapsto": "↦",
    "times": "×", "div": "÷", "cdot": "·", "bullet": "•", "ast": "∗",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "sim": "∼", "propto": "∝",
    "pm": "±", "mp": "∓", "infty": "∞", "partial": "∂", "nabla": "∇",
    "sum": "∑", "prod": "∏", "int": "∫", "sqrt": "√",
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "theta": "θ", "lambda": "λ", "mu": "μ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "phi": "φ", "omega": "ω",
    "Delta": "Δ", "Sigma": "Σ", "Omega": "Ω",
    "circ": "∘", "degree": "°", "ldots": "…", "cdots": "⋯", "dots": "…",
}
# 긴 명령부터 alternation. `(?![a-zA-Z])`로 `\to`가 `\theta`/`\total`을 오염시키지 않게 한다.
_LETTER_CMD_RE = re.compile(
    r"\\(" + "|".join(sorted(_LATEX_TO_UNICODE, key=len, reverse=True)) + r")(?![a-zA-Z])"
)
# 기호/공백/이스케이프 명령(backslash+구두점 위주 — letter 명령의 접두가 아니라
# 경계에서 쪼개져도 다른 명령을 오염시키지 않는다). `\left`/`\right`는 일부러 제외:
# `\rightarrow`의 접두 `\right`를 bare로 지우면 스트리밍 토큰 경계 분할 시 화살표가
# 깨질 수 있다. 드문 `\left(`·`\right)`는 평문으로 두는 편이 안전하다.
_SYMBOL_CMDS = {
    r"\quad": "  ", r"\qquad": "    ", r"\,": " ", r"\;": " ", r"\:": " ", r"\!": "",
    r"\%": "%", r"\$": "$", r"\&": "&", r"\#": "#", r"\_": "_", r"\{": "{", r"\}": "}",
}
# 인라인 `$...$`가 통화가 아니라 수식임을 가리는 표시.
_MATH_INDICATOR = re.compile(r"[\\^_]")

# HTML을 렌더링하지 않는 화면용 줄바꿈 보정. 실제 개행으로 바꾸면 마크다운
# 표의 셀 하나가 여러 행으로 갈라지므로 항상 공백을 쓴다. 전체 HTML unescape는
# 링크 URL 등을 바꿀 수 있어 하지 않고, br 및 그 escape 표기만 처리한다.
_HTML_BREAK_RE = re.compile(
    r"(?:\\?<|&lt;)br[ \t]*/?[ \t]*(?:\\?>|&gt;)", re.IGNORECASE
)


# 같은 문자가 비정상적으로 길게 이어지는 런(run) — 모델 반복 루프의 전형적 흔적.
# 2026-09-15 실측: gemini 가 마크다운 표 구분선 `| :---- |` 를 쓰다 루프에 빠져 대시만
# max_tokens 까지 124,956자를 뿜었고, 30,000자 trim 이 표 본문까지 잘라 "(응답이 너무 길어
# 일부만 표시됩니다.)" 만 남았다. 정상 문서에서 같은 문자가 120자 이상 이어지는 경우는
# 사실상 없으므로(가장 긴 정상 구분선·밑줄도 수십 자), 그 이상이면 루프로 보고 3자로 접는다.
# 3자를 남기는 이유: `---`(표 구분선·수평선)·`===`(setext 제목) 은 3자로도 마크다운 문법이
# 유지된다. 코드 블록 안까지 일괄 적용한다 — 120자 동일 문자 런은 코드에서도 정상이 아니다.
_REPEAT_RUN_THRESHOLD = 120
_REPEAT_RUN_KEEP = 3
_REPEAT_RUN_RE = re.compile(r"(.)\1{%d,}" % (_REPEAT_RUN_THRESHOLD - 1), re.DOTALL)


def collapse_repeated_runs(text: str) -> str:
    """같은 문자가 `_REPEAT_RUN_THRESHOLD`자 이상 이어지는 런을 `_REPEAT_RUN_KEEP`자로 접는다.

    oversize trim(30,000자) **앞에서** 호출해야 의미가 있다 — 루프 잔해를 먼저 걷어내야
    실제 본문 길이로 trim 여부가 결정되고, 표 본문 등 정상 내용이 살아남는다.
    반복이 없으면 원문을 그대로 반환한다(빈 문자열·None 포함).
    """
    if not text or len(text) < _REPEAT_RUN_THRESHOLD:
        return text
    collapsed, n = _REPEAT_RUN_RE.subn(lambda m: m.group(1) * _REPEAT_RUN_KEEP, text)
    if n:
        logger.warning(
            "[sanitize] 반복 문자 런 %d개 접음 (%d → %d chars) — 모델 반복 루프 의심",
            n, len(text), len(collapsed),
        )
    return collapsed


def normalize_html_breaks(text: str) -> str:
    """표·링크 구조를 유지하면서 HTML 줄바꿈 표기를 공백으로 바꾼다."""
    if not text:
        return text
    return _HTML_BREAK_RE.sub(" ", text)


def _convert_latex_commands(s: str) -> str:
    """수식 내용의 LaTeX 명령을 유니코드/평문으로 바꾼다."""
    s = _LETTER_CMD_RE.sub(lambda m: _LATEX_TO_UNICODE[m.group(1)], s)
    for k, v in _SYMBOL_CMDS.items():
        s = s.replace(k, v)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)  # \frac{a}{b} → (a)/(b)
    s = re.sub(r"([\^_])\{([^{}]*)\}", r"\1\2", s)                        # x^{2} → x^2
    return s


def normalize_math_markup(text: str) -> str:
    """모델이 출력한 LaTeX/수식을 프론트가 그대로 그릴 수 있는 평문으로 정규화한다.

    프론트 마크다운 렌더러가 LaTeX 를 지원하지 않아 `$\\rightarrow$` 같은 출력이
    생으로 노출되던 문제를, 렌더러 도입 없이 백엔드에서 결정적으로 해소한다.

    처리:
        - `$$...$$`·`\\[...\\]`·`\\(...\\)` 래퍼 제거 + 내부 명령 변환
        - 인라인 `$...$`는 내부에 수식 표시(`\\`·`^`·`_`)가 있을 때만 변환
          → `$100 와 $200` 같은 통화 표기는 건드리지 않는다
        - 래퍼 밖 bare 명령(`\\rightarrow` 등)도 변환

    Args:
        text: LLM 응답 본문 (None/빈 문자열/`$`·`\\` 없으면 그대로 반환)

    Returns:
        수식이 유니코드 평문으로 치환된 문자열
    """
    if not text or ("$" not in text and "\\" not in text):
        return text
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: _convert_latex_commands(m.group(1)), text, flags=re.S)
    text = re.sub(r"\\\[(.+?)\\\]", lambda m: _convert_latex_commands(m.group(1)), text, flags=re.S)
    text = re.sub(r"\\\((.+?)\\\)", lambda m: _convert_latex_commands(m.group(1)), text, flags=re.S)

    def _inline(m: "re.Match") -> str:
        inner = m.group(1)
        return _convert_latex_commands(inner) if _MATH_INDICATOR.search(inner) else m.group(0)

    text = re.sub(r"\$([^\n$]+?)\$", _inline, text)
    return _convert_latex_commands(text)


def _maskable(name: str) -> bool:
    """이 이름을 화이트리스트에 넣어도 안전한지 판단한다.

    언더스코어가 없는 이름(`search`, `calculator`, `database`)은 **제외한다**. 도구명이자
    동시에 평범한 영단어라, 화이트리스트에 넣으면 답변 본문에 등장하는 모든 "search" ·
    "database" 가 "내부 도구"로 치환돼 정상 문장이 깨진다. 도구명 노출을 막으려다 답변
    품질을 망가뜨리는 쪽이 더 나쁘다는 판단 — 이 이름들은 프롬프트 금지 규칙에만 맡긴다.
    """
    return bool(name) and "_" in name and name not in _NEVER_MASK


def _collect_tool_names() -> set:
    """마스킹 대상 도구 이름을 소스 3종에서 모은다.

    - ``dynamic``  : ``tool_loader._cache`` — llm_tools.json 기반 동적 API 도구
    - ``registry`` : ``ToolRegistry`` — 정적 스킬 + MCP 원격 도구(런타임 등록)
    - ``config``   : ``tools_config.yaml`` 의 tools 키

    ``config`` 가 중요하다. MCP 등록은 부팅 후 백그라운드 + 백오프 재시도이고 재시도가
    소진되면 그 프로세스 생애 동안 등록이 안 된다. 그래도 LLM 은 프롬프트·과거 대화 맥락에서
    배운 이름을 뱉을 수 있으므로, 파일 기반 소스가 그 구멍을 막는다.

    소스별로 예외를 **독립 처리**한다 — 하나가 터져도 나머지로 마스킹이 살아남게. 셋을
    한 try 로 묶으면 소스 하나의 실패가 마스킹 전체를 무력화한다.

    Returns:
        set: 마스킹 대상 도구 이름 집합 (``_maskable`` 필터 통과분)
    """
    # 순환 임포트 회피를 위해 함수 내부에서 임포트한다. registry 는 register() 안에서
    # 이 모듈의 reset_whitelist_cache() 를 부르므로 **상호 의존**이다 — 어느 한쪽이라도
    # 모듈 레벨 import 로 올리면 부팅 시 ImportError 가 난다.
    from agent_arch.core.config import settings

    sources = {
        part.strip()
        for part in (settings.sanitize_tool_name_sources or "").split(",")
        if part.strip()
    }
    raw: set = set()
    counts: dict = {}

    if "dynamic" in sources:
        try:
            from agent_arch.services.api_registry.loader import tool_loader

            if tool_loader._cache is None:
                tool_loader.load_from_json()
            got = set((tool_loader._cache or {}).keys())
        except Exception as e:
            logger.warning("[sanitize] 동적 도구 캐시 로드 실패 — 해당 소스 제외: %s", e)
            got = set()
        counts["dynamic"] = len(got)
        raw |= got

    if "registry" in sources:
        try:
            # 주의: 서브모듈 import 라도 파이썬이 부모 패키지 app.agents.tools 의 __init__.py 를
            # 먼저 실행하므로, 이 한 줄이 전 정적 스킬을 import·등록시킨다(실측: 이 import 직후
            # list_tool_names() == 35). 즉 "패키지 대신 모듈을 import 해서 부수효과를 피한다"는
            # 불가능하다. 운영에서는 main.py 가 부팅 시 이미 import 하므로 첫 요청 시점엔 로드
            # 완료 상태고, 아직이면 첫 sanitize 호출이 그 import 를 동기로 끌고 온다.
            # 재진입은 안전하다 — 등록마다 reset_whitelist_cache() 가 불려도 그건 이 수집
            # **도중**이고, _WHITELIST_RE 최종 대입은 수집이 끝난 뒤라 스냅샷이 유실되지 않는다.
            from agent_arch.agents.tools.registry import ToolRegistry

            got = set(ToolRegistry.list_tool_names())
        except Exception as e:
            logger.warning("[sanitize] 레지스트리 조회 실패 — 해당 소스 제외: %s", e)
            got = set()
        counts["registry"] = len(got)
        raw |= got

    if "config" in sources:
        try:
            got = set((settings.tools_config or {}).get("tools", {}).keys())
        except Exception as e:
            logger.warning("[sanitize] tools_config 조회 실패 — 해당 소스 제외: %s", e)
            got = set()
        counts["config"] = len(got)
        raw |= got

    names = {n for n in raw if _maskable(n)}
    # 부팅 로그로 소스별 커버리지를 드러낸다. MCP 등록 타이밍 문제(등록 전 스냅샷이 굳는
    # 현상)가 재발하면 registry 카운트가 0/작은 값으로 남는 것으로 즉시 진단된다.
    logger.info(
        "[sanitize] 화이트리스트 빌드: %s → %d개(제외 %d)",
        " ".join(f"{k}={v}" for k, v in counts.items()) or "소스 없음",
        len(names),
        len(raw) - len(names),
    )
    return names


def _build_whitelist() -> re.Pattern:
    """수집된 도구 이름으로 화이트리스트 정규식을 빌드한다.

    가장 긴 이름부터 매칭하도록 길이 내림차순으로 alternation을 구성한다.

    Returns:
        실제 도구명 alternation 정규식. 이름이 하나도 없으면 매칭 안 되는 패턴 반환.
    """
    global _WHITELIST_RE

    names = sorted(_collect_tool_names(), key=len, reverse=True)
    if not names:
        # 빈 alternation은 모든 입력을 매칭하므로, 절대 매칭되지 않는 패턴 사용.
        # 운영에서는 config 소스(파일 기반)가 항상 살아 있어 도달하지 않는다 — 소스를
        # 전부 끈 경우나 테스트 격리용 경로다.
        _WHITELIST_RE = re.compile(r"(?!)")
        return _WHITELIST_RE

    # 길이 내림차순: prefix가 다른 도구 이름의 시작과 충돌하는 것을 방지
    pattern = (
        r"(?<![A-Za-z0-9_])"
        r"(?:" + "|".join(re.escape(n) for n in names) + r")"
        r"(?![A-Za-z0-9_])"
    )
    _WHITELIST_RE = re.compile(pattern)
    return _WHITELIST_RE


def reset_whitelist_cache() -> None:
    """화이트리스트 정규식 캐시를 초기화한다.

    도구 정의가 런타임에 변경되었을 때 호출한다. ``ToolRegistry.register/unregister/clear``
    가 매번 호출하므로(→ registry.py), MCP 원격 도구가 부팅 후 뒤늦게 등록돼도 다음 요청
    때 화이트리스트가 다시 만들어진다. 이 훅이 없으면 lazy 빌드된 첫 스냅샷이 프로세스
    생애 동안 고정돼, MCP 등록 전에 들어온 첫 요청 하나가 그 프로세스 내내 메일·일정
    도구명을 못 걸리게 만든다(2026-08 실제 누출 원인).
    """
    global _WHITELIST_RE
    _WHITELIST_RE = None


# URL 본체 — 공백·괄호·꺾쇠·따옴표·백틱 앞에서 끝난다. 마크다운 링크의 닫는 `)` 도 여기서
# 끊긴다. 아래 두 절(내부 식별자 차단·가이드 위키 주소 차단)이 같은 정의를 쓴다.
_URL_BODY = r"https?://[^\s<>()\[\]\"'`]+"


# ── 내부 식별자 노출 차단 ────────────────────────────────────────────────────
# Mattermost 의 post/channel/file id 는 **소문자+숫자 26자**다. 도구가 그 값을 결과로
# 주는데(파일을 지목하려면 필요하다), 모델이 그걸 답변 본문에 그대로 옮겨 적는다.
# 실측(2026-09-15): "에이전트_요금제_안내.xlsx (file_id: 8paerwxuyb8jfch8z8o14xe97o)".
# 사용자에게는 아무 의미가 없고 파일명만 가린다. **모델은 도구 인자로 계속 쓰되 화면에서만**
# 지운다 — 도구 호출 인자는 이 함수를 거치지 않으므로 기능에 영향이 없다.
#
# 26자 고정이라 한국어·영단어와 겹치지 않는다. 다만 순수 숫자(주문번호 등)나 대문자가
# 섞인 값은 건드리지 않게 **소문자+숫자 혼합**만 잡는다(둘 다 최소 1자 포함).
#
# ⚠️ **URL 안의 id 는 예외다.** 메신저 딥링크는 `…/{team}/pl/{post_id}` 라 id 가 곧 주소의
# 목적지다. 여기를 지우면 `https://chat.example.com/team/pl/` 만 남아 **파란 링크로
# 보이지만 눌러도 아무 데도 가지 않는다** — 화면상 "링크가 잘린" 것처럼 보인다.
# 실측(2026-09-16): 사용자가 "메시지 바로가기"를 눌러도 반응이 없었고, 그 답변을 지어낸
# 링크로 오판해 `answer_guard.redact_fabricated_permalinks` 를 넣었는데도 증상이 그대로였다
# (가드는 정화 **전에** 돌아 온전한 26자 URL 을 보고 통과시키고, 그 뒤 여기서 깨뜨렸다).
# 파일 다운로드 주소(`/api/v4/files/{file_id}`)도 같은 이유로 보존한다.
# 노출 차단의 대상은 어디까지나 **본문에 맨몸으로 적힌 id**(`file_id: 8paerw…`)다.
_MM_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?=[a-z0-9]{26}(?![A-Za-z0-9]))(?=[a-z0-9]*[a-z])(?=[a-z0-9]*[0-9])[a-z0-9]{26}"
)
# 주소로 보이는 스팬 — 이 안의 id 는 지우지 않는다. 스킴이 있는 URL 뿐 아니라 스킴을 생략한
# 표기(`chat.example.com/…`)도 렌더러가 링크로 만들므로 함께 본다.
_URL_SPAN_RE = re.compile(
    _URL_BODY
    + r"|(?<![A-Za-z0-9@.])[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}/[^\s<>()\[\]\"'`]*"
)
# id 를 지운 뒤 **값이 통째로 빈 줄**은 줄 자체를 없앤다. 실측(2026-09-15): 목록 항목
# "- **파일 ID**: ``" 가 백틱만 남은 채 화면에 남았다 — 사용자에게는 빈 라벨일 뿐이다.
_ID_ONLY_LINE = re.compile(
    r"^[ \t]*(?:[-*+]|\d+\.)?[ \t]*\**[ \t]*"
    r"(?:파일\s*)?(?:ID|아이디|id|file[_ ]?id|post[_ ]?id|channel[_ ]?id|root[_ ]?id)"
    r"[ \t]*\**[ \t]*[:：][ \t]*[`'\"\u201c\u201d]*[ \t]*$\n?",
    re.MULTILINE | re.IGNORECASE,
)

# id 를 지우고 남는 껍데기: "(file_id: )" · "(post_id: , ...)" 같은 빈 라벨 조각.
_ID_LABEL_LEFTOVER = re.compile(
    r"[\(\[]\s*(?:file_id|post_id|channel_id|root_id|id)\s*[:=]?\s*[\)\]]"
    r"|\s*[,·]?\s*(?:file_id|post_id|channel_id|root_id)\s*[:=]\s*(?=[\)\]\n,]|$)"
)


def strip_internal_ids(text: str) -> str:
    """답변 본문에서 내부 식별자(Mattermost 26자 id)를 지운다.

    **주소 안의 id 는 남긴다** — 메신저 딥링크(`…/pl/{post_id}`)·파일 주소
    (`…/api/v4/files/{file_id}`)에서 id 는 목적지 그 자체라, 지우면 눌러도 아무 데도
    가지 않는 링크가 된다. 맨몸으로 적힌 id(`file_id: 8paerw…`)만 지운다.

    Args:
        text: LLM 응답 본문.

    Returns:
        id 와 그로 인해 남는 빈 라벨 조각이 제거된 문자열. 주소 안의 id 는 그대로.
    """
    if not text:
        return text
    spans = [m.span() for m in _URL_SPAN_RE.finditer(text)]

    def _drop_unless_in_url(m: re.Match) -> str:
        # 스킴도 호스트도 없는 경로(`[보기](/team/pl/{id})`)와, 스트리밍에서 URL 이
        # head 로 쪼개져 앞부분이 이번 조각에 없는 경우까지 덮으려고 바로 앞의 `/` 도 본다.
        if m.start() > 0 and text[m.start() - 1] == "/":
            return m.group(0)
        return m.group(0) if any(s <= m.start() < e for s, e in spans) else ""

    out = _MM_ID_RE.sub(_drop_unless_in_url, text)
    if out == text:
        return text
    out = _ID_ONLY_LINE.sub("", out)
    out = _ID_LABEL_LEFTOVER.sub("", out)
    # "파일명 ()" 처럼 빈 괄호만 남는 경우 정리.
    out = re.sub(r"\s*[\(\[]\s*[\)\]]", "", out)
    return re.sub(r"[ \t]{2,}", " ", out)


# ── 가이드 위키(Confluence) 주소 차단 ────────────────────────────────────────
# 가이드 KB 원문은 Confluence 위키(example.atlassian.net)다. 일반 사용자는 접근 권한이 없어
# 그 주소가 답변에 나가면 눌러도 열리지 않는 링크만 남는다. 프롬프트로 "page_url 은 노출
# 금지"를 걸어 두었지만 실측(2026-09-16) 소용이 없었다 — html_normalizer 가 본문 <a href>
# 를 [텍스트](URL) 로 보존하는데 위키 페이지끼리 건 내부 링크도 같이 들어가서, 같은 문단의
# "본문에 있는 URL 은 넣어라"는 규칙이 오히려 위키 주소를 허용했다. 그래서 프롬프트가 아니라
# 코드로 막는다. 인제스션(html_normalizer)·도구 직렬화(guide_kb)·최종 출력(여기) 세 곳이
# 이 상수를 단일 출처로 쓴다. 가이드 KB 는 항상 이 위키에서 관리하므로 도메인은 하드코딩한다
# — 호스트 접미 매칭이라 `example.atlassian.net` 처럼 어떤 서브도메인이 와도 잡힌다.
WIKI_URL_HOST_SUFFIXES: tuple = ("atlassian.net",)

# `[라벨](URL "제목")` 마크다운 링크(앞의 `!` 는 이미지). 라벨 안에 대괄호가 없는 일반 형태만 본다.
_MD_LINK_RE = re.compile(
    r"(!?)\[([^\[\]]*)\]\(\s*<?(" + _URL_BODY + r")>?(?:\s+\"[^\"]*\")?\s*\)"
)
_AUTOLINK_RE = re.compile(r"<(" + _URL_BODY + r")>")
_BARE_URL_RE = re.compile(_URL_BODY)
# 목록 항목이 위키 링크 하나뿐이면 줄을 통째로 없앤다(글자만 남기면 "- 휴가 신청 가이드" 같은
# 링크 아닌 링크 목록이 된다). 뒤에 붙은 문장부호 정도는 허용.
_LINK_ONLY_BULLET_RE = re.compile(
    r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+!?\[[^\[\]]*\]\(\s*<?(" + _URL_BODY + r")>?(?:\s+\"[^\"]*\")?\s*\)"
    r"[ \t]*[.,;:]?[ \t]*$\n?",
    re.MULTILINE,
)
# 링크를 지운 뒤 껍데기만 남은 목록 항목("- ") 과 빈 괄호.
_EMPTY_BULLET_LINE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]*$\n?", re.MULTILINE)
_EMPTY_PAREN = re.compile(r"[ \t]*[\(（]\s*[\)）]")
# 답변 말미 "참고 링크" 머리말. 이 섹션은 코드 소유다(react/guide_links.py 가 도구 결과에서 결정적으로
# 붙인다) — 모델이 쓴 머리말은 어떤 변형이든 걷어낸다. 실측 변형: `🔗 참고 링크`, `**참고 링크**`(🔗 없음,
# 2026-09-17 스크린샷), `### 🔗 참고 링크`, `참고링크:`. 굵게 표시가 🔗 안팎 어디에 붙어도 잡는다.
_REF_HDR_CORE = (
    r"(?:#{1,6}[ \t]*)?(?:\*\*|__)?(?:🔗[ \t]*)?(?:\*\*|__)?참고[ \t]?링크(?:\*\*|__)?[ \t]*:?[ \t]*(?:\*\*|__)?"
)
# 머리말만 있는 줄(strip_wiki_links 의 "항목 다 잃은 머리말" 판정용).
_REF_LINKS_HEADER = re.compile(r"^[ \t]*" + _REF_HDR_CORE + r"[ \t]*$")
# 줄 어디든 놓인 머리말 — `pre`(앞 문장, 모델이 줄바꿈을 빼먹은 "…접수됩니다. 🔗 참고 링크"), `hdr`, `rest`
# (한 줄형 `**참고 링크**: [A](u), [B](u)` 의 링크들). 줄 중간 머리말은 _ref_header_ok 가 🔗 가 있을 때만
# 인정한다 — 문장 속 "참고 링크는…" 오탐 방지, 그리고 스트리밍 보류 대안(🔗)과 같은 기준.
_REF_LINKS_ANY_RE = re.compile(r"^(?P<pre>.*?)(?P<hdr>" + _REF_HDR_CORE + r")[ \t]*(?P<rest>.*)$", re.MULTILINE)
# 머리말 뒤 한 줄형 나머지 — 마크다운 링크(대상은 None·빈 값 포함 무엇이든)만 나열됐는가.
_LINKS_ONLY_RE = re.compile(r"^(?:[ \t,;:：·、]*!?\[[^\[\]]*\]\([^()\n]*\))+[ \t.,;:：·、]*$")
# 모델식 항목 — 목록 줄 하나가 마크다운 링크 하나. 코드가 붙이는 각주(`- URL` 생 주소)는 이 패턴이 아니므로
# 정화기가 건드리지 않는다 — 이 차이가 "모델 섹션은 지우고 코드 각주는 남기는" 기준.
_LINK_ITEM_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+!?\[[^\[\]]*\]\([^()\n]*\)[ \t]*[.,;:]?[ \t]*$")
# 링크를 소개하는 **유도 문구** — "더 자세한 내용은 아래 가이드를 참고해 주세요" 꼴. 뒤따르던 링크가
# 근거 없는 주소라 전부 지워지면 이 문장만 고아로 남아 말이 안 된다(2026-09-21 신고 화면).
# 좁게 잡는다: 줄이 통째로 이 꼴일 때만, 그리고 **링크가 실제로 사라졌을 때만** 걷어낸다
# (`strip_unallowed_links`). "담당자에게 문의해 주세요" 같은 정상 문장은 매치되지 않는다.
_LEADIN_CORE = (
    r"[^\n]{0,80}?(?:가이드|도움말|매뉴얼|문서)[^\n]{0,30}?(?:참고|참조|확인)[^\n]{0,20}?"
    r"(?:하세요|해\s?주세요|해\s?주시기\s?바랍니다|하시기\s?바랍니다|바랍니다|해\s?보세요)"
)
#: 절대 경로만 적힌 마크다운 링크 — `[가이드](/wiki/spaces/…)`. `_MD_LINK_RE` 는 http(s) 만 잡아서
#: 이 꼴은 정화기 어디에도 걸리지 않는다. 채팅 패널에서 열면 **앱 오리진 + 그 경로**라 없는 페이지다.
#: 가이드 턴에서만(정책 있을 때만) 라벨 글자로 되돌린다.
_MD_PATH_LINK_RE = re.compile(r"(!?)\[([^\[\]]*)\]\(\s*(/[^()\s]*)\s*\)")
_LEADIN_PREFIX = r"^[ \t]*(?:[-*+>][ \t]*)?(?:\*\*|__)?"
#: 줄 안 어디서 시작하는 유도 문구(스트리밍 보류 판정용).
_LEADIN_LINE_RE = re.compile(_LEADIN_PREFIX + _LEADIN_CORE, re.MULTILINE)
#: 유도 문구만 있는 줄 — 뒤에 목록이 붙는 꼴.
_LEADIN_ONLY_RE = re.compile(_LEADIN_PREFIX + _LEADIN_CORE + r"(?:\*\*|__)?[ \t]*[.!:：]?[ \t]*$")
#: 한 줄형 — 유도 문구 뒤에 링크들이 바로 붙는 꼴(신고 화면이 이 모양이다).
_LEADIN_WITH_LINKS_RE = re.compile(
    _LEADIN_PREFIX + _LEADIN_CORE + r"(?:\*\*|__)?[ \t]*[.!:：]?[ \t]*"
    r"(?P<links>(?:[ \t,;:：·、]*!?\[[^\[\]]*\]\([^()\n]*\))+)[ \t.,;:：]*$\n?",
    re.MULTILINE,
)


def _ref_header_ok(m: re.Match) -> bool:
    """_REF_LINKS_ANY_RE 매치가 진짜 머리말인가. 줄 중간이면 단어 경계 + 🔗/굵게/헤딩 표식이 있어야 한다."""
    pre, hdr = m.group("pre"), m.group("hdr")
    if not pre.strip():
        return True
    if not (pre[-1].isspace() or pre[-1] in ".!?。)）"):
        return False
    return "🔗" in hdr
# 죽은 링크 — 모델이 URL 을 몰라 빈칸을 메운 `[라벨](None)`·`[라벨]()`·`[라벨](null)`. 라벨 글자만 남긴다.
_DEAD_MD_LINK_RE = re.compile(r"\[([^\[\]]*)\]\(\s*(?:None|none|null|NULL|nil|#|없음|N/A|n/a)?\s*\)")
_LIST_ITEM_LINE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)[ \t]+\S")
# 스트리밍 보류 판정용 — 아직 다 오지 않은 목록 줄("-", "1", "1." 만 도착)도 항목으로 친다.
_LIST_ITEM_OR_PARTIAL = re.compile(r"^[ \t]*(?:(?:[-*+]|\d+\.)(?:[ \t]+\S.*)?|\d*)[ \t]*$")

# 스트리밍에서 URL·마크다운 링크가 head/tail 로 쪼개지지 않게 보류할 최대 길이. Confluence 주소는
# 페이지 제목이 퍼센트 인코딩되어(한글 1자 = 9자) 200~400자까지 늘어난다 — _LOOKBEHIND 로는 모자란다.
# "🔗 참고 링크" 머리말 + 항목 목록도 이 창 안에서 한 head 에 둔다(답변 말미의 짧은 목록).
_LINK_HOLD_MAX = 600
# head 끝에 걸친 미완성 조각: `[라벨` / `[라벨]` / `[라벨](https://…` / 생 URL / `https:/` 같은 URL
# 접두 토막 / 위키 링크 하나뿐인 목록 줄(줄바꿈이 아직 안 옴) / 🔗 로 시작하는 줄(머리말 진행 중).
# 링크가 줄 첫 항목이면 앞의 "- " 마커까지 같은 head 에 둔다 — 마커만 먼저 나가면 줄 삭제가 반쪽이 된다.
# 마지막 두 대안: "참고 링크" 머리말이 **도착 중인 줄**(`**`/`**참고 링`/`참고 링크: [A](u),` — 아직 줄바꿈이
# 안 옴, `\Z` = 버퍼 끝)과 줄 중간 🔗. 🔗 없는 굵은 머리말은 위 🔗 대안이 못 잡아 head 경계에서 갈리면 화면에만
# 샌다(1자 스트림 실측: 머리말이 버퍼 맨 앞에 놓여 cut 이 `*|*참고` 사이에 떨어졌다). 마크업만 온 줄 끝("**")도
# 붙든다 — 다음 글자가 참 이 아니면 바로 풀린다. 줄이 닫히면 9) 규칙이 머리말+목록을 묶는다.
_OPEN_LINK_TAIL_RE = re.compile(
    r"(?:^[ \t]*(?:[-*+]|\d+\.)[ \t]+!?\[[^\[\]]*\]\([^\s()<>]*\)[ \t]*[.,;:]?[ \t]*$"
    r"|^[ \t]*(?:#+[ \t]*)?(?:\*\*)?🔗[^\n]*$"
    r"|^[ \t]*(?:#{1,6}[ \t]*)?(?:\*{1,2}|_{1,2})?(?:🔗[ \t]*)?(?:\*{1,2}|_{1,2})?(?:참(?:고(?:[ \t]?(?:링(?:크[^\n]*)?)?)?)?)?\Z"
    r"|[ \t]*(?:\*{1,2}|_{1,2})?🔗[^\n]*\Z"
    r"|(?:^[ \t]*(?:[-*+]|\d+\.)[ \t]+!?)?\[[^\[\]]*(?:\](?:\(\s*<?[^\s()<>]*)?)?$"
    r"|https?://[^\s<>()\[\]\"'`]*$"
    r"|(?<![A-Za-z])h(?:t(?:t(?:p(?:s?(?::(?://?)?)?)?)?)?)?$)",
    re.MULTILINE,
)


def is_wiki_url(url: str) -> bool:
    """URL 의 호스트가 가이드 위키 도메인(``WIKI_URL_HOST_SUFFIXES``)인지 판정한다.

    Args:
        url: 판정할 URL 문자열. 빈 값·파싱 불가 값은 ``False``.

    Returns:
        호스트가 위키 도메인과 같거나 그 서브도메인이면 ``True``.
    """
    if not url:
        return False
    try:
        host = (urlsplit(url.strip()).hostname or "").lower()
    except ValueError:
        return False
    return any(host == suffix or host.endswith("." + suffix) for suffix in WIKI_URL_HOST_SUFFIXES)


def strip_wiki_links(text: str) -> str:
    """답변 본문에서 가이드 위키 주소를 지운다.

    ``[라벨](위키 URL)`` 은 라벨 글자만 남겨 문장을 보존하고(``자세한 절차는 [휴가 신청
    가이드](https://…atlassian.net/…)를`` → ``자세한 절차는 휴가 신청 가이드를``), 이미지
    링크·``<위키 URL>`` 자동링크·생 URL 은 통째로 지운다. 위키 링크 하나만 있던 목록 항목은
    줄을 없애고, 그래서 항목이 하나도 남지 않은 "🔗 참고 링크" 머리말도 걷어낸다(머리말 앞
    빈 줄은 남긴다 — 스트리밍 head 경계와 무관하게 같은 결과가 나오게). 위키가 아닌
    URL(문의 폼·다운로드 센터 등 실무 주소)은 건드리지 않는다.

    Args:
        text: LLM 응답 본문.

    Returns:
        위키 주소와 그 잔재가 제거된 문자열. 위키 주소가 없으면 입력을 그대로 반환.
    """
    if not text:
        return text
    lowered = text.lower()
    if not any(suffix in lowered for suffix in WIKI_URL_HOST_SUFFIXES):
        return text

    def _bullet(m: re.Match) -> str:
        return "" if is_wiki_url(m.group(1)) else m.group(0)

    def _md(m: re.Match) -> str:
        bang, label, url = m.group(1), m.group(2), m.group(3)
        if not is_wiki_url(url):
            return m.group(0)
        return "" if bang else label.strip()

    out = _LINK_ONLY_BULLET_RE.sub(_bullet, text)
    out = _MD_LINK_RE.sub(_md, out)
    out = _AUTOLINK_RE.sub(lambda m: "" if is_wiki_url(m.group(1)) else m.group(0), out)
    out = _BARE_URL_RE.sub(lambda m: "" if is_wiki_url(m.group(0)) else m.group(0), out)
    if out == text:
        return text
    out = _EMPTY_PAREN.sub("", out)
    out = _EMPTY_BULLET_LINE.sub("", out)
    # 항목을 다 잃은 "🔗 참고 링크" 머리말 제거 — 다음 비어 있지 않은 줄이 목록 항목이 아니면 머리말만 남은 것.
    lines = out.split("\n")
    kept: list = []
    for i, line in enumerate(lines):
        if _REF_LINKS_HEADER.match(line):
            nxt = next((l for l in lines[i + 1 :] if l.strip()), "")
            if not _LIST_ITEM_LINE.match(nxt):
                continue
        kept.append(line)
    return "\n".join(kept)


def _drop_orphan_leadins(original: str, out: str) -> str:
    """목록을 소개하던 유도 문구가 항목을 다 잃었으면 문구도 걷어낸다.

    원문에서 "…가이드를 참고해 주세요" 뒤에 목록이 있었는데 정화 후 그 목록이 사라졌다면,
    남은 문장은 가리킬 곳이 없다. 원문에 목록이 **있었던** 문구만 후보로 삼아, 정상 문장을
    지우지 않는다.
    """
    lines = original.split("\n")
    candidates = set()
    for i, line in enumerate(lines):
        if _LEADIN_ONLY_RE.match(line):
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if _LIST_ITEM_LINE.match(nxt):
                candidates.add(line.strip())
    if not candidates:
        return out
    olines = out.split("\n")
    kept: list = []
    for i, line in enumerate(olines):
        if line.strip() in candidates:
            nxt = next((l for l in olines[i + 1:] if l.strip()), "")
            if not _LIST_ITEM_LINE.match(nxt):
                continue
        kept.append(line)
    return "\n".join(kept)


def strip_unallowed_links(text: str, permits) -> str:
    """근거 없는 주소를 지운다 — 판정은 호출자가 넘긴 ``permits(url)`` 콜백이 한다.

    왜 필요한가(2026-09-21 실측): 가이드 KB 답변 말미에 ``더 자세한 내용은 아래 가이드를 참고해
    주세요.`` + ``[Agent 사용 심층 가이드](…)`` 가 붙는데 **주소를 모델이 지어냈다.** 가이드 페이지의
    진짜 주소는 LLM 에게 가지 않으므로(`tools/guide_kb.py`), 가이드 턴의 답변에 있는 주소 중 **모델이
    실제로 본 텍스트에 없던 것**은 전부 환각이다. 판정 규칙은 `agents/react/link_policy.py` 가 소유하고
    (`app.utils` 가 `app.agents` 를 참조하지 않게) 여기는 콜백만 받는다.

    처리 모양은 ``strip_wiki_links`` 와 같다 — 마크다운 링크는 **라벨 글자만 남겨 문장을 보존**하고,
    이미지·자동링크·생 URL 은 통째로 지운다. 유도 문구는 링크를 다 잃었을 때만 함께 걷어낸다.

    Args:
        text: LLM 응답 본문.
        permits: ``url -> bool``. ``None`` 이면(정책 없음) 입력을 그대로 돌려준다.

    Returns:
        근거 없는 주소가 제거된 문자열. 손댈 것이 없으면 입력을 그대로 반환.
    """
    if not text or permits is None or ("http" not in text and "](/" not in text):
        return text

    def _all_unallowed(links: str) -> bool:
        urls = [m.group(3) for m in _MD_LINK_RE.finditer(links)]
        return bool(urls) and not any(permits(u) for u in urls)

    def _md(m: "re.Match") -> str:
        bang, label, url = m.group(1), m.group(2), m.group(3)
        if permits(url):
            return m.group(0)
        return "" if bang else label.strip()

    # ① 한 줄형 "유도 문구 + 링크들" — 링크가 전부 근거 없으면 줄째 지운다. 라벨만 남기면
    #    "…참고해 주세요. Agent 사용 심층 가이드 DMP 사용 가이드" 같은 꼬리가 남는다.
    out = _LEADIN_WITH_LINKS_RE.sub(
        lambda m: "" if _all_unallowed(m.group("links")) else m.group(0), text)
    out = _LINK_ONLY_BULLET_RE.sub(lambda m: "" if not permits(m.group(1)) else m.group(0), out)
    out = _MD_LINK_RE.sub(_md, out)
    out = _AUTOLINK_RE.sub(lambda m: "" if not permits(m.group(1)) else m.group(0), out)
    out = _BARE_URL_RE.sub(lambda m: "" if not permits(m.group(0)) else m.group(0), out)
    # 절대 경로 링크(`[가이드](/wiki/…)`)도 가이드 턴에서는 라벨만 남긴다 — 열면 앱 오리진에
    # 그 경로가 붙어 없는 페이지로 간다(신고된 "엉뚱한 페이지"의 다른 얼굴).
    out = _MD_PATH_LINK_RE.sub(lambda m: "" if m.group(1) else m.group(2).strip(), out)
    if out == text:
        return text
    out = _EMPTY_PAREN.sub("", out)
    out = _EMPTY_BULLET_LINE.sub("", out)
    # 빈 줄 압축(`\n{3,}`)은 **하지 않는다** — 스트리밍에서 빈 줄이 head 경계 너머에 있으면 압축이
    # 갈려 화면에만 빈 줄이 하나 더 남는다(화면 ≠ 저장본). 같은 이유로 `strip_wiki_links` 도 압축하지
    # 않는다. 빈 줄 하나는 렌더링에 보이지 않지만, 두 본문이 다른 것은 나중에 비싸게 물린다.
    return _drop_orphan_leadins(text, out)


def strip_reference_links_section(text: str) -> str:
    """모델이 직접 쓴 "참고 링크" 섹션을 통째로 지운다 — 이 섹션은 코드가 소유한다.

    배경(실측 2026-09-16~17, stg=vllm·로컬=gemini 모두): 프롬프트가 답변 말미 "🔗 참고 링크" 목록을
    요구했는데 가이드 청크 본문 대부분에 URL 이 없다. 모델은 섹션을 만들고 라벨은 청크 제목으로,
    URL 자리는 `None`·제목을 영어로 번역한 가짜 경로·무의미 반복으로 메웠다. 형식이 원문에 없는 것을
    요구하는 구조라 프롬프트로는 안 잡힌다. 그래서 모델이 쓴 섹션은 여기서 지우고,
    ``app/agents/react/guide_links.py`` 가 이번 턴 도구 결과에 실제로 있는 URL 로만 다시 붙인다.

    지우는 범위: 머리말 줄(🔗·굵게·헤딩·콜론 변형, 줄 중간이면 표식 있을 때만) + 뒤따르는 빈 줄·목록 항목 —
    단 항목이 **전부 마크다운 링크**(`- [라벨](URL)`, 대상이 None 이어도)일 때만. 코드 각주(`- URL`)는 이 꼴이
    아니라 남는다. 섹션 밖의 죽은 링크 `[라벨](None)` 은 라벨 글자만 남긴다.

    Args:
        text: LLM 응답 본문.

    Returns:
        섹션이 제거된 문자열. 바꿀 것이 없으면 입력을 그대로 반환.
    """
    if not text or ("참고" not in text and "](" not in text):
        return text
    lines = text.split("\n")
    kept: list = []
    i = 0
    changed = False
    while i < len(lines):
        m = _REF_LINKS_ANY_RE.match(lines[i]) if "참고" in lines[i] else None
        if m and _ref_header_ok(m):
            rest = m.group("rest") or ""
            j = i + 1
            while j < len(lines) and (not lines[j].strip() or _LIST_ITEM_LINE.match(lines[j])):
                j += 1
            items = [l for l in lines[i + 1 : j] if l.strip()]
            inline_ok = bool(_LINKS_ONLY_RE.match(rest))                       # 한 줄형 `**참고 링크**: [A](u), [B](u)`
            block_ok = not rest.strip() and bool(items) and all(_LINK_ITEM_RE.match(l) for l in items)
            if inline_ok or block_ok:
                changed = True
                if m.group("pre"):
                    kept.append(m.group("pre"))   # 같은 줄 앞의 문장은 보존
                i = j if block_ok else i + 1
                continue
        kept.append(lines[i])
        i += 1
    out = "\n".join(kept) if changed else text
    if "](" in out:
        out2 = _DEAD_MD_LINK_RE.sub(lambda mm: mm.group(1).strip(), out)
        changed = changed or out2 != out
        out = out2
    return re.sub(r"\n{3,}", "\n\n", out) if changed else text


def sanitize_tool_names(text: str, *, link_permits=None) -> str:
    """LLM 본문을 정화한다 — 도구명 마스킹 + 중복 흡수 + 수식·HTML 줄바꿈 보정.

    도구명: 화이트리스트(정확 매칭) → 안전망 패턴 순으로 두 단계 마스킹한다.
    한국어 응답 안에 박힌 ASCII 식별자도 단어 경계로 정확히 잡는다.
    `link_permits` 가 주어지면(가이드 검색을 쓴 턴) `strip_unallowed_links` 가 근거 없는
    주소까지 걷어낸다 — 판정 규칙은 `agents/react/link_policy.py` 소유, 여기는 콜백만 받는다.

    이어서 `_absorb_redundancy` 가 치환으로 생긴 중복("내부 도구 도구를", 감싼 백틱,
    치환어만 남은 괄호)을 흡수하고, `strip_internal_ids`·`strip_wiki_links` 로 내부
    식별자와 가이드 위키 주소를 지우고, `strip_reference_links_section` 으로 모델이 쓴
    "참고 링크" 섹션(코드 소유 — guide_links 가 도구 결과로 다시 붙임)을 걷어내고, `normalize_math_markup`으로 LaTeX/수식을 유니코드
    평문으로 바꾼다.

    스트리밍(StreamSanitizer)·최종 응답(react loop)·비스트리밍 경로가 모두 이
    함수를 거치므로, 출력 정화는 여기 한 곳에서 일괄 보장된다.

    Args:
        text: LLM 응답 본문 (None/빈 문자열은 그대로 반환)

    Returns:
        도구 이름이 "내부 도구"로 치환되고 수식이 평문화된 문자열
    """
    if not text:
        return text
    # 가이드 에이전트-힌트 마커는 내부 신호이므로 표시 전에 제거한다(사용자 비노출).
    if GUIDE_AGENT_HINT in text:
        text = text.replace(GUIDE_AGENT_HINT, "")
    wl = _WHITELIST_RE if _WHITELIST_RE is not None else _build_whitelist()
    text = wl.sub(_REPLACEMENT, text)
    text = _PATTERN_FALLBACK.sub(_REPLACEMENT, text)
    text = _absorb_redundancy(text)
    text = strip_internal_ids(text)
    text = strip_wiki_links(text)
    text = strip_reference_links_section(text)
    # 근거 없는 링크 제거는 반드시 **참고 링크 섹션 처리 뒤**다 — 먼저 돌면 `- [라벨](URL)` 이
    # `- 라벨` 이 되어 _LINK_ITEM_RE 가 모델 섹션을 못 알아보고 머리말만 화면에 남는다.
    text = strip_unallowed_links(text, link_permits)
    return normalize_html_breaks(normalize_math_markup(text))


class StreamSanitizer:
    """토큰 스트림용 버퍼링 정화기.

    LLM 토큰이 도구 이름 중간에서 잘려 들어오는 경우를 대비해
    마지막 `_LOOKBEHIND` 자를 보류하고, 그 앞 부분만 즉시 정화하여 yield한다.
    스트림 종료 시 `flush()`로 잔여 버퍼를 마지막 정화 처리한다.

    Examples:
        >>> s = StreamSanitizer()
        >>> out = []
        >>> for chunk in ["앞서 logist", "ics_service_", "list_assets로 조회"]:
        ...     out.append(s.feed(chunk))
        >>> out.append(s.flush())
        >>> "".join(out)
        '앞서 내부 도구로 조회'
    """

    # 가장 긴 도구 이름(약 65자) + 여유. _PATTERN_FALLBACK이 이 길이 안에서 종결되어야 한다.
    _LOOKBEHIND = 80

    def __init__(self, *, link_permits=None) -> None:
        self._buffer = ""
        # 링크 허용 판정 콜백(`link_policy.TurnLinkPolicy.permits`). 정책 객체는 도구 결과가
        # 도착할 때마다 **제자리로** 갱신되므로, 바인딩된 이 콜백이 턴 내내 최신 상태를 읽는다.
        self._link_permits = link_permits

    def feed(self, chunk: str) -> str:
        """새 토큰을 버퍼에 추가하고 정화 가능한 head를 반환한다.

        Args:
            chunk: 새로 도착한 LLM 토큰

        Returns:
            정화된 텍스트 head. 버퍼가 _LOOKBEHIND 이하면 빈 문자열.
        """
        if not chunk:
            return ""
        self._buffer += chunk
        if len(self._buffer) <= self._LOOKBEHIND:
            return ""
        emit_len = self._safe_emit_len(
            self._buffer, len(self._buffer) - self._LOOKBEHIND, self._link_permits)
        if emit_len <= 0:
            # 수식/LaTeX 스팬이 경계에 걸쳐 안전하게 자를 위치가 없음 → 더 버퍼링.
            return ""
        head, self._buffer = self._buffer[:emit_len], self._buffer[emit_len:]
        return sanitize_tool_names(head, link_permits=self._link_permits)

    def flush(self) -> str:
        """스트림 종료 시 잔여 버퍼를 정화하여 반환한다."""
        out = sanitize_tool_names(self._buffer, link_permits=self._link_permits)
        self._buffer = ""
        return out

    @staticmethod
    def _safe_emit_len(buf: str, cut: int, link_permits=None) -> int:
        """수식/LaTeX 스팬을 쪼개지 않는 안전한 emit 경계를 구한다.

        `sanitize_tool_names`(→ `normalize_math_markup`)가 head 단위로 적용되므로,
        head 가 `$...$`·`\\(...\\)`·`\\[...\\]` 또는 `\\command` 중간에서 끝나면 변환이
        깨진다(예: `$\\rightarrow` 만 떼면 `$→` 잔재, `\\ri`+`ghtarrow` 로 갈리면 미변환).
        그런 경우 스팬 시작 앞으로 cut 을 당겨 닫힘이 올 때까지 보류한다.

        무한 보류 방지: 스팬 시작이 `_LOOKBEHIND` 보다 멀면(짧은 인라인 수식이 아니라
        통화·일반 `$` 로 판단) 당기지 않는다. 인라인 수식은 짧으므로 이 창 안에서 닫힌다.
        """
        if cut <= 0:
            return 0
        lb = StreamSanitizer._LOOKBEHIND
        # 1) 닫히지 않은 $ (홀수 개) 가 가까이 있으면 그 $ 앞에서 자른다 → $...$ 보존
        if buf.count("$", 0, cut) % 2 == 1:
            last = buf.rfind("$", 0, cut)
            if last != -1 and cut - last <= lb:
                cut = last
        # 2) 닫히지 않은 \( … \) / \[ … \] 안에서 자르지 않는다
        for opener, closer in (("\\(", "\\)"), ("\\[", "\\]")):
            if buf.count(opener, 0, cut) > buf.count(closer, 0, cut):
                last = buf.rfind(opener, 0, cut)
                if last != -1 and cut - last <= lb:
                    cut = last
        # 3) 끝에 걸친 \command (또는 \( 시작 토막) 앞에서 자른다
        m = re.search(r"\\[A-Za-z(]*$", buf[:cut])
        if m and m.start() < cut and cut - m.start() <= lb:
            cut = m.start()
        # 4) ASCII 식별자(도구명 후보) 한복판에서 자르지 않는다.
        #    _LOOKBEHIND 만큼 보류해도 cut 자체가 이름 중간에 떨어지면 앞 head 에는
        #    `mail_list_`, 다음 head 에는 `messages` 만 남아 **양쪽 다 매칭되지 않는다**.
        #    이름이 버퍼 끝에서 정확히 _LOOKBEHIND 근처에 놓일 때 재현된다 — 2026-08-19
        #    실측 누출: 화면에만 `mail_list_messages` 가 그대로 나가고 DB 저장본
        #    (LoopResult 전문 정화)은 정상이라 화면을 봐야만 보였다.
        if 0 < cut < len(buf) and _IDENT_CHAR.match(buf[cut]) and _IDENT_CHAR.match(buf[cut - 1]):
            start = cut
            while start > 0 and _IDENT_CHAR.match(buf[start - 1]):
                start -= 1
            # 식별자 런이 창보다 길면(base64 등) 당기지 않는다 — 무한 보류 방지.
            if cut - start <= lb:
                cut = start
        # 5) 도구명 매치의 **뒤쪽 문맥**이 head 에 다 들어오지 않으면 매치 앞으로 당긴다.
        #    흡수 규칙(_absorb_redundancy)은 치환어 뒤의 백틱·분류어·닫는 괄호를 봐야
        #    하는데, 그게 tail 에 남으면 흡수가 발동하지 못한다. 그러면 화면에는
        #    "내부 도구 도구를" 이 그대로 나가고 저장본(전문 정화)만 깨끗해진다 — 4)와
        #    같은 종류의 "화면만 깨지는" 버그다.
        if cut > 0:
            for pat in (
                _WHITELIST_RE if _WHITELIST_RE is not None else _build_whitelist(),
                _PATTERN_FALLBACK,
            ):
                for m in pat.finditer(buf, max(0, cut - lb)):
                    if m.start() >= cut:
                        break  # 매치가 head 밖에서 시작 — 이번 emit 과 무관
                    if cut - m.end() >= _ABSORB_WINDOW:
                        continue  # 뒤쪽 문맥이 이미 head 안 — 흡수 가능
                    cand = m.start()
                    # 여는 백틱·괄호도 **같은 head** 에 있어야 흡수가 온전히 돈다. 매치
                    # 시작까지만 당기면 head 끝에 "(`" 만 남고 다음 head 에 이름이 들어가
                    # `내부 도구` 처럼 반쪽만 정리된 결과가 나온다(실측).
                    while cand > 0 and buf[cand - 1] == "`":
                        cand -= 1
                    w = max(0, cand - lb)
                    if buf.count("(", w, cand) > buf.count(")", w, cand):
                        opener = buf.rfind("(", w, cand)
                        if opener != -1:
                            cand = opener
                    while cand > 0 and buf[cand - 1] in " \t":
                        cand -= 1
                    if cut - cand <= lb:
                        cut = cand
                        break
        # 6) 여는 백틱·괄호를 head 끝에 남기지 않는다. 5)는 버퍼에 **이미 있는** 매치만
        #    보호하는데, 토큰이 잘게 오면 이름이 아직 도착하지 않은 채 "도구(`" 까지만
        #    흘러나가고(1자 스트림 실측) 다음 head 에 들어온 이름이 반쪽만 정리된다
        #    (`내부 도구` 또는 내부 도구). 닫힘이 올 때까지 보류한다 — 1)의 `$` 처리와
        #    같은 방식이고, 창을 넘으면 포기해 무한 보류를 막는다.
        if cut > 0:
            w = max(0, cut - lb)
            if buf.count("(", w, cut) > buf.count(")", w, cut):
                opener = buf.rfind("(", w, cut)
                if opener != -1:
                    cut = opener
                    while cut > 0 and buf[cut - 1] in " \t":
                        cut -= 1  # 괄호 앞 공백까지 같은 head 에 둔다(A4 가 함께 흡수)
            w = max(0, cut - lb)
            if buf.count("`", w, cut) % 2 == 1:
                last = buf.rfind("`", w, cut)
                if last != -1:
                    cut = last
        # 7) HTML 줄바꿈 태그도 한 head 안에서 치환되어야 한다. 특히 '<'와
        #    'br>' 사이, '&lt;' 엔티티 안에서 잘리면 최종 저장본과 화면이 달라진다.
        for m in _HTML_BREAK_RE.finditer(buf):
            if m.start() >= cut:
                break
            if m.start() < cut < m.end():
                cut = m.start()
                break
        # 8) URL·마크다운 링크 한복판에서 자르지 않는다. strip_wiki_links 는 호스트를 봐야
        #    위키 주소인지 알 수 있는데, head 에 `[라벨](https://example.atl` 까지만 들어가면
        #    판정도 치환도 못 하고 그대로 화면에 나간다(저장본은 전문 정화라 멀쩡 — 4)·5)와
        #    같은 "화면만 새는" 유형). `[라벨]` 까지만 온 경우도 `(` 가 뒤따를 수 있어 잡는다.
        #    위키 링크 하나뿐인 목록 줄은 줄바꿈까지 함께 지우므로 `\n` 이 올 때까지 줄 전체를
        #    보류한다. Confluence 주소가 _LOOKBEHIND 보다 훨씬 길어 창은 _LINK_HOLD_MAX 로 따로.
        if cut > 0:
            m = _OPEN_LINK_TAIL_RE.search(buf, max(0, cut - _LINK_HOLD_MAX), cut)
            if m and m.start() < cut and cut - m.start() <= _LINK_HOLD_MAX:
                cut = m.start()
        # 9) "🔗 참고 링크" 머리말은 항목이 전부 위키 링크였으면 함께 걷어내는데, 그 판정은 항목을
        #    다 봐야 할 수 있다. 머리말 뒤가 아직 목록(또는 빈 줄)뿐이면 목록이 끝날 때까지 머리말과
        #    항목을 같은 head 에 둔다 — 답변 말미의 짧은 목록이라 창(_LINK_HOLD_MAX)을 넘지 않는다.
        if cut > 0:
            w = max(0, cut - _LINK_HOLD_MAX)
            header = None
            for hm in _REF_LINKS_ANY_RE.finditer(buf, w, cut):
                if _ref_header_ok(hm):
                    header = hm
            if header is not None:
                hstart = header.start("hdr")
                after = buf[header.end("hdr") : cut].split("\n")
                first, rest_lines = after[0], after[1:]
                first_ok = (not first.strip()) or bool(_LINKS_ONLY_RE.match(first))   # 한 줄형 나머지 또는 빈 꼬리
                if first_ok and all(_LIST_ITEM_OR_PARTIAL.match(l) for l in rest_lines):
                    cut = hstart
                else:
                    # 목록이 끝났다(비목록 줄 도착). 머리말+목록이 **한 head** 에 통째로 들어가게 첫 비목록
                    # 줄의 시작으로 cut 을 옮긴다 — 기본 cut(len-80)이 목록 한복판에 떨어지면 머리말만 먼저
                    # 나가 지워지고 항목은 다음 head 에 고아로 남아 그대로 화면에 샌다(1자 스트림 실측).
                    pos = header.end("hdr") + len(first) + 1
                    for line in rest_lines:
                        if line.strip() and not _LIST_ITEM_OR_PARTIAL.match(line):
                            break
                        pos += len(line) + 1
                    if hstart < pos <= cut:
                        cut = pos
        # 10) 링크를 소개하는 유도 문구("… 가이드를 참고해 주세요")와 뒤따르는 링크를 한 head 에 둔다.
        #     링크가 전부 근거 없으면 문구까지 함께 걷어내는데(strip_unallowed_links), 경계가 그 사이에
        #     떨어지면 **문구만 먼저 화면에 나가 고아로 남는다**(저장본은 전문 정화라 멀쩡 — 8)·9)와 같은
        #     "화면만 새는" 유형). 9)와 같은 창을 쓰고, 정책이 없는 턴에는 아예 돌지 않는다.
        if cut > 0 and link_permits is not None:
            lead = None
            # endpos 를 cut 으로 묶지 않는다 — 문구가 cut 을 걸치고 있으면(아직 도착 중) 매치 자체가
            # 사라져 보류가 풀린다. 시작점만 cut 앞이면 후보로 본다.
            for lm in _LEADIN_LINE_RE.finditer(buf, max(0, cut - _LINK_HOLD_MAX)):
                if lm.start() >= cut:
                    break
                lead = lm
            if lead is not None and cut - lead.start() <= _LINK_HOLD_MAX:
                # 문구가 소개하는 블록의 끝 — 문구 줄 + 뒤따르는 빈 줄·목록 항목까지.
                nl = buf.find("\n", lead.end())
                if nl == -1:
                    block_end = len(buf)          # 줄이 아직 안 끝났다(한 줄형 링크 도착 중)
                else:
                    block_end = nl + 1
                    for line in buf[block_end:].split("\n"):
                        if line.strip() and not _LIST_ITEM_OR_PARTIAL.match(line):
                            break
                        block_end += len(line) + 1
                    block_end = min(block_end, len(buf))
                if block_end > cut:
                    cut = lead.start()            # 블록이 head 밖으로 걸친다 → 통째로 보류
        return max(cut, 0)
