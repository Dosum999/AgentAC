"""정규식 규칙 — 치환류(이메일·전화)와 차단류(주민번호·카드·계좌·비밀번호·API 키).

설계 원칙:
- ``\\b`` 를 쓰지 않는다. 한글은 ``\\w`` 라 "메일user@example.com로" 같은 붙여쓰기에서
  경계가 어긋난다. 좌우를 명시적 lookaround 로 잡는다.
- 숫자류는 좌우에 숫자·하이픈이 없을 때만(주문번호·송장번호의 일부를 자르지 않게).
- 차단류는 **키워드 앵커**(계좌는 은행·계좌 어휘, 비밀번호는 키워드+구분자)와
  검증기(카드 Luhn, 주민번호 생년월일)로 오탐을 줄인다. 그래도 남는 오탐은 shadow
  집계로 보고 ``PII_GUARD_BLOCK_KINDS`` 로 단계 활성한다.
- 계좌 규칙은 매치 전체가 아니라 ``num`` 그룹만 바꾼다(키워드는 보존). ``RegexRule.group``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Optional

from agent_arch.llm.pii.tokens import Kind

# ── 공통 경계 ────────────────────────────────────────────────────────────────
_NO_DIGIT_L = r"(?<![0-9-])"
_NO_DIGIT_R = r"(?![0-9-])"

#: 한글 조사 — 이름 뒤에 붙어도 이름 경계로 인정한다(lookahead 로만 보고 소비하지 않는다).
PARTICLE = (
    r"(?:께서|에게서|한테서|으로서|으로써|이라고|라고|이랑|랑|에게|한테|부터|까지|으로|로|이며|이나|이든"
    r"|께|님|씨|군|양|이|가|은|는|을|를|의|과|와|도|만|요|에|야|아)"
)
#: 존칭·직급 — 2자 이름은 이것이 뒤따를 때만 이름으로 본다.
HONORIFIC = (
    r"(?:대표이사|부사장|본부장|센터장|연구원|매니저|팀장|과장|대리|부장|이사|차장|상무|전무|사장|사원|주임"
    r"|선임|책임|수석|대표|실장|소장|원장|회장|교수|박사|기사|프로|팀|님|씨|PM|CTO|CEO|CFO|COO)"
)
#: 이름 뒤 허용 꼬리: (공백)직급? 조사{0,3} 그리고 한글이 더 이어지지 않음.
NAME_TAIL = rf"(?=(?:\s?{HONORIFIC})?(?:{PARTICLE}){{0,3}}(?![가-힣]))"
#: 2자 이름·흔한 단어 이름: 직급/존칭이 반드시 뒤따라야 한다.
NAME_TAIL_STRICT = rf"(?=\s?{HONORIFIC}(?:{PARTICLE}){{0,2}}(?![가-힣]))"
NAME_LEFT = r"(?<![가-힣A-Za-z0-9])"
ASCII_ID_LEFT = r"(?<![A-Za-z0-9._-])"
ASCII_ID_RIGHT = r"(?![A-Za-z0-9._-])"


def luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = ord(ch) - 48
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class RegexRule:
    """하나의 정규식 규칙.

    Attributes:
        kind: 토큰 종류.
        pattern: 컴파일된 정규식.
        tier: 낮을수록 먼저 채택(겹침 해소용). 0=보호, 1=차단류, 2=이메일, 3=전화, 4=사번.
        group: 치환할 그룹 이름(None 이면 매치 전체).
        validator: 매치를 받아 채택 여부를 돌려준다(None 이면 항상 채택).
        deny_json_keys: 이 키 아래의 값에는 적용하지 않는다(도구 결과 JSON).
        deny_before: 매치 직전 20자에 이 패턴이 있으면 버린다.
    """

    kind: Kind
    pattern: re.Pattern
    tier: int
    group: Optional[str] = None
    validator: Optional[Callable[[re.Match], bool]] = None
    deny_json_keys: Optional[re.Pattern] = None
    deny_before: Optional[re.Pattern] = None


# ── 보호 스팬(치환 금지) ───────────────────────────────────────────────────────
#: 이미 치환된 토큰·차단 라벨 — 이중 래핑(supervisor→worker)에서도 다시 건드리지 않는다.
EXISTING_TOKEN = re.compile(
    r"\[\[(?:PERSON|EMAIL|PHONE|HANDLE|EMPNO)_[0-9a-f]{8}(?:[0-9a-f]{4})?\]\]|\[차단됨:[^\]]{1,10}\]"
)
#: URL 안은 치환하지 않는다 — 영문 ID 가 경로에 들어 있을 때 링크를 깨지 않기 위해.
URL = re.compile(r"https?://[^\s<>\"')\]]+")

# ── 치환류 ────────────────────────────────────────────────────────────────────
EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])(?P<local>[A-Za-z0-9._%+-]+)@(?P<domain>[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
    r"(?![A-Za-z0-9-]|\.[A-Za-z])"
)

# 한국 전화: 010-1234-5678 / 01012345678 / 02-123-4567 / 031)123-4567 / +82-10-1234-5678 / 010.1234.5678
PHONE_KR = re.compile(
    r"(?<![0-9.-])(?:\+82[-\s]?0?|0)(?P<area>1[016789]|2|[3-6][1-5]|50\d?|70|80)"
    r"[-.\s)]?(?P<mid>\d{3,4})[-.\s]?(?P<last>\d{4})(?![0-9-])"
)


def _phone_ok(m: re.Match) -> bool:
    digits = re.sub(r"\D", "", m.group(0))
    if digits.startswith("82"):
        digits = "0" + digits[2:].lstrip("0")
    return 9 <= len(digits) <= 11


#: ``@handle`` — 기본 규칙 목록에는 넣지 않는다(코드의 데코레이터·봇 멘션까지 잡는다).
#: 어휘(lexicon)의 handle 별칭이 이 자리를 맡는다.
HANDLE = re.compile(r"(?<![A-Za-z0-9._%+-])@(?P<id>[A-Za-z0-9][A-Za-z0-9._-]{2,})(?![A-Za-z0-9._-])")

# ── 차단류 ────────────────────────────────────────────────────────────────────
# 주민번호: 생년월일 유효성 + 뒷자리 첫 자리 1~8(외국인 5~8 포함).
SSN_KR = re.compile(
    r"(?<![0-9-])(?P<birth>\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01]))-?(?P<tail>[1-8]\d{6})(?![0-9-])"
)

CREDIT_CARD = re.compile(r"(?<![0-9-])(?P<n>\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4})(?![0-9-])")


def _card_ok(m: re.Match) -> bool:
    return luhn_ok(re.sub(r"\D", "", m.group("n")))


_ACCOUNT_KW = (
    r"(?:계좌번호|계좌|입금|송금|예금주|은행|bank|account|acct"
    r"|신한|농협|수협|씨티|카카오뱅크|토스뱅크|케이뱅크|새마을금고|우체국|기업은행|국민은행|우리은행|하나은행"
    r"|SC제일|대구은행|부산은행|경남은행|광주은행|전북은행|제주은행|iM뱅크)"
)
ACCOUNT = re.compile(
    rf"(?i)(?P<kw>{_ACCOUNT_KW})[^\n0-9]{{0,15}}(?<![0-9-])(?P<num>\d{{2,6}}(?:-\d{{2,6}}){{1,3}}|\d{{10,14}})(?![0-9-])"
)
_ACCOUNT_DENY_BEFORE = re.compile(r"(?i)(주문|송장|운송장|택배|order|tracking|waybill|invoice)")
_ACCOUNT_DENY_KEYS = re.compile(r"(?i)(order|tracking|waybill|invoice|송장|주문)")


def _account_ok(m: re.Match) -> bool:
    return 10 <= len(re.sub(r"\D", "", m.group("num"))) <= 14


# API 키: 고정밀 패밀리만. 일반 고엔트로피 탐지는 하지 않는다(base64 이미지·file_id·UUID 오탐).
APIKEY = re.compile(
    r"(?<![A-Za-z0-9_-])(?:"
    r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"
    r"|AIza[0-9A-Za-z_-]{35}"
    r"|gh[pousr]_[A-Za-z0-9]{36,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"
    r")(?![A-Za-z0-9_-])"
)
BEARER = re.compile(r"(?i)\bbearer\s+(?P<v>[A-Za-z0-9._~+/=-]{20,})")

# 비밀번호: 키워드 + 구분자(":"/"=" 또는 조사+공백) + 값. 값만 차단, 키워드는 보존.
PASSWORD = re.compile(
    r"(?i)(?P<kw>password|passwd|pwd|비밀번호|패스워드|비번|secret|api[_-]?key)"
    r"\s*(?:[:=]|(?:은|는|이|가)\s)\s*[\"'`]?(?P<v>[^\s\"'`,，]{6,})"
)


def _password_ok(m: re.Match) -> bool:
    v = m.group("v")
    return bool(re.search(r"[A-Za-z0-9]", v)) and not re.fullmatch(r"[가-힣]+", v)


# ── 규칙 목록 ─────────────────────────────────────────────────────────────────
TIER_PROTECT, TIER_BLOCK, TIER_EMAIL, TIER_PHONE, TIER_EMPNO, TIER_LEXICON = 0, 1, 2, 3, 4, 5

RULES: tuple[RegexRule, ...] = (
    RegexRule(Kind.SSN, SSN_KR, TIER_BLOCK),
    RegexRule(Kind.CARD, CREDIT_CARD, TIER_BLOCK, group="n", validator=_card_ok),
    RegexRule(Kind.ACCOUNT, ACCOUNT, TIER_BLOCK, group="num", validator=_account_ok,
              deny_json_keys=_ACCOUNT_DENY_KEYS, deny_before=_ACCOUNT_DENY_BEFORE),
    RegexRule(Kind.APIKEY, APIKEY, TIER_BLOCK),
    RegexRule(Kind.APIKEY, BEARER, TIER_BLOCK, group="v"),
    RegexRule(Kind.PASSWORD, PASSWORD, TIER_BLOCK, group="v", validator=_password_ok),
    RegexRule(Kind.EMAIL, EMAIL, TIER_EMAIL),
    RegexRule(Kind.PHONE, PHONE_KR, TIER_PHONE, validator=_phone_ok),
)


def empno_rule(pattern: str) -> RegexRule:
    """회사별 사번 형식(설정 ``PII_GUARD_EMPNO_REGEX``)으로 사번 규칙을 만든다."""
    return RegexRule(Kind.EMPNO, re.compile(rf"(?<![A-Za-z0-9-])(?:{pattern})(?![A-Za-z0-9-])"), TIER_EMPNO)
