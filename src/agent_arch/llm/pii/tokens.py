"""자리표시 토큰 — 종류, 정규화, HMAC 발급, 충돌 처리, 렌더.

토큰 = ``[[PERSON_3f9a1c2d]]``. hex 는 HMAC-SHA256(비밀‖범위, "KIND:정규화값") 앞 8자라
같은 대화(범위)에서는 같은 값이 항상 같은 토큰이 되고, 매핑을 저장할 필요가 없다.

**일괄 발급**(``TokenIssuer.assign``): 호출 한 번에 모인 값 전체를 정렬해 8자 충돌을
검사하고, 충돌 그룹은 **전원** 12자로 발급한다. 어느 값이 먼저 등장했든 결과가 같아
이력 절단과 무관하게 결정적이다(이력의 assistant tool_calls 를 다시 치환해 보낼 때
Gemini thought_signature 짝이 깨지지 않는 조건).
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import re
from enum import StrEnum
from typing import Iterable

logger = logging.getLogger(__name__)


class Kind(StrEnum):
    """토큰 종류. 복원형은 응답에서 원문으로 되돌리고, 차단형은 라벨로만 바꾼다."""

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    HANDLE = "HANDLE"
    EMPNO = "EMPNO"
    SSN = "SSN"
    CARD = "CARD"
    ACCOUNT = "ACCOUNT"
    PASSWORD = "PASSWORD"
    APIKEY = "APIKEY"

    @property
    def restorable(self) -> bool:
        return self in RESTORABLE


RESTORABLE = frozenset({Kind.PERSON, Kind.EMAIL, Kind.PHONE, Kind.HANDLE, Kind.EMPNO})
BLOCKABLE = frozenset({Kind.SSN, Kind.CARD, Kind.ACCOUNT, Kind.PASSWORD, Kind.APIKEY})

#: 차단형 라벨 — 복원 없음.
BLOCKED_LABEL: dict[Kind, str] = {
    Kind.SSN: "[차단됨:주민번호]",
    Kind.CARD: "[차단됨:카드번호]",
    Kind.ACCOUNT: "[차단됨:계좌번호]",
    Kind.PASSWORD: "[차단됨:비밀번호]",
    Kind.APIKEY: "[차단됨:API키]",
}

#: 복원표에 없는 토큰(모델이 지어냄)을 본문에서 대신하는 라벨.
UNKNOWN_LABEL: dict[Kind, str] = {
    Kind.PERSON: "(확인되지 않은 사람)",
    Kind.EMAIL: "(확인되지 않은 이메일)",
    Kind.PHONE: "(확인되지 않은 번호)",
    Kind.HANDLE: "(확인되지 않은 계정)",
    Kind.EMPNO: "(확인되지 않은 사번)",
}

SHORT_HEX = 8
LONG_HEX = 12

_WS = re.compile(r"\s+")
_NON_DIGIT = re.compile(r"\D")


def normalize(kind: Kind, raw: str) -> str:
    """토큰 발급용 정규화 — 표기 차이(대소문자·하이픈·공백)가 다른 토큰을 만들지 않게."""
    value = (raw or "").strip()
    if kind is Kind.EMAIL:
        return value.lower()
    if kind is Kind.PHONE:
        return _NON_DIGIT.sub("", value)
    if kind is Kind.HANDLE:
        return value.lstrip("@").lower()
    if kind is Kind.PERSON:
        return _WS.sub("", value)
    return value


def scope_id(conversation_id: str | None, user_sub: str | None = None) -> str:
    """HMAC 범위 — 대화 ID, 없으면(첫 턴 lazy create·게스트) 사용자, 그것도 없으면 익명."""
    if conversation_id:
        return f"conv:{conversation_id}"
    if user_sub:
        return f"user:{user_sub}"
    return "anon"


def render(kind: Kind, hexid: str) -> str:
    """``[[KIND_hex]]``. 이메일 도메인·핸들의 ``@`` 는 호출자가 토큰 밖에 붙인다."""
    return f"[[{kind.value}_{hexid}]]"


def canonical_key(kind: Kind | str, hexid: str) -> str:
    """복원표 키 — 종류 대문자 + hex 소문자(모델이 대소문자를 바꿔도 같은 키)."""
    k = kind.value if isinstance(kind, Kind) else str(kind).upper()
    return f"{k}_{hexid.lower()}"


class TokenIssuer:
    """범위(대화)별 토큰 발급기.

    Args:
        secret: 서버 비밀. 대화 ID 만으로 키를 만들면 대화 ID 를 아는 클라이언트가
            임직원 명단으로 토큰을 역산할 수 있어 서버 비밀을 섞는다.
        scope: ``scope_id()`` 결과.
    """

    def __init__(self, secret: str | bytes, scope: str):
        if not secret:
            raise ValueError("PII token secret 이 비어 있다")
        key = secret if isinstance(secret, bytes) else secret.encode("utf-8")
        self._key = hashlib.sha256(key + b"\x00" + scope.encode("utf-8")).digest()
        self.scope = scope

    def digest(self, kind: Kind, normalized: str) -> str:
        msg = f"{kind.value}:{normalized}".encode("utf-8")
        return hmac.new(self._key, msg, hashlib.sha256).hexdigest()

    def assign(self, pairs: Iterable[tuple[Kind, str]]) -> dict[tuple[Kind, str], str]:
        """(종류, 정규화값) 집합에 hex 를 배정한다. 8자 접두가 겹치는 그룹은 전원 12자.

        Returns:
            {(kind, normalized): hexid}
        """
        unique = sorted(set(pairs), key=lambda p: (p[0].value, p[1]))
        digests = {p: self.digest(*p) for p in unique}
        by_short: dict[str, list[tuple[Kind, str]]] = {}
        for p, d in digests.items():
            by_short.setdefault(d[:SHORT_HEX], []).append(p)
        out: dict[tuple[Kind, str], str] = {}
        for short, group in by_short.items():
            if len(group) == 1:
                out[group[0]] = short
                continue
            logger.warning("[pii_guard] 토큰 8자 충돌 — 12자로 발급 (scope=%s n=%d)", self.scope, len(group))
            for p in group:
                out[p] = digests[p][:LONG_HEX]
        return out
