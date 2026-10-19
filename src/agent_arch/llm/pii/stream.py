"""스트림 복원 버퍼 — 토큰이 청크 경계에 걸쳐 쪼개져 와도 복원한다.

provider 는 delta 를 그대로 ``StreamEvent(text)`` 로 내므로 ``[[PERSON_3f9a1c2d]]`` 가
``[[PER`` / ``SON_3f9a`` / ``1c2d]]`` 로 나뉘어 올 수 있다. 꼬리가 **토큰의 접두일 때만**
보류하고(일반 문장은 지연 0), 상한(``hold_max``)을 넘으면 강제로 내보낸다.
``StreamSanitizer`` 와 같은 feed/flush 계약이다.
"""
from __future__ import annotations

import re

from agent_arch.llm.pii.engine import Restorer

_KINDS = ("PERSON", "EMAIL", "PHONE", "HANDLE", "EMPNO")


def _proper_prefixes(word: str) -> str:
    # P|PE|PER|... 를 중첩 optional 로: P(?:E(?:R(?:S(?:O(?:N)?)?)?)?)?
    body = ""
    for ch in reversed(word[1:]):
        body = f"(?:{ch}{body})?"
    return word[0] + body


_KIND_PREFIX = "|".join(_proper_prefixes(k) for k in _KINDS)
#: 버퍼 끝이 "토큰이 되는 중" 인 꼴(종류 접두 또는 여는 괄호로 시작). search 로 가장 이른 위치를 찾는다.
_PARTIAL_RE = re.compile(
    rf"(?:(?:(?:\\?\[){{1,2}}\s{{0,2}})?(?:{_KIND_PREFIX})(?:_[0-9A-Fa-f]{{0,12}})?"
    rf"(?:\s{{0,2}}(?:\\?\]){{0,2}}\\?)?(?:@[A-Za-z0-9.-]{{0,64}})?|(?:\\?\[){{1,2}}\\?\s{{0,2}}|\\)\Z",
    re.IGNORECASE,
)
#: 여닫는 괄호 수가 맞는 완성 토큰 — 이것은 다음 청크를 기다릴 필요가 없다.
_CLOSED_RE = re.compile(
    r"(?:(?:\\?\[){2}\s*[A-Z]+_[0-9A-Fa-f]{8,12}\s*(?:\\?\]){2}|\\?\[\s*[A-Z]+_[0-9A-Fa-f]{8,12}\s*\\?\])",
    re.IGNORECASE,
)


class StreamRestorer:
    """``feed(chunk) -> 내보낼 텍스트``, ``flush() -> 잔여``. ``unresolved`` 에 미등록 토큰 누적."""

    def __init__(self, restorer: Restorer, *, hold_max: int = 96, unknown: str = "label"):
        self._restorer = restorer
        self._buf = ""
        self._hold_max = hold_max
        self._unknown = unknown
        self.unresolved: list[str] = []
        self.restored = 0

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        cut = self._cut_point(self._buf)
        if len(self._buf) - cut > self._hold_max:
            cut = len(self._buf)
        if cut <= 0:
            return ""
        head, self._buf = self._buf[:cut], self._buf[cut:]
        return self._restore(head)

    def flush(self) -> str:
        head, self._buf = self._buf, ""
        return self._restore(head) if head else ""

    def _restore(self, text: str) -> str:
        r = self._restorer.restore_text(text, unknown=self._unknown)
        self.unresolved.extend(r.unresolved)
        self.restored += r.restored
        return r.value

    @staticmethod
    def _cut_point(buf: str) -> int:
        """버퍼에서 안전하게 내보낼 수 있는 길이. 꼬리가 토큰 접두가 아니면 전부."""
        m = _PARTIAL_RE.search(buf)
        if m is None:
            return len(buf)
        tail = buf[m.start():]
        # 꼬리가 괄호까지 닫힌 완성 토큰이고 이메일 종류가 아니면 도메인을 기다릴 필요가 없다.
        # (괄호 없는 토큰은 hex 가 12자로 더 올 수 있어 다음 청크까지 기다린다.)
        if _CLOSED_RE.fullmatch(tail) and not tail.upper().lstrip("\\[ ").startswith("EMAIL"):
            return len(buf)
        return m.start()
