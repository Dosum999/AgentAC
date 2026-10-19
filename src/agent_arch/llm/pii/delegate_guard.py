"""ChatGPT 위탁 레인(app/services/chatgpt/delegate) 용 가명화.

위탁 레인은 ``LLMFactory`` 를 거치지 않고 OpenAI Responses API 를 직접 부르므로 래퍼가
닿지 않는다. 같은 엔진으로 이력·질문 텍스트를 치환하고 출력 델타를 복원한다.

이미지(base64)·첨부(OpenAI 파일 업로드)는 치환이 불가능하다. enforce 회사에서는 그런
위탁을 **거부**하고(``AttachmentsRefused``) 호출자가 일반 답변 경로로 되돌린다. 내장
web_search 는 모델이 토큰만 보므로 검색어에 원문이 들어갈 수 없다.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterable, List, Optional

from agent_arch.agents.context import get_conversation_id, get_user_context
from agent_arch.llm.base import Message
from agent_arch.llm.pii import audit
from agent_arch.llm.pii.engine import Restorer, Substituter, Substitution
from agent_arch.llm.pii.guard_llm import PLACEHOLDER_HINT
from agent_arch.llm.pii.lexicon import Lexicon
from agent_arch.llm.pii.policy import Mode, block_kinds, resolve_mode, rules_for_settings, secret
from agent_arch.llm.pii.stream import StreamRestorer
from agent_arch.llm.pii.tokens import TokenIssuer, scope_id
from agent_arch.llm.pii.turn import build_turn_lexicon

logger = logging.getLogger(__name__)

PROVIDER = "openai"


class AttachmentsRefused(RuntimeError):
    """enforce 회사의 위탁에 이미지·첨부가 있다 — 외부로 보낼 수 없다."""


class DelegateGuard:
    """위탁 1회분. ``prepare()`` 로 만들고, 입력은 ``messages``/``user_text``, 출력은 ``feed``/``flush``."""

    def __init__(self, mode: Mode, messages: List[Message], user_text: str, sub: Optional[Substitution],
                 restorer: Optional[Restorer], scope: str, scan_ms: float, lexicon_size: int,
                 lexicon: Optional[Lexicon] = None):
        self.mode = mode
        self.messages = messages
        self.user_text = user_text
        self.sub = sub
        self._restorer = restorer
        self._stream = StreamRestorer(restorer) if restorer is not None else None
        self.scope, self.scan_ms, self.lexicon_size = scope, scan_ms, lexicon_size
        self._lexicon = lexicon or Lexicon.empty()

    @property
    def instructions_suffix(self) -> str:
        if self.mode == "enforce" and self.sub is not None and self.sub.token_count:
            return "\n- " + PLACEHOLDER_HINT
        return ""

    @classmethod
    async def prepare(cls, user_text: str, messages: Any, *, company: Optional[str],
                      images: Optional[Iterable] = None, attachments: Optional[Iterable] = None) -> "DelegateGuard":
        ctx = get_user_context() or {}
        scope = scope_id(get_conversation_id(), ctx.get("sub"))
        mode = await resolve_mode(PROVIDER, company)
        convo = [m for m in (messages or []) if getattr(m, "role", None) in ("user", "assistant") and not getattr(m, "tool_calls", None)]
        if mode == "off":
            return cls("off", list(messages or []), user_text, None, None, scope, 0.0, 0)
        if mode == "enforce" and (list(images or []) or list(attachments or [])):
            logger.warning("[pii_guard] enforce 회사의 ChatGPT 위탁에 이미지·첨부가 있어 거부 (company=%s)", company)
            raise AttachmentsRefused("이미지·첨부는 가명화할 수 없어 외부 위탁에 보내지 않습니다")
        t0 = time.perf_counter()
        lexicon = await build_turn_lexicon()
        sub = cls._substitute([*convo, Message(role="user", content=user_text or "")], lexicon, scope)
        scan_ms = (time.perf_counter() - t0) * 1000
        if mode == "shadow":
            return cls("shadow", list(messages or []), user_text, sub, None, scope, scan_ms, len(lexicon), lexicon)
        return cls("enforce", sub.messages[:-1], sub.messages[-1].content or "", sub, Restorer(sub.restore_map),
                   scope, scan_ms, len(lexicon), lexicon)

    @staticmethod
    def _substitute(messages: List[Message], lexicon: Lexicon, scope: str) -> Substitution:
        return Substituter(TokenIssuer(secret(), scope), lexicon, rules=rules_for_settings(),
                           block_kinds=block_kinds()).substitute_messages(messages)

    # ── 출력 ──────────────────────────────────────────────────────────────────
    def feed(self, delta: str) -> str:
        return self._stream.feed(delta) if self._stream is not None else (delta or "")

    def flush(self) -> str:
        return self._stream.flush() if self._stream is not None else ""

    def finish(self, body: str) -> None:
        """응답이 끝난 뒤 기록. shadow 는 응답 원문을 스캔해 echo 를 집계한다."""
        if self.mode == "off" or self.sub is None:
            return
        if self.mode == "shadow":
            echo = {}
            try:
                echo = audit.echo_stats(self._substitute([Message(role="assistant", content=body or "")], self._lexicon, self.scope))
            except Exception:  # noqa: BLE001
                pass
            audit.log_shadow(provider="chatgpt", model=None, scope=self.scope, sub=self.sub, scan_ms=self.scan_ms,
                             lexicon_size=self.lexicon_size, echo=echo)
            return
        audit.log_enforce(provider="chatgpt", model=None, scope=self.scope, sub=self.sub, scan_ms=self.scan_ms,
                          lexicon_size=self.lexicon_size,
                          unresolved=self._stream.unresolved if self._stream is not None else None)
