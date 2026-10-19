"""``PIIGuardLLM`` — 외부 provider 를 감싸 호출 직전에 치환하고 응답에서 복원하는 래퍼.

``LLMFactory.create`` 가 대상 provider(gemini·openai·anthropic)에만 씌운다. 호출부는
``BaseLLM`` 계약(``invoke``/``stream``/``bind_tools``/``model``/``stateless`` …)만 보므로
수정하지 않는다. 위임 속성은 ``__getattr__`` 에 기대지 않고 **명시**한다 — supervisor 가
워커 래퍼에 ``bind_tools`` 한 뒤 ``get_bound_tools`` 를 읽어도 어긋나지 않게.

모드별 동작:
- off: inner 로 그대로 위임(치환·복원 없음).
- shadow: 치환을 **계산만** 하고 원문을 보낸다. 응답도 스캔해 "모델이 되돌려 말한
  식별자 수" 를 기록한다. 로그에 값은 없다.
- enforce: 치환본을 보내고 응답 텍스트·tool_calls 인자를 복원한다. 치환·복원이
  예외로 실패하면 **텍스트 없이** ``done(error)`` 만 낸다 — 그래프 ``_route_after_model`` 은
  text 가 비면 fallback 으로 가므로 원문이 새지 않는다. 미복원 토큰은 본문에선 라벨로
  바뀌고, 인자에선 남겨 executor 가 실행을 막는다.

이미지(``Message.images``)는 치환 대상이 아니다(문서 3장 논의 중). enforce 에서 외부로
나가면 경고 로그 1줄만 남긴다.
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, List, Optional

from agent_arch.agents.context import get_conversation_id, get_user_context
from agent_arch.llm.base import BaseLLM, LLMResponse, Message, StreamEvent, ToolCall
from agent_arch.llm.pii import audit
from agent_arch.llm.pii.engine import Restorer, Substituter, Substitution
from agent_arch.llm.pii.lexicon import Lexicon
from agent_arch.llm.pii.policy import Mode, block_kinds, resolve_mode, rules_for_settings, secret
from agent_arch.llm.pii.stream import StreamRestorer
from agent_arch.llm.pii.tokens import TokenIssuer, scope_id
from agent_arch.llm.pii.turn import build_turn_lexicon

logger = logging.getLogger(__name__)

#: 치환이 한 건이라도 있으면 system 끝에 붙이는 안내. 모델이 토큰을 추측·변형·반문하지 않게.
PLACEHOLDER_HINT = (
    "[자리표시 안내] `[[PERSON_…]]`, `[[EMAIL_…]]@도메인`, `[[PHONE_…]]`, `[[HANDLE_…]]`, `[[EMPNO_…]]` 꼴은 "
    "개인정보 보호를 위해 치환된 사람·연락처 자리표시입니다. 실제 값을 추측하거나 묻지 말고, "
    "답변과 도구 인자에 그 꼴 그대로(대괄호·글자 변경 없이) 옮겨 쓰세요. 자리표시끼리는 서로 다른 대상입니다."
)


class PIIGuardError(RuntimeError):
    """enforce 에서 치환·복원이 실패했다 — 응답을 내보내지 않는다."""


class _Prepared:
    __slots__ = ("mode", "messages", "sub", "restorer", "scan_ms", "lexicon_size", "scope", "images", "lexicon")

    def __init__(self, mode: Mode, messages: List[Message], sub: Optional[Substitution], restorer: Optional[Restorer],
                 scan_ms: float, lexicon_size: int, scope: str, images: int, lexicon: Optional[Lexicon] = None):
        self.mode, self.messages, self.sub, self.restorer = mode, messages, sub, restorer
        self.scan_ms, self.lexicon_size, self.scope, self.images = scan_ms, lexicon_size, scope, images
        self.lexicon = lexicon or Lexicon.empty()


class PIIGuardLLM(BaseLLM):
    """외부 provider 래퍼. ``inner`` 가 실제 provider 인스턴스."""

    def __init__(self, inner: BaseLLM, provider_name: str):
        # BaseLLM.__init__ 은 config 로 속성을 만든다 — 아래 property 들이 inner 로 위임하므로
        # 여기 값은 쓰이지 않지만, 계약(속성 존재)을 지키기 위해 부른다.
        super().__init__(dict(getattr(inner, "config", {}) or {}))
        self.inner = inner
        self.provider_name = (provider_name or "").lower()

    # ── 위임 속성(명시) ──────────────────────────────────────────────────────
    @property
    def config(self) -> dict:  # type: ignore[override]
        return self.inner.config

    @config.setter
    def config(self, value: dict) -> None:
        # BaseLLM.__init__ 이 대입한다 — inner 가 아직 없을 때는 무시.
        if getattr(self, "inner", None) is not None:
            self.inner.config = value

    @property
    def model(self):  # type: ignore[override]
        return self.inner.model

    @model.setter
    def model(self, value) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner.model = value

    @property
    def temperature(self):  # type: ignore[override]
        return self.inner.temperature

    @temperature.setter
    def temperature(self, value) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner.temperature = value

    @property
    def max_tokens(self):  # type: ignore[override]
        return self.inner.max_tokens

    @max_tokens.setter
    def max_tokens(self, value) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner.max_tokens = value

    @property
    def supports_function_calling(self):  # type: ignore[override]
        return self.inner.supports_function_calling

    @supports_function_calling.setter
    def supports_function_calling(self, value) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner.supports_function_calling = value

    @property
    def supports_streaming(self):  # type: ignore[override]
        return self.inner.supports_streaming

    @supports_streaming.setter
    def supports_streaming(self, value) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner.supports_streaming = value

    @property
    def stateless(self):  # type: ignore[override]
        return self.inner.stateless

    @stateless.setter
    def stateless(self, value) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner.stateless = value

    @property
    def _bound_tools(self):  # type: ignore[override]
        return self.inner._bound_tools

    @_bound_tools.setter
    def _bound_tools(self, value) -> None:
        if getattr(self, "inner", None) is not None:
            self.inner._bound_tools = value

    async def bind_tools(self, tools: List[Any]) -> "BaseLLM":
        await self.inner.bind_tools(tools)
        return self

    def get_bound_tools(self) -> List[Any]:
        return self.inner.get_bound_tools()

    def _convert_tool_to_schema(self, tool: Any) -> dict:
        return self.inner._convert_tool_to_schema(tool)

    def __getattr__(self, name: str) -> Any:
        # provider 전용 속성(client, api_key …)은 inner 로. __init__ 전엔 inner 가 없다.
        inner = self.__dict__.get("inner")
        if inner is None or name == "inner":
            raise AttributeError(name)
        return getattr(inner, name)

    def __repr__(self) -> str:
        return f"PIIGuardLLM({self.inner!r})"

    # ── 준비: 모드 판정 + 치환 ───────────────────────────────────────────────
    async def _prepare(self, messages: List[Message]) -> _Prepared:
        ctx = get_user_context() or {}
        company = ctx.get("company") or ctx.get("active_company")
        mode = await resolve_mode(self.provider_name, company)
        scope = scope_id(get_conversation_id(), ctx.get("sub"))
        images = sum(len(m.images or []) for m in messages)
        if mode == "off":
            return _Prepared("off", messages, None, None, 0.0, 0, scope, images)
        t0 = time.perf_counter()
        lexicon = await build_turn_lexicon()
        sub = self._substitute(messages, lexicon, scope)
        scan_ms = (time.perf_counter() - t0) * 1000
        if mode == "shadow":
            return _Prepared("shadow", messages, sub, None, scan_ms, len(lexicon), scope, images, lexicon)
        out = sub.messages
        if sub.token_count:
            out = _with_hint(out)
        if images:
            logger.warning("[pii_guard] enforce 인데 이미지 %d장이 외부(%s)로 나간다 — 이미지는 치환 범위 밖", images, self.provider_name)
        return _Prepared("enforce", out, sub, Restorer(sub.restore_map), scan_ms, len(lexicon), scope, images, lexicon)

    def _substitute(self, messages: List[Message], lexicon: Lexicon, scope: str) -> Substitution:
        issuer = TokenIssuer(secret(), scope)
        return Substituter(issuer, lexicon, rules=rules_for_settings(), block_kinds=block_kinds()).substitute_messages(messages)

    def _shadow_echo(self, prepared: _Prepared, text: str, tool_calls: Optional[List[ToolCall]]) -> dict:
        """shadow: 원문 응답을 스캔해 모델이 되돌려 말한 식별자 수."""
        try:
            msgs = [Message(role="assistant", content=text or "")]
            if tool_calls:
                msgs.append(Message(role="assistant", content="", tool_calls=[tc.model_dump() for tc in tool_calls]))
            resp_sub = self._substitute(msgs, prepared.lexicon, prepared.scope)
            return audit.echo_stats(resp_sub)
        except Exception:  # noqa: BLE001
            return {}

    # ── invoke ──────────────────────────────────────────────────────────────
    async def invoke(self, messages: List[Message], **kwargs) -> LLMResponse:
        try:
            prepared = await self._prepare(messages)
        except Exception as e:  # noqa: BLE001
            return self._fail(e, "invoke")
        resp = await self.inner.invoke(prepared.messages, **kwargs)
        if prepared.mode == "off":
            return resp
        if prepared.mode == "shadow":
            audit.log_shadow(provider=self.provider_name, model=self.model, scope=prepared.scope, sub=prepared.sub,
                             scan_ms=prepared.scan_ms, lexicon_size=prepared.lexicon_size,
                             echo=self._shadow_echo(prepared, resp.content, resp.tool_calls), images=prepared.images)
            return resp
        try:
            text = prepared.restorer.restore_text(resp.content or "")
            calls, unresolved = _restore_calls(prepared.restorer, resp.tool_calls)
            unresolved = text.unresolved + unresolved
        except Exception as e:  # noqa: BLE001
            return self._fail(e, "invoke")
        audit.log_enforce(provider=self.provider_name, model=self.model, scope=prepared.scope, sub=prepared.sub,
                          scan_ms=prepared.scan_ms, lexicon_size=prepared.lexicon_size, unresolved=unresolved,
                          images=prepared.images)
        return resp.model_copy(update={"content": text.value, "tool_calls": calls})

    def _fail(self, e: Exception, where: str) -> LLMResponse:
        logger.error("[pii_guard] enforce %s 실패 — 응답을 내보내지 않는다: %r", where, e, exc_info=True)
        return LLMResponse(content="", tool_calls=None, finish_reason="error", usage=None)

    # ── stream ──────────────────────────────────────────────────────────────
    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[StreamEvent]:
        try:
            prepared = await self._prepare(messages)
        except Exception as e:  # noqa: BLE001
            logger.error("[pii_guard] enforce stream 준비 실패 — 텍스트 없이 종료: %r", e, exc_info=True)
            yield StreamEvent(type="done", finish_reason="error")
            return
        if prepared.mode == "off":
            async for ev in self.inner.stream(messages, **kwargs):
                yield ev
            return
        if prepared.mode == "shadow":
            text_buf: list[str] = []
            calls: Optional[List[ToolCall]] = None
            async for ev in self.inner.stream(messages, **kwargs):
                if ev.type == "text" and ev.content:
                    text_buf.append(ev.content)
                elif ev.type == "tool_calls":
                    calls = ev.tool_calls
                yield ev
            audit.log_shadow(provider=self.provider_name, model=self.model, scope=prepared.scope, sub=prepared.sub,
                             scan_ms=prepared.scan_ms, lexicon_size=prepared.lexicon_size,
                             echo=self._shadow_echo(prepared, "".join(text_buf), calls), images=prepared.images)
            return

        buf = StreamRestorer(prepared.restorer)
        unresolved_calls: list[str] = []
        try:
            async for ev in self.inner.stream(prepared.messages, **kwargs):
                if ev.type == "text":
                    out = buf.feed(ev.content or "")
                    if out:
                        yield StreamEvent(type="text", content=out)
                elif ev.type == "tool_calls":
                    tail = buf.flush()
                    if tail:
                        yield StreamEvent(type="text", content=tail)
                    calls, unresolved_calls = _restore_calls(prepared.restorer, ev.tool_calls)
                    yield StreamEvent(type="tool_calls", tool_calls=calls)
                elif ev.type == "done":
                    tail = buf.flush()
                    if tail:
                        yield StreamEvent(type="text", content=tail)
                    audit.log_enforce(provider=self.provider_name, model=self.model, scope=prepared.scope,
                                      sub=prepared.sub, scan_ms=prepared.scan_ms, lexicon_size=prepared.lexicon_size,
                                      unresolved=buf.unresolved + unresolved_calls, images=prepared.images)
                    yield ev
                else:
                    yield ev
        except PIIGuardError as e:
            logger.error("[pii_guard] enforce stream 복원 실패 — 종료: %r", e, exc_info=True)
            yield StreamEvent(type="done", finish_reason="error")


def _restore_calls(restorer: Restorer, tool_calls: Optional[List[ToolCall]]) -> tuple[Optional[List[ToolCall]], list[str]]:
    if not tool_calls:
        return tool_calls, []
    out: List[ToolCall] = []
    unresolved: list[str] = []
    for tc in tool_calls:
        r = restorer.restore_value(tc.arguments or {})
        unresolved.extend(r.unresolved)
        out.append(tc.model_copy(update={"arguments": r.value}) if r.restored or r.unresolved else tc)
    return out, unresolved


def _with_hint(messages: List[Message]) -> List[Message]:
    """system 메시지 끝에 자리표시 안내를 붙인다. system 이 없으면 맨 앞에 하나 만든다."""
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "system":
            if PLACEHOLDER_HINT in (m.content or ""):
                return messages
            return [*messages[:i], m.model_copy(update={"content": f"{m.content or ''}\n\n{PLACEHOLDER_HINT}".strip()}), *messages[i + 1:]]
    return [Message(role="system", content=PLACEHOLDER_HINT), *messages]
