"""Anthropic(Claude) LLM 프로바이더 구현.

Messages API(`https://api.anthropic.com/v1/messages`)를 httpx 로 직접 호출한다.

**SDK 를 쓰지 않는 이유**: `anthropic` 패키지가 이 이미지에 없다. 도면 분석에서 회사별
API 키로 Claude 를 고를 수 있게 하려고 프로바이더를 추가하는 것인데, 그 때문에 의존성
설치와 이미지 리빌드를 요구하면 배포 경로가 길어진다. Messages API 는 요청·응답 스키마가
단순하고 스트리밍도 표준 SSE 라 httpx 로 충분하다. (SDK 를 도입할 때는 이 클래스의
`_request`/`_stream_lines` 만 갈아끼우면 된다.)

OpenAI 계열과 다른 네 가지:
    ① system 은 메시지 배열이 아니라 최상위 `system` 필드다.
    ② `max_tokens` 가 필수다(생략하면 400).
    ③ PDF 는 `document` 블록으로 보낸다 — 이미지 블록으로 보내면 거부된다.
    ④ 도구 호출은 메시지 content 블록이다 — assistant 의 `tool_use` 블록과, 바로 다음
       user 메시지의 `tool_result` 블록이 id 로 짝을 이뤄야 한다(짝이 깨지면 400).
       그래서 ``_convert_messages`` 가 짝을 검사해 고아 블록은 텍스트로 내린다.

도구 호출은 2026-10 에 붙였다. 그 전까지는 도면 분석(도구 없음) 전용이었고, 도면 레인은
bind_tools 를 부르지 않으므로 여전히 tools 없이 호출된다.
"""

import json
import logging
from typing import Any, AsyncIterator, ClassVar, Dict, List, Optional, Sequence

import httpx

from agent_arch.llm.base import BaseLLM, LLMResponse, Message, StreamEvent, ToolCall
from agent_arch.llm.factory import register_llm_provider
from agent_arch.services.monitoring.error_events import event_extra
from agent_arch.services.usage.meter import meter_llm_call

logger = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_API_VERSION = "2023-06-01"

# Claude 가 문서로 직접 읽는 형식. 그 외 이미지 MIME 은 image 블록으로 보낸다.
_DOCUMENT_MIME_TYPES = frozenset({"application/pdf"})

# stop_reason → 표준 finish_reason (BaseLLM 규약: stop|tool_calls|length|error)
_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
}


def _tool_choice_param(tool_mode: str, allowed_tools: Optional[Sequence[str]]) -> Dict[str, Any]:
    """tool_mode(AUTO|ANY|NONE) + allowed_tools → Anthropic ``tool_choice`` 값.

    OpenAI 계열의 ``app.llm.tool_choice._tool_choice_param`` 과 같은 규약이다.
    ANY + allowed_tools 하나면 그 도구로 고정(guide-first)하고, 여러 개면 ``any`` 로 완화한다
    (Anthropic 도 도구 하나만 지정할 수 있다). NONE 은 ``{"type": "none"}`` — tools 를 아예
    빼지 않는 이유는 히스토리에 tool_use/tool_result 블록이 있으면 tools 정의가 있어야
    400 이 안 나기 때문이다.
    """
    if tool_mode == "ANY":
        if allowed_tools and len(allowed_tools) == 1:
            return {"type": "tool", "name": allowed_tools[0]}
        return {"type": "any"}
    if tool_mode == "NONE":
        return {"type": "none"}
    return {"type": "auto"}


def _tool_call_as_text(name: str, arguments: Any) -> str:
    """짝이 깨진 tool_use 를 사람이 읽는 한 줄로 내린다."""
    try:
        args = json.dumps(arguments or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        args = str(arguments)
    return f"[도구 호출: {name}({args})]"


@register_llm_provider("anthropic")
class AnthropicLLM(BaseLLM):
    """Claude(Anthropic Messages API) 구현.

    Attributes:
        api_key: Anthropic API 키.
        timeout: 요청 타임아웃(초). 도면 판독은 수십 초가 걸린다.
    """

    # factory.py 가 settings 에서 읽어올 키 (호출자가 kwargs 로 준 값이 우선)
    _settings_config: ClassVar[Dict[str, str]] = {"api_key": "anthropic_api_key"}

    def __init__(self, config: Dict[str, Any]):
        """AnthropicLLM 을 초기화한다.

        Args:
            config: 설정 딕셔너리.
                - api_key: Anthropic API 키 (필수)
                - model: 모델 이름 (예: claude-sonnet-4-5)
                - max_tokens: 최대 생성 토큰 (Anthropic 은 필수 파라미터)
                - temperature·top_p: 받되 보내지 않는다(Claude 5 계열 폐기, _payload 주석)
                - base_url: 프록시·게이트웨이를 쓸 때만 (선택)
                - timeout: 요청 타임아웃(초)

        Raises:
            ValueError: API 키가 없는 경우.
        """
        super().__init__(config)
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        self.api_key = api_key
        self.base_url = (config.get("base_url") or _API_URL).rstrip("/")
        if not self.base_url.endswith("/messages"):
            self.base_url = f"{self.base_url}/v1/messages"
        self.timeout = float(config.get("timeout") or 300.0)

    # ── 요청 조립 ────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }

    def _convert_messages(
        self, messages: List[Message], tools_defined: bool = False
    ) -> tuple[Optional[str], List[dict]]:
        """표준 Message 를 Anthropic 형식으로 변환한다.

        도구 호출 블록의 짝 규칙(Anthropic 은 이걸 어기면 400):
            - assistant 의 ``tool_use`` 는 **바로 다음** user 메시지에 같은 id 의
              ``tool_result`` 가 있어야 한다.
            - ``tool_result`` 는 직전 assistant 의 ``tool_use`` id 여야 한다.
            - 둘 다 요청에 ``tools`` 가 정의돼 있어야 한다.
        대화 압축·히스토리 top_k 로 짝이 깨진 블록, 또는 tools 없이 부르는 레인(검증·요약)
        에 섞여 온 블록은 **텍스트로 내린다**. OpenAI 계열은 이런 입력을 관대하게 받아서
        그쪽 변환부에는 이 방어가 없다.

        Args:
            messages: 대화 메시지.
            tools_defined: 이번 요청에 tools 를 실어 보내는지. False 면 도구 블록을 전부
                텍스트로 내린다.

        Returns:
            (system 문자열 또는 None, messages 배열).
        """
        system_parts: List[str] = []
        out: List[dict] = []
        # 직전 assistant 가 낸 tool_use id → 블록 참조. 다음 user 턴의 tool_result 가 소비한다.
        pending: Dict[str, dict] = {}

        def _demote_pending() -> None:
            """짝을 못 찾은 tool_use 를 텍스트로 바꾼다(블록 객체를 제자리에서 수정)."""
            for block in pending.values():
                name, args = block.get("name", ""), block.get("input")
                block.clear()
                block.update({"type": "text", "text": _tool_call_as_text(name, args)})
            pending.clear()

        for msg in messages:
            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
                continue

            role = "assistant" if msg.role == "assistant" else "user"
            blocks: List[dict] = []

            if msg.role == "tool":
                # tool_result 는 user role 로 간다. 짝(pending)에 있을 때만 블록, 아니면 텍스트.
                call_id = msg.tool_call_id or ""
                if tools_defined and call_id in pending:
                    pending.pop(call_id)
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": msg.content or "",
                    })
                else:
                    blocks.append({
                        "type": "text",
                        "text": f"[도구 결과: {msg.name or call_id}]\n{msg.content or ''}",
                    })
                if out and out[-1]["role"] == "user":
                    out[-1]["content"].extend(blocks)
                else:
                    out.append({"role": "user", "content": blocks})
                continue

            # tool 메시지가 아닌 턴이 왔는데 아직 짝을 못 찾은 tool_use 가 있으면 영영 짝이
            # 없다(user 텍스트 턴이든 다음 assistant 턴이든) → 텍스트로 내린다.
            if pending:
                _demote_pending()
            # 파일·이미지를 텍스트보다 앞에 둔다 — 지시문이 뒤에 오면 모델이 문서를
            # 읽기 전에 지시를 해석해 "첨부가 없다" 고 답하는 경우가 있다.
            for item in msg.images or []:
                if item.mime_type in _DOCUMENT_MIME_TYPES:
                    blocks.append({
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": item.mime_type,
                            "data": item.data,
                        },
                    })
                else:
                    blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": item.mime_type,
                            "data": item.data,
                        },
                    })
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            if role == "assistant" and msg.tool_calls:
                for tc in msg.tool_calls:
                    call_id = str(tc.get("id") or f"call_{tc.get('name', '')}")
                    name = str(tc.get("name") or "")
                    args = tc.get("arguments") or {}
                    if tools_defined:
                        block = {"type": "tool_use", "id": call_id, "name": name, "input": args}
                        pending[call_id] = block
                    else:
                        block = {"type": "text", "text": _tool_call_as_text(name, args)}
                    blocks.append(block)
            if not blocks:
                continue

            # 연속된 같은 role 은 합친다 — Anthropic 은 role 교대를 요구한다.
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": blocks})

        # 히스토리 끝이 tool_use 로 끝나면(결과를 못 받은 채 압축된 턴) 그것도 짝이 없다.
        _demote_pending()

        return ("\n\n".join(system_parts) or None), out

    def _tool_schemas(self) -> List[dict]:
        """바인딩된 도구 스키마. 도구 호출을 켜지 않았거나 바인딩이 없으면 빈 목록."""
        if not (self._bound_tools and self.supports_function_calling):
            return []
        return [self._convert_tool_to_schema(t) for t in self._bound_tools]

    def _payload(self, messages: List[Message], kwargs: dict, stream: bool) -> dict:
        tools = self._tool_schemas()
        system, converted = self._convert_messages(messages, tools_defined=bool(tools))
        payload: Dict[str, Any] = {
            "model": kwargs.get("model") or self.model,
            # Anthropic 은 max_tokens 필수 — 없으면 400 이라 기본값을 반드시 채운다.
            "max_tokens": int(kwargs.get("max_tokens") or self.max_tokens or 4096),
            "messages": converted,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = _tool_choice_param(
                kwargs.get("tool_mode", "AUTO"), kwargs.get("allowed_tools")
            )
        # temperature·top_p·top_k 는 **보내지 않는다**. Claude Opus 4.6 이후 모델은 샘플링 인자를
        # 폐기해 1.0 외의 값은 400 이다(2026-10-12 실측: "`temperature` is deprecated for this
        # model" — 라우터·스트림이 통째로 죽었다). yaml 이나 호출부(도면 판독 temperature=0.0,
        # 도면 레인 tuning top_p)가 값을 줘도 여기서 무시한다 — 다른 프로바이더와 공유하는
        # 호출 계약을 바꾸지 않고 이 프로바이더만 모델 기본에 맡긴다.
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _usage(raw: Optional[dict]) -> Optional[Dict[str, int]]:
        """Anthropic usage 를 표준 키로 옮긴다."""
        if not raw:
            return None
        prompt = int(raw.get("input_tokens") or 0)
        completion = int(raw.get("output_tokens") or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }

    @staticmethod
    def _error_message(resp: httpx.Response, body: str) -> str:
        """오류 응답에서 사람이 읽을 메시지를 뽑는다."""
        try:
            payload = json.loads(body)
            err = payload.get("error") or {}
            detail = err.get("message") or ""
        except (ValueError, TypeError):
            detail = body[:200]
        return f"Anthropic {resp.status_code}: {detail}"

    # ── 호출 ─────────────────────────────────────────────────────

    def _convert_tool_to_schema(self, tool: Any) -> Dict[str, Any]:
        """BaseTool 을 Anthropic tools 항목으로 변환한다.

        OpenAI 의 ``function.parameters`` 가 여기서는 ``input_schema`` 다. 그 외 구조는 같다.

        Args:
            tool: BaseTool 인스턴스(name·description·parameters·required_params).

        Returns:
            ``{"name", "description", "input_schema"}`` 딕셔너리.
        """
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": {
                "type": "object",
                "properties": tool.parameters,
                "required": tool.required_params or [],
            },
        }

    @staticmethod
    def _parse_tool_uses(blocks: List[dict]) -> List[ToolCall]:
        """응답 content 의 ``tool_use`` 블록을 ToolCall 로 옮긴다."""
        calls: List[ToolCall] = []
        for block in blocks:
            if block.get("type") != "tool_use":
                continue
            name = str(block.get("name") or "unknown")
            args = block.get("input")
            if not isinstance(args, dict):
                args = {}
            calls.append(ToolCall(
                id=str(block.get("id") or f"call_{name}"),
                name=name,
                arguments=args,
            ))
        return calls

    async def invoke(self, messages: List[Message], **kwargs) -> LLMResponse:
        """Messages API 를 1회 호출한다.

        도구가 바인딩돼 있고 ``supports_function_calling`` 이면 tools 를 실어 보내고,
        응답의 ``tool_use`` 블록을 ToolCall 로 돌려준다(도면 레인은 바인딩이 없어 그대로).

        Args:
            messages: 대화 메시지.
            **kwargs: model·max_tokens·tool_mode·allowed_tools override(temperature·top_p 는 무시).

        Returns:
            LLMResponse. 텍스트 블록을 이어붙인 content + tool_calls.

        Raises:
            RuntimeError: API 가 2xx 를 주지 않은 경우.
        """
        payload = self._payload(messages, kwargs, stream=False)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.base_url, json=payload, headers=self._headers())
            body = resp.text
            if resp.status_code >= 400:
                raise RuntimeError(self._error_message(resp, body))
            data = resp.json()

        content_blocks = data.get("content") or []
        text = "".join(
            block.get("text") or ""
            for block in content_blocks
            if block.get("type") == "text"
        )
        tool_calls = self._parse_tool_uses(content_blocks)
        usage = self._usage(data.get("usage"))
        meter_llm_call("anthropic", payload.get("model") or self.model, usage, source="anthropic.invoke")
        finish_reason = _STOP_REASON_MAP.get(data.get("stop_reason") or "", "stop")
        if tool_calls:
            finish_reason = "tool_calls"
        return LLMResponse(
            content=text,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def stream(self, messages: List[Message], **kwargs) -> AsyncIterator[StreamEvent]:
        """Messages API 를 스트리밍으로 호출한다.

        표준 SSE 를 그대로 읽는다. 텍스트 델타는 즉시 흘리고, ``tool_use`` 블록은
        ``content_block_start`` 에서 id·name 을 잡고 ``input_json_delta`` 를 모아
        ``content_block_stop`` 에서 완성한다. 완성된 호출은 스트림 끝에 tool_calls 이벤트
        한 번으로 내고, 마지막에 done 이벤트를 정확히 한 번 낸다(BaseLLM 규약).

        Claude 는 도구를 부르기 전에 짧은 텍스트("확인해 보겠습니다")를 먼저 내는 일이
        흔하다. OpenAI 계열처럼 "먼저 온 종류로 모드를 굳히고 나머지를 버리면" 그 뒤의
        tool_use 가 사라져 도구 체인이 끊기므로, 여기서는 둘 다 흘린다 — react 그래프는
        text_buf 와 iter_tool_calls 를 따로 모으므로 둘이 같이 와도 된다.

        Args:
            messages: 대화 메시지.
            **kwargs: model·max_tokens·tool_mode·allowed_tools override(temperature·top_p 는 무시).

        Yields:
            StreamEvent: text N회 → (tool_calls 1회) → done 1회. 실패 시 done(finish_reason="error").
        """
        payload = self._payload(messages, kwargs, stream=True)
        finish_reason = "stop"
        usage: Optional[Dict[str, int]] = None
        input_tokens = 0
        output_tokens = 0
        # content block index → {"id", "name", "buf"} (tool_use 블록만)
        tool_accum: Dict[int, Dict[str, Any]] = {}
        tool_calls: List[ToolCall] = []

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", self.base_url, json=payload, headers=self._headers()
                ) as resp:
                    if resp.status_code >= 400:
                        body = (await resp.aread()).decode("utf-8", "replace")
                        logger.warning(
                            "Anthropic 스트림 실패: %s", self._error_message(resp, body),
                            extra=event_extra(provider="anthropic", status_code=resp.status_code),
                        )
                        yield StreamEvent(type="done", finish_reason="error")
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        raw = line[len("data:"):].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        try:
                            event = json.loads(raw)
                        except (ValueError, TypeError):
                            continue

                        etype = event.get("type")
                        if etype == "content_block_delta":
                            delta = event.get("delta") or {}
                            dtype = delta.get("type")
                            if dtype == "text_delta" and delta.get("text"):
                                yield StreamEvent(type="text", content=delta["text"])
                            elif dtype == "input_json_delta":
                                entry = tool_accum.get(int(event.get("index") or 0))
                                if entry is not None:
                                    entry["buf"] += delta.get("partial_json") or ""
                        elif etype == "content_block_start":
                            block = event.get("content_block") or {}
                            if block.get("type") == "tool_use":
                                tool_accum[int(event.get("index") or 0)] = {
                                    "id": block.get("id"),
                                    "name": block.get("name"),
                                    "buf": "",
                                }
                        elif etype == "content_block_stop":
                            entry = tool_accum.pop(int(event.get("index") or 0), None)
                            if entry is not None:
                                tool_calls.append(self._finalize_tool_call(entry))
                        elif etype == "message_start":
                            input_tokens = int(
                                ((event.get("message") or {}).get("usage") or {})
                                .get("input_tokens") or 0
                            )
                        elif etype == "message_delta":
                            stop = (event.get("delta") or {}).get("stop_reason")
                            if stop:
                                finish_reason = _STOP_REASON_MAP.get(stop, "stop")
                            output_tokens = int(
                                (event.get("usage") or {}).get("output_tokens") or output_tokens
                            )
                        elif etype == "error":
                            logger.warning(
                                "Anthropic 스트림 오류 이벤트: %r", event.get("error"),
                                extra=event_extra(provider="anthropic"),
                            )
                            yield StreamEvent(type="done", finish_reason="error")
                            return
        except (httpx.HTTPError, RuntimeError) as e:
            logger.warning(
                "Anthropic 스트림 예외: %s: %r", type(e).__name__, e,
                extra=event_extra(provider="anthropic", exc=e),
            )
            yield StreamEvent(type="done", finish_reason="error")
            return

        # 스트림이 content_block_stop 없이 끝난 블록(연결 단절 등)도 버리지 않고 완성한다.
        for idx in sorted(tool_accum):
            tool_calls.append(self._finalize_tool_call(tool_accum[idx]))
        if tool_calls:
            yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
            finish_reason = "tool_calls"

        if input_tokens or output_tokens:
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        meter_llm_call("anthropic", payload.get("model") or self.model, usage, source="anthropic.stream")
        yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)

    @staticmethod
    def _finalize_tool_call(entry: Dict[str, Any]) -> ToolCall:
        """스트림에서 모은 tool_use 블록 하나를 ToolCall 로 완성한다.

        partial_json 조각을 이어붙인 결과가 JSON 이 아니면(절단 등) 빈 인자로 내린다 —
        OpenAI 스트림의 ``_finalize_tool_calls`` 와 같은 방침.
        """
        name = entry.get("name") or "unknown"
        buf = entry.get("buf") or ""
        try:
            args = json.loads(buf) if buf.strip() else {}
        except json.JSONDecodeError:
            logger.warning("[anthropic_stream] tool_use input JSON parse 실패: %s", buf[:200])
            args = {}
        if not isinstance(args, dict):
            args = {}
        return ToolCall(id=str(entry.get("id") or f"call_{name}"), name=name, arguments=args)
