import base64
import uuid
import logging
import time
from typing import ClassVar, List, Dict, Any, AsyncIterator, Optional

from google import genai
from google.genai import types
from langsmith import traceable

from agent_arch.services.langsmith.langsmith_utils import record_llm_run
from agent_arch.services.monitoring.error_events import event_extra
from agent_arch.services.usage.meter import meter_llm_call
from agent_arch.llm.base import BaseLLM, Message, LLMResponse, StreamEvent, ToolCall
from agent_arch.llm.transient import open_with_retry
from agent_arch.llm.factory import register_llm_provider


_SKIP_SIGNATURE = b"skip_thought_signature_validator"


def _requires_signature(model: Optional[str]) -> bool:
    """functionCall 에 thought_signature 가 필수인 모델(Gemini 3 이상). 2.5 계열은 검사하지 않는다."""
    import re as _re
    m = _re.match(r"(?:models/)?gemini-(\d+)", str(model or ""))
    return bool(m) and int(m.group(1)) >= 3


def _call_id(fc) -> str:
    """Gemini 가 function_call.id 를 비워 보내면 고유 id 를 만든다.

    예전 폴백 `call_{name}` 은 같은 도구를 한 턴에 여러 번(병렬) 부르면 id 가 겹쳤다(운영 서버 실측 2026-09-22:
    brave_llm_context 4건이 모두 `call_brave_llm_context`). Gemini 는 이름·순서로 응답을 맞추지만 우리 코드(검증기 증거
    사실·예산 가드·결과 매칭)는 tool_call_id 로 찾으므로 앞 결과가 뒤 결과에 덮였다. 이름 복원은 `__` 앞부분.
    """
    return getattr(fc, "id", None) or f"call_{fc.name}__{uuid.uuid4().hex[:8]}"

logger = logging.getLogger(__name__)


@register_llm_provider("gemini")
class GeminiLLM(BaseLLM):
    """Google Gemini LLM 구현 클래스.

    Google Generative AI SDK(google-genai)를 사용하여 LLM 기능을 제공합니다.
    Thinking 모델의 thought_signature를 캡처·복원하여 멀티턴 tool calling을 지원합니다.
    """

    # factory.py가 settings에서 읽어올 설정 키 매핑
    _settings_config: ClassVar[Dict[str, str]] = {"api_key": "gemini_api_key"}

    def __init__(self, config: Dict[str, Any]):
        """GeminiLLM 인스턴스를 초기화합니다.

        Args:
            config: Gemini 설정 딕셔너리
                - api_key: Google API 키 (필수)
                - model: 모델 이름
                - temperature: 생성 온도 (0.0–1.0)
                - max_tokens: 최대 생성 토큰 수
                - top_p: nucleus sampling 파라미터

        Raises:
            ValueError: API 키가 없는 경우
        """
        super().__init__(config)
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("Gemini API key is required")

        self.top_p = config.get("top_p", 0.95)
        # 기본 thinking_budget (yaml). None 이면 미설정(모델 기본 thinking).
        # 호출 시 kwargs["thinking_budget"] 로 개별 override 가능.
        self.thinking_budget = config.get("thinking_budget")
        self.client = genai.Client(api_key=api_key)

    def _build_generation_config(
        self,
        system_msgs: List[Message],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """invoke() / stream() 공통 generation config 를 구성한다.

        Args:
            system_msgs: role="system" 메시지 리스트.
            kwargs: 호출 시 전달된 추가 인자 (tool_mode 등).

        Returns:
            types.GenerateContentConfig 생성용 dict.
        """
        config_kwargs: Dict[str, Any] = dict(
            temperature=self.temperature,
            # 호출 시 max_tokens kwargs 가 오면 존중 (demo 분류 호출의 토큰 제한 등).
            max_output_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=self.top_p,
        )
        if system_msgs:
            config_kwargs["system_instruction"] = "\n\n".join(
                m.content for m in system_msgs
            )
        # SDK 자동 함수호출(AFC) 비활성 — **항상**. 도구 실행은 오케스트레이터(MCP)가 수동
        # 처리하므로 SDK가 functionCall 을 자체 실행하면 안 된다.
        # 주의: AFC 비활성은 thought_signature 400 의 해법이 아니다. 히스토리 contents 에
        # functionCall part 가 있으면 gemini 3.x 가 서버측에서 시그니처를 강제 검증하며,
        # 이는 AFC·thinking_budget 설정과 무관하다(실측 확인). 시그니처 없는 tool_call 을
        # replay 해야 하는 경로(병렬 eval)는 호출 전에 평문으로 펼쳐 functionCall 을 제거한다.
        config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
            disable=True
        )
        # thinking_budget: 호출 kwargs 우선, 없으면 yaml 기본값(self.thinking_budget).
        # 둘 다 None 이면 모델 기본 thinking 유지.
        thinking_budget = kwargs.get("thinking_budget", self.thinking_budget)
        if thinking_budget is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=thinking_budget
            )
        if self._bound_tools and self.supports_function_calling:
            schemas = [self._convert_tool_to_schema(t) for t in self._bound_tools]
            config_kwargs["tools"] = [types.Tool(function_declarations=schemas)]
            mode = kwargs.get("tool_mode", "AUTO")
            fcc_kwargs: Dict[str, Any] = {"mode": mode}
            # guide-first: ANY + allowed_tools 면 그 도구(들)만 호출 가능하게 고정.
            allowed = kwargs.get("allowed_tools")
            if mode == "ANY" and allowed:
                fcc_kwargs["allowed_function_names"] = list(allowed)
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(**fcc_kwargs)
            )
            # (AFC 비활성은 위에서 무조건 처리 — 도구 바인딩 여부와 무관)
        # 구조화 출력: 키·타입을 API 가 강제한다(도면 판독). 응답 본문은 그 스키마의 JSON 이다.
        response_schema = kwargs.get("response_schema")
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema
        return config_kwargs

    def _extract_usage(self, chunk_or_response) -> Optional[Dict[str, int]]:
        """Gemini 응답(또는 마지막 청크)에서 usage_metadata 를 추출한다.

        Args:
            chunk_or_response: Gemini API 응답 또는 스트림 청크.

        Returns:
            토큰 사용량 dict 또는 None.
        """
        um = getattr(chunk_or_response, "usage_metadata", None) if chunk_or_response else None
        if not um:
            return None
        usage = {
            "prompt_tokens": um.prompt_token_count or 0,
            "completion_tokens": um.candidates_token_count or 0,
            "total_tokens": um.total_token_count or 0,
        }
        # 입력 중 암묵/명시 캐시 적중분(prompt_token_count 에 포함) — 외부 원가 계산용. 적중이
        # 있을 때만 싣는다(없으면 계측이 0 으로 읽는다).
        cached = getattr(um, "cached_content_token_count", None)
        if isinstance(cached, int) and not isinstance(cached, bool) and cached > 0:
            usage["cached_tokens"] = cached
        return usage

    def _derive_finish_reason(
        self,
        chunk_or_response,
        accumulated_tool_calls: List[ToolCall],
    ) -> str:
        """tool_calls 누적 여부 + candidates[0].finish_reason 으로 종료 사유를 결정한다.

        Args:
            chunk_or_response: 마지막 청크 또는 응답 객체.
            accumulated_tool_calls: 누적된 ToolCall 리스트.

        Returns:
            "tool_calls" | "malformed_function_call" | "length" | "stop".
        """
        if accumulated_tool_calls:
            return "tool_calls"
        candidates = getattr(chunk_or_response, "candidates", None) if chunk_or_response else None
        if candidates:
            raw = getattr(candidates[0], "finish_reason", None)
            if raw == types.FinishReason.MALFORMED_FUNCTION_CALL:
                return "malformed_function_call"
            if raw == types.FinishReason.MAX_TOKENS:
                return "length"
        return "stop"

    @traceable(run_type="llm", name="gemini_invoke")
    async def invoke(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """Gemini API를 호출합니다.

        Args:
            messages: 대화 메시지 리스트
            **kwargs: 추가 파라미터 (tool_mode: "AUTO" | "ANY" | "NONE")

        Returns:
            LLMResponse: Gemini 응답 객체

        Raises:
            RuntimeError: API 호출 실패 시
        """
        system_msgs = [m for m in messages if m.role == "system"]
        content_msgs = [m for m in messages if m.role != "system"]

        gemini_messages = self._convert_messages(content_msgs)
        config_kwargs = self._build_generation_config(system_msgs, kwargs)

        try:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=gemini_messages,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
            except Exception as first_error:
                if not self._is_thinking_config_rejected(first_error, config_kwargs):
                    raise
                logger.warning(
                    "Gemini %s 가 thinking_config 를 거부 — 제거하고 재시도합니다.", self.model
                )
                retry_kwargs = {k: v for k, v in config_kwargs.items() if k != "thinking_config"}
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=gemini_messages,
                    config=types.GenerateContentConfig(**retry_kwargs),
                )

            usage = self._extract_usage(response)
            record_llm_run(self.model, usage=usage)
            meter_llm_call("gemini", self.model, usage, source="gemini.invoke")

            return self._parse_response(response)

        except Exception as e:
            raise RuntimeError(f"Gemini API error: {str(e)}")

    def _is_thinking_config_rejected(self, error: Exception, config_kwargs: dict) -> bool:
        """thinking_config 때문에 400 이 난 것으로 보이는지 판별한다.

        thinking 을 끌 수 없는 모델에 `thinking_budget=0` 을 보내면 Gemini 가
        400 INVALID_ARGUMENT 를 준다(실측: gemini-3.5-flash-lite).

        **`param_fallback=True` 로 만든 인스턴스에서만 동작한다.** 이 재시도는 회사가 임의의
        모델을 고를 수 있는 경로(도면 분석 레인)를 위한 안전망이고, 에이전트 대화처럼
        우리가 모델과 인자를 통제하는 경로의 실패는 그대로 드러나야 한다 — 조용한 재시도가
        생기면 잘못된 설정이 숨는다.

        Args:
            error: API 예외.
            config_kwargs: 이번 호출에 쓴 설정.

        Returns:
            재시도할 가치가 있으면 True.
        """
        if not self.config.get("param_fallback"):
            return False
        if "thinking_config" not in config_kwargs:
            return False
        text = str(error)
        return "INVALID_ARGUMENT" in text or "400" in text

    @traceable(run_type="llm", name="gemini_stream")
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[StreamEvent]:
        """Gemini 스트리밍 API를 호출합니다.

        텍스트 토큰을 즉시 yield 하고, function_call part 는 누적 후
        스트림 종료 시 한 번에 yield 합니다. 마지막에 반드시 done 이벤트를 yield 합니다.

        Args:
            messages: 대화 메시지 리스트
            **kwargs: 추가 파라미터 (tool_mode 등)

        Yields:
            StreamEvent: text / tool_calls / done 이벤트.
        """
        system_msgs = [m for m in messages if m.role == "system"]
        content_msgs = [m for m in messages if m.role != "system"]

        gemini_messages = self._convert_messages(content_msgs)
        config_kwargs = self._build_generation_config(system_msgs, kwargs)

        accumulated_tool_calls: List[ToolCall] = []
        mode: Optional[str] = None  # "text" | "tool_calls"
        first_token_time: Optional[float] = None
        t_start = time.monotonic()
        last_chunk = None

        async def _open_stream():
            """스트림을 연다.

            thinking_config 거부면 제거하고 한 번 더 시도하고, 일시 오류(503·429·연결
            단절)는 지수 백오프로 재시도한다 — 이 시점엔 토큰을 하나도 흘리지 않아
            재시도가 안전하다. dev 로그 실측에서 이 프로바이더 실패 39건이 전부 503/429
            였고, 그 턴의 사용자는 답변 대신 안내 메시지를 받았다.
            """
            try:
                return await open_with_retry(
                    lambda: self.client.aio.models.generate_content_stream(
                        model=self.model,
                        contents=gemini_messages,
                        config=types.GenerateContentConfig(**config_kwargs),
                    ),
                    provider="gemini_stream",
                )
            except Exception as first_error:
                if not self._is_thinking_config_rejected(first_error, config_kwargs):
                    raise
                logger.warning(
                    "Gemini %s 가 thinking_config 를 거부 — 제거하고 재시도합니다(stream).",
                    self.model,
                )
                retry_kwargs = {k: v for k, v in config_kwargs.items() if k != "thinking_config"}
                return await open_with_retry(
                    lambda: self.client.aio.models.generate_content_stream(
                        model=self.model,
                        contents=gemini_messages,
                        config=types.GenerateContentConfig(**retry_kwargs),
                    ),
                    provider="gemini_stream",
                )

        try:
            async for chunk in await _open_stream():
                last_chunk = chunk
                candidates = getattr(chunk, "candidates", None) or []
                for candidate in candidates:
                    content_obj = getattr(candidate, "content", None)
                    parts = getattr(content_obj, "parts", None) if content_obj else None
                    if not parts:
                        continue

                    for part in parts:
                        # function_call part
                        fc = getattr(part, "function_call", None)
                        if fc and getattr(fc, "name", None):
                            # function_call 우선 — 추론(thought)/text 가 먼저 와도 tool_calls 로
                            # 승격, 절대 드롭 금지. (thinking 모델이 thought 를 먼저 흘린 뒤
                            # function_call 을 내면 기존 로직이 그걸 버려 도구호출이 증발했다.)
                            if mode != "tool_calls":
                                if mode == "text":
                                    logger.debug(
                                        "[gemini_stream] 선행 text → tool_calls 승격: %s", fc.name
                                    )
                                mode = "tool_calls"
                            sig = getattr(part, "thought_signature", None)
                            sig_b64 = base64.b64encode(sig).decode("ascii") if sig else None
                            accumulated_tool_calls.append(ToolCall(
                                id=_call_id(fc),
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                                thought_signature=sig_b64,
                            ))
                            continue

                        # text part
                        text = getattr(part, "text", None)
                        if text:
                            if mode is None:
                                mode = "text"
                            if mode == "text":
                                if first_token_time is None:
                                    first_token_time = time.monotonic() - t_start
                                yield StreamEvent(type="text", content=text)
                            else:
                                logger.debug(
                                    "[gemini_stream] tool_calls 모드 commit 후 text 무시 (%d chars)",
                                    len(text),
                                )

            # tool_calls 누적이 있으면 한 번에 yield
            if accumulated_tool_calls:
                yield StreamEvent(type="tool_calls", tool_calls=accumulated_tool_calls)

            usage = self._extract_usage(last_chunk)
            finish_reason = self._derive_finish_reason(last_chunk, accumulated_tool_calls)
            record_llm_run(self.model, usage=usage, first_token_secs=first_token_time)
            meter_llm_call("gemini", self.model, usage, source="gemini.stream")
            yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)

        except Exception as e:
            logger.exception("[gemini_stream] error: %s", e, extra=event_extra(provider="gemini"))
            # 마지막 청크를 이미 받았다면 usage 가 그 안에 있다 — 버리지 않는다.
            usage = self._extract_usage(last_chunk)
            if usage:
                meter_llm_call("gemini", self.model, usage, source="gemini.stream_error")
            yield StreamEvent(type="done", finish_reason="error", usage=usage)

    # Gemini Schema proto가 지원하는 필드 목록
    _GEMINI_SCHEMA_FIELDS = {"type", "description", "enum", "format", "items", "properties", "required", "nullable"}

    def _convert_tool_to_schema(self, tool: Any) -> Dict[str, Any]:
        """BaseTool을 Gemini Function Declaration 포맷으로 변환합니다.

        Args:
            tool: BaseTool 인스턴스

        Returns:
            Gemini Function Declaration 스키마 딕셔너리
        """
        cleaned = {
            name: self._clean_schema_property(schema)
            for name, schema in tool.parameters.items()
        }
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "OBJECT",
                "properties": cleaned,
                "required": tool.required_params or []
            }
        }

    def _clean_schema_property(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """JSON Schema property를 Gemini Schema proto 호환 형식으로 정제합니다.

        - type 값을 대문자로 변환 ('string' → 'STRING')
        - Gemini가 지원하지 않는 필드 제거 (default, minimum, maximum 등)
        - items (ARRAY), properties (OBJECT) 재귀 처리
        """
        prop: Dict[str, Any] = {}
        for key, val in schema.items():
            if key not in self._GEMINI_SCHEMA_FIELDS:
                continue
            if key == "type" and isinstance(val, str):
                prop["type"] = val.upper()
            elif key == "items" and isinstance(val, dict):
                prop["items"] = self._clean_schema_property(val)
            elif key == "properties" and isinstance(val, dict):
                prop["properties"] = {
                    k: self._clean_schema_property(v) for k, v in val.items()
                }
            else:
                prop[key] = val
        return prop

    def _convert_messages(self, messages: List[Message]) -> List[Any]:
        """Message 리스트를 Gemini 포맷으로 변환합니다.

        이미지가 있는 경우 inline_data 파트를 텍스트보다 앞에 배치합니다.
        tool_calls가 있는 assistant 메시지는 thought_signature를 복원하여
        types.Part 객체로 변환합니다.

        Args:
            messages: Message 객체 리스트

        Returns:
            Gemini API에 전달할 메시지 리스트
        """
        gemini_messages = []
        i = 0

        while i < len(messages):
            msg = messages[i]
            role = "model" if msg.role == "assistant" else msg.role

            # Tool calls 처리 (model turn) — thought_signature 복원
            if msg.tool_calls:
                parts = []
                for tc in msg.tool_calls:
                    sig_b64 = tc.get("thought_signature")
                    sig_bytes = base64.b64decode(sig_b64) if sig_b64 else None
                    if sig_bytes is None and _requires_signature(self.model):
                        # 모델이 만들지 않은 호출(선실행·이력 재구성)에는 서명이 없다. Gemini 3 계열은 서명 없는 functionCall 을 400
                        # 으로 거절한다(2026-09-23: 웹 선실행 턴 전부 "조회하지 못했습니다"). 공식 문서의 대체 서명값으로 검증을 건너뛴다.
                        sig_bytes = _SKIP_SIGNATURE

                    fc_kwargs: Dict[str, Any] = dict(
                        name=tc.get("name"),
                        args=tc.get("arguments", {}),
                    )
                    fc_id = tc.get("id")
                    if fc_id:
                        fc_kwargs["id"] = fc_id

                    part_kwargs: Dict[str, Any] = dict(
                        function_call=types.FunctionCall(**fc_kwargs)
                    )
                    if sig_bytes:
                        part_kwargs["thought_signature"] = sig_bytes

                    parts.append(types.Part(**part_kwargs))

                gemini_messages.append({"role": "model", "parts": parts})
                i += 1
                continue

            # Tool 응답 처리: 연속된 tool 메시지를 하나의 user turn으로 묶음
            if msg.role == "tool":
                tool_parts = []
                while i < len(messages) and messages[i].role == "tool":
                    t = messages[i]
                    func_name = t.name or (t.tool_call_id or "").replace("call_", "", 1).split("__")[0]
                    try:
                        import json as _json
                        parsed = _json.loads(t.content)
                        response_data = parsed if isinstance(parsed, dict) else {"result": parsed}
                    except Exception:
                        response_data = {"result": t.content}
                    tool_parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response=response_data,
                        )
                    ))
                    i += 1
                gemini_messages.append({"role": "user", "parts": tool_parts})
                continue

            parts = []

            # 이미지 파트 먼저 (Gemini 권장 순서)
            if msg.images:
                for img in msg.images:
                    parts.append({
                        "inline_data": {
                            "mime_type": img.mime_type,
                            "data": img.data
                        }
                    })

            if msg.content:
                parts.append({"text": msg.content})

            gemini_messages.append({"role": role, "parts": parts})
            i += 1

        return gemini_messages

    def _parse_response(self, response) -> LLMResponse:
        """Gemini 응답을 LLMResponse로 변환합니다.

        Part 레벨의 thought_signature를 캡처하여 ToolCall에 base64 str로 저장합니다.

        Args:
            response: Gemini API 응답 객체

        Returns:
            LLMResponse: 표준화된 응답 객체
        """
        content = ""
        tool_calls = []

        try:
            if hasattr(response, "text") and response.text:
                content = response.text
        except Exception:
            pass  # function_call만 있는 응답 → text 없음, 정상

        if hasattr(response, "candidates") and response.candidates:
            for candidate in response.candidates:
                content_obj = getattr(candidate, "content", None)
                if content_obj is None:
                    continue
                parts = getattr(content_obj, "parts", None)
                if not parts:
                    continue
                for part in parts:
                    if hasattr(part, "function_call") and part.function_call and part.function_call.name:
                        fc = part.function_call

                        # Part 레벨에서 thought_signature 캡처 (bytes → base64 str)
                        sig = getattr(part, "thought_signature", None)
                        sig_b64 = base64.b64encode(sig).decode("ascii") if sig else None

                        tool_calls.append(ToolCall(
                            id=_call_id(fc),
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                            thought_signature=sig_b64,
                        ))

        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif hasattr(response, "candidates") and response.candidates:
            raw_fr = getattr(response.candidates[0], "finish_reason", None)
            if raw_fr == types.FinishReason.MALFORMED_FUNCTION_CALL:
                logger.warning(
                    "Gemini MALFORMED_FUNCTION_CALL 감지 — 도구 호출 실패. "
                    "tool_mode=NONE으로 orchestrator에서 재시도 필요."
                )
                finish_reason = "malformed_function_call"

        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            um = response.usage_metadata
            usage = {
                "prompt_tokens": um.prompt_token_count or 0,
                "completion_tokens": um.candidates_token_count or 0,
                "total_tokens": um.total_token_count or 0,
            }

        return LLMResponse(
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            finish_reason=finish_reason,
            usage=usage,
        )
