"""OpenAI(클라우드) LLM 프로바이더 구현.

OpenAI Chat Completions API(`https://api.openai.com/v1`)를 통해 GPT 계열 모델과
통신한다. vLLM 구현과 동일한 openai.AsyncOpenAI 클라이언트 및 OpenAI 호환 파싱
로직(OpenAICompatLLM)을 재사용하며, 멀티모달(이미지) 입력을 추가로 지원한다.

파일명을 ``openai_provider.py`` 로 둔 이유: ``openai.py`` 는 ``import openai``(SDK)와
충돌하기 때문이다.
"""

import json
import re
import logging
import time
from typing import ClassVar, List, Dict, Any, AsyncIterator, Optional

import openai
from langsmith import traceable

from agent_arch.services.langsmith.langsmith_utils import record_llm_run
from agent_arch.services.monitoring.error_events import event_extra
from agent_arch.services.usage.meter import cache_token_fields as _cache_usage, meter_llm_call
from agent_arch.llm.openai_compat import OpenAICompatLLM
from agent_arch.llm.base import Message, LLMResponse, StreamEvent, ToolCall
from agent_arch.llm.transient import open_with_retry
from agent_arch.llm.factory import register_llm_provider
from agent_arch.llm import openai_responses as _resp

logger = logging.getLogger(__name__)

from agent_arch.llm.tool_choice import _TOOL_CHOICE_MAP, _tool_choice_param  # noqa: F401


@register_llm_provider("openai")
class OpenAILLM(OpenAICompatLLM):
    """OpenAI 클라우드 LLM 구현 클래스.

    OpenAI Chat Completions API 를 통해 GPT 계열 모델과 통신한다.
    텍스트·이미지(멀티모달) 입력과 함수 호출(tool calling), 스트리밍을 지원한다.

    Attributes:
        client: openai.AsyncOpenAI 클라이언트.
    """

    # factory.py 가 settings 에서 읽어올 설정 키 매핑
    _settings_config: ClassVar[Dict[str, str]] = {"api_key": "openai_api_key"}

    def __init__(self, config: Dict[str, Any]):
        """OpenAILLM 인스턴스를 초기화한다.

        Args:
            config: OpenAI 설정 딕셔너리.
                - api_key: OpenAI API 키 (필수)
                - model: 모델 이름 (예: gpt-5.4-mini)
                - base_url: API 베이스 URL (선택, 미지정 시 OpenAI 기본값)
                - temperature: 생성 온도
                - max_tokens: 최대 생성 토큰 수

        Raises:
            ValueError: API 키가 없는 경우.
        """
        super().__init__(config)
        api_key = config.get("api_key")
        if not api_key:
            raise ValueError("OpenAI API key is required")

        # base_url 미지정 시 openai SDK 기본값(https://api.openai.com/v1) 사용.
        base_url = config.get("base_url") or None
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _convert_messages_to_oai(self, messages: List[Message]) -> list:
        """메시지 리스트를 OpenAI Chat Completions API 형식으로 변환한다.

        vllm 의 변환 로직에 멀티모달(이미지) 지원을 추가한다.
        - role="assistant" + tool_calls → OpenAI 도구 호출 형식.
        - role="tool" → tool_call_id 포함 형식.
        - msg.images 가 있는 메시지 → content 를 텍스트/이미지 파트 배열(멀티모달)로 구성.

        Args:
            messages: 변환할 메시지 리스트.

        Returns:
            OpenAI API 형식의 메시지 딕셔너리 리스트.
        """
        result = []
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                oai_tool_calls = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    }
                    for tc in msg.tool_calls
                ]
                result.append({
                    "role": "assistant",
                    "content": msg.content or None,
                    "tool_calls": oai_tool_calls,
                })
            elif msg.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "content": msg.content,
                })
            elif msg.images:
                # 멀티모달: content 를 텍스트 + 파일/이미지 파트 배열로 구성.
                # ImageContent.data 는 순수 base64 (data URL 접두어 제거됨).
                #
                # PDF 는 image_url 로 보내면 거부된다 — Chat Completions 의 `file`
                # 파트로 보내야 모델이 문서로 읽는다(도면 판독이 이 경로를 쓴다).
                content: List[Dict[str, Any]] = []
                if msg.content:
                    content.append({"type": "text", "text": msg.content})
                for idx, img in enumerate(msg.images):
                    if img.mime_type == "application/pdf":
                        content.append({
                            "type": "file",
                            "file": {
                                "filename": f"document-{idx + 1}.pdf",
                                "file_data": f"data:application/pdf;base64,{img.data}",
                            },
                        })
                    else:
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img.mime_type};base64,{img.data}"
                            },
                        })
                result.append({"role": msg.role, "content": content})
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    def _rejected_param(self, error: Exception, params: Dict[str, Any]) -> Optional[str]:
        """모델이 거부한 인자 이름을 돌려준다 — 재시도할 가치가 없으면 None.

        회사가 임의의 모델을 고를 수 있는 경로(도면 분석 레인)를 위한 안전망이다. 우리가
        얹는 인자를 그 모델이 거부하면 400 이 나고, 도면 판독·판정·리포트가 전부 죽는다
        (실측: gpt-5.6-sol → "Unsupported value: 'temperature' does not support 0.0 with
        this model. Only the default (1) value is supported.").

        **`param_fallback=True` 인스턴스에서만 동작한다.** 에이전트 대화처럼 우리가 모델과
        인자를 통제하는 경로의 실패는 그대로 드러나야 한다 — 조용한 재시도가 생기면 잘못된
        설정이 숨는다(gemini._is_thinking_config_rejected 와 같은 방침).

        Args:
            error: OpenAI 예외.
            params: 이번 호출에 쓴 파라미터. 여기 있는 인자만 뗄 수 있다.

        Returns:
            떼어낼 파라미터 이름. 없으면 None.
        """
        if not self.config.get("param_fallback"):
            return None
        text = str(error)
        if "400" not in text:
            return None
        # OpenAI 는 거부된 인자 이름을 error.param 으로 준다 —
        # {'message': "Unsupported value: 'temperature' ...", 'param': 'temperature',
        #  'code': 'unsupported_value'}
        m = re.search(r"'param':\s*'([A-Za-z_]+)'", text)
        name = m.group(1) if m else None
        # model 자체가 거부된 경우(chat 모델이 아님 등)는 뗄 수 없다 — 그대로 실패시킨다.
        if not name or name == "model" or name not in params:
            return None
        return name

    # ── Responses API 경로(config api=responses) ────────────────────────────────────
    # gpt-6.1-sol 은 Chat Completions 에서 함수 도구를 못 쓴다(app/llm/openai_responses.py 머리말). 그 모델을 쓰는
    # 레인만 이 경로를 켠다 — 기본은 지금까지의 Chat Completions 그대로.

    def _uses_responses_api(self) -> bool:
        return self.config.get("api") == "responses"

    def _responses_params(self, messages: List[Message], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Responses 호출 인자. temperature 는 싣지 않는다 — 이 경로를 쓰는 추론형 모델은 기본값(1) 외에는 거부한다."""
        params: Dict[str, Any] = {
            "model": self.model,
            "input": _resp.to_input(messages),
            "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            "store": False,
        }
        effort = kwargs.get("reasoning_effort") or self.config.get("reasoning_effort")
        if effort:
            params["reasoning"] = {"effort": effort}
        user = kwargs.get("user_sub")
        if user:
            params["user"] = user
        if self._bound_tools and self.supports_function_calling:
            params["tools"] = [_resp.to_tool(self._convert_tool_to_schema(t)) for t in self._bound_tools]
            params["tool_choice"] = _resp.to_tool_choice(kwargs.get("tool_mode", "AUTO"), kwargs.get("allowed_tools"))
        return params

    async def _invoke_responses(self, messages: List[Message], **kwargs) -> LLMResponse:
        try:
            response = await self.client.responses.create(**self._responses_params(messages, kwargs))
        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
        result = _resp.parse_response(response)
        record_llm_run(self.model, usage=result.usage)
        meter_llm_call("openai", self.model, result.usage, source="openai.responses")
        return result

    async def _stream_responses(self, messages: List[Message], **kwargs) -> AsyncIterator[StreamEvent]:
        """본문 델타는 바로 내보내고, 함수 호출은 끝난 항목(output_item.done)을 모아 스트림 끝에 한 번에 낸다.

        함수 호출이 있으면 앞서 본문이 나왔어도 tool_calls 로 끝낸다(gemini 스트림과 같은 방침 — 도구 호출을 버리지 않는다).
        """
        params = self._responses_params(messages, kwargs)
        params["stream"] = True
        try:
            stream = await open_with_retry(lambda: self.client.responses.create(**params),
                                           provider="openai_responses_stream")
        except openai.OpenAIError as e:
            logger.exception("[openai_responses] error: %s", e, extra=event_extra(provider="openai"))
            yield StreamEvent(type="done", finish_reason="error")
            return

        t_start = time.monotonic()
        first_token_time: Optional[float] = None
        calls: List[ToolCall] = []
        final = None
        failed = False
        try:
            async for event in stream:
                kind = getattr(event, "type", "")
                if kind == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.monotonic() - t_start
                        yield StreamEvent(type="text", content=delta)
                elif kind == "response.output_item.done":
                    item = getattr(event, "item", None)
                    if getattr(item, "type", "") == "function_call":
                        calls.append(_resp.tool_call_of(item))
                elif kind in ("response.completed", "response.incomplete"):
                    final = getattr(event, "response", None)
                elif kind in ("response.failed", "error"):
                    final = getattr(event, "response", None)
                    failed = True
                    logger.error("[openai_responses] 스트림 실패: %s", getattr(final, "error", None) or event,
                                 extra=event_extra(provider="openai"))
        except openai.OpenAIError as e:
            logger.exception("[openai_responses] error: %s", e, extra=event_extra(provider="openai"))
            failed = True

        usage = _resp.usage_of(getattr(final, "usage", None))
        record_llm_run(self.model, usage=usage, first_token_secs=first_token_time)
        meter_llm_call("openai", self.model, usage, source="openai.responses_stream")
        if calls:
            yield StreamEvent(type="tool_calls", tool_calls=calls)
        yield StreamEvent(type="done", usage=usage,
                          finish_reason="error" if failed and not calls else _resp.finish_of(final, bool(calls)))

    @traceable(run_type="llm", name="openai_invoke")
    async def invoke(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """OpenAI Chat Completions API 를 호출한다.

        Args:
            messages: 대화 메시지 리스트.
            **kwargs: 추가 파라미터 (temperature, max_tokens, tool_mode, user_sub).

        Returns:
            LLMResponse: OpenAI 응답 객체.

        Raises:
            RuntimeError: API 호출 실패 시.
        """
        if self._uses_responses_api():
            return await self._invoke_responses(messages, **kwargs)
        oai_messages = self._convert_messages_to_oai(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            # GPT-5 계열은 max_tokens 를 거부하고 max_completion_tokens 를 요구한다.
            # 내부 인자(kwargs/self.max_tokens)는 그대로 두고 OpenAI 파라미터 키만 변경.
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        # 추론형 모델의 추론 강도 — 호출자가 줄 때만 싣는다(옵트인). 추론 토큰은 max_completion_tokens 를 함께 쓴다
        # (실측: gpt-6-luna 가 2000 토큰을 추론에 다 써 본문이 빈 채 finish=length).
        effort = kwargs.get("reasoning_effort") or self.config.get("reasoning_effort")
        if effort:
            params["reasoning_effort"] = effort
        # user_sub 이 있으면 OpenAI user 필드로 전달 (남용 모니터링/추적용).
        user = kwargs.get("user_sub")
        if user:
            params["user"] = user

        # tool calling 지원 시 도구 스키마와 tool_choice 추가
        if self._bound_tools and self.supports_function_calling:
            params["tools"] = [
                self._convert_tool_to_schema(t) for t in self._bound_tools
            ]
            tool_mode = kwargs.get("tool_mode", "AUTO")
            params["tool_choice"] = _tool_choice_param(tool_mode, kwargs.get("allowed_tools"))

        # 거부되는 인자를 떼며 최대 2회 재시도한다(temperature·max_completion_tokens 등).
        # 2회로 묶는 이유: 무한 재시도가 되면 진짜 실패가 지연으로만 보인다.
        for _ in range(2):
            try:
                response = await self.client.chat.completions.create(**params)
                break
            except openai.OpenAIError as e:
                dropped = self._rejected_param(e, params)
                if not dropped:
                    raise RuntimeError(f"OpenAI API error: {str(e)}")
                params.pop(dropped, None)
                logger.warning(
                    "[openai] 모델이 거부한 인자를 떼고 재시도: model=%s param=%s "
                    "(도면 판독은 이 인자 없이 실행되므로 같은 도면이 매번 같은 값을 주지 "
                    "않을 수 있다)",
                    self.model, dropped,
                )
        else:
            raise RuntimeError(
                f"OpenAI API error: 인자를 떼고 재시도해도 호출이 실패했습니다 (model={self.model})"
            )

        try:
            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    **_cache_usage(getattr(response.usage, "prompt_tokens_details", None)),
                }
            record_llm_run(self.model, usage=usage)
            meter_llm_call("openai", self.model, usage, source="openai.invoke")
            return self._parse_openai_response(response)

        except openai.OpenAIError as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")

    def _finalize_tool_calls(self, accum: Dict[int, Dict[str, Any]]) -> List[ToolCall]:
        """OpenAI 스트리밍에서 누적된 tool_calls 델타를 ToolCall 리스트로 완성한다.

        Args:
            accum: {index: {"id": ..., "name": ..., "arguments_buf": "..."}}.

        Returns:
            인덱스 순으로 정렬된 ToolCall 리스트. arguments JSON parse 실패 시 빈 dict fallback.
        """
        result: List[ToolCall] = []
        for idx in sorted(accum.keys()):
            entry = accum[idx]
            args_buf = entry.get("arguments_buf", "")
            try:
                args = json.loads(args_buf) if args_buf else {}
            except json.JSONDecodeError:
                logger.warning(
                    "[openai_stream] tool_call args JSON parse 실패: %s", args_buf
                )
                args = {}
            name = entry.get("name") or "unknown"
            result.append(ToolCall(
                id=entry.get("id") or f"call_{name}",
                name=name,
                arguments=args,
            ))
        return result

    @traceable(run_type="llm", name="openai_stream")
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[StreamEvent]:
        """OpenAI 스트리밍 API 를 호출한다.

        텍스트 토큰은 즉시 yield 하고, tool_call 델타는 인덱스별로 누적 후 스트림
        종료 시 한 번에 yield 한다. 마지막에 반드시 done 이벤트를 yield 한다.

        Args:
            messages: 대화 메시지 리스트.
            **kwargs: 추가 파라미터 (temperature, max_tokens, tool_mode, user_sub).

        Yields:
            StreamEvent: text / tool_calls / done 이벤트.
        """
        if self._uses_responses_api():
            async for event in self._stream_responses(messages, **kwargs):
                yield event
            return
        oai_messages = self._convert_messages_to_oai(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            # GPT-5 계열은 max_tokens 를 거부하고 max_completion_tokens 를 요구한다.
            "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        # 추론 강도 — invoke 와 같은 옵트인(스트리밍에서만 빠져 있었다).
        effort = kwargs.get("reasoning_effort") or self.config.get("reasoning_effort")
        if effort:
            params["reasoning_effort"] = effort
        user = kwargs.get("user_sub")
        if user:
            params["user"] = user

        if self._bound_tools and self.supports_function_calling:
            params["tools"] = [
                self._convert_tool_to_schema(t) for t in self._bound_tools
            ]
            tool_mode = kwargs.get("tool_mode", "AUTO")
            params["tool_choice"] = _tool_choice_param(tool_mode, kwargs.get("allowed_tools"))

        first_token_time: Optional[float] = None
        t_start = time.monotonic()
        tool_call_accum: Dict[int, Dict[str, Any]] = {}
        mode: Optional[str] = None  # "text" | "tool_calls"
        usage: Optional[Dict[str, int]] = None
        last_finish_reason: Optional[str] = None

        # invoke 와 같은 방침 — 모델이 거부하는 인자를 떼고 재시도한다. 리포트 생성은
        # stream 경로라, invoke 만 고치면 검증·판독은 통과하고 리포트에서 죽는다.
        # 재시도는 **첫 청크를 받기 전**에만 가능하다(이미 토큰을 내보낸 뒤엔 되돌릴 수 없다).
        stream = None
        for _ in range(2):
            try:
                # 일시 오류(429·503·연결 단절)는 여기서 백오프 재시도한다. 결제 소진
                # (insufficient_quota)은 재시도해도 같은 답이 오므로 즉시 올라온다.
                stream = await open_with_retry(
                    lambda: self.client.chat.completions.create(**params),
                    provider="openai_stream",
                )
                break
            except openai.OpenAIError as e:
                dropped = self._rejected_param(e, params)
                if not dropped:
                    logger.exception(
                        "[openai_stream] error: %s", e, extra=event_extra(provider="openai")
                    )
                    yield StreamEvent(type="done", finish_reason="error")
                    return
                params.pop(dropped, None)
                logger.warning(
                    "[openai_stream] 모델이 거부한 인자를 떼고 재시도: model=%s param=%s",
                    self.model, dropped,
                )
        if stream is None:
            logger.error(
                "[openai_stream] 인자를 떼고 재시도해도 실패: model=%s", self.model,
                extra=event_extra(provider="openai"),
            )
            yield StreamEvent(type="done", finish_reason="error")
            return

        try:
            async for chunk in stream:
                # usage 전용 청크 (choices=[]) 처리
                if not chunk.choices:
                    if getattr(chunk, "usage", None):
                        u = chunk.usage
                        usage = {
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens,
                            **_cache_usage(getattr(u, "prompt_tokens_details", None)),
                        }
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                if getattr(choice, "finish_reason", None):
                    last_finish_reason = choice.finish_reason

                if not delta:
                    continue

                # tool_calls 델타 누적
                delta_tcs = getattr(delta, "tool_calls", None)
                if delta_tcs:
                    if mode is None:
                        mode = "tool_calls"
                    if mode == "tool_calls":
                        for delta_tc in delta_tcs:
                            idx = delta_tc.index
                            entry = tool_call_accum.setdefault(idx, {
                                "id": None,
                                "name": None,
                                "arguments_buf": "",
                            })
                            if getattr(delta_tc, "id", None):
                                entry["id"] = delta_tc.id
                            fn = getattr(delta_tc, "function", None)
                            if fn:
                                if getattr(fn, "name", None):
                                    entry["name"] = fn.name
                                if getattr(fn, "arguments", None):
                                    entry["arguments_buf"] += fn.arguments
                    else:
                        logger.debug(
                            "[openai_stream] text 모드 commit 후 tool_calls 무시"
                        )
                    continue

                # text 청크
                if delta.content:
                    if mode is None:
                        mode = "text"
                    if mode == "text":
                        if first_token_time is None:
                            first_token_time = time.monotonic() - t_start
                        yield StreamEvent(type="text", content=delta.content)
                    else:
                        logger.debug(
                            "[openai_stream] tool_calls 모드 commit 후 text 무시 (%d chars)",
                            len(delta.content),
                        )

            # 누적 tool_calls 완성 후 yield
            if mode == "tool_calls":
                finalized = self._finalize_tool_calls(tool_call_accum)
                if finalized:
                    yield StreamEvent(type="tool_calls", tool_calls=finalized)

            if mode == "tool_calls":
                finish_reason = "tool_calls"
            elif last_finish_reason:
                finish_reason = last_finish_reason
            else:
                finish_reason = "stop"

            record_llm_run(self.model, usage=usage, first_token_secs=first_token_time)
            meter_llm_call("openai", self.model, usage, source="openai.stream")
            yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)

        except openai.OpenAIError as e:
            logger.exception("[openai_stream] error: %s", e, extra=event_extra(provider="openai"))
            # 스트림 도중 끊겨도 이미 받은 usage 는 실어 보낸다(vllm 과 동일 규약).
            if usage:
                meter_llm_call("openai", self.model, usage, source="openai.stream_error")
            yield StreamEvent(type="done", finish_reason="error", usage=usage)
