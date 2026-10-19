import json
import logging
import re
import time
from typing import ClassVar, List, Dict, Any, AsyncIterator, Iterable, Optional
import httpx
import openai
from langsmith import traceable
from agent_arch.services.langsmith.langsmith_utils import record_llm_run
from agent_arch.services.monitoring.error_events import event_extra
from agent_arch.services.usage.meter import meter_llm_call
from agent_arch.llm.openai_compat import OpenAICompatLLM
from agent_arch.llm.base import Message, LLMResponse, StreamEvent, ToolCall
from agent_arch.llm.transient import open_with_retry
from agent_arch.llm.factory import register_llm_provider

logger = logging.getLogger(__name__)

from agent_arch.llm.tool_choice import _TOOL_CHOICE_MAP, _tool_choice_param  # noqa: F401

# Gemma 4 thinking 채널 구분자 (공백/줄바꿈 변형 허용)
_THINK_LEAD = "<|channel>thought"
_CHANNEL_END = "<channel|>"
_THINKING_MARKER_RE = re.compile(r"<\|channel>thought\s*<channel\|>")
_RESPONSE_MARKER_RE = re.compile(r"<\|channel>response\s*<channel\|>")


class _ThinkingFSM:
    """Gemma 4 thinking 채널 토큰을 제거하는 상태 머신.

    text 모드 청크 단위로 feed 하여 emit 가능한 응답 조각만 yield 한다.
    완전 형식(<thinking_marker>...<response_marker>...)과
    단축 형식(<thinking_marker>...) 모두 처리한다.
    """

    def __init__(self):
        self.buffer = ""
        self.in_thinking = False
        self.in_response = False

    def feed(self, chunk: str) -> Iterable[str]:
        """청크를 받아 emit 가능한 응답 조각들을 yield 한다.

        Args:
            chunk: 새로 도착한 텍스트 청크.

        Yields:
            응답 모드로 commit 된 구간의 텍스트 조각.
        """
        if not chunk:
            return

        if self.in_response:
            yield chunk
            return

        self.buffer += chunk

        if self.in_thinking:
            m = _RESPONSE_MARKER_RE.search(self.buffer)
            if m:
                self.in_thinking = False
                self.in_response = True
                remainder = self.buffer[m.end():].lstrip("\n")
                self.buffer = ""
                if remainder:
                    yield remainder
            return

        # 모드 미결정: thinking 마커 여부 판별
        m = _THINKING_MARKER_RE.match(self.buffer)
        if m:
            self.in_thinking = True
            self.buffer = self.buffer[m.end():]
            m2 = _RESPONSE_MARKER_RE.search(self.buffer)
            if m2:
                self.in_thinking = False
                self.in_response = True
                remainder = self.buffer[m2.end():].lstrip("\n")
                self.buffer = ""
                if remainder:
                    yield remainder
        elif _THINK_LEAD.startswith(self.buffer):
            # buffer 가 <|channel>thought 의 prefix → 계속 누적
            return
        elif self.buffer.startswith(_THINK_LEAD):
            rest = self.buffer[len(_THINK_LEAD):]
            rest_stripped = rest.lstrip()
            if rest_stripped and not _CHANNEL_END.startswith(rest_stripped):
                self.in_response = True
                out = self.buffer
                self.buffer = ""
                yield out
            # else: 공백 누적 중이거나 <channel|> prefix 대기 → 계속 누적
        else:
            self.in_response = True
            out = self.buffer
            self.buffer = ""
            yield out

    def flush(self) -> str:
        """스트림 종료 시 잔여 버퍼를 반환한다.

        thinking 모드로 끝난 경우(response 마커가 끝까지 안 도착) 누적된 버퍼는
        ``<|channel>thought<channel|>`` 마커 뒤의 본문이므로, ``_strip_thinking()``
        (invoke 경로) 와 동일하게 단축 형식으로 간주해 응답으로 반환한다.
        """
        out = self.buffer
        self.buffer = ""
        return out


# ── Gemma4 평문 tool_call 누출 폴백 ───────────────────────────────────────────
# vLLM 서버의 ``--tool-call-parser gemma4`` 는 ``<|tool_call>call:NAME{...}<tool_call|>``
# 의 특수토큰 래퍼에 의존한다. 모델이 (긴 인자·멀티스텝 드리프트로) 래퍼를 빠뜨리면
# structured tool_calls 가 아니라 평문 ``call:NAME{...}`` 가 content 로 흘러 사용자
# 화면에 그대로 노출된다(서버 0.23.0 도 평문 폴백 없음 — 설계). 또한 깨진 reasoning
# 채널 마커가 한 줄짜리 ``thought`` 로 흘러 함께 노출된다.
#
# 이 가드는 text 스트림의 *선두만* 버퍼링해 분류한다:
#   - ``[thought\n]call:NAME{...}`` 누출 → 화면 노출을 막고 가능하면 ToolCall 로 복구
#   - 그 외 일반 텍스트 → 즉시 통과(라이브 스트리밍 UX 유지)
# 복구된 호출도 승인필요 액션은 HITL 카드(confirm=False)로 사람이 인자를 검수한다.
_LEAK_THOUGHT_LEAD_RE = re.compile(r"^\s*thought\s*\n")      # 한 줄짜리 깨진 reasoning 마커
_LEAK_CALL_OPEN_RE = re.compile(r"call:[\w.\-]+\s*\{")        # 'call:NAME{' 누출 시작
_LEAK_CALL_BUILDING_RE = re.compile(r"call:[\w.\-]*")        # '{' 직전까지의 빌드중 prefix
_LEAK_CALL_NAME_RE = re.compile(r"call:([\w.\-]+)\s*\{")     # 이름 + 여는 중괄호
_TOOL_CALL_WRAP_RE = re.compile(r"<\|tool_call>|<tool_call\|>")
_LEAK_WRAP_LEAD_RE = re.compile(r"^\s*<\|tool_call>")         # 누출 선두의 <|tool_call> 래퍼
_TOOL_CALL_WRAP_OPEN = "<|tool_call>"                         # 청크 분할 시 부분 프리픽스 판정용
_GEMMA_STR_DELIM = '<|"|>'                                    # Gemma4 문자열 구분자(token 52)
_LEAK_PROBE_CHARS = 24                                        # 비-call 선두 분류 한도
_LEAK_CALL_MAX = 96                                           # 'call:' 빌드 대기 한도(도구명 최장 ~65)


def _coerce_arg_value(raw: str):
    """누출 인자 문자열을 적절한 타입으로 변환한다 (best-effort)."""
    s = raw.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none"):
        return None
    # 선두 0 이 있는 숫자열(전화번호·사번 등)은 정수화하면 0 이 날아가 문자열 유지
    bare = s.lstrip("-")
    if re.fullmatch(r"-?\d+", s) and not (len(bare) > 1 and bare[0] == "0"):
        return int(s)
    if re.fullmatch(r"-?\d+\.\d+", s):
        return float(s)
    return s


def _parse_gemma_call_args(body: str) -> dict:
    """``key:value,key:value`` 형식(따옴표 없는 키/값)을 dict 로 파싱한다.

    Gemma4 평문 누출은 표준 JSON 이 아니다(키 따옴표 없음, 값에 콜론/공백/한국어
    포함: ``clock_in:08:00``, ``reason:출근 시간 정정 요청``). 값은 다음 ``key:`` 직전
    까지로 잡아 콜론·공백을 보존한다. 평면 인자 대상의 best-effort 파서.
    """
    body = body.replace(_GEMMA_STR_DELIM, "")
    args: dict = {}
    for m in re.finditer(
        r"([A-Za-z_]\w*)\s*:\s*(.*?)(?=,\s*[A-Za-z_]\w*\s*:|$)", body, re.DOTALL
    ):
        args[m.group(1)] = _coerce_arg_value(m.group(2))
    return args


def _extract_leak_call(text: str) -> Optional[tuple]:
    """text 에서 첫 ``call:NAME{...}`` 를 *균형 중괄호*로 잘라 ``(name, body)`` 반환.

    그리디 매칭(마지막 ``}`` 까지)은 중첩 객체나 뒤따르는 평문의 ``}`` 를 본문에
    빨아들여 인자를 오염시킨다. 여는 ``{`` 부터 깊이를 세어 *짝 맞는* ``}`` 에서
    끊는다. 닫는 중괄호가 없으면(잘림) None → 복구 실패(억제).
    """
    m = _LEAK_CALL_NAME_RE.search(text)
    if not m:
        return None
    depth, i, n = 1, m.end(), len(text)
    while i < n:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return m.group(1), text[m.end():i]
        i += 1
    return None


class _ToolCallLeakGuard:
    """text 스트림 선두를 버퍼링해 평문 ``call:`` 누출을 가로채는 상태 머신.

    ``feed`` 는 화면에 보여줄 텍스트 조각만 yield 한다. 누출로 판정되면 텍스트를
    내보내지 않고 본문을 캡처하며, 스트림 종료 후 ``recovered_tool_call`` 로 복구한다.
    """

    def __init__(self):
        self._buf = ""
        self._is_leak = False
        self._passthrough = False  # 일반 텍스트 확정 → 이후 그대로 통과

    def feed(self, chunk: str):
        if not chunk:
            return
        if self._passthrough:
            yield chunk
            return
        if self._is_leak:
            self._buf += chunk  # 누출 본문 캡처(출력 안 함)
            return
        self._buf += chunk
        yield from self._classify()

    def _classify(self):
        probe = _LEAK_THOUGHT_LEAD_RE.sub("", self._buf, count=1)
        # 누출이 <|tool_call> 래퍼로 시작하는 변종 — 선두 래퍼를 떼고 안쪽 call:NAME{ 을 본다.
        # (떼지 않으면 '<|tool_call>' 이 call:/thought 프리픽스가 아니라 passthrough 로 새어나감)
        probe = _LEAK_WRAP_LEAD_RE.sub("", probe, count=1)
        stripped = probe.lstrip()
        if _LEAK_CALL_OPEN_RE.search(stripped):
            self._is_leak = True
            self._buf = stripped
            return
        # 'call:NAME' 빌드 중('{' 미도착)이면 한도까지 더 받는다(도구명이 길 수 있음).
        # 'call: 안녕'처럼 식별자 외 문자가 오면 빌드중이 아니므로 일반 텍스트로 흘린다.
        if _LEAK_CALL_BUILDING_RE.fullmatch(stripped) and len(stripped) < _LEAK_CALL_MAX:
            return
        # 아직 thought/call lead 의 prefix 일 수 있으면 짧은 동안 더 받는다.
        # 한도는 *공백 제거 후* 길이 기준 — 선두 공백 청크가 한도를 소진해 누출되는 것 방지.
        if len(stripped) < _LEAK_PROBE_CHARS and self._maybe_leak_prefix(stripped):
            return
        # 일반 텍스트 확정 — 선두 깨진 'thought' 마커만 떼고 통과
        self._passthrough = True
        out, self._buf = probe, ""
        if out:
            yield out

    @staticmethod
    def _maybe_leak_prefix(s: str) -> bool:
        if not s:
            return True
        # 부분 래퍼(<|tool_call> 가 청크로 쪼개져 옴)도 아직 누출일 수 있으니 더 받는다.
        return (
            "call:".startswith(s)
            or "thought".startswith(s.lower())
            or _TOOL_CALL_WRAP_OPEN.startswith(s)
        )

    def flush(self):
        if self._passthrough or self._is_leak:
            return
        out = _LEAK_THOUGHT_LEAD_RE.sub("", self._buf, count=1)
        self._buf = ""
        if out:
            yield out

    def leak_detected(self) -> bool:
        """평문 call: 누출을 감지(억제)했는지 여부 (복구 성공과 무관)."""
        return self._is_leak

    def recovered_tool_call(self) -> Optional["ToolCall"]:
        """누출 캡처본을 ToolCall 로 복구한다. 닫는 중괄호 부재 등 복구 불가 시 None(억제)."""
        if not self._is_leak:
            return None
        extracted = _extract_leak_call(_TOOL_CALL_WRAP_RE.sub("", self._buf))
        if extracted is None:
            return None
        name, body = extracted
        # 빈 본문({})은 정상적인 무인자 호출이므로 드롭하지 않는다.
        return ToolCall(id=f"call_{name}", name=name, arguments=_parse_gemma_call_args(body))


#: 컨텍스트 초과 재시도의 하한. 이보다 짧게 잡으면 문장이 끊겨 답변 가치가 없다.
_MIN_MAX_TOKENS = 256


def _rejects_tool_choice(error: Exception) -> bool:
    """vLLM 서버가 도구 호출 파서 없이 떠서 tool_choice 를 거부했는가.

    실측 문구: '"auto" tool choice requires --enable-auto-tool-choice and
    --tool-call-parser to be set' / 'tool_choice="required" requires --tool-call-parser'.
    서버 기동 옵션 문제라 클라이언트가 고칠 수 없다 — 도구를 떼고 답이라도 받는 편이
    사용자에게 낫다(안 그러면 "죄송합니다" 안내로 끝난다).
    """

    text = str(error).lower()
    return "tool choice requires" in text or "tool-call-parser" in text


def _exceeds_context(error: Exception) -> bool:
    """컨텍스트 한도를 넘겨 거부됐는가.

    실측 문구: "This model's maximum context length is 16384 tokens. However, you
    requested 1024 output tokens and your prompt contains at least 15361 input tokens".
    출력 여유만 줄이면 답할 수 있는 경우가 많다.
    """

    text = str(error).lower()
    return "maximum context length" in text or "context_length_exceeded" in text


def _drop_max_tokens_for_context(params: Dict[str, Any], tag: str) -> bool:
    """컨텍스트 초과 거부에 대응해 ``max_tokens`` 를 떼고 재시도하게 한다.

    **거부 문구의 입력 토큰 수로 여유를 계산하면 안 된다.** "prompt contains at
    least N input tokens" 의 N 은 실제 프롬프트 길이가 아니라 넘침을 유발하는
    하한(= 한도 − 요청한 출력 + 1)이다. 실측 2026-09-21: 한도 65,536 에 8,192 를
    요청하니 N=57,345 로 보고됐고(실제 입력은 60,231), 그 값으로 계산한 8,127 로
    재시도해도 또 거부됐다 — 매번 요청값에서 1씩만 깎이는 셈이라 재시도가 무의미하다.
    예전 ``//4`` 방식도 한 번만 줄여 프롬프트가 한도에 붙은 대화에서는 실패했다
    (8192→2048 로도 1토큰 초과).

    vLLM 은 ``max_tokens`` 가 없으면 ``max_model_len − prompt_tokens`` 를 상한으로
    잡는다. 즉 남은 자리를 서버가 정확히 계산해 준다 — 클라이언트가 프롬프트를
    토크나이즈해 볼 필요가 없다. 실측: 프롬프트 57,001 / 60,326 / 61,751 토큰 모두
    ``max_tokens`` 생략 시 통과, 8192 명시 시 뒤 둘은 거부.

    답변이 과하게 길어질 걱정은 없다 — 이 경로는 요청한 ``max_tokens`` 조차 안
    들어갈 만큼 자리가 없을 때만 타므로, 서버가 잡는 상한은 원래 요청값보다 작다.

    Args:
        params: 재시도할 요청 파라미터(제자리에서 수정된다).
        tag: 로그 접두사.

    Returns:
        떼어낼 게 있어 재시도할 가치가 있으면 True. 이미 뗀 상태면 False —
        그때는 프롬프트 자체가 한도를 넘은 것이라 호출자가 올려보내야 한다.
    """
    if params.get("max_tokens") is None:
        return False
    logger.warning(
        "[%s] 컨텍스트 초과 — max_tokens %s 를 떼고 재시도(서버가 남은 자리를 상한으로 잡는다)",
        tag, params["max_tokens"],
    )
    params.pop("max_tokens", None)
    return True


@register_llm_provider("vllm")
class VllmLLM(OpenAICompatLLM):
    """vLLM LLM 구현 클래스.

    OpenAI 호환 API를 통해 로컬 vLLM 서버와 통신합니다.
    """

    # factory.py가 settings에서 읽어올 설정 키 매핑
    _settings_config: ClassVar[Dict[str, str]] = {"base_url": "vllm_base_url", "api_key": "vllm_api_key"}

    def __init__(self, config: Dict[str, Any]):
        """VllmLLM 인스턴스를 초기화합니다.

        Args:
            config (Dict[str, Any]): vLLM 설정이 포함된 딕셔너리
                - base_url: vLLM 서버 URL (기본값: http://localhost:8001)
                - model: 모델 이름
                - temperature: 생성 온도
                - max_tokens: 최대 생성 토큰 수
        """
        super().__init__(config)
        base_url = config.get("base_url", "http://localhost:8001")
        # 키는 서버를 `--api-key` 로 띄웠을 때만 필요하다(VLLM_API_KEY). 없으면 더미값 — openai.AsyncOpenAI 가
        # 빈 값을 거부하고, 인증 없는 서버는 값을 보지 않는다. env 가 빈 문자열이어도 더미로 간다.
        api_key = config.get("api_key") or "token-abc"
        self.client = openai.AsyncOpenAI(
            base_url=f"{base_url}/v1",
            api_key=api_key,
            http_client=httpx.AsyncClient(verify=False),
        )

    def _strip_thinking(self, content: str) -> str:
        """Gemma 4 thinking 채널 토큰을 제거하고 실제 응답만 반환한다.

        모델 출력 형식이 두 가지일 수 있다:
        - 완전 형식: <thinking_marker>[추론]<response_marker>[응답]
        - 단축 형식: <thinking_marker>[응답] (response 마커 생략)

        Args:
            content: 모델 원본 출력 문자열.

        Returns:
            thinking 섹션이 제거된 응답 문자열.
        """
        m = _RESPONSE_MARKER_RE.search(content)
        if m:
            return content[m.end():].lstrip("\n")
        # response 마커 없이 thinking 마커만 있는 경우 → 마커만 제거
        m = _THINKING_MARKER_RE.match(content)
        if m:
            return content[m.end():]
        return content

    def _extra_body(self, **overrides: Any) -> Dict[str, Any]:
        """OpenAI 스펙 밖의 vLLM 전용 파라미터를 모아 반환한다.

        ``repetition_penalty`` 만 하드코딩돼 있던 자리다. Qwen3.8-Flash-Next 처럼
        **서버 기본이 thinking ON** 인 모델은 ``chat_template_kwargs`` 로 꺼 줘야
        추론 토큰이 지연·비용으로 새지 않는다. 그 값을 YAML(providers.*)에서 선언할
        수 있게 열어 둔다 — 모델을 갈아끼울 때 코드를 고치지 않기 위함이다.

        Args:
            **overrides: 호출부가 이번 요청에만 덮어쓸 값.

        Returns:
            ``extra_body`` 로 실을 딕셔너리. 빈 값은 담지 않는다 — 모르는 파라미터를
            거부하는 백엔드로 바꿔 끼울 때 그대로 터지지 않게 한다.
        """
        body: Dict[str, Any] = {
            # 기존 동작 유지: 설정이 없으면 Gemma4 시절 기본값 1.1
            "repetition_penalty": self.config.get("repetition_penalty", 1.1),
        }
        ctk = self.config.get("chat_template_kwargs")
        if ctk:
            body["chat_template_kwargs"] = dict(ctk)
        for k, v in overrides.items():
            if v is None:
                continue
            if k == "chat_template_kwargs":
                body.setdefault("chat_template_kwargs", {}).update(v)
            else:
                body[k] = v
        return body

    def _convert_messages_to_oai(self, messages: List[Message]) -> list:
        """메시지 리스트를 OpenAI Chat Completions API 형식으로 변환한다.

        role="assistant" 메시지에 tool_calls가 있으면 OpenAI 도구 호출 형식으로 변환하고,
        role="tool" 메시지는 tool_call_id를 포함한 형식으로 변환한다.

        Args:
            messages: 변환할 메시지 리스트

        Returns:
            OpenAI API 형식의 메시지 딕셔너리 리스트
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
            else:
                result.append({"role": msg.role, "content": msg.content})
        return result

    @traceable(run_type="llm", name="vllm_invoke")
    async def invoke(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """vLLM API를 호출합니다.

        Args:
            messages (List[Message]): 대화 메시지 리스트
            **kwargs: 추가 파라미터

        Returns:
            LLMResponse: vLLM 응답 객체

        Raises:
            RuntimeError: API 호출 실패 시
        """
        oai_messages = self._convert_messages_to_oai(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "extra_body": self._extra_body(
                chat_template_kwargs=kwargs.get("chat_template_kwargs"),
            ),
        }

        # tool calling 지원 시 도구 스키마와 tool_choice 추가
        if self._bound_tools and self.supports_function_calling:
            params["tools"] = [
                self._convert_tool_to_schema(t) for t in self._bound_tools
            ]
            tool_mode = kwargs.get("tool_mode", "AUTO")
            params["tool_choice"] = _tool_choice_param(tool_mode, kwargs.get("allowed_tools"))

        try:
            try:
                response = await self.client.chat.completions.create(**params)
            except openai.OpenAIError as first:
                # stream() 과 같은 계약 — 컨텍스트 초과면 출력 여유만 줄여 한 번 더.
                # (이 경로에는 원래 재시도가 없어, 긴 대화가 그대로 500 으로 떨어졌다)
                if not (_exceeds_context(first)
                        and _drop_max_tokens_for_context(params, "vllm_invoke")):
                    raise
                response = await self.client.chat.completions.create(**params)

            usage = None
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            record_llm_run(self.model, usage=usage)
            meter_llm_call("vllm", self.model, usage, source="vllm.invoke")
            llm_response = self._parse_openai_response(response)
            llm_response.content = self._strip_thinking(llm_response.content)
            return llm_response

        except openai.OpenAIError as e:
            raise RuntimeError(f"vLLM API error: {str(e)}")

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
                    "[vllm_stream] tool_call args JSON parse 실패: %s", args_buf
                )
                args = {}
            name = entry.get("name") or "unknown"
            result.append(ToolCall(
                id=entry.get("id") or f"call_{name}",
                name=name,
                arguments=args,
            ))
        return result

    @traceable(run_type="llm", name="vllm_stream")
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[StreamEvent]:
        """vLLM 스트리밍 API를 호출합니다.

        텍스트 토큰은 thinking 마커를 제거한 후 즉시 yield 하고, tool_call 델타는
        인덱스별로 누적 후 스트림 종료 시 한 번에 yield 합니다.
        마지막에 반드시 done 이벤트를 yield 합니다.

        Args:
            messages: 대화 메시지 리스트
            **kwargs: 추가 파라미터 (tool_mode 등)

        Yields:
            StreamEvent: text / tool_calls / done 이벤트.
        """
        oai_messages = self._convert_messages_to_oai(messages)

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": oai_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "stream": True,
            "stream_options": {"include_usage": True},
            "extra_body": self._extra_body(
                chat_template_kwargs=kwargs.get("chat_template_kwargs"),
            ),
        }
        if self._bound_tools and self.supports_function_calling:
            params["tools"] = [
                self._convert_tool_to_schema(t) for t in self._bound_tools
            ]
            tool_mode = kwargs.get("tool_mode", "AUTO")
            params["tool_choice"] = _tool_choice_param(tool_mode, kwargs.get("allowed_tools"))

        first_token_time: Optional[float] = None
        t_start = time.monotonic()
        fsm = _ThinkingFSM()
        guard = _ToolCallLeakGuard()
        tool_call_accum: Dict[int, Dict[str, Any]] = {}
        mode: Optional[str] = None  # "text" | "tool_calls"
        usage: Optional[Dict[str, int]] = None
        last_finish_reason: Optional[str] = None

        async def _open():
            return await self.client.chat.completions.create(**params)

        try:
            # 열기 실패의 대부분은 일시 오류(연결 단절·429·503)다. 아직 토큰을 하나도
            # 흘리지 않은 시점이라 재시도가 안전하다 — 흘린 뒤 재시도하면 같은 문장이 두 번
            # 나간다. 재시도해도 안 되는 실패(401·400·결제 소진)는 즉시 올라온다.
            try:
                stream = await open_with_retry(_open, provider="vllm_stream")
            except openai.OpenAIError as first:
                # 서버 기동 옵션 때문에 tool_choice 가 거부되면 도구를 떼고 답이라도 받는다.
                if _rejects_tool_choice(first) and params.get("tools"):
                    logger.warning(
                        "[vllm_stream] 서버가 tool_choice 를 거부 — 도구 없이 재시도: %s", first
                    )
                    params.pop("tools", None)
                    params.pop("tool_choice", None)
                    stream = await open_with_retry(_open, provider="vllm_stream")
                # 컨텍스트 초과면 출력 여유를 줄여 한 번 더 — 프롬프트는 건드리지 않는다.
                elif _exceeds_context(first) and _drop_max_tokens_for_context(
                    params, "vllm_stream"
                ):
                    stream = await open_with_retry(_open, provider="vllm_stream")
                else:
                    raise

            async for chunk in stream:
                # usage 전용 청크 (choices=[]) 처리
                if not chunk.choices:
                    if getattr(chunk, "usage", None):
                        u = chunk.usage
                        usage = {
                            "prompt_tokens": u.prompt_tokens,
                            "completion_tokens": u.completion_tokens,
                            "total_tokens": u.total_tokens,
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
                    # tool_call 은 '행동' 출력 — 추론 서두 텍스트가 먼저 와도 우선한다.
                    # text 모드였으면 tool_calls 로 승격(선행 텍스트는 폐기). 절대 드롭 금지:
                    # 경량 모델이 "조회하겠습니다" 류 서두를 흘린 뒤 tool_call 을 내는 경우
                    # 기존 로직은 그 tool_call 을 버려 멀티스텝이 통째로 깨졌다.
                    if mode != "tool_calls":
                        if mode == "text":
                            logger.debug("[vllm_stream] 선행 text → tool_calls 승격(서두 폐기)")
                        mode = "tool_calls"
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
                    continue

                # text 청크
                if delta.content:
                    if mode is None:
                        mode = "text"
                    if mode == "text":
                        # FSM(thinking 마커 제거) → 누출 가드(평문 call: 차단) 순으로 통과
                        for emitted in fsm.feed(delta.content):
                            for shown in guard.feed(emitted):
                                if not shown:
                                    continue
                                if first_token_time is None:
                                    first_token_time = time.monotonic() - t_start
                                yield StreamEvent(type="text", content=shown)
                    else:
                        logger.debug(
                            "[vllm_stream] tool_calls 모드 commit 후 text 무시 (%d chars)",
                            len(delta.content),
                        )

            # FSM 잔여분 → 가드 통과 → 가드 잔여 flush (text 모드일 때만 의미 있음)
            recovered: Optional[ToolCall] = None
            if mode == "text":
                for shown in guard.feed(fsm.flush()):
                    if not shown:
                        continue
                    if first_token_time is None:
                        first_token_time = time.monotonic() - t_start
                    yield StreamEvent(type="text", content=shown)
                for shown in guard.flush():
                    if not shown:
                        continue
                    if first_token_time is None:
                        first_token_time = time.monotonic() - t_start
                    yield StreamEvent(type="text", content=shown)
                recovered = guard.recovered_tool_call()

            # tool_calls yield — 구조화 누적 우선, 없으면 평문 'call:' 누출 폴백 복구
            if tool_call_accum:
                finalized = self._finalize_tool_calls(tool_call_accum)
                if finalized:
                    yield StreamEvent(type="tool_calls", tool_calls=finalized)
                    mode = "tool_calls"
            elif recovered is not None:
                logger.warning(
                    "[vllm_stream] 평문 'call:' 누출 감지 → tool_call '%s' 폴백 복구"
                    " (서버 tool-call-parser 미검출)",
                    recovered.name,
                )
                yield StreamEvent(type="tool_calls", tool_calls=[recovered])
                mode = "tool_calls"
            elif mode == "text" and guard.leak_detected():
                # 누출은 감지·억제했으나 복구 실패(잘림 등) — 빈 응답으로 루프가 재시도한다.
                logger.warning(
                    "[vllm_stream] 평문 'call:' 누출 감지했으나 복구 실패 → 억제(루프 재시도 유도)"
                )

            if mode == "tool_calls":
                finish_reason = "tool_calls"
            elif last_finish_reason:
                finish_reason = last_finish_reason
            else:
                finish_reason = "stop"

            record_llm_run(self.model, usage=usage, first_token_secs=first_token_time)
            meter_llm_call("vllm", self.model, usage, source="vllm.stream")
            yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)

        except openai.OpenAIError as e:
            logger.exception("[vllm_stream] error: %s", e, extra=event_extra(provider="vllm"))
            # usage 를 이미 받은 뒤 끊긴 경우가 있다 — 버리면 그 턴이 "토큰 미기록"으로
            # 남아 크레딧이 실제보다 적게 잡힌다. 받은 만큼은 실어 보낸다.
            if usage:
                meter_llm_call("vllm", self.model, usage, source="vllm.stream_error")
            yield StreamEvent(type="done", finish_reason="error", usage=usage)
