from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator, Literal
from pydantic import BaseModel


class ImageContent(BaseModel):
    """메시지에 첨부된 이미지 모델.

    base64 인코딩된 순수 이미지 데이터만 포함합니다.
    data URL 접두어(data:image/...;base64,)는 제거된 상태여야 합니다.
    """
    data: str
    mime_type: str


class Message(BaseModel):
    """LLM 메시지 모델.

    대화형 LLM API에서 사용하는 메시지 형식을 정의합니다.
    """
    role: str  # system, user, assistant, tool
    content: str
    images: Optional[List[ImageContent]] = None
    name: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    # user 메시지: 사용자가 직접 쓴 질문(첨부 본문이 content 앞에 붙은 턴에서만). 프로바이더로는 보내지 않는다.
    question: Optional[str] = None


class ToolCall(BaseModel):
    """도구 호출 정보 모델.

    LLM이 함수 호출을 요청할 때의 정보를 담습니다.
    """
    id: str
    name: str
    arguments: Dict[str, Any]
    thought_signature: Optional[str] = None  # base64 인코딩된 bytes, Gemini Thinking 모델 전용


class LLMResponse(BaseModel):
    """LLM 응답 모델.

    LLM API 호출 결과를 표준화된 형식으로 담습니다.
    """
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str  # stop, tool_calls, length, etc.
    usage: Optional[Dict[str, int]] = None


class StreamEvent(BaseModel):
    """LLM 스트림의 단일 이벤트.

    어댑터(Gemini/vLLM)가 토큰 단위 응답과 tool_call 진행 상태를
    단일 시퀀스로 표현하기 위한 표준 모델.

    이벤트 타입별 필드 규약:
        - type="text":  content 필수 (텍스트 토큰 조각).
        - type="tool_calls": tool_calls 필수 (어댑터가 누적/완성한 결과).
        - type="done":  finish_reason 필수, usage 선택. 스트림 마지막에 정확히 1회 yield.
            finish_reason 값: "stop" | "tool_calls" | "length" |
                              "malformed_function_call" | "error".

    Attributes:
        type: 이벤트 종류.
        content: 텍스트 토큰 (type="text" 일 때).
        tool_calls: 완성된 도구 호출 목록 (type="tool_calls" 일 때).
        finish_reason: 종료 사유 (type="done" 일 때).
        usage: 토큰 사용량 (type="done" 일 때, 어댑터가 제공 가능한 경우).
    """

    type: Literal["text", "tool_calls", "done"]
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class BaseLLM(ABC):
    """LLM 프로바이더 추상 기본 클래스.

    모든 LLM 프로바이더(Gemini, OpenAI, vLLM 등)는 이 인터페이스를 구현해야 합니다.
    공통 API를 제공하여 프로바이더 간 전환을 용이하게 합니다.

    Attributes:
        config: LLM 설정 딕셔너리
        model: 모델 이름
        temperature: 생성 온도 (0.0-1.0)
        max_tokens: 최대 생성 토큰 수
        supports_function_calling: 함수 호출 지원 여부
        supports_streaming: 스트리밍 지원 여부
        _bound_tools: 바인딩된 도구 리스트
    """

    def __init__(self, config: Dict[str, Any]):
        """BaseLLM 인스턴스를 초기화합니다.

        Args:
            config (Dict[str, Any]): LLM 설정이 포함된 딕셔너리
        """
        self.config = config
        self.model = config.get("model")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 4096)
        self.supports_function_calling = config.get("supports_function_calling", False)
        self.supports_streaming = config.get("supports_streaming", False)
        # stateless=True: 시스템 프롬프트/히스토리 없이 user 메시지만 전달 (외부 독립 서비스용)
        self.stateless = config.get("stateless", False)
        self._bound_tools: List[Any] = []

    @abstractmethod
    async def invoke(
        self,
        messages: List[Message],
        **kwargs
    ) -> LLMResponse:
        """LLM을 동기 방식으로 호출합니다.

        Args:
            messages (List[Message]): 대화 메시지 리스트
            **kwargs: 추가 파라미터 (temperature, max_tokens 등)

        Returns:
            LLMResponse: LLM 응답 객체

        Raises:
            NotImplementedError: 구현되지 않은 경우
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[StreamEvent]:
        """LLM을 스트리밍 방식으로 호출합니다.

        이벤트 시퀀스 규약:
            1. tool_calls 모드:
                type="tool_calls" 1회 → type="done" (finish_reason="tool_calls")
            2. text 모드:
                type="text" N회 → type="done" (finish_reason="stop"|"length")
            3. 오류:
                type="done" (finish_reason="error") — text/tool_calls 없이 즉시 종료

        kwargs 규약:
            - tool_mode: Literal["AUTO", "ANY", "NONE"] — invoke() 와 동일.
            - user_sub: str | None — LangSmith 메타데이터.

        Args:
            messages (List[Message]): 대화 메시지 리스트
            **kwargs: 추가 파라미터 (tool_mode, user_sub 등)

        Yields:
            StreamEvent: 단일 이벤트.

        Raises:
            NotImplementedError: 구현되지 않은 경우
        """
        pass

    async def bind_tools(self, tools: List[Any]) -> "BaseLLM":
        """도구를 LLM에 바인딩합니다.

        도구 목록을 내부에 저장한다. 모든 프로바이더가 동일하게 동작하므로
        기본 구현을 제공한다. 프로바이더별 추가 처리가 필요하면 오버라이드한다.

        Args:
            tools (List[Any]): BaseTool을 상속한 도구 리스트

        Returns:
            BaseLLM: 메서드 체이닝을 위해 자기 자신 반환
        """
        self._bound_tools = tools
        return self

    @abstractmethod
    def _convert_tool_to_schema(self, tool: Any) -> Dict[str, Any]:
        """BaseTool을 프로바이더별 스키마로 변환합니다.

        Args:
            tool (Any): BaseTool 인스턴스

        Returns:
            Dict[str, Any]: 프로바이더별 도구 스키마

        Raises:
            NotImplementedError: 구현되지 않은 경우
        """
        pass

    def get_bound_tools(self) -> List[Any]:
        """현재 바인딩된 도구 리스트를 반환합니다.

        Returns:
            List[Any]: 바인딩된 도구 리스트
        """
        return self._bound_tools

    async def invoke_with_retry(
        self,
        messages: List[Message],
        max_retries: int = 3,
        **kwargs
    ) -> LLMResponse:
        """재시도 로직이 포함된 LLM 호출을 수행합니다.

        네트워크 오류나 일시적인 실패에 대해 지수 백오프로 재시도합니다.

        Args:
            messages (List[Message]): 대화 메시지 리스트
            max_retries (int): 최대 재시도 횟수 (기본값: 3)
            **kwargs: 추가 파라미터

        Returns:
            LLMResponse: LLM 응답 객체

        Raises:
            Exception: 모든 재시도 실패 시 마지막 예외 발생

        Examples:
            >>> response = await llm.invoke_with_retry(
            ...     messages=[Message(role="user", content="안녕하세요")],
            ...     max_retries=5
            ... )
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.invoke(messages, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # 지수 백오프 대기
                    await self._wait_exponential(attempt)
                    continue

        raise last_error

    async def _wait_exponential(self, attempt: int):
        """지수 백오프 방식으로 대기합니다.

        Args:
            attempt (int): 현재 시도 횟수 (0부터 시작)
        """
        import asyncio
        wait_time = min(2 ** attempt, 30)  # 최대 30초
        await asyncio.sleep(wait_time)

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환합니다.

        Returns:
            str: 클래스 이름과 모델 정보
        """
        return f"{self.__class__.__name__}(model={self.model})"
