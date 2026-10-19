"""OpenAI Chat Completions API 호환 LLM 프로바이더 기본 클래스.

vllm이 사용하는 도구 스키마 변환 및 응답 파싱 로직을 제공한다.
"""

import json
from typing import Any, Dict

from agent_arch.llm.base import BaseLLM, LLMResponse, ToolCall


class OpenAICompatLLM(BaseLLM):
    """OpenAI Chat Completions API 호환 프로바이더 기본 클래스.

    _convert_tool_to_schema()와 _parse_openai_response()를 제공하여
    OpenAI 호환 프로바이더(vllm)의 도구 스키마 변환·응답 파싱 로직을 제공한다.
    """

    def _convert_tool_to_schema(self, tool: Any) -> Dict[str, Any]:
        """BaseTool을 OpenAI Function Calling 형식의 스키마로 변환한다.

        Args:
            tool: BaseTool 인스턴스.

        Returns:
            Dict[str, Any]: OpenAI 호환 도구 스키마.
        """
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required_params or [],
                },
            },
        }

    def _parse_openai_response(self, response: Any) -> LLMResponse:
        """OpenAI SDK 응답 객체를 LLMResponse로 변환한다.

        vllm.py에서 사용하는 파싱 로직을 통합 제공한다.

        Args:
            response: openai.types.chat.ChatCompletion 객체.

        Returns:
            LLMResponse: 표준화된 응답 객체.
        """
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        content = message.content or "" if message else ""
        tool_calls = []

        if message and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        finish_reason = "stop"
        if tool_calls:
            finish_reason = "tool_calls"
        elif choice and choice.finish_reason == "length":
            finish_reason = "length"

        usage = response.usage
        return LLMResponse(
            content=content,
            tool_calls=tool_calls or None,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            },
        )
