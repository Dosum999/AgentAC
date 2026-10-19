"""OpenAI Responses API(`/v1/responses`) 변환 — OpenAILLM 의 옵트인 경로(config ``api: responses``).

왜 따로 두나: gpt-6.1-sol 은 Chat Completions 에서 함수 도구를 쓰면 400 이다 — 추론 강도를 주면 "Function tools with
reasoning_effort are not supported ... use /v1/responses", 'none' 을 주면 "does not support 'none'"(2026-10-16 실측,
둘 다 거부). Responses API 에서는 도구 + reasoning.effort(low~max) 가 함께 된다.

규약은 Chat Completions 경로와 같다 — 입력은 우리 Message, 출력은 LLMResponse / StreamEvent. 서버에 대화를 남기지
않는다(store=False): 매 호출이 전체 대화를 보낸다(Chat Completions 경로와 같은 방식). 추론 항목은 되돌려 보내지 않는다.
"""

import json
from typing import Any, Dict, List, Optional, Sequence

from agent_arch.llm.base import LLMResponse, Message, ToolCall


def to_input(messages: List[Message]) -> list:
    """우리 메시지 → Responses ``input`` 항목.

    - assistant + tool_calls → (본문 메시지) + function_call 항목들
    - tool → function_call_output
    - 이미지·PDF → input_image / input_file 파트
    """
    items: list = []
    for msg in messages:
        if msg.role == "assistant" and msg.tool_calls:
            if msg.content:
                items.append({"role": "assistant", "content": msg.content})
            for tc in msg.tool_calls:
                items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                })
        elif msg.role == "tool":
            items.append({"type": "function_call_output", "call_id": msg.tool_call_id, "output": msg.content or ""})
        elif msg.images:
            parts: List[Dict[str, Any]] = []
            if msg.content:
                parts.append({"type": "input_text", "text": msg.content})
            for idx, img in enumerate(msg.images):
                if img.mime_type == "application/pdf":
                    parts.append({"type": "input_file", "filename": f"document-{idx + 1}.pdf",
                                  "file_data": f"data:application/pdf;base64,{img.data}"})
                else:
                    parts.append({"type": "input_image", "image_url": f"data:{img.mime_type};base64,{img.data}"})
            items.append({"role": msg.role, "content": parts})
        else:
            items.append({"role": msg.role, "content": msg.content or ""})
    return items


def to_tool(chat_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Chat Completions 도구 스키마({type, function:{...}}) → Responses 도구({type, name, description, parameters})."""
    fn = chat_schema.get("function") or {}
    return {"type": "function", "name": fn.get("name"), "description": fn.get("description") or "",
            "parameters": fn.get("parameters") or {"type": "object", "properties": {}}}


def to_tool_choice(tool_mode: str, allowed_tools: Optional[Sequence[str]]) -> Any:
    """tool_mode(AUTO|ANY|NONE) → Responses ``tool_choice``. 규칙은 app.llm.tool_choice 와 같다(함수 하나만 고정 가능)."""
    if tool_mode == "ANY" and allowed_tools and len(allowed_tools) == 1:
        return {"type": "function", "name": allowed_tools[0]}
    return {"ANY": "required", "NONE": "none"}.get(tool_mode, "auto")


def usage_of(u: Any) -> Optional[Dict[str, int]]:
    if u is None:
        return None
    from agent_arch.services.usage.meter import cache_token_fields

    return {"prompt_tokens": getattr(u, "input_tokens", 0) or 0,
            "completion_tokens": getattr(u, "output_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
            **cache_token_fields(getattr(u, "input_tokens_details", None))}


def parse_args(raw: Optional[str]) -> Dict[str, Any]:
    try:
        args = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        return {}
    return args if isinstance(args, dict) else {}


def tool_call_of(item: Any) -> ToolCall:
    name = getattr(item, "name", None) or "unknown"
    return ToolCall(id=getattr(item, "call_id", None) or f"call_{name}", name=name,
                    arguments=parse_args(getattr(item, "arguments", None)))


def finish_of(response: Any, has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    details = getattr(response, "incomplete_details", None)
    if getattr(response, "status", None) == "incomplete" and getattr(details, "reason", None) == "max_output_tokens":
        return "length"
    return "stop"


def parse_response(response: Any) -> LLMResponse:
    """Responses 응답 객체 → LLMResponse(본문은 message 항목의 output_text 를 잇는다)."""
    text: List[str] = []
    calls: List[ToolCall] = []
    for item in getattr(response, "output", None) or []:
        kind = getattr(item, "type", "")
        if kind == "function_call":
            calls.append(tool_call_of(item))
        elif kind == "message":
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", "") == "output_text":
                    text.append(getattr(part, "text", "") or "")
    return LLMResponse(content="".join(text), tool_calls=calls or None,
                       finish_reason=finish_of(response, bool(calls)),
                       usage=usage_of(getattr(response, "usage", None)) or
                       {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
