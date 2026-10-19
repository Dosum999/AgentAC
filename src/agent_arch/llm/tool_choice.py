"""OpenAI 호환(vllm·openai) 공급자의 tool_choice 파라미터 계산.

tool_mode 규약(Gemini FunctionCallingConfig 와 동일): AUTO | ANY | NONE.
allowed_tools 가 함께 오고 ANY 이면 그 도구 하나로 고정한다(guide-first).
OpenAI 규격은 함수 1개만 지정할 수 있어 여러 개면 ``required`` 로 완화한다.
"""
from typing import Any, Optional, Sequence

_TOOL_CHOICE_MAP = {
    "AUTO": "auto",
    "ANY": "required",
    "NONE": "none",
}


def _tool_choice_param(tool_mode: str, allowed_tools: Optional[Sequence[str]]) -> Any:
    """tool_mode + allowed_tools → OpenAI ``tool_choice`` 값."""
    if tool_mode == "ANY" and allowed_tools and len(allowed_tools) == 1:
        return {"type": "function", "function": {"name": allowed_tools[0]}}
    return _TOOL_CHOICE_MAP.get(tool_mode, "auto")
