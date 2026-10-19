"""tool_mode=ANY 에 allowed_tools 를 함께 주면 공급자별로 '그 도구만' 강제한다."""
from unittest.mock import MagicMock

import pytest

from agent_arch.llm.gemini import GeminiLLM
from agent_arch.llm.vllm import _tool_choice_param as vllm_tool_choice
from agent_arch.llm.openai_provider import _tool_choice_param as openai_tool_choice


def _tool(name):
    t = MagicMock()
    t.name = name
    t.description = "d"
    t.parameters = {"query": {"type": "string", "description": "q"}}
    t.required_params = ["query"]
    return t


def test_gemini_allowed_function_names_when_any_with_allowed_tools():
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.temperature = 0.1
    llm.max_tokens = 100
    llm.top_p = 0.9
    llm.thinking_budget = None
    llm.supports_function_calling = True
    llm._bound_tools = [_tool("search_user_guide_kb"), _tool("hr_x")]
    cfg = llm._build_generation_config([], {"tool_mode": "ANY", "allowed_tools": ["search_user_guide_kb"]})
    fcc = cfg["tool_config"].function_calling_config
    assert fcc.mode == "ANY"
    assert list(fcc.allowed_function_names) == ["search_user_guide_kb"]


def test_gemini_no_allowed_names_by_default():
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.temperature = 0.1
    llm.max_tokens = 100
    llm.top_p = 0.9
    llm.thinking_budget = None
    llm.supports_function_calling = True
    llm._bound_tools = [_tool("a")]
    cfg = llm._build_generation_config([], {"tool_mode": "AUTO"})
    fcc = cfg["tool_config"].function_calling_config
    assert fcc.mode == "AUTO"
    assert not fcc.allowed_function_names


@pytest.mark.parametrize("fn", [vllm_tool_choice, openai_tool_choice])
def test_openai_compatible_tool_choice_pins_single_tool(fn):
    assert fn("AUTO", None) == "auto"
    assert fn("ANY", None) == "required"
    assert fn("NONE", None) == "none"
    assert fn("ANY", ["search_user_guide_kb"]) == {
        "type": "function", "function": {"name": "search_user_guide_kb"}
    }
    # OpenAI 규격은 도구 1개만 지정 가능 → 여러 개면 required 로 완화
    assert fn("ANY", ["a", "b"]) == "required"
    # AUTO 에 allowed_tools 는 의미 없음 → 무시
    assert fn("AUTO", ["a"]) == "auto"
