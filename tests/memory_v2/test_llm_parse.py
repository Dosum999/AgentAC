"""메모리 레인 생성 + 관대한 JSON 파서."""
from __future__ import annotations

import pytest

from agent_arch.services.memory_v2 import llm as llm_module
from agent_arch.services.memory_v2.llm import extract_json, prompt_sha, repair_json, strip_fence


def test_strip_fence_variants():
    assert strip_fence('```json\n{"a":1}\n```') == '{"a":1}'
    assert strip_fence('```\n[1,2]\n```') == "[1,2]"
    assert strip_fence('{"a":1}') == '{"a":1}'


def test_extract_object_from_fenced_text():
    data = extract_json('```json\n{"scene_name":"메일 루틴","atoms":[]}\n```', "object")
    assert data == {"scene_name": "메일 루틴", "atoms": []}


def test_extract_array_embedded_in_prose():
    data = extract_json('판정 결과입니다: [{"new":"n1","action":"skip"}] 이상.', "array")
    assert data == [{"new": "n1", "action": "skip"}]


def test_repair_trailing_comma_and_bad_priority():
    raw = '{"atoms":[{"atom_type":"persona","content":"x","priority": high,},]}'
    data = extract_json(raw, "object")
    assert data is not None
    assert data["atoms"][0]["priority"] == 50


def test_quoted_priority_becomes_number():
    assert '"priority": 70' in repair_json('{"priority": "70"}')
    assert '"priority": -1' in repair_json('{"priority": "-1"}')


def test_kind_mismatch_returns_none():
    assert extract_json('{"a":1}', "array") is None
    assert extract_json("[1]", "object") is None
    assert extract_json("", "object") is None


def test_prompt_sha_is_stable_16_hex():
    a, b = prompt_sha("hello"), prompt_sha("hello")
    assert a == b and len(a) == 16 and int(a, 16) >= 0
    assert prompt_sha("hello2") != a


def test_create_memory_llm_uses_lane(monkeypatch):
    captured = {}

    def fake_create(provider, **kwargs):
        captured["provider"] = provider
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(llm_module.LLMFactory, "create", staticmethod(fake_create))
    monkeypatch.setattr(llm_module.settings, "llm_providers", {
        "memory": {"provider": "gemini", "model": "gemini-2.5-flash", "max_tokens": 2048, "temperature": 0.1},
    }, raising=False)
    llm_module.create_memory_llm()
    assert captured["provider"] == "gemini"
    assert captured["model"] == "gemini-2.5-flash"
    assert captured["max_tokens"] == 2048
    assert captured["temperature"] == 0.1


def test_create_memory_llm_follows_env_default_provider(monkeypatch):
    """레인이 provider 를 안 적으면 배포 기본값(DEFAULT_LLM_PROVIDER)을 따른다.

    예전엔 레인 기본값에 vllm 이 박혀 있어, vLLM 을 내리면 대화는 다른 프로바이더로 돌아도
    메모리 추출만 계속 죽은 주소를 쳤다(2026-09-18 실측). 모델도 안 적으므로 그 프로바이더의
    기본 모델을 쓴다 — gemma 모델명이 gemini 로 실려 가는 사고를 막는다.
    """
    captured = {}
    monkeypatch.setattr(llm_module.LLMFactory, "create", staticmethod(lambda p, **kw: captured.update({"p": p, **kw})))
    monkeypatch.setattr(llm_module.settings, "llm_providers", {}, raising=False)
    monkeypatch.setattr(llm_module.settings, "default_llm_provider", "gemini", raising=False)
    llm_module.create_memory_llm()
    assert captured["p"] == "gemini"
    assert "model" not in captured
    assert captured["max_tokens"] == 4096      # 배치에 필요한 토큰은 레인이 계속 정한다


@pytest.mark.asyncio
async def test_invoke_text_returns_content_model_latency(monkeypatch):
    class _Resp:
        content = ' {"ok": true} '
        finish_reason = "stop"

    class _LLM:
        model = "fake-model"

        async def invoke(self, messages, tool_mode=None):
            return _Resp()

    monkeypatch.setattr(llm_module, "create_memory_llm", lambda: _LLM())
    content, model, latency = await llm_module.invoke_text([])
    assert content == '{"ok": true}' and model == "fake-model" and latency >= 0
