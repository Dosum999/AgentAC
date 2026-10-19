"""`LLMFactory.create_with_fallback` 은 배포 기본값을 따른다.

왜 있는가 — 2026-09-18 운영: `fallback_chain` 선두가 `vllm` 이라, vLLM 을 내린 뒤에도 대화
자동 압축이 계속 죽은 주소를 쳐 턴마다 `압축 LLM 호출 실패 … ConnectTimeout` 으로 끝났다.
memory·verify 레인에서 이미 겪은 것과 같은 사고이고(7238cab), 이 체인만 남아 있었다.

체인은 **생성 실패만** 잡고 호출 실패는 못 잡는다 — 생성은 거의 언제나 성공하므로 선두에
적힌 프로바이더가 사실상 고정된다. 그래서 여기 프로바이더를 박으면 안 된다.
"""
from __future__ import annotations

import pytest

from agent_arch.core.config import settings
from agent_arch.llm.factory import LLMFactory


def test_shipped_config_pins_no_provider():
    """레포가 들고 나가는 값이 비어 있어야 한다 — 이 파일이 회귀를 막는 자리다."""
    assert not settings.llm_providers.get("fallback_chain")


def test_empty_chain_follows_the_deployment_default(monkeypatch):
    picked = []
    monkeypatch.setattr(settings, "llm_providers", {"fallback_chain": []}, raising=False)
    monkeypatch.setattr(settings, "default_llm_provider", "gemini", raising=False)
    monkeypatch.setattr(LLMFactory, "create",
                        classmethod(lambda cls, p=None, **kw: picked.append(p) or object()))
    LLMFactory.create_with_fallback()
    assert picked == ["gemini"]


def test_missing_key_follows_the_deployment_default(monkeypatch):
    picked = []
    monkeypatch.setattr(settings, "llm_providers", {}, raising=False)
    monkeypatch.setattr(settings, "default_llm_provider", "gemini", raising=False)
    monkeypatch.setattr(LLMFactory, "create",
                        classmethod(lambda cls, p=None, **kw: picked.append(p) or object()))
    LLMFactory.create_with_fallback()
    assert picked == ["gemini"]


def test_explicit_chain_is_still_honoured_in_order(monkeypatch):
    """임시로 특정 프로바이더에 고정하고 싶을 때는 여전히 적을 수 있다."""
    seen = []

    def _create(cls, p=None, **kw):
        seen.append(p)
        if p == "vllm":
            raise RuntimeError("서빙 중단")
        return object()

    monkeypatch.setattr(settings, "llm_providers",
                        {"fallback_chain": ["vllm", "gemini"]}, raising=False)
    monkeypatch.setattr(LLMFactory, "create", classmethod(_create))
    LLMFactory.create_with_fallback()
    assert seen == ["vllm", "gemini"], "생성 실패는 다음 후보로 넘어가야 한다"


def test_all_failures_raise(monkeypatch):
    monkeypatch.setattr(settings, "llm_providers", {"fallback_chain": ["vllm"]}, raising=False)
    monkeypatch.setattr(LLMFactory, "create",
                        classmethod(lambda cls, p=None, **kw: (_ for _ in ()).throw(RuntimeError("x"))))
    with pytest.raises(RuntimeError, match="Failed to create any LLM provider"):
        LLMFactory.create_with_fallback()
