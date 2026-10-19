"""정책(모드·provider·회사 opt-in)과 감사(값 비노출·redact_for_log)."""
import logging

import pytest

from agent_arch.core.config import settings
from agent_arch.llm.base import Message
from agent_arch.llm.pii import audit, policy, wrap_if_external, unwrap_llm
from agent_arch.llm.pii.engine import Substituter
from agent_arch.llm.pii.guard_llm import PIIGuardLLM
from agent_arch.llm.pii.lexicon import build_lexicon
from agent_arch.llm.pii.tokens import Kind, TokenIssuer


# ── policy ───────────────────────────────────────────────────────────────────
def test_global_mode_normalizes_and_rejects_unknown(monkeypatch, caplog):
    monkeypatch.setattr(settings, "pii_guard_mode", " Enforce ", raising=False)
    assert policy.global_mode() == "enforce"
    monkeypatch.setattr(settings, "pii_guard_mode", "bogus", raising=False)
    with caplog.at_level(logging.WARNING):
        assert policy.global_mode() == "off"


def test_guarded_providers_default_excludes_vllm(monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_providers", "gemini,openai,anthropic", raising=False)
    assert policy.is_guarded_provider("Gemini") and policy.is_guarded_provider("anthropic")
    assert not policy.is_guarded_provider("vllm") and not policy.is_guarded_provider(None)
    monkeypatch.setattr(settings, "pii_guard_providers", "gemini, vllm", raising=False)
    assert policy.is_guarded_provider("vllm")


def test_block_kinds_parse(monkeypatch, caplog):
    monkeypatch.setattr(settings, "pii_guard_block_kinds", "ssn, apikey,bogus", raising=False)
    with caplog.at_level(logging.WARNING):
        assert policy.block_kinds() == {Kind.SSN, Kind.APIKEY}


def test_secret_falls_back_to_secret_key(monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_secret", "", raising=False)
    monkeypatch.setattr(settings, "secret_key", "jwt-key", raising=False)
    assert policy.secret() == "jwt-key"
    monkeypatch.setattr(settings, "pii_guard_secret", "own", raising=False)
    assert policy.secret() == "own"


@pytest.mark.asyncio
async def test_resolve_mode_requires_company_opt_in(monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_mode", "shadow", raising=False)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "", raising=False)
    assert await policy.resolve_mode("gemini", "(주)회사") == "off"          # 회사 설정 조회 전엔 꺼짐(opt-in)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "(주)회사, 회사SB", raising=False)
    assert await policy.resolve_mode("gemini", "(주)회사") == "shadow"
    assert await policy.resolve_mode("gemini", "회사SB") == "shadow"
    assert await policy.resolve_mode("gemini", "다른회사") == "off"
    assert await policy.resolve_mode("gemini", None) == "off"
    assert await policy.resolve_mode("vllm", "(주)회사") == "off"
    monkeypatch.setattr(settings, "pii_guard_mode", "off", raising=False)
    assert await policy.resolve_mode("gemini", "(주)회사") == "off"


# ── wrap / unwrap ────────────────────────────────────────────────────────────
class _Dummy:
    config = {}
    model = "m"


def test_wrap_if_external_by_provider_and_mode(monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_mode", "shadow", raising=False)
    monkeypatch.setattr(settings, "pii_guard_providers", "gemini,openai,anthropic", raising=False)
    d = _Dummy()
    assert wrap_if_external(d, "vllm") is d
    w = wrap_if_external(d, "gemini")
    assert isinstance(w, PIIGuardLLM) and w.inner is d
    assert wrap_if_external(w, "gemini") is w                              # 재래핑 금지
    assert unwrap_llm(w) is d and unwrap_llm(d) is d
    monkeypatch.setattr(settings, "pii_guard_mode", "off", raising=False)
    assert wrap_if_external(d, "gemini") is d


def test_factory_wraps_external_provider(monkeypatch):
    from agent_arch.llm.factory import LLMFactory

    monkeypatch.setattr(settings, "pii_guard_mode", "shadow", raising=False)
    monkeypatch.setitem(LLMFactory._providers, "gemini", lambda config: _Dummy())
    monkeypatch.setitem(LLMFactory._providers, "vllm", lambda config: _Dummy())
    monkeypatch.setattr("agent_arch.llm.factory.settings.get_llm_config", lambda name: {})
    assert isinstance(LLMFactory.create("gemini"), PIIGuardLLM)
    assert not isinstance(LLMFactory.create("vllm"), PIIGuardLLM)


def test_provider_of_sees_through_wrapper(monkeypatch):
    import agent_arch.llm.gemini  # noqa: F401 — 먼저 등록시켜 두어야 provider_of 의 지연 import 가 덮어쓰지 않는다
    from agent_arch.agents.react.verify import provider_of
    from agent_arch.llm.factory import LLMFactory

    class _G:
        config = {}

    monkeypatch.setitem(LLMFactory._providers, "gemini", _G)
    assert provider_of(PIIGuardLLM(_G(), "gemini")) == "gemini"
    assert provider_of(_G()) == "gemini"


def test_evidence_budget_sees_through_wrapper():
    from agent_arch.agents.react import evidence_budget
    from agent_arch.llm.vllm import VllmLLM

    v = VllmLLM.__new__(VllmLLM)
    v.config = {}
    assert evidence_budget._small_window(PIIGuardLLM(v, "vllm")) is True


# ── audit ────────────────────────────────────────────────────────────────────
def _sub():
    lex = build_lexicon([("김보라", "user@example.com")])
    return Substituter(TokenIssuer("s", "conv:1"), lex).substitute_messages([
        Message(role="user", content="김보라님 010-1234-5678 주민번호 900101-1234567 계좌 신한 110-123-456789")])


def test_summarize_counts_observe_separately():
    s = audit.summarize(_sub())
    assert s["tokens"] == 2 and s["blocked"] == {"SSN": 1}
    assert s["by_kind"] == {"PERSON": 1, "PHONE": 1, "SSN": 1, "ACCOUNT(observe)": 1}
    assert s["by_source"] == {"directory": 1, "regex": 3} and s["est_extra_tokens"] == 14


def test_shadow_log_has_no_values_unless_reveal_in_dev(monkeypatch, caplog):
    sub = _sub()
    with caplog.at_level(logging.INFO, logger="agent_arch.llm.pii.audit"):
        audit.log_shadow(provider="gemini", model="m", scope="conv:1", sub=sub, scan_ms=1.0, lexicon_size=2)
    msg = caplog.records[-1].getMessage()
    assert "김보라" not in msg and "1234" not in msg and "'value'" not in msg
    monkeypatch.setattr(settings, "pii_guard_shadow_reveal", True, raising=False)
    monkeypatch.setattr(settings, "environment", "development", raising=False)
    with caplog.at_level(logging.INFO, logger="agent_arch.llm.pii.audit"):
        audit.log_shadow(provider="gemini", model="m", scope="conv:1", sub=sub, scan_ms=1.0, lexicon_size=2)
    assert "'value': '김보라'" in caplog.records[-1].getMessage()
    monkeypatch.setattr(settings, "environment", "production", raising=False)
    assert audit.reveal_allowed() is False


def test_redact_for_log_shapes_only():
    out = audit.redact_for_log({"to": ["a@b.kr"], "body": "김보라님께", "n": 2, "ok": True, "x": None, "d": {"k": "v"}})
    assert out == {"to": "<list:1>", "body": "<str:5>", "n": 2, "ok": True, "x": None, "d": {"k": "<str:1>"}}
