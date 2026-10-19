"""LLMFactory.normalize_provider() 별칭 정규화 동작 검증.

프론트엔드 호환을 위해 추가된 `gemma -> vllm` 같은 별칭 매핑이
대소문자/공백/None/빈 문자열/미등록 값에 대해 안전하게 동작하는지 확인한다.
"""
import pytest

from agent_arch.llm.factory import LLMFactory, _PROVIDER_ALIASES


class TestNormalizeProvider:
    """LLMFactory.normalize_provider 단위 테스트."""

    def test_gemma_resolves_to_vllm(self):
        """프론트엔드 별칭 'gemma'는 canonical 'vllm'으로 정규화된다."""
        assert LLMFactory.normalize_provider("gemma") == "vllm"

    def test_uppercase_alias_normalized(self):
        """대문자 입력도 별칭 매핑이 동작한다."""
        assert LLMFactory.normalize_provider("GEMMA") == "vllm"

    def test_alias_with_whitespace(self):
        """앞뒤 공백은 trim된다."""
        assert LLMFactory.normalize_provider("  gemma  ") == "vllm"

    def test_canonical_name_passthrough(self):
        """canonical 이름은 소문자로 통일되어 반환된다."""
        assert LLMFactory.normalize_provider("vllm") == "vllm"
        assert LLMFactory.normalize_provider("VLLM") == "vllm"

    def test_other_providers_unchanged(self):
        """별칭이 아닌 프로바이더는 소문자 정규화만 적용된다."""
        assert LLMFactory.normalize_provider("gemini") == "gemini"
        assert LLMFactory.normalize_provider("Gemini") == "gemini"
        assert LLMFactory.normalize_provider("VLLM") == "vllm"
        assert LLMFactory.normalize_provider("openai") == "openai"

    def test_none_passthrough(self):
        """None은 settings fallback을 위해 그대로 반환된다."""
        assert LLMFactory.normalize_provider(None) is None

    def test_empty_string_passthrough(self):
        """빈 문자열도 그대로 반환되어 호출자의 fallback에 위임된다."""
        assert LLMFactory.normalize_provider("") == ""

    def test_unknown_provider_passthrough(self):
        """미등록 값은 정규화하지 않고 그대로 반환되어 create() 단계에서 거부된다."""
        assert LLMFactory.normalize_provider("chatgpt") == "chatgpt"
        assert LLMFactory.normalize_provider("mistral") == "mistral"

    def test_claude_resolves_to_anthropic(self):
        """사용자 노출 이름 'claude'는 canonical 'anthropic'으로 정규화된다."""
        assert LLMFactory.normalize_provider("claude") == "anthropic"
        assert LLMFactory.normalize_provider("CLAUDE") == "anthropic"

    def test_alias_map_contains_gemma(self):
        """현 시점의 별칭 매핑 사양: gemma -> vllm."""
        assert _PROVIDER_ALIASES.get("gemma") == "vllm"


class TestCreateWithAlias:
    """LLMFactory.create()가 별칭을 통해 올바른 프로바이더 클래스를 인스턴스화하는지 확인."""

    def test_create_with_gemma_returns_vllm_class(self, monkeypatch):
        """create('gemma')가 'vllm'으로 등록된 클래스를 인스턴스화한다."""
        # 데코레이터 등록을 보장하기 위해 모듈 import
        import agent_arch.llm.vllm  # noqa: F401

        captured = {}

        def fake_provider_class(config):
            captured["instantiated"] = True
            return object()

        monkeypatch.setitem(LLMFactory._providers, "vllm", fake_provider_class)
        monkeypatch.setattr(
            "agent_arch.llm.factory.settings.get_llm_config",
            lambda name: {},
        )

        result = LLMFactory.create("gemma")
        assert captured.get("instantiated") is True
        assert result is not None



    def test_create_with_unknown_raises_valueerror(self):
        """미등록 프로바이더는 ValueError를 던진다 (별칭 정규화 후에도 동일)."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            LLMFactory.create("nonexistent_provider")
