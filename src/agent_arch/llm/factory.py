import logging
from typing import Dict, Any, Optional
from agent_arch.llm.base import BaseLLM
from agent_arch.core.config import settings

logger = logging.getLogger(__name__)


# 외부 별칭 → canonical 프로바이더 이름. 프론트엔드 호환을 위한 입력 정규화 전용.
_PROVIDER_ALIASES: Dict[str, str] = {
    "gemma": "vllm",
    # 사용자 노출 이름. 구현·설정·회사 허용목록의 canonical id 는 "anthropic" 하나다.
    "claude": "anthropic",
}


class LLMFactory:
    """LLM 프로바이더 팩토리 클래스.

    설정에 따라 적절한 LLM 프로바이더 인스턴스를 생성하고 관리합니다.
    프로바이더 등록, 생성, fallback 처리를 담당합니다.

    Attributes:
        _providers: 등록된 LLM 프로바이더들의 딕셔너리
    """

    _providers: Dict[str, type] = {}

    @classmethod
    def register_provider(cls, name: str, provider_class: type):
        """LLM 프로바이더를 등록합니다.

        Args:
            name (str): 프로바이더 이름 (gemini, openai, vllm 등)
            provider_class (type): BaseLLM을 상속한 프로바이더 클래스

        Examples:
            >>> LLMFactory.register_provider("gemini", GeminiLLM)
        """
        cls._providers[name.lower()] = provider_class

    @classmethod
    def normalize_provider(cls, provider: Optional[str]) -> Optional[str]:
        """프로바이더 별칭을 canonical 이름으로 정규화한다.

        빈 값(None / "")은 그대로 반환해 호출자의 settings fallback 로직에 위임한다.
        등록되지 않은 값은 그대로 반환되어 후속 검증(LLMFactory.create) 단계에서
        ValueError로 거부된다.

        Args:
            provider: 외부에서 전달된 프로바이더 이름 (별칭 포함)

        Returns:
            정규화된 canonical 이름 또는 빈 값 그대로
        """
        if not provider:
            return provider
        normalized = provider.strip().lower()
        return _PROVIDER_ALIASES.get(normalized, normalized)

    @classmethod
    def create(cls, provider: str = None, **kwargs) -> BaseLLM:
        """LLM 프로바이더 인스턴스를 생성합니다.

        Args:
            provider (str): 프로바이더 이름. None이면 기본값 사용. 별칭(예: "gemma")도 허용.
            **kwargs: 추가 설정. config를 오버라이드합니다

        Returns:
            BaseLLM: LLM 프로바이더 인스턴스

        Raises:
            ValueError: 등록되지 않은 프로바이더인 경우

        Examples:
            >>> llm = LLMFactory.create("gemini", temperature=0.5)
            >>> llm = LLMFactory.create()  # 기본 프로바이더 사용
            >>> llm = LLMFactory.create("gemma")  # 별칭 → vllm으로 정규화
        """
        provider_name = cls.normalize_provider(provider) or settings.default_llm_provider
        provider_name = provider_name.lower()

        if provider_name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(
                f"Unknown LLM provider: {provider_name}. "
                f"Available providers: {available}"
            )

        # YAML에서 프로바이더 설정 로드
        config = settings.get_llm_config(provider_name)

        # kwargs로 설정 오버라이드
        config.update(kwargs)

        # 프로바이더 클래스에 선언된 _settings_config로 settings 값을 동적 주입
        # 새 프로바이더 추가 시 factory.py 수정 없이 클래스에 _settings_config만 선언하면 됨
        #
        # 단, **호출자가 명시한 kwargs 가 우선**한다. 예전에는 이 주입이 kwargs 뒤에
        # 실행돼 `create("openai", api_key=사용자키)` 가 조용히 .env 키로 바뀌었다 —
        # 회사별 API 키(BYO)를 넘길 수 없던 원인이다.
        provider_class = cls._providers[provider_name]
        for config_key, settings_attr in getattr(provider_class, "_settings_config", {}).items():
            if kwargs.get(config_key) not in (None, ""):
                continue
            val = getattr(settings, settings_attr, None)
            if val is not None:
                config[config_key] = val

        # 외부 provider 면 가명화 래퍼로 감싼다(app/llm/pii). 대상이 아니거나 모드 off 면 그대로.
        from agent_arch.llm.pii import wrap_if_external
        return wrap_if_external(provider_class(config), provider_name)

    @classmethod
    def create_with_fallback(cls) -> BaseLLM:
        """Fallback 체인을 사용해 LLM을 생성합니다.

        설정된 fallback_chain 순서대로 시도하여 사용 가능한 첫 번째 프로바이더를 반환합니다.
        체인이 비어 있으면 **DEFAULT_LLM_PROVIDER** 를 쓴다 — 그것이 기본값이고, YAML 에
        프로바이더를 박아 두는 것은 예외적인 선택이다.

        ⚠️ 이 체인은 **생성 실패만** 잡는다. 클라이언트 생성은 거의 언제나 성공하므로
        호출 시점에 죽는 프로바이더(서빙 중단·타임아웃)는 뒤로 넘어가지 않고 그 자리에서
        실패한다. 그래서 선두에 프로바이더를 적어 두면 그 레인만 따로 죽는다
        (2026-09-18: 선두 `vllm` 이 내려가 대화 자동 압축이 턴마다 실패했다).

        Returns:
            사용 가능한 첫 번째 LLM 프로바이더

        Raises:
            RuntimeError: 모든 프로바이더 생성 실패 시

        Examples:
            >>> llm = LLMFactory.create_with_fallback()
        """
        fallback_chain = settings.llm_providers.get("fallback_chain", [])

        if not fallback_chain:
            fallback_chain = [settings.default_llm_provider]

        last_error = None

        for provider_name in fallback_chain:
            try:
                llm = cls.create(provider_name)
            except Exception as e:
                last_error = e
                continue
            # 어느 프로바이더가 골라졌는지 남긴다 — 호출 실패는 이 함수가 못 잡으므로,
            # 레인이 죽었을 때 "무엇을 치고 있었나" 를 로그 한 줄로 알 수 있어야 한다.
            logger.debug("[llm] fallback 선택: %s", provider_name)
            return llm

        raise RuntimeError(
            f"Failed to create any LLM provider. Last error: {last_error}"
        )

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """등록된 모든 프로바이더 이름을 반환합니다.

        Returns:
            등록된 프로바이더 이름 리스트

        Examples:
            >>> providers = LLMFactory.get_available_providers()
            >>> print(providers)
            ['gemini', 'openai', 'vllm']
        """
        return list(cls._providers.keys())


# 프로바이더 자동 등록을 위한 데코레이터 헬퍼
def register_llm_provider(name: str):
    """LLM 프로바이더 자동 등록 데코레이터.

    클래스 정의 시 자동으로 팩토리에 등록됩니다.

    Args:
        name (str): 등록할 프로바이더 이름

    Returns:
        callable: 데코레이터 함수

    Examples:
        >>> @register_llm_provider("gemini")
        ... class GeminiLLM(BaseLLM):
        ...     pass
    """
    def decorator(cls):
        LLMFactory.register_provider(name, cls)
        return cls
    return decorator
