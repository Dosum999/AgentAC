"""LLM provider 허용목록 정책.

요청이 고른 provider 를 허용목록으로 게이트하고, 미허용·미지정은 **배포 기본값
(DEFAULT_LLM_PROVIDER)으로 폴백**한다. 허용목록은 ``LLM_ALLOWED_PROVIDERS``(콤마 구분)로 주고,
비어 있으면 게이트 없이 요청값(정규화) 또는 기본값을 쓴다.

⚠️ **폴백에 프로바이더 이름을 박지 않는다.** 폴백을 특정 provider 상수로 두면, 그 서빙을 내린
순간 폴백이 "안전한 기본"이 아니라 "확정 실패"가 된다(대화 압축·메모리 레인에서 같은 사고를 겪었다).
어느 모델로 돌릴지는 배포가 ``DEFAULT_LLM_PROVIDER`` 하나로 정하고, 이 모듈은 "허용된 것 중에서
고른다"만 한다. 테넌트별 허용목록이 필요하면 ``get_company_allowed_providers`` 를 그 저장소로
갈아 끼운다(시그니처에 company 를 남겨 둔 이유).
"""
import logging
from typing import Optional

from agent_arch.core.config import settings
from agent_arch.llm.factory import LLMFactory

logger = logging.getLogger(__name__)


def _fallback_provider() -> str:
    """폴백 provider — **배포 기본값을 따른다**(호출 시점에 읽는다)."""
    return LLMFactory.normalize_provider(settings.default_llm_provider) or ""


def _configured_allowed() -> list[str]:
    raw = str(getattr(settings, "llm_allowed_providers", "") or "")
    return [p for p in (LLMFactory.normalize_provider(x.strip()) for x in raw.split(",")) if p]


async def get_company_allowed_providers(company: Optional[str]) -> list[str]:
    """사용 가능한 LLM provider 목록. 폴백 provider(배포 기본값)는 항상 포함한다.

    Args:
        company: 테넌트 키(현재는 쓰지 않는다 — 테넌트별 정책 저장소를 붙일 자리).
    """
    allowed = _configured_allowed()
    fallback = _fallback_provider()
    if fallback and fallback not in allowed:
        allowed = [fallback, *allowed]
    return allowed


async def resolve_effective_provider(requested: Optional[str], company: Optional[str]) -> str:
    """요청 provider 를 허용목록으로 게이트해 실제 사용할 provider 를 정한다.

    허용목록이 비어 있으면 게이트 없이 ``요청(정규화) 또는 기본값``. 있으면 목록 밖·미지정은 기본값.
    """
    normalized = LLMFactory.normalize_provider(requested)
    if not _configured_allowed():
        return normalized or settings.default_llm_provider
    allowed = await get_company_allowed_providers(company)
    if normalized and normalized in allowed:
        return normalized
    return _fallback_provider() or settings.default_llm_provider
