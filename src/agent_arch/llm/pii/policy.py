"""가드 정책 — 전역 모드, 대상 provider, 회사 opt-in.

실효 모드 = 전역 ``PII_GUARD_MODE``(off|shadow|enforce) ∧ provider 가 대상 목록
(``PII_GUARD_PROVIDERS``) ∧ 회사가 켜져 있음. 회사 스위치는 이사님 요청대로 **opt-in,
기본 비활성화**다: ``PII_GUARD_FORCE_COMPANIES`` 목록이 있으면 그것이 결정하고(운영
강제·로컬 테스트), 없으면 common_app 회사 설정(``agent_pii_guard_enabled``)을 따른다
(회사 설정 조회는 어휘 PR 에서 붙는다 — 그 전까지는 강제 목록만 켠다).

vLLM 은 기본 목록에 없다(사내 전제). 임대 GPU 를 외부로 보려면 목록에 넣는다.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from agent_arch.core.config import settings
from agent_arch.llm.pii.tokens import Kind

logger = logging.getLogger(__name__)

Mode = Literal["off", "shadow", "enforce"]
_MODES = ("off", "shadow", "enforce")


def _csv(value: Optional[str]) -> frozenset[str]:
    return frozenset(p.strip().lower() for p in (value or "").split(",") if p.strip())


def global_mode() -> Mode:
    raw = str(getattr(settings, "pii_guard_mode", "off") or "off").strip().lower()
    if raw not in _MODES:
        logger.warning("[pii_guard] PII_GUARD_MODE=%r 를 모른다 — off 로 간주", raw)
        return "off"
    return raw  # type: ignore[return-value]


def guarded_providers() -> frozenset[str]:
    return _csv(getattr(settings, "pii_guard_providers", "gemini,openai,anthropic"))


def is_guarded_provider(provider_name: Optional[str]) -> bool:
    return bool(provider_name) and provider_name.strip().lower() in guarded_providers()


def block_kinds() -> frozenset[Kind]:
    out = set()
    for name in _csv(getattr(settings, "pii_guard_block_kinds", "SSN,APIKEY,PASSWORD")):
        try:
            out.add(Kind(name.upper()))
        except ValueError:
            logger.warning("[pii_guard] PII_GUARD_BLOCK_KINDS 의 %r 를 모른다 — 무시", name)
    return frozenset(out)


def secret() -> str:
    """토큰 HMAC 비밀. 전용 값이 없으면 JWT 서명 키를 쓴다(항상 있다)."""
    return str(getattr(settings, "pii_guard_secret", "") or getattr(settings, "secret_key", "") or "")


def forced_companies() -> Optional[frozenset[str]]:
    """``PII_GUARD_FORCE_COMPANIES`` — 비어 있으면 None(회사 설정을 따른다)."""
    raw = (getattr(settings, "pii_guard_force_companies", "") or "").strip()
    if not raw:
        return None
    return frozenset(c.strip() for c in raw.split(",") if c.strip())


async def company_enabled(company: Optional[str]) -> bool:
    """회사가 가드를 켰는가. 강제 목록이 있으면 그것이, 없으면 회사 설정이 결정한다."""
    forced = forced_companies()
    if forced is not None:
        return bool(company) and company in forced
    return await _company_setting_enabled(company)


LAST_KNOWN_KEY = "pii_guard:last_known:{company}"
LAST_KNOWN_TTL = 24 * 3600


async def _company_setting_enabled(company: Optional[str]) -> bool:
    """common_app ``agent_pii_guard_enabled``(플래그 키 ``pii_guard``). 키가 없거나 조회 실패면 마지막으로 읽은 값.

    common_app 은 정의 미등록·조회 실패를 전부 False 로 돌려주고 agent 쪽 조회도 실패하면 {} 다.
    opt-in 키라 그대로 두면 짧은 장애에 가드가 꺼진다 — 마지막으로 읽은 True 를 24시간 보관해
    조회 실패(키 부재)에서만 그 값을 쓴다. 명시 False 는 즉시 반영한다.
    """
    if not company:
        return False
    from agent_arch.services.web.guard import fetch_company_agent_features

    features = await fetch_company_agent_features(company)
    key = LAST_KNOWN_KEY.format(company=company)
    if isinstance(features, dict) and "pii_guard" in features:
        enabled = bool(features.get("pii_guard"))
        try:
            from agent_arch.memory.redis_store import redis_store
            if enabled:
                await redis_store.set(key, True, expire=LAST_KNOWN_TTL)
            else:
                await redis_store.delete(key)
        except Exception:  # noqa: BLE001
            pass
        return enabled
    try:
        from agent_arch.memory.redis_store import redis_store
        last = await redis_store.get(key)
    except Exception:  # noqa: BLE001
        last = None
    if last:
        logger.warning("[pii_guard] 회사 플래그 조회 실패 — 마지막으로 읽은 켬 값을 유지 (company=%s)", company)
    return bool(last)


def rules_for_settings():
    """기본 정규식 규칙 + 회사 사번 형식(``PII_GUARD_EMPNO_REGEX``)이 있으면 사번 규칙."""
    from agent_arch.llm.pii.patterns import RULES, empno_rule

    raw = (getattr(settings, "pii_guard_empno_regex", "") or "").strip()
    if not raw:
        return RULES
    try:
        return (*RULES, empno_rule(raw))
    except Exception:  # noqa: BLE001
        logger.warning("[pii_guard] PII_GUARD_EMPNO_REGEX 컴파일 실패 — 사번 규칙 없이 진행: %r", raw)
        return RULES


async def resolve_mode(provider_name: Optional[str], company: Optional[str]) -> Mode:
    mode = global_mode()
    if mode == "off" or not is_guarded_provider(provider_name):
        return "off"
    if not await company_enabled(company):
        return "off"
    return mode
