"""사용자 개인설정 조회 캐시 wrapper.

매 대화 턴(``append_turn``)마다 호출되는 핫패스이므로 PostgreSQL 부하를 줄이기 위해
Redis 60초 캐시를 둔다. 행이 없는 사용자도 sentinel 마커로 negative-cache해
미등록 사용자도 매 턴 DB miss가 발생하지 않게 한다.

흐름:
    get_for_user(user_sub)
      ├─ Redis HIT (pref:v1:{sub})
      │    ├─ "__NEGATIVE__" → None 반환 (행 없음 = 전역 기본값 사용)
      │    └─ dict → UserPreferences 재구성 후 반환
      └─ Redis MISS
           ├─ repository.find_preferences → UserPreferences | None
           └─ Redis SET (행 있으면 직렬화, 없으면 sentinel) → 반환

캐시 무효화는 PUT/DELETE 라우터에서 ``invalidate(user_sub)`` 호출.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from agent_arch.memory.redis_store import redis_store
from agent_arch.services.preferences.repository import (
    UserPreferences,
    find_preferences,
)

logger = logging.getLogger(__name__)

# 캐시 키 prefix — 스키마 변경 시 v2로 올려 자연 무효화
_CACHE_KEY_PREFIX = "pref:v1:"

# 미등록 사용자용 negative cache 마커 — Mattermost 권한 캐시와 동일 패턴
_NEGATIVE_MARKER = "__NEGATIVE__"

# 캐시 TTL (초). 사용자 설정 변경은 드물고 짧게 stale해도 영향 미미.
_CACHE_TTL_SECONDS = 60


def _cache_key(user_sub: str) -> str:
    """사용자 sub로 캐시 키를 생성한다."""
    return f"{_CACHE_KEY_PREFIX}{user_sub}"


def _serialize(prefs: UserPreferences) -> dict:
    """dataclass를 캐시 저장용 dict로 변환한다. datetime은 ISO 8601 문자열."""
    return {
        "user_sub": prefs.user_sub,
        "max_history": prefs.max_history,
        "session_ttl_seconds": prefs.session_ttl_seconds,
        "created_at": prefs.created_at.isoformat(),
        "updated_at": prefs.updated_at.isoformat(),
    }


def _deserialize(cached: dict) -> Optional[UserPreferences]:
    """캐시 dict를 ``UserPreferences``로 복원한다. 스키마 mismatch 시 None."""
    try:
        return UserPreferences(
            user_sub=cached["user_sub"],
            max_history=cached["max_history"],
            session_ttl_seconds=cached["session_ttl_seconds"],
            created_at=datetime.fromisoformat(cached["created_at"]),
            updated_at=datetime.fromisoformat(cached["updated_at"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("개인설정 캐시 스키마 mismatch — 재조회 (err=%s)", exc)
        return None


async def get_for_user(user_sub: Optional[str]) -> Optional[UserPreferences]:
    """사용자 개인설정을 조회한다 (캐시 우선).

    Args:
        user_sub: Keycloak user sub. None/빈 문자열이면 즉시 None.

    Returns:
        ``UserPreferences`` 또는 None.
        None은 "행이 없음 → 호출자가 전역 기본값을 적용해야 함"을 의미한다.
    """
    if not user_sub:
        return None

    key = _cache_key(user_sub)

    # 1. 캐시 조회
    cached = await redis_store.get(key)
    if cached == _NEGATIVE_MARKER:
        return None
    if isinstance(cached, dict):
        prefs = _deserialize(cached)
        if prefs is not None:
            return prefs
        # mismatch는 아래 재조회로 흐른다

    # 2. 캐시 미스 → DB 조회
    prefs = await find_preferences(user_sub)
    if prefs is None:
        # 미등록 사용자도 캐시 — 매 턴 DB miss 회피
        await redis_store.set(key, _NEGATIVE_MARKER, expire=_CACHE_TTL_SECONDS)
        return None

    # 3. 결과 캐시
    await redis_store.set(key, _serialize(prefs), expire=_CACHE_TTL_SECONDS)
    return prefs


async def invalidate(user_sub: str) -> bool:
    """사용자 개인설정 캐시를 즉시 삭제한다.

    PUT/DELETE 라우터가 DB 변경 직후 호출한다.

    Args:
        user_sub: 사용자 Keycloak sub.

    Returns:
        삭제 성공 여부.
    """
    if not user_sub:
        return False
    return await redis_store.delete(_cache_key(user_sub))
