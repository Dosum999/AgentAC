"""사용자 개인설정 서비스.

Redis 대화 이력 정책(보존 메시지 수 + TTL)을 사용자 단위로 저장·조회.

레이어:
    repository.py  agent.user_preferences CRUD (asyncpg)
    service.py     Redis 60s 캐시 wrapper + negative cache
"""
from __future__ import annotations

from agent_arch.services.preferences.repository import (
    UserPreferences,
    delete_preferences,
    find_preferences,
    upsert_preferences,
)
from agent_arch.services.preferences.service import (
    get_for_user,
    invalidate,
)

__all__ = [
    "UserPreferences",
    "find_preferences",
    "upsert_preferences",
    "delete_preferences",
    "get_for_user",
    "invalidate",
]
