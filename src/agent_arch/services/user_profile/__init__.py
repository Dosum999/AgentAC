"""사용자 프로필 캐시 서비스.

호스트 프론트가 호출한 Frappe ``get_current_user`` 응답을 ui-components를 통해
백엔드에 동기 등록받아 Redis(핫 캐시) + Postgres(영속) 양쪽에 보관한다.

레이어:
    repository.py  agent.users CRUD (asyncpg)
    service.py     Redis 30분 캐시 + sha256 변경 감지로 DB write 절약
"""
from __future__ import annotations

from agent_arch.services.user_profile.normalizer import (
    InvalidUserProfileError,
    normalize_user_profile,
)
from agent_arch.services.user_profile.repository import (
    UserProfileRow,
    delete_profile_row,
    find_profile_hash,
    find_profile_row,
    find_profile_row_by_email,
    upsert_profile_row,
)
from agent_arch.services.user_profile.service import (
    REDIS_PROFILE_TTL_SECONDS,
    compute_profile_hash,
    get_profile,
    invalidate,
    upsert_profile,
)

__all__ = [
    "REDIS_PROFILE_TTL_SECONDS",
    "InvalidUserProfileError",
    "UserProfileRow",
    "compute_profile_hash",
    "delete_profile_row",
    "find_profile_hash",
    "find_profile_row",
    "find_profile_row_by_email",
    "get_profile",
    "invalidate",
    "normalize_user_profile",
    "upsert_profile",
    "upsert_profile_row",
]
