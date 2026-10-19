"""사용자 프로필 캐시 서비스.

ui-components가 호출한 ``POST /user/profile`` 요청을 받아 Redis + Postgres에
프로필을 보관한다. 일반 요청 경로(``auth.get_current_user``)는 Redis를 우선 조회해
``request.state.user``를 구성한다.

흐름 요약:

    upsert_profile(user_sub, profile)
      ├─ profile_hash 계산 (sha256 canonical JSON)
      ├─ DB에서 기존 hash 조회
      │    ├─ 동일 → DB write 생략 (updated=False)
      │    └─ 변경 → INSERT ... ON CONFLICT DO UPDATE (updated=True)
      └─ Redis SET user_profile:{sub} (TTL 30분) — 변경 여부와 무관

    get_profile(user_sub)
      ├─ Redis HIT  →  반환
      └─ Redis MISS →  DB 조회 → Redis backfill → 반환 (또는 None)

캐시 무효화는 ``invalidate(user_sub)`` 호출. 본 PR에서는 자동 무효화 트리거가
없고 등록(upsert) 또는 명시 호출만 갱신·삭제 경로다.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

from agent_arch.memory.redis_store import redis_store
from agent_arch.services.user_profile.normalizer import normalize_user_profile
from agent_arch.services.user_profile.repository import (
    find_profile_hash,
    find_profile_row,
    upsert_profile_row,
)

logger = logging.getLogger(__name__)

# Redis 키 prefix — 스키마 변경 시 v2로 올려 자연 무효화
_CACHE_KEY_PREFIX = "user_profile:"

# Redis 캐시 TTL (초). 24시간 — DB가 SOT이고 TTL 만료해도 get_profile()이 자동
# DB backfill 하므로 데이터 손실 위험이 없다. SESSION_TTL(대화 이력)과 같은 단위라
# 운영 시 직관적. 호스트가 매 마운트/refetch마다 재등록하므로 실제 만료는 드물다.
REDIS_PROFILE_TTL_SECONDS = 86400


def _cache_key(user_sub: str) -> str:
    """사용자 sub로 Redis 캐시 키를 생성한다."""
    return f"{_CACHE_KEY_PREFIX}{user_sub}"


def compute_profile_hash(profile: dict) -> str:
    """프로필 dict의 결정적 sha256 hash를 계산한다.

    ``sort_keys=True``로 키 순서를 정규화하고, ``separators=(",", ":")``로
    공백을 제거해 같은 내용은 항상 같은 hash가 되도록 한다. 한글 회사명 등
    유니코드 문자열은 ``ensure_ascii=False``로 그대로 두어 표준 비교가 가능하다.

    Args:
        profile: Frappe ``get_current_user`` 응답.

    Returns:
        64자 hex 문자열.
    """
    canonical = json.dumps(
        profile,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


async def upsert_profile(user_sub: str, profile: dict) -> bool:
    """프로필을 Redis + Postgres에 동기화한다.

    Args:
        user_sub: Gateway X-User-Sub (UUID).
        profile: 레거시 Frappe 또는 REST v1 ``MyContext`` 응답. 저장 전에
            Agent 내부 평면 프로필로 정규화한다.

    Returns:
        DB가 실제로 갱신되었는지 여부. False는 "동일 profile_hash → DB write 생략,
        Redis만 갱신"을 의미한다.
    """
    profile = normalize_user_profile(profile)
    profile_hash = compute_profile_hash(profile)
    existing_hash = await find_profile_hash(user_sub)
    updated = existing_hash != profile_hash

    if updated:
        await upsert_profile_row(
            user_sub=user_sub,
            email=profile.get("user"),
            is_guest=bool(profile.get("is_guest")),
            profile=profile,
            profile_hash=profile_hash,
        )

    # Redis는 변경 여부와 무관하게 항상 갱신 — TTL 슬라이딩 효과 + 핫 캐시 보장.
    await redis_store.set(
        _cache_key(user_sub),
        profile,
        expire=REDIS_PROFILE_TTL_SECONDS,
    )

    return updated


async def get_profile(user_sub: Optional[str]) -> Optional[dict]:
    """사용자 프로필을 조회한다 (Redis 우선, DB fallback + backfill).

    Args:
        user_sub: Keycloak user sub. None/빈 문자열이면 즉시 None.

    Returns:
        Frappe profile JSON. 미등록이면 None.
    """
    if not user_sub:
        return None

    cached = await redis_store.get(_cache_key(user_sub))
    if isinstance(cached, dict):
        return cached

    row = await find_profile_row(user_sub)
    if row is None:
        return None

    # DB hit → Redis backfill (TTL 슬라이딩)
    await redis_store.set(
        _cache_key(user_sub),
        row.profile,
        expire=REDIS_PROFILE_TTL_SECONDS,
    )
    return row.profile


async def invalidate(user_sub: str) -> bool:
    """Redis 프로필 캐시를 즉시 삭제한다. DB 행은 유지된다.

    Args:
        user_sub: 사용자 Keycloak sub.

    Returns:
        Redis 삭제 성공 여부.
    """
    if not user_sub:
        return False
    return await redis_store.delete(_cache_key(user_sub))
