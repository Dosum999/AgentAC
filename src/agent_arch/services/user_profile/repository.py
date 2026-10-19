"""``agent.users`` 테이블 CRUD (asyncpg).

저장 모델은 단일 JSONB 컬럼(``profile``) — Frappe ``get_current_user`` 응답을
가공 없이 통째로 보관한다. 변경 감지는 별도 ``profile_hash`` 컬럼(sha256)에 의존하며,
서비스 레이어가 동일 hash인 경우 DB write 자체를 생략한다.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from agent_arch.db.connection import db_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UserProfileRow:
    """``agent.users`` 한 행.

    Attributes:
        user_sub: Keycloak user sub (PK).
        email: Frappe profile.user.
        is_guest: 비로그인 게스트 꼬리표.
        profile: Frappe ``get_current_user`` 응답 전체.
        profile_hash: 해당 ``profile``의 sha256 canonical hash.
        created_at: 최초 저장 시각.
        updated_at: 마지막 갱신 시각 (트리거로 자동 갱신).
    """

    user_sub: str
    email: Optional[str]
    is_guest: bool
    profile: dict
    profile_hash: str
    created_at: datetime
    updated_at: datetime


async def find_profile_row(user_sub: str) -> Optional[UserProfileRow]:
    """사용자 프로필 행을 조회한다. 미등록·DB 미연결이면 None."""
    pool = db_pool.get_pool()
    if pool is None:
        return None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT user_sub, email, is_guest, profile, profile_hash,
                   created_at, updated_at
              FROM agent.users
             WHERE user_sub = $1
            """,
            user_sub,
        )
    return _to_profile_row(row) if row is not None else None


def _to_profile_row(row) -> UserProfileRow:
    """asyncpg Record → ``UserProfileRow``.

    asyncpg는 JSONB를 dict가 아닌 str로 반환할 수 있어 안전하게 정규화한다.
    """
    profile = row["profile"]
    if isinstance(profile, str):
        profile = json.loads(profile)

    return UserProfileRow(
        user_sub=row["user_sub"],
        email=row["email"],
        is_guest=row["is_guest"],
        profile=profile,
        profile_hash=row["profile_hash"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def find_profile_row_by_email(email: str) -> Optional[UserProfileRow]:
    """이메일로 사용자 프로필 행을 역조회한다 (게스트 제외).

    Gateway 헤더 없이 외부 채널(Mattermost 봇 등)에서 들어온 요청의 신원을
    이메일로 복원하기 위한 경로다. 정상 요청 경로는 항상 user_sub를 쓴다.

    같은 이메일이 여러 sub에 붙어 있을 수 있어(계정 재발급 등) 가장 최근에
    갱신된 행 하나만 돌려준다.

    Args:
        email: Frappe ``profile.user`` 이메일.

    Returns:
        프로필 행. 미등록·게스트 전용·DB 미연결이면 None.
    """
    pool = db_pool.get_pool()
    if pool is None or not email:
        return None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT user_sub, email, is_guest, profile, profile_hash,
                   created_at, updated_at
              FROM agent.users
             WHERE email = $1 AND is_guest = FALSE
             ORDER BY updated_at DESC
             LIMIT 1
            """,
            email,
        )
    return _to_profile_row(row) if row is not None else None


async def find_profile_hash(user_sub: str) -> Optional[str]:
    """현재 저장된 ``profile_hash``만 조회한다 (변경 감지 핫패스용)."""
    pool = db_pool.get_pool()
    if pool is None:
        return None

    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT profile_hash FROM agent.users WHERE user_sub = $1",
            user_sub,
        )


async def upsert_profile_row(
    *,
    user_sub: str,
    email: Optional[str],
    is_guest: bool,
    profile: dict,
    profile_hash: str,
) -> Optional[UserProfileRow]:
    """프로필을 INSERT 또는 UPDATE 한다.

    호출자(서비스 레이어)가 hash 비교 후 변경이 있을 때만 호출하는 전제다.
    동일 hash라도 호출되면 같은 값으로 덮어쓰며 ``updated_at``만 갱신된다.

    Returns:
        저장된 행. DB 미연결이면 None.
    """
    pool = db_pool.get_pool()
    if pool is None:
        logger.warning("[user_profile] DB 미연결 — upsert 스킵 user_sub=%s", user_sub)
        return None

    profile_json = json.dumps(profile, ensure_ascii=False)

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO agent.users
                (user_sub, email, is_guest, profile, profile_hash)
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (user_sub) DO UPDATE SET
                email        = EXCLUDED.email,
                is_guest     = EXCLUDED.is_guest,
                profile      = EXCLUDED.profile,
                profile_hash = EXCLUDED.profile_hash
            RETURNING user_sub, email, is_guest, profile, profile_hash,
                      created_at, updated_at
            """,
            user_sub, email, is_guest, profile_json, profile_hash,
        )

    stored_profile = row["profile"]
    if isinstance(stored_profile, str):
        stored_profile = json.loads(stored_profile)

    return UserProfileRow(
        user_sub=row["user_sub"],
        email=row["email"],
        is_guest=row["is_guest"],
        profile=stored_profile,
        profile_hash=row["profile_hash"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def delete_profile_row(user_sub: str) -> bool:
    """프로필 행을 삭제한다 (관리·테스트용)."""
    pool = db_pool.get_pool()
    if pool is None:
        return False
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM agent.users WHERE user_sub = $1",
            user_sub,
        )
    return result.endswith(" 1")
