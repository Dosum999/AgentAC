"""``agent.user_preferences`` 테이블 CRUD.

merge-patch 시맨틱: ``upsert``에 None을 넘기면 해당 컬럼은 기존 값을 유지한다
(``COALESCE($N, table.col)``). 새 행이면 컬럼 DEFAULT(20, 86400)가 적용된다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from agent_arch.db.connection import db_pool

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class UserPreferences:
    """``agent.user_preferences`` 한 행.

    Attributes:
        user_sub: Keycloak user sub (PK).
        max_history: 대화 이력 최대 보존 메시지 수 (짝수).
        session_ttl_seconds: Redis session 키 TTL (초).
        created_at: 최초 저장 시각.
        updated_at: 마지막 갱신 시각 (트리거로 자동 갱신).
    """

    user_sub: str
    max_history: int
    session_ttl_seconds: int
    created_at: datetime
    updated_at: datetime


async def find_preferences(user_sub: str) -> Optional[UserPreferences]:
    """사용자 개인설정을 조회한다. 미등록이거나 DB 미연결이면 None."""
    pool = db_pool.get_pool()
    if pool is None:
        return None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT user_sub, max_history, session_ttl_seconds,
                   created_at, updated_at
              FROM agent.user_preferences
             WHERE user_sub = $1
            """,
            user_sub,
        )
    if row is None:
        return None
    return UserPreferences(
        user_sub=row["user_sub"],
        max_history=row["max_history"],
        session_ttl_seconds=row["session_ttl_seconds"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def upsert_preferences(
    *,
    user_sub: str,
    max_history: Optional[int],
    session_ttl_seconds: Optional[int],
) -> Optional[UserPreferences]:
    """개인설정을 INSERT 또는 UPDATE 한다 (merge-patch).

    None으로 들어온 컬럼은 UPDATE 시 기존 값 유지. 새 행이면 컬럼 DEFAULT가 적용된다.

    Args:
        user_sub: Keycloak user sub.
        max_history: 갱신할 값. None이면 기존 값 유지(신규 행은 DEFAULT 20).
        session_ttl_seconds: 갱신할 값. None이면 기존 값 유지(신규 행은 DEFAULT 86400).

    Returns:
        저장된 행. DB 미연결이면 None.
    """
    pool = db_pool.get_pool()
    if pool is None:
        logger.warning(
            "[preferences] DB 미연결 상태 — upsert 스킵 user_sub=%s", user_sub
        )
        return None

    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO agent.user_preferences
                (user_sub, max_history, session_ttl_seconds)
            VALUES (
                $1,
                COALESCE($2, 20),
                COALESCE($3, 86400)
            )
            ON CONFLICT (user_sub) DO UPDATE SET
                max_history         = COALESCE($2, agent.user_preferences.max_history),
                session_ttl_seconds = COALESCE($3, agent.user_preferences.session_ttl_seconds)
            RETURNING user_sub, max_history, session_ttl_seconds,
                      created_at, updated_at
            """,
            user_sub, max_history, session_ttl_seconds,
        )
    return UserPreferences(
        user_sub=row["user_sub"],
        max_history=row["max_history"],
        session_ttl_seconds=row["session_ttl_seconds"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


async def delete_preferences(user_sub: str) -> bool:
    """개인설정 행을 삭제한다. 다음 GET부터 전역 기본값으로 복귀한다.

    Returns:
        실제 삭제된 행이 있으면 True. 미연결·미존재면 False.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return False
    async with pool.acquire() as conn:
        result = await conn.execute(
            "DELETE FROM agent.user_preferences WHERE user_sub = $1",
            user_sub,
        )
    # asyncpg execute는 'DELETE N' 형태 반환
    return result.endswith(" 1")
