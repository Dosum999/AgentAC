"""대화방(agent.conversations) CRUD 레포지토리.

한 사용자(user_sub)는 N개의 대화방을 가지며, 각 대화방은 N개의 agent.messages를 가진다.

소유자 격리: 모든 함수는 SQL WHERE 절에 user_sub 매칭을 강제한다.
라우터가 검증을 누락해도 데이터 누수가 일어나지 않는 SQL-레벨 안전망.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from agent_arch.db.connection import db_pool

logger = logging.getLogger(__name__)


def derive_conversation_title(text: Optional[str], max_len: int = 40) -> Optional[str]:
    """첫 사용자 질의에서 대화방 제목을 파생한다(공백 정규화 + 길이 제한).

    lazy-create 시점에 호출해 목록에 의미 있는 제목이 뜨게 한다. 개행/연속 공백은
    한 칸으로 접고 ``max_len``을 넘으면 말줄임(…)한다. 빈 입력이면 None을 반환해
    호출자/UI가 created_at 폴백을 쓰게 한다.

    Args:
        text: 사용자 입력 텍스트(첫 질의).
        max_len: 제목 최대 길이(기본 40자).

    Returns:
        정규화된 제목 문자열. 입력이 비면 None.
    """
    if not text:
        return None
    normalized = " ".join(text.split())
    if not normalized:
        return None
    if len(normalized) <= max_len:
        return normalized
    return normalized[:max_len].rstrip() + "…"


# 사용자 대화 목록·검색에서 숨기는 방 — 예약 실행 결과를 담는 방(app/services/schedules). 결과는 헤더 알림으로
# 받고, 방은 사용량 집계(agent.messages)와 감사용으로만 남긴다. 단건 조회(get_conversation)는 그대로 된다.
HIDDEN_KIND_SQL = "COALESCE({alias}metadata->>'kind', '') <> 'schedule'"


async def create_conversation(
    user_sub: str,
    title: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[str]:
    """신규 대화방을 생성하고 UUID 문자열을 반환한다.

    title은 호출자가 넘긴 값을 그대로 저장한다. lazy-create 경로(save_turn / async
    accept)는 ``derive_conversation_title(첫 질의)``로 채워 목록에 의미 있는 제목이
    뜨게 한다. None이면 UI가 created_at으로 폴백 표시한다.

    Args:
        user_sub: 소유자 sub.
        title: 대화방 제목. 보통 None. 명시 입력 경로(v2)에서만 채워짐.
        metadata: 방 메타. ``{"kind": "schedule"}`` 이면 목록·검색에서 숨는다.

    Returns:
        생성된 대화방 UUID 문자열. DB 비활성화 시 None.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None

    row = await pool.fetchrow(
        """
        INSERT INTO agent.conversations (user_sub, title, metadata)
        VALUES ($1, $2, $3::jsonb)
        RETURNING id
        """,
        user_sub,
        title,
        json.dumps(metadata, ensure_ascii=False) if metadata else None,
    )
    return str(row["id"]) if row else None


async def list_conversations(
    user_sub: str,
    limit: int,
    cursor: Optional[tuple[datetime, str]] = None,
    include_archived: bool = False,
) -> list[dict]:
    """본인 대화방 목록을 last_message_at 역순으로 조회한다.

    단순 구현 — message_count·rated_count 같은 집계 컬럼은 포함하지 않는다.
    집계가 필요한 경로는 list_conversations_for_user 의 LEFT JOIN 버전을 사용한다.

    Args:
        user_sub: 소유자 sub.
        limit: 페이지 크기. 호출자가 has_more 판단을 위해 limit+1을 넘길 수 있음.
        cursor: (last_message_at, id) 튜플. 이 값보다 과거 페이지를 반환.
        include_archived: True면 archived 방도 포함.

    Returns:
        dict 리스트. asyncpg row를 dict로 변환.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    clauses = ["user_sub = $1", HIDDEN_KIND_SQL.format(alias="")]
    params: list[Any] = [user_sub]

    if not include_archived:
        clauses.append("archived_at IS NULL")

    if cursor is not None:
        params.append(cursor[0])
        params.append(cursor[1])
        clauses.append(
            f"(COALESCE(last_message_at, created_at), id) "
            f"< (${len(params) - 1}::timestamp, ${len(params)}::uuid)"
        )

    params.append(limit)
    where = " AND ".join(clauses)
    sql = f"""
        SELECT id, user_sub, title, created_at, updated_at,
               last_message_at, archived_at, metadata
          FROM agent.conversations
         WHERE {where}
         ORDER BY COALESCE(last_message_at, created_at) DESC NULLS LAST, id DESC
         LIMIT ${len(params)}
    """
    rows = await pool.fetch(sql, *params)
    return [dict(r) for r in rows]


async def get_conversation(conversation_id: str, user_sub: str) -> Optional[dict]:
    """본인 소유 대화방 1건을 조회한다.

    타인 소유 또는 존재하지 않으면 None — 호출자는 404로 변환.

    Args:
        conversation_id: 대화방 UUID 문자열.
        user_sub: 소유자 sub.

    Returns:
        대화방 dict. 없으면 None.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None

    row = await pool.fetchrow(
        """
        SELECT id, user_sub, title, created_at, updated_at,
               last_message_at, archived_at, metadata
          FROM agent.conversations
         WHERE id = $1::uuid AND user_sub = $2
        """,
        conversation_id,
        user_sub,
    )
    return dict(row) if row else None


async def update_conversation(
    conversation_id: str,
    user_sub: str,
    title: Optional[str] = None,
    archived: Optional[bool] = None,
) -> bool:
    """대화방의 title 또는 archived 상태를 갱신한다 (PATCH 의미).

    title=None, archived=None이면 no-op.
    archived=True면 archived_at=NOW(), False면 NULL로.

    Args:
        conversation_id: 대상 대화방 UUID.
        user_sub: 소유자 sub (격리).
        title: 새 제목. None이면 변경하지 않음.
        archived: True/False면 archived_at 토글, None이면 변경하지 않음.

    Returns:
        업데이트된 행이 있으면 True.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return False

    sets: list[str] = []
    params: list[Any] = []

    if title is not None:
        params.append(title)
        sets.append(f"title = ${len(params)}")

    if archived is True:
        sets.append("archived_at = NOW()")
    elif archived is False:
        sets.append("archived_at = NULL")

    if not sets:
        return False

    params.append(conversation_id)
    params.append(user_sub)
    sql = f"""
        UPDATE agent.conversations
           SET {", ".join(sets)}
         WHERE id = ${len(params) - 1}::uuid
           AND user_sub = ${len(params)}
    """
    result = await pool.execute(sql, *params)
    return result.endswith(" 1")


async def list_conversations_for_user(
    user_sub: str,
    limit: int,
    cursor: Optional[tuple[datetime, str]] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    order: str = "desc",
) -> list[dict]:
    """본인 대화방 목록을 집계 컬럼과 함께 페이지네이션 조회한다.

    단순 목록만 반환하는 ``list_conversations`` 위에 ``message_count`` /
    ``rated_count`` LEFT JOIN 집계를 더한 버전. 라우터 GET /me/conversation 진입점이다.

    Args:
        user_sub: 소유자 sub.
        limit: 페이지 크기. 본 함수가 ``LIMIT limit+1``로 SELECT — 호출자가
            ``has_more``를 판단.
        cursor: ``(sort_key, id)`` 튜플. ``sort_key``는
            ``COALESCE(last_message_at, created_at)``.
        since: 시작 시각 — ``COALESCE(last_message_at, created_at)`` 기준.
        until: 종료 시각 — 동일 기준.
        order: ``"desc"``(최신순 기본) | ``"asc"``(과거순). cursor 비교 연산자가
            정렬 방향에 맞춰 반전된다 (``list_messages_in_conversation``과 같은 규약).

    Returns:
        dict 리스트 (최대 ``limit+1`` 개). 컬럼:
        id, title, created_at, last_message_at, message_count, rated_count,
        positive_count, negative_count,
        first_query(첫 질의 — title 이 NULL 일 때 라우터가 제목 폴백으로 사용).
        archived 방은 항상 제외 (v1 정책).

        ``rated_count == positive_count + negative_count`` 가 항상 성립한다 —
        세 집계 모두 ``rated_sql`` 하나에서 판정 기준을 가져오기 때문이다.

    Note:
        성능: 사용자당 대화방이 매우 많아지면(예: 새로고침 모델 + 수일 사용 → 수백 방)
        GROUP BY 비용이 늘 수 있다. FILTER 집계가 2개에서 4개로 늘었으니 실측 후
        회귀하면 ``agent.conversations``에 ``message_count``/``rated_count``/
        ``negative_count`` 캐시 컬럼 추가 후보 (v1.1).
    """
    from agent_arch.utils.rating import _POSITIVE_PRESETS, rated_sql

    pool = db_pool.get_pool()
    if pool is None:
        return []

    # 평가 판정은 classify_rating 과 같은 기준(빈 문자열 = 미평가)을 쓴다.
    rated = rated_sql("m.rating")
    clauses = ["c.user_sub = $1", "c.archived_at IS NULL", HIDDEN_KIND_SQL.format(alias="c.")]
    params: list[Any] = [user_sub]

    if cursor is not None:
        params.append(cursor[0])
        params.append(cursor[1])
        op = "<" if order == "desc" else ">"
        clauses.append(
            f"(COALESCE(c.last_message_at, c.created_at), c.id) "
            f"{op} (${len(params) - 1}::timestamp, ${len(params)}::uuid)"
        )
    if since is not None:
        params.append(since)
        clauses.append(f"COALESCE(c.last_message_at, c.created_at) >= ${len(params)}")
    if until is not None:
        params.append(until)
        clauses.append(f"COALESCE(c.last_message_at, c.created_at) <= ${len(params)}")

    # positives 배열은 **LIMIT 파라미터 append 앞**에 넣는다 — 그래야 위 cursor/since/until
    # 블록이 계산해 둔 ``$n`` 번호가 밀리지 않고, ``LIMIT ${len(params)}`` 도 그대로
    # 마지막을 가리킨다. asyncpg 는 placeholder 가 SELECT 절에 먼저 등장하는 것과
    # append 순서가 다른 걸 신경 쓰지 않는다.
    params.append(list(_POSITIVE_PRESETS))
    pos_pos = len(params)
    params.append(limit + 1)
    where = " AND ".join(clauses)
    # 정렬 키는 "마지막 활동 시각, 없으면 생성 시각". COALESCE 가 NULL 을 없애므로
    # 빈 방도 자기 생성 시각으로 다른 방들 사이에 섞인다(최신순이면 갓 만든 빈 방이 위로
    # 온다 — 기존 동작이고 정렬 방향과 무관하다). ``NULLS LAST`` 는 COALESCE 가 걷힐 때를
    # 대비한 방어일 뿐 실제로는 발동하지 않는다.
    sort_key = "COALESCE(c.last_message_at, c.created_at)"
    order_sql = (
        f"{sort_key} DESC NULLS LAST, c.id DESC"
        if order == "desc"
        else f"{sort_key} ASC NULLS LAST, c.id ASC"
    )
    sql = f"""
        SELECT c.id, c.title, c.created_at, c.last_message_at,
               COUNT(m.id)::int                                       AS message_count,
               COUNT(m.id) FILTER (WHERE {rated})::int                AS rated_count,
               COUNT(m.id) FILTER (
                   WHERE m.rating = ANY(${pos_pos}::text[])
               )::int                                                 AS positive_count,
               COUNT(m.id) FILTER (
                   WHERE {rated} AND m.rating <> ALL(${pos_pos}::text[])
               )::int                                                 AS negative_count,
               (SELECT fm.query
                  FROM agent.messages fm
                 WHERE fm.conversation_id = c.id
                   AND fm.query IS NOT NULL AND fm.query <> ''
                   AND (fm.metadata->>'pos_internal') IS DISTINCT FROM 'true'
                 ORDER BY fm.created_at ASC, fm.id ASC
                 LIMIT 1)                                             AS first_query
          FROM agent.conversations c
          -- POS 플래닝 턴은 집계(message_count/rated_count)에서 제외 — 메시지 목록 API 의
          -- 숨김 필터와 기준을 맞춰 "화면엔 1쌍인데 2건" 불일치를 막는다. ON 절에 두는
          -- 이유: LEFT JOIN 의미론상 조인 조건이 정위치(WHERE 면 빈 방 행이 사라짐).
          LEFT JOIN agent.messages m ON m.conversation_id = c.id
               AND (m.metadata->>'pos_internal') IS DISTINCT FROM 'true'
         WHERE {where}
         GROUP BY c.id
         ORDER BY {order_sql}
         LIMIT ${len(params)}
    """
    rows = await pool.fetch(sql, *params)
    return [dict(r) for r in rows]


async def touch_conversation(conversation_id: str) -> None:
    """save_turn 직후 호출하여 last_message_at을 NOW()로 갱신한다.

    updated_at은 트리거가 자동 갱신. 본 함수는 소유자 검증을 하지 않는다 —
    save_turn 호출 직후 같은 트랜잭션 흐름이라 conversation_id의 출처가 신뢰됨.

    Args:
        conversation_id: 대상 대화방 UUID.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return

    await pool.execute(
        "UPDATE agent.conversations SET last_message_at = NOW() WHERE id = $1::uuid",
        conversation_id,
    )
