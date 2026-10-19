"""장기 사용자 메모리 record CRUD (agent.memory_records).

Phase 1 의 memory-records 층. PG=구조화 SoT, 벡터는 Qdrant(이전 설계).

소유자 격리: 조회/수정 함수는 SQL WHERE 절에 user_sub 매칭을 강제한다(회사 스코프와
함께 멀티테넌트 누수를 SQL 레벨에서 차단). 인덱싱 잡(list_unindexed/mark_indexed)만
시스템 전역이라 user_sub를 받지 않는다.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from agent_arch.db.connection import db_pool
from agent_arch.models.memory import MemoryRecordDraft

logger = logging.getLogger(__name__)

# 읽기 시 dict로 정규화할 컬럼 집합 — SELECT 는 항상 이 컬럼을 반환한다.
_SELECT_COLS = """
    id, user_sub, company_scope, domain_scope, record_type, content, payload,
    source_kind, support_count, confidence, source_message_ids,
    source_conversation_ids, status, superseded_by, embedding_indexed,
    created_at, updated_at, last_seen_at, expires_at
"""


def _row_to_dict(row: Any) -> dict:
    """asyncpg Row를 정규화한 dict로 변환한다.

    jsonb(payload)는 asyncpg 기본 코덱상 문자열로 오므로 json.loads로 dict화한다.
    UUID / UUID[] / Decimal은 각각 str / list[str] / float로 정규화해 상위 계층이
    직렬화·비교를 단순하게 하도록 한다.
    """
    d = dict(row)
    payload = d.get("payload")
    if isinstance(payload, str):
        try:
            d["payload"] = json.loads(payload)
        except (ValueError, TypeError):
            d["payload"] = {}
    elif payload is None:
        d["payload"] = {}
    for k in ("id", "superseded_by"):
        if d.get(k) is not None:
            d[k] = str(d[k])
    for k in ("source_message_ids", "source_conversation_ids"):
        d[k] = [str(x) for x in (d.get(k) or [])]
    if d.get("confidence") is not None:
        d["confidence"] = float(d["confidence"])
    return d


async def insert_memory_record(user_sub: str, draft: MemoryRecordDraft) -> Optional[str]:
    """draft를 새 active record로 영속하고 UUID를 반환한다.

    sor_derivable은 INSERT에서 생략한다(DB DEFAULT FALSE) — SoR 파생 후보는 파이프라인이
    이 함수 호출 전에 걸러야 하며, DB CHECK가 최종 트립와이어로 남는다.

    Args:
        user_sub: 소유자 sub.
        draft: 영속할 후보. confidence 미지정 시 정책값(resolved_confidence)을 쓴다.

    Returns:
        생성된 record UUID 문자열. DB 비활성화 시 None.

    Raises:
        ValueError: draft.sor_derivable=True (저장 금지선 위반, 방어적 차단).
    """
    if draft.sor_derivable:
        raise ValueError("sor_derivable=True인 후보는 저장할 수 없습니다(SoR는 라이브 조회)")

    pool = db_pool.get_pool()
    if pool is None:
        return None

    row = await pool.fetchrow(
        """
        INSERT INTO agent.memory_records
            (user_sub, company_scope, domain_scope, record_type, content, payload,
             source_kind, support_count, confidence,
             source_message_ids, source_conversation_ids)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10::uuid[], $11::uuid[])
        RETURNING id
        """,
        user_sub,
        draft.company_scope,
        draft.domain_scope,
        draft.record_type,
        draft.content,
        json.dumps(draft.payload, ensure_ascii=False),
        draft.source_kind,
        draft.support_count,
        draft.resolved_confidence(),
        draft.source_message_ids,
        draft.source_conversation_ids,
    )
    return str(row["id"]) if row else None


async def get_memory_record(record_id: str, user_sub: str) -> Optional[dict]:
    """본인 소유 record 1건을 조회한다(상태 무관). 없으면 None."""
    pool = db_pool.get_pool()
    if pool is None:
        return None
    row = await pool.fetchrow(
        f"SELECT {_SELECT_COLS} FROM agent.memory_records "
        f"WHERE id = $1::uuid AND user_sub = $2",
        record_id,
        user_sub,
    )
    return _row_to_dict(row) if row else None


async def list_active_records(
    user_sub: str,
    *,
    record_type: Optional[str] = None,
    company_scope: Optional[str] = None,
    domain_scope: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """활성·미만료 record를 confidence 내림차순으로 조회한다.

    company_scope/domain_scope는 "전역(NULL) + 해당 스코프" 를 함께 포함한다
    (전역 record는 어떤 회사/도메인 컨텍스트에서도 유효하므로).

    Args:
        user_sub: 소유자.
        record_type: 지정 시 해당 타입만.
        company_scope: 지정 시 (NULL 이거나 == 값) 인 record.
        domain_scope: 지정 시 (NULL 이거나 == 값) 인 record.
        limit: 최대 건수.

    Returns:
        정규화 dict 리스트.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    clauses = [
        "user_sub = $1",
        "status = 'active'",
        "deleted_at IS NULL",
        "(expires_at IS NULL OR expires_at > NOW())",
    ]
    params: list[Any] = [user_sub]

    if record_type is not None:
        params.append(record_type)
        clauses.append(f"record_type = ${len(params)}")
    if company_scope is not None:
        params.append(company_scope)
        clauses.append(f"(company_scope IS NULL OR company_scope = ${len(params)})")
    if domain_scope is not None:
        params.append(domain_scope)
        clauses.append(f"(domain_scope IS NULL OR domain_scope = ${len(params)})")

    params.append(limit)
    sql = (
        f"SELECT {_SELECT_COLS} FROM agent.memory_records "
        f"WHERE {' AND '.join(clauses)} "
        f"ORDER BY confidence DESC, last_seen_at DESC LIMIT ${len(params)}"
    )
    rows = await pool.fetch(sql, *params)
    return [_row_to_dict(r) for r in rows]


async def find_alias_by_canonical(
    user_sub: str, canonical_kind: str, canonical_id: str
) -> Optional[dict]:
    """동일 canonical 대상의 활성 entity_alias record를 찾는다(reconcile 중복 판정).

    payload @> containment 로 조회(GIN idx_memory_records_alias 가속).
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None
    row = await pool.fetchrow(
        f"""
        SELECT {_SELECT_COLS} FROM agent.memory_records
         WHERE user_sub = $1 AND record_type = 'entity_alias'
           AND status = 'active' AND deleted_at IS NULL
           AND payload @> jsonb_build_object('canonical_kind', $2::text,
                                             'canonical_id', $3::text)
         LIMIT 1
        """,
        user_sub,
        canonical_kind,
        canonical_id,
    )
    return _row_to_dict(row) if row else None


async def find_alias_by_name(user_sub: str, canonical_name: str) -> Optional[dict]:
    """canonical_name이 같은 활성 entity_alias record를 찾는다(canonical_id 없을 때 fallback).

    도구가 대상을 id로 해석하지 못한 경우(canonical_id 부재) 중복 alias 누적을 막기 위한
    이름 기반 dedup. payload @> containment 로 조회(GIN idx_memory_records_alias 가속).
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None
    row = await pool.fetchrow(
        f"""
        SELECT {_SELECT_COLS} FROM agent.memory_records
         WHERE user_sub = $1 AND record_type = 'entity_alias'
           AND status = 'active' AND deleted_at IS NULL
           AND payload @> jsonb_build_object('canonical_name', $2::text)
         ORDER BY confidence DESC LIMIT 1
        """,
        user_sub,
        canonical_name,
    )
    return _row_to_dict(row) if row else None


async def lookup_alias(user_sub: str, alias: str) -> Optional[dict]:
    """별칭 문자열로 활성 entity_alias record를 해석한다(alias→canonical).

    Phase 2 메모리 도구 및 도구 인자 자동충족에서 "scrum"→회사 scrum 채널 해석에 쓴다.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None
    row = await pool.fetchrow(
        f"""
        SELECT {_SELECT_COLS} FROM agent.memory_records
         WHERE user_sub = $1 AND record_type = 'entity_alias'
           AND status = 'active' AND deleted_at IS NULL
           AND payload @> jsonb_build_object('aliases', jsonb_build_array($2::text))
         ORDER BY confidence DESC LIMIT 1
        """,
        user_sub,
        alias,
    )
    return _row_to_dict(row) if row else None


async def search_memory_keyword(
    user_sub: str,
    query: str,
    *,
    record_type: Optional[str] = None,
    company_scope: Optional[str] = None,
    domain_scope: Optional[str] = None,
    limit: int = 5,
) -> list[dict]:
    """content pg_trgm word_similarity로 record를 검색한다(하이브리드의 키워드 half).

    벡터 half는 Qdrant(app/services/memory/indexer.py)가 담당하며, 상위 계층에서
    RRF로 융합한다. `<%`(word_similarity)는 짧은 쿼리가 긴 content 안의 "단어"에
    매칭되는지를 보므로, `%`(전체 유사도)보다 record 검색에 적합하다. GIN trgm 인덱스 사용.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    clauses = [
        "user_sub = $1",
        "status = 'active'",
        "deleted_at IS NULL",
        "(expires_at IS NULL OR expires_at > NOW())",
        "$2 <% content",
    ]
    params: list[Any] = [user_sub, query]

    if record_type is not None:
        params.append(record_type)
        clauses.append(f"record_type = ${len(params)}")
    if company_scope is not None:
        params.append(company_scope)
        clauses.append(f"(company_scope IS NULL OR company_scope = ${len(params)})")
    if domain_scope is not None:
        params.append(domain_scope)
        clauses.append(f"(domain_scope IS NULL OR domain_scope = ${len(params)})")

    params.append(limit)
    sql = (
        f"SELECT {_SELECT_COLS}, word_similarity($2, content) AS score "
        f"FROM agent.memory_records WHERE {' AND '.join(clauses)} "
        f"ORDER BY score DESC, confidence DESC LIMIT ${len(params)}"
    )
    rows = await pool.fetch(sql, *params)
    return [_row_to_dict(r) for r in rows]


async def reinforce_record(
    record_id: str,
    user_sub: str,
    *,
    support_count: int,
    confidence: float,
    add_message_ids: Optional[list[str]] = None,
    payload: Optional[dict] = None,
) -> bool:
    """재관측된 record를 강화한다(support_count·confidence·provenance 갱신).

    source_message_ids는 기존값과 add_message_ids의 합집합(중복 제거)으로 갱신한다.
    payload가 주어지면 통째로 교체한다(예: entity_alias 별칭 목록 확장 후 전달).
    last_seen_at은 NOW()로 갱신하고 embedding_indexed=FALSE로 되돌려 재인덱싱을 유도한다.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return False

    payload_json = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    result = await pool.execute(
        """
        UPDATE agent.memory_records
           SET support_count = $3,
               confidence = $4,
               last_seen_at = NOW(),
               embedding_indexed = FALSE,
               source_message_ids = (
                   SELECT COALESCE(array_agg(DISTINCT e), '{}')
                     FROM unnest(source_message_ids || $5::uuid[]) AS e
               ),
               payload = COALESCE($6::jsonb, payload)
         WHERE id = $1::uuid AND user_sub = $2 AND status = 'active'
        """,
        record_id,
        user_sub,
        support_count,
        confidence,
        add_message_ids or [],
        payload_json,
    )
    return result.endswith(" 1")


async def supersede_record(old_id: str, user_sub: str, new_id: str) -> bool:
    """old record를 new record로 대체 표시한다(reconcile update/merge 감사추적)."""
    pool = db_pool.get_pool()
    if pool is None:
        return False
    result = await pool.execute(
        """
        UPDATE agent.memory_records
           SET status = 'superseded', superseded_by = $3::uuid
         WHERE id = $1::uuid AND user_sub = $2 AND status = 'active'
        """,
        old_id,
        user_sub,
        new_id,
    )
    return result.endswith(" 1")


async def soft_delete_user_memory(user_sub: str) -> int:
    """사용자의 모든 활성/대체 record를 soft delete한다(삭제권 행사).

    clear_user_conversations(Redis/PG 대화 삭제)와 짝을 이룬다. Qdrant 포인트 정리는
    호출부(indexer)가 담당한다.

    Returns:
        삭제 표시된 행 수.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return 0
    result = await pool.execute(
        """
        UPDATE agent.memory_records
           SET status = 'deleted', deleted_at = NOW()
         WHERE user_sub = $1 AND status <> 'deleted'
        """,
        user_sub,
    )
    # "UPDATE N" → N 파싱
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


async def soft_delete_record(record_id: str, user_sub: str) -> bool:
    """본인 소유 record 1건을 soft delete한다(잊기/정정 — 특정 기억만 제거).

    소유자 격리를 SQL WHERE 로 강제하고, 이미 삭제된 것은 건드리지 않는다(멱등).
    Qdrant 포인트 정리는 호출부(service.forget_memory)가 담당한다.

    Args:
        record_id: 삭제할 record UUID.
        user_sub: 소유자 sub(격리).

    Returns:
        실제로 삭제 표시된 경우 True, 없거나 이미 삭제됐으면 False.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return False
    result = await pool.execute(
        """
        UPDATE agent.memory_records
           SET status = 'deleted', deleted_at = NOW()
         WHERE id = $1::uuid AND user_sub = $2 AND status <> 'deleted'
        """,
        record_id,
        user_sub,
    )
    try:
        return int(result.split()[-1]) > 0
    except (ValueError, IndexError):
        return False


# ─── 상위 요약층 (agent.memory_summaries — user_profile / topic_track) ────────

_SUMMARY_COLS = """
    id, user_sub, company_scope, kind, title, content, evidence_record_ids,
    source_record_count, status, created_at, updated_at, last_built_at
"""


def _summary_row_to_dict(row: Any) -> dict:
    d = dict(row)
    if d.get("id") is not None:
        d["id"] = str(d["id"])
    d["evidence_record_ids"] = [str(x) for x in (d.get("evidence_record_ids") or [])]
    return d


async def active_record_count(user_sub: str, company_scope: Optional[str] = None) -> int:
    """활성·미만료 record 수(요약 재빌드 트리거 기준)."""
    pool = db_pool.get_pool()
    if pool is None:
        return 0
    if company_scope is None:
        return int(await pool.fetchval(
            "SELECT count(*) FROM agent.memory_records "
            "WHERE user_sub=$1 AND status='active' AND deleted_at IS NULL "
            "AND (expires_at IS NULL OR expires_at > NOW())",
            user_sub,
        ) or 0)
    return int(await pool.fetchval(
        "SELECT count(*) FROM agent.memory_records "
        "WHERE user_sub=$1 AND status='active' AND deleted_at IS NULL "
        "AND (expires_at IS NULL OR expires_at > NOW()) "
        "AND (company_scope IS NULL OR company_scope=$2)",
        user_sub, company_scope,
    ) or 0)


async def get_active_summary(
    user_sub: str, kind: str, company_scope: Optional[str] = None
) -> Optional[dict]:
    """유저의 활성 요약 1건(user_profile) 또는 kind별 첫 건을 조회한다."""
    pool = db_pool.get_pool()
    if pool is None:
        return None
    row = await pool.fetchrow(
        f"""
        SELECT {_SUMMARY_COLS} FROM agent.memory_summaries
         WHERE user_sub=$1 AND kind=$2 AND status='active'
           AND ($3::text IS NULL OR company_scope IS NOT DISTINCT FROM $3)
         ORDER BY updated_at DESC LIMIT 1
        """,
        user_sub, kind, company_scope,
    )
    return _summary_row_to_dict(row) if row else None


async def list_summaries(
    user_sub: str, kind: str, company_scope: Optional[str] = None, limit: int = 20
) -> list[dict]:
    """유저의 활성 요약을 kind별로 나열한다(topic_track 다건)."""
    pool = db_pool.get_pool()
    if pool is None:
        return []
    rows = await pool.fetch(
        f"""
        SELECT {_SUMMARY_COLS} FROM agent.memory_summaries
         WHERE user_sub=$1 AND kind=$2 AND status='active'
           AND ($3::text IS NULL OR company_scope IS NOT DISTINCT FROM $3)
         ORDER BY updated_at DESC LIMIT $4
        """,
        user_sub, kind, company_scope, limit,
    )
    return [_summary_row_to_dict(r) for r in rows]


async def search_summaries_keyword(
    user_sub: str, query: str, kind: str, *, limit: int = 5
) -> list[dict]:
    """요약(topic_track 등) content 를 pg_trgm word_similarity 로 검색한다."""
    pool = db_pool.get_pool()
    if pool is None or not (query or "").strip():
        return []
    rows = await pool.fetch(
        f"""
        SELECT {_SUMMARY_COLS}, word_similarity($2, coalesce(title,'') || ' ' || content) AS score
          FROM agent.memory_summaries
         WHERE user_sub=$1 AND kind=$3 AND status='active'
           AND $2 <% (coalesce(title,'') || ' ' || content)
         ORDER BY score DESC LIMIT $4
        """,
        user_sub, query, kind, limit,
    )
    return [_summary_row_to_dict(r) for r in rows]


async def replace_topic_tracks(
    user_sub: str, tracks: list[dict], company_scope: Optional[str] = None
) -> int:
    """유저의 topic_track 전체를 새 집합으로 교체한다(기존 active→deleted 후 insert).

    MVP는 증분 merge 대신 재생성. tracks 원소: {title, content, evidence_record_ids}.

    Returns:
        새로 insert 된 트랙 수.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return 0
    # 기존 활성 트랙 → deleted (재생성 방식). 개별 실패해도 다음 트리거가 자가치유.
    await pool.execute(
        """
        UPDATE agent.memory_summaries SET status='deleted'
         WHERE user_sub=$1 AND kind='topic_track' AND status='active'
           AND company_scope IS NOT DISTINCT FROM $2
        """,
        user_sub, company_scope,
    )
    n = 0
    for t in tracks[:20]:
        title = (t.get("title") or "").strip()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        await pool.execute(
            """
            INSERT INTO agent.memory_summaries
                (user_sub, company_scope, kind, title, content, evidence_record_ids)
            VALUES ($1, $2, 'topic_track', $3, $4, $5::uuid[])
            """,
            user_sub, company_scope, title or None, content,
            [str(x) for x in (t.get("evidence_record_ids") or [])],
        )
        n += 1
    return n


async def upsert_user_profile(
    user_sub: str,
    content: str,
    *,
    evidence_record_ids: Optional[list[str]] = None,
    source_record_count: int = 0,
    company_scope: Optional[str] = None,
) -> Optional[str]:
    """활성 user_profile 을 갱신하거나(있으면) 신규 생성한다. id 반환."""
    pool = db_pool.get_pool()
    if pool is None:
        return None
    ev = evidence_record_ids or []
    # 기존 활성 프로필 UPDATE 시도 → 없으면 INSERT (partial-unique 회피).
    row = await pool.fetchrow(
        """
        UPDATE agent.memory_summaries
           SET content=$2, evidence_record_ids=$3::uuid[], source_record_count=$4,
               last_built_at=NOW()
         WHERE user_sub=$1 AND kind='user_profile' AND status='active'
           AND company_scope IS NOT DISTINCT FROM $5
         RETURNING id
        """,
        user_sub, content, ev, source_record_count, company_scope,
    )
    if row:
        return str(row["id"])
    row = await pool.fetchrow(
        """
        INSERT INTO agent.memory_summaries
            (user_sub, company_scope, kind, content, evidence_record_ids, source_record_count)
        VALUES ($1, $2, 'user_profile', $3, $4::uuid[], $5)
        RETURNING id
        """,
        user_sub, company_scope, content, ev, source_record_count,
    )
    return str(row["id"]) if row else None


# ─── 과거 대화 원문 검색 (recall_conversation 도구용) ─────────────────────────


async def search_user_conversations(
    user_sub: str, query: str, *, limit: int = 5
) -> list[dict]:
    """사용자의 과거 대화 원문(agent.messages)을 pg_trgm 으로 검색한다.

    record(정제된 사실)로는 못 담는 "실제로 오갔던 대화"를 회상하기 위한 것.
    query/answer 양쪽에 word_similarity(`<%`)로 매칭하고, 세션(대화방)을 가로질러
    본인(user_sub) 것만 조회한다. 소유자 격리는 SQL WHERE 로 강제.

    Args:
        user_sub: 소유자.
        query: 검색어(자연어).
        limit: 최대 건수.

    Returns:
        [{id, conversation_id, created_at, query, answer, score}] 리스트(유사도순).
    """
    pool = db_pool.get_pool()
    if pool is None or not user_sub or not (query or "").strip():
        return []
    rows = await pool.fetch(
        """
        SELECT id::text, conversation_id::text, created_at, query, answer,
               GREATEST(word_similarity($2, query),
                        word_similarity($2, COALESCE(answer, ''))) AS score
          FROM agent.messages
         WHERE user_sub = $1
           AND ($2 <% query OR $2 <% COALESCE(answer, ''))
           -- POS 플래닝 턴 제외 — 판단 턴과 query 가 동일해 회상 결과가 중복 히트하고
           -- answer="" 행이 섞인다. 이력 조회 API 의 숨김 필터와 같은 기준.
           AND (metadata->>'pos_internal') IS DISTINCT FROM 'true'
         ORDER BY score DESC, created_at DESC
         LIMIT $3
        """,
        user_sub,
        query,
        limit,
    )
    out: list[dict] = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"]) if d.get("id") else None
        d["conversation_id"] = str(d["conversation_id"]) if d.get("conversation_id") else None
        if d.get("score") is not None:
            d["score"] = float(d["score"])
        out.append(d)
    return out


# ─── 인덱싱 잡 (시스템 전역 — user_sub 미격리) ────────────────────────────────


async def list_unindexed_records(limit: int = 200) -> list[dict]:
    """Qdrant 미동기화 활성 record를 오래된 순으로 조회한다(인덱서 배치용)."""
    pool = db_pool.get_pool()
    if pool is None:
        return []
    rows = await pool.fetch(
        f"""
        SELECT {_SELECT_COLS} FROM agent.memory_records
         WHERE embedding_indexed = FALSE AND status = 'active' AND deleted_at IS NULL
         ORDER BY created_at ASC LIMIT $1
        """,
        limit,
    )
    return [_row_to_dict(r) for r in rows]


async def mark_embedding_indexed(record_ids: list[str]) -> int:
    """record들을 Qdrant 동기화 완료로 표시한다."""
    if not record_ids:
        return 0
    pool = db_pool.get_pool()
    if pool is None:
        return 0
    result = await pool.execute(
        "UPDATE agent.memory_records SET embedding_indexed = TRUE "
        "WHERE id = ANY($1::uuid[])",
        record_ids,
    )
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0
