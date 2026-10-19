"""장기 사용자 메모리 v2 저장소 (agent.memory_atoms / checkpoints / generation_log + L0 읽기).

PG=진실의 원천, Qdrant=L1 검색 인덱스(app/services/memory_v2/indexer.py).
소유자 격리: 모든 조회/수정은 SQL WHERE 에 user_sub 매칭을 강제한다. 인덱싱 잡
(list_unindexed_atoms / mark_atoms_indexed)만 시스템 전역이다.

커서 규약: L0 커서는 (created_at, id) 튜플이다. mark_job_success 는 커서가 **단조 전진**할
때만 갱신한다 — 재시도·중복 실행이 겹쳐도 옛 스냅샷으로 되감기지 않는다(같은 L0 재추출 방지).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from agent_arch.db.connection import db_pool

logger = logging.getLogger(__name__)

_NIL_UUID = "00000000-0000-0000-0000-000000000000"
_EPOCH = datetime(1970, 1, 1)

_ATOM_COLS = """
    id, user_sub, company_scope, atom_type, content, priority, scene_name, payload,
    source_message_ids, timestamps, version, status, superseded_by, generation_id,
    observed_count, embedding_indexed, created_at, updated_at, last_seen_at, expires_at
"""

_MSG_COLS = "id, conversation_id, user_sub, query, answer, tool_calls, metadata, created_at"
# L0 읽기 공통 필터: POS 플래닝 턴(pos_internal)과 실패·중단·가드 교정 턴(capture_skip)은 기억의 근거가 아니다.
_L0_FILTER = (
    "COALESCE(metadata->>'pos_internal', '') <> 'true' "
    "AND COALESCE(metadata->>'capture_skip', '') = ''"
)


def _loads(value: Any, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return default
    return value if value is not None else default


def _atom_row(row: Any) -> dict:
    d = dict(row)
    d["payload"] = _loads(d.get("payload"), {})
    if not isinstance(d["payload"], dict):
        d["payload"] = {}
    for k in ("id", "superseded_by", "generation_id"):
        if d.get(k) is not None:
            d[k] = str(d[k])
    d["source_message_ids"] = [str(x) for x in (d.get("source_message_ids") or [])]
    d["timestamps"] = list(d.get("timestamps") or [])
    return d


def _msg_row(row: Any) -> dict:
    d = dict(row)
    for k in ("id", "conversation_id"):
        if d.get(k) is not None:
            d[k] = str(d[k])
    d["tool_calls"] = _loads(d.get("tool_calls"), None)
    d["metadata"] = _loads(d.get("metadata"), {}) or {}
    return d


# ─── 체크포인트 / 스케줄 ───────────────────────────────────────────────────

async def bump_pending(
    user_sub: str,
    message_id: str,
    *,
    immediate: bool = False,
    idle_seconds: float = 600.0,
    default_threshold: int = 5,
) -> Optional[dict]:
    """캡처 1턴을 체크포인트에 반영한다(pending +1, next_due 갱신).

    첫 캡처는 커서를 (이 메시지의 created_at, nil uuid) 로 두어 **이 메시지부터** 배치에
    들어가게 한다(과거 이력은 백필 잡의 몫). 웜업 임계 1 이라 즉시 due.
    이후 캡처는 pending 이 임계(웜업 값, 졸업 후 default_threshold)에 닿거나 immediate
    (지시 마커 턴)면 즉시, 아니면 유휴 idle_seconds 뒤로 예약한다(캡처마다 리셋 = 디바운스).
    """
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO agent.memory_checkpoints
            (user_sub, last_cursor_at, last_cursor_id, pending_count, warmup_threshold, next_due_at)
        VALUES ($1,
                COALESCE((SELECT created_at FROM agent.messages WHERE id = $2::uuid), NOW()),
                $6::uuid, 1, 1, NOW())
        ON CONFLICT (user_sub) DO UPDATE SET
            pending_count = agent.memory_checkpoints.pending_count + 1,
            next_due_at = CASE
                WHEN $3::boolean THEN NOW()
                WHEN agent.memory_checkpoints.pending_count + 1 >= (
                        CASE WHEN agent.memory_checkpoints.warmup_threshold > 0
                             THEN agent.memory_checkpoints.warmup_threshold ELSE $5 END)
                    THEN NOW()
                ELSE NOW() + make_interval(secs => $4)
            END,
            updated_at = NOW()
        RETURNING user_sub, pending_count, warmup_threshold, next_due_at
        """,
        user_sub, message_id, bool(immediate), float(idle_seconds), int(default_threshold), _NIL_UUID,
    )
    return dict(row) if row else None


async def get_checkpoint(user_sub: str) -> Optional[dict]:
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        "SELECT * FROM agent.memory_checkpoints WHERE user_sub = $1", user_sub,
    )
    if not row:
        return None
    d = dict(row)
    if d.get("last_cursor_id") is not None:
        d["last_cursor_id"] = str(d["last_cursor_id"])
    return d


async def list_due_users(limit: int = 10) -> list[str]:
    """지금 L1 잡을 돌려야 할 유저(pending>0, due 경과)를 due 순으로 돌려준다."""
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        """
        SELECT user_sub FROM agent.memory_checkpoints
         WHERE pending_count > 0 AND next_due_at IS NOT NULL AND next_due_at <= NOW()
         ORDER BY next_due_at
         LIMIT $1
        """,
        int(limit),
    )
    return [r["user_sub"] for r in rows]


async def clear_pending(user_sub: str) -> None:
    """커서 이후 새 메시지가 없을 때 — 예약을 지운다(다음 캡처가 다시 만든다)."""
    pool = db_pool.get_pool()
    await pool.execute(
        """
        UPDATE agent.memory_checkpoints
           SET pending_count = 0, next_due_at = NULL, retry_count = 0, updated_at = NOW()
         WHERE user_sub = $1
        """,
        user_sub,
    )


async def mark_job_success(
    user_sub: str,
    *,
    cursor_at: datetime,
    cursor_id: str,
    processed: int,
    has_more: bool,
    scene_name: Optional[str],
    atoms_added: int,
    default_threshold: int = 5,
    idle_seconds: float = 600.0,
) -> None:
    """L1 잡 성공 반영: 커서 단조 전진, pending 차감, 웜업 임계 ×2, 백로그면 즉시 재예약.

    SET 절의 모든 표현식은 갱신 전 행 값을 본다(PostgreSQL 규칙) — pending_count 를 두 번
    참조해도 같은 옛 값이다.
    """
    pool = db_pool.get_pool()
    await pool.execute(
        """
        UPDATE agent.memory_checkpoints SET
            last_cursor_at = CASE
                WHEN last_cursor_at IS NULL OR (last_cursor_at, last_cursor_id) < ($2::timestamp, $3::uuid)
                THEN $2::timestamp ELSE last_cursor_at END,
            last_cursor_id = CASE
                WHEN last_cursor_at IS NULL OR (last_cursor_at, last_cursor_id) < ($2::timestamp, $3::uuid)
                THEN $3::uuid ELSE last_cursor_id END,
            pending_count = GREATEST(0, pending_count - $4),
            next_due_at = CASE
                WHEN $5::boolean THEN NOW()
                WHEN GREATEST(0, pending_count - $4) > 0 THEN NOW() + make_interval(secs => $8)
                ELSE NULL END,
            warmup_threshold = CASE
                WHEN warmup_threshold = 0 THEN 0
                WHEN warmup_threshold * 2 >= $7 THEN 0
                ELSE warmup_threshold * 2 END,
            last_l1_at = NOW(),
            retry_count = 0,
            last_error = NULL,
            last_scene_name = COALESCE($6, last_scene_name),
            atoms_since_persona = atoms_since_persona + $9,
            updated_at = NOW()
        WHERE user_sub = $1
        """,
        user_sub, cursor_at, cursor_id, int(processed), bool(has_more), scene_name,
        int(default_threshold), float(idle_seconds), int(atoms_added),
    )


async def mark_job_failure(
    user_sub: str,
    error: str,
    *,
    retry_delay_seconds: float = 30.0,
    max_retries: int = 5,
) -> int:
    """L1 잡 실패 반영: 커서 유지, retry_count+1, 지연 재예약. 상한 도달 시 포기(pending 0).

    포기해도 커서는 그대로라 다음 캡처가 pending 을 다시 올리면 같은 L0 부터 재시도된다
    (유실 없음, 자연 재시도).
    """
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        UPDATE agent.memory_checkpoints SET
            retry_count = retry_count + 1,
            last_error = LEFT($2, 500),
            next_due_at = CASE WHEN retry_count + 1 >= $4 THEN NULL
                               ELSE NOW() + make_interval(secs => $3) END,
            pending_count = CASE WHEN retry_count + 1 >= $4 THEN 0 ELSE pending_count END,
            updated_at = NOW()
        WHERE user_sub = $1
        RETURNING retry_count
        """,
        user_sub, error or "", float(retry_delay_seconds), int(max_retries),
    )
    return int(row["retry_count"]) if row else 0


# ─── L0 읽기 (agent.messages) ──────────────────────────────────────────────

async def fetch_messages_after(
    user_sub: str,
    cursor_at: Optional[datetime],
    cursor_id: Optional[str],
    limit: int,
) -> list[dict]:
    """커서 이후의 L0 턴을 오래된 순으로 돌려준다(pos_internal 제외). 커서 None = 처음부터(백필)."""
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        f"""
        SELECT {_MSG_COLS}
          FROM agent.messages
         WHERE user_sub = $1
           AND (created_at, id) > ($2::timestamp, $3::uuid)
           AND {_L0_FILTER}
         ORDER BY created_at, id
         LIMIT $4
        """,
        user_sub, cursor_at or _EPOCH, cursor_id or _NIL_UUID, int(limit),
    )
    return [_msg_row(r) for r in rows]


async def fetch_messages_before(
    user_sub: str,
    cursor_at: Optional[datetime],
    cursor_id: Optional[str],
    limit: int,
) -> list[dict]:
    """커서 이전(포함)의 최근 L0 턴 limit 개를 오래된 순으로 돌려준다(배경 맥락용)."""
    if cursor_at is None or limit <= 0:
        return []
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        f"""
        SELECT {_MSG_COLS}
          FROM agent.messages
         WHERE user_sub = $1
           AND (created_at, id) <= ($2::timestamp, $3::uuid)
           AND {_L0_FILTER}
         ORDER BY created_at DESC, id DESC
         LIMIT $4
        """,
        user_sub, cursor_at, cursor_id or _NIL_UUID, int(limit),
    )
    return [_msg_row(r) for r in reversed(rows)]


# ─── L1 atom ───────────────────────────────────────────────────────────────

async def insert_atom(
    user_sub: str,
    *,
    atom_type: str,
    content: str,
    priority: int,
    scene_name: Optional[str],
    payload: dict,
    source_message_ids: list[str],
    timestamps: list[datetime],
    version: int = 1,
    generation_id: Optional[str] = None,
    company_scope: Optional[str] = None,
) -> str:
    """새 active atom 을 넣고 id 를 돌려준다."""
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO agent.memory_atoms
            (user_sub, company_scope, atom_type, content, priority, scene_name, payload,
             source_message_ids, timestamps, version, generation_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::uuid[], $9::timestamp[], $10, $11::uuid)
        RETURNING id
        """,
        user_sub, company_scope, atom_type, content, int(priority), scene_name,
        json.dumps(payload or {}, ensure_ascii=False),
        list(source_message_ids or []), list(timestamps or []), int(version), generation_id,
    )
    return str(row["id"])


async def get_active_atoms_by_ids(user_sub: str, ids: list[str]) -> list[dict]:
    """본인 소유 active atom 만 하이드레이트한다(타 유저 id·삭제분은 조용히 제외)."""
    if not ids:
        return []
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        f"""
        SELECT {_ATOM_COLS} FROM agent.memory_atoms
         WHERE user_sub = $1 AND id = ANY($2::uuid[]) AND status = 'active'
        """,
        user_sub, list(dict.fromkeys(ids)),
    )
    return [_atom_row(r) for r in rows]


async def supersede_atoms(user_sub: str, ids: list[str], new_id: str) -> list[str]:
    """대상 atom 들을 superseded 로 바꾸고 new_id 를 가리키게 한다. 실제 바뀐 id 목록 반환."""
    if not ids:
        return []
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        """
        UPDATE agent.memory_atoms
           SET status = 'superseded', superseded_by = $3::uuid, updated_at = NOW()
         WHERE user_sub = $1 AND id = ANY($2::uuid[]) AND status = 'active'
        RETURNING id
        """,
        user_sub, list(ids), new_id,
    )
    return [str(r["id"]) for r in rows]


async def touch_atoms(user_sub: str, ids: list[str]) -> None:
    """skip 판정 — 기존 atom 의 관측 카운터·last_seen 만 올린다(게이트 아님, 관측용)."""
    if not ids:
        return
    pool = db_pool.get_pool()
    await pool.execute(
        """
        UPDATE agent.memory_atoms
           SET observed_count = observed_count + 1, last_seen_at = NOW(), updated_at = NOW()
         WHERE user_sub = $1 AND id = ANY($2::uuid[]) AND status = 'active'
        """,
        user_sub, list(ids),
    )


async def search_atoms_keyword(
    user_sub: str,
    query: str,
    *,
    limit: int = 5,
    atom_type: Optional[str] = None,
) -> list[dict]:
    """pg_trgm word_similarity(`<%`)로 본인 active atom 을 검색한다(하이브리드의 키워드 half)."""
    q = (query or "").strip()
    if not q:
        return []
    pool = db_pool.get_pool()
    params: list[Any] = [user_sub, q]
    where = ["user_sub = $1", "status = 'active'", "$2 <% content"]
    if atom_type:
        params.append(atom_type)
        where.append(f"atom_type = ${len(params)}")
    params.append(int(limit))
    rows = await pool.fetch(
        f"""
        SELECT {_ATOM_COLS}, word_similarity($2, content) AS score
          FROM agent.memory_atoms
         WHERE {' AND '.join(where)}
         ORDER BY score DESC, priority DESC
         LIMIT ${len(params)}
        """,
        *params,
    )
    out = []
    for r in rows:
        d = _atom_row(r)
        d["score"] = float(r["score"]) if r["score"] is not None else 0.0
        out.append(d)
    return out


async def list_active_atoms(
    user_sub: str,
    *,
    atom_type: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    pool = db_pool.get_pool()
    params: list[Any] = [user_sub]
    where = ["user_sub = $1", "status = 'active'"]
    if atom_type:
        params.append(atom_type)
        where.append(f"atom_type = ${len(params)}")
    params.append(int(limit))
    rows = await pool.fetch(
        f"""
        SELECT {_ATOM_COLS} FROM agent.memory_atoms
         WHERE {' AND '.join(where)}
         ORDER BY priority = -1 DESC, priority DESC, last_seen_at DESC
         LIMIT ${len(params)}
        """,
        *params,
    )
    return [_atom_row(r) for r in rows]


async def list_alias_hints(user_sub: str, limit: int = 30) -> list[str]:
    """추출 프롬프트의 [알려진 별칭] 힌트 — "별칭1/별칭2 → 정식이름" 문자열 목록."""
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        """
        SELECT payload FROM agent.memory_atoms
         WHERE user_sub = $1 AND atom_type = 'alias' AND status = 'active'
         ORDER BY priority DESC, last_seen_at DESC
         LIMIT $2
        """,
        user_sub, int(limit),
    )
    hints: list[str] = []
    for r in rows:
        p = _loads(r["payload"], {})
        if not isinstance(p, dict):
            continue
        aliases = [str(a) for a in (p.get("aliases") or []) if a]
        canon = p.get("canonical_name")
        if aliases and canon:
            # id 유무를 같이 적는다 — 추출기가 "이미 아는 별칭"이라 넘기지 않고, 이 턴의 도구 요약에서
            # id 를 확인했으면 canonical_id 를 채워 다시 내게 하기 위함(id 는 회상 레시피의 재료).
            state = "id 있음" if p.get("canonical_id") else "id 없음"
            hints.append(f"{'/'.join(aliases[:3])} → {canon} ({state})")
    return hints


async def list_unindexed_atoms(limit: int = 200) -> list[dict]:
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        f"""
        SELECT {_ATOM_COLS} FROM agent.memory_atoms
         WHERE status = 'active' AND embedding_indexed = FALSE
         ORDER BY created_at
         LIMIT $1
        """,
        int(limit),
    )
    return [_atom_row(r) for r in rows]


async def mark_atoms_indexed(ids: list[str]) -> None:
    if not ids:
        return
    pool = db_pool.get_pool()
    await pool.execute(
        "UPDATE agent.memory_atoms SET embedding_indexed = TRUE WHERE id = ANY($1::uuid[])",
        list(ids),
    )


# ─── 생성 로그 ─────────────────────────────────────────────────────────────

async def insert_generation_log(
    *,
    user_sub: str,
    layer: str,
    status: str,
    model: Optional[str],
    prompt_sha: Optional[str],
    input_message_ids: list[str],
    output_atom_ids: list[str],
    decision_counts: dict,
    latency_ms: Optional[int],
    error: Optional[str] = None,
    raw_output: Optional[str] = None,
    log_id: Optional[str] = None,
) -> str:
    """생성 로그 한 행. log_id 를 주면 그 id 로 넣는다(atom.generation_id 와 선연결용)."""
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO agent.memory_generation_log
            (id, user_sub, layer, status, model, prompt_sha, input_message_ids, output_atom_ids,
             decision_counts, latency_ms, error, raw_output)
        VALUES (COALESCE($1::uuid, uuid_generate_v4()), $2, $3, $4, $5, $6, $7::uuid[], $8::uuid[],
                $9::jsonb, $10, $11, $12)
        RETURNING id
        """,
        log_id, user_sub, layer, status, model, prompt_sha,
        list(input_message_ids or []), list(output_atom_ids or []),
        json.dumps(decision_counts or {}, ensure_ascii=False),
        latency_ms, (error or None), (raw_output[:4000] if raw_output else None),
    )
    return str(row["id"])


# ─── 회상 · 도구용 조회 (S2) ───────────────────────────────────────────────

async def list_strict_instructions(user_sub: str, limit: int = 8) -> list[dict]:
    """priority -1(전역 강제) instruction — 질의와 무관하게 항상 주입되는 유일한 부류."""
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        f"""
        SELECT {_ATOM_COLS} FROM agent.memory_atoms
         WHERE user_sub = $1 AND status = 'active' AND atom_type = 'instruction' AND priority = -1
         ORDER BY last_seen_at DESC
         LIMIT $2
        """,
        user_sub, int(limit),
    )
    return [_atom_row(r) for r in rows]


async def soft_delete_atom(user_sub: str, atom_id: str) -> Optional[dict]:
    """본인 active atom 1건을 deleted 로. 지운 행의 (atom_type, content) 반환, 없으면 None(멱등)."""
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        UPDATE agent.memory_atoms
           SET status = 'deleted', deleted_at = NOW(), updated_at = NOW()
         WHERE user_sub = $1 AND id = $2::uuid AND status = 'active'
        RETURNING id, atom_type, content, scene_name
        """,
        user_sub, atom_id,
    )
    if not row:
        return None
    d = dict(row)
    d["id"] = str(d["id"])
    return d


async def get_active_persona(user_sub: str, company_scope: Optional[str] = None) -> Optional[dict]:
    """활성 L3 페르소나. 회사 스코프 본이 있으면 그것, 없으면 유저 전역(NULL) 본."""
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, content, version, source_atom_count, company_scope, updated_at
          FROM agent.memory_persona
         WHERE user_sub = $1 AND status = 'active'
           AND (company_scope IS NULL OR company_scope = $2)
         ORDER BY (company_scope IS NOT NULL) DESC
         LIMIT 1
        """,
        user_sub, company_scope,
    )
    if not row:
        return None
    d = dict(row)
    d["id"] = str(d["id"])
    return d


async def list_scenes(user_sub: str, *, limit: int = 8, with_content: bool = False) -> list[dict]:
    """활성 L2 장면을 heat 순으로(내비게이션·L2 빌드 입력)."""
    pool = db_pool.get_pool()
    cols = "id, name, title, summary, heat, version, evidence_atom_ids, updated_at" + (", content" if with_content else "")
    rows = await pool.fetch(
        f"""
        SELECT {cols} FROM agent.memory_scenes
         WHERE user_sub = $1 AND status = 'active'
         ORDER BY heat DESC, updated_at DESC
         LIMIT $2
        """,
        user_sub, int(limit),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"])
        d["evidence_atom_ids"] = [str(x) for x in (d.get("evidence_atom_ids") or [])]
        out.append(d)
    return out


async def get_scene(user_sub: str, name: str) -> Optional[dict]:
    """장면 문서 전문(memory_read_scene 도구)."""
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        SELECT id, name, title, summary, content, heat, version, updated_at
          FROM agent.memory_scenes
         WHERE user_sub = $1 AND status = 'active' AND name = $2
        """,
        user_sub, (name or "").strip(),
    )
    if not row:
        return None
    d = dict(row)
    d["id"] = str(d["id"])
    return d


# ─── L2 장면 · L3 페르소나 쓰기 (S3) ──────────────────────────────────────

async def list_atoms_since(user_sub: str, since: Optional[datetime], limit: int = 120) -> list[dict]:
    """since 이후 생성된 active atom(L2 증분 입력). since None = 전체(상한 내, 오래된 순)."""
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        f"""
        SELECT {_ATOM_COLS} FROM agent.memory_atoms
         WHERE user_sub = $1 AND status = 'active' AND created_at > $2::timestamp
         ORDER BY created_at
         LIMIT $3
        """,
        user_sub, since or _EPOCH, int(limit),
    )
    return [_atom_row(r) for r in rows]


async def count_active_atoms(user_sub: str) -> int:
    pool = db_pool.get_pool()
    return int(await pool.fetchval(
        "SELECT COUNT(*) FROM agent.memory_atoms WHERE user_sub = $1 AND status = 'active'", user_sub,
    ) or 0)


async def count_active_scenes(user_sub: str) -> int:
    pool = db_pool.get_pool()
    return int(await pool.fetchval(
        "SELECT COUNT(*) FROM agent.memory_scenes WHERE user_sub = $1 AND status = 'active'", user_sub,
    ) or 0)


async def decay_scene_heat(user_sub: str, factor: float = 0.9) -> None:
    """모든 활성 장면의 heat 를 factor 배로 식힌다(L2 실행마다 — 최근 관측이 위로 오게)."""
    pool = db_pool.get_pool()
    await pool.execute(
        "UPDATE agent.memory_scenes SET heat = heat * $2 WHERE user_sub = $1 AND status = 'active'",
        user_sub, float(factor),
    )


async def upsert_scene(
    user_sub: str,
    *,
    name: str,
    title: str,
    summary: str,
    content: str,
    evidence_atom_ids: list[str],
    heat_delta: float,
    company_scope: Optional[str] = None,
) -> str:
    """활성 장면을 이름으로 upsert 한다(부분 유니크 인덱스 추론). 갱신 시 version+1, heat 누적."""
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO agent.memory_scenes
            (user_sub, company_scope, name, title, summary, content, evidence_atom_ids, heat)
        VALUES ($1, $2, $3, $4, $5, $6, $7::uuid[], $8)
        ON CONFLICT (user_sub, COALESCE(company_scope, ''), name) WHERE status = 'active'
        DO UPDATE SET
            title = EXCLUDED.title,
            summary = EXCLUDED.summary,
            content = EXCLUDED.content,
            evidence_atom_ids = (
                SELECT ARRAY(SELECT DISTINCT e FROM unnest(agent.memory_scenes.evidence_atom_ids || EXCLUDED.evidence_atom_ids) AS e)
            ),
            heat = agent.memory_scenes.heat + EXCLUDED.heat,
            version = agent.memory_scenes.version + 1,
            last_built_at = NOW(),
            updated_at = NOW()
        RETURNING id
        """,
        user_sub, company_scope, name, title, summary or "", content,
        list(evidence_atom_ids or []), float(heat_delta),
    )
    return str(row["id"])


async def resolve_evidence(user_sub: str, ids: list[str]) -> dict[str, Optional[str]]:
    """장면 근거 id → 지금 살아 있는 atom id. 대체(superseded)는 체인을 따라가고, 끝이 삭제·유실이면 None.

    장면의 근거 목록은 atom 이 update/merge 로 새 id 를 받아도 갱신되지 않는다 — 옛 id 를 그대로 보면
    "살아 있는 기억을 대체한 것"과 "지워진 기억"을 구별할 수 없다.
    """
    if not ids:
        return {}
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        """
        WITH RECURSIVE chain(orig, cur, depth) AS (
            SELECT id, id, 0 FROM agent.memory_atoms WHERE user_sub = $1 AND id = ANY($2::uuid[])
            UNION ALL
            SELECT c.orig, a.superseded_by, c.depth + 1
              FROM chain c JOIN agent.memory_atoms a ON a.id = c.cur
             WHERE a.user_sub = $1 AND a.status = 'superseded' AND a.superseded_by IS NOT NULL AND c.depth < 20
        )
        SELECT c.orig, c.cur FROM chain c JOIN agent.memory_atoms a ON a.id = c.cur
         WHERE a.user_sub = $1 AND a.status = 'active'
        """,
        user_sub, list(dict.fromkeys(ids)),
    )
    live = {str(r["orig"]): str(r["cur"]) for r in rows}
    return {i: live.get(i) for i in ids}


async def rewrite_scene(
    user_sub: str, scene_id: str, *, title: str, summary: str, content: str, evidence_atom_ids: list[str],
) -> bool:
    """장면 하나를 제자리에서 다시 쓴다 — 근거는 **교체**(upsert_scene 의 합집합과 다르다). 이름·heat 유지."""
    pool = db_pool.get_pool()
    status = await pool.execute(
        """
        UPDATE agent.memory_scenes
           SET title = $3, summary = $4, content = $5, evidence_atom_ids = $6::uuid[],
               version = version + 1, last_built_at = NOW(), updated_at = NOW()
         WHERE user_sub = $1 AND id = $2::uuid AND status = 'active'
        """,
        user_sub, scene_id, title, summary or "", content, list(dict.fromkeys(evidence_atom_ids or [])),
    )
    return str(status).endswith("1")


async def delete_scene_by_id(user_sub: str, scene_id: str) -> bool:
    """이름이 아니라 id 로 지운다 — 같은 이름의 장면이 회사 범위별로 여럿일 수 있다."""
    pool = db_pool.get_pool()
    status = await pool.execute(
        """
        UPDATE agent.memory_scenes SET status = 'deleted', updated_at = NOW()
         WHERE user_sub = $1 AND id = $2::uuid AND status = 'active'
        """,
        user_sub, scene_id,
    )
    return str(status).endswith("1")


async def delete_scene(user_sub: str, name: str) -> bool:
    pool = db_pool.get_pool()
    status = await pool.execute(
        """
        UPDATE agent.memory_scenes SET status = 'deleted', updated_at = NOW()
         WHERE user_sub = $1 AND name = $2 AND status = 'active'
        """,
        user_sub, name,
    )
    return str(status).endswith("1")


async def upsert_persona(
    user_sub: str,
    content: str,
    *,
    source_atom_count: int,
    company_scope: Optional[str] = None,
) -> str:
    """활성 페르소나를 upsert(유저·회사스코프당 1개). 갱신 시 version+1."""
    pool = db_pool.get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO agent.memory_persona (user_sub, company_scope, content, source_atom_count)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_sub, COALESCE(company_scope, '')) WHERE status = 'active'
        DO UPDATE SET
            content = EXCLUDED.content,
            source_atom_count = EXCLUDED.source_atom_count,
            version = agent.memory_persona.version + 1,
            updated_at = NOW()
        RETURNING id
        """,
        user_sub, company_scope, content, int(source_atom_count),
    )
    return str(row["id"])


async def list_active_persona_scopes(user_sub: str) -> list[Optional[str]]:
    """활성 페르소나가 있는 회사 범위 목록(범위 없는 본은 None)."""
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        "SELECT DISTINCT company_scope FROM agent.memory_persona WHERE user_sub = $1 AND status = 'active'",
        user_sub,
    )
    return [r["company_scope"] for r in rows]


async def delete_persona(user_sub: str, company_scope: Optional[str] = None) -> bool:
    pool = db_pool.get_pool()
    status = await pool.execute(
        """
        UPDATE agent.memory_persona SET status = 'deleted', updated_at = NOW()
         WHERE user_sub = $1 AND status = 'active' AND COALESCE(company_scope, '') = COALESCE($2, '')
        """,
        user_sub, company_scope,
    )
    return str(status).endswith("1")


async def mark_l2_done(user_sub: str) -> None:
    pool = db_pool.get_pool()
    await pool.execute(
        "UPDATE agent.memory_checkpoints SET last_l2_at = NOW(), updated_at = NOW() WHERE user_sub = $1", user_sub,
    )


async def mark_l3_done(user_sub: str) -> None:
    pool = db_pool.get_pool()
    await pool.execute(
        """
        UPDATE agent.memory_checkpoints
           SET last_l3_at = NOW(), atoms_since_persona = 0, updated_at = NOW()
         WHERE user_sub = $1
        """,
        user_sub,
    )


# ─── 관리 · 백필 (S4) ──────────────────────────────────────────────────────

async def seed_checkpoint(user_sub: str, cursor_at: datetime, cursor_id: str, pending: int) -> None:
    """백필용: 커서를 임의 지점으로 놓고 즉시 due 로 만든다(웜업 0 = 기본 N 배치)."""
    pool = db_pool.get_pool()
    await pool.execute(
        """
        INSERT INTO agent.memory_checkpoints
            (user_sub, last_cursor_at, last_cursor_id, pending_count, warmup_threshold, next_due_at)
        VALUES ($1, $2, $3::uuid, $4, 0, NOW())
        ON CONFLICT (user_sub) DO UPDATE SET
            last_cursor_at = $2, last_cursor_id = $3::uuid, pending_count = $4, warmup_threshold = 0,
            next_due_at = NOW(), retry_count = 0, last_error = NULL, updated_at = NOW()
        """,
        user_sub, cursor_at, cursor_id, int(pending),
    )


async def count_messages_after(user_sub: str, cursor_at: Optional[datetime], cursor_id: Optional[str]) -> int:
    pool = db_pool.get_pool()
    return int(await pool.fetchval(
        f"""
        SELECT COUNT(*) FROM agent.messages
         WHERE user_sub = $1 AND (created_at, id) > ($2::timestamp, $3::uuid)
           AND {_L0_FILTER}
        """,
        user_sub, cursor_at or _EPOCH, cursor_id or _NIL_UUID,
    ) or 0)


async def list_users_with_messages(min_count: int = 20, since: Optional[datetime] = None) -> list[dict]:
    """백필 대상 후보: 메시지 수가 min_count 이상인 user_sub 와 건수(많은 순)."""
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        f"""
        SELECT user_sub, COUNT(*) AS n
          FROM agent.messages
         WHERE user_sub IS NOT NULL AND created_at > $2::timestamp
           AND {_L0_FILTER}
         GROUP BY user_sub HAVING COUNT(*) >= $1
         ORDER BY n DESC
        """,
        int(min_count), since or _EPOCH,
    )
    return [{"user_sub": r["user_sub"], "count": int(r["n"])} for r in rows]


async def soft_delete_all_atoms(user_sub: str) -> int:
    pool = db_pool.get_pool()
    status = await pool.execute(
        """
        UPDATE agent.memory_atoms SET status = 'deleted', deleted_at = NOW(), updated_at = NOW()
         WHERE user_sub = $1 AND status <> 'deleted'
        """,
        user_sub,
    )
    return _affected(status)


async def soft_delete_all_scenes(user_sub: str) -> int:
    pool = db_pool.get_pool()
    status = await pool.execute(
        "UPDATE agent.memory_scenes SET status = 'deleted', updated_at = NOW() WHERE user_sub = $1 AND status = 'active'",
        user_sub,
    )
    return _affected(status)


async def delete_checkpoint(user_sub: str) -> None:
    pool = db_pool.get_pool()
    await pool.execute("DELETE FROM agent.memory_checkpoints WHERE user_sub = $1", user_sub)


async def list_generation_logs(user_sub: str, limit: int = 20) -> list[dict]:
    pool = db_pool.get_pool()
    rows = await pool.fetch(
        """
        SELECT id, layer, status, model, prompt_sha, decision_counts, latency_ms, error, created_at,
               cardinality(input_message_ids) AS input_count, cardinality(output_atom_ids) AS output_count
          FROM agent.memory_generation_log
         WHERE user_sub = $1 ORDER BY created_at DESC LIMIT $2
        """,
        user_sub, int(limit),
    )
    out = []
    for r in rows:
        d = dict(r)
        d["id"] = str(d["id"])
        d["decision_counts"] = _loads(d.get("decision_counts"), {})
        out.append(d)
    return out


def _affected(status: Any) -> int:
    """asyncpg execute 상태 문자열("UPDATE 3")에서 영향 행 수."""
    try:
        return int(str(status).split()[-1])
    except (ValueError, IndexError):
        return 0
