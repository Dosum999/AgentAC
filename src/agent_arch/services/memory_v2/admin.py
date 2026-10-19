"""관리 기능 — 내 기억 조회/초기화(셀프서비스 API), 백필, v1 별칭 시드.

백필은 스케줄러와 **같은 L1 잡**을 커서만 과거로 놓고 반복 호출한다(코드 경로 분기 없음).
초기화는 3층(atom·scene·persona)+Qdrant+체크포인트를 한 번에 지운다 — v1 이 벌크 삭제권을
구현만 하고 배선하지 않았던 공백을 채운다.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from agent_arch.db.repository import memory_v2 as repo
from agent_arch.services.memory_v2 import dedup, layers, pipeline, place as place_mod, writer
from agent_arch.services.memory_v2.models import AtomDraft

logger = logging.getLogger(__name__)

_NIL_UUID = "00000000-0000-0000-0000-000000000000"
_EPOCH = datetime(1970, 1, 1)
UPPER_LOCK_RETRIES = 4
UPPER_LOCK_WAIT_SECONDS = 15.0


def _iso(v: Any) -> Optional[str]:
    return v.isoformat() if isinstance(v, datetime) else None


def _public_payload(payload: Any) -> dict:
    """사용자에게 보여줄 payload — 채점용 내부 키(장소·출처 등급)는 뺀다.

    장소 키는 `mattermost:6uwx…` 같은 불투명한 문자열이라 내 기억 화면에 그대로 뜨면 뜻 없는
    글자만 늘어난다. 나중에 채널 **이름**으로 보여줄 수 있게 되면 그때 제대로 된 필드로 낸다.
    """
    if not isinstance(payload, dict):
        return {}
    return {k: v for k, v in payload.items()
            if k not in (place_mod.PAYLOAD_PLACES, place_mod.PAYLOAD_ORIGIN)}


async def export_user_memory(user_sub: str, *, limit: int = 100) -> dict:
    """내 기억 한 장: atom(우선순위순) · 장면(heat 순) · 페르소나 · 체크포인트 · 최근 생성 로그."""
    atoms = await repo.list_active_atoms(user_sub, limit=limit)
    scenes = await repo.list_scenes(user_sub, limit=15, with_content=True)
    persona = await repo.get_active_persona(user_sub)
    cp = await repo.get_checkpoint(user_sub) or {}
    logs = await repo.list_generation_logs(user_sub, limit=10)
    return {
        "atoms": [
            {
                "id": a["id"], "type": a["atom_type"], "content": a["content"], "priority": a["priority"],
                "scene": a.get("scene_name"), "payload": _public_payload(a.get("payload")), "version": a.get("version"),
                "created_at": _iso(a.get("created_at")), "last_seen_at": _iso(a.get("last_seen_at")),
            }
            for a in atoms
        ],
        "scenes": [
            {
                "name": s["name"], "title": s["title"], "summary": s.get("summary"), "content": s.get("content"),
                "heat": float(s.get("heat") or 0), "version": s.get("version"), "updated_at": _iso(s.get("updated_at")),
            }
            for s in scenes
        ],
        "persona": (
            {"content": persona["content"], "version": persona.get("version"), "updated_at": _iso(persona.get("updated_at"))}
            if persona else None
        ),
        "checkpoint": {
            "pending_count": cp.get("pending_count"), "next_due_at": _iso(cp.get("next_due_at")),
            "last_l1_at": _iso(cp.get("last_l1_at")), "last_l2_at": _iso(cp.get("last_l2_at")), "last_l3_at": _iso(cp.get("last_l3_at")),
        } if cp else None,
        "recent_generations": [
            {"layer": g["layer"], "status": g["status"], "model": g.get("model"), "counts": g.get("decision_counts"),
             "latency_ms": g.get("latency_ms"), "error": g.get("error"), "at": _iso(g.get("created_at"))}
            for g in logs
        ],
    }


async def reset_user_memory(user_sub: str) -> dict:
    """내 기억 전부 초기화(soft-delete) + Qdrant 포인트 삭제 + 체크포인트 삭제. 건수 반환."""
    atoms = await repo.soft_delete_all_atoms(user_sub)
    scenes = await repo.soft_delete_all_scenes(user_sub)
    persona = await repo.delete_persona(user_sub)
    await repo.delete_checkpoint(user_sub)
    qdrant_ok = True
    try:
        from agent_arch.services.memory_v2.indexer import get_v2_indexer
        await (await get_v2_indexer()).delete_user_points(user_sub)
    except Exception as e:
        qdrant_ok = False
        logger.warning("memory_v2 reset: Qdrant 포인트 삭제 실패(user=%s): %s", user_sub[:8], e)
    from agent_arch.services.memory_v2.recall import invalidate_stable_block
    await invalidate_stable_block(user_sub)
    logger.info("memory_v2 reset user=%s atoms=%d scenes=%d persona=%s", user_sub[:8], atoms, scenes, persona)
    return {"atoms": atoms, "scenes": scenes, "persona": bool(persona), "qdrant": qdrant_ok}


def alias_draft_from_v1(record: dict) -> Optional[AtomDraft]:
    """v1 entity_alias record → v2 alias AtomDraft. aliases/canonical_name 없으면 None."""
    p = record.get("payload") or {}
    aliases = [str(a).strip() for a in (p.get("aliases") or []) if str(a).strip()]
    canon = str(p.get("canonical_name") or "").strip()
    if not aliases or not canon:
        return None
    # v1 데이터 품질 필터(백필 실측): 정식 이름이 별칭과 같으면 정보가 없고("watchdog → watchdog"),
    # 봇 이름이 별칭이면 v1 오염("에이전트 → 사용자 실명")이다. 둘 다 시드하지 않는다.
    from agent_arch.services.memory_v2.extractor import BOT_SELF_MARKERS
    if any(any(m.lower() in a.lower() for m in BOT_SELF_MARKERS) for a in aliases):
        return None
    informative = [a for a in aliases if a.lower() != canon.lower()]
    cid = str(p.get("canonical_id") or "").strip()
    has_id = bool(cid) and cid.lower() != canon.lower()
    if not informative and not has_id:
        return None                              # 이름도 id 도 새 정보가 없다("watchdog → watchdog")
    if informative and not has_id:
        aliases = informative                    # id 가 없으면 이름 그대로인 별칭은 정보가 없다
    # id 가 있으면 별칭이 정식 이름과 같아도 유지한다 — "watchdog" 라고 말했을 때 채널 id 로
    # 바로 가는 것이 이 기억의 가치다(라이브 실측: 'watchdog 채널' 만 남겨 'watchdog에' 가 어휘 매칭 실패).
    payload = {"aliases": aliases[:10], "canonical_name": canon}
    for k in ("canonical_kind", "canonical_id"):
        if p.get(k):
            payload[k] = str(p[k])
    support = int(record.get("support_count") or 1)
    priority = min(85, 40 + support * 3)         # 71회 관측된 watchdog 은 85, 1회는 43
    content = f"사용자가 '{aliases[0]}'(으)로 지칭하는 대상은 {canon}이다."
    return AtomDraft(
        atom_type="alias", content=content, priority=priority, payload=payload,
        source_message_ids=list(record.get("source_message_ids") or [])[:20],
    )


async def seed_aliases_from_v1(user_sub: str) -> int:
    """v1 별칭(검증된 ROI)을 v2 alias atom 으로 시드한다. 배치 판정기를 거치므로 재실행 안전."""
    from agent_arch.db.repository.memory_records import list_active_records
    records = await list_active_records(user_sub, record_type="entity_alias", limit=100)
    drafts = [d for d in (alias_draft_from_v1(r) for r in records) if d]
    if not drafts:
        return 0
    judge = await dedup.judge_batch(user_sub, drafts)
    stats = await writer.apply_decisions(user_sub, drafts, judge.decisions, generation_id=None, scene_name=None)
    logger.info("memory_v2 alias seed user=%s v1=%d written=%d skipped=%d", user_sub[:8], len(records), stats.written, stats.skipped)
    return stats.written


async def backfill_user(
    user_sub: str,
    *,
    since: Optional[datetime] = None,
    reset: bool = False,
    seed_aliases: bool = False,
    max_rounds: int = 500,
    upper_layers: bool = True,
    resume: bool = False,
) -> dict:
    """유저 한 명의 L0 를 since 부터 배치로 재추출한다(= 스케줄러와 같은 L1 잡 반복).

    resume=True 면 커서를 되감지 않고 기존 체크포인트 커서부터 이어간다(중단된 백필 재개용 —
    처음부터 다시 돌리면 판정기가 대부분 skip 하니 결과는 같지만 LLM 비용을 다시 낸다).

    Returns: {"rounds", "processed", "stored", "updated", "merged", "skipped", "failed", "aliases", "upper"}.
    """
    summary: dict[str, Any] = {"user_sub": user_sub, "rounds": 0, "processed": 0, "stored": 0, "updated": 0,
                               "merged": 0, "skipped": 0, "failed": 0, "aliases": 0, "upper": None}
    if reset:
        summary["reset"] = await reset_user_memory(user_sub)
    if seed_aliases:
        summary["aliases"] = await seed_aliases_from_v1(user_sub)

    cursor_at, cursor_id = since or _EPOCH, _NIL_UUID
    if resume:
        cp = await repo.get_checkpoint(user_sub)
        if cp and cp.get("last_cursor_at"):
            cursor_at, cursor_id = cp["last_cursor_at"], cp.get("last_cursor_id") or _NIL_UUID
    pending = await repo.count_messages_after(user_sub, cursor_at, cursor_id)
    await repo.seed_checkpoint(user_sub, cursor_at, cursor_id, pending)
    summary["pending_at_start"] = pending

    llm_down = False
    while summary["rounds"] < max_rounds:
        res = await pipeline.run_l1_job(user_sub)
        summary["rounds"] += 1
        summary["processed"] += res.processed
        if res.stats:
            summary["stored"] += res.stats.stored
            summary["updated"] += res.stats.updated
            summary["merged"] += res.stats.merged
            summary["skipped"] += res.stats.skipped
        if res.status == "failed":
            summary["failed"] += 1
            if not res.processed:               # LLM 호출 실패 — 재시도 예약은 스케줄러 몫, 여기선 멈춘다
                llm_down = True
                break
        if res.status in ("skipped_locked", "no_checkpoint", "no_new") or not res.has_more:
            break

    # 상위층은 이번에 쓴 게 없어도 돌린다(중단된 백필의 재개처럼 L1 은 끝났는데 L2/L3 만 남은 경우).
    # LLM 이 죽어 있으면 상위층도 실패할 테니 그때만 건너뛴다.
    if upper_layers and not llm_down:
        # 다른 실행(직전 forget 이 예약한 상위층 등)이 락을 쥐고 있으면 잠시 기다려 다시 시도한다.
        out: dict = {"scenes": None, "persona": "locked"}
        for attempt in range(UPPER_LOCK_RETRIES):
            out = await layers.run_upper_layers(user_sub, force_scenes=True, force_persona=True, fresh_persona=True)
            if out.get("persona") != "locked":
                break
            await asyncio.sleep(UPPER_LOCK_WAIT_SECONDS)
        sc = out.get("scenes")
        persona = out.get("persona")
        summary["upper"] = {
            "scenes": (sc.created, sc.updated, sc.deleted, sc.error) if sc else None,
            "persona": persona if persona in (None, "locked", "skipped") else f"{len(persona)} chars",
        }
    logger.info("memory_v2 backfill %s", summary)
    return summary
