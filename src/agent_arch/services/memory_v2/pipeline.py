"""L1 잡 — 유저 하나의 커서 이후 L0 를 읽어 추출 → 판정 → 쓰기 → 커서 전진 → 생성 로그.

불변식:
- 유저 락(Redis SET NX) 아래에서만 돈다. 겹치면 skipped_locked.
- 커서는 성공했을 때만, 그리고 단조로만 전진한다(repository.mark_job_success).
- LLM **호출** 실패는 커서를 두고 재시도(≤5, 30초). **파싱** 실패는 커서를 전진시키고 실패
  로그를 남긴다 — 같은 배치를 5번 되풀이해 파이프라인이 그 자리에 굳는 것보다, 한 배치를
  건너뛰고 생성 로그로 빈도를 드러내는 편이 낫다.
- 결과는 INFO 한 줄로 남긴다(v1 의 관측성 0 을 여기서 끝낸다).
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from agent_arch.db.repository import memory_v2 as repo
from agent_arch.memory.redis_store import redis_store
from agent_arch.services.memory_v2 import dedup, extractor, writer
from agent_arch.services.memory_v2.capture import DEFAULT_THRESHOLD, IDLE_SECONDS
from agent_arch.services.memory_v2.models import L1JobResult, WriteStats

logger = logging.getLogger(__name__)

NEW_BATCH = 10          # 배치당 신규 턴 상한(추출기 입력 창과 일치 — 넘치면 has_more 로 즉시 재예약)
BACKGROUND = 5          # 배경 맥락 턴 수(추출 금지)
LOCK_TTL_SECONDS = 300
RETRY_DELAY_SECONDS = 30.0
MAX_RETRIES = 5


def _lock_key(user_sub: str) -> str:
    return f"memory2:lock:{user_sub}"


async def run_l1_job(user_sub: str) -> L1JobResult:
    """유저 락을 잡고 배치 하나를 처리한다."""
    key = _lock_key(user_sub)
    if not await redis_store.set_nx(key, "1", expire=LOCK_TTL_SECONDS):
        return L1JobResult(user_sub=user_sub, status="skipped_locked")
    try:
        result = await _run_locked(user_sub)
    except Exception as e:
        logger.exception("memory_v2 L1 잡 예외(user=%s)", user_sub[:8])
        try:
            await repo.mark_job_failure(
                user_sub, f"{type(e).__name__}: {e}",
                retry_delay_seconds=RETRY_DELAY_SECONDS, max_retries=MAX_RETRIES,
            )
        except Exception:
            logger.exception("memory_v2 실패 기록도 실패(user=%s)", user_sub[:8])
        result = L1JobResult(user_sub=user_sub, status="failed", error=f"{type(e).__name__}: {e}"[:300])
    finally:
        try:
            await redis_store.delete(key)
        except Exception:
            pass
    _log_result(result)
    return result


def _log_result(r: L1JobResult) -> None:
    s = r.stats or WriteStats()
    logger.info(
        "memory_v2 L1 user=%s status=%s processed=%d stored=%d updated=%d merged=%d skipped=%d "
        "scene=%s has_more=%s%s",
        r.user_sub[:8], r.status, r.processed, s.stored, s.updated, s.merged, s.skipped,
        (r.scene_name or "-")[:30], r.has_more, f" error={r.error}" if r.error else "",
    )


async def _run_locked(user_sub: str) -> L1JobResult:
    cp = await repo.get_checkpoint(user_sub)
    if not cp:
        return L1JobResult(user_sub=user_sub, status="no_checkpoint")
    cursor_at, cursor_id = cp.get("last_cursor_at"), cp.get("last_cursor_id")

    new_msgs = await repo.fetch_messages_after(user_sub, cursor_at, cursor_id, NEW_BATCH + 1)
    has_more = len(new_msgs) > NEW_BATCH
    new_msgs = new_msgs[:NEW_BATCH]
    if not new_msgs:
        await repo.clear_pending(user_sub)
        return L1JobResult(user_sub=user_sub, status="no_new")

    background = await repo.fetch_messages_before(user_sub, cursor_at, cursor_id, BACKGROUND)
    try:
        known_aliases = await repo.list_alias_hints(user_sub)
    except Exception:
        known_aliases = []

    extraction = await extractor.extract_atoms(
        new_msgs, background,
        last_scene_name=cp.get("last_scene_name"), known_aliases=known_aliases,
    )
    last = new_msgs[-1]
    new_cursor_at, new_cursor_id = last["created_at"], last["id"]

    # LLM 호출 자체가 실패 → 커서 유지, 재시도 예약.
    if extraction.error and not extraction.raw:
        retry = await repo.mark_job_failure(
            user_sub, extraction.error,
            retry_delay_seconds=RETRY_DELAY_SECONDS, max_retries=MAX_RETRIES,
        )
        await repo.insert_generation_log(
            user_sub=user_sub, layer="l1", status="failed", model=extraction.model,
            prompt_sha=extraction.prompt_sha, input_message_ids=extraction.input_message_ids,
            output_atom_ids=[], decision_counts={}, latency_ms=extraction.latency_ms,
            error=f"{extraction.error} (retry {retry}/{MAX_RETRIES})",
        )
        return L1JobResult(user_sub=user_sub, status="failed", processed=0, error=extraction.error)

    gen_id = str(uuid.uuid4())
    stats = WriteStats()
    judge_error: Optional[str] = None
    judge_latency = 0
    judge_raw = ""
    pool_size = 0

    if extraction.drafts:
        # 장소·출처 등급은 판정·쓰기 **전에** 코드가 찍는다(place 모듈 계약).
        extractor.stamp_places(new_msgs, extraction.drafts)
        judge = await dedup.judge_batch(user_sub, extraction.drafts)
        judge_error, judge_latency, judge_raw, pool_size = judge.error, judge.latency_ms, judge.raw, judge.pool_size
        ts_for = {
            i: extractor.message_timestamps(new_msgs, d.source_message_ids)
            for i, d in enumerate(extraction.drafts)
        }
        stats = await writer.apply_decisions(
            user_sub, extraction.drafts, judge.decisions,
            generation_id=gen_id, scene_name=extraction.scene_name, timestamps_for=ts_for,
        )

    if extraction.error:            # parse_failed — 커서는 전진, 로그에 원문 보관
        status, error = "failed", extraction.error
    elif not extraction.drafts:
        status, error = "empty", None
    else:
        status, error = "succeeded", judge_error

    counts = {**stats.counts(), "drafts": len(extraction.drafts), "pool": pool_size}
    raw_log = extraction.raw if not judge_raw else f"{extraction.raw}\n--- dedup ---\n{judge_raw}"
    await repo.insert_generation_log(
        log_id=gen_id, user_sub=user_sub, layer="l1", status=status, model=extraction.model,
        prompt_sha=extraction.prompt_sha, input_message_ids=extraction.input_message_ids,
        output_atom_ids=stats.output_atom_ids, decision_counts=counts,
        latency_ms=extraction.latency_ms + judge_latency, error=error, raw_output=raw_log,
    )
    await repo.mark_job_success(
        user_sub, cursor_at=new_cursor_at, cursor_id=new_cursor_id, processed=len(new_msgs),
        has_more=has_more, scene_name=extraction.scene_name, atoms_added=stats.written,
        default_threshold=DEFAULT_THRESHOLD, idle_seconds=IDLE_SECONDS,
    )
    if stats.written:
        # 전역 지시가 바뀌었을 수 있다 — 안정 블록 캐시를 지워 다음 턴이 다시 만들게 한다.
        from agent_arch.services.memory_v2.recall import invalidate_stable_block
        await invalidate_stable_block(user_sub)
    # 상위층(L2 장면 → L3 페르소나) 예약 — atom 이 실제로 쓰였을 때만. 모순 update 가 있었으면
    # 페르소나를 fresh 로 다시 써서 뒤집힌 지시가 유령처럼 남지 않게 한다.
    if stats.written and not has_more:
        from agent_arch.services.memory_v2.layers import schedule_upper_layers
        schedule_upper_layers(
            user_sub, force_persona=stats.updated > 0, fresh_persona=stats.updated > 0,
        )
    return L1JobResult(
        user_sub=user_sub, status="done" if status == "succeeded" else status,
        processed=len(new_msgs), stats=stats, has_more=has_more,
        scene_name=extraction.scene_name, error=error,
    )
