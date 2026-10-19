"""상위층(L2 장면 → L3 페르소나) 실행 규칙과 예약.

L1 잡이 atom 을 썼을 때 10초 뒤 돈다. L2 는 직전 실행에서 15분 이상 지났을 때만(배치가 몰려도 LLM
호출이 폭주하지 않게), L3 는 페르소나가 없는데 atom 이 5개 이상이거나 50개가 더 쌓였을 때, 또는
강제(forget·모순 update 뒤 fresh)일 때. 유저 락으로 겹침을 막는다.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from agent_arch.db.repository import memory_v2 as repo
from agent_arch.memory.redis_store import redis_store
from agent_arch.services.memory_v2 import persona as persona_mod
from agent_arch.services.memory_v2 import scenes as scenes_mod

logger = logging.getLogger(__name__)

DELAY_AFTER_L1_SECONDS = 10.0
SCENES_MIN_INTERVAL = timedelta(minutes=15)
LOCK_TTL_SECONDS = 300


def scenes_due(last_l2_at: Optional[datetime], now: Optional[datetime] = None) -> bool:
    if last_l2_at is None:
        return True
    # last_l2_at 은 DB NOW()(세션 UTC)의 naive 값 — 비교 기준도 naive UTC 여야 한다. datetime.now()
    # 는 컨테이너 로컬(NKS 는 KST)이라 9시간 차이로 늘 '주기 지남'이 된다.
    return (now or datetime.now(timezone.utc).replace(tzinfo=None)) - last_l2_at >= SCENES_MIN_INTERVAL


def persona_due(cp: dict, atom_count: int, has_persona: bool) -> bool:
    if atom_count < persona_mod.PERSONA_MIN_ATOMS:
        return False
    if not has_persona:
        return True
    return int(cp.get("atoms_since_persona") or 0) >= persona_mod.PERSONA_REBUILD_EVERY


MAX_RERUNS = 3


def _rerun_key(user_sub: str) -> str:
    return f"memory2:upper:rerun:{user_sub}"


async def _mark_rerun(user_sub: str, want: dict) -> None:
    """락에 걸린 요청을 버리지 않고 '끝나면 한 번 더'로 남긴다(강제·fresh 플래그는 OR 로 합친다).

    한 턴에 기억을 여러 개 지우면 forget 마다 상위층이 예약되는데, 락 때문에 첫 실행만 돌고 나머지는 버려졌다.
    첫 실행이 뒤따른 삭제보다 먼저 시작하면 그 기억을 쓴 장면·페르소나가 고쳐지지 않는다.
    """
    try:
        prev = await redis_store.get(_rerun_key(user_sub)) or {}
        merged = {k: bool(prev.get(k)) or bool(want.get(k)) for k in ("force_scenes", "force_persona", "fresh_persona")}
        merged["company_scope"] = want.get("company_scope") or prev.get("company_scope")
        await redis_store.set(_rerun_key(user_sub), merged, expire=LOCK_TTL_SECONDS)
    except Exception:
        logger.debug("memory_v2 상위층 재실행 표시 실패(무시)", exc_info=True)


async def _take_rerun(user_sub: str) -> Optional[dict]:
    try:
        pending = await redis_store.getdel(_rerun_key(user_sub))
    except Exception:
        return None
    return pending if isinstance(pending, dict) else None


async def run_upper_layers(
    user_sub: str,
    *,
    force_scenes: bool = False,
    force_persona: bool = False,
    fresh_persona: bool = False,
    company_scope: Optional[str] = None,
) -> dict:
    """L2 → L3 를 규칙에 따라 실행한다. 반환: {"scenes": ScenesResult|None, "persona": str|None|"skipped"}.

    락에 걸리면 재실행 표시만 남기고 끝낸다 — 락 보유자가 한 번 더 돈다(최대 MAX_RERUNS).
    """
    key = f"memory2:lock:upper:{user_sub}"
    want = {"force_scenes": force_scenes, "force_persona": force_persona,
            "fresh_persona": fresh_persona, "company_scope": company_scope}
    if not await redis_store.set_nx(key, "1", expire=LOCK_TTL_SECONDS):
        await _mark_rerun(user_sub, want)
        return {"scenes": None, "persona": "locked"}
    out: dict = {"scenes": None, "persona": "skipped"}
    try:
        for _ in range(MAX_RERUNS + 1):
            out = await _run_once(user_sub, **want)
            pending = await _take_rerun(user_sub)
            if not pending:
                break
            want = {**want, **pending}
    finally:
        try:
            await redis_store.delete(key)
        except Exception:
            pass
    # 마지막 확인과 락 해제 사이에 들어온 표시 — 새로 예약해 락을 다시 잡게 한다.
    late = await _take_rerun(user_sub)
    if late:
        schedule_upper_layers(user_sub, delay=0, **late)
    return out


async def _run_once(
    user_sub: str,
    *,
    force_scenes: bool = False,
    force_persona: bool = False,
    fresh_persona: bool = False,
    company_scope: Optional[str] = None,
) -> dict:
    out: dict = {"scenes": None, "persona": "skipped"}
    try:
        cp = await repo.get_checkpoint(user_sub) or {}
        # 지워진 기억을 근거로 쓴 장면부터 고친다 — 장면 갱신은 새 atom 만 보고, 페르소나는 장면 본문을 읽으므로
        # 이걸 먼저 하지 않으면 지운 내용이 장면을 거쳐 페르소나로 돌아온다. 대상 장면이 없으면 쿼리 두 번이다.
        repair = await scenes_mod.repair_scenes(user_sub)
        out["repair"] = repair
        if force_scenes or scenes_due(cp.get("last_l2_at")):
            res = await scenes_mod.rebuild_scenes(user_sub, since=cp.get("last_l2_at"), company_scope=company_scope)
            out["scenes"] = res
            if not res.error:
                await repo.mark_l2_done(user_sub)
        atom_count = await repo.count_active_atoms(user_sub)
        has_persona = (await repo.get_active_persona(user_sub, company_scope)) is not None
        if force_persona and fresh_persona:
            # fresh(기억을 지웠거나 모순 update) 는 **모든 범위**의 페르소나를 다시 쓴다. 회상은 회사 범위 본을 먼저
            # 읽는데, 삭제 경로는 범위를 모른 채 예약돼 범위 없는 본만 다시 써졌다 — 회사 범위 본에 지운 규칙이 남아
            # 계속 적용됐다(실측 2026-09-29 로컬: '— 에이전트 드림' 삭제 뒤에도 테스트 테넌트 범위 페르소나 v2 가 그대로).
            scopes = list(dict.fromkeys([company_scope, *await repo.list_active_persona_scopes(user_sub)]))
            for scope in scopes:
                out["persona"] = await persona_mod.rebuild_persona(user_sub, fresh=True, company_scope=scope)
        elif force_persona or persona_due(cp, atom_count, has_persona):
            out["persona"] = await persona_mod.rebuild_persona(user_sub, fresh=fresh_persona, company_scope=company_scope)
        sc = out.get("scenes")
        if (sc is not None and sc.changed) or repair.changed or isinstance(out.get("persona"), str):
            from agent_arch.services.memory_v2.recall import invalidate_stable_block
            await invalidate_stable_block(user_sub)
    except Exception:
        logger.exception("memory_v2 상위층 실행 실패(user=%s)", user_sub[:8])
    return out


def schedule_upper_layers(user_sub: str, *, delay: float = DELAY_AFTER_L1_SECONDS, **kwargs) -> Optional[asyncio.Task]:
    """백그라운드로 run_upper_layers 를 예약한다(이벤트 루프 없으면 None)."""
    async def _later():
        if delay > 0:
            await asyncio.sleep(delay)
        await run_upper_layers(user_sub, **kwargs)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None
    return loop.create_task(_later())
