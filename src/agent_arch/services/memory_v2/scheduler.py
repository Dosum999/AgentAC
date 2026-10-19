"""메모리 v2 스케줄러 — lifespan 백그라운드 루프. due 유저를 집어 L1 잡을 돌린다.

폴링(30초) + PG next_due_at 기반이라 재시작에도 예약이 살아 있다. 레플리카가 여럿이어도
잡은 유저 락(Redis) 아래에서 돌고 커서는 단조라 정합성이 유지된다(중복 실행은 skipped_locked).
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from agent_arch.db.repository.memory_v2 import list_due_users
from agent_arch.services.memory_v2.pipeline import run_l1_job

logger = logging.getLogger(__name__)

POLL_SECONDS = 30.0
MAX_CONCURRENT = 3
BATCH_USERS = 10
SYNC_EVERY_TICKS = 10        # ≈5분마다 Qdrant 미동기 atom 을 밀어 넣는다(임베딩 실패분 복구)


class MemoryV2Scheduler:
    def __init__(self, *, poll_seconds: float = POLL_SECONDS) -> None:
        self.poll_seconds = poll_seconds
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        self._ticks = 0

    def start(self) -> asyncio.Task:
        self._stopping = False
        self._task = asyncio.create_task(self.run(), name="memory-v2-scheduler")
        return self._task

    async def stop(self) -> None:
        self._stopping = True
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    async def run(self) -> None:
        logger.info("memory_v2 스케줄러 시작(poll=%.0fs, concurrency=%d)", self.poll_seconds, MAX_CONCURRENT)
        while not self._stopping:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("memory_v2 스케줄러 tick 실패")
            await asyncio.sleep(self.poll_seconds)

    async def tick(self) -> int:
        """due 유저를 최대 BATCH_USERS 명 처리한다. 처리한 유저 수 반환. 주기적으로 인덱스 잔여분도 동기화."""
        self._ticks += 1
        if self._ticks % SYNC_EVERY_TICKS == 0:
            await self._sync_index()
        users = await list_due_users(BATCH_USERS)
        if not users:
            return 0
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def _one(u: str) -> None:
            async with sem:
                await run_l1_job(u)

        await asyncio.gather(*[_one(u) for u in users], return_exceptions=True)
        return len(users)

    async def _sync_index(self) -> int:
        """embedding_indexed=false 로 남은 atom(임베딩 일시 실패분)을 Qdrant 에 밀어 넣는다. 실패는 로그만."""
        try:
            from agent_arch.services.memory_v2.indexer import get_v2_indexer
            n = await (await get_v2_indexer()).sync_pending(200)
            if n:
                logger.info("memory_v2 인덱스 잔여 동기화 %d건", n)
            return n
        except Exception as e:
            logger.warning("memory_v2 인덱스 잔여 동기화 실패: %s: %s", type(e).__name__, e)
            return 0


scheduler = MemoryV2Scheduler()
