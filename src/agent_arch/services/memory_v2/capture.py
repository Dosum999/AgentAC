"""캡처 훅 — 턴 저장 직후 체크포인트만 건드린다(LLM 없음, 사용자 대기 없음).

save_logged_in_turn 이 PG INSERT 에 성공하면 on_turn_saved 가 fire-and-forget 으로
pending +1 / next_due 갱신을 한다. 추출은 스케줄러(scheduler.py)가 배치로 돈다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

IDLE_SECONDS = 600.0        # 마지막 캡처 후 이만큼 조용하면 배치 실행
DEFAULT_THRESHOLD = 5       # 웜업 졸업 후 N턴마다
# 이 마커가 있으면 배치를 기다리지 않고 즉시 due — "기억해/앞으로 …" 는 다음 대화방에서 바로 쓰인다.
IMMEDIATE_MARKERS: tuple[str, ...] = (
    "기억해", "기억하", "잊지", "앞으로", "항상", "매번", "무조건", "절대",
    "쓰지 마", "쓰지마", "하지 마", "하지마", "말고", "대신",
)


def mode() -> str:
    return str(getattr(settings, "memory_v2_mode", "off") or "off").strip().lower()


def enabled() -> bool:
    """v2 쓰기 경로(캡처·추출·저장)가 켜져 있나. shadow 와 active 둘 다 True."""
    return bool(settings.enable_user_memory) and mode() in ("shadow", "active")


def is_active() -> bool:
    """회상·도구가 v2 를 보는 단계인가(v1 추출은 이때 멈춘다)."""
    return bool(settings.enable_user_memory) and mode() == "active"


def is_immediate(user_input: Optional[str]) -> bool:
    text = (user_input or "").strip()
    return bool(text) and any(m in text for m in IMMEDIATE_MARKERS)


async def bump(user_sub: str, message_id: str, user_input: Optional[str]) -> Optional[dict]:
    """체크포인트 반영(await 버전). 실패는 경고만 — 메모리는 턴을 깨지 않는다."""
    from agent_arch.db.repository.memory_v2 import bump_pending
    try:
        row = await bump_pending(
            user_sub, message_id,
            immediate=is_immediate(user_input),
            idle_seconds=IDLE_SECONDS,
            default_threshold=DEFAULT_THRESHOLD,
        )
        if row:
            logger.debug(
                "memory_v2 capture user=%s pending=%s warmup=%s due=%s",
                user_sub[:8], row.get("pending_count"), row.get("warmup_threshold"), row.get("next_due_at"),
            )
        return row
    except Exception as e:
        logger.warning("memory_v2 capture 실패(user=%s): %s: %s", user_sub[:8], type(e).__name__, e)
        return None


def on_turn_saved(user_sub: Optional[str], message_id: Optional[str], user_input: Optional[str]) -> bool:
    """save_logged_in_turn 훅. 게이트 통과 시 백그라운드 태스크를 띄우고 True."""
    if not enabled() or not user_sub or not message_id:
        return False
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return False
    loop.create_task(bump(user_sub, message_id, user_input))
    return True
