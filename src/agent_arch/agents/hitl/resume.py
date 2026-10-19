"""HITL 재개 로직 — interrupt 로 멈춘 액션을 사람 승인 값으로 이어 실행한다.

HTTP 관심사(인증/소유권/요청·응답)는 ``app/api/v1/agent.py`` 의 엔드포인트가 담당하고,
여기서는 체크포인트 상태 조회 / 서버측 검증 / Command(resume) 재개만 담당한다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional

from langgraph.types import Command

from agent_arch.agents.hitl.graph import REJECT_SENTINEL, get_action_graph, interrupt_payload

logger = logging.getLogger(__name__)

# 승인 재개 잠금. 액션 실행(메일 발송·DM·일정 등록)은 몇 초가 걸리고, 그동안 승인 버튼은
# 살아 있다 — 메신저 카드는 콜백 응답이 와야 버튼이 사라지므로 클라이언트가 막을 방법이
# 없다. 그 틈에 한 번 더 누르면 두 요청이 **같은 interrupt 체크포인트**를 보고 둘 다
# 통과해 액션이 두 번 실행된다(2026-09-18 실측: 승인 콜백 2건이 1초 간격으로 들어와
# mattermost_skill_send_dm 이 두 번 나갔다).
_LOCK_TTL_SECONDS = 180
_LOCK_KEY_PREFIX = "hitl:resume:"
# Redis 가 없을 때의 대비책. HITL 체크포인터가 MemorySaver(단일 프로세스)라 이 잠금만으로도
# 현재 배치에서는 충분하지만, 체크포인터가 프로세스 밖으로 나가면 Redis 쪽이 답이다.
_local_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class ResumeInProgress(RuntimeError):
    """같은 대화의 승인이 이미 처리 중이다. 두 번째 클릭은 여기서 멈춘다."""


@asynccontextmanager
async def resume_guard(conversation_id: str) -> AsyncIterator[None]:
    """한 대화의 승인 재개를 한 번에 하나만 통과시킨다.

    ``load_pending`` 부터 ``resume_action`` 까지를 통째로 감싸야 한다 — 둘 사이가 열려
    있으면 두 요청이 같은 대기 항목을 집어 든다. 잡지 못하면 ``ResumeInProgress`` 를
    올리고, 호출자는 "이미 처리 중"으로 답한다(사용자에게는 두 번째 클릭이 무시된 것으로
    보인다).

    Raises:
        ResumeInProgress: 같은 대화의 승인이 이미 진행 중.
    """
    key = f"{_LOCK_KEY_PREFIX}{conversation_id}"
    token = uuid.uuid4().hex
    redis = None
    try:
        from agent_arch.memory.redis_store import redis_store

        redis = redis_store.redis
    except Exception:  # noqa: BLE001 — Redis 미구성은 잠금 실패가 아니다
        redis = None

    if redis is not None:
        try:
            if not await redis.set(key, token, nx=True, ex=_LOCK_TTL_SECONDS):
                raise ResumeInProgress(conversation_id)
        except ResumeInProgress:
            raise
        except Exception:  # noqa: BLE001 — Redis 장애로 승인을 막지는 않는다
            logger.warning("[hitl] 재개 잠금 Redis 실패 — 프로세스 내 잠금으로 대체", exc_info=True)
            redis = None

    local = _local_locks[conversation_id]
    if local.locked():
        if redis is not None:
            await _release(redis, key, token)
        raise ResumeInProgress(conversation_id)
    async with local:
        try:
            yield
        finally:
            if redis is not None:
                await _release(redis, key, token)


async def _release(redis, key: str, token: str) -> None:
    """내가 건 잠금만 푼다 — TTL 로 풀린 뒤 남이 새로 건 잠금을 지우지 않도록."""
    try:
        if await redis.get(key) in (token, token.encode()):
            await redis.delete(key)
    except Exception:  # noqa: BLE001 — 못 풀어도 TTL 이 걷어간다
        logger.debug("[hitl] 재개 잠금 해제 실패(무시)", exc_info=True)


# ── 승인 대기열 사본(Redis) — 재시작에도 승인 카드가 살아 있게 ─────────────────────────────
# 체크포인터가 MemorySaver(프로세스 메모리)라 에이전트 파드가 재시작되면 그 전에 띄운 승인 카드가 전부 죽는다 —
# 눌러도 load_pending 이 None 이라 "이미 처리된 요청입니다" 만 뜬다(2026-10-13 운영: 수주서 수정 카드를 띄운 뒤
# 다른 배포로 파드가 바뀌었다). 큐를 체크포인트에 넣을 때 같은 큐를 Redis 에 두고, 체크포인트가 비어 있으면
# 그 사본으로 다시 interrupt 상태를 세운다. 실행·거절로 소비된 항목은 사본에서도 빠진다(남은 것만 둔다) —
# 그래서 "실제로 처리된 카드" 는 여전히 "이미 처리된 요청입니다" 가 된다.
_QUEUE_KEY_PREFIX = "hitl:queue:"
_QUEUE_TTL_SECONDS = 3 * 24 * 3600  # 메신저 카드는 며칠 뒤에도 눌린다


def _redis():
    try:
        from agent_arch.memory.redis_store import redis_store

        return redis_store.redis
    except Exception:  # noqa: BLE001 — Redis 미구성이면 사본 없이(종전 동작)
        return None


async def save_queue_snapshot(conversation_id: str, queue: list) -> None:
    """승인 대기 큐(남은 항목)의 사본을 둔다. 비면 지운다. 실패해도 승인 흐름은 막지 않는다."""
    redis = _redis()
    if redis is None:
        return
    key = f"{_QUEUE_KEY_PREFIX}{conversation_id}"
    try:
        if queue:
            await redis.set(key, json.dumps(list(queue), ensure_ascii=False, default=str), ex=_QUEUE_TTL_SECONDS)
        else:
            await redis.delete(key)
    except Exception:  # noqa: BLE001
        logger.warning("[hitl] 승인 대기열 사본 저장 실패(무시)", exc_info=True)


async def _restore_if_lost(graph, conversation_id: str) -> None:
    """체크포인트가 비어 있고(재시작) 사본이 있으면 사본 큐로 interrupt 상태를 다시 세운다."""
    state = await graph.aget_state(_config(conversation_id))
    if state is not None and getattr(state, "next", None):
        return
    redis = _redis()
    if redis is None:
        return
    try:
        raw = await redis.get(f"{_QUEUE_KEY_PREFIX}{conversation_id}")
        queue = json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        logger.warning("[hitl] 승인 대기열 사본 읽기 실패(무시)", exc_info=True)
        return
    if not queue:
        return
    logger.info("[hitl] 체크포인트 유실(재시작) → 사본 큐 %d건으로 복원 conversation=%s", len(queue), conversation_id)
    await graph.ainvoke(
        {"pending_queue": queue, "index": 0, "results": [], "result": None}, _config(conversation_id)
    )


async def _sync_snapshot(graph, conversation_id: str) -> None:
    """재개 뒤 남은 큐로 사본을 맞춘다 — 소비된 항목은 빠지고, 다 끝났으면 지운다."""
    try:
        state = await graph.aget_state(_config(conversation_id))
        values = getattr(state, "values", None) or {}
        queue = values.get("pending_queue") or []
        idx = values.get("index", 0) or 0
        remaining = queue[idx:] if (state is not None and getattr(state, "next", None)) else []
    except Exception:  # noqa: BLE001
        logger.warning("[hitl] 승인 대기열 상태 확인 실패 — 사본을 지운다", exc_info=True)
        remaining = []
    await save_queue_snapshot(conversation_id, remaining)


def _config(conversation_id: str, metadata: Optional[dict] = None) -> dict:
    config: dict = {"configurable": {"thread_id": conversation_id}}
    if metadata:
        # LangSmith run metadata 로 기록된다. resume 그래프 run 에 회사/사용자를 부착해
        # 회사별 LLM 사용량 집계에서 "미매핑"으로 빠지지 않도록 한다.
        config["metadata"] = {k: v for k, v in metadata.items() if v is not None}
    return config


async def load_pending(conversation_id: str) -> Optional[dict]:
    """thread 가 interrupt(승인대기) 상태면 '현재' 승인 대기 항목(큐 메타 포함)을 반환한다.

    멀티-액션 큐에서는 현재 인덱스 항목을 돌려준다. None 인 경우: 체크포인트 없음(초안
    단계 미경유), 이미 재개되어 소비됨, 또는 큐 소진.
    """
    graph = get_action_graph()
    await _restore_if_lost(graph, conversation_id)
    state = await graph.aget_state(_config(conversation_id))
    # state.next 가 비어 있으면 재개할 노드가 없음 = 미interrupt 이거나 이미 완료/소비.
    if state is None or not getattr(state, "next", None):
        return None
    values = getattr(state, "values", None) or {}
    queue = values.get("pending_queue")
    if not queue:
        pa = values.get("pending_action")
        queue = [pa] if pa else []
    idx = values.get("index", 0)
    if not queue or idx >= len(queue):
        return None
    item = dict(queue[idx])
    item.update(queue_index=idx, queue_total=len(queue), is_last=idx == len(queue) - 1)
    return item


def validate_resume_values(pending: dict, values: dict) -> None:
    """승인 값이 체크포인트 draft 의 허용 필드 안에 있는지 검증한다.

    도구 자체는 체크포인트(state)에서만 읽으므로 위조 불가하다. 인자는 클라이언트가
    수정해 보낼 수 있으나, 초안/폼에 선언되지 않은 필드를 주입해 도구에 예상 밖 kwarg 가
    전달되는 것을 차단한다(defense-in-depth).

    Raises:
        ValueError: 허용되지 않은 필드가 포함된 경우.
    """
    allowed = set((pending.get("arguments") or {}).keys())
    form = (pending.get("draft_view") or {}).get("form") or {}
    allowed |= set((form.get("values") or {}).keys())
    extra = set(values or {}) - allowed - {"confirm"}
    if extra:
        raise ValueError(f"승인 값에 허용되지 않은 필드가 있습니다: {sorted(extra)}")


async def resume_action(
    conversation_id: str,
    values: dict,
    *,
    reject: bool = False,
    user_info: Optional[dict] = None,
) -> dict:
    """현재 액션을 재개한다 — 승인이면 confirm=True 실행, 거절이면 그 액션만 스킵.

    실행/스킵 후 큐에 다음 항목이 남아 있으면 그 draft 페이로드를 ``next_pending`` 으로 함께
    반환한다(없으면 None=마지막). 호출 전 호출자가 ``set_user_context`` 로 사용자 명의를
    설정해야 도구가 본인 명의로 실행된다(``workspace.py:_commit`` 와 동일).

    Returns:
        {"success": bool, "view"?, "error"?, "rejected"?, "next_pending": dict|None}
    """
    graph = get_action_graph()
    # ⚠ 빈 dict 로 재개하면 안 된다 — langgraph 의 Command(resume={}) 는 falsy 라
    #   interrupt 노드가 재실행되지 않아 "값 무수정 승인"이 전부 무동작(502)이 된다
    #   (2026-08-13 accounting E2E 실측, af0cc87 — 이슈 revert 로 유실된 픽스 재적용).
    #   confirm 은 validate_resume_values 허용 키이고 실행부(_execute_one)가 pop 하므로
    #   진공 값으로 안전하다.
    resume_value = REJECT_SENTINEL if reject else (values or {"confirm": True})
    # orchestrator._preprocess_request 와 동일하게 트레이스에 회사/사용자를 부착한다.
    # 미부착 시 resume 그래프 run 이 company 없는 root 로 기록되어 사용량 대시보드에서
    # "미매핑"으로 집계된다.
    metadata = None
    if user_info:
        metadata = {
            "user_id": user_info.get("sub"),
            "user_email": user_info.get("email"),
            "company": user_info.get("company"),
            "session_id": user_info.get("session_id"),
        }
    await _restore_if_lost(graph, conversation_id)
    final = await graph.ainvoke(
        Command(resume=resume_value), _config(conversation_id, metadata)
    )
    await _sync_snapshot(graph, conversation_id)
    result = (final.get("result") if isinstance(final, dict) else None) or {
        "success": False, "error": "재개 결과가 없습니다."
    }
    # 큐에 다음 항목이 남아 다시 interrupt 했으면 그 페이로드(다음 draft) 반환.
    next_pending = interrupt_payload(final)
    return {**result, "next_pending": next_pending}
