"""내 장기 기억 셀프서비스 API (/api/v1/me/memory).

무엇을 기억하고 있는지 보고, 잘못된 기억 하나를 지우거나 전부 초기화한다. 자동 추출은 완벽할 수
없으므로 사용자가 직접 교정할 길이 있어야 한다 — 기억의 가치는 정확도에 달려 있다.

메모리 v2(MEMORY_V2_MODE=shadow|active)가 켜져 있을 때만 동작한다. 인증은 main.py 의
get_current_user 의존성이 걸고, 소유자 격리는 request.state.user["sub"] 로 강제한다.
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, HTTPException, Query, Request, Response, status

from agent_arch.services.memory_v2 import admin as memory_admin
from agent_arch.services.memory_v2 import capture as memory_flags
from agent_arch.services.memory_v2 import explicit as memory_explicit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/me/memory", tags=["memory"])


def _require_user_sub(request: Request) -> str:
    """``request.state.user["sub"]`` 강제 추출(conversation/preferences 라우터와 동일 패턴)."""
    user = getattr(request.state, "user", None) or {}
    sub = user.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="user_sub missing")
    return sub


def _require_enabled() -> None:
    if not memory_flags.enabled():
        raise HTTPException(status_code=409, detail="장기 메모리 v2 가 비활성 상태입니다(MEMORY_V2_MODE).")


@router.get("")
async def get_my_memory(
    request: Request,
    limit: int = Query(100, ge=1, le=500, description="atom 최대 개수"),
) -> dict:
    """내 기억 전체: atom(우선순위순) · 장면 · 페르소나 · 처리 상태 · 최근 생성 로그."""
    sub = _require_user_sub(request)
    _require_enabled()
    return await memory_admin.export_user_memory(sub, limit=limit)


@router.delete("/atoms/{atom_id}", status_code=status.HTTP_204_NO_CONTENT)
async def forget_my_atom(request: Request, atom_id: str) -> Response:
    """기억 1건 삭제(soft). 없거나 이미 지워졌으면 404, 비-UUID 는 400. 페르소나·장면은 fresh 로 재작성된다."""
    sub = _require_user_sub(request)
    _require_enabled()
    try:
        uuid.UUID(atom_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="atom_id 가 UUID 가 아닙니다.")
    removed = await memory_explicit.forget_atom(sub, atom_id)
    if removed is None:
        raise HTTPException(status_code=404, detail="해당 기억이 없거나 이미 지워졌습니다.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("")
async def reset_my_memory(request: Request) -> dict:
    """내 기억 전부 초기화(atom·장면·페르소나 soft-delete + 벡터 인덱스 삭제 + 처리 상태 리셋). 건수 반환."""
    sub = _require_user_sub(request)
    _require_enabled()
    return await memory_admin.reset_user_memory(sub)
