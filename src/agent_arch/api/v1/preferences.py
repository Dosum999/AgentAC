"""사용자 개인설정 API 라우터.

Redis 대화 이력 정책(보존 메시지 수 + TTL)을 본인 단위로 저장·조회한다.
인증은 라우터 등록부(``app/main.py``)에서 ``Depends(get_current_user)``로 적용한다.

엔드포인트:
    GET    /api/v1/me/preferences           현재 설정 조회 (행 없으면 전역 기본값)
    PUT    /api/v1/me/preferences           merge-patch 갱신
    DELETE /api/v1/me/preferences           행 삭제 → 다음 GET부터 is_default=true
    DELETE /api/v1/me/preferences/history   현재 사용자 대화 컨텍스트 즉시 비우기

PUT은 Redis 대화 키를 건드리지 않는다(side-effect-free). 새 정책은 다음
``append_turn`` 호출 시점부터 자연 적용된다.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, Response, status

from agent_arch.memory.conversation import (
    MAX_HISTORY as DEFAULT_MAX_HISTORY,
    SESSION_TTL as DEFAULT_SESSION_TTL,
    clear_user_conversations,
)
from agent_arch.models.preferences import (
    UserPreferencesResponse,
    UserPreferencesUpdate,
)
from agent_arch.services.preferences import (
    UserPreferences,
    delete_preferences,
    find_preferences,
    invalidate,
    upsert_preferences,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/me/preferences", tags=["preferences"])


def _require_user_sub(request: Request) -> str:
    """``request.state.user["sub"]``를 강제 추출한다.

    Gateway가 X-User-Sub를 검증해 통과시키면 ``get_current_user`` 미들웨어가
    채워두므로 정상 흐름에서는 항상 존재. 누락은 미들웨어 오류로 간주해 401.
    """
    user = getattr(request.state, "user", None) or {}
    sub = user.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="user_sub missing")
    return sub


def _to_response(prefs: UserPreferences) -> UserPreferencesResponse:
    """Repository dataclass를 응답 모델로 변환한다."""
    return UserPreferencesResponse(
        user_sub=prefs.user_sub,
        max_history=prefs.max_history,
        session_ttl_seconds=prefs.session_ttl_seconds,
        is_default=False,
        updated_at=prefs.updated_at.isoformat(),
    )


def _default_response(user_sub: str) -> UserPreferencesResponse:
    """행이 없는 사용자에 대한 read-time fallback 응답."""
    return UserPreferencesResponse(
        user_sub=user_sub,
        max_history=DEFAULT_MAX_HISTORY,
        session_ttl_seconds=DEFAULT_SESSION_TTL,
        is_default=True,
        updated_at=None,
    )


@router.get("", response_model=UserPreferencesResponse)
async def get_preferences(request: Request) -> UserPreferencesResponse:
    """현재 사용자의 개인설정을 조회한다.

    행이 없으면 전역 기본값으로 응답하고 ``is_default=true``를 표시한다
    (FE가 404 분기를 짤 필요가 없도록 read-time fallback).
    """
    user_sub = _require_user_sub(request)
    prefs = await find_preferences(user_sub)
    if prefs is None:
        return _default_response(user_sub)
    return _to_response(prefs)


@router.put("", response_model=UserPreferencesResponse)
async def update_preferences(
    request: Request,
    body: UserPreferencesUpdate,
) -> UserPreferencesResponse:
    """현재 사용자의 개인설정을 부분 갱신한다 (merge-patch).

    누락된 필드는 기존 값을 유지한다. 행이 없으면 신규 생성하며 누락 필드는
    컬럼 DEFAULT(20 / 86400)로 채워진다. Redis 대화 키는 건드리지 않으며
    새 정책은 다음 ``append_turn`` 호출부터 자연 반영된다.

    Raises:
        HTTPException: 500 — DB 미연결로 upsert 실패.
    """
    user_sub = _require_user_sub(request)

    prefs = await upsert_preferences(
        user_sub=user_sub,
        max_history=body.max_history,
        session_ttl_seconds=body.session_ttl_seconds,
    )
    if prefs is None:
        raise HTTPException(status_code=500, detail="preferences upsert failed")

    await invalidate(user_sub)
    return _to_response(prefs)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def delete_preferences_route(request: Request) -> Response:
    """현재 사용자의 개인설정 행을 삭제한다. 다음 GET부터 ``is_default=true`` 복귀."""
    user_sub = _require_user_sub(request)
    await delete_preferences(user_sub)
    await invalidate(user_sub)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
async def clear_user_history(request: Request) -> Response:
    """현재 사용자의 Redis 대화 컨텍스트를 즉시 비운다.

    개인설정 변경 자체는 ``append_turn`` 시점에 자연 반영(side-effect-free)되지만,
    사용자가 "지금 즉시 대화를 비우고 싶다"고 명시할 때를 위한 별도 엔드포인트.
    """
    user_sub = _require_user_sub(request)
    await clear_user_conversations(user_sub)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
