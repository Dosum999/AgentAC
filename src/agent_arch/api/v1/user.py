"""사용자 프로필 등록 API 라우터.

ui-components가 호스트 프론트의 ``get_current_user`` 응답을 agent-backend에
동기 등록한다. 본 엔드포인트가 가장 먼저 호출되어야 후속 ``/agent/*``
요청에서 ``auth.get_current_user`` 미들웨어가 Redis profile을 사용할 수 있다.

검증 정책:
    - Gateway가 주입한 ``X-User-Sub`` ↔ body의 ``user_sub`` 일치 (403 if mismatch)
    - ``profile.user``가 이메일 문자열이거나 email/id가 있는 REST 사용자 객체
      (422 if missing)
    - profile 내부 권한 필드는 호스트 프론트 신뢰 — 별도 검증 없음
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request, status

from agent_arch.models.user_profile import (
    UserProfileRegisterRequest,
    UserProfileRegisterResponse,
)
from agent_arch.services.user_profile import (
    InvalidUserProfileError,
    normalize_user_profile,
    upsert_profile,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/user", tags=["user"])


@router.post("/profile", response_model=UserProfileRegisterResponse)
async def register_user_profile(
    body: UserProfileRegisterRequest,
    request: Request,
) -> UserProfileRegisterResponse:
    """ui-components가 호스트 프론트의 get_current_user 결과를 등록한다.

    호출 시점: 호스트 프론트에서 ``get_current_user``가 호출/갱신될 때마다
    (마운트, 회사 전환, 권한 변경 등). 백엔드는 동일 profile_hash인 경우
    DB write를 생략하므로 멱등하게 여러 번 호출해도 안전하다.

    Raises:
        HTTPException 401: ``X-User-Sub`` 헤더 누락.
        HTTPException 403: ``X-User-Sub`` ≠ ``body.user_sub``.
        HTTPException 422: ``profile.user``의 사용자 식별자 누락.
    """
    gateway_sub = request.headers.get("X-User-Sub")
    if not gateway_sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Sub header",
        )
    if gateway_sub != body.user_sub:
        # 보안: 다른 사용자의 sub를 body에 넣어 등록하려는 시도 차단
        logger.warning(
            "[user_profile] X-User-Sub mismatch — gateway=%s body=%s",
            gateway_sub, body.user_sub,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="user_sub mismatch with X-User-Sub",
        )

    try:
        profile = normalize_user_profile(body.profile)
    except InvalidUserProfileError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    updated = await upsert_profile(body.user_sub, profile)

    # PII 보호: profile 본문은 절대 로그에 출력하지 않음 — sub와 is_guest만 기록.
    logger.info(
        "[user_profile] registered user_sub=%s is_guest=%s updated=%s",
        body.user_sub,
        bool(profile.get("is_guest")),
        updated,
    )

    return UserProfileRegisterResponse(success=True, updated=updated)
