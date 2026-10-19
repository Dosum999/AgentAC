"""사용자 프로필 등록 API 요청/응답 모델.

UI 컴포넌트가 호스트 프론트의 ID 공급자 사용자 조회 응답을
``POST /user/profile``로 등록할 때 사용한다.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfileRegisterRequest(BaseModel):
    """POST /user/profile 요청 바디.

    ``profile``은 레거시 사용자 조회 응답과 REST v1 ``MyContext``를
    함께 받기 위해 의도적으로 ``dict``로 둔다. 라우터에서 두 계약을 Agent 내부
    평면 프로필로 정규화하고 핵심 사용자 식별자를 검증한다.

    Attributes:
        user_sub: 호스트 프론트가 보유한 Keycloak sub (UUID).
            Gateway X-User-Sub와 일치해야 한다.
        profile: 레거시 또는 REST v1 사용자 context 응답.
    """

    user_sub: str = Field(
        ...,
        description="Keycloak sub (UUID). Gateway X-User-Sub와 일치해야 함",
    )
    profile: dict = Field(
        ...,
        description="레거시 사용자 조회 응답 또는 REST v1 MyContext (loose schema)",
    )


class UserProfileRegisterResponse(BaseModel):
    """POST /user/profile 응답 바디.

    Attributes:
        success: 등록 처리 성공 여부 (현 구현에서는 항상 True; 실패 케이스는 HTTPException).
        updated: DB가 실제로 갱신되었는지. False는 "동일 profile_hash → DB write 생략,
            Redis만 갱신"을 의미한다.
    """

    success: bool
    updated: bool = Field(
        ...,
        description="DB 변경 발생 여부 (False = profile 동일, Redis만 갱신)",
    )
