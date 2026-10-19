"""사용자 개인설정 API 요청/응답 모델.

Redis 대화 이력 정책(보존 메시지 수 + TTL)을 사용자 단위로 저장·조회한다.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UserPreferencesUpdate(BaseModel):
    """PUT /api/v1/me/preferences 요청 바디 — merge-patch 시맨틱.

    누락된 필드는 기존 값을 유지한다(RFC 7396 merge-patch 유사).
    프론트엔드가 부분 갱신만 보낼 수 있도록 모든 필드가 optional이다.

    Attributes:
        max_history: 대화 이력 최대 보존 메시지 수. user/assistant 페어 단위로
            트림되므로 짝수만 허용 (2 ≤ N ≤ 100).
        session_ttl_seconds: Redis session:{user_sub} 키 TTL (초).
            10분(600) ~ 7일(604800).
    """

    max_history: Optional[int] = Field(default=None, ge=2, le=100)
    session_ttl_seconds: Optional[int] = Field(default=None, ge=600, le=604800)

    @field_validator("max_history")
    @classmethod
    def _must_be_even(cls, value: Optional[int]) -> Optional[int]:
        """max_history는 user/assistant 페어 단위 트림과 맞추기 위해 짝수만 허용."""
        if value is not None and value % 2 != 0:
            raise ValueError("max_history는 짝수여야 합니다 (user/assistant 페어)")
        return value


class UserPreferencesResponse(BaseModel):
    """GET / PUT /api/v1/me/preferences 응답 바디.

    행이 없는 사용자는 전역 기본값으로 채워서 반환하고 is_default=True를 표시한다.
    프론트엔드가 404 분기 없이 항상 같은 스키마를 받을 수 있도록 read-time fallback.

    Attributes:
        user_sub: Keycloak user sub.
        max_history: 현재 적용 중인 보존 메시지 수.
        session_ttl_seconds: 현재 적용 중인 TTL (초).
        is_default: True면 사용자가 설정한 값이 아니라 전역 기본값을 채워준 것.
        updated_at: 마지막 갱신 시각 (ISO 8601). is_default=True면 None.
    """

    user_sub: str
    max_history: int
    session_ttl_seconds: int
    is_default: bool
    updated_at: Optional[str] = None
