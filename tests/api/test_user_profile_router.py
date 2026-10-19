"""POST /user/profile 라우터 단위 테스트.

검증 항목:
    - 정상 등록 → 200 + updated 플래그 전달
    - X-User-Sub 헤더 누락 → 401
    - X-User-Sub ↔ body.user_sub 불일치 → 403
    - profile.user 누락 → 422
    - 동일 profile 재등록 → updated=False
    - is_guest=true profile → 차단 없이 등록
"""
from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from agent_arch.api.v1 import user as user_router
from agent_arch.models.user_profile import UserProfileRegisterRequest


def _make_request(headers: Dict[str, str]) -> MagicMock:
    """FastAPI Request mock — headers.get만 흉내."""
    req = MagicMock()
    req.headers = headers
    return req


@pytest.fixture
def upsert_mock(monkeypatch):
    """agent_arch.api.v1.user.upsert_profile를 캡처용 mock으로 교체.

    Returns:
        호출 기록 dict({"calls": [...], "result": bool}). result는 기본 True.
    """
    state: Dict[str, Any] = {"calls": [], "result": True}

    async def _upsert(user_sub: str, profile: dict) -> bool:
        state["calls"].append({"user_sub": user_sub, "profile": profile})
        return state["result"]

    monkeypatch.setattr(user_router, "upsert_profile", _upsert)
    return state


def _profile(user_email: Any = "user@example.com", **overrides: Any) -> dict:
    base = {
        "user": user_email,
        "full_name": "테스트",
        "is_guest": False,
        "company": "ACME",
        "companies": ["ACME"],
    }
    base.update(overrides)
    if user_email is None:
        base.pop("user")
    return base


class TestRegisterUserProfile:
    """register_user_profile 핸들러의 검증/위임 동작."""

    @pytest.mark.asyncio
    async def test_success(self, upsert_mock):
        """정상 케이스: X-User-Sub 일치 + profile.user 존재 → success/updated."""
        body = UserProfileRegisterRequest(user_sub="u-1", profile=_profile())
        request = _make_request({"X-User-Sub": "u-1"})

        resp = await user_router.register_user_profile(body, request)

        assert resp.success is True
        assert resp.updated is True
        assert upsert_mock["calls"] == [{"user_sub": "u-1", "profile": body.profile}]

    @pytest.mark.asyncio
    async def test_returns_updated_false_when_same_profile(self, upsert_mock):
        """upsert가 False를 반환하면 응답도 updated=False."""
        upsert_mock["result"] = False
        body = UserProfileRegisterRequest(user_sub="u-1", profile=_profile())
        request = _make_request({"X-User-Sub": "u-1"})

        resp = await user_router.register_user_profile(body, request)

        assert resp.success is True
        assert resp.updated is False

    @pytest.mark.asyncio
    async def test_guest_profile_allowed(self, upsert_mock):
        """is_guest=true도 차단 없이 등록된다 (꼬리표 정책)."""
        body = UserProfileRegisterRequest(
            user_sub="u-guest",
            profile=_profile(is_guest=True),
        )
        request = _make_request({"X-User-Sub": "u-guest"})

        resp = await user_router.register_user_profile(body, request)

        assert resp.success is True
        assert upsert_mock["calls"][0]["profile"]["is_guest"] is True

    @pytest.mark.asyncio
    async def test_rest_v1_profile_is_normalized_before_upsert(self, upsert_mock):
        body = UserProfileRegisterRequest(
            user_sub="u-rest",
            profile={
                "user": {
                    "id": "jpark@example.com",
                    "email": "jpark@example.com",
                    "full_name": "박정훈",
                    "image": "",
                },
                "selected_company_id": "ACME",
                "employee": None,
                "companies": [],
                "authorization": {
                    "company_id": "ACME",
                    "role_type": "EMPLOYEE",
                    "permissions": {"eapproval": ["eapproval.access"]},
                    "service_access": [],
                    "role_assignments": [
                        {"id": "ROLE-1", "label": "전자결재 담당자"}
                    ],
                },
            },
        )
        request = _make_request({"X-User-Sub": "u-rest"})

        resp = await user_router.register_user_profile(body, request)

        assert resp.success is True
        saved = upsert_mock["calls"][0]["profile"]
        assert saved["user"] == "jpark@example.com"
        assert saved["full_name"] == "박정훈"
        assert saved["company"] == "ACME"
        assert saved["permissions"] == {"eapproval": ["eapproval.access"]}
        assert saved["assigned_roles"] == ["전자결재 담당자"]

    @pytest.mark.asyncio
    async def test_missing_x_user_sub_raises_401(self, upsert_mock):
        body = UserProfileRegisterRequest(user_sub="u-1", profile=_profile())
        request = _make_request({})  # X-User-Sub 누락

        with pytest.raises(HTTPException) as exc:
            await user_router.register_user_profile(body, request)

        assert exc.value.status_code == 401
        assert upsert_mock["calls"] == []  # 서비스 호출 없음

    @pytest.mark.asyncio
    async def test_sub_mismatch_raises_403(self, upsert_mock):
        """다른 사용자 sub를 body에 끼워 보내는 시도는 차단."""
        body = UserProfileRegisterRequest(user_sub="u-victim", profile=_profile())
        request = _make_request({"X-User-Sub": "u-attacker"})

        with pytest.raises(HTTPException) as exc:
            await user_router.register_user_profile(body, request)

        assert exc.value.status_code == 403
        assert upsert_mock["calls"] == []

    @pytest.mark.asyncio
    async def test_missing_profile_user_raises_422(self, upsert_mock):
        body = UserProfileRegisterRequest(user_sub="u-1", profile=_profile(user_email=None))
        request = _make_request({"X-User-Sub": "u-1"})

        with pytest.raises(HTTPException) as exc:
            await user_router.register_user_profile(body, request)

        assert exc.value.status_code == 422
        assert upsert_mock["calls"] == []

    @pytest.mark.asyncio
    async def test_empty_profile_user_string_raises_422(self, upsert_mock):
        """profile.user가 빈 문자열인 경우도 거부 (falsy 검증)."""
        body = UserProfileRegisterRequest(user_sub="u-1", profile=_profile(user_email=""))
        request = _make_request({"X-User-Sub": "u-1"})

        with pytest.raises(HTTPException) as exc:
            await user_router.register_user_profile(body, request)

        assert exc.value.status_code == 422

    @pytest.mark.asyncio
    async def test_rest_v1_profile_without_user_identity_raises_422(self, upsert_mock):
        body = UserProfileRegisterRequest(
            user_sub="u-1",
            profile=_profile(user_email={"id": "", "email": ""}),
        )
        request = _make_request({"X-User-Sub": "u-1"})

        with pytest.raises(HTTPException) as exc:
            await user_router.register_user_profile(body, request)

        assert exc.value.status_code == 422
        assert "profile.user.email" in exc.value.detail
        assert upsert_mock["calls"] == []
