"""auth.py 듀얼 모드 (Redis profile 우선 + 헤더 fallback) 단위 테스트.

검증 항목:
    - X-User-Sub 누락 → 401
    - Redis profile hit → user dict가 Frappe profile에서 어댑팅됨 (is_guest 포함)
    - Redis profile miss → 헤더 기반 fallback으로 user dict 구성 (is_guest 미포함)
    - Redis/DB 오류 → fallback이 흡수해 401 대신 헤더로 동작
    - X-Session-Id는 두 경로 모두 헤더에서 그대로 흘러감
    - 선택 회사는 캐시 hit 여도 X-Forwarded-Company(Gateway 세션=SOT)를 따름
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock
from urllib.parse import quote

import pytest
from fastapi import HTTPException

from agent_arch.middleware import auth as auth_module


def _make_request(
    headers: Optional[Dict[str, str]] = None,
) -> MagicMock:
    """FastAPI Request mock — headers + state를 흉내.

    state는 dict가 아닌 SimpleNamespace-like attribute access여야 한다.
    """
    req = MagicMock()
    req.headers = headers or {}
    # state는 attr 할당이 가능해야 함 (request.state.user = ...)
    req.state = MagicMock()
    return req


@pytest.fixture
def cache_mock(monkeypatch):
    """auth_module.get_profile_from_cache를 캡처용 mock으로 교체.

    Returns:
        {"return_value": dict | None, "raise_exc": Exception | None, "calls": [...]}
    """
    state: Dict[str, Any] = {"return_value": None, "raise_exc": None, "calls": []}

    async def _get(user_sub: str):
        state["calls"].append(user_sub)
        if state["raise_exc"] is not None:
            raise state["raise_exc"]
        return state["return_value"]

    monkeypatch.setattr(auth_module, "get_profile_from_cache", _get)
    return state


class TestGetCurrentUserDualMode:
    """get_current_user의 Redis hit / miss / 오류 경로."""

    @pytest.mark.asyncio
    async def test_missing_x_user_sub_raises_401(self, cache_mock):
        request = _make_request(headers={})

        with pytest.raises(HTTPException) as exc:
            await auth_module.get_current_user(request)

        assert exc.value.status_code == 401
        assert cache_mock["calls"] == []  # Redis 조회 안 함

    @pytest.mark.asyncio
    async def test_redis_hit_uses_profile(self, cache_mock):
        """Redis hit: Frappe profile에서 user dict가 어댑팅된다."""
        cache_mock["return_value"] = {
            "user": "user@example.com",
            "full_name": "길동 홍",  # 정규화로 "홍길동"
            "is_guest": False,
            "company": "ACME",
            "companies": ["ACME", "ACME_HK"],
            "role_type": "EMPLOYEE",
            "employee": {
                "department": "ENG",
                "designation": "Engineer",
                "employee_number": "E-1234",
            },
        }
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-Session-Id": "sess-abc",
        })

        user = await auth_module.get_current_user(request)

        assert user["sub"] == "u-1"
        assert user["email"] == "user@example.com"
        assert user["username"] == "user@example.com"
        assert user["name"] == "홍길동"  # 한국 이름 정규화
        assert user["company"] == "ACME"
        assert user["companies"] == ["ACME", "ACME_HK"]
        assert user["role_type"] == "EMPLOYEE"
        assert user["employee"]["department"] == "ENG"
        assert "roles" not in user  # 권한 raw는 의도적으로 노출 안 함
        assert user["is_guest"] is False
        assert request.state.user is user
        assert request.state.company == "ACME"
        assert request.state.session_id == "sess-abc"

    @pytest.mark.asyncio
    async def test_redis_hit_with_guest_profile(self, cache_mock):
        cache_mock["return_value"] = {
            "user": "guest@example.com",
            "full_name": "Guest",
            "is_guest": True,
        }
        request = _make_request(headers={"X-User-Sub": "u-guest"})

        user = await auth_module.get_current_user(request)

        assert user["is_guest"] is True

    @pytest.mark.asyncio
    async def test_redis_miss_falls_back_to_headers(self, cache_mock):
        """Redis miss: 헤더 기반 fallback. is_guest 키는 없음(헤더에서 추출 안 함)."""
        cache_mock["return_value"] = None  # miss
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-User-Name": "John Doe",
            "X-User-Username": "user@example.com",
            "X-Forwarded-User": "user@example.com",
            "X-Forwarded-Company": "ACME",
            "X-Session-Id": "sess-abc",
        })

        user = await auth_module.get_current_user(request)

        assert user["sub"] == "u-1"
        assert user["email"] == "user@example.com"
        assert user["name"] == "John Doe"  # 한국 이름 아니므로 그대로
        assert user["company"] == "ACME"
        assert "is_guest" not in user  # 헤더 fallback에선 채워지지 않음
        assert request.state.session_id == "sess-abc"

    @pytest.mark.asyncio
    async def test_redis_error_falls_back_to_headers(self, cache_mock):
        """Redis/DB 조회 중 예외 → 무시하고 헤더 fallback (안전망)."""
        cache_mock["raise_exc"] = RuntimeError("redis down")
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-Forwarded-Company": "ACME",
        })

        user = await auth_module.get_current_user(request)

        assert user["sub"] == "u-1"
        assert user["company"] == "ACME"
        assert request.state.user is user

    @pytest.mark.asyncio
    async def test_url_encoded_x_user_sub(self, cache_mock):
        """URL 인코딩된 X-User-Sub도 디코딩되어 사용된다."""
        cache_mock["return_value"] = None
        request = _make_request(headers={
            "X-User-Sub": "abc%2D123",  # "abc-123"
        })

        user = await auth_module.get_current_user(request)
        assert user["sub"] == "abc-123"
        # cache 조회도 디코딩된 값으로
        assert cache_mock["calls"] == ["abc-123"]

    @pytest.mark.asyncio
    async def test_session_id_always_from_header(self, cache_mock):
        """Redis profile에 session 정보가 없어도 X-Session-Id는 항상 헤더에서."""
        cache_mock["return_value"] = {
            "user": "user@example.com",
            "is_guest": False,
        }
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-Session-Id": "sess-from-header",
        })

        await auth_module.get_current_user(request)

        assert request.state.session_id == "sess-from-header"


class TestSelectedCompanyFollowsGatewayHeader:
    """선택 회사(``company``)의 출처 — 캐시가 아니라 Gateway 세션이 SOT.

    회사 전환은 ``PUT /auth/selected-company`` 가 세션에 동기 반영하고 이후 모든 요청에
    ``X-Forwarded-Company`` 로 실려 온다. profile 캐시는 프론트의 fire-and-forget
    재등록으로만 갱신되므로 전환 직후엔 이전 회사가 남아 있을 수 있다 — 그때 캐시를
    믿으면 경비청구 기안자·참석자가 이전 회사로 상신된다(2026-09-10 실측).
    """

    OLD = "(주)회사"
    NEW = "(주)고객사 A"

    def _cached(self, company):
        return {
            "user": "u@e.com",
            "full_name": "홍길동",
            "company": company,
            "companies": [self.OLD, self.NEW],
        }

    @pytest.mark.asyncio
    async def test_stale_cache_is_corrected_by_header(self, cache_mock):
        """캐시가 전환 전 회사 → 헤더의 새 회사로 교정된다."""
        cache_mock["return_value"] = self._cached(self.OLD)
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-Forwarded-Company": self.NEW,
        })

        user = await auth_module.get_current_user(request)

        assert user["company"] == self.NEW
        assert request.state.company == self.NEW
        # 소속 목록은 권한 계보라 건드리지 않는다
        assert user["companies"] == [self.OLD, self.NEW]

    @pytest.mark.asyncio
    async def test_percent_encoded_header_is_decoded(self, cache_mock):
        """게이트웨이는 회사명을 percent-encode 해 보낸다 — 디코딩 후 비교/적용."""
        cache_mock["return_value"] = self._cached(self.OLD)
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-Forwarded-Company": quote(self.NEW, safe=""),
        })

        user = await auth_module.get_current_user(request)

        assert user["company"] == self.NEW

    @pytest.mark.asyncio
    async def test_no_header_keeps_cached_company(self, cache_mock):
        """헤더 없는 진입(MM 봇 등)은 종전대로 캐시 값을 쓴다."""
        cache_mock["return_value"] = self._cached(self.OLD)
        request = _make_request(headers={"X-User-Sub": "u-1"})

        user = await auth_module.get_current_user(request)

        assert user["company"] == self.OLD

    @pytest.mark.asyncio
    async def test_matching_header_is_noop(self, cache_mock):
        """캐시와 헤더가 같으면 그대로 — 교정 분기가 값을 흔들지 않는다."""
        cache_mock["return_value"] = self._cached(self.NEW)
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-Forwarded-Company": self.NEW,
        })

        user = await auth_module.get_current_user(request)

        assert user["company"] == self.NEW

    @pytest.mark.asyncio
    async def test_empty_header_does_not_blank_cached_company(self, cache_mock):
        """빈 헤더는 값이 아니다 — 캐시 회사를 지워버리면 안 된다."""
        cache_mock["return_value"] = self._cached(self.OLD)
        request = _make_request(headers={
            "X-User-Sub": "u-1",
            "X-Forwarded-Company": "",
        })

        user = await auth_module.get_current_user(request)

        assert user["company"] == self.OLD


class TestProfileToUserDict:
    """_profile_to_user_dict 단위 — Frappe → user dict 어댑팅."""

    def test_full_profile(self):
        profile = {
            "user": "user@example.com",
            "full_name": "John Doe",
            "is_guest": False,
            "company": "ACME",
            "companies": ["ACME"],
            "role_type": "GA",
            "employee": {
                "department": "ENG",
                "designation": "Engineer",
                "employee_number": "E-1234",
            },
            # 권한 raw 는 어댑팅에서 의도적으로 버린다.
            "assigned_roles": ["agent_user", "admin"],
            "permissions": {"hr_service": ["read"]},
        }
        result = auth_module._profile_to_user_dict("u-1", profile)

        assert result["sub"] == "u-1"
        assert result["email"] == "user@example.com"
        assert result["username"] == "user@example.com"
        assert result["name"] == "John Doe"
        assert result["company"] == "ACME"
        assert result["companies"] == ["ACME"]
        assert result["role_type"] == "GA"
        assert result["employee"] == {
            "department": "ENG",
            "designation": "Engineer",
            "employee_number": "E-1234",
        }
        assert result["is_guest"] is False
        # 권한 raw 는 절대 user dict 로 넘어오면 안 된다.
        assert "roles" not in result
        assert "assigned_roles" not in result
        assert "permissions" not in result

    def test_korean_name_normalized(self):
        """한국 이름은 성+이름 순으로 재조합."""
        profile = {"user": "u@e.com", "full_name": "길동 홍"}
        result = auth_module._profile_to_user_dict("u-1", profile)
        assert result["name"] == "홍길동"

    def test_empty_full_name_returns_none(self):
        """full_name이 빈 문자열이면 name 키는 None."""
        profile = {"user": "u@e.com", "full_name": ""}
        result = auth_module._profile_to_user_dict("u-1", profile)
        assert result["name"] is None

    def test_missing_fields_are_none(self):
        """profile에 키가 빠져도 KeyError가 아니라 None으로."""
        profile = {"user": "u@e.com"}
        result = auth_module._profile_to_user_dict("u-1", profile)
        assert result["companies"] is None
        assert result["company"] is None
        assert result["role_type"] is None
        assert result["employee"] is None
        assert result["is_guest"] is False  # bool(None) = False
