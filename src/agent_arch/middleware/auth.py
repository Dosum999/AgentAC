"""사용자 인증 미들웨어 (FastAPI dependency).

경로:
    0) ``AUTH_MODE=dev`` 면 게이트웨이 없이 ``DEV_USER_*`` 고정 사용자로 통과(로컬 개발 전용).
    1) Gateway가 주입한 ``X-User-Sub``로 Redis profile을 조회한다.
    2) hit → ID 공급자 profile(JSON)을 user dict로 어댑팅해 ``request.state.user``에 채운다.
    3) miss → 기존 헤더 기반 파싱으로 fallback (Redis 캐시 미스 안전망).

단, **선택 회사(``company``)는 캐시 hit 여도 헤더를 따른다** — 선택 회사의 SOT는
Gateway Redis 세션이고 profile 캐시는 프론트 재등록으로만 갱신되기 때문이다.
자세한 근거는 ``get_current_user`` 안의 주석.

ContextVar 키(``sub``, ``email``, ``company``, ``session_id`` 등)는 두 경로에서 동일
포맷을 유지한다 — ``DynamicAPITool`` 같은 하위 코드는 변경 없이 그대로 동작.
"""
import json
import logging
import re
from typing import Optional
from urllib.parse import unquote

from fastapi import HTTPException, Request, status

from agent_arch.services.user_profile import get_profile as get_profile_from_cache

logger = logging.getLogger(__name__)

_KOREAN_RE = re.compile(r'^[가-힯]+$')


def _normalize_korean_name(name: str) -> str:
    """Keycloak이 서양식 순서(given_name family_name)로 보내는 한국 이름을 성+이름 순으로 재조합한다.

    공백 하나로 분리되고 양쪽이 모두 한글인 경우에만 역순으로 재조합한다.

    Args:
        name (str): 정규화할 이름 문자열.

    Returns:
        str: 성+이름 순으로 재조합된 이름. 조건 미충족 시 원본 반환.

    Examples:
        >>> _normalize_korean_name("길동 홍")
        '홍길동'
        >>> _normalize_korean_name("John Doe")
        'John Doe'
    """
    parts = name.split(" ")
    if len(parts) == 2 and all(_KOREAN_RE.match(p) for p in parts):
        given, family = parts
        return family + given
    return name


def _parse_gateway_header(value: str | None):
    """Gateway 헤더 값을 디코딩하고 파싱합니다.

    URL 인코딩된 문자열을 디코딩하고, JSON 배열/객체 문자열은 Python 객체로 변환합니다.

    Args:
        value (str | None): 헤더 값 문자열. None이면 None 반환

    Returns:
        str | list | dict | None: 디코딩/파싱된 값
        - None: 입력이 None이거나 빈 문자열인 경우
        - str: 일반 문자열
        - list: JSON 배열 문자열이 파싱된 리스트
        - dict: JSON 객체 문자열이 파싱된 딕셔너리

    Examples:
        >>> _parse_gateway_header("test%40example.com")
        'test@example.com'
        >>> _parse_gateway_header("%5B%22admin%22%2C%20%22user%22%5D")
        ['admin', 'user']
        >>> _parse_gateway_header(None)
        None
    """
    if not value:
        return None

    decoded = unquote(value).strip()
    if not decoded:
        return None

    # JSON 배열 또는 객체인 경우 파싱 시도
    if decoded[0] in "[{":
        try:
            return json.loads(decoded)
        except json.JSONDecodeError:
            return decoded

    return decoded


def _profile_to_user_dict(user_sub: str, profile: dict) -> dict:
    """ID 공급자 사용자 조회 응답 → user dict 포맷 (LLM 컨텍스트 소스).

    응답 매핑:
        - ``profile.user`` (이메일) → ``email``, ``username``
        - ``profile.full_name`` → ``name`` (한국 이름은 성+이름 순으로 정규화)
        - ``profile.companies`` → ``companies``
        - ``profile.company`` → ``company`` (현재 선택된 회사)
        - ``profile.role_type`` → ``role_type`` (PA / GA / CA / EMPLOYEE)
        - ``profile.employee`` → ``employee`` (부서·직급·사번 dict; None 가능)
        - ``profile.is_guest`` → ``is_guest``

    권한 raw(``assigned_roles``, ``permissions``, ``service_access``)는 의도적으로
    어댑팅에서 제외한다. 권한은 별도 트랙(ID 공급자 ``permissions`` 기반 코드 분기)
    에서 다루며, LLM 컨텍스트로 raw role 문자열이 들어가면 격상 환각의 씨앗이 됨.

    Args:
        user_sub: Gateway가 검증한 Keycloak sub.
        profile: ID 공급자 사용자 조회 응답 JSON.

    Returns:
        ``get_current_user``가 ``request.state.user``에 채우는 표준 dict.
    """
    full_name = profile.get("full_name") or ""
    return {
        "sub": user_sub,
        "email": profile.get("user"),
        "name": _normalize_korean_name(full_name) or None,
        "username": profile.get("user"),
        "companies": profile.get("companies"),
        "company": profile.get("company"),
        "role_type": profile.get("role_type"),
        "employee": profile.get("employee"),
        "is_guest": bool(profile.get("is_guest")),
    }


def _build_user_from_headers(request: Request, user_sub: str) -> dict:
    """Redis profile 캐시 미스 시 fallback — Gateway 확장 헤더에서 user dict를 구성한다."""
    companies = _parse_gateway_header(request.headers.get("X-User-Companies"))
    company = _parse_gateway_header(request.headers.get("X-Forwarded-Company"))

    # X-Forwarded-Company가 없으면 X-User-Companies의 첫 값을 사용
    if not company and isinstance(companies, list) and companies:
        company = companies[0]
    if not company and isinstance(companies, str):
        company = companies

    return {
        "sub": user_sub,
        "email": _parse_gateway_header(
            request.headers.get("X-Forwarded-User")
            or request.headers.get("X-User-Username")
        ),
        "name": _normalize_korean_name(
            _parse_gateway_header(request.headers.get("X-User-Name")) or ""
        ) or None,
        "username": _parse_gateway_header(request.headers.get("X-User-Username")),
        "roles": _parse_gateway_header(request.headers.get("X-User-Roles")),
        "companies": companies,
        "company": company,
    }


async def _try_get_cached_profile(user_sub: str) -> Optional[dict]:
    """Redis profile 조회 — 캐시·DB 오류는 fallback이 흡수하도록 None으로 흡수."""
    try:
        return await get_profile_from_cache(user_sub)
    except Exception:
        logger.warning(
            "auth: Redis/DB profile 조회 실패 — 헤더 fallback으로 진행",
            exc_info=True,
        )
        return None


def _dev_user() -> Optional[dict]:
    """``AUTH_MODE=dev`` 면 ``DEV_USER_*`` 로 고정 사용자를 만든다. 아니면 None(기본 = 게이트웨이 헤더 필수)."""
    from agent_arch.core.config import settings

    if str(getattr(settings, "auth_mode", "") or "").lower() != "dev":
        return None
    return {
        "sub": settings.dev_user_sub,
        "email": settings.dev_user_email,
        "name": settings.dev_user_name,
        "username": settings.dev_user_email,
        "companies": [settings.dev_user_company] if settings.dev_user_company else [],
        "company": settings.dev_user_company or None,
        "is_guest": False,
    }


async def get_current_user(request: Request) -> dict:
    """Gateway가 전달한 ``X-User-Sub``로 사용자 정보를 구성하는 FastAPI dependency.

    듀얼 경로:
        1) Redis profile (``user_profile:{sub}``) hit → profile 어댑팅
        2) miss → 기존 헤더 기반 파싱 (``X-User-Name``, ``X-User-Roles`` 등)

    Args:
        request (Request): FastAPI Request 객체.

    Returns:
        dict: 사용자 정보 딕셔너리. 다음 키를 포함한다:
            - ``sub``: Keycloak sub (UUID)
            - ``email``: 이메일 (선택)
            - ``name``: 이름 (선택, 한국 이름은 정규화)
            - ``username``: 사용자 아이디 (선택)
            - ``roles``: 권한 목록 (선택)
            - ``companies``: 소속 회사 목록 (선택)
            - ``company``: 현재 선택된 회사 (선택)
            - ``is_guest``: 비로그인 게스트 꼬리표 (Redis 경로에서만)

    Raises:
        HTTPException: ``X-User-Sub`` 헤더 누락 시 401.
    """
    dev_user = _dev_user()
    if dev_user is not None and not request.headers.get("X-User-Sub"):
        # 로컬 개발(AUTH_MODE=dev): 게이트웨이 없이 고정 사용자로 통과한다. 헤더가 오면 헤더를 따른다.
        request.state.user = dev_user
        request.state.company = dev_user.get("company")
        request.state.session_id = None
        request.state.source_path = None
        return dev_user

    user_sub = _parse_gateway_header(request.headers.get("X-User-Sub"))
    if not user_sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-User-Sub header",
        )

    cached = await _try_get_cached_profile(user_sub)
    if cached is not None:
        user = _profile_to_user_dict(user_sub, cached)
        source = "redis"
        # 선택 회사만은 캐시가 아니라 헤더를 따른다.
        #
        # 선택 회사의 SOT는 Gateway Redis 세션이다 — 헤더 셀렉터의 회사 전환은
        # ``PUT /auth/selected-company`` 가 ID 공급자 + 세션에 동기 반영하고, 이후 모든
        # 요청에 ``X-Forwarded-Company`` 로 실려 온다. 반면 이쪽 profile 캐시는
        # ``POST /user/profile`` 재등록으로만 갱신되고(자동 무효화 없음, TTL 만료도
        # 같은 DB 행을 backfill) 그 재등록은 프론트가 실패를 삼키는 fire-and-forget
        # 이라, 캐시를 신뢰하면 전환 후에도 이전 회사로 업무 요청이 이전 회사로 나간다.
        #
        # 헤더가 있을 때만 덮으므로 Gateway를 경유하지 않는 진입(MM 봇 등)은 종전대로
        # 캐시 값을 쓴다. ``companies``(소속 목록)는 권한 계보라 건드리지 않는다.
        header_company = _parse_gateway_header(
            request.headers.get("X-Forwarded-Company")
        )
        if header_company and header_company != user.get("company"):
            logger.info(
                "auth: 선택 회사를 게이트웨이 헤더로 교정 — user_sub=%s cached=%s header=%s",
                user_sub, user.get("company"), header_company,
            )
            user["company"] = header_company
    else:
        # Redis profile 미등록 시 헤더 기반 fallback.
        user = _build_user_from_headers(request, user_sub)
        source = "headers_fallback"

    request.state.user = user
    request.state.company = user.get("company")
    request.state.session_id = request.headers.get("X-Session-Id")
    # 게이트웨이가 _encode_header_value(quote, safe="") 로 percent-encode 해 보내므로
    # 다른 게이트웨이 헤더(X-User-Sub 등)와 동일하게 _parse_gateway_header 로 unquote 한다.
    request.state.source_path = _parse_gateway_header(request.headers.get("X-Source-Path"))

    logger.debug(
        "auth.source=%s user_sub=%s is_guest=%s",
        source, user_sub, user.get("is_guest"),
    )
    logger.debug("auth.source_path=%s", request.state.source_path)

    return user
