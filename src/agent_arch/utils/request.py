import logging
from typing import Any, Dict

from fastapi import Request

from agent_arch.agents.source_routing import resolve_memory_surface, resolve_service
from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

#: 서비스 축을 판정하지 못한 경로 — 경로당 한 번만 경고한다. 턴마다 찍으면(실측 하루
#: 수백 턴) 로그가 밀려 정작 다른 신호를 덮는다.
_warned_unresolved_paths: set[str] = set()


def build_user_info(http_request: Request) -> Dict[str, Any]:
    """미들웨어 사용자 정보를 API 레이어 표준 포맷으로 변환한다.

    Args:
        http_request (Request): FastAPI HTTP 요청 객체

    Returns:
        Dict[str, Any]: 사용자 정보 딕셔너리 (sub, email, company 등 포함)
    """
    user = http_request.state.user
    user_info = {
        "sub": user["sub"],
        "email": user.get("email"),
        "company": user.get("company") or getattr(http_request.state, "company", None),
    }

    # 확장 사용자 정보 — Redis profile 또는 Gateway 헤더에서 채워진 값
    # is_guest / role_type / employee는 Redis profile 경로에서만 채워지므로
    # 헤더 fallback일 때는 누락된다.
    for key in (
        "name", "username", "companies", "role_type", "employee", "is_guest"
    ):
        value = user.get(key)
        if value is not None:
            user_info[key] = value

    # X-Session-Id 헤더 기반 세션 ID (Gateway → agent-backend → Gateway Token Exchange)
    session_id = getattr(http_request.state, "session_id", None)
    if session_id:
        user_info["session_id"] = session_id

    # X-Source-Path 헤더 기반 UX 힌트 (권한 분기 사용 금지).
    # agent_mode / guide_categories fallback 매핑의 입력으로만 사용.
    source_path = getattr(http_request.state, "source_path", None)
    if source_path:
        user_info["source_path"] = source_path

    # 사용량 관측용 서비스 축. 인증된 에이전트 면이 전부 이 함수를 거치므로 여기 한 곳에서
    # 판정한다 — 호출부마다 붙이면 새 라우터가 생길 때마다 조용히 미분류가 된다.
    service = resolve_service(http_request.url.path)
    if service:
        user_info["service"] = service
    else:
        _warn_unresolved_service(http_request.url.path, source_path)

    # 기억의 자리(앱 전용 창). 사용량 서비스 축과 같은 이유로 여기 한 곳에서 판정한다 —
    # 값이 붙은 턴의 기억은 그 창에 매이고(memory_v2/place), 값이 없으면 장소 중립이라
    # 이 기능 이전과 순위·자격이 같다. 판정 근거가 경로 하나라 새 창이 생기면
    # source_routing._MEMORY_SURFACE_BY_PREFIX 한 줄로 끝난다.
    surface = resolve_memory_surface(http_request.url.path)
    if surface:
        user_info["memory_surface"] = surface

    # 세션 쿠키 포워딩 (Gateway 미경유 직접 호출 호환)
    session_cookie = http_request.cookies.get(settings.session_cookie_name)
    if session_cookie:
        user_info["session_cookie"] = session_cookie

    return user_info


def _warn_unresolved_service(path: str, source_path: Any) -> None:
    """서비스 축 판정 실패를 **경로와 함께** 남긴다.

    지금까지 이 실패는 완전히 조용했다. 화면에는 '미분류'만 뜨고 어느 경로가 판정에서
    빠졌는지는 어디에도 없어서, 2026-09-03 dev 의 미분류 129턴을 추적하는 데 DB·컨테이너·
    네트워크를 전부 뒤져야 했다. 경로가 로그에 한 줄 남으면 그 조사는 grep 한 번이다.

    같은 경로를 반복해 찍지 않는다 — 새 면이 붙었을 때의 **첫 신호**가 목적이고, 그 뒤로는
    사용량 관측의 `unclassified_service_tokens` 가 규모를 보여준다.
    """

    if path in _warned_unresolved_paths:
        return
    _warned_unresolved_paths.add(path)
    logger.warning(
        "사용량 서비스 축 미판정 — 이 경로의 사용량이 '미분류'로 적재된다 "
        "[path=%s source_path=%s api_v1_prefix=%s]. "
        "새 면이라면 source_routing._SERVICE_BY_PREFIX 에 추가할 것.",
        path, source_path, settings.api_v1_prefix,
    )
