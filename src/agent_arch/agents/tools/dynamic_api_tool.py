import logging
import re
from typing import Any, Dict, List
from urllib.parse import quote as _url_quote

import httpx

logger = logging.getLogger(__name__)

from agent_arch.agents.context import get_user_context

from agent_arch.agents.permissions import (
    GateMode,
    PermissionRequirement,
    normalize_service,
    permission_index,
)
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.core.config import settings
from agent_arch.services.api_registry.models import ExternalToolDef
from agent_arch.services.api_registry.service_config import get_base_url

# 요청당 하나의 AsyncClient를 재사용하기보다 각 호출에서 생성한다.
# (생명 주기 관리가 단순하고 연결 수가 적으므로 충분)
_HTTP_TIMEOUT = 30.0

# ── HTTP 오류 정제 ─────────────────────────────────────────────────────────

_HTTP_ERROR_CATEGORY: dict[int, str] = {
    400: "bad_request",
    401: "permission_denied",
    403: "permission_denied",
    404: "not_found",
    408: "timeout",
    429: "rate_limited",
    500: "server_error",
    502: "service_unavailable",
    503: "service_unavailable",
    504: "timeout",
}

_ERROR_DESCRIPTION: dict[str, str] = {
    "bad_request":         "입력 조건이 올바르지 않아 요청을 처리할 수 없습니다.",
    "permission_denied":   "해당 데이터를 조회할 수 있는 권한이 없습니다.",
    "not_found":           "요청한 데이터를 찾을 수 없습니다.",
    "timeout":             "시스템이 응답하지 않습니다. 잠시 후 다시 시도해 주세요.",
    "rate_limited":        "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
    "server_error":        "시스템 내부 오류가 발생했습니다.",
    "service_unavailable": "시스템이 일시적으로 사용할 수 없습니다.",
    "request_failed":      "요청을 처리할 수 없습니다.",
}


def _sanitize_http_error(e: httpx.HTTPStatusError) -> str:
    """HTTPStatusError를 LLM 친화적 에러 문자열로 변환한다.

    URL, 응답 바디, 상태 코드 숫자를 노출하지 않는다.
    400 에러는 API 서버가 반환한 detail 필드를 (길이 제한 후) 사용한다.

    Args:
        e: httpx.HTTPStatusError 인스턴스.

    Returns:
        "{카테고리}: {설명}" 형태의 문자열.
    """
    status = e.response.status_code
    category = _HTTP_ERROR_CATEGORY.get(status)
    if category is None:
        category = "request_failed" if status < 500 else "server_error"

    if category == "bad_request":
        try:
            body = e.response.json()
            detail = body.get("detail") or body.get("message") or body.get("error")
            if detail and isinstance(detail, str) and len(detail) <= 200:
                return f"{category}: {detail}"
        except Exception:
            pass

    return f"{category}: {_ERROR_DESCRIPTION[category]}"


def _sanitize_generic_error() -> str:
    """알 수 없는 예외를 LLM 친화적 에러 문자열로 변환한다.

    원본 예외 메시지를 노출하지 않는다.
    디버깅 정보는 호출 측에서 logger.exception()으로 기록한다.

    Returns:
        고정된 사용자 친화적 에러 문자열.
    """
    return f"request_failed: {_ERROR_DESCRIPTION['request_failed']}"


class DynamicAPITool(BaseTool):
    """llm_tools.json 항목 하나를 실행 가능한 Tool로 래핑한다.

    외부 서비스 API를 Gateway 헤더 포워딩 방식으로 호출한다.
    런타임에 동적으로 생성되며 Qdrant에서 검색된 API 도구 정의를 기반으로 동작한다.
    """

    def __init__(self, tool_def: ExternalToolDef):
        self._def = tool_def
        # BaseTool.__init__ 이전에 _def를 할당해야 get_name() 등이 동작한다.
        super().__init__(config={})
        self._register_permission()

    def _register_permission(self) -> None:
        """런타임에 PermissionIndex에 권한 메타를 등록한다.

        우선순위:
            1. ``tool_def.required_permissions`` 가 채워져 있으면 그대로 등록.
            2. 없으면 ``external_service`` 를 정규화해 ACCESS_ONLY fallback.
            3. ``external_service`` 도 없으면 등록하지 않음 → 게이트가 deny.

        2번 fallback의 효과: ``permissions[{service}]``의 ``"{service}.access"`` 만 검사되므로 그 서비스에
        접근권이 있는 사용자에게 신규 도구가 즉시 노출된다 (운영 친화).
        """
        rp = self._def.required_permissions
        if rp:
            try:
                req = PermissionRequirement.from_dict(rp)
                permission_index.register(self._def.tool_name, req)
                return
            except (KeyError, ValueError, TypeError) as e:
                logger.warning(
                    "권한 메타 파싱 실패 — fallback으로 진행 (tool=%s): %s",
                    self._def.tool_name, e,
                )

        ext = self._def.external_service or ""
        if not ext:
            logger.warning(
                "권한 메타 fallback 불가 — external_service 비어있음 (tool=%s) → deny",
                self._def.tool_name,
            )
            return
        permission_index.register(
            self._def.tool_name,
            PermissionRequirement(
                service=normalize_service(ext),
                mode=GateMode.ACCESS_ONLY,
            ),
        )

    # ── BaseTool 추상 메서드 구현 ──────────────────────────────────────────

    def get_name(self) -> str:
        """도구 이름을 반환한다.

        Returns:
            str: 도구 정의에 명시된 이름
        """
        return self._def.tool_name

    def get_description(self) -> str:
        """도구 설명을 반환한다.

        Returns:
            str: 도구 정의에 명시된 설명
        """
        return self._def.description

    def get_parameters(self) -> Dict[str, Any]:
        """도구 파라미터 스키마를 반환한다 (JSON Schema 형식).

        Returns:
            Dict[str, Any]: 도구 정의에 명시된 파라미터 스키마
        """
        return self._def.properties

    def get_required_params(self) -> List[str]:
        """필수 파라미터 리스트를 반환한다.

        Returns:
            List[str]: 도구 정의에 명시된 필수 파라미터 이름 리스트
        """
        return self._def.required_params

    # ── 실행 메서드 ─────────────────────────────────────────────────────────


    @staticmethod
    def _header_value(value: str) -> str:
        """HTTP 헤더 값에 비ASCII 문자(한글 등)가 있으면 percent-encoding으로 변환한다.

        HTTP/1.1 헤더는 ASCII만 허용하므로 한글 회사명 등이 그대로 전달되면
        'ascii' codec 오류가 발생한다. 수신 측은 URL-decode로 복원할 수 있다.

        Args:
            value (str): 헤더 값 문자열

        Returns:
            str: ASCII 문자열 또는 percent-encoded 문자열
        """
        if not value:
            return ""
        try:
            value.encode("ascii")
            return value
        except UnicodeEncodeError:
            return _url_quote(value, safe="")

    async def _real_execute(self, **kwargs) -> ToolResult:
        """Gateway 헤더를 포함하여 외부 API를 실제 호출한다.

        사용자 컨텍스트에서 인증 정보를 추출하여 헤더나 쿠키에 포함하고,
        외부 서비스의 API를 호출한다.

        Args:
            **kwargs: 도구 파라미터

        Returns:
            ToolResult: API 호출 결과 또는 오류 정보
        """
        ctx = get_user_context()
        headers = {
            "X-User-Sub": self._header_value(ctx.get("sub", "")),
            "X-Forwarded-Company": self._header_value(ctx.get("company", "")),
            "X-Forwarded-User": self._header_value(ctx.get("email", "")),
            "Content-Type": "application/json",
            "Origin": "https://agent.example.com",
        }
        cookies: Dict[str, str] = {}
        # 서비스별 인증 방식 분기
        if self._def.external_service == "hr_service":
            # HR 서비스: session_id 쿠키 방식
            if session_id := ctx.get("session_id"):
                cookies[settings.session_cookie_name] = session_id
                logger.debug("[DynamicAPITool] auth=session_id for HR service")
            else:
                logger.warning("[DynamicAPITool] auth=NONE for HR service - no session_id")
        else:
            # 기타 서비스: fallback 로직
            if session_id := ctx.get("session_id"):
                cookies[settings.session_cookie_name] = session_id
                logger.debug("[DynamicAPITool] auth=session_id (fallback)")
            elif session_cookie := ctx.get("session_cookie"):
                cookies[settings.session_cookie_name] = session_cookie
                logger.debug("[DynamicAPITool] auth=session_cookie (legacy fallback)")
            else:
                logger.warning("[DynamicAPITool] auth=NONE")

        try:
            base_url = get_base_url(self._def.external_service)
        except ValueError as e:
            return ToolResult(success=False, output=None, error=str(e))

        url = base_url.rstrip("/") + self._def.endpoint_path
        method = self._def.http_method.upper()

        try:
            async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT, follow_redirects=True) as client:
                if method == "GET":
                    url, query_params = self._resolve_path_and_params(url, dict(kwargs))
                    resp = await client.get(url, headers=headers, params=query_params, cookies=cookies)
                else:
                    url = self._resolve_path(url, dict(kwargs))
                    path_vars = set(re.findall(r"\{(\w+)\}", self._def.endpoint_path))
                    body_params = {k: v for k, v in kwargs.items() if k not in path_vars}
                    json_body = {"data": body_params}
                    resp = await client.request(
                        method, url, headers=headers, json=json_body, cookies=cookies
                    )

            resp.raise_for_status()
            raw = resp.json()
            return ToolResult(success=True, output=raw)

        except httpx.HTTPStatusError as e:
            logger.warning(
                "외부 API 오류 (tool=%s, status=%s): %s",
                self._def.tool_name,
                e.response.status_code,
                e.response.text[:500],
            )
            return ToolResult(
                success=False,
                output=None,
                error=_sanitize_http_error(e),
            )
        except Exception as e:
            logger.exception("도구 실행 중 예기치 않은 오류 (tool=%s)", self._def.tool_name)
            return ToolResult(success=False, output=None, error=_sanitize_generic_error())

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────

    def _resolve_path_and_params(self, url: str, kwargs: dict):
        """URL path 파라미터를 치환하고 나머지 kwargs를 query string으로 반환한다.

        Args:
            url (str): path 파라미터가 포함된 URL 템플릿
            kwargs (dict): 파라미터 딕셔너리

        Returns:
            tuple[str, dict]: (치환된 URL, 남은 파라미터 딕셔너리)
        """
        path_vars = re.findall(r"\{(\w+)\}", url)
        for var in path_vars:
            value = kwargs.pop(var, "")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            url = url.replace(f"{{{var}}}", str(value))
        # LLM이 integer 파라미터를 float(1.0 등)으로 생성하는 경우 int로 정규화
        query_params = {
            k: int(v) if isinstance(v, float) and v.is_integer() else v
            for k, v in kwargs.items()
        }
        return url, query_params

    def _resolve_path(self, url: str, kwargs: dict) -> str:
        """URL path 파라미터만 치환하고 URL을 반환한다.

        Args:
            url (str): path 파라미터가 포함된 URL 템플릿
            kwargs (dict): 파라미터 딕셔너리

        Returns:
            str: path 파라미터가 치환된 URL
        """
        path_vars = re.findall(r"\{(\w+)\}", url)
        for var in path_vars:
            value = kwargs.get(var, "")
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            url = url.replace(f"{{{var}}}", str(value))
        return url
