"""DynamicAPITool HTTP 오류 정제 단위 테스트.

_sanitize_http_error(), _sanitize_generic_error() 의 동작과
DynamicAPITool._real_execute() 에서 예외가 올바르게 정제된 ToolResult로 반환되는지 검증한다.
"""
import json
import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.agents.tools.dynamic_api_tool import (
    _sanitize_http_error,
    _sanitize_generic_error,
    _ERROR_DESCRIPTION,
    DynamicAPITool,
)
from agent_arch.agents.tools.base import ToolResult


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _make_http_error(status_code: int, body: str = "") -> httpx.HTTPStatusError:
    """지정한 상태 코드와 바디로 HTTPStatusError를 생성한다."""
    request = httpx.Request("GET", "http://internal/api/v1/test")
    response = httpx.Response(
        status_code=status_code,
        content=body.encode(),
        headers={"content-type": "application/json"},
    )
    return httpx.HTTPStatusError(
        message=str(status_code),
        request=request,
        response=response,
    )


def _make_json_error(status_code: int, **fields) -> httpx.HTTPStatusError:
    """JSON 바디를 가진 HTTPStatusError를 생성한다."""
    return _make_http_error(status_code, json.dumps(fields, ensure_ascii=False))


def _make_tool(service: str = "hr_service") -> DynamicAPITool:
    """테스트용 DynamicAPITool 인스턴스를 생성한다."""
    tool_def = MagicMock()
    tool_def.tool_name = "test_tool"
    tool_def.description = "테스트 도구"
    tool_def.properties = {}
    tool_def.required_params = []
    tool_def.external_service = service
    tool_def.endpoint_path = "/api/v1/employees"
    tool_def.http_method = "GET"
    return DynamicAPITool(tool_def)


# ── _sanitize_http_error ───────────────────────────────────────────────────────

class TestSanitizeHttpError:
    """_sanitize_http_error() 동작 검증."""

    def test_401_returns_permission_denied(self):
        result = _sanitize_http_error(_make_http_error(401))
        assert result.startswith("permission_denied:")

    def test_403_returns_permission_denied(self):
        result = _sanitize_http_error(_make_http_error(403))
        assert result.startswith("permission_denied:")
        assert "권한" in result

    def test_404_returns_not_found(self):
        result = _sanitize_http_error(_make_http_error(404))
        assert result.startswith("not_found:")
        assert "찾을 수 없" in result

    def test_500_returns_server_error(self):
        result = _sanitize_http_error(_make_http_error(500))
        assert result.startswith("server_error:")

    def test_502_returns_service_unavailable(self):
        result = _sanitize_http_error(_make_http_error(502))
        assert result.startswith("service_unavailable:")

    def test_503_returns_service_unavailable(self):
        result = _sanitize_http_error(_make_http_error(503))
        assert result.startswith("service_unavailable:")

    def test_504_returns_timeout(self):
        result = _sanitize_http_error(_make_http_error(504))
        assert result.startswith("timeout:")

    def test_408_returns_timeout(self):
        result = _sanitize_http_error(_make_http_error(408))
        assert result.startswith("timeout:")

    def test_429_returns_rate_limited(self):
        result = _sanitize_http_error(_make_http_error(429))
        assert result.startswith("rate_limited:")

    def test_400_with_detail_uses_detail(self):
        """400 오류에서 API detail 필드를 추출해 사용한다."""
        err = _make_json_error(400, detail="사원번호 형식이 올바르지 않습니다.")
        result = _sanitize_http_error(err)
        assert result.startswith("bad_request:")
        assert "사원번호 형식이 올바르지 않습니다." in result

    def test_400_with_message_uses_message(self):
        """400 오류에서 message 필드를 추출해 사용한다."""
        err = _make_json_error(400, message="잘못된 날짜 형식입니다.")
        result = _sanitize_http_error(err)
        assert result.startswith("bad_request:")
        assert "잘못된 날짜 형식입니다." in result

    def test_400_with_error_uses_error(self):
        """400 오류에서 error 필드를 추출해 사용한다."""
        err = _make_json_error(400, error="invalid_param")
        result = _sanitize_http_error(err)
        assert result.startswith("bad_request:")
        assert "invalid_param" in result

    def test_400_detail_takes_precedence_over_message(self):
        """detail, message 둘 다 있으면 detail을 우선한다."""
        err = _make_json_error(400, detail="detail 값", message="message 값")
        result = _sanitize_http_error(err)
        assert "detail 값" in result

    def test_400_detail_too_long_falls_back_to_generic(self):
        """400 detail이 200자 초과이면 일반 메시지로 대체한다."""
        err = _make_json_error(400, detail="A" * 201)
        result = _sanitize_http_error(err)
        assert result == f"bad_request: {_ERROR_DESCRIPTION['bad_request']}"

    def test_400_detail_exactly_200_chars_is_allowed(self):
        """200자 정확히 일치하면 detail을 사용한다."""
        detail = "B" * 200
        err = _make_json_error(400, detail=detail)
        result = _sanitize_http_error(err)
        assert detail in result

    def test_400_non_json_body_falls_back_to_generic(self):
        """400 바디가 JSON이 아니면 일반 메시지로 대체한다."""
        err = _make_http_error(400, body="Bad Request")
        result = _sanitize_http_error(err)
        assert result == f"bad_request: {_ERROR_DESCRIPTION['bad_request']}"

    def test_400_numeric_detail_falls_back_to_generic(self):
        """400 detail이 문자열이 아니면 일반 메시지로 대체한다."""
        err = _make_json_error(400, detail=12345)
        result = _sanitize_http_error(err)
        assert result == f"bad_request: {_ERROR_DESCRIPTION['bad_request']}"

    def test_unknown_4xx_returns_request_failed(self):
        """매핑에 없는 4xx(예: 418)는 request_failed를 반환한다."""
        result = _sanitize_http_error(_make_http_error(418))
        assert result.startswith("request_failed:")

    def test_unknown_5xx_returns_server_error(self):
        """매핑에 없는 5xx(예: 511)는 server_error를 반환한다."""
        result = _sanitize_http_error(_make_http_error(511))
        assert result.startswith("server_error:")

    def test_no_raw_status_code_in_output(self):
        """출력에 HTTP 상태 코드 숫자가 포함되지 않는다."""
        for status in (401, 403, 404, 500, 502, 503, 504):
            result = _sanitize_http_error(_make_http_error(status))
            assert str(status) not in result, (
                f"상태 코드 {status}가 출력에 노출됨: {result}"
            )

    def test_no_url_in_output(self):
        """출력에 요청 URL이 포함되지 않는다."""
        result = _sanitize_http_error(_make_http_error(403))
        assert "http" not in result
        assert "/api/" not in result

    def test_no_response_body_in_output_for_5xx(self):
        """5xx 오류에서 응답 바디가 출력에 포함되지 않는다."""
        err = _make_http_error(500, body='{"stack_trace": "line 42 in app.py"}')
        result = _sanitize_http_error(err)
        assert "stack_trace" not in result
        assert "line 42" not in result


# ── _sanitize_generic_error ───────────────────────────────────────────────────

class TestSanitizeGenericError:
    """_sanitize_generic_error() 동작 검증."""

    def test_returns_request_failed_prefix(self):
        assert _sanitize_generic_error().startswith("request_failed:")

    def test_returns_exact_expected_message(self):
        expected = f"request_failed: {_ERROR_DESCRIPTION['request_failed']}"
        assert _sanitize_generic_error() == expected

    def test_is_deterministic(self):
        """항상 동일한 값을 반환한다."""
        assert _sanitize_generic_error() == _sanitize_generic_error()


# ── DynamicAPITool._real_execute() 연동 검증 ─────────────────────────────────

class TestDynamicAPIToolErrorSanitization:
    """_real_execute()가 예외를 정제된 ToolResult로 반환하는지 검증한다."""

    def _patched_execute(self, tool: DynamicAPITool, mock_client_setup):
        """공통 패치 컨텍스트를 제공하는 헬퍼."""
        return (
            patch(
                "agent_arch.agents.tools.dynamic_api_tool.get_user_context",
                return_value={
                    "sub": "user1",
                    "company": "company1",
                    "email": "user@example.com",
                    "session_id": "test-session-id",
                },
            ),
            patch(
                "agent_arch.agents.tools.dynamic_api_tool.get_base_url",
                return_value="http://hr-service",
            ),
            patch(
                "agent_arch.agents.tools.dynamic_api_tool.settings",
                session_cookie_name="dmp_sid",
            ),
            patch("httpx.AsyncClient", new_callable=lambda: lambda **kw: mock_client_setup),
        )

    def _make_mock_client(self, side_effect) -> MagicMock:
        """GET 요청에 side_effect를 가진 모의 AsyncClient를 생성한다."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = side_effect
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        return mock_ctx

    @pytest.mark.asyncio
    async def test_http_403_returns_permission_denied(self):
        """403 오류 시 permission_denied 카테고리가 ToolResult에 담긴다."""
        tool = _make_tool()
        err = _make_http_error(403)
        mock_ctx = self._make_mock_client(side_effect=err)

        with (
            patch("agent_arch.agents.tools.dynamic_api_tool.get_user_context", return_value={"sub": "u", "company": "c", "email": "e@e.com", "session_id": "sid"}),
            patch("agent_arch.agents.tools.dynamic_api_tool.get_base_url", return_value="http://hr-service"),
            patch("agent_arch.agents.tools.dynamic_api_tool.settings", session_cookie_name="dmp_sid"),
            patch("httpx.AsyncClient", return_value=mock_ctx),
        ):
            result = await tool._real_execute()

        assert isinstance(result, ToolResult)
        assert result.success is False
        assert result.error.startswith("permission_denied:")
        assert "403" not in result.error
        assert "/api/" not in result.error

    @pytest.mark.asyncio
    async def test_http_404_returns_not_found(self):
        """404 오류 시 not_found 카테고리가 ToolResult에 담긴다."""
        tool = _make_tool()
        err = _make_http_error(404)
        mock_ctx = self._make_mock_client(side_effect=err)

        with (
            patch("agent_arch.agents.tools.dynamic_api_tool.get_user_context", return_value={"sub": "u", "company": "c", "email": "e@e.com", "session_id": "sid"}),
            patch("agent_arch.agents.tools.dynamic_api_tool.get_base_url", return_value="http://hr-service"),
            patch("agent_arch.agents.tools.dynamic_api_tool.settings", session_cookie_name="dmp_sid"),
            patch("httpx.AsyncClient", return_value=mock_ctx),
        ):
            result = await tool._real_execute()

        assert result.success is False
        assert result.error.startswith("not_found:")

    @pytest.mark.asyncio
    async def test_http_500_returns_server_error(self):
        """500 오류 시 server_error 카테고리가 ToolResult에 담긴다."""
        tool = _make_tool()
        err = _make_http_error(500, body='{"traceback": "...line 99..."}')
        mock_ctx = self._make_mock_client(side_effect=err)

        with (
            patch("agent_arch.agents.tools.dynamic_api_tool.get_user_context", return_value={"sub": "u", "company": "c", "email": "e@e.com", "session_id": "sid"}),
            patch("agent_arch.agents.tools.dynamic_api_tool.get_base_url", return_value="http://hr-service"),
            patch("agent_arch.agents.tools.dynamic_api_tool.settings", session_cookie_name="dmp_sid"),
            patch("httpx.AsyncClient", return_value=mock_ctx),
        ):
            result = await tool._real_execute()

        assert result.success is False
        assert result.error.startswith("server_error:")
        assert "traceback" not in result.error
        assert "line 99" not in result.error

    @pytest.mark.asyncio
    async def test_generic_exception_hides_original_message(self):
        """일반 예외의 원본 메시지가 ToolResult에 노출되지 않는다."""
        tool = _make_tool()
        secret = "비밀 내부 에러 /internal/path token=abc123"
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=RuntimeError(secret))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("agent_arch.agents.tools.dynamic_api_tool.get_user_context", return_value={"sub": "u", "company": "c", "email": "e@e.com", "session_id": "sid"}),
            patch("agent_arch.agents.tools.dynamic_api_tool.get_base_url", return_value="http://hr-service"),
            patch("agent_arch.agents.tools.dynamic_api_tool.settings", session_cookie_name="dmp_sid"),
            patch("httpx.AsyncClient", return_value=mock_ctx),
        ):
            result = await tool._real_execute()

        assert result.success is False
        assert result.error.startswith("request_failed:")
        assert secret not in result.error
        assert "RuntimeError" not in result.error
        assert "/internal/" not in result.error

    @pytest.mark.asyncio
    async def test_generic_exception_returns_tool_result_instance(self):
        """일반 예외 발생 시 반환값은 항상 ToolResult 인스턴스다."""
        tool = _make_tool()
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=ConnectionError("refused"))
        mock_ctx = MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("agent_arch.agents.tools.dynamic_api_tool.get_user_context", return_value={"sub": "u", "company": "c", "email": "e@e.com", "session_id": "sid"}),
            patch("agent_arch.agents.tools.dynamic_api_tool.get_base_url", return_value="http://hr-service"),
            patch("agent_arch.agents.tools.dynamic_api_tool.settings", session_cookie_name="dmp_sid"),
            patch("httpx.AsyncClient", return_value=mock_ctx),
        ):
            result = await tool._real_execute()

        assert isinstance(result, ToolResult)
        assert result.success is False
