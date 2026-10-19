"""MCP 서버 도구 1개를 ``BaseTool``로 노출하는 어댑터.

정적 스킬과 달리 클래스 단위가 아니라 **인스턴스 단위**로 정의된다 —
이름/설명/파라미터 스키마를 서버 ``list_tools`` 결과에서 받아 생성한다
(DynamicAPITool과 같은 방식). 권한은 provider가 등록 시점에
``permission_index``에 직접 register한다.

호출은 stateless: 도구 실행 1건마다 새 HTTP 연결을 열고 현재 요청
사용자의 이메일을 ``X-User-Email`` 헤더로 전달한다. 사용자별 자격증명
(앱 비밀번호 등)은 MCP 서버 쪽에서 해결하므로 백엔드는 보관하지 않는다.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from agent_arch.agents.context import get_user_context
from agent_arch.agents.permissions.spec import PermissionRequirement
from agent_arch.extensions import extensions
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.agents.tools.mcp.transport import open_mcp_session, user_context_headers
from agent_arch.services.usage.meter import harvest_mcp_usage
from agent_arch.services.monitoring.error_events import event_extra

logger = logging.getLogger(__name__)

# 도구 호출 상한 — IMAP 폴더 스캔 등 느린 연산 여유 포함
_CALL_TIMEOUT_SECONDS = 60.0


def _sanitize_parameters(input_schema: Optional[Dict[str, Any]], hidden: Iterable[str] = ()) -> Dict[str, Any]:
    """MCP input_schema.properties → BaseTool 파라미터 스키마.

    MCPServer(구 FastMCP)가 붙이는 ``title`` 키는 LLM 바인딩에 노이즈라 제거한다. ``hidden`` 에 든 파라미터는
    모델에 노출하지 않는다(정책이 기본값·강제값으로 채우는 운영 인자, 헤더성 인자 등).
    """
    properties = (input_schema or {}).get("properties", {}) or {}
    hidden_set = set(hidden or ())
    cleaned: Dict[str, Any] = {}
    for name, schema in properties.items():
        if name in hidden_set:
            continue
        if isinstance(schema, dict):
            cleaned[name] = {k: v for k, v in schema.items() if k != "title"}
        else:
            cleaned[name] = schema
    return cleaned


def _enum_ok(schema: Any, value: Any) -> bool:
    """스칼라 enum 이 선언된 파라미터면 값이 그 안에 있는지. enum 이 없거나 스칼라가 아니면 통과."""
    allowed = schema.get("enum") if isinstance(schema, dict) else None
    if not isinstance(allowed, list) or isinstance(value, (list, dict)):
        return True
    return value in allowed


class RemoteMCPTool(BaseTool):
    """원격 MCP 도구 어댑터. ``@register_tool`` 데코레이터를 타지 않고
    provider가 인스턴스를 직접 register한다."""

    def __init__(
        self,
        *,
        server_url: str,
        service_token: str,
        name: str,
        description: str,
        input_schema: Optional[Dict[str, Any]],
        permission: PermissionRequirement,
        exposure: str = "core",
        prompt_guidance: str = "",
        hidden_params: Iterable[str] = (),
        default_args: Optional[Dict[str, Any]] = None,
        forced_args: Optional[Dict[str, Any]] = None,
        server_name: Optional[str] = None,
    ):
        self._server_url = server_url
        # MCP 서버 키(provider 의 MCPServerSpec.name). 호출 실패를 오류 이벤트 mcp_server 로 묶는다.
        self._server_name = server_name
        self._service_token = service_token
        self._name = name
        self._description = description
        self._hidden = frozenset(hidden_params or ())
        self._default_args = dict(default_args or {})
        self._forced_args = dict(forced_args or {})
        self._parameters = _sanitize_parameters(input_schema, self._hidden)
        self._required = [r for r in (input_schema or {}).get("required", []) if r not in self._hidden]
        # 인스턴스 속성으로 ClassVar 를 가린다 — 워커 지침 수집(_bound_tools_guidance)은 getattr 로 읽는다.
        self.prompt_guidance = prompt_guidance or ""
        # ClassVar 섀도잉 — permission_index 등록은 provider 책임이지만,
        # 코드 어디서든 tool.permission_required를 읽는 경로와 일관성 유지.
        self.permission_required = permission
        self.exposure = exposure  # 노출 티어 (exposure.resolve_exposure가 해석)
        super().__init__()

    def get_name(self) -> str:
        return self._name

    def get_description(self) -> str:
        return self._description

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters

    def get_required_params(self) -> List[str]:
        return self._required

    def _merge_arguments(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """모델 인자 위에 정책을 얹는다: 기본값(모델이 비운 것만) → 모델 인자 → 강제값(항상). 숨긴 인자는 모델이 줘도 버린다.

        스키마 enum 밖의 값은 모델이 비운 것으로 친다(기본값이 있으면 기본값). 실측(2026-09-23): search_lang="ko-hkr" 한 글자
        오타로 서버가 호출 전체를 -32602 로 거절해 검색 결과를 통째로 잃었다. 조회를 살리는 쪽이 낫다.
        """
        given = {k: v for k, v in kwargs.items()
                 if k not in self._hidden and v is not None and _enum_ok(self._parameters.get(k), v)}
        return {**self._default_args, **given, **self._forced_args}

    async def _demo_execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output={
                "message": f"[DEMO] {self._name} 호출 (MCP 서버 미연동 모의 응답)",
                "params": kwargs,
            },
        )

    async def _real_execute(self, **kwargs) -> ToolResult:
        user_ctx = get_user_context() or {}
        email = user_ctx.get("email")
        if not email:
            return ToolResult(
                success=False,
                output=None,
                error=(
                    "사용자 컨텍스트에 이메일이 없어 메일 도구를 사용할 수 없습니다 "
                    "(인증되지 않은 요청에서 사용자 도구 호출 시도)."
                ),
            )

        # 사람 이름 인자는 디렉터리로 실제 계정을 확정한다(extensions.person_resolver).
        # 모호하면 해석기가 되묻기용 ToolResult 를 settled 로 돌려준다.
        if extensions.person_resolver is not None:
            kwargs, settled = await extensions.person_resolver(self._name, kwargs)
            if settled is not None:
                return settled
        try:
            result = await self._call_remote(self._merge_arguments(kwargs), user_context_headers(user_ctx))
        except Exception as e:
            logger.warning(
                "[mcp] 도구 호출 실패 name=%s: %s: %r", self._name, type(e).__name__, e,
                extra=event_extra("mcp_tool", exc=e, mcp_server=self._server_name or "unknown"),
            )
            return ToolResult(
                success=False,
                output=None,
                error=f"MCP 서버 호출 실패 ({self._name}) — 서버 상태를 확인하세요: {e}",
            )
        return _to_tool_result(self._name, result)

    async def _call_remote(self, arguments: Dict[str, Any], user_headers: Dict[str, str]):
        """원격 호출 본체. 테스트에서 이 메서드를 패치해 전송 계층을 격리한다."""
        async with open_mcp_session(
            self._server_url,
            service_token=self._service_token,
            user_headers=user_headers,
            timeout_seconds=_CALL_TIMEOUT_SECONDS,
        ) as session:
            return await session.call_tool(self._name, arguments)


def _to_tool_result(tool_name: str, result) -> ToolResult:
    """``CallToolResult`` → ``ToolResult`` 변환.

    structured_content가 있으면 그대로, 없으면 text content를 JSON 파싱
    시도 후 실패 시 원문 문자열로 반환한다.

    (필드명은 mcp 2.x 에서 전부 snake_case 로 바뀌었다 — isError/structuredContent
    로 읽으면 AttributeError 다. 생성자 kwarg 만 camelCase 를 계속 받아준다.)
    """
    texts = [c.text for c in (result.content or []) if getattr(c, "text", None)]
    if result.is_error:
        return ToolResult(
            success=False,
            output=None,
            error="; ".join(texts) or f"MCP 도구 '{tool_name}' 실행 오류 (상세 없음)",
        )

    output: Any = result.structured_content
    if output is None:
        joined = "\n".join(texts)
        try:
            output = json.loads(joined)
        except ValueError:
            output = joined
    # MCP 가 데이터를 실은 실패를 정상 응답으로 돌려준다(`success: false` + `error`).
    # 예외로 던지면 데이터(동명이인 후보 등)가 사라지기 때문이다 — 여기서 다시 실패로 세운다.
    if isinstance(output, dict) and output.get("success") is False:
        data = {k: v for k, v in output.items() if k not in ("success", "error")}
        return ToolResult(
            success=False,
            output=data or None,
            error=str(output.get("error") or f"MCP 도구 '{tool_name}' 실행 실패"),
        )
    return ToolResult(success=True, output=harvest_mcp_usage(tool_name, output))
