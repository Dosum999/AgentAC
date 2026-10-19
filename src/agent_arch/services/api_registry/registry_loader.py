"""ToolRegistry(정적 스킬·MCP 도구) → 인덱스용 ExternalToolDef 변환.

retrieved 티어 도구가 질의 검색에 걸리려면 동적 API와 같은 컬렉션에
인덱싱돼야 한다. 단, registry 항목은 **이름+설명만** 싣는다 — 파라미터·
권한·실행 정보는 ToolRegistry의 실물 인스턴스가 소유하며, 검색 히트 시
이름으로 실물을 resolve해 쓴다 (권한 이중 등록 원천 차단).
"""
from __future__ import annotations

from typing import List

from agent_arch.agents.permissions import normalize_service
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.services.api_registry.models import ExternalToolDef


def _domain_of(tool) -> str | None:
    """도구의 도메인(정규화된 권한 서비스 키)을 도출한다. PUBLIC/무소속은 None."""
    req = getattr(tool, "permission_required", None)
    service = getattr(req, "service", "") if req else ""
    return normalize_service(service) if service else None


def defs_from_registry(only_names: List[str] | None = None) -> List[ExternalToolDef]:
    """등록된 도구를 인덱스용 정의로 변환한다.

    Args:
        only_names: 지정 시 해당 이름의 도구만 (MCP 증분 인덱싱용).
            None이면 enabled 전체.

    Returns:
        origin="registry"인 ExternalToolDef 리스트
    """
    tools = ToolRegistry.get_enabled_tools()
    if only_names is not None:
        wanted = set(only_names)
        tools = [t for t in tools if t.name in wanted]

    return [
        ExternalToolDef(
            tool_name=t.name,
            description=t.description,
            origin="registry",
            domain=_domain_of(t),
            is_external=False,
            enabled=True,
            parameters_schema={},
        )
        for t in tools
    ]
