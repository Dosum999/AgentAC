"""도구 노출(Exposure) 정책 — "어디서 왔나(소스)"와 "언제 LLM에 보이나"의 분리.

배경:
    도구 소스가 3종(정적 스킬·MCP·동적 API)으로 늘면서, 소스가 노출 방식을
    강제하는 구조(정적=상시, 동적=검색)로는 도구 증가가 곧 상시 바인딩 증가로
    이어진다. 본 모듈은 노출 결정을 한 곳으로 모으고 도구별 정책 메타를 둔다.

티어:
    CORE         항상 바인딩. 정원 상한(_CORE_CAP) 초과 시 경고.
    RETRIEVED    질의 유사도 검색에 걸릴 때만 바인딩 (3단계부터 유효 —
                 현재는 동적 API 도구가 사실상 이 티어).
    DISCOVERABLE 예약. 메타도구(tool_search) 기반 2단 노출 — 미구현이므로
                 지정 시 경고 후 CORE로 폴백 (도구가 사라지는 사고 방지).

1단계 계약: 모든 등록 도구의 기본값이 CORE이므로 본 모듈 도입 전후의
바인딩 결과는 동일하다. 노출 변경(티어 강등)은 tool_selection eval로
전후 비교 후에만 수행한다 (evidence-gated).
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import List, Optional, Sequence

from agent_arch.agents.permissions.gate import is_allowed
from agent_arch.agents.tools.base import BaseTool
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

# CORE 정원 상한 — 초과해도 차단하지 않고 경고만 한다. 수치 근거가 쌓이기
# 전까지는 운영 신호용 (현행 정적 도구가 이미 ~25개라 도입 직후 경고가 정상).
_CORE_CAP = 15

_cap_warned = False
_flag_off_warned = False
_fallback_warned: set[str] = set()


class Exposure(str, Enum):
    """도구 노출 티어. 자세한 의미는 모듈 docstring 참조."""

    CORE = "core"
    RETRIEVED = "retrieved"
    DISCOVERABLE = "discoverable"


def resolve_exposure(tool: BaseTool) -> Exposure:
    """도구의 유효 노출 티어를 해석한다.

    우선순위: tools_config.yaml 의 ``exposure`` > 클래스/인스턴스 속성 > CORE.
    잘못된 문자열·미구현 티어(DISCOVERABLE)는 경고 후 CORE 폴백 — 설정 실수로
    도구가 조용히 사라지는 것보다 과노출이 안전하다.
    """
    raw = settings.get_tool_config(tool.name).get("exposure") or getattr(
        tool, "exposure", Exposure.CORE
    )
    try:
        exposure = Exposure(raw)
    except ValueError:
        _warn_fallback(tool.name, f"알 수 없는 exposure '{raw}'")
        return Exposure.CORE

    if exposure is Exposure.DISCOVERABLE:
        _warn_fallback(tool.name, "DISCOVERABLE은 미구현(2단 노출 도입 전)")
        return Exposure.CORE
    return exposure


def core_tools() -> List[BaseTool]:
    """상시 바인딩 대상(CORE) 도구를 반환한다.

    orchestrator가 요청마다 호출한다 — MCP처럼 늦게 등록되는 도구도
    다음 요청부터 반영되는 기존 특성을 유지한다.

    안전 폴백: UNIFIED_TOOL_INDEXING이 꺼져 있으면 retrieved 도구는 검색에
    걸릴 수 없으므로(인덱스 부재) core로 승격해 바인딩을 유지한다 —
    플래그/설정 불일치로 도구가 조용히 사라지는 사고 방지.
    """
    unified = settings.unified_tool_indexing
    tools: List[BaseTool] = []
    promoted = 0
    for t in ToolRegistry.get_enabled_tools():
        tier = resolve_exposure(t)
        if tier is Exposure.CORE:
            tools.append(t)
        elif tier is Exposure.RETRIEVED and not unified:
            tools.append(t)
            promoted += 1
    if promoted:
        _warn_flag_off(promoted)
    _warn_core_cap(len(tools))
    return tools


def plan_tools(
    core: Sequence[BaseTool],
    retrieved: Sequence[BaseTool],
    profile: Optional[dict],
) -> List[BaseTool]:
    """티어별 후보를 조립해 최종 바인딩 목록을 만든다.

    조립 규칙:
        1) core 전부 + retrieved (이름 중복 시 core 인스턴스 우선)
        2) 권한 게이트(is_allowed) 통과분만 — profile=None이면 전부 deny

    Args:
        core: 상시 바인딩 후보 (보통 ``core_tools()`` 결과)
        retrieved: 질의 검색 결과 (동적 API 도구 등)
        profile: Frappe profile. 게이트 평가용.

    Returns:
        bind_tools에 넘길 도구 리스트 (core → retrieved 순서 유지)
    """
    seen: set[str] = set()
    candidates: List[BaseTool] = []
    for tool in [*core, *retrieved]:
        if tool.name in seen:
            continue
        seen.add(tool.name)
        candidates.append(tool)

    return [t for t in candidates if is_allowed(t.name, profile)]


def with_domain_gateway(tools: Sequence[BaseTool], gateway_name: str) -> List[BaseTool]:
    """도메인 관문 메타도구(find_*_tools)를 도구 목록에 더해 반환한다.

    채널별 선택 노출(per-channel exposure)용 primitive. 특정 진입점(전용 앱 채널·음성 등)
    에서만 도메인을 열고 싶을 때, 그 진입점에서 이 함수로 관문을 static_tools 에 얹는다.
    일반 경로에서는 호출하지 않으므로 해당 도메인이 발견·호출·광고되지 않는다(fail-closed 기본값).

    관문 자체는 tools_config.yaml 에서 전역 OFF(enabled=false)로 둘 수 있고, 그래도
    ``ToolRegistry.get`` 은 import 시 등록된 인스턴스를 돌려주므로 여기서 얹을 수 있다.
    관문을 얹는 것은 "노출 후보"에 넣는 것일 뿐, 실제 바인딩은 이후 ``plan_tools`` 의
    권한 게이트(``{service}.access``)를 통과해야 한다 — 즉 권한 부여가 아니다.

    Args:
        tools: 기존 도구 목록(보통 ``orchestrator.static_tools``).
        gateway_name: 얹을 관문 도구 이름(예: ``"find_<domain>_tools"``).

    Returns:
        관문을 더한 새 리스트. 관문 미등록이거나 이미 목록에 있으면 원본 사본 그대로.
    """
    gateway = ToolRegistry.get(gateway_name)
    if gateway is None or any(t.name == gateway.name for t in tools):
        return list(tools)
    return [*tools, gateway]


def _warn_core_cap(count: int) -> None:
    global _cap_warned
    if count > _CORE_CAP and not _cap_warned:
        _cap_warned = True
        logger.warning(
            "[exposure] CORE 도구 %d개 — 상한(%d) 초과. 저빈도 도구의 "
            "retrieved 강등 검토 필요 (tools_config.yaml exposure 항목)",
            count, _CORE_CAP,
        )


def _warn_flag_off(promoted: int) -> None:
    global _flag_off_warned
    if not _flag_off_warned:
        _flag_off_warned = True
        logger.warning(
            "[exposure] UNIFIED_TOOL_INDEXING off — retrieved 도구 %d개를 "
            "core로 승격 유지 (강등을 적용하려면 플래그를 켜세요)", promoted,
        )


def _warn_fallback(tool_name: str, reason: str) -> None:
    if tool_name not in _fallback_warned:
        _fallback_warned.add(tool_name)
        logger.warning("[exposure] %s: %s — CORE로 폴백", tool_name, reason)
