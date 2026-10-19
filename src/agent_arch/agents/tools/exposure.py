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
from typing import List, Optional, Sequence, Tuple

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

    채널별 선택 노출(per-channel exposure)용 primitive. 특정 진입점에서만 도메인을
    열고 싶을 때, 그 진입점에서 이 함수로 관문을 static_tools 에 얹는다 — 예: 음성
    ``/audio/*``, MES 채팅 ``/agent/mes/*``. 일반 경로에서는 호출하지 않으므로 해당
    도메인이 발견·호출·광고되지 않는다(fail-closed 기본값).

    관문 자체는 tools_config.yaml 에서 전역 OFF(enabled=false)로 둘 수 있고, 그래도
    ``ToolRegistry.get`` 은 import 시 등록된 인스턴스를 돌려주므로 여기서 얹을 수 있다.
    관문을 얹는 것은 "노출 후보"에 넣는 것일 뿐, 실제 바인딩은 이후 ``plan_tools`` 의
    권한 게이트(``{service}.access``)를 통과해야 한다 — 즉 권한 부여가 아니다.

    Args:
        tools: 기존 도구 목록(보통 ``orchestrator.static_tools``).
        gateway_name: 얹을 관문 도구 이름(예: ``"find_logistics_tools"``).

    Returns:
        관문을 더한 새 리스트. 관문 미등록이거나 이미 목록에 있으면 원본 사본 그대로.
    """
    gateway = ToolRegistry.get(gateway_name)
    if gateway is None or any(t.name == gateway.name for t in tools):
        return list(tools)
    return [*tools, gateway]


# 물류 전용 봇(음성 /audio/* · MES 앱 채팅 /agent/mes/*)이 고정으로 쓰는 LLM provider.
#
# 이 두 채널은 앱이 ``llm_provider`` 를 안 실어 보내서 서버 기본값(.env DEFAULT_LLM_PROVIDER)
# 으로 떨어지는데, 그 값이 바뀌면(현재 openai — 크레딧 소진 429) 물류 봇이 통째로 죽는다.
# 채널이 이미 agent_mode·도구를 강제하는 전용 봇이므로 provider 도 채널 계약에 포함시킨다.
#
# 왜 gemini 인가:
#   - providers.gemini 설정(temperature 0.4 · thinking_budget 0)이 도구 체이닝 정밀도를
#     기준으로 맞춰진 값이라 물류 REACT 루프에 그대로 맞다(config/llm_providers.yaml 주석).
#   - ``_VISION_PROVIDERS``(gemini·openai) 에 속해 주력 LLM 을 이미지 분석에 재사용한다 —
#     현장 사진 입력 시 vision_llm_provider(기본 openai) 별도 인스턴스를 타지 않는다.
#   - 병렬 eval 매핑(_PARALLEL_PROVIDER_MAP)에 gemini→vllm 이 있어 비교 데이터가 계속 쌓인다.
#
# 사용법: ``resolve_effective_provider`` 의 ``requested`` 인자로 넘긴다. 회사별 provider
# 게이트를 켜면 허용목록 검사를 그대로 통과해야 하므로, 게이트를 우회해 orchestrator 에
# 직접 넣지 않는다(미허용 회사는 vllm 폴백).
LOGISTICS_BOT_LLM_PROVIDER = "gemini"


# MES 앱 PoC 채널이 직접 바인딩하는 레거시 조회 도구(동적 API)의 도메인 키.
# llm_tools.json 의 external_service="logistics_service" 가 normalize_service 를 거쳐
# 이 키가 된다 (permissions/services.py SERVICE_ALIAS — 그 주석이 근거).
_MESPOC_DOMAIN = "mespoc"

# 동적 조회 도구 인스턴스 캐시. 정의는 기동 시 한 번 로드되고 런타임에 바뀌지 않으며,
# DynamicAPITool 생성자는 정의만 캡처하고 사용자 컨텍스트는 실행 시점에 읽으므로
# (dynamic_api_tool.py `_execute`) 요청 간 인스턴스 재사용이 안전하다 — ToolRegistry 의
# 정적 스킬과 같은 수명이다.
_mespoc_browse_cache: Optional[List[BaseTool]] = None


# MES 앱 PoC 채널이 도메인 도구와 함께 유지하는 공용 도구(core 에서 살려 두는 이름).
#
# - calculator: 수량·비율 계산은 도메인과 무관하게 쓰인다.
# - 메모리 4종: 회상 안정 블록이 "주입된 기억으로 부족하면 memory_search / '기억해둬'는
#   memory_remember / 지워 달라면 memory_forget" 을 지시한다(memory_v2 recall.TOOLS_GUIDE).
#   도구가 없으면 그 지시가 죽고, 특히 "저장 도구가 없다고 답하지 말 것"까지 못 박혀 있어
#   "기억해둬" 요청이 막다른 길이 된다. MES 앱은 일반 채팅과 분리된 창이므로 그 창 안에서
#   기억을 찾고 남기는 경로가 따로 있어야 한다. 나머지 메모리 도구(memory_read_scene·
#   recall_conversation)는 retrieved 티어라 find_memory_tools 관문으로 발견된다.
#
# 이름으로 고르는 이유: tools_config 에서 끈 도구는 core 에 애초에 없으므로, 여기서도
# 자동으로 빠진다(전역 스위치를 우회하지 않는다).
_CHANNEL_KEEP_TOOLS: frozenset[str] = frozenset(
    {
        "calculator",
        "find_memory_tools",
        "memory_search",
        "memory_remember",
        "memory_forget",
    }
)


def _mespoc_browse_tools() -> List[BaseTool]:
    """MES 앱 PoC 도메인의 동적 조회 도구(목록·현황·대시보드)를 돌려준다.

    큐레이션 스킬 10종(ToolRegistry 등록)은 LOT·시리얼·제조코드로 **단건**을 집는
    진입만 받는다. "제조작업지시 뭐 있지?" 같은 목록·현황 질문에 답할 도구는 동적
    API 쪽(``mespoc_service_*``)에만 있는데, 이들은 ToolRegistry 에 없어 관문
    (``find_mespoc_tools``) 검색으로만 발견된다. 그런데 채널이 스킬 10종을 이미 직접
    바인딩해 주므로 모델은 "내 도구 목록에 목록 조회가 없다 = 그 기능은 없다"로
    단정하고 관문을 열지 않는다 — 실제로 운영에서 "제조작업지시를 직접 조회하는
    기능은 가지고 있지 않습니다"라고 답하고 도구를 한 개도 부르지 않았다(2026-09-22
    이력). 그래서 조회 도구도 채널이 직접 바인딩한다(관문은 fallback 으로 유지).

    노출 범위는 이 함수를 부르는 채널(음성 ``/audio/*`` · MES 채팅 ``/agent/mes/*``)로
    한정된다 — 일반 채팅은 이 경로를 타지 않고, 관문도 전역 OFF 라 그대로 차단이다.
    실제 실행 권한은 이후 ``plan_tools`` 의 권한 게이트가 판단한다(노출≠권한).

    Returns:
        ``DynamicAPITool`` 리스트. 정의를 읽지 못하면 빈 리스트 — 그 경우 채널은
        스킬 10종 + 관문만으로 종전대로 동작한다.
    """
    global _mespoc_browse_cache
    if _mespoc_browse_cache is not None:
        return _mespoc_browse_cache
    # 순환 import 회피 (exposure ← tools ← api_registry ← agents)
    from agent_arch.agents.tools.dynamic_api_tool import DynamicAPITool
    from agent_arch.services.api_registry import tool_loader

    try:
        defs = tool_loader.defs_by_domain(_MESPOC_DOMAIN)
    except Exception as e:  # 파일 부재·파싱 실패 — 채널을 죽이지 않는다
        logger.warning("[exposure] MES PoC 조회 도구 로드 실패 — 스킬만 바인딩: %s", e)
        return []
    _mespoc_browse_cache = [
        DynamicAPITool(d) for d in defs if d.origin == "dynamic_api"
    ]
    return _mespoc_browse_cache


def apply_logistics_bot(orchestrator) -> None:
    """orchestrator 를 'MES 앱 PoC 봇'으로 만든다 (음성 /audio/* · MES 채팅 /agent/mes/* 공용).

    2026-08-24 도메인 스왑: 이 채널이 바인딩하는 도구는 레거시 물류 서비스 연동
    스킬(mespoc_*)이다 — "logistics" 도메인·find_logistics_tools 는 물류 MCP
    (erp, 일반 채팅)가 소유하므로 여기서 건드리지 않는다. agent_mode 명
    "logistics" 는 프론트·오케스트레이터 계약이라 유지한다(= MES 앱 PoC 채널).

    두 채널(음성·MES채팅)이 동일하게 쓰는 채널 스코핑 primitive:

    1. ``agent_mode = "logistics"`` 강제 — 전략상 REACT(도구 실행)로 동작하고
       LOGISTICS_SYSTEM_PROMPT 를 써 "뭘 할 수 있냐"에 PoC 물류만 소개한다. 앱
       프론트가 보내는 모드(guide 등)를 무시한다.
    2. ``static_tools`` 를 공용 도구(``_CHANNEL_KEEP_TOOLS`` — calculator·메모리 4종)
       + PoC 스킬(``mespoc_`` 접두사) 전체 + PoC 조회 도구(``_mespoc_browse_tools``)
       + 관문(find_mespoc_tools, fallback) 로 재구성 — 다른 **업무 도메인**의 탐색도구·
       디렉터리를 제거한다. 스킬은 단건 진입(LOT·
       시리얼·제조코드)이고 목록·현황은 조회 도구 몫이라 둘 다 필요하다(근거는
       ``_mespoc_browse_tools`` 주석). PoC 도구를 관문 뒤로 숨기지 않는 이유는,
       관문+sticky 발견에만 의존하면 후속 행동 턴에서 직전에 '호출된' 도구(주로 조회)만
       되살아나고 상태변경 등 write 도구가 안 살아나, 에이전트가 능력을 못 찾고 되묻는
       멀티턴 실패가 나기 때문이다(단일 도메인이라 잡탕 노이즈 우려 없음). 다른 도메인
       스킬은 각자의 find_*_tools 가 없으면 발견·호출이 **불가능**하므로 PoC 외 노출은
       그대로 차단된다.

    attribute 오버라이드는 안전 — orchestrator __init__ 은 mode 를 저장만 하고, 전략·
    프롬프트·바인딩은 run/stream 시점 self.agent_mode/static_tools 로 결정된다. 권한
    게이트(plan_tools)도 그대로 적용된다 — mespoc 은 PUBLIC 이라 사용권 검사는 없지만
    미인증·게스트 deny 는 남는다(근거: skills/mes_poc 권한 상수 주석).

    Args:
        orchestrator: AgentOrchestrator 인스턴스(``agent_mode``/``static_tools`` 속성 보유).
    """
    orchestrator.agent_mode = "logistics"
    # PoC 봇은 이미 채널단에서 단일 도메인(도구 10여 개)으로 좁혀져 있어, 여러 도메인
    # 잡탕을 막으려던 발견 계층(관문 + sticky)이 오히려 후속 행동 턴에서 write 도구를
    # 못 살려 되묻게 만든다. 그래서 PoC 도구(mespoc_ 접두사) 전체를 직접 static 으로
    # 바인딩한다. 관문은 fallback 으로 함께 남겨 등록 누락 시 자기치유되게 하고, 권한
    # 게이트(PUBLIC — 미인증·게스트 deny)는 이후 plan_tools 에서 그대로 적용된다
    # (노출≠실행권한).
    kept = [t for t in orchestrator.static_tools if t.name in _CHANNEL_KEEP_TOOLS]
    poc_tools = [
        tool
        for name, tool in ToolRegistry.get_all().items()
        if name.startswith("mespoc_")
    ]
    orchestrator.static_tools = with_domain_gateway(
        kept + poc_tools + _mespoc_browse_tools(), "find_mespoc_tools"
    )


# SCMS 채널이 노출하는 도구(등록명).
#
# 조회 4종(scms_*)은 scms-mcp 서버가 기동 시 등록한다(mcp/provider.py `_SCMS_POLICIES`).
# 서버가 안 떠 있으면 registry 에 없어 그냥 빠지므로, 이름을 미리 적어 둬도 안전하다 —
# 아래 조립이 `ToolRegistry.get_all()` 에 실제로 있는 것만 집기 때문이다.
#
# 쓰기 도구(견적요청 등록·견적 제출·낙찰)를 추가할 때는 여기만 고치면 안 된다:
# hitl/policy.py 의 APPROVAL_REQUIRED_TOOLS 에 함께 넣어야 승인 없이 실행되지 않는다.
_SCMS_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # 공급사 추천 — 2026-09-30 내림. 솔버 오류로 모든 요청이 실패한다(cpmpy 1.0 이 소수 단가 곱을 거부).
        # 고쳐서 다시 켜면 SCMS 프롬프트(웹·메신저 사본)에서 뺀 안내 문장도 되돌린다 — 프롬프트 위 주석 참조.
        # "supplier_recommend",
        "scms_purchase_order_draft",
        # scms-mcp 조회 도구 — 현황
        "scms_context",
        "scms_my_status",
        "scms_board",
        "scms_order",
        # scms-mcp 조회 도구 — 업무 영역
        "scms_settlement",
        "scms_quality",
        "scms_vendor",
        "scms_my_performance",
        "scms_pivot",
        "scms_master",
        # scms-mcp 조회 도구 — 프로세스 안내(입고/검수 7단계)
        "scms_process",
        # scms-mcp 조회 도구 — 정산 회차 해설
        "scms_payment",
        # scms-mcp 조회 도구 — 견적 비교·낙찰 보조
        "scms_bids",
        # scms-mcp 변경 요청 검토(단가·수량·납기 변경 요청 → 차이·반영 화면·회신 재료)
        "scms_change_review",
        # scms-mcp 업무 절차 안내 관문("어떻게 하지?" 의 기본 도구)
        "scms_guide",
        # scms-mcp 체결된 계약서 읽기(서명본·지급조건·발주와 어긋난 곳)
        "scms_contract",
        # scms-mcp 정정요청 및 문의(VOC) — 관점별 가능 액션까지
        "scms_voc",
        # scms-mcp 품목 과거 단가 이력(견적 기준선)
        "scms_price_history",
        # scms-mcp 계약서 파트 쓰기 — 승인 필수(hitl/policy). 메신저에서는 계약서 카드 버튼이 주로 부른다.
        "scms_award",
        "scms_order_change",
        "scms_change_respond",
    }
)


# 메신저 SCMS 봇에만 더 얹는 도구(apply_scms_messenger_bot). registry 에 있는 것만 집는다 — MCP(mattermost_read_file)가
# 안 떠 있으면 그냥 빠진다.
_SCMS_MESSENGER_EXTRA_TOOLS: frozenset[str] = frozenset({
    "scms_quote_agreement",   # 견적서 → 합의본 → 네고 → 양측 확인
    "scms_quote_submit",      # 공급사 전용 카드(scms_card) 전달 → 클릭 때 SCMS 견적 적재
    "mattermost_read_file",   # 견적서 밖의 파일(발주 내역 등)을 사용자가 물을 때 — 보조
    "scms_contract_start",    # 계약서 파트 시작·재개 — 단계에 맞는 카드를 당사자에게 보낸다
})


def apply_scms_bot(orchestrator) -> None:
    """orchestrator 를 'SCMS 공급망 전용 봇'으로 만든다 (SCMS 채팅 /agent/scms/*).

    ``apply_logistics_bot`` 과 같은 채널 스코핑 primitive 이며, 두 가지를 강제한다:

    1. ``agent_mode = "scms"`` 고정 — 앱 프론트가 보내는 모드(guide 등)를 무시한다.
       scms 모드는 (a) 전략상 REACT(도구 루프)로 동작해 소싱 도구가 실제 실행되고 —
       guide/demo 는 각각 KB 검색·DirectLLM 이라 도구를 안 돈다 — (b) SCMS 전용 시스템
       프롬프트(SCMS_SYSTEM_PROMPT)를 써 "뭘 할 수 있냐"에 조달만 소개한다.
    2. ``static_tools`` 를 calculator + SCMS 도구로 재구성 — 다른 도메인 탐색도구
       (find_messenger/mail/hr/... )·디렉터리·메모리를 제거한다. 프롬프트가 다른 도메인을
       광고하지 않으므로 도구도 조달로 좁혀 광고·실행을 일치시킨다.

    물류 채널과 달리 도메인 관문(find_*_tools)을 얹지 않는다 — SCMS 관문이 없고 도구가
    한 개라 관문 뒤에 숨길 이유가 없다(관문+sticky 발견은 후속 행동 턴에서 도구를 못
    살리는 실패가 있었다 — apply_logistics_bot 주석 참조).

    attribute 오버라이드는 안전 — orchestrator __init__ 은 mode 를 저장만 하고, 전략·
    프롬프트·바인딩은 run/stream 시점 self.agent_mode/static_tools 로 결정된다. 노출을
    얹은 뒤에도 요청자 권한 게이트(plan_tools)는 그대로 적용되므로, 이 오버라이드가
    권한을 부여하지는 않는다.

    Args:
        orchestrator: AgentOrchestrator 인스턴스(``agent_mode``/``static_tools`` 속성 보유).
    """
    orchestrator.agent_mode = "scms"
    kept = [t for t in orchestrator.static_tools if t.name == "calculator"]
    scms_tools = [
        tool for name, tool in ToolRegistry.get_all().items() if name in _SCMS_TOOL_NAMES
    ]
    orchestrator.static_tools = kept + scms_tools
    _use_scms_model(orchestrator)


def _scms_lane_cfg() -> dict:
    from agent_arch.core.config import settings

    return (getattr(settings, "llm_providers", None) or {}).get("scms_agent") or {}


def scms_agent_lane() -> Tuple[str, Optional[str]]:
    """SCMS 전용 봇의 대화 프로바이더·모델 (config/llm_providers.yaml: scms_agent). 블록이 없으면 ("", None)."""
    lane = _scms_lane_cfg()
    return str(lane.get("provider") or ""), lane.get("model") or None


def _use_scms_model(orchestrator) -> None:
    """정해진 프로바이더가 SCMS 레인 프로바이더면 레인 모델로 바꿔 끼운다(그 프로바이더의 전역 모델이 아니라).

    프로바이더는 진입점이 정한다(SCMS_CHANNEL_LLM_PROVIDER → 레인 → 회사 허용 정책). 정책이 다른 프로바이더로 내렸거나
    운영자가 다른 프로바이더를 고정했으면 그 프로바이더의 기본 모델을 그대로 쓴다 — 그 프로바이더에 sol 을 넘길 수 없다.
    """
    provider, model_name = scms_agent_lane()
    if not model_name or not provider or getattr(orchestrator, "llm_provider", None) != provider:
        return
    from agent_arch.llm.factory import LLMFactory

    try:
        extra = {k: v for k, v in _scms_lane_cfg().items() if k in ("api", "reasoning_effort") and v not in (None, "")}
        orchestrator.llm = LLMFactory.create(provider, model=model_name, **extra)
    except Exception:  # noqa: BLE001 — 모델을 못 만들면 프로바이더 기본 모델로 간다
        logger.warning("[exposure] SCMS 레인 모델(%s/%s)을 만들지 못함 — 프로바이더 기본 모델로", provider, model_name,
                       exc_info=True)


# 공개 범위(scms.deal.visibility) — 원청이 있을 수 있는 채널에서 플랫폼 쪽 사람이 부르면 이 조회 도구들을 뺀다.
# 플랫폼(회사)은 SCMS 에서 발주사로만 서므로 그 세션의 조회 결과는 사실상 전부 하도급 쪽(금형사 발주·견적·정산·계약)이다.
# 답은 채널 전원이 본다. 절차 안내(scms_process·scms_guide)와 쓰기 흐름(발주요청·견적 합의)은 남긴다.
_SCMS_INTERNAL_LOOKUP_TOOLS: frozenset[str] = _SCMS_TOOL_NAMES - {"scms_purchase_order_draft", "scms_process", "scms_guide"}


SCMS_LOOKUP_WITHHELD_REASON = (
    "이 채널에는 원청 쪽 사람이 있을 수 있어 SCMS 내부 조회를 하지 않습니다(답은 채널 전원이 봅니다). "
    "조회 결과를 말하지 말고, 필요하면 에이전트에게 DM 으로 물어 달라고 안내하세요."
)


def withhold_scms_lookup(orchestrator) -> List[str]:
    """이번 턴에서 SCMS 내부 조회 도구를 막는다(정적 목록에서 빼고, 턴 컨텍스트로 바인딩·실행도 막는다)."""
    removed = [t.name for t in orchestrator.static_tools if t.name in _SCMS_INTERNAL_LOOKUP_TOOLS]
    orchestrator.static_tools = [t for t in orchestrator.static_tools if t.name not in _SCMS_INTERNAL_LOOKUP_TOOLS]
    orchestrator.withheld_tools = {name: SCMS_LOOKUP_WITHHELD_REASON for name in _SCMS_INTERNAL_LOOKUP_TOOLS}
    return removed


def apply_scms_messenger_bot(orchestrator) -> None:
    """메신저 대화를 SCMS 전용 봇으로 만든다 (SCMS 전용 웹훅 /webhooks/mattermost/scms 의 멘션).

    ``apply_scms_bot`` 을 그대로 걸고, 발주요청 초안 도구만 메신저용으로 바꾼다. 웹 채널의
    ``scms_purchase_order_draft`` 는 승인 뒤 SCMS 프론트가 화면을 열어 초안을 만드는데
    메신저에는 그 프론트가 없다 — 승인이 아무 일도 안 일으킨다. 대신 서버가 초안을 만들고
    링크를 주는 ``scms_order_request_draft``(스레드 도면은 도구가 직접 판독한다)를 켠다. 이 도구는
    ``scms_order_enabled`` 가 켜진 요청에서만 ``_prepare_tools`` 가 결정적으로 붙인다.

    Args:
        orchestrator: AgentOrchestrator 인스턴스.
    """
    apply_scms_bot(orchestrator)
    orchestrator.static_tools = [
        t for t in orchestrator.static_tools if t.name != "scms_purchase_order_draft"
    ]
    # 메신저에서만 더 얹는 도구 — 견적 합의(scms_quote)와 파일 읽기. 웹 SCMS 패널(_SCMS_TOOL_NAMES)에는 넣지 않는다:
    # 패널에는 Mattermost 파일이 없고, 합의본은 방(채널) 단위로 쌓인다. 등록 도구는 승인 필수(hitl/policy).
    have = {t.name for t in orchestrator.static_tools}
    orchestrator.static_tools += [
        tool for name, tool in ToolRegistry.get_all().items()
        if name in _SCMS_MESSENGER_EXTRA_TOOLS and name not in have
    ]
    orchestrator.scms_order_enabled = True
    # 시스템 프롬프트는 메신저용 사본으로(prompts/scms_messenger) — 웹 SCMS 채팅 프롬프트와 따로 관리한다.
    orchestrator.prompt_surface = "messenger"


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
