"""도메인 확장 지점(extension points).

에이전트 코어(라우터·관문·HITL·오케스트레이터·PII)는 특정 업무 도메인을 import 하지 않는다.
도메인이 코어에 끼어들어야 하는 자리는 전부 여기 슬롯으로 열려 있고, 앱 조립부(composition root)가
기동 시 채운다. 슬롯이 비어 있으면 코어는 "그 기능 없음"으로 결정적으로 동작한다(fail-closed).

코어가 업무 도메인(구매·물류·예약)·용어 사전 출처·사내 디렉터리 모듈을 직접 import 하면, 도메인을 하나
끌 때마다 코어 4곳을 함께 고쳐야 한다. 그래서 이 자리들을 슬롯으로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


# (tool_name, kwargs) -> (해석된 kwargs, settled). settled 가 None 이 아니면(모호·미존재) 그 ToolResult 를
# 그대로 돌려주고 원격 호출을 하지 않는다 — 되묻기는 모델 몫이다.
PersonResolver = Callable[[str, dict], Awaitable[Tuple[dict, Optional[Any]]]]
# 승인된 비가역 액션이 성공한 뒤 도메인 후처리. 반환 문자열은 결과 카드의 result_note 에 덧붙는다.
PostApprovalHook = Callable[[dict, dict], Awaitable[Optional[str]]]


@dataclass
class Extensions:
    # 검색 인덱스를 거치지 않고 기능 플래그로 붙는 도메인 도구(domain -> 도구 목록 팩토리).
    domain_tools: Dict[str, Callable[[], list]] = field(default_factory=dict)
    # 라우터가 도메인을 후보로 낼지(domain -> 사용 가능 여부). 미등록 도메인은 True.
    domain_available: Dict[str, Callable[[], bool]] = field(default_factory=dict)
    # 도구 이름 -> 승인 후 후처리 훅.
    post_approval: Dict[str, PostApprovalHook] = field(default_factory=dict)
    # MCP 도구 인자의 사람 이름 -> 식별자 해석.
    person_resolver: Optional[PersonResolver] = None
    # PII 가드가 쓰는 디렉터리 클라이언트 팩토리(.enabled 속성과 fetch 대상).
    directory_client: Optional[Callable[[], Any]] = None
    # 용어 사전 출처. services/glossary 가 캐시·매칭·교정을 하고 출처만 여기서 받는다.
    glossary_source: Optional[Callable[[], Awaitable[list]]] = None
    # 모드 -> 기본 시스템 프롬프트(코어가 아는 production/guide 외 전용 채널 모드).
    mode_prompts: Dict[str, Callable[[str], str]] = field(default_factory=dict)
    # 가이드 모드 지식베이스 도구 팩토리.
    guide_tool: Optional[Callable[[], Any]] = None
    # 병렬(그림자) 평가 결과 저장소. 키워드 인자로 한 행을 받는다(orchestrator._run_parallel_eval).
    eval_sink: Optional[Callable[..., Awaitable[Any]]] = None


extensions = Extensions()


def domain_tools(domain: str) -> list:
    factory = extensions.domain_tools.get(domain)
    return list(factory()) if factory else []


def domain_available(domain: str) -> bool:
    check = extensions.domain_available.get(domain)
    return check() if check else True

