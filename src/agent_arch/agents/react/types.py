"""ReAct 루프 이벤트 및 결과 타입 정의.

ProgressEvent: 루프 진행 중 스트리밍 SSE로 변환할 이벤트
LoopResult:    루프 완료 후 반환되는 최종 결과 (항상 마지막으로 yield됨)
merge_usage:   LLM 호출 간 토큰 수를 누적하는 유틸리티 함수
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class ProgressEvent:
    """ReAct 루프 진행 이벤트.

    스트리밍 모드에서는 SSE JSON으로 변환되어 클라이언트에 전달된다.
    비스트리밍 모드(run())에서는 무시된다.

    Attributes:
        event_type: 이벤트 종류
        content: 클라이언트에 표시할 메시지
        name: tool_start / tool_end 시 도구 이름 (선택)
            (link_policy 이벤트는 content 에 정책 JSON 을 싣는다)
    """

    event_type: Literal[
        "thinking",
        "searching",
        "answering",
        "tool_start",
        "tool_end",
        "token",
        # 이번 턴의 링크 허용 정책 스냅샷(JSON). orchestrator 가 StreamSanitizer 에 먹이고
        # **SSE 로는 내보내지 않는다** — 화면 계약이 아니라 정화기에 상태를 넘기는 내부 통로다
        # (`agents/react/link_policy.py`). 도구 결과가 확정된 직후, 답변 토큰보다 먼저 나간다.
        "link_policy",
    ]
    content: str = ""
    name: str = ""


@dataclass
class LoopResult:
    """ReAct 루프 최종 결과.

    react_loop_events / guide_loop_events가 마지막으로 yield하는 객체.
    run()에서는 이 값을 수신해 DB에 저장하고 응답 dict를 구성한다.
    stream()에서는 content를 클라이언트로 전송한 뒤 DB에 저장한다.

    Attributes:
        content: 최종 응답 텍스트
        tool_calls: 이번 턴에 실행된 도구 호출 목록 (DB 저장용)
        tool_results: 이번 턴 각 도구의 실행 결과 (Redis tool_ctx 저장용)
            [{"tool_call_id": str, "name": str, "content": str}, ...]
        llm_invoke_count: 총 LLM 호출 횟수
        token_usage: 누적 토큰 사용량
        finish_reason: 마지막 LLM 응답의 finish_reason
        guide_hits: guide 모드 전용 — KB 검색 히트 수
        abstain: guide 모드 전용 — KB 검색이 기권(가이드로 답할 수 없는 질문)한 경우 True.
            상위 로직이 "AI 에이전트 모드로 전환할까요?" 제안 신호(suggested_mode)로 승격한다.
        abstain_reason: guide 모드 전용 — 기권 사유(REASON_* 상수). abstain=True일 때만 의미.
        pending_action: HITL 전용 — 승인필요 액션이 confirm=False 초안만 만들고
            턴을 종결했을 때의 {action_id, tool_name, arguments, draft_view}.
            None이면 일반 턴. 설정되면 orchestrator가 HITL 그래프에 체크포인트해
            사람 승인(재개) 대기 상태로 둔다.
    """

    content: str
    tool_calls: list = field(default_factory=list)
    tool_results: list = field(default_factory=list)
    llm_invoke_count: int = 0
    token_usage: dict = field(
        default_factory=lambda: {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
    )
    finish_reason: str = "stop"
    guide_hits: int = 0
    abstain: bool = False
    abstain_reason: Optional[str] = None
    pending_action: Optional[dict] = None
    # 멀티-액션 HITL: 한 턴에 승인필요 액션이 여럿이면 모두 담는 큐. pending_action 은
    # 하위호환을 위해 큐[0]을 미러. None이면 일반 턴. 큐 순서대로 각자 승인/거절 받는다.
    pending_queue: Optional[list] = None
    # 그래프 전용 스텝 계측(LangSmith 메타데이터 + DB metadata 컬럼). react_loop_events 는 None.
    # {node_trace, node_timings, iterations, llm_invoke_count, retry_any, discovered_tools, finish_reason}
    step_meta: Optional[dict] = None


def merge_usage(acc: dict, usage: Optional[dict]) -> None:
    """LLM 호출 결과의 토큰 사용량을 누적 딕셔너리에 합산한다.

    Args:
        acc: 누적용 딕셔너리 {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        usage: LLMResponse.usage 딕셔너리 또는 None
    """
    if not usage:
        return
    acc["prompt_tokens"] += usage.get("prompt_tokens", 0)
    acc["completion_tokens"] += usage.get("completion_tokens", 0)
    acc["total_tokens"] += usage.get("total_tokens", 0)
