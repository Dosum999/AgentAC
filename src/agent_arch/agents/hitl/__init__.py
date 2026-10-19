"""HITL(Human-In-The-Loop) — 비가역 액션의 코드 강제 일시정지/재개.

승인이 필요한 액션 도구에 도달하면 LLM이 아니라 시스템이 강제로 멈추고
(interrupt + checkpoint), 사람 승인 시 정확히 그 지점부터 재개(Command resume)한다.
전체 ReAct 루프는 그대로 두고 액션 실행 지점만 LangGraph로 감싼다.
"""
from agent_arch.agents.hitl.policy import (
    APPROVAL_REQUIRED_TOOLS,
    REJECTED_DRAFT_NOTE_HEADER,
    action_done_message,
    build_rejected_draft_note,
    requires_approval,
)
from agent_arch.agents.hitl.graph import (
    HitlState,
    build_action_graph,
    execute_action_node,
    get_action_graph,
    interrupt_payload,
)
from agent_arch.agents.hitl.resume import (
    ResumeInProgress,
    StalePendingAction,
    load_pending,
    resume_action,
    resume_guard,
    validate_resume_values,
)

__all__ = [
    "APPROVAL_REQUIRED_TOOLS",
    "REJECTED_DRAFT_NOTE_HEADER",
    "action_done_message",
    "build_rejected_draft_note",
    "requires_approval",
    "HitlState",
    "build_action_graph",
    "execute_action_node",
    "get_action_graph",
    "interrupt_payload",
    "ResumeInProgress",
    "StalePendingAction",
    "load_pending",
    "resume_action",
    "resume_guard",
    "validate_resume_values",
]
