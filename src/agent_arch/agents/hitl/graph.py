"""HITL 액션 미니그래프 — interrupt(일시정지) + checkpoint + Command(재개).

비가역 액션 실행 지점만 감싸는 단일 노드 StateGraph. 추론 노드가 없으므로
커스텀 ``Message`` ↔ LangChain ``BaseMessage`` 변환이 전혀 필요없다.

흐름:
    1) ReAct 루프가 승인필요 도구를 ``confirm=False`` 로만 실행해 draft view 를 만들고
       ``pending_action`` 으로 넘긴다(부수효과 0).
    2) 이 그래프를 ``ainvoke({"pending_action": ...}, thread_id=conversation_id)`` 하면
       ``execute_action`` 노드가 ``interrupt(draft)`` 로 강제 정지하고 checkpointer 가
       상태를 영속한다(LLM 은 실행 자체가 불가).
    3) 사람 승인 시 ``ainvoke(Command(resume=approved_values), thread_id=...)`` 로 재개 →
       interrupt 지점부터 이어져 ``confirm=True`` 로 실제 실행한다.

실행 시 사용자 명의(게이트웨이 헤더 복원)는 호출자(resume 엔드포인트)가
``set_user_context`` 로 현재 async 컨텍스트에 설정한다.
"""
from __future__ import annotations

import logging
from typing import Any, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.extensions import extensions

logger = logging.getLogger(__name__)


# 거절 센티넬 — /resume 가 거절 시 Command(resume=REJECT_SENTINEL) 로 전달한다.
REJECT_SENTINEL = {"__hitl_reject__": True}


class HitlState(TypedDict, total=False):
    """HITL 미니그래프 상태.

    pending_action: 단일 액션 {action_id, tool_name, arguments(confirm 제외), draft_view} (하위호환).
    pending_queue: 멀티 액션 큐 [위 dict, ...]. 있으면 우선, 없으면 [pending_action].
    index: 현재 처리할 큐 인덱스(체크포인터가 보존 — 재개 간 진행 위치).
    results: 각 액션 실행 결과 누적 [{tool_name, action_id, success, view?, error?, rejected?}].
    result: 직전(=방금) 실행 결과. 하위호환 — 단일/현재 액션 결과 소비자용.
    """

    pending_action: dict
    pending_queue: list
    index: int
    results: list
    result: Optional[dict]


def _queue_of(state: HitlState) -> list:
    """처리할 큐를 얻는다. pending_queue 우선, 없으면 단일 pending_action 1건."""
    q = state.get("pending_queue")
    if q:
        return q
    pa = state.get("pending_action")
    return [pa] if pa else []


async def _execute_one(tool_name: str, draft_args: dict, approved: Any) -> dict:
    """승인된 단일 액션을 confirm=True 로 실제 실행하고 결과 dict 를 반환한다."""
    tool = ToolRegistry.get(tool_name)
    if tool is None:
        logger.warning("[hitl] resume 실행 실패 — 도구 미등록: %s", tool_name)
        return {"tool_name": tool_name, "success": False, "error": f"{tool_name} 도구 미등록"}
    # 초안 인자 base + 승인 시 (수정 반영) 값. confirm 은 코드가 강제(위조 무시).
    exec_args = {**draft_args, **(approved if isinstance(approved, dict) else {})}
    exec_args.pop("confirm", None)
    exec_args.pop("__hitl_reject__", None)
    # 가명화 자리표시가 인자에 남아 있으면 실행하지 않는다(executor 와 같은 검사 — resume 은 executor 를 거치지 않는다).
    from agent_arch.agents.react.executor import unresolved_placeholder_error
    leftover = unresolved_placeholder_error(tool_name, exec_args)
    if leftover is not None:
        logger.warning("[hitl] %s 인자에 자리표시 토큰 잔존 — 실행 차단", tool_name)
        return {"tool_name": tool_name, "success": False, "error": leftover}
    result = await tool.execute(confirm=True, **exec_args)
    if not result.success:
        logger.warning("[hitl] %s 실행 실패: %s", tool_name, result.error)
        return {"tool_name": tool_name, "success": False, "error": result.error}
    out = result.output if isinstance(result.output, dict) else {}
    r: dict = {"tool_name": tool_name, "success": True, "view": out.get("view")}
    hook = extensions.post_approval.get(tool_name)
    if hook is not None:
        # 도메인 후처리(연쇄 생성·팀 알림 등). 실패해도 본 액션은 이미 실행됐으니 결과는 그대로 두고
        # 사유만 남긴다. 돌려준 문장은 승인 카드 결과 줄의 첫 줄 뒤에 붙인다.
        try:
            from agent_arch.agents.context import get_conversation_id, get_user_context

            note = await hook(out, {"user": get_user_context() or {}, "conversation_id": get_conversation_id()})
            view = out.get("view") if isinstance(out.get("view"), dict) else None
            if note and view is not None:
                head, _, rest = str(view.get("result_note") or "").partition("\n")
                view["result_note"] = "\n".join(x for x in (head, note, rest) if x)
        except Exception:
            logger.exception("[hitl] %s 승인 후처리 실패 — 본 실행 결과는 유지", tool_name)
    if out.get("held"):
        # 호출은 성공했지만 목적(상신)까지 못 가고 임시보관으로 끝난 경우(전자결재 create 계약).
        # 고정 완료 문구("상신했어요")를 내면 거짓 성공이 되므로 도구의 사유 문장을 그대로 올린다.
        r["held"] = True
        r["message"] = str(out.get("message") or "").strip() or None
        r["reason_code"] = out.get("reason_code")
        logger.info("[hitl] %s 임시보관 종료 — %s", tool_name, out.get("reason") or r["reason_code"])
    return r


async def execute_action_node(state: HitlState) -> dict:
    """큐의 '현재 인덱스' 액션 1건을 승인 대기(interrupt) → 재개 시 실행한다.

    노드는 항목 1개만 처리하고 index 를 증가시켜 return 한다. 다음 항목은 조건부 엣지가
    노드로 되돌려 처리 → Command(resume) 한 번이 정확히 1개 액션을 실행한다. (인-노드
    루프를 쓰면 재개마다 이전 항목까지 재실행돼 이중 발송이 나므로 쓰지 않는다.)
    interrupt 이전 코드는 재개 시 재실행되므로 부수효과를 두지 않는다.
    """
    queue = _queue_of(state)
    idx = state.get("index", 0)
    results = state.get("results", [])
    # 방어: 인덱스가 큐 범위를 벗어나면(이론상 stale state) 크래시 대신 안전 종료.
    if not queue or idx >= len(queue):
        return {"index": idx}
    item = queue[idx]
    tool_name = item["tool_name"]
    draft_args = item.get("arguments", {}) or {}

    # ── 코드 강제 일시정지 ── interrupt() 반환값 = Command(resume=...) 승인 값/거절 센티넬.
    approved = interrupt(
        {
            "action_id": item.get("action_id"),
            "tool_name": tool_name,
            "arguments": draft_args,
            "draft_view": item.get("draft_view"),
            "queue_index": idx,
            "queue_total": len(queue),
            "is_last": idx == len(queue) - 1,
        }
    )

    # ── 재개 후에만 ── 거절이면 스킵, 아니면 실제 실행.
    if isinstance(approved, dict) and approved.get("__hitl_reject__"):
        r = {"action_id": item.get("action_id"), "tool_name": tool_name,
             "success": False, "rejected": True}
    else:
        r = await _execute_one(tool_name, draft_args, approved)
        r.setdefault("action_id", item.get("action_id"))

    return {"results": results + [r], "index": idx + 1, "result": r}


def _route_next(state: HitlState):
    """큐에 남은 항목이 있으면 노드로 되돌리고(다음 interrupt), 없으면 종료한다."""
    return "execute_action" if state.get("index", 0) < len(_queue_of(state)) else END


def build_action_graph(checkpointer):
    """체크포인터를 주입해 컴파일된 HITL 액션 그래프를 반환한다.

    각 큐 항목마다 interrupt(정지)하고, 조건부 엣지로 큐를 순회한다. Command(resume)
    한 번이 정확히 한 액션을 실행하고 다음 항목 interrupt 에서 다시 정지한다(체크포인터가
    index 를 보존). 단일 액션이면 큐 길이 1로 동작(하위호환).

    Args:
        checkpointer: LangGraph checkpointer (로컬/테스트=MemorySaver,
            프로덕션=RedisSaver). interrupt 재개에 필수.

    Returns:
        compile 된 그래프 (ainvoke / Command(resume) 지원).
    """
    graph = StateGraph(HitlState)
    graph.add_node("execute_action", execute_action_node)
    graph.add_edge(START, "execute_action")
    graph.add_conditional_edges("execute_action", _route_next)
    return graph.compile(checkpointer=checkpointer)


def interrupt_payload(result: Any) -> Optional[dict]:
    """``ainvoke`` 결과에서 interrupt 페이로드(draft 등)를 꺼낸다. 없으면 None.

    그래프가 일시정지했는지(=승인 대기) 판별하고 draft view 를 얻는 데 사용한다.
    """
    if not isinstance(result, dict):
        return None
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    first = interrupts[0]
    return getattr(first, "value", None)


_ACTION_GRAPH = None


def get_action_graph():
    """프로세스 단일 HITL 액션 그래프를 반환한다(lazy singleton).

    체크포인터는 요청 간(초안→승인은 별도 HTTP 요청) 상태를 공유해야 하므로
    동일 인스턴스를 재사용한다. 백엔드(memory/postgres) 선택은
    ``app.agents.graphs.checkpointer.get_checkpointer()`` 단일 주입 지점이 담당한다
    (설정 ``LANGGRAPH_CHECKPOINTER``). 기본 memory=in-process·단일 워커 한정.
    """
    global _ACTION_GRAPH
    if _ACTION_GRAPH is None:
        from agent_arch.agents.graphs.checkpointer import get_checkpointer

        _ACTION_GRAPH = build_action_graph(get_checkpointer())
        logger.info("[hitl] action graph compiled")
    return _ACTION_GRAPH
