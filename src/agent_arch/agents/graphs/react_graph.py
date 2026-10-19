"""ReAct 루프의 LangGraph StateGraph 구현 (Phase 1).

``app/agents/react/loop.py`` 의 ``react_loop_events`` 와 **동일한 LoopResult** 를 산출하는
low-level ``StateGraph``. 커스텀 노드가 ``BaseLLM`` 을 직접 호출한다(LangChain 메시지/
prebuilt 미사용) — sanitize·thought_signature·tool_ctx·anti-thrash·provider-carry 등
축적된 동작을 그대로 보존하기 위함. loop.py 의 헬퍼(_build_pending_action / 의도 정규식 /
fallback 상수)를 **재사용** 해 로직 드리프트를 원천 차단한다(패리티 by construction).

단계:
    - **1a(현재)**: 비스트리밍 ``run()`` 패리티 — 토큰을 밖으로 방출하지 않고 최종
      ``LoopResult`` 만 산출. ``run_react_graph()`` 가 그래프를 ainvoke 하고
      ``final_state["result"]`` 를 돌려준다. orchestrator 배선은 1c.
    - **1b**: ``get_stream_writer()`` 로 토큰/ProgressEvent 방출(스트리밍).
    - **1c**: orchestrator 가 flag(USE_LANGGRAPH_REACT)로 react_loop_events ↔ 그래프 분기.

노드/엣지(loop.py 블록 ↔ 노드 1:1):
    start → loop_guard
    loop_guard  ─(_route_loop)→  stopped | call_model | force_answer
    call_model  ─(_route_after_model)→  stopped | malformed | answer | retry_any | fallback | execute
    retry_any → loop_guard
    execute     ─(_route_after_execute)→  stopped | END(HITL 대기) | loop_guard
    answer / fallback / malformed / force_answer / stopped → END

체크포인터 없음(단일 턴, interrupt 없음). 상태에 BaseLLM·도구 인스턴스·커스텀 Message 등
비직렬화 객체를 담으므로 checkpointer 를 붙이려면 커스텀 직렬화가 필요(추후 과제).
"""
from __future__ import annotations

from typing import Optional

import inspect
import json
import logging
import re
import time
from urllib.parse import urlsplit

from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer
from langsmith import get_current_run_tree

from agent_arch.core.config import settings
from agent_arch.agents.cancellation import is_cancelled
from agent_arch.agents.context import get_user_context
from agent_arch.agents.react.thread_first import (
    THREAD_TOOL_NAME,
    pick_thread_tool,
    scope_to_current,
)
from agent_arch.agents.react.author_first import (
    AUTHOR_TOOL_NAME,
    pick_author_tool,
)
from agent_arch.agents.react.howto import GUIDE_TOOL_NAME, pick_forced_tool

# guide-first 강제 턴에서 검색어로 쓰는 질문 문장의 최대 길이(임베딩 입력 보호)
_GUIDE_QUERY_MAX_LEN = 500
from agent_arch.agents.graphs.state import ReactState
from agent_arch.agents.hitl.policy import requires_approval
from agent_arch.agents.hitl.answer_guard import (
    build_file_link_map,
    find_unsourced_person_names,
    guard_phantom_action,
    has_direct_action,
    has_unnamed_author,
    needs_draft_cancel_guard,
    redact_fabricated_permalinks,
    redact_unsourced_names,
    requester_context,
)
from agent_arch.agents.react.executor import execute_tools
from agent_arch.agents.react import verify as _verify
from agent_arch.agents.react import evidence_budget as _evidence
from agent_arch.agents.react.provenance import apply_sources_footer
from agent_arch.agents.react.guide_links import apply_guide_links_footer
from agent_arch.agents.react.link_policy import TurnLinkPolicy
from agent_arch.agents.react.messenger_links import apply_file_links, file_refs
from agent_arch.agents.react.types import LoopResult, ProgressEvent, merge_usage
from agent_arch.agents.tools.tool_search import drain_discovered_tools, reset_discovered_tools
from agent_arch.llm.base import Message
from agent_arch.memory.conversation import COMPACT_SUMMARY_MARKER
from agent_arch.utils.language import DEFAULT_LANGUAGE, looks_off_target
from agent_arch.utils.sanitize import collapse_repeated_runs, sanitize_tool_names

# loop.py 헬퍼/상수 재사용 — 로직 단일 출처(패리티 보장). 언더스코어 심볼이지만 모듈
# 레벨이라 import 가능. 여기서 재정의하면 두 경로가 갈라질 수 있어 반드시 재사용한다.
from agent_arch.agents.react.loop import (
    _build_pending_action,
    _wants_thread_reply,
    _wants_react_latest,
    _SAFE_MSG_FALLBACK,
    _SAFE_MSG_MALFORMED,
    _SAFE_MSG_NO_TOOLS,
    _MAX_RESPONSE_CHARS,
    _MM_SEND_TOOLS,
    _MM_REACT_TOOL,
)
from agent_arch.services.monitoring.error_events import event_extra

logger = logging.getLogger(__name__)


def _emit(event: ProgressEvent) -> None:
    """스트리밍 소비자(custom stream)가 있으면 ProgressEvent 를 방출한다.

    ``stream_react_graph`` 의 ``astream(stream_mode=[...,"custom"])`` 가 소비한다.
    ``run_react_graph`` (ainvoke) 처럼 custom 소비자가 없으면 no-op — 그래프 실행에
    영향 주지 않도록 방어한다(1a 비스트리밍 경로 무변경 보장).
    """
    try:
        writer = get_stream_writer()
    except Exception:
        return
    if writer is None:
        return
    try:
        writer(event)
    except Exception:
        pass


# 언어 가드 선버퍼 크기 — 답변 앞부분을 이만큼 붙들고 언어를 판정한다.
# looks_off_target 은 이질 문자가 20자 미만이면 판정을 보류하므로 그보다 넉넉히.
_LANG_SNIFF_CHARS = 80


class _AnswerLanguageGate:
    """목표 언어가 아닌 답변이 화면에 다 찍힌 뒤에야 교체되는 것을 막는 토큰 방출 게이트.

    언어 가드(``_regenerate_answer_language``)는 스트림이 **끝난 뒤** 최종 답변 노드에서
    돈다. 토큰을 그대로 흘려보내면 사용자는 다른 언어 답변을 끝까지 보고, 재생성본은 DB
    에만 남는다(화면 ≠ 히스토리). SSE 에는 본문 교체 이벤트가 없어 되돌릴 수도 없다.

    그래서 답변 앞부분 ``_LANG_SNIFF_CHARS`` 만큼을 붙들고 언어를 판정한다:
      - **목표 언어** → 즉시 flush 하고 이후 토큰은 그대로 통과(체감 지연 ≈ 첫 문장 한 줄).
      - **다른 언어** → 한 글자도 내보내지 않고 ``withheld=True``. 최종 답변 노드가 목표
        언어로 재생성한 본문(=저장본과 동일)을 한 번에 방출한다.

    목표 언어 답변이 다른 스크립트(코드·로그)로 시작하면 보류로 오판할 수 있는데, 그때도
    전체 본문이 목표 언어면 가드가 원문을 그대로 돌려주므로 **스트리밍이 한 번에 표시될
    뿐 내용은 같다**. ``settings.korean_answer_guard=false`` 면 게이트도 통째로 비활성.
    """

    def __init__(self, *, language: str = DEFAULT_LANGUAGE, hold_all: bool = False) -> None:
        # hold_all: 이름 환각 위험(도구 결과에 @handle 작성자가 있음)이 있는 턴은 답변 전체를
        # 붙든다. 이름은 앞 80자만 봐서는 판정할 수 없고(본문 어디에나 나온다), 틀린 이름이
        # 화면에 다 찍힌 뒤 저장본만 고치면 화면 ≠ 히스토리가 된다. 위험한 턴에만 켠다.
        self._hold_all = hold_all
        self._language = language
        self._open = (not settings.korean_answer_guard) and not hold_all
        self._buf: list[str] = []
        self.withheld = hold_all

    def feed(self, chunk: str) -> None:
        """텍스트 청크 1개 — 게이트가 열려 있으면 즉시 방출, 아니면 판정될 때까지 보류."""
        if self._open:
            _emit(ProgressEvent("token", content=chunk))
            return
        if self.withheld:
            return  # 이미 보류 결정 — 최종 답변 노드가 재생성본을 방출한다.
        self._buf.append(chunk)
        if sum(len(c) for c in self._buf) >= _LANG_SNIFF_CHARS:
            self._decide()

    def close(self) -> None:
        """스트림 종료 — 판정 임계에 못 미친 짧은 답변을 여기서 마무리 판정한다."""
        if self._hold_all:
            return  # 이름 검증 전에는 한 글자도 내보내지 않는다.
        if not self._open and not self.withheld and self._buf:
            self._decide()

    def flush_partial(self, text: str) -> None:
        """중단 등으로 재생성 기회가 없을 때, 보류분을 원문 그대로라도 내보낸다."""
        if self._hold_all:
            return  # 검증 못 한 이름을 내보내느니 최종 답변 노드에 맡긴다.
        if not self._open and not self._buf and not self.withheld:
            return
        if not self._open and text:
            _emit(ProgressEvent("token", content=text))

    def _decide(self) -> None:
        text = "".join(self._buf)
        self._buf = []
        if looks_off_target(text, self._language):
            self.withheld = True
            logger.warning(
                "[lang_guard] 목표 언어(%s) 아닌 답변 감지 → 스트리밍 보류, 재생성본만 방출 (preview=%r)",
                self._language, text[:60],
            )
            return
        self._open = True
        _emit(ProgressEvent("token", content=text))


def _instrument(name: str):
    """노드를 감싸 실행시간을 재고 node_trace 에 (name, elapsed) 를 남긴다(스텝 집계용).

    node_trace 는 operator.add 리듀서라 노드 경계를 넘어 누적된다. 노드가 반환한 상태
    업데이트에 자신의 (이름, 소요초)만 얹으므로 원 노드 로직은 건드리지 않는다.
    """
    def deco(fn):
        if inspect.iscoroutinefunction(fn):
            async def _aw(state):
                t = time.monotonic()
                upd = await fn(state)
                upd = dict(upd) if upd else {}
                upd["node_trace"] = [(name, round(time.monotonic() - t, 4))]
                return upd
            _aw.__name__ = getattr(fn, "__name__", name)
            return _aw

        def _w(state):
            t = time.monotonic()
            upd = fn(state)
            upd = dict(upd) if upd else {}
            upd["node_trace"] = [(name, round(time.monotonic() - t, 4))]
            return upd
        _w.__name__ = getattr(fn, "__name__", name)
        return _w
    return deco


def _budget_summary(final_state: dict) -> dict:
    try:
        from agent_arch.agents.react.budget import summarize
        return summarize(final_state.get("collected_tool_calls"), final_state.get("collected_tool_results"))
    except Exception:
        return {}


def _build_step_meta(final_state: dict) -> dict:
    """그래프 최종 state 에서 스텝 집계 신호를 뽑는다(LangSmith 메타데이터 + DB 저장용)."""
    trace = final_state.get("node_trace") or []
    timings: dict = {}
    for entry in trace:
        try:
            n, e = entry
        except (TypeError, ValueError):
            continue
        timings[n] = round(timings.get(n, 0.0) + float(e), 4)
    names = [n for n, _ in trace if isinstance(trace, list)] if trace else []
    result = final_state.get("result")
    return {
        "node_trace": names,                                  # 방문 경로
        "node_timings": timings,                              # 노드별 누적 시간(초)
        "iterations": final_state.get("iteration", 0),
        "llm_invoke_count": final_state.get("llm_invoke_count", 0),
        "retry_any": names.count("retry_any"),
        "discovered_tools": final_state.get("discovered_tool_names") or [],
        "finish_reason": getattr(result, "finish_reason", None),
        # 이름 가드 관측 — "이번 주에 모델이 이름을 지어냈나"를 트랜스크립트 없이 답하기 위한
        # 유일한 신호다. **armed=0 은 안심이 아니라 경보다**: 도구 출력 형태가 바뀌어
        # 가드가 조용히 죽은 상태가 정확히 그렇게 보인다(2026-09-03 실측: 30일 메신저 턴
        # 791건 / 마스킹 생애 1건).
        "name_guard": final_state.get("name_guard") or {},
        "guide_first": bool(final_state.get("guide_first", False)),  # 가이드 검색 강제 적용 여부
        # 턴 예산(react/budget): 실행된 웹 호출·중복 차단·상한 차단 수 — 이상치 턴(14회 반복) 감시용.
        "web_budget": _budget_summary(final_state),
        # 답변 검증 관측(react/verify) — shadow 실측의 유일한 자료: 라운드별 판정·행동·증거 사실.
        "verify": final_state.get("verify_meta") or [],
        "sufficiency": final_state.get("suff_meta") or [],
    }


def _attach_step_meta(final_state: dict, result) -> None:
    """스텝 신호를 LoopResult.step_meta 에 싣고 LangSmith 현재 run 에 메타/태그로 남긴다."""
    if not isinstance(final_state, dict):
        return
    step_meta = _build_step_meta(final_state)
    if result is not None:
        result.step_meta = step_meta
    try:
        run = get_current_run_tree()
        if run is not None:
            run.add_metadata({"react_steps": step_meta})
            fr = step_meta.get("finish_reason")
            if fr:
                run.add_tags([f"react_finish:{fr}"])
    except Exception:
        logger.debug("[react_graph] LangSmith 스텝 메타 부착 실패(무시)", exc_info=True)


def _stopped_result(state: ReactState) -> LoopResult:
    """사용자 중단(stop) 시 산출할 결과 — react_loop_events._stopped_result 와 동형.

    중단 턴은 저장하지 않으므로 content 는 비운다. finish_reason="stopped" 로
    orchestrator 가 저장/HITL/eval 을 스킵한다.
    """
    return LoopResult(
        content="",
        tool_calls=state["collected_tool_calls"],
        tool_results=state["collected_tool_results"],
        llm_invoke_count=state["llm_invoke_count"],
        token_usage=state["token_usage"],
        finish_reason="stopped",
    )


# ── 노드 ──────────────────────────────────────────────────────────────────────

def _draft_sig(d: dict) -> tuple:
    """초안의 논리적 시그니처(tool_name + 인자) — 재초안(같은 발송) dedup 용."""
    return (d.get("tool_name"), json.dumps(d.get("arguments") or {}, sort_keys=True, ensure_ascii=False))


def _accumulate_drafts(prior: list, new: list) -> list:
    """누적 초안 + 새 초안을 시그니처 dedup 병합(같은 액션 재초안이면 최신으로 갱신)."""
    by_sig = {_draft_sig(d): d for d in prior}
    for d in new:
        by_sig[_draft_sig(d)] = d
    return list(by_sig.values())


def _final_loop_result(state: ReactState, content: str, finish_reason: str,
                       *, invoke: Optional[int] = None, token_usage: Optional[dict] = None) -> LoopResult:
    """터미널 LoopResult 빌더 — 누적된 pending_drafts 가 있으면 action_pending 으로 전환.

    승인 대기 초안을 답변 시점(모든 조회/요약 완료 후)에 붙여, HITL 을 마지막으로 미룬다.
    """
    drafts = state.get("pending_drafts") or []
    # 승인 카드(drafts)가 없는데 LLM 이 "삭제했어요"·"[승인] 버튼 누르세요"를 환각하면 제거.
    # 승인 필요 도구는 루프에서 초안(confirm=False)으로만 돌고 실제 실행·완료문구는 코드 소관.
    # 단 승인 없이 곧바로 실행되는 도구(memory_forget 등)가 이번 턴에 실제로 돌았으면
    # 완료 주장은 정당하다 — 그 규칙만 끄고 버튼 환각 규칙은 유지한다.
    content = guard_phantom_action(
        content, bool(drafts),
        allow_done_claims=has_direct_action(state.get("collected_tool_calls")),
    )
    return LoopResult(
        # 근거 없는 링크 제거(가이드 턴에서만 발동) — 화면 토큰도 같은 정책으로 정화된다
        # (orchestrator 의 StreamSanitizer, `link_policy` 이벤트로 전달). 화면 = 저장본.
        content=sanitize_tool_names(content, link_permits=_link_policy(state).permits),
        tool_calls=state["collected_tool_calls"],
        tool_results=state["collected_tool_results"],
        llm_invoke_count=(invoke if invoke is not None else state["llm_invoke_count"]),
        token_usage=(token_usage if token_usage is not None else state["token_usage"]),
        finish_reason=("action_pending" if drafts else finish_reason),
        pending_action=(drafts[0] if drafts else None),
        pending_queue=(drafts or None),
    )


# 지시어가 있는 턴에 **현재 채널로 묶을** 도구들.
#   search_knowledge  : 기본이 전 채널이라 "이 스레드" 질문에서 다른 채널을 끌어온다(사고 원인).
#   get_channel_messages: 모델이 channel_keyword 로 채널을 찍으면 엉뚱한 채널이 된다.
_SCOPED_SEARCH_TOOL = "mattermost_search_knowledge"

#: 제품·서비스 이름. **회사명이 아니라 검색어**다 — 사용자가 찾는 문서 제목에 이 말이 들어
#: 있다. 모델이 "회사명은 query 에 남기지 마라" 규칙을 여기까지 적용해 지워 버리는 일이
#: 있어(2026-09-16 실측) 시스템이 되살린다.
_PRODUCT_TERMS = ("에이전트", "Agent", "agent", "회사 Talk", "DMP")
#: 그 말이 **너를 부르는 말**일 때는 검색어가 아니다 — `@에이전트`, "에이전트야".
#: ⚠️ 메신저 봇은 **언제나 `@에이전트` 로 불린다.** "부르는 말이 있으면 건너뛴다" 로 짜면
#: 메신저 경로에서 가드가 **한 번도 안 돈다**(테스트로 잡았다). 그래서 건너뛰는 게 아니라
#: 부르는 부분만 **걷어낸 뒤** 그 말이 여전히 남아 있는지로 판단한다 —
#: "@에이전트 에이전트의 권한 문서" 는 부르는 말 하나, 주제어 하나다.
_VOCATIVE_RE = re.compile(r"@\s*에이전트|에이전트\s*(?:야|아)(?![가-힣])")
_SCOPED_CHANNEL_TOOL = "mattermost_skill_get_channel_messages"


def _origin_ctx() -> dict:
    """이 턴의 현재 위치(채널·스레드). 메신저가 아니면 빈 dict."""
    origin = (get_user_context() or {}).get("origin")
    return origin if isinstance(origin, dict) else {}


def _origin_root_id() -> Optional[str]:
    """이 턴의 스레드 루트 post_id. 메신저가 아니거나 스레드 밖이면 None.

    봇이 `user_info["origin"]` 으로 실어 보낸 값(프롬프트 `[현재 위치]` 블록과 같은 출처)을
    요청 스코프 컨텍스트에서 읽는다.
    """
    return _origin_ctx().get("root_id") or None



def _start_node(state: ReactState) -> dict:
    """루프 진입 — 누적 상태 초기화.

    발견 대기열 reset 은 여기가 아니라 execute 노드에서 한다: LangGraph 는 노드마다
    컨텍스트를 격리(copy_context)하므로 start 에서 reset 해도 execute 로 전파되지 않는다
    (tests/agents/graphs test_contextvar_* 로 실측 확인).
    """
    all_tools = state.get("all_tools") or []
    known = {t.name for t in all_tools}
    _remember_history_urls(state.get("messages") or [])
    # thread-first: "이 스레드" 를 가리키는 질문이면 첫 호출을 스레드 조회로 고정한다.
    # guide-first 보다 **먼저** 본다 — 둘 다 걸리는 문장("이 스레드에서 말한 설정 방법")은
    # 지금 눈앞의 스레드가 답이지 가이드 문서가 아니다.
    input_text = state.get("input") or ""
    force_tool = pick_thread_tool(input_text, known, _origin_root_id())
    # author-first: "누가 ~한 글" 이면 작성자 검색으로 고정한다. thread 보다는 **뒤**(눈앞의
    # 스레드가 더 구체적이다), guide 보다는 **앞**이다 — "홍길동 이사님이 공유한 설정 방법"
    # 은 가이드 문서가 아니라 그 사람이 올린 글이 정답이다.
    force_tool = force_tool or pick_author_tool(input_text, known)
    # guide-first: 방법·설정·증상 질문이면 첫 호출을 가이드 검색으로 고정(후속 턴 검색 생략 방지).
    # supervisor 경로의 input 은 라우터 재작성 서브쿼리라 원문의 질문 신호("검색이 안 돼요")가
    # 사라지는 턴이 있다(9/28 실측, no_search 의 근본 원인) → 서브쿼리가 미매치면 **턴 원문**
    # (get_turn_input, 이미지·초안 블록 주입 전 raw)으로 한 번 더 판정한다. 호출자가
    # turn_input_guide_fallback 을 켠 턴(단일 도메인·단일 서브런)에서만. 원문이 없는 경로(승인
    # 재개 등)는 종전과 같다.
    force_tool = force_tool or pick_forced_tool(input_text, known)
    guide_query_text = None   # 원문으로만 판정된 턴에만 채운다 — 실행부가 길이와 무관하게 이 값으로 검색
    if force_tool is None and state.get("turn_input_guide_fallback"):
        from agent_arch.agents.context import get_turn_input
        turn_text = get_turn_input()
        if turn_text and turn_text.strip() != input_text.strip():
            force_tool = pick_forced_tool(turn_text, known)
            if force_tool == GUIDE_TOOL_NAME:
                # 원문으로만 판정된 턴은 검색어도 원문 — 서브쿼리는 신호를 잃은 문장이라 검색에도 약하다.
                guide_query_text = turn_text
                logger.info("[react_graph] guide-first — 서브쿼리는 미매치, 턴 원문으로 강제: %r", turn_text[:60])
    # 거절된 초안이 있으면 그 도구를 지목한다 — 위 pick_* 보다 뒤에 두어, 사용자가 명시적으로
    # 다른 것을 가리킨 턴(스레드·작성자·가이드)은 그쪽이 이긴다.
    if state.get("pending_draft_force"):
        force_tool = force_tool or _pin_tool_for_rejected_draft(known)
    # 초안이 진행 중이면(수집 중·거절됨) 첫 호출을 **도구 호출로 강제**한다
    # (ANY — 어떤 도구인지는 모델이 고른다).
    #
    # 모델이 도구를 부르지 않고 제가 답을 지어내는 일이 있다. 실측(구매 요청 초안):
    # 선택값을 묻는 결과를 받은 뒤 사용자가 "넘어가" 라고 답해도 도구를 다시 부르지 않고
    # **같은 질문을 새로 만들어** 냈다 — 그때 후보 분류가 도구가 준 목록에 없는 값이었다
    # (지어낸 증거). 같은 화면이 무한히 반복됐다.
    #
    # 지목(`allowed_tools`)이 아니라 ANY 인 이유: 수집 중에도 사용자는 다른 것을 묻는다
    # ("이 발주 어디까지 왔어"). 지목하면 그 질문까지 초안 도구로 끌려간다.
    #
    # 켤지는 **호출자가 정한다**(`pending_draft_force`). 일반 채널은 supervisor 의
    # `_bias_to_action_for_pending_draft` 가 같은 일을 하므로 켜지 않는다 — 판단 근거는
    # orchestrator `_pending_draft_force` 에 있다.
    force_reason = _force_tool_reason(known) if state.get("pending_draft_force") else None
    if force_reason:
        logger.info("[react_graph] 첫 호출 도구 강제(ANY) — 사유=%s", force_reason)
    return {
        "force_tool_name": force_tool,
        # 집계·검색어 치환 게이트 — thread/author 강제는 가이드가 아니다(2026-10-13 교정, 강제 실행은 force_tool_name).
        "guide_first": force_tool == GUIDE_TOOL_NAME,
        "guide_query_text": guide_query_text,
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "collected_tool_calls": [],
        "collected_tool_results": [],
        "known_tool_names": known,
        "used_tool_names": [],
        "llm_invoke_count": 0,
        "used_tools": False,
        # 진행 중 초안이면 첫 호출을 도구로 강제한다(위 force_reason 참조).
        # ⚠ 이 키는 여기 **한 번만** 나와야 한다 — dict 리터럴은 같은 키가 겹치면 뒤가
        # 이긴다. 위쪽에 따로 넣었다가 이 줄(False)에 조용히 덮여, 판정 로그는 찍히는데
        # 강제는 안 걸리는 상태로 세 회차를 헤맸다.
        "forced_any_mode": bool(force_reason),
        "forced_any_done": False,
        "iteration": 0,
        "iter_tool_calls": (pre := _prefetch_tool_calls(state.get("prefetch_calls"), known)),
        "prefetch_phase": bool(pre),
        "auto_fetch_done": False,
        "suff_rounds": 0,
        "suff_meta": [],
        "finish_reason": "stop",
        "text_buf": [],
        "verify_rounds": 0,
        "rewrite_done": False,
        "history_evidence": _history_evidence(state.get("messages") or []),
        "verify_meta": [],
        "verify_decision": "",
        "cancelled": False,
        "answer_withheld": False,
        "result": None,
        "pending_drafts": [],
    }


def _force_tool_reason(known_tool_names: set) -> Optional[str]:
    """첫 호출을 도구로 강제할 이유. 없으면 None.

    진행 중인 초안이 있으면 그 턴은 조회가 아니라 **실행**이다 — 수집 중(되물을 값이 남음)
    이거나 거절됨(값을 고쳐 다시 만들 상태). supervisor 의 `_pending_draft_stashes` 와 같은
    두 stash 를 본다.

    **첨부만으로는 강제하지 않는다.** 도면을 올렸다고 발주요청인 것은 아니다 — 첨부 턴을
    강제했더니 "이 도면 재질이 뭐야?" 가 답 끝에 묻지도 않은 초안 질문을 달고 수집 상태까지
    남겼다(실측 2026-09-30, 2/2). 무엇을 할지는 모델이 사용자 말로 고른다.

    **초안의 도구가 이 턴에 바인딩돼 있을 때만** 켠다. 도구가 안 붙은 턴(채널·권한 스코핑)에
    켜면 엉뚱한 도구를 부르게 된다. 실패하면 강제하지 않는다(fail-open).
    """
    known = known_tool_names or set()
    try:
        from agent_arch.agents.context import (
            get_collecting_draft_context,
            get_rejected_draft_context,
        )

        for label, stash in (("수집중", get_collecting_draft_context()),
                             ("거절됨", get_rejected_draft_context())):
            name = stash.get("tool_name") if isinstance(stash, dict) else None
            if name and name in known:
                return f"{label}:{name}"
        return None
    except Exception:  # noqa: BLE001 — 부가기능(fail-open)
        return None


def _pin_tool_for_rejected_draft(known_tool_names: set) -> Optional[str]:
    """거절된 초안이 있으면 그 도구를 **지목**한다(`allowed_tools`). 없으면 None.

    `tool_mode="ANY"` 로는 부족했다 — 실측(구매 요청 초안): 거절 뒤 "다시 만들어줘" 턴에
    ANY 가 걸렸는데도(로그 `사유=거절됨:…`) 모델이 도구를 부르지 않고 **카드를 글로 흉내**
    냈다. `answer_guard` 가 그 환각 문구를 걷어냈지만 초안은 만들어지지 않았다.

    ANY 는 "도구를 부르긴 해라" 까지만 강제하고, 모델이 텍스트를 내면 `_retry_any_node` 가
    잡아야 하는데 그건 **빈 응답일 때만** 발동한다. 그럴듯한 텍스트가 나오면 그대로 끝난다.

    지목이 위험한 경우("수집 중에 다른 것을 묻는다")와 달리, **거절 직후 정정 턴은 그 초안을
    다시 만드는 것 말고 할 일이 거의 없다.** 그래서 이 상태에서만 지목한다.
    """
    try:
        from agent_arch.agents.context import get_rejected_draft_context

        stash = get_rejected_draft_context()
        if isinstance(stash, dict) and stash.get("pin_used"):
            # 지목은 거절 직후 한 턴만 — 소비 표시가 붙었으면 모델이 고르게 둔다
            # (orchestrator `rejected_pin_used`). ANY 강제는 그대로 걸린다.
            return None
        name = stash.get("tool_name") if isinstance(stash, dict) else None
        return name if name and name in (known_tool_names or set()) else None
    except Exception:  # noqa: BLE001 — 부가기능(fail-open)
        return None


_HISTORY_EVIDENCE_CHARS = 200_000


def _history_evidence(messages: list) -> str:
    """이번 턴 이전 대화(앞 답·앞 도구 결과·사용자 입력)의 본문. 마지막 메시지(이번 턴 질문)와 system 은 뺀다.

    실측(2026-09-28 실측 vllm 유로): 앞 턴에서 smbs.biz 로 확인한 공식값 1,538.04원을 이번 턴 답에 쓰자, 검증기가
    "이번 턴 도구 결과에 없다"며 두 번 되돌렸다(고쳐 쓰기 2회, +40초). 앞 턴 근거도 근거다.
    """
    parts = [str(getattr(m, "content", "") or "") for m in (messages or [])[:-1] if getattr(m, "role", None) != "system"]
    return "\n".join(parts)[-_HISTORY_EVIDENCE_CHARS:]


def _remember_history_urls(messages: list) -> None:
    """이 대화의 이전 답변에 나온 URL 은 web_fetch 로 열 수 있게 한다 — 이전 검색에서 나와 사용자에게 이미 보인 주소다.

    실측(실측 2026-09-28): 후속 질의에서 모델이 앞 답의 씨티은행 URL 을 열려다 url_not_seen 으로 막혔고, 그 제약을
    사용자에게 그대로 말하며 되물었다. 모델이 지어낸 주소를 막는 원칙과는 충돌하지 않는다(답에 실제로 쓰였던 주소만).
    """
    from agent_arch.agents.context import remember_web_urls, web_search_enabled
    if not web_search_enabled():
        return
    urls = []
    for m in messages:
        if getattr(m, "role", None) == "assistant" and isinstance(getattr(m, "content", None), str):
            urls += re.findall(r"https?://[^\s<>\"')\]]+", m.content)
    if urls:
        remember_web_urls(u.rstrip(".,") for u in urls)


def _prefetch_tool_calls(calls, known: set) -> list:
    """선실행 호출 목록 → ToolCall. 이 턴에 바인딩되지 않은 도구는 버린다(스코프 밖 실행 금지).

    선실행은 execute 노드를 그대로 지난다 — 도구 호출 메시지·결과 기록·예산 가드·각주·검증기가 모델이 부른 호출과
    똑같이 본다. 모델은 첫 호출에서 이미 결과를 쥔 채로 시작한다.
    """
    from uuid import uuid4

    from agent_arch.llm.base import ToolCall

    out = []
    for c in calls or []:
        name = (c or {}).get("name")
        if name in known:
            out.append(ToolCall(id=f"prefetch_{name}__{uuid4().hex[:8]}", name=name, arguments=dict(c.get("arguments") or {})))
    return out


_AUTO_FETCH_MAX = 2
_AUTO_FETCH_CHARS = 12_000


_TOKEN_RE = re.compile(r"[0-9a-z가-힣]{2,}")


# 실시간 시세·변환 페이지는 현재값을 한 줄로 단언한다("1.00 EUR = 1,544.18 KRW"). 통계·이력 페이지(ECB 그래프·"history")는 아니다.
# 관련도(제목·URL 단어)만으로 고르자 '오늘 유로 기준환율'에 ECB 그래프·Wise 이력 페이지를 열었다(2026-09-28) — 사이트가 아니라 내용 형태로 가점.
_QUOTE_BONUS = 3
_QUOTE_RE = re.compile(r"(?<![\d.])1(?:\.0+)?\s*([A-Z]{3})\s*=\s*[\d,]+(?:\.\d+)?\s*([A-Z]{3})")


def _states_quote(content, codes: set) -> int:
    """검색어의 통화 코드 쌍으로 '1 A = 숫자 B'를 단언하는 행이면 1."""
    if not content or len(codes) < 2:
        return 0
    return int(any({m.group(1), m.group(2)} <= codes for m in _QUOTE_RE.finditer(str(content))))


def _auto_fetch_urls(results: list, calls: Optional[list] = None, limit: int = _AUTO_FETCH_MAX) -> list[str]:
    """선실행 본문 조각 결과에서 원 페이지를 열 URL — 검색어 핵심 토큰이 제목·URL 에 많이 든 순, 호스트는 서로 다르게.

    처음엔 "검색어별 1순위"였는데 한국어 검색어의 1순위가 네이버 금융 종목 페이지·기술 블로그라 엉뚱한 페이지를 열었다
    (2026-09-23 EPDM). 제목·URL 에 대상어(epdm·rubber·price·단가)가 든 페이지가 가격지수 원 페이지다. 숫자(연도·월)는 뺀다.
    """
    from agent_arch.services.web.brave_output import CONTEXT_TOOL

    queries = {c.get("id"): str((c.get("arguments") or {}).get("query") or "") for c in (calls or []) if isinstance(c, dict)}
    scored: list[tuple[int, int, str, str]] = []
    order = 0
    for r in results or []:
        if r.get("name") != CONTEXT_TOOL:
            continue
        try:
            data = json.loads(r.get("content") or "")
        except (TypeError, ValueError):
            continue
        q = queries.get(r.get("tool_call_id"), "")
        tokens = {t for t in _TOKEN_RE.findall(q.lower()) if not t.isdigit()}
        codes = {t.upper() for t in tokens if len(t) == 3 and t.isalpha() and t.isascii()}
        for x in data.get("results") or []:
            url = x.get("url") if isinstance(x, dict) else None
            if not isinstance(url, str):
                continue
            hay = f"{x.get('title') or ''} {url}".lower()
            score = sum(1 for t in tokens if t in hay) if tokens else 1
            score += _QUOTE_BONUS * _states_quote(x.get("content"), codes)
            order += 1
            if score:
                scored.append((-score, order, url, (urlsplit(url).hostname or "")))
    out: list[str] = []
    hosts: set[str] = set()
    for _neg, _o, url, host in sorted(scored):
        if url in out or host in hosts:
            continue
        out.append(url); hosts.add(host)
        if len(out) >= limit:
            break
    return out


def _auto_fetch_node(state: ReactState) -> dict:
    """선실행 검색 뒤 상위 페이지를 코드가 연다(web_fetch). 모델이 web_fetch 를 부르지 않아도 최신 본문이 손에 들어온다.

    실측(2026-09-23, gemini-2.5-flash): 시스템 지침·도구 결과 안내·검증기 규칙 어느 자리에 "조각에 없으면 원 페이지를 열어라"를
    넣어도 web_fetch 를 한 번도 부르지 않고 "9월 단가는 확인되지 않는다"로 답했다(IMARC 원 페이지엔 9/18 갱신된 9월 표가 있었다).
    검색 범위를 선실행으로 보장한 것과 같은 이유로, 페이지 열기도 코드가 보장한다.
    """
    from agent_arch.llm.base import ToolCall
    from agent_arch.services.web.brave_output import FETCH_TOOL
    from uuid import uuid4

    if state.get("auto_fetch_done"):
        return {"prefetch_phase": False, "iter_tool_calls": []}
    calls = []
    if state.get("auto_fetch_enabled") and FETCH_TOOL in (state.get("known_tool_names") or set()):
        calls = [ToolCall(id=f"prefetch_{FETCH_TOOL}__{uuid4().hex[:8]}", name=FETCH_TOOL,
                          arguments={"url": u, "max_chars": _AUTO_FETCH_CHARS})
                 for u in _auto_fetch_urls(state.get("collected_tool_results") or [], state.get("collected_tool_calls") or [])]
    if calls:
        logger.info("[react_graph] 선실행 페이지 자동 열기 %s", [c.arguments["url"] for c in calls])
    return {"auto_fetch_done": True, "iter_tool_calls": calls, "prefetch_phase": bool(calls)}


def _route_after_auto_fetch(state: ReactState) -> str:
    if state.get("prefetch_phase") and state.get("iter_tool_calls"):
        return "execute"
    from agent_arch.agents.react.sufficiency import rounds_budget
    from agent_arch.agents.react.evidence_budget import over_budget
    if state.get("prefetch_calls") and (state.get("suff_rounds") or 0) < rounds_budget():
        if over_budget(state["messages"], llm=state["llm"]):
            logger.info("[sufficiency] 웹 근거 총량 예산 초과 — 더 찾지 않고 답으로")
            return "loop_guard"
        return "sufficiency"                     # 웹 선실행이 있었던 턴만: 답하기 전에 근거가 충분한지 본다
    return "loop_guard"


async def _sufficiency_node(state: ReactState) -> dict:
    """근거 충분성 판정 → 부족하면 새 검색어·열 URL 을 코드가 실행(다음 execute), 충분하거나 판정 실패면 모델로.

    레퍼런스 공통 설계(답 생성과 분리된 판정 → 부족하면 재검색 → 예산 소진 시 있는 근거로 답). 소극적인 모델(gemini)이
    첫 결과로 "확인되지 않는다"고 답하던 실측(2026-09-28 유로 기준환율)을 모델 성향이 아니라 코드로 막는다.
    """
    from agent_arch.agents.react import sufficiency as S

    n = (state.get("suff_rounds") or 0) + 1
    calls_log, results = state.get("collected_tool_calls") or [], state.get("collected_tool_results") or []
    digest, tried, urls = S.build_digest(calls_log, results, state.get("input") or "")
    try:
        verdict = await S.judge(state.get("input") or "", digest, tried, n, S.rounds_budget(),
                                provider=_verify.provider_of(state.get("llm")))
    except Exception:
        logger.warning("[sufficiency] 판정 실패 → 있는 근거로 진행", exc_info=True)
        verdict = {"sufficient": True, "error": True, "queries": [], "open": []}
    calls = S.plan_calls(verdict, tried, urls, state.get("known_tool_names") or set())
    logger.info("[sufficiency] round=%d sufficient=%s missing=%r → 추가 %s", n, verdict.get("sufficient"),
                verdict.get("missing"), [c.arguments.get("query") or c.arguments.get("url") for c in calls])
    if calls:
        _emit(ProgressEvent("thinking", content="근거가 부족해 다른 방향으로 더 찾아보고 있어요..."))
    meta = list(state.get("suff_meta") or []) + [{"round": n, **{k: verdict.get(k) for k in ("sufficient", "missing", "queries", "open")}}]
    upd = {"suff_rounds": n, "suff_meta": meta, "iter_tool_calls": calls, "prefetch_phase": bool(calls), "auto_fetch_done": True}
    if not verdict.get("sufficient") and verdict.get("missing") and n >= S.rounds_budget():
        # 마지막 라운드까지 부족 — 판정이 짚은 빈칸을 답하는 모델에게 넘긴다. 넘기지 않으면 모델이 비슷한 값(시장환율)을 묻는 값
        # (공식 매매기준율)처럼 섞어 썼다(2026-09-28 유로 기준환율: 판정은 "오늘 공식 매매기준율 없음"을 정확히 짚었다).
        from agent_arch.agents.react.history import FEEDBACK_NAME
        from agent_arch.llm.base import Message
        messages = state["messages"]
        messages.append(Message(role="user", name=FEEDBACK_NAME, content=(
            "[근거 점검 — 사용자에게 보이지 않는 메시지] 여러 방향으로 더 찾았지만 판정기는 다음이 빠졌다고 봤다: "
            f"{verdict['missing']}\n판정기가 틀렸을 수 있다 — 도구 결과를 직접 다시 보고 판단하라. 답은 **확인된 가장 최신 값으로 "
            "시작하라**(그 값의 기준·시점을 붙여서). '찾을 수 없다·확인되지 않는다'로 시작하거나 끝맺지 마라. 정말 확인하지 못한 "
            "부분만 답 끝에 한 줄로 적고, 그 값을 비슷한 값(다른 기준·다른 시점)으로 대신하지 마라. 이 메시지 자체는 언급하지 마라.")))
        upd["messages"] = messages
    return upd


def _route_after_sufficiency(state: ReactState) -> str:
    return "execute" if state.get("iter_tool_calls") else "loop_guard"


def _route_after_start(state: ReactState) -> str:
    """선실행 호출이 있으면 바로 execute, 없으면 평소대로."""
    return "execute" if state.get("iter_tool_calls") else "loop_guard"


def _loop_guard_node(state: ReactState) -> dict:
    """iteration 경계 패스스루 — 라우팅 앵커. (_route_loop 가 분기)"""
    return {}


async def _call_model_node(state: ReactState) -> dict:
    """② LLM stream 소비 — 한 iteration 분의 호출. (1a: 토큰 밖으로 방출 안 함)"""
    llm = state["llm"]
    forced = state.get("forced_any_mode", False)
    # ANY 재시도 요청이 있으면 이번 iteration 은 ANY, 아니면 AUTO.
    tool_mode = "ANY" if forced else "AUTO"
    # guide-first: 첫 iteration 한정으로 가이드 검색 도구 하나만 허용해 검색을 강제한다.
    force_tool = state.get("force_tool_name") if state["iteration"] == 0 else None
    stream_kwargs = {"tool_mode": tool_mode}
    if force_tool:
        stream_kwargs = {"tool_mode": "ANY", "allowed_tools": [force_tool]}

    # iteration>0 진행 표시(react_loop_events 와 동일 위치·문구).
    if state["iteration"] > 0:
        _emit(ProgressEvent("thinking", content="심층 분석중..."))

    _evidence.compact(state["messages"], llm=llm)       # 웹 결과가 예산을 넘었으면 부르기 전에 줄인다(결정론적 절단)

    text_buf: list[str] = []
    iter_tool_calls: list = []
    finish_reason = "stop"
    usage = None
    cancelled = False
    # 목표 언어가 아니면 방출을 보류한다 — 뒤에서 도는 언어 가드가 화면까지 고칠 수 있도록.
    # 이름 환각 위험(도구 결과에 @handle 작성자)이 있으면 답변 전체를 붙든다 — 이름은 본문
    # 어디에나 나와 앞부분만으로 판정할 수 없다.
    _lang = state.get("target_language") or DEFAULT_LANGUAGE
    gate = _AnswerLanguageGate(
        language=_lang,
        # 용어 교정은 답변이 끝난 뒤에 돈다. SSE 에 본문 교체가 없어 흘려보내면 저장본만
        # 고쳐지므로, 고칠 것이 있는 턴은 이름 가드와 같은 방식으로 붙든다.
        hold_all=(needs_draft_cancel_guard(state.get("input", ""))
                  or _name_guard_needed(state) or _verify.hold_needed(state)
                  or _link_guard_needed(state) or await _glossary_hold(_lang)),
    )

    async for event in llm.stream(state["messages"], **stream_kwargs):
        # 체크포인트 b(토큰 단위): break 시 aclose() 가 업스트림 스트림 정리.
        if is_cancelled():
            cancelled = True
            break
        if event.type == "text":
            if event.content:
                text_buf.append(event.content)
                gate.feed(event.content)
        elif event.type == "tool_calls":
            iter_tool_calls = event.tool_calls or []
        elif event.type == "done":
            finish_reason = event.finish_reason or "stop"
            usage = event.usage

    if cancelled or is_cancelled():
        # 중단 턴은 재생성 기회가 없으므로 보류분을 그대로 흘려 부분답변을 보존한다
        # (orchestrator 가 emitted 로 부분답변을 저장한다).
        partial = "".join(text_buf)
        if needs_draft_cancel_guard(state.get("input", "")):
            partial = guard_phantom_action(
                partial, bool(state.get("pending_drafts")),
                allow_done_claims=has_direct_action(state.get("collected_tool_calls")),
            )
        gate.flush_partial(partial)
        # 이번 invoke 는 카운트하지 않고 stopped 로 종결(loop.py 와 동일).
        return {"cancelled": True}

    gate.close()
    tu = state["token_usage"]
    merge_usage(tu, usage)
    upd = {
        "text_buf": text_buf,
        "iter_tool_calls": iter_tool_calls,
        "finish_reason": finish_reason,
        "token_usage": tu,
        "llm_invoke_count": state["llm_invoke_count"] + 1,
        "answer_withheld": gate.withheld,
    }
    if forced:
        upd["forced_any_mode"] = False
        upd["forced_any_done"] = True
    if force_tool:
        upd["force_tool_name"] = None  # 1회 소비
    return upd


def _evidence_text(state: ReactState) -> str:
    """이 턴에 모델이 실제로 본 것 — 이름 대조의 근거.

    대화 이력·도구 결과는 전문을 넣지만 **시스템 프롬프트 본문은 넣지 않는다**.
    프롬프트에 예시로 적힌 이름이 화이트리스트가 되기 때문이다 — 실제로 이름 규칙 조항의
    반례에 실명이 박혀 있어 사고를 낸 바로 그 이름이 영구히 탐지 불가였다(2026-09-03).
    대신 ``[요청자 정보]`` 블록만 되살린다. 요청자 실명은 거기에만 있어서, 통째로 빼면
    사용자 본인 이름이 환각으로 오탐된다.

    압축 요약(``name == COMPACT_SUMMARY_MARKER``)은 ``role="system"`` 이지만 **지난 대화
    내용 그 자체**라 정당한 근거다 — 전문을 유지한다.
    """
    parts = []
    for m in state["messages"]:
        c = getattr(m, "content", None)
        if not (isinstance(c, str) and c):
            continue
        if getattr(m, "role", None) == "system" and getattr(m, "name", None) != COMPACT_SUMMARY_MARKER:
            block = requester_context(c)
            if block:
                parts.append(block)
            continue
        parts.append(c)
    return "\n".join(parts)


def _name_guard_needed(state: ReactState) -> bool:
    """이름 환각 위험이 있는 턴인가 — 도구 결과에 ``@handle`` 작성자가 섞였는지로 본다.

    웹 검색 결과는 무장 신호에서 뺀다. 가드의 대상은 사내 도구가 작성자를 영문 ID 로만 줄 때의 음차 환각인데, 웹 페이지
    본문의 ``@트위터계정``·이메일이 무장시켜 표 셀 규칙이 "변동률·서비스·지역" 같은 일반 명사를 이름으로 잡고 답 전체에서
    "이름미확인"으로 바꿨다(실측 실사용 2026-09-23: EPDM 단가표 헤더 "월별 이름미확인"). 대조 근거에는 그대로 남는다.
    """
    if not settings.name_answer_guard:
        return False
    from agent_arch.services.web.brave_output import WEB_TOOLS
    tool_texts = [
        c for m in state["messages"]
        if getattr(m, "role", None) == "tool" and getattr(m, "name", None) not in WEB_TOOLS
        and isinstance((c := getattr(m, "content", None)), str)
    ]
    return has_unnamed_author("\n".join(tool_texts))


def _link_guard_needed(state: ReactState) -> bool:
    """파일 링크 날조 위험이 있는 턴인가 — 도구 결과에 메신저 첨부 히트가 있는지로 본다.

    모델은 ``file_id`` 를 보면 ``/{team}/files/{id}`` 라는 **없는 주소**를 조립한다
    (실측 2026-09-18). 링크 가드가 그걸 진짜 permalink 로 바꾸는데, 가드는 최종 본문에서만
    도므로 붙들지 않으면 **화면에는 죽은 링크가 찍힌 뒤 저장본만 고쳐진다** — 이름 가드가
    hold 를 도입한 이유와 같다. 첨부가 걸린 턴만 좁게 붙든다.
    """
    return any(
        isinstance(c := r.get("content"), str) and '"file_id"' in c
        for r in (state.get("collected_tool_results") or [])
    )


def _memory_rule_reminder() -> str:
    """이 턴 회상의 사용자 규칙 리마인더(없으면 빈 문자열). 기억 장애가 도구 루프를 막지 않게 삼킨다."""
    try:
        from agent_arch.agents.context import get_memory_recall
        from agent_arch.services.memory_v2.recall import rule_reminder
        return rule_reminder(get_memory_recall())
    except Exception:
        logger.debug("[react_graph] 규칙 리마인더 생략", exc_info=True)
        return ""


def _attach_rule_reminder(content: str, reminder: str) -> str:
    """도구 결과에 리마인더를 붙인다 — 결과 구조를 깨지 않게.

    결과가 JSON 객체면 키 하나(``user_rules``)로 넣는다. 뒤에 평문을 이어 붙이면 JSON 파싱이
    깨져 프로바이더가 결과 전체를 문자열 하나로 감싸고(gemini functionResponse {"result": "..."}),
    모델이 구조화된 결과를 못 읽어 빈 응답을 냈다(실측 2026-09-28, 8회 중 6회). JSON 이 아니면 뒤에 붙인다.
    """
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        parsed = None
    if isinstance(parsed, dict):
        return json.dumps({**parsed, "user_rules": reminder}, ensure_ascii=False)
    return f"{content}\n\n{reminder}"


def _link_policy(state: ReactState) -> TurnLinkPolicy:
    """이번 턴의 링크 허용 정책 — 가이드 검색이 돌았으면 근거 없는 주소를 답변에서 걷어낸다.

    근거로 인정하는 것: ① 도구 결과 본문(가이드 청크의 문의 폼, `mattermost_open_screen` 의
    채널 url, 웹 검색 결과 …) ② 시스템 프롬프트가 코드로 심는 고정 링크(다운로드 센터·앱스토어)
    ③ 사용자가 직접 적어 보낸 주소. 대화 이력은 넣지 않는다 — 과거에 샌 가짜 주소가 화이트리스트가
    된다. 판정 규칙은 `react/link_policy.py` 참조.
    """
    policy = TurnLinkPolicy()
    policy.observe_results(state.get("collected_tool_results"))
    for m in state.get("messages") or []:
        if getattr(m, "role", None) == "system":
            policy.allow_text(getattr(m, "content", None))
    policy.allow_text(state.get("input"))
    return policy


async def _regenerate_names(state: ReactState, content: str) -> tuple[str, dict]:
    """근거에 없는 사람 이름이 있으면 1회 재생성하고, 그래도 남으면 마스킹한다.

    실측 사고: 도구가 작성자를 영문 ID 로만 주자 모델이 음차해 ``jdoe`` 를 "김민수"(실제
    김민서), ``lho`` 를 "이호"(실제 홍길동)로 적었다. 틀린 이름은 **다른 사람에게 발언을
    잘못 붙이는** 피해라, 재생성으로도 안 고쳐지면 원문 유지가 아니라 마스킹한다.
    """
    # 위험한 턴(도구 결과에 @handle 작성자)에서만 돈다. 이유가 둘이다:
    #  ① 시연용 프롬프트는 **의도적으로** 가상의 한국 이름을 만든다. 항상
    #     검사하면 그 정상 동작을 마스킹해 버린다.
    #  ② 스트림 게이트(_name_guard_needed)와 조건이 같아야 한다. 붙들지 않은 턴에서 뒤늦게
    #     본문을 바꾸면 화면 ≠ 히스토리가 된다.
    meta: dict = {"armed": False}
    if not _name_guard_needed(state):
        return content, meta
    evidence = _evidence_text(state)
    bad = find_unsourced_person_names(content, evidence)
    meta.update(armed=True, detected=list(bad), regenerated=False, masked=[])
    if not bad:
        return content, meta
    from agent_arch.llm.base import Message

    logger.warning("[name_guard] 근거 없는 이름 감지 %r → 재생성 시도", bad[:3])
    best, best_bad = content, bad
    try:
        reask = list(state["messages"]) + [
            Message(role="user", content=(
                "[시스템] 방금 답변에 도구 결과에 없는 사람 이름이 있습니다: "
                + ", ".join(bad)
                + ". 사람 이름은 도구 결과에 적힌 문자열만 그대로 옮기세요. "
                "`@`로 시작하는 핸들은 실명 미등록 표시이므로 `@`를 뗀 한글 이름을 만들지 말고 "
                "그대로 쓰고, 확인되지 않은 이름은 아예 쓰지 마세요. 같은 내용으로 다시 작성하세요."
            )),
        ]
        resp = await state["llm"].invoke(reask, tool_mode="NONE")
        meta["regenerated"] = True
        regenerated = (resp.content or "").strip()
        if regenerated:
            again = find_unsourced_person_names(regenerated, evidence)
            if len(again) < len(best_bad):
                best, best_bad = regenerated, again
    except Exception:
        logger.warning("[name_guard] 재생성 중 오류 → 마스킹으로 처리", exc_info=True)
    meta["masked"] = list(best_bad)
    return redact_unsourced_names(best, best_bad), meta


async def _glossary_hold(language: str) -> bool:
    """이 턴은 용어 교정·번역 표기 검증을 위해 스트리밍을 붙들어야 하는가."""
    from agent_arch.agents.context import get_turn_input
    from agent_arch.services.glossary import hold_stream_for_glossary

    return await hold_stream_for_glossary(language, get_turn_input())


async def _ensure_translation_spellings(state: ReactState, content: str) -> str:
    """번역 요청 턴이면 답변에 사전의 목표 표기가 들어 있는지 확인하고, 없으면 1회 재생성한다.

    "에이전트를 중국어로 번역하면?"은 한국어로 답하므로 최종 교정(외국어 답변 전용)이 손댈 수
    없다. 모델이 사전 대신 제 번역을 쓰면 그대로 나갔다(2026-09-23 보고). 여기서는 표기를
    지어내 끼워 넣지 않고 — 문장 어디에 넣을지는 모델 몫이다 — 빠졌는지만 코드로 판정한다.
    재생성해도 빠지면 원문을 두고 경고를 남긴다(얼마나 새는지 셀 수 있게).

    질문은 턴 원문(get_turn_input)으로 판정한다 — state["input"] 에는 이미지·초안 블록이
    붙어 있을 수 있다.
    """
    from agent_arch.agents.context import get_turn_input
    from agent_arch.agents.prompts.system import render_glossary_reask
    from agent_arch.llm.base import Message
    from agent_arch.services.glossary import (
        enforce_glossary,
        missing_spellings,
        question_translation_targets,
    )

    targets = await question_translation_targets(get_turn_input())
    missing = missing_spellings(content, targets)
    if not missing:
        return content
    language = state.get("target_language") or DEFAULT_LANGUAGE
    logger.warning("[glossary] 번역 요청 답변에 사전 표기 누락 %r → 재생성", missing)
    try:
        reask = list(state["messages"]) + [
            Message(role="assistant", content=content),
            Message(role="user", content=render_glossary_reask(missing, language)),
        ]
        resp = await state["llm"].invoke(reask, tool_mode="NONE")
        regenerated = await enforce_glossary((resp.content or "").strip(), language)
        if regenerated and not missing_spellings(regenerated, targets):
            return regenerated
        logger.warning("[glossary] 재생성 후에도 사전 표기 누락 → 원문 유지 (missing=%r)", missing)
    except Exception:
        logger.warning("[glossary] 번역 표기 재생성 중 오류 → 원문 유지", exc_info=True)
    return content


async def _enforce_glossary(content: str, language: str) -> str:
    """최종 답변의 표기를 용어 사전의 확정 표기로 되돌린다.

    실제 구현은 `agent_arch/services/glossary.enforce_glossary` 가 소유한다 — supervisor
    합성 답변도 같은 함수를 거쳐야 하므로 그래프 안에 두지 않는다.
    """
    from agent_arch.services.glossary import enforce_glossary

    return await enforce_glossary(content, language)


async def _regenerate_answer_language(state: ReactState, content: str) -> str:
    """답변이 목표 언어가 아니면 도구 없이(tool_mode=NONE) 목표 언어로 1회 재생성한다.

    프롬프트의 답변 언어 지시를 gemini 가 다른 언어 입력에서 미러링하며 무시하는 경우를
    결정적으로 보강한다. 재생성 실패/여전히 다른 언어면 원문을 그대로 둔다(답변을 깨지
    않음). settings.korean_answer_guard=false 면 비활성.

    목표 언어는 state["target_language"] — orchestrator 가 요청 필드 또는 입력 감지로
    정해 주입한다. 없으면 한국어(기존 동작).
    """
    language = state.get("target_language") or DEFAULT_LANGUAGE
    # 표기 교정은 재생성 여부와 무관하게 **모든 답변**이 거쳐야 한다. 이 함수가 답변이
    # 반드시 통과하는 유일한 지점이라 여기 한 곳에 모은다.
    if not settings.korean_answer_guard or not looks_off_target(content, language):
        return await _enforce_glossary(content, language)
    from agent_arch.agents.prompts.system import render_answer_language_reask
    from agent_arch.llm.base import Message

    logger.warning("[lang_guard] 목표 언어(%s) 아닌 답변 감지 → 재생성 시도 (preview=%r)",
                   language, content[:80])
    try:
        reask = list(state["messages"]) + [
            Message(role="user", content=render_answer_language_reask(language)),
        ]
        resp = await state["llm"].invoke(reask, tool_mode="NONE")
        regenerated = (resp.content or "").strip()
        if regenerated and not looks_off_target(regenerated, language):
            return await _enforce_glossary(regenerated, language)
        logger.warning("[lang_guard] 재생성 결과가 비었거나 여전히 목표 언어가 아님 → 원문 유지")
    except Exception:
        logger.warning("[lang_guard] 재생성 중 오류 → 원문 유지", exc_info=True)
    return await _enforce_glossary(content, language)


async def _answer_node(state: ReactState) -> dict:
    """⑤ 텍스트 최종 답변(비어있지 않음) — oversize trim + 언어 가드 후 LoopResult."""
    # 모델 반복 루프 잔해(같은 문자 120자+ 런)는 trim **앞에서** 접는다 — loop.py 와 동일 순서.
    final_content = collapse_repeated_runs("".join(state["text_buf"]))
    if state["used_tools"] and len(final_content) > _MAX_RESPONSE_CHARS:
        logger.warning("[react_graph] oversized response (%d chars) → trim", len(final_content))
        final_content = final_content[:_MAX_RESPONSE_CHARS] + "\n\n(응답이 너무 길어 일부만 표시됩니다.)"
    # 목표 언어가 아니면 그 언어로 재생성(도구 없이). C2(guard_phantom_action)와 같은 최종 지점.
    final_content = await _regenerate_answer_language(state, final_content)
    final_content = await _ensure_translation_spellings(state, final_content)
    # 근거에 없는 사람 이름(음차 환각)이면 재생성 → 그래도 남으면 마스킹.
    final_content, name_guard = await _regenerate_names(state, final_content)
    # 지어낸 메신저 딥링크를 걷어낸다 — 이름 환각과 같은 등급이다. 파일 주소(`/files/{id}`)는
    # 그 턴 도구 결과에 실려 온 진짜 글 permalink 로 **바꿔 단다**.
    final_content, _ = redact_fabricated_permalinks(
        final_content,
        file_links=build_file_link_map(state["collected_tool_results"]),
    )
    # 첨부 링크의 **목적지**는 코드가 정한다 — 모델이 파일과 주소를 뒤바꿔 짝지었다(2026-09-18).
    final_content, _ = apply_file_links(
        final_content,
        file_refs(state["collected_tool_results"]),
        download_base=settings.mattermost_base_url,
    )
    # 누적된 승인 초안이 있으면 여기서 action_pending 으로 전환(HITL 을 답변 시점으로 미룸).
    result = _final_loop_result(state, final_content, state["finish_reason"])
    # 게이트가 다른 언어를 감지해 스트리밍을 보류했다면, 화면에는 아직 아무것도 안 나갔다.
    # 재생성본을 여기서 한 번에 방출한다 — 저장본(result.content)과 정확히 같은
    # 텍스트라 화면과 히스토리가 어긋나지 않는다.
    if state.get("answer_withheld") and result.content:
        _emit(ProgressEvent("token", content=result.content))
    return {"result": result, "name_guard": name_guard}


async def _retry_any_node(state: ReactState) -> dict:
    """빈 응답 + 도구 있음 + ANY 미시도 → ANY 모드로 강제 재시도(도구 바인딩 유지)."""
    logger.warning("[react_graph] no tool calls + empty content, tools available → retry ANY")
    await state["llm"].bind_tools(state["all_tools"])
    return {"forced_any_mode": True, "iteration": state["iteration"] + 1}


def _fallback_node(state: ReactState) -> dict:
    """ANY 재시도 후에도 실패 or 도구 0개 → 안전 메시지."""
    no_tools = not state["all_tools"]
    fallback_msg = _SAFE_MSG_NO_TOOLS if no_tools else _SAFE_MSG_FALLBACK
    logger.warning(
        "[react_graph] fallback: %s",
        "도구 탐색 실패(0개)" if no_tools else "ANY 재시도 후에도 도구호출 없음",
    )
    # call_model 이 빈 응답이라 토큰을 안 흘렸으므로, 사용자가 빈 화면을 안 보도록 방출.
    _emit(ProgressEvent("token", content=fallback_msg))
    return {"result": _final_loop_result(state, fallback_msg, "fallback")}


async def _forced_none_answer(
    state: ReactState,
    *,
    empty_msg: str,
    swallow_errors: bool,
    log_label: str,
) -> dict:
    """도구 없이(``tool_mode="NONE"``) 한 번 더 스트리밍해 답변을 확정하는 공통 골격.

    ``_malformed_node`` 와 ``_force_answer_node`` 가 같은 일을 하는데, 둘 다 토큰을
    **직접** 방출하고 최종 답변 노드의 가드를 거치지 않았다. 즉 언어 가드도 이름 가드도
    안 돌았다. 특히 ``_force_answer_node`` 는 max_iterations 소진 경로 — **도구를 여러 번
    돈 긴 메신저 턴이 정확히 여기로 떨어진다.** 위험이 가장 높은 턴이 유일하게 무방비였다.

    ``_answer_node`` 와 **같은 순서**(언어 → 이름)로 가드를 태우고, 보류된 경우 저장본과
    똑같은 텍스트를 한 번에 방출한다(화면 = 히스토리).

    Args:
        empty_msg: 재시도가 빈 응답일 때 내보낼 문구. 두 노드의 유일한 실질 차이다.
        swallow_errors: 스트림 예외를 삼킬지. ``_malformed_node`` 는 반드시 삼켜야 한다 —
            그래프 밖으로 흘리면 스트림이 종료 이벤트 없이 끊겨 화면이 무한 로딩으로 남는다.
        log_label: 진단 로그 접두어.
    """
    llm = state["llm"]
    await llm.bind_tools([])
    _lang = state.get("target_language") or DEFAULT_LANGUAGE
    gate = _AnswerLanguageGate(
        language=_lang,
        hold_all=(needs_draft_cancel_guard(state.get("input", ""))
                  or _name_guard_needed(state) or _link_guard_needed(state)
                  or await _glossary_hold(_lang)),
    )
    text: list[str] = []
    finish = "stop"
    usage = None
    try:
        async for event in llm.stream(state["messages"], tool_mode="NONE"):
            if event.type == "text":
                if event.content:
                    text.append(event.content)
                    gate.feed(event.content)
            elif event.type == "done":
                finish = event.finish_reason or "stop"
                usage = event.usage
    except Exception:
        if not swallow_errors:
            raise
        # loop.py 의 malformed 재시도 실패와 같은 자리 — 두 실행 경로 모두 이벤트로 남긴다.
        logger.warning("[react_graph] %s 재시도 실패 — 폴백 문구로 종결", log_label, exc_info=True,
                       extra=event_extra("exception"))
        finish = "error"
    gate.close()

    tu = state["token_usage"]
    merge_usage(tu, usage)
    count = state["llm_invoke_count"] + 1
    final_content = "".join(text)
    logger.info("[react_graph] %s 재시도 결과 finish=%s len=%d",
                log_label, finish, len(final_content))

    name_guard: dict = {"armed": False}
    had_model_content = bool(final_content.strip())
    if not had_model_content:
        # 빈 응답 — 토큰이 하나도 안 나가면 SSE 가 내용 없이 끝나 화면이 무한 로딩으로 남는다
        # (실측 2026-08-18). 가드를 태울 본문이 없으므로 폴백 문구를 그대로 방출한다.
        final_content = empty_msg
        _emit(ProgressEvent("token", content=final_content))
    else:
        final_content = await _regenerate_answer_language(state, final_content)
        final_content = await _ensure_translation_spellings(state, final_content)
        final_content, name_guard = await _regenerate_names(state, final_content)
        final_content, _ = redact_fabricated_permalinks(
            final_content,
            file_links=build_file_link_map(state["collected_tool_results"]),
        )
        final_content, _ = apply_file_links(
            final_content,
            file_refs(state["collected_tool_results"]),
            download_base=settings.mattermost_base_url,
        )
    result = _final_loop_result(state, final_content, finish, invoke=count, token_usage=tu)
    if had_model_content and gate.withheld and result.content:
        # 가드 전 원문을 내보내면 저장본만 고쳐진다. 정상 경로와 같은 최종본문을 쓴다.
        _emit(ProgressEvent("token", content=result.content))

    return {
        "llm_invoke_count": count,
        "token_usage": tu,
        "name_guard": name_guard,
        "result": result,
    }


async def _malformed_node(state: ReactState) -> dict:
    """③ malformed_function_call → tool_mode=NONE 으로 재 스트림 후 종결.

    ⚠️ **재시도가 비면 반드시 폴백 문구를 방출한다.** 안 하면 토큰이 하나도 안 나가고
    result.content 도 빈 문자열이라, SSE 가 내용 없이 끝나 화면이 무한 로딩으로 남는다
    (실측 2026-08-18: 되묻기에 "1번"/"3번" 으로 답한 턴 두 건). 같은 tool_mode=NONE 재시도를
    하는 형제 노드 _force_answer_node 에는 이 가드가 있는데 여기만 빠져 있었다.

    예외도 삼킨다 — 그래프 밖으로 흘리면 스트림이 종료 이벤트 없이 끊겨 역시 무한 로딩이 된다.

    진단 로그: malformed 는 2026-07-31 이후 29건 발생했는데 실패 원인이 로그에 남지 않아
    좁힐 수 없었다. 재시도의 finish_reason·길이를 남겨 다음 발생 때
    (a) 토큰 소진(length) (b) 순수 빈 응답(stop, len=0) (c) 예외(error) 를 구별한다.
    """
    logger.warning("[react_graph] MALFORMED_FUNCTION_CALL → tool_mode=NONE 재호출")
    return await _forced_none_answer(
        state,
        empty_msg=_SAFE_MSG_MALFORMED,
        swallow_errors=True,
        log_label="malformed",
    )


async def _execute_node(state: ReactState) -> dict:
    """④ tool 실행 — 의도 자동주입 + HITL confirm 강제 + 병렬 실행 + 발견도구 머지 + HITL 대기 판정."""
    llm = state["llm"]
    input_text = state["input"]
    iter_tool_calls = state["iter_tool_calls"]
    all_tools = state["all_tools"]
    messages = state["messages"]

    # 답글 의도 자동 주입: '답글/댓글/스레드' → reply_to_latest=True (명시값 있으면 유지).
    if _wants_thread_reply(input_text):
        for tc in iter_tool_calls:
            if tc.name in _MM_SEND_TOOLS:
                a = tc.arguments or {}
                if not (a.get("root_id") or a.get("reply_to_post_id") or a.get("reply_to_latest")):
                    tc.arguments = {**a, "reply_to_latest": True}

    # 리액션 의도 자동 주입: '이모지/리액션/체크' → react_to_latest=True.
    if _wants_react_latest(input_text):
        for tc in iter_tool_calls:
            if tc.name == _MM_REACT_TOOL:
                a = tc.arguments or {}
                if not a.get("post_id") and not a.get("react_to_latest"):
                    tc.arguments = {**a, "react_to_latest": True}

    # 스레드 조회의 root_id 는 **모델이 아니라 시스템이** 채운다. 프롬프트로 알려줘도 모델이
    # 다른 post_id 를 넣으면 다시 엉뚱한 스레드를 읽는다 — 이 값은 봇이 이미 알고 있다.
    origin_root = _origin_root_id()
    if origin_root:
        for tc in iter_tool_calls:
            if tc.name == THREAD_TOOL_NAME:
                a = tc.arguments or {}
                if a.get("root_id") != origin_root:
                    tc.arguments = {**a, "root_id": origin_root}

    # Preserve retrieval intent; a post permalink does not require a URL in its body.
    link_forced_ids: set = set()
    for tc in iter_tool_calls:
        if tc.name == AUTHOR_TOOL_NAME:
            a = tc.arguments or {}
            tc.arguments = {**a, "search_intent": input_text}
            # A post permalink request does not require an external URL in the body.
            if a.get("must_contain_link"):
                link_forced_ids.add(tc.id)

    # 모델이 제품 이름을 회사명으로 오해해 `query` 에서 지우면 되살린다.
    #
    # 실측(2026-09-16): "에이전트의 권한과 역할 문서를 언급한 링크 찾아줘" → `query="권한 역할
    # 문서"`. `에이전트` 가 빠지자 권한 얘기가 든 기술제안서가 통째로 걸려 엉뚱한 답이 나갔다.
    # 직전 다섯 턴은 `에이전트` 를 남겨 정답을 찾았다 — 프롬프트 규칙("회사명은 query 에 남기지
    # 마라")이 제품 이름까지 먹은 것이다. 프롬프트에도 예외를 적었지만 확률적이라 코드로 받친다.
    for tc in iter_tool_calls:
        if tc.name != _SCOPED_SEARCH_TOOL:
            continue
        a = tc.arguments or {}
        q = str(a.get("query") or "")
        if not q.strip():
            continue                      # 주제어 없는 활동형 질문 — 최근순 경로다
        # 부르는 말을 먼저 걷어낸다 — 남아 있으면 그건 주제어다.
        topic_text = _VOCATIVE_RE.sub(" ", input_text)
        for term in _PRODUCT_TERMS:
            if term.casefold() in topic_text.casefold() and term.casefold() not in q.casefold():
                q = f"{term} {q}".strip()
                logger.info("[react_graph] 질의에서 지워진 제품명 복원: %r → %r",
                            a.get("query"), q)
        if q != a.get("query"):
            tc.arguments = {**a, "query": q}

    # "이 스레드"·"이 채널" 턴은 조회를 **지금 채널로 묶는다.** 지시어가 없으면 묶지 않는다
    # — 사내 지식검색("지난주 품질 이슈 찾아줘")의 본래 용도가 전사 검색이기 때문이다.
    # 묶었는데 0건이면 아래에서 범위를 풀어 다시 부른다(판정이 틀려도 답을 잃지 않는다).
    scoped_ids: set = set()
    origin_channel = _origin_ctx().get("channel_id")
    if origin_channel and scope_to_current(input_text):
        for tc in iter_tool_calls:
            a = tc.arguments or {}
            if tc.name == _SCOPED_SEARCH_TOOL and not a.get("channel_ids"):
                tc.arguments = {**a, "channel_ids": [origin_channel]}
                scoped_ids.add(tc.id)
            elif tc.name == _SCOPED_CHANNEL_TOOL and a.get("channel_id") != origin_channel:
                narrowed = {**a, "channel_id": origin_channel}
                narrowed.pop("channel_keyword", None)   # 키워드가 남으면 모델이 찍은 채널이 이긴다
                tc.arguments = narrowed
        if scoped_ids or any(tc.name == _SCOPED_CHANNEL_TOOL for tc in iter_tool_calls):
            logger.info("[react_graph] 현재 채널로 조회 범위 고정: channel_id=%s", origin_channel)

    # guide-first 강제 턴: 가이드 검색어는 LLM 의 축약 검색어 대신 질문 문장을 그대로 쓴다.
    # 실측(2026-09-04): "시급제 연장근무수당 조정" 같은 축약어는 급여 문단만 잡고,
    # 질문 문장("07:40 출근 ... 연장 20분 없애는 방법")은 근태 설정 문단을 1위로 잡았다.
    # 턴 원문으로만 강제된 턴은 원문을 검색어로 쓴다(guide_query_text, 2026-10-13) — 도구의 원문
    # 폴백(#577)과 같은 문장이라 중복 제거돼 임베딩이 늘지 않는다.
    if state.get("guide_first") and state["iteration"] == 0 and input_text:
        raw_query = (state.get("guide_query_text") or "").strip()[:_GUIDE_QUERY_MAX_LEN]
        question = input_text.strip()[:_GUIDE_QUERY_MAX_LEN]
        for tc in iter_tool_calls:
            if tc.name == GUIDE_TOOL_NAME:
                a = tc.arguments or {}
                if raw_query:
                    # 원문 폴백 턴: 원문이 서브쿼리·모델 검색어보다 짧아도 교체(길이 비교는 축약어 방지용이지 원문엔 무관).
                    tc.arguments = {**a, "query": raw_query}
                elif len(question) > len(str(a.get("query") or "")):
                    tc.arguments = {**a, "query": question}

    # HITL: 승인필요 액션은 confirm=False 강제(초안만). 한 턴 계획분을 기억해 둔다.
    hitl_pending_tcs = []
    if settings.use_langgraph_hitl:
        for tc in iter_tool_calls:
            if requires_approval(tc.name):
                tc.arguments = {**tc.arguments, "confirm": False}
                hitl_pending_tcs.append(tc)

    messages.append(Message(
        role="assistant",
        content="",
        tool_calls=[tc.model_dump() for tc in iter_tool_calls],
    ))

    # 체크포인트 c(도구 실행 직전): 부수효과 도구가 돌기 전에 멈춘다.
    if is_cancelled():
        return {"cancelled": True, "used_tools": True, "messages": messages}

    # ── 발견 대기열 리셋은 반드시 이 노드 안에서 ──────────────────────────────────
    # LangGraph 는 노드마다 컨텍스트를 격리(copy_context)하므로 start 노드의 reset 은
    # 여기로 전파되지 않는다(실측). reset → execute_tools(gather 자식이 _enqueue) →
    # drain 을 execute 노드 단일 컨텍스트에 묶어야 발견 도구가 유실되지 않는다. 도구
    # 발견은 execute_tools 안에서만 일어나므로 iteration 마다 리셋해도 semantics 동일
    # (기존 루프도 drain 시 동일 리스트를 비운다).
    reset_discovered_tools()
    collected_tool_calls = state["collected_tool_calls"]
    from agent_arch.agents.react.budget import executed_web
    web_rounds = state.get("web_rounds") or 0
    exec_results = await execute_tools(iter_tool_calls, all_tools,
                                       prior_calls=collected_tool_calls, prior_results=state["collected_tool_results"],
                                       web_rounds=web_rounds)
    # 선실행(웹 조사 계획)은 모델이 고른 단계가 아니라 단계 예산에서 뺀다 — 세면 모델이 검색 한 번 더 하고 나면 web_fetch 로
    # 원 페이지를 열 단계가 남지 않았다(2026-09-23 실측: "단계 상한 도달 tool=web_fetch").
    if executed_web(exec_results) and not state.get("prefetch_phase"):
        web_rounds += 1
    if scoped_ids:
        exec_results = await _widen_empty_scoped(exec_results, scoped_ids, all_tools)
    if link_forced_ids:
        exec_results = await _widen_empty_linked(exec_results, link_forced_ids, all_tools)
    collected_tool_calls.extend(tc.model_dump() for tc in iter_tool_calls)

    # tool_search 가 발견·게이트 통과시킨 도구 머지 — 다음 iteration bind_tools 부터 호출 가능.
    known_tool_names = state["known_tool_names"]
    discovered = drain_discovered_tools()
    newly_discovered = [t for t in discovered if t.name not in known_tool_names]
    if newly_discovered:
        all_tools.extend(newly_discovered)
        known_tool_names.update(t.name for t in newly_discovered)
        logger.info("[react_graph] tool_search 발견 도구 바인딩: %s", [t.name for t in newly_discovered])

    collected_tool_results = state["collected_tool_results"]
    used_tool_names = state["used_tool_names"]
    # 이 턴에 회상된 사용자 규칙을 이번 도구 결과 묶음의 **마지막 결과 뒤**에 한 번 붙인다(도구 무관).
    # 모델은 무엇을 되물을지 고르는 순간 가장 가까운 도구 데이터를 따르므로, 규칙이 앞쪽 기억 블록에만
    # 있으면 도구가 '받아야 할 값'으로 나열한 항목을 또 묻는다. LLM 이 읽는 메시지에만 싣고 저장되는
    # 도구 결과(collected_tool_results → tool_ctx·이력)에는 넣지 않는다 — 규칙은 매 턴 회상이 다시 준다.
    reminder = _memory_rule_reminder() if exec_results else ""
    if reminder:
        # 관측(INFO): 운영에서 리마인더가 실제로 붙었는지 — 없으면 "안 붙었다"와 "붙었는데 모델이 물었다"를 못 가른다.
        last = exec_results[-1][1]
        logger.info(
            "[memory_rules] 리마인더 부착 tool=%s rules=%d form=%s",
            exec_results[-1][0].name, reminder.count("\n- "),
            "json" if isinstance(last, str) and last.lstrip().startswith("{") else "text",
        )
    for i, (tc, result_content) in enumerate(exec_results):
        used_tool_names.append(tc.name)
        shown = _attach_rule_reminder(result_content, reminder) if reminder and i == len(exec_results) - 1 else result_content
        messages.append(Message(
            role="tool",
            content=shown,
            name=tc.name,
            tool_call_id=tc.id,
        ))
        collected_tool_results.append({
            "tool_call_id": tc.id,
            "name": tc.name,
            "content": result_content,
        })

    await llm.bind_tools(all_tools)

    # 링크 허용 정책을 화면 쪽(orchestrator 의 StreamSanitizer)에 알린다 — 도구가 돌아야 정책이
    # 켜지므로 최종 답변 토큰보다 항상 먼저 나간다. 비스트리밍(ainvoke)에서는 writer 가 없어
    # no-op 이고, 그 경로는 _final_loop_result 의 전문 정화로 덮인다.
    policy = _link_policy(state)
    if policy.active:
        _emit(ProgressEvent("link_policy", content=policy.dumps()))

    # 같은 실패를 반복하고 있는지 — 상한을 넘으면 _route_loop 가 루프를 끊는다.
    sig = _failure_signature(exec_results)
    streak = (state.get("fail_streak") or 0) + 1 if sig and sig == state.get("fail_signature") else (1 if sig else 0)
    if sig and streak > 1:
        logger.warning("[react_graph] 같은 도구 실패 %d회 연속: %s", streak, sig[:160])

    upd = {
        "used_tools": True,
        "fail_signature": sig,
        "fail_streak": streak,
        "web_rounds": web_rounds,
        "messages": messages,
        "all_tools": all_tools,
        "collected_tool_calls": collected_tool_calls,
        "collected_tool_results": collected_tool_results,
        "known_tool_names": known_tool_names,
        "used_tool_names": used_tool_names,
        "iteration": state["iteration"] + 1,
    }
    if newly_discovered:
        upd["discovered_tool_names"] = [t.name for t in newly_discovered]  # 스텝 집계용

    # HITL: 승인필요 도구의 유효 draft 를 '누적'만 하고 pause 하지 않는다(마지막으로 미룸).
    # 즉시 멈추면 워커가 조회(읽기)·요약 등 나머지 작업을 못 함. 초안을 pending_drafts 에
    # 모아두고 루프를 계속해 LLM 이 조회·요약을 마치게 한 뒤, 최종 답변(answer/force_answer)
    # 시점에 이 큐로 승인 대기한다. (부수효과 0 — confirm=False 초안만.)
    if hitl_pending_tcs:
        new_drafts = [
            pa for tc in hitl_pending_tcs
            if (pa := _build_pending_action(tc, collected_tool_results)).get("draft_view")
        ]
        if new_drafts:
            merged = _accumulate_drafts(state.get("pending_drafts") or [], new_drafts)
            upd["pending_drafts"] = merged
            logger.info(
                "[react_graph] HITL 초안 누적(pause 보류) — 누적 %d건: %s",
                len(merged), [p["tool_name"] for p in merged],
            )
        else:
            logger.info("[react_graph] HITL: 승인필요 도구 draft view 없음 → 루프 계속(복구)")
    return upd


# 같은 도구가 같은 이유로 이만큼 연속 실패하면 루프를 끊는다. 모델이 인자를 못 만드는
# 상황은 몇 번 더 돈다고 풀리지 않는다 — 2026-09-14 운영에서 create_file 이 13회 연속
# 같은 에러로 실패하며 5분을 돌았고, 그 시간 내내 사용자에게는 아무 신호도 없었다.
_MAX_SAME_FAILURES = 3


def _failure_signature(exec_results) -> Optional[str]:
    """이번 iteration 의 도구 실패를 한 줄 서명으로 만든다. 실패가 없으면 None.

    도구명 + 에러 앞부분으로 묶는다 — 인자만 조금씩 바꾼 재시도도 같은 서명이 되어야
    "같은 실패를 반복 중"으로 잡힌다.

    Args:
        exec_results: ``execute_tools`` 가 돌려준 (ToolCall, 결과문자열) 목록.

    Returns:
        ``"도구명:에러앞부분|…"`` 서명. 실패가 하나도 없으면 ``None``.
    """
    sigs = sorted(
        f"{tc.name}:{(content or '')[:80]}"
        for tc, content in exec_results
        if isinstance(content, str) and content.startswith("[ERROR]")
    )
    return "|".join(sigs) or None


def _is_empty_search(content: str) -> bool:
    """검색 결과가 0건인지. 판정 못 하면 False(섣불리 재검색하지 않는다)."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("count") == 0:
        return True
    results = data.get("results")
    return isinstance(results, list) and not results


async def _widen_empty_scoped(exec_results, scoped_ids: set, all_tools: list):
    """현재 채널로 묶어 검색했는데 0건이면 **범위를 풀어 한 번 더** 부른다.

    지시어 판정은 휴리스틱이라 언젠가 틀린다. 틀렸을 때의 대가를 "한 번 더 검색"으로
    묶어 두면, 판정 실패가 **답을 못 찾는 사고로 이어지지 않는다**(같은 소스의 기간
    자동 확장과 같은 발상).

    Args:
        exec_results: ``execute_tools`` 결과 [(ToolCall, 결과문자열)].
        scoped_ids: 범위를 주입한 tool_call id 집합.
        all_tools: 이번 턴 바인딩된 도구들.

    Returns:
        재검색 결과로 교체된 exec_results(순서 보존).
    """
    retry = [
        (i, tc) for i, (tc, content) in enumerate(exec_results)
        if tc.id in scoped_ids and _is_empty_search(content)
    ]
    if not retry:
        return exec_results

    out = list(exec_results)
    for idx, tc in retry:
        widened = {k: v for k, v in (tc.arguments or {}).items() if k != "channel_ids"}
        tc.arguments = widened
        logger.info("[react_graph] 현재 채널 0건 → 범위 해제 재검색: %s", tc.name)
        again = await execute_tools([tc], all_tools)
        if again:
            out[idx] = again[0]
    return out


async def _widen_empty_linked(exec_results, link_forced_ids: set, all_tools: list):
    """링크 있는 글로 한정해 0건이면 **조건을 풀어 한 번 더** 부른다.

    `must_contain_link` 는 URL 이 든 글만 남기는 하드 필터라, 사용자가 "링크" 라고 말했어도
    정작 그 글이 파일 첨부였거나 URL 없이 제목만 적혀 있으면 0건이 된다. 판정이 틀렸을 때의
    대가를 "한 번 더 검색" 으로 묶어 둔다(`_widen_empty_scoped` 와 같은 발상).

    Args:
        exec_results: ``execute_tools`` 결과 [(ToolCall, 결과문자열)].
        link_forced_ids: `must_contain_link` 를 주입한 tool_call id 집합.
        all_tools: 이번 턴 바인딩된 도구들.

    Returns:
        재검색 결과로 교체된 exec_results(순서 보존).
    """
    retry = [
        (i, tc) for i, (tc, content) in enumerate(exec_results)
        if tc.id in link_forced_ids and _is_empty_search(content)
    ]
    if not retry:
        return exec_results

    out = list(exec_results)
    for idx, tc in retry:
        tc.arguments = {k: v for k, v in (tc.arguments or {}).items()
                        if k != "must_contain_link"}
        logger.info("[react_graph] 링크 한정 0건 → 조건 해제 재검색: %s", tc.name)
        again = await execute_tools([tc], all_tools)
        if again:
            out[idx] = again[0]
    return out


async def _force_answer_node(state: ReactState) -> dict:
    """max_iterations 소진(또는 같은 실패 반복) → tool_mode=NONE 강제 최종 답변."""
    repeated = (state.get("fail_streak") or 0) >= _MAX_SAME_FAILURES
    logger.warning("[react_graph] %s, tool_mode=NONE 강제 스트리밍",
                   "같은 도구 실패 반복" if repeated else "max_iterations 소진")
    return await _forced_none_answer(
        state,
        empty_msg=(
            "요청하신 작업을 완료하지 못했습니다. 같은 오류가 반복돼 중단했어요 — "
            "조금 더 구체적으로 다시 말씀해 주시겠어요?"
            if repeated else
            "최대 반복 횟수를 초과하여 답변을 생성하지 못했습니다."
        ),
        swallow_errors=False,
        log_label="force_answer",
    )


async def _salvage_node(state: ReactState) -> dict:
    """빈 응답이 ANY 재시도 뒤에도 이어졌는데 웹 근거는 모였다 → 근거를 절반 예산으로 줄이고 도구 없이 답한다.

    실측(2026-09-28 vllm EPDM): 근거를 가장 많이 모은 턴이 컨텍스트 초과로 빈 응답을 두 번 내고 "조회하지 못했습니다"로 끝났다.
    """
    cap = max(20_000, (_evidence.budget(state["llm"]) or 60_000) // 2)
    saved = _evidence.compact(state["messages"], cap)
    logger.warning("[react_graph] 빈 응답 반복 — 웹 근거를 %d자 줄이고 도구 없이 답한다", saved)
    return await _forced_none_answer(state, empty_msg=_SAFE_MSG_FALLBACK, swallow_errors=True, log_label="salvage")


def _stopped_node(state: ReactState) -> dict:
    """사용자 중단 종결."""
    return {"result": _stopped_result(state)}


# ── 라우팅 ────────────────────────────────────────────────────────────────────

def _route_loop(state: ReactState) -> str:
    """iteration 경계(체크포인트 a) — 중단/반복실패/반복/소진 분기."""
    if is_cancelled():
        return "stopped"
    if (state.get("fail_streak") or 0) >= _MAX_SAME_FAILURES:
        logger.warning(
            "[react_graph] 같은 실패 %d회 — 루프 중단하고 답변으로 넘어간다: %s",
            state["fail_streak"], (state.get("fail_signature") or "")[:160],
        )
        return "force_answer"
    if state["iteration"] < state["max_iterations"]:
        return "call_model"
    return "force_answer"


async def _verify_node(state: ReactState) -> dict:
    """④′ 답변 검증 — 최종 답 초안이 요청을 충족하는지 본다(react/verify). shadow 는 기록만, on 은 보강 루프.

    검증기 실패는 pass 로 흡수한다(검증기가 죽어 답이 막히면 안 된다). 보강은 횟수 상한·반복 상한·승인 초안
    없음 조건을 모두 만족할 때만 돈다. 보강 시 초안을 assistant 메시지로 남기고 그 뒤에 지시(system)를 붙여
    모델이 초안을 이어 고치게 한다.
    """
    draft = "".join(state["text_buf"])
    rounds = int(state.get("verify_rounds") or 0)
    mode = _verify.mode()
    verdict: dict
    facts: dict = {}
    usage = None
    try:
        verdict, facts, usage = await _verify.run_verifier(state.get("input") or "", draft, state)
    except Exception as e:  # noqa: BLE001 — 검증기 장애는 흐름을 막지 않는다
        logger.warning("[verify] 검증기 실패 → pass 처리: %s", e)
        verdict = {"verdict": "pass", "issues": [], "action": {"type": "none", "detail": ""}, "error": str(e)[:200]}
    tu = state["token_usage"]
    merge_usage(tu, usage)
    meta = {"mode": mode, "round": rounds + 1, **{k: verdict.get(k) for k in ("verdict", "issues", "action", "latency_ms", "model", "error", "parse_error") if verdict.get(k) is not None},
            "facts": {k: facts.get(k) for k in ("tool_calls", "tool_errors", "web_searches", "context_reads", "empty_searches", "results_total", "results_with_date", "answer_urls_not_in_results", "duplicate_calls", "budget_capped") if k in facts}}
    upd: dict = {
        "verify_meta": list(state.get("verify_meta") or []) + [meta],
        "token_usage": tu,
        "llm_invoke_count": state["llm_invoke_count"] + (1 if usage is not None or "error" not in verdict else 0),
        "verify_decision": "answer",
    }
    a_type = (verdict.get("action") or {}).get("type")
    can_revise = (
        mode == _verify.MODE_ON and verdict.get("verdict") == "revise"
        and rounds < _verify.max_rounds()
        and a_type not in ("none", "rewrite")          # rewrite 는 아래 1회 고쳐 쓰기로 끝낸다(다시 검증하지 않는다)
        and not state.get("pending_drafts")
        and state["iteration"] + 1 < state["max_iterations"]
        # 더 찾으라는 보강(search·fetch)은 웹 근거 예산 안에서만 — 넘치면 컨텍스트를 넘겨 답 자체를 잃는다(2026-09-28 EPDM)
        and not ((verdict.get("action") or {}).get("type") in ("search", "fetch") and _evidence.over_budget(state["messages"], llm=state["llm"]))
    )
    logger.info("[verify] mode=%s round=%d verdict=%s action=%s issues=%s → %s",
                mode, rounds + 1, verdict.get("verdict"), (verdict.get("action") or {}).get("type"),
                verdict.get("issues"), "revise" if can_revise else "answer")
    # 검증기가 "근거는 결과에 있는데 초안이 어긋난다"(rewrite)를 냈다 → 도구 없이 1회 고쳐 쓰고 끝낸다(라운드와 무관, 재검증 없음).
    # 실측(2026-09-28): gemini 유로는 2라운드에 rewrite 가 나왔는데 라운드가 없어 틀린 초안이 나갔고, 실측 vllm 유로는
    # rewrite → 재작성 → 재검증 rewrite → 고쳐 쓰기로 추론 켠 답을 세 번 써 79초가 걸렸다. 고쳐 쓰기는 한 번이면 된다.
    final_rewrite = (
        mode == _verify.MODE_ON and verdict.get("verdict") == "revise" and a_type == "rewrite"
        and not state.get("rewrite_done") and not state.get("pending_drafts")
    )
    if can_revise or final_rewrite:
        msgs = state["messages"]
        msgs.append(Message(role="assistant", content=draft))
        msgs.append(_verify.feedback_message(verdict))
        _emit(ProgressEvent("thinking", content="답변을 검증하고 보강하고 있어요..."))
    if can_revise:
        upd.update({"messages": msgs, "iteration": state["iteration"] + 1, "verify_rounds": rounds + 1,
                    "text_buf": [], "verify_decision": "revise"})
    elif final_rewrite:
        logger.info("[verify] rewrite → 도구 없이 1회 고쳐 쓰고 끝낸다")
        upd.update({"messages": msgs, "verify_decision": "rewrite", "rewrite_done": True})
    return upd


async def _rewrite_node(state: ReactState) -> dict:
    """도구 없이(tool_mode=NONE) 검증 지시대로 고쳐 쓴다. 비거나 실패하면 원 초안 그대로 — 답을 잃지 않는다.
    방출은 하지 않는다: on 모드 검증 턴은 토큰을 붙들고 있고(answer_withheld), 최종 답변 노드가 가드를 거친 본문을 한 번에 내보낸다."""
    llm = state["llm"]
    draft = "".join(state["text_buf"])
    text: list[str] = []
    usage = None
    try:
        await llm.bind_tools([])
        async for event in llm.stream(state["messages"], tool_mode="NONE"):
            if event.type == "text" and event.content:
                text.append(event.content)
            elif event.type == "done":
                usage = event.usage
    except Exception:
        logger.warning("[verify] 고쳐 쓰기 실패 — 원 초안 유지", exc_info=True)
    tu = state["token_usage"]
    merge_usage(tu, usage)
    fixed = "".join(text)
    logger.info("[verify] 고쳐 쓰기 결과 len=%d (원 초안 %d)", len(fixed), len(draft))
    return {"text_buf": [fixed] if fixed.strip() else [draft], "token_usage": tu,
            "llm_invoke_count": state["llm_invoke_count"] + 1, "answer_withheld": True}


def _route_after_verify(state: ReactState) -> str:
    d = state.get("verify_decision")
    return "loop_guard" if d == "revise" else "rewrite" if d == "rewrite" else "answer"


def _route_after_model(state: ReactState) -> str:
    """call_model 후 분기 — react_loop_events 의 종료조건 로직과 동형(+ 답변 검증 분기)."""
    if state.get("cancelled"):
        return "stopped"
    fr = state["finish_reason"]
    if fr == "malformed_function_call":
        return "malformed"
    if fr != "tool_calls" or not state["iter_tool_calls"]:
        if "".join(state["text_buf"]):
            return "verify" if _verify.should_verify(state) else "answer"
        if state["all_tools"] and not state["forced_any_done"]:
            return "retry_any"
        if _evidence._web_tool_messages(state["messages"]):
            return "salvage"             # 모은 웹 근거가 있으면 폴백 문구 대신 줄여서 도구 없이 답한다
        return "fallback"
    return "execute"


def _route_after_execute(state: ReactState) -> str:
    """execute 후 분기 — 중단/HITL 대기(END)/다음 iteration."""
    if state.get("cancelled"):
        return "stopped"
    if state.get("result") is not None:
        return "end"  # HITL 대기 초안 → 종결
    if state.get("prefetch_phase"):
        return "auto_fetch"  # 선실행 단계: 검색 뒤 상위 페이지 열기(1회), 그다음 모델로
    return "loop_guard"


# ── 빌드/실행 ─────────────────────────────────────────────────────────────────

def _build_react_graph():
    """ReAct StateGraph 를 컴파일한다(체크포인터 없음)."""
    g = StateGraph(ReactState)
    # 노드는 _instrument 로 감싸 실행시간·방문을 node_trace 에 남긴다(스텝 집계).
    g.add_node("start", _instrument("start")(_start_node))
    g.add_node("loop_guard", _instrument("loop_guard")(_loop_guard_node))
    g.add_node("call_model", _instrument("call_model")(_call_model_node))
    g.add_node("execute", _instrument("execute")(_execute_node))
    g.add_node("retry_any", _instrument("retry_any")(_retry_any_node))
    g.add_node("verify", _instrument("verify")(_verify_node))
    g.add_node("auto_fetch", _instrument("auto_fetch")(_auto_fetch_node))
    g.add_node("sufficiency", _instrument("sufficiency")(_sufficiency_node))
    g.add_node("answer", _instrument("answer")(_answer_node))
    g.add_node("fallback", _instrument("fallback")(_fallback_node))
    g.add_node("malformed", _instrument("malformed")(_malformed_node))
    g.add_node("force_answer", _instrument("force_answer")(_force_answer_node))
    g.add_node("salvage", _instrument("salvage")(_salvage_node))
    g.add_node("rewrite", _instrument("rewrite")(_rewrite_node))
    g.add_node("stopped", _instrument("stopped")(_stopped_node))

    g.add_edge(START, "start")
    g.add_conditional_edges("start", _route_after_start, {"execute": "execute", "loop_guard": "loop_guard"})
    g.add_conditional_edges("loop_guard", _route_loop, {
        "stopped": "stopped", "call_model": "call_model", "force_answer": "force_answer",
    })
    g.add_conditional_edges("call_model", _route_after_model, {
        "stopped": "stopped", "malformed": "malformed", "answer": "answer", "verify": "verify",
        "retry_any": "retry_any", "fallback": "fallback", "salvage": "salvage", "execute": "execute",
    })
    g.add_conditional_edges("verify", _route_after_verify, {"answer": "answer", "loop_guard": "loop_guard", "rewrite": "rewrite"})
    g.add_edge("rewrite", "answer")
    g.add_edge("retry_any", "loop_guard")
    g.add_conditional_edges("execute", _route_after_execute, {
        "stopped": "stopped", "end": END, "loop_guard": "loop_guard", "auto_fetch": "auto_fetch",
    })
    g.add_conditional_edges("auto_fetch", _route_after_auto_fetch,
                            {"execute": "execute", "loop_guard": "loop_guard", "sufficiency": "sufficiency"})
    g.add_conditional_edges("sufficiency", _route_after_sufficiency, {"execute": "execute", "loop_guard": "loop_guard"})
    for term in ("answer", "fallback", "malformed", "force_answer", "salvage", "stopped"):
        g.add_edge(term, END)
    return g.compile()


_REACT_GRAPH = None


def get_react_graph():
    """프로세스 단일 ReAct 그래프(lazy singleton). llm/도구는 매 실행 state 로 주입."""
    global _REACT_GRAPH
    if _REACT_GRAPH is None:
        _REACT_GRAPH = _build_react_graph()
        logger.info("[react_graph] compiled")
    return _REACT_GRAPH


async def run_react_graph(llm, input, messages, all_tools, search_fn, max_iterations: int = 10,
                          target_language: str = DEFAULT_LANGUAGE,
                          pending_draft_force: bool = False,
                          prefetch: Optional[list] = None,
                          auto_fetch: bool = False,
                          turn_input_guide: bool = False) -> LoopResult:
    """ReAct 그래프를 ainvoke 하고 최종 LoopResult 를 반환한다(비스트리밍, run() 패리티).

    react_loop_events 와 동일 시그니처. llm 은 호출 전 이미 bind_tools 된 상태여야 한다
    (_prepare_tools 가 진입 전 바인딩 — react_loop_events 와 동일 계약).

    target_language: 답변 언어 가드가 목표로 삼을 언어(ko/en/zh/ja). 기본은 한국어라
    넘기지 않는 호출자는 이관 전과 동일하게 동작한다.
    """
    graph = get_react_graph()
    init: ReactState = {
        "llm": llm,
        "input": input,
        "messages": messages,
        "all_tools": all_tools,
        "search_fn": search_fn,
        "max_iterations": max_iterations,
        "target_language": target_language,
        "pending_draft_force": pending_draft_force,
        "prefetch_calls": list(prefetch or []),
        "auto_fetch_enabled": bool(auto_fetch),
        "turn_input_guide_fallback": bool(turn_input_guide),
    }
    # 재귀 한도: iteration 당 loop_guard/call_model/execute ~3 superstep + 여유.
    recursion_limit = max(25, (max_iterations + 2) * 6)
    final = await graph.ainvoke(init, config={"recursion_limit": recursion_limit})
    result = final.get("result")
    _attach_step_meta(final, result)
    apply_sources_footer(result)   # 웹 출처 각주(결정적) — 답변 본문에 남아 다음 턴이 근거를 본다
    apply_guide_links_footer(result)   # 가이드 참고 링크 각주(결정적) — 청크 본문에 있는 URL 만, 정화 뒤에
    return result


async def stream_react_graph(llm, input, messages, all_tools, search_fn, max_iterations: int = 10,
                             target_language: str = DEFAULT_LANGUAGE,
                             pending_draft_force: bool = False,
                             prefetch: Optional[list] = None,
                             auto_fetch: bool = False,
                             turn_input_guide: bool = False):
    """ReAct 그래프를 스트리밍하며 ProgressEvent 들을 흘리고 마지막에 LoopResult 를 yield.

    ``react_loop_events`` 와 동일한 소비 계약(ProgressEvent* 다음 LoopResult 1개)을
    재현한다 — orchestrator ``_stream_react`` 가 교체 소비 가능(1c).

    노드가 ``get_stream_writer()`` 로 방출한 ProgressEvent 는 ``custom`` 모드로,
    최종 ``LoopResult`` 는 ``values`` 모드의 마지막 state 에서 얻는다. 토큰은 즉시
    yield 하고 LoopResult 는 종료 후 마지막에 yield 한다(터미널 노드가 last 라 순서 보존).
    """
    graph = get_react_graph()
    init: ReactState = {
        "llm": llm,
        "input": input,
        "messages": messages,
        "all_tools": all_tools,
        "search_fn": search_fn,
        "max_iterations": max_iterations,
        "target_language": target_language,
        "pending_draft_force": pending_draft_force,
        "prefetch_calls": list(prefetch or []),
        "auto_fetch_enabled": bool(auto_fetch),
        "turn_input_guide_fallback": bool(turn_input_guide),
    }
    recursion_limit = max(25, (max_iterations + 2) * 6)
    final_result = None
    final_state = None
    async for mode, chunk in graph.astream(
        init, stream_mode=["custom", "values"], config={"recursion_limit": recursion_limit}
    ):
        if mode == "custom":
            # 노드가 _emit 한 ProgressEvent — 즉시 흘린다.
            yield chunk
        elif mode == "values":
            if isinstance(chunk, dict):
                final_state = chunk
                if chunk.get("result") is not None:
                    final_result = chunk["result"]
    if final_result is not None:
        _attach_step_meta(final_state or {}, final_result)
        footer = apply_sources_footer(final_result)
        if footer:
            yield ProgressEvent("token", content=footer)   # 화면에도 같은 각주 — 화면 = 히스토리
        guide_footer = apply_guide_links_footer(final_result)   # 가이드 참고 링크(청크 본문의 URL 만) — 같은 방식
        if guide_footer:
            yield ProgressEvent("token", content=guide_footer)
        yield final_result
