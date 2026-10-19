"""표준 ReAct 루프.

run()과 stream() 양쪽에서 소비하는 단일 비동기 제너레이터를 제공한다.

소비 패턴:
    비스트리밍(run()):
        async for event in react_loop_events(...):
            if isinstance(event, LoopResult):
                result = event

    스트리밍(stream()):
        async for event in react_loop_events(...):
            if isinstance(event, ProgressEvent):
                if event.event_type == "token":
                    yield SSE token JSON
                else:
                    yield SSE progress JSON
            elif isinstance(event, LoopResult):
                # 최종 누적 텍스트가 content 에 들어있음 (DB 저장용)

LangGraph 전환 시:
    이 함수 내부 블록이 각 노드와 1:1 대응된다.
    ① 동적 도구 재검색  →  "refresh_tools" 노드
    ② llm.stream()    →  "call_model" 노드
    ③ malformed 처리   →  conditional edge → "force_answer" 노드
    ④ tool 실행        →  "execute_tools" 노드
    ⑤ 루프 종료 조건   →  conditional edge → END
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import AsyncIterator, Awaitable, Callable, List

from agent_arch.core.config import settings
from agent_arch.llm.runaway import stop_on_runaway
from agent_arch.llm.base import BaseLLM, Message, ToolCall
from agent_arch.agents.hitl.policy import requires_approval
from agent_arch.agents.react.types import LoopResult, ProgressEvent, merge_usage
from agent_arch.agents.cancellation import is_cancelled
from agent_arch.agents.react.executor import execute_tools
from agent_arch.agents.react.budget import executed_web
from agent_arch.agents.react.author_first import pick_author_tool
from agent_arch.agents.react.link_policy import TurnLinkPolicy
from agent_arch.agents.tools.tool_search import drain_discovered_tools, reset_discovered_tools
from agent_arch.utils.sanitize import collapse_repeated_runs, sanitize_tool_names
from agent_arch.services.monitoring.error_events import event_extra

logger = logging.getLogger(__name__)


def _build_pending_action(tc: ToolCall, tool_results: list[dict]) -> dict:
    """승인필요 도구의 draft view + (confirm 제외) 인자를 pending_action 으로 구성한다.

    tool_results 에서 해당 tool_call 의 직렬화 결과(JSON 문자열)를 찾아 ``view`` 를 꺼낸다.
    실제 실행은 사람 승인(HITL 그래프 재개) 후에만 일어나므로 여기서는 초안만 담는다.
    """
    draft_view = None
    for tr in tool_results:
        if tr.get("tool_call_id") == tc.id:
            content = tr.get("content")
            if isinstance(content, str) and '"view"' in content:
                try:
                    draft_view = json.loads(content).get("view")
                except (ValueError, TypeError):
                    draft_view = None
            break
    args = {k: v for k, v in tc.arguments.items() if k != "confirm"}
    return {
        "action_id": tc.id,
        "tool_name": tc.name,
        "arguments": args,
        "draft_view": draft_view,
    }

_SAFE_MSG_FALLBACK = "요청하신 정보를 조회하지 못했습니다. 잠시 후 다시 시도해 주세요."
# 도구 탐색 실패(쓸 수 있는 도구가 0개) 전용 — 모델 일시 실패와 구분해 명확히 안내.
_SAFE_MSG_NO_TOOLS = (
    "요청과 관련된 기능을 찾지 못했어요. 다른 표현으로 다시 말씀해 주시거나, "
    "어떤 작업을 원하시는지 조금 더 구체적으로 알려주세요."
)
# 모델이 도구 호출 인자를 만들다 실패(malformed_function_call)하고 텍스트 재시도까지 비었을 때.
# 원인이 모델 출력이라 사용자가 할 수 있는 건 재시도뿐이므로, 짧게 다시 말해달라고 안내한다.
# ⚠️ 이 문구가 없으면 토큰이 하나도 방출되지 않아 화면이 무한 로딩으로 남는다.
_SAFE_MSG_MALFORMED = (
    "요청을 처리하지 못했어요. 같은 내용을 조금 더 구체적으로 다시 말씀해 주시겠어요?"
)
_MAX_RESPONSE_CHARS = 30_000

# 답글 의도 패턴 — 메신저 메시지 작성 시 reply_to_latest 자동 주입 판단용(결정적 보정).
_REPLY_INTENT_RE = re.compile(r"(답글|댓글|대댓글|스레드|thread|reply)", re.IGNORECASE)
_MM_SEND_TOOLS = ("mattermost_post_message", "mattermost_skill_send_dm")

# 리액션 의도 패턴 — react 시 react_to_latest 자동 주입 판단용.
_REACT_INTENT_RE = re.compile(r"(리액션|이모지|이모티콘|반응|체크|emoji|react)", re.IGNORECASE)
_MM_REACT_TOOL = "mattermost_skill_react"


def _wants_thread_reply(text: str) -> bool:
    """사용자 입력이 '답글/댓글/스레드' 의도인지 — reply_to_latest 자동 주입 판단."""
    return bool(text and _REPLY_INTENT_RE.search(text))


def _wants_react_latest(text: str) -> bool:
    """사용자 입력이 '리액션/이모지/체크' 의도인지 — react_to_latest 자동 주입 판단."""
    return bool(text and _REACT_INTENT_RE.search(text))


async def react_loop_events(
    llm: BaseLLM,
    input: str,
    messages: list,
    all_tools: list,
    search_fn: Callable[[str], Awaitable[list]],
    max_iterations: int = 10,
) -> AsyncIterator[ProgressEvent | LoopResult]:
    """표준 ReAct 루프를 실행하며 진행 이벤트와 최종 결과를 yield한다.

    messages와 all_tools는 in-place로 수정된다.
    항상 마지막에 LoopResult 하나를 yield하고 종료한다.

    텍스트 답변 회차에서는 llm.stream() 의 text 청크를 즉시 ProgressEvent("token") 으로
    yield 하여 사용자에게 토큰 단위 스트리밍 UX 를 제공한다.
    도구 호출 회차에서는 토큰을 흘리지 않고 tool_calls 를 누적해 도구 실행으로 진행한다.

    Args:
        llm: LLM 인스턴스 (stream + bind_tools 사용)
        input: 사용자 입력 텍스트
        messages: 대화 메시지 리스트 (system + history + user, in-place 수정)
        all_tools: 사용 가능한 도구 리스트 (in-place 확장될 수 있음)
        search_fn: 동적 도구 재검색 콜백 (orchestrator._search_dynamic_tools)
        max_iterations: 최대 반복 횟수

    Yields:
        ProgressEvent: 루프 진행 이벤트 (thinking/searching/tool_start/tool_end/token)
        LoopResult: 최종 결과 (항상 마지막으로 yield됨)
    """
    # ── 상태 초기화 ──────────────────────────────────────────────────────────
    token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    collected_tool_calls: list[dict] = []
    web_rounds = 0  # 웹 호출을 실행한 스텝 수(react/budget 단계 상한)
    collected_tool_results: list[dict] = []
    known_tool_names: set = {t.name for t in all_tools}
    used_tool_names: list[str] = []
    llm_invoke_count = 0
    used_tools = False
    _forced_any_mode = False
    _forced_any_done = False

    # 발견 대기열을 이 요청 컨텍스트의 새 리스트로 초기화 — 도메인 탐색 도구가
    # gather 자식 Task 에서 적재해도 이 리스트(동일 객체)로 부모가 drain 가능.
    reset_discovered_tools()

    # 링크 허용 정책 — 가이드 검색이 돈 턴에만 켜져, 근거 없는 주소(지어낸 가이드 문서 링크)를
    # 답변에서 걷어낸다. 근거 = 도구 결과 + 시스템 프롬프트의 고정 링크 + 사용자 입력
    # (`react/link_policy.py`). 그래프 경로(`graphs/react_graph.py`)와 같은 정책이다.
    policy = TurnLinkPolicy()
    for _m in messages:
        if getattr(_m, "role", None) == "system":
            policy.allow_text(getattr(_m, "content", None))
    policy.allow_text(input)

    def _stopped_result() -> LoopResult:
        """사용자 중단(stop) 시 yield 할 결과.

        중단 턴은 저장하지 않으므로 content 는 비워 둔다(부분텍스트 누적 불필요).
        finish_reason="stopped" 로 orchestrator 가 저장/HITL/eval 을 스킵하게 한다.
        """
        return LoopResult(
            content="",
            tool_calls=collected_tool_calls,
            tool_results=collected_tool_results,
            llm_invoke_count=llm_invoke_count,
            token_usage=token_usage,
            finish_reason="stopped",
        )

    # ── ReAct 루프 ────────────────────────────────────────────────────────────
    for iteration in range(max_iterations):

        # 사용자 중단 (체크포인트 a — iteration 경계): 다음 LLM 호출 전에 멈춘다.
        if is_cancelled():
            yield _stopped_result()
            return

        # (A안) 자동 사전검색 재검색 제거 — 매 iteration 원본 질의로 top-k 를 다시
        # 뽑아 무관 도메인 도구(hr/mail/drive …)를 주입해 모델이 헤매던 원인.
        # 추가 도구 발견은 오직 LLM 이 도메인 탐색 도구(find_<도메인>_tools)를
        # 호출 → 아래 drain_discovered_tools 머지 경로로만 이뤄진다.

        if iteration > 0:
            yield ProgressEvent("thinking", content="심층 분석중...")

        # ANY 재시도 요청이 있으면 이번 iteration에서 ANY 모드 사용
        if _forced_any_mode:
            tool_mode = "ANY"
            _forced_any_mode = False
            _forced_any_done = True
        else:
            tool_mode = "AUTO"

        # ② LLM stream 소비
        text_buf: list[str] = []
        iter_tool_calls: List[ToolCall] = []
        finish_reason = "stop"
        usage = None

        t0 = time.monotonic()
        pinned = pick_author_tool(input, known_tool_names) if iteration == 0 else None
        stream_options = {"tool_mode": "ANY", "allowed_tools": [pinned]} if pinned else {"tool_mode": tool_mode}
        async for event in stop_on_runaway(llm.stream(messages, **stream_options)):
            # 사용자 중단 (체크포인트 b — 토큰 단위): break 시 async gen aclose() 가
            # GeneratorExit 를 주입해 업스트림 LLM 스트림(httpx)이 정리된다.
            if is_cancelled():
                break
            if event.type == "text":
                if event.content:
                    text_buf.append(event.content)
                    yield ProgressEvent("token", content=event.content)
            elif event.type == "tool_calls":
                iter_tool_calls = event.tool_calls or []
            elif event.type == "done":
                finish_reason = event.finish_reason or "stop"
                usage = event.usage

        # 토큰 스트림 도중 중단됐으면 stopped 로 종결(이번 invoke 는 카운트하지 않음).
        if is_cancelled():
            yield _stopped_result()
            return

        llm_invoke_count += 1
        merge_usage(token_usage, usage)
        logger.debug(
            "[react_loop] iter=%d tool_mode=%s elapsed=%.2fs "
            "finish_reason=%s tool_calls=%s",
            iteration, tool_mode, time.monotonic() - t0, finish_reason,
            [tc.name for tc in iter_tool_calls],
        )

        # ③ malformed_function_call 처리 → tool_mode=NONE 으로 재 스트림
        if finish_reason == "malformed_function_call":
            logger.warning("[react_loop] MALFORMED_FUNCTION_CALL → tool_mode=NONE 재호출")
            await llm.bind_tools([])
            retry_text: list[str] = []
            retry_finish = "stop"
            retry_usage = None
            try:
                async for event in stop_on_runaway(llm.stream(messages, tool_mode="NONE")):
                    if event.type == "text":
                        if event.content:
                            retry_text.append(event.content)
                            yield ProgressEvent("token", content=event.content)
                    elif event.type == "done":
                        retry_finish = event.finish_reason or "stop"
                        retry_usage = event.usage
            except Exception:
                # 예외를 밖으로 흘리면 스트림이 종료 이벤트 없이 끊겨 화면이 무한 로딩으로 남는다.
                logger.warning("[react_loop] malformed 재시도 실패 — 폴백 문구로 종결", exc_info=True,
                               extra=event_extra("exception"))
                retry_finish = "error"
            llm_invoke_count += 1
            merge_usage(token_usage, retry_usage)
            final_content = sanitize_tool_names("".join(retry_text), link_permits=policy.permits)
            # 진단: 다음 발생 때 (a)length=토큰 소진 (b)stop+len=0=빈 응답 (c)error=예외 를 구별.
            logger.info(
                "[react_loop] malformed 재시도 결과 finish=%s len=%d",
                retry_finish, len(final_content),
            )
            if not final_content.strip():
                # ⚠️ 재시도가 비면 토큰이 하나도 안 나가고 content 도 빈 문자열이라 화면이
                # 무한 로딩으로 남는다(실측 2026-08-18). 반드시 안내를 방출한다.
                final_content = _SAFE_MSG_MALFORMED
                yield ProgressEvent("token", content=final_content)
            yield LoopResult(
                content=final_content,
                tool_calls=collected_tool_calls,
                tool_results=collected_tool_results,
                llm_invoke_count=llm_invoke_count,
                token_usage=token_usage,
                finish_reason=retry_finish,
            )
            return

        # ⑤ 루프 종료 조건 — tool_calls 없음 → 최종 답변 분기
        if finish_reason != "tool_calls" or not iter_tool_calls:
            # 모델 반복 루프 잔해(같은 문자 120자+ 런)는 trim **앞에서** 접는다 — 그래야
            # 정상 본문 길이로 trim 여부가 결정돼 표 본문 등이 살아남는다(react_graph 와 패리티).
            final_content = collapse_repeated_runs("".join(text_buf))

            # 비정상적으로 큰 응답 → 이미 사용자에게 흘렸으므로 trim + 안내문구
            #   (정책: 재 invoke 하면 사용자에게 중복 출력 → trim 채택)
            if used_tools and len(final_content) > _MAX_RESPONSE_CHARS:
                logger.warning(
                    "[react_loop] oversized response (%d chars) → trim",
                    len(final_content),
                )
                final_content = (
                    final_content[:_MAX_RESPONSE_CHARS]
                    + "\n\n(응답이 너무 길어 일부만 표시됩니다.)"
                )

            # 실제 답변 텍스트가 있을 때만 그대로 종료한다.
            # (빈 응답을 used_tools 라는 이유로 무음 종료하던 버그 수정 — 빈 응답은
            #  스트리밍된 토큰이 0개라 재시도해도 중복 출력 위험이 없다.)
            if final_content:
                yield LoopResult(
                    content=sanitize_tool_names(final_content, link_permits=policy.permits),
                    tool_calls=collected_tool_calls,
                    tool_results=collected_tool_results,
                    llm_invoke_count=llm_invoke_count,
                    token_usage=token_usage,
                    finish_reason=finish_reason,
                )
                return

            # 빈 응답 + 도구 있음 + 아직 ANY 재시도 안 함 → ANY 모드로 강제 재시도.
            # (used_tools 든 아니든 동일 처리. ANY 는 도구 바인딩을 유지해 구조화된 tool_call 을 끌어내 멀티스텝을 잇는다.)
            if all_tools and not _forced_any_done:
                logger.warning(
                    "[react_loop] no tool calls + empty content, tools available → retry ANY"
                )
                _forced_any_mode = True
                await llm.bind_tools(all_tools)
                continue

            # ANY 재시도 후에도 실패 or 도구 없음 → 안전 메시지 반환
            # token 으로도 흘려 사용자가 빈 화면을 보지 않도록 한다 (guide_loop 와 일관).
            # 쓸 수 있는 도구가 0개면 "탐색 실패"로 명확히 — 모델 일시 실패와 구분.
            no_tools = not all_tools
            fallback_msg = _SAFE_MSG_NO_TOOLS if no_tools else _SAFE_MSG_FALLBACK
            logger.warning(
                "[react_loop] fallback: %s",
                "도구 탐색 실패(0개)" if no_tools else "ANY 재시도 후에도 도구호출 없음",
            )
            yield ProgressEvent("token", content=fallback_msg)
            yield LoopResult(
                content=fallback_msg,
                tool_calls=collected_tool_calls,
                tool_results=collected_tool_results,
                llm_invoke_count=llm_invoke_count,
                token_usage=token_usage,
                finish_reason="fallback",
            )
            return

        # ④ tool 실행
        used_tools = True
        for tc in iter_tool_calls:
            if tc.name == "mattermost_skill_find_posts_by_author":
                tc.arguments = {**(tc.arguments or {}), "search_intent": input}

        # ── 답글 의도 자동 주입(3번): '답글/댓글/스레드' 요청 → reply_to_latest=True ──
        # 모델이 reply_to_latest 를 안 챙겨 새 글로 올라가는 걸 코드로 결정적 보정한다.
        # 명시적 root_id/reply_to_post_id/reply_to_latest 가 이미 있으면 건드리지 않는다.
        if _wants_thread_reply(input):
            for tc in iter_tool_calls:
                if tc.name in _MM_SEND_TOOLS:
                    a = tc.arguments or {}
                    if not (a.get("root_id") or a.get("reply_to_post_id") or a.get("reply_to_latest")):
                        tc.arguments = {**a, "reply_to_latest": True}

        # ── 리액션 의도 자동 주입: '이모지/리액션/체크' 요청 → react_to_latest=True ──
        # post_id 없이 호출되면 채널 최근 글에 리액션하도록(channel_id 는 모델이 제공).
        if _wants_react_latest(input):
            for tc in iter_tool_calls:
                if tc.name == _MM_REACT_TOOL:
                    a = tc.arguments or {}
                    if not a.get("post_id") and not a.get("react_to_latest"):
                        tc.arguments = {**a, "react_to_latest": True}

        # ── HITL: 승인필요 액션은 confirm=False 강제(초안만) ──────────────────────
        # LLM 이 confirm=True 로 조기 실행하는 것을 코드로 차단한다(프롬프트 유도가
        # 아니라 코드 강제). 한 턴에 계획된 승인필요 도구를 '모두' 기억해 두고, 정상
        # 실행 흐름을 그대로 태운 뒤 '유효한 draft view'가 나온 것들만 큐로 묶어 일시
        # 정지한다(아래). view 없는 항목(미발견/에러)은 큐에서 제외 → 1건도 없으면
        # 멈추지 않고 루프를 이어 LLM 이 복구하게 둔다.
        # 기능 플래그와 무관하게 항상 강제한다 — 플래그가 꺼진 배포에서 비가역 액션이 승인 없이
        # 나가는 구멍이 있었다. 실제 실행은 /resume(HITL 그래프)에서만 일어난다.
        hitl_pending_tcs = []
        for tc in iter_tool_calls:
            if requires_approval(tc.name):
                tc.arguments = {**tc.arguments, "confirm": False}
                hitl_pending_tcs.append(tc)

        messages.append(Message(
            role="assistant",
            content="",
            tool_calls=[tc.model_dump() for tc in iter_tool_calls],
        ))

        # 사용자 중단 (체크포인트 c — 도구 실행 직전): 메일 발송 등 부수효과 도구가
        # 돌기 전에 멈춰 외부 호출을 막는다.
        if is_cancelled():
            yield _stopped_result()
            return

        exec_results = await execute_tools(iter_tool_calls, all_tools,
                                           prior_calls=collected_tool_calls, prior_results=collected_tool_results,
                                           web_rounds=web_rounds)
        if executed_web(exec_results):
            web_rounds += 1
        collected_tool_calls.extend(tc.model_dump() for tc in iter_tool_calls)

        # tool_search가 발견·게이트 통과시킨 도구를 머지 — 다음 iteration의
        # bind_tools(아래)부터 네이티브 호출 가능해진다.
        discovered = drain_discovered_tools()
        newly_discovered = [t for t in discovered if t.name not in known_tool_names]
        if newly_discovered:
            all_tools.extend(newly_discovered)
            known_tool_names.update(t.name for t in newly_discovered)
            logger.info(
                "[react_loop] tool_search 발견 도구 바인딩: %s",
                [t.name for t in newly_discovered],
            )

        for tc, result_content in exec_results:
            used_tool_names.append(tc.name)
            messages.append(Message(
                role="tool",
                content=result_content,
                name=tc.name,
                tool_call_id=tc.id,
            ))
            collected_tool_results.append({
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": result_content,
            })

        # 정책 스냅샷을 화면 쪽(orchestrator 의 StreamSanitizer)에 먼저 알린다.
        policy.observe_results(collected_tool_results)
        if policy.active:
            yield ProgressEvent("link_policy", content=policy.dumps())

        await llm.bind_tools(all_tools)

        # ── HITL 일시정지 판정: 승인필요 도구들이 '유효한 draft view'를 만들었을 때만 ──
        # 멈춘다. view 없는 항목(도구 미발견/실행 에러)은 큐에서 제외 → 1건도 없으면
        # 멈추지 않고 루프를 이어 LLM 이 복구하게 둔다(일반 NOT FOUND 복구 흐름과 동일).
        # 여러 승인필요 액션이 한 턴에 계획되면 모두 큐에 담아 순차 승인/거절하게 한다.
        if hitl_pending_tcs:
            queue = [
                pa for tc in hitl_pending_tcs
                if (pa := _build_pending_action(tc, collected_tool_results)).get("draft_view")
            ]
            if queue:
                logger.info(
                    "[react_loop] HITL 일시정지 — 승인필요 액션 %d건 초안: %s",
                    len(queue), [p["tool_name"] for p in queue],
                )
                yield LoopResult(
                    content=sanitize_tool_names("".join(text_buf), link_permits=policy.permits),
                    tool_calls=collected_tool_calls,
                    tool_results=collected_tool_results,
                    llm_invoke_count=llm_invoke_count,
                    token_usage=token_usage,
                    finish_reason="action_pending",
                    pending_action=queue[0],   # 하위호환: 단수 = 큐[0]
                    pending_queue=queue,
                )
                return
            logger.info(
                "[react_loop] HITL: 승인필요 도구 draft view 없음(미발견/에러) → 루프 계속(복구)",
            )

    # ── max_iterations 소진 → 강제 최종 답변 ──────────────────────────────────
    logger.warning("[react_loop] max_iterations=%d 소진, tool_mode=NONE 강제 스트리밍", max_iterations)
    await llm.bind_tools([])
    forced_text: list[str] = []
    forced_finish = "stop"
    forced_usage = None
    async for event in stop_on_runaway(llm.stream(messages, tool_mode="NONE")):
        if event.type == "text":
            if event.content:
                forced_text.append(event.content)
                yield ProgressEvent("token", content=event.content)
        elif event.type == "done":
            forced_finish = event.finish_reason or "stop"
            forced_usage = event.usage
    llm_invoke_count += 1
    merge_usage(token_usage, forced_usage)
    if forced_text:
        # 강제 NONE 스트림이 텍스트를 만들었다면 토큰은 이미 사용자에게 흘러갔다.
        final_content = "".join(forced_text)
    else:
        # 강제 NONE 도 빈 응답 → fallback 문구를 token 으로 명시 전달 후 LoopResult 마무리.
        final_content = "최대 반복 횟수를 초과하여 답변을 생성하지 못했습니다."
        yield ProgressEvent("token", content=final_content)
    yield LoopResult(
        content=sanitize_tool_names(final_content, link_permits=policy.permits),
        tool_calls=collected_tool_calls,
        tool_results=collected_tool_results,
        llm_invoke_count=llm_invoke_count,
        token_usage=token_usage,
        finish_reason=forced_finish,
    )
