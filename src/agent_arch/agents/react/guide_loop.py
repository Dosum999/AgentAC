"""가이드 모드 ReAct 루프.

_run_guide_mode()와 _stream_guide_mode()를 단일 비동기 제너레이터로 통합한다.

guide 모드 특성:
    - tool_mode=AUTO: 검색 여부는 LLM 자율 판단. 가이드 질문에서의 검색
      보장은 시스템 프롬프트(prompts/system.py L249)에 위임한다.
    - guide_hits 카운터: 평가/로깅용으로 LoopResult에 그대로 노출.
    - max_iterations는 최대 3으로 제한된다 (가이드 KB는 depth가 얕음)
    - 빈 결과·도메인 밖·인사 같은 상황은 시스템 프롬프트의 안내 지시
      (L353-383)에 따라 LLM 이 직접 응답하도록 위임. 하드코딩 fallback 은
      LLM 이 한 글자도 응답을 못 만든 극단 케이스에만 발동한다.
"""
from __future__ import annotations

import logging
import time
from typing import AsyncIterator, List, Optional

from agent_arch.llm.runaway import stop_on_runaway
from agent_arch.llm.base import BaseLLM, Message, ToolCall
from agent_arch.agents.react.types import LoopResult, ProgressEvent, merge_usage
from agent_arch.agents.react.executor import find_tool, serialize_result
from agent_arch.agents.react.guide_links import guide_links_footer
from agent_arch.agents.react.link_policy import TurnLinkPolicy
from agent_arch.agents.cancellation import is_cancelled
from agent_arch.utils.sanitize import GUIDE_AGENT_HINT, sanitize_tool_names

logger = logging.getLogger(__name__)

_GUIDE_NO_RESPONSE_MSG = (
    "찾으시는 내용을 가이드에서 바로 안내드리기 어려웠어요. "
    "어떤 서비스의 어떤 기능이 궁금하신지 조금 더 구체적으로 알려주시면 "
    "더 정확히 도와드릴게요."
)
_GUIDE_MAX_ITERATIONS = 3


async def guide_loop_events(
    llm: BaseLLM,
    input: str,
    messages: list,
    guide_tool,
    max_iterations: int = _GUIDE_MAX_ITERATIONS,
    categories: Optional[List[str]] = None,
) -> AsyncIterator[ProgressEvent | LoopResult]:
    """가이드 모드 ReAct 루프를 실행하며 진행 이벤트와 최종 결과를 yield한다.

    표준 ReAct 루프(react_loop_events)와 동일한 stream 기반 패턴을 따른다.
    최종 답변 회차의 텍스트 토큰을 ProgressEvent("token") 으로 즉시 yield 하여
    토큰 스트리밍 UX 를 제공한다.

    Args:
        llm: LLM 인스턴스 (stream + bind_tools 사용)
        input: 사용자 입력 텍스트
        messages: 대화 메시지 리스트 (in-place 수정)
        guide_tool: 가이드 KB 검색 도구 (GuideKBTool 인스턴스)
        max_iterations: 최대 반복 횟수 (기본값: 3)
        categories: 검색 대상 카테고리 화이트리스트. ``None`` 이면 전체 검색.
            LLM tool_calls 인자와 충돌하지 않도록 ``_categories`` underscore prefix
            로 ``tool.execute`` 에 주입된다 (GuideKBTool 이 pop 으로 분리).

    Yields:
        ProgressEvent: 루프 진행 이벤트 (thinking/searching/token)
        LoopResult: 최종 결과 (항상 마지막으로 yield됨, guide_hits 필드 포함)
    """
    all_tools = [guide_tool]
    await llm.bind_tools(all_tools)

    # 링크 허용 정책 — 가이드 검색이 돌면 켜지고, 그때부터 "모델이 실제로 본 텍스트에 있던
    # 주소"만 답변에 남는다(지어낸 가이드 문서 링크 차단, `link_policy` 참조). 시스템 프롬프트가
    # 코드로 심는 고정 링크(다운로드 센터 등)와 사용자가 적어 보낸 주소도 근거로 인정한다.
    policy = TurnLinkPolicy()
    for m in messages:
        if getattr(m, "role", None) == "system":
            policy.allow_text(getattr(m, "content", None))
    policy.allow_text(input)

    token_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    collected_tool_calls: list[dict] = []
    collected_tool_results: list[dict] = []
    llm_invoke_count = 0
    guide_hits = 0
    # 마지막 KB 검색의 기권(abstain) 신호. 가이드로 답할 수 없는 질문으로 판정되면
    # 상위(orchestrator)가 suggested_mode="agent" 로 승격해 "AI 에이전트 모드로
    # 전환할까요?" 제안을 프론트에 내려보낸다. 여러 번 검색하면 마지막 값을 채택한다.
    last_abstain = False
    last_abstain_reason: Optional[str] = None
    text_buf: list[str] = []
    finish_reason = "stop"

    def _stopped_result() -> LoopResult:
        """사용자 중단(stop) 시 yield 할 결과. 중단 턴은 저장하지 않으므로 content 비움."""
        return LoopResult(
            content="",
            tool_calls=collected_tool_calls,
            tool_results=collected_tool_results,
            llm_invoke_count=llm_invoke_count,
            token_usage=token_usage,
            finish_reason="stopped",
            guide_hits=guide_hits,
        )

    yield ProgressEvent("thinking", content="질의를 이해하는 중...")

    for iteration in range(min(max_iterations, _GUIDE_MAX_ITERATIONS)):
        # 사용자 중단 (체크포인트 a — iteration 경계): 다음 LLM 호출 전에 멈춘다.
        if is_cancelled():
            yield _stopped_result()
            return

        # 진행 상황 이벤트: 첫 번째는 검색, 이후는 재시도 안내
        if iteration == 0:
            yield ProgressEvent("searching", content="관련 가이드를 검색하는 중...")
        else:
            yield ProgressEvent("thinking", content="검색어를 바꾸어 다시 확인하는 중...")

        # 검색 여부는 LLM 자율 판단 — trivial 질문(인사 등)은 검색 없이 응답,
        # 가이드 질문은 시스템 프롬프트의 "답변 전 가이드 확인" 지시가 검색 유도.
        tool_mode = "AUTO"

        iter_text: list[str] = []
        iter_tool_calls: List[ToolCall] = []
        iter_finish = "stop"
        iter_usage = None

        t0 = time.monotonic()
        async for event in stop_on_runaway(llm.stream(messages, tool_mode=tool_mode)):
            # 사용자 중단 (체크포인트 b — 토큰 단위): 업스트림 스트림을 정리하며 빠져나온다.
            if is_cancelled():
                break
            if event.type == "text":
                if event.content:
                    iter_text.append(event.content)
                    yield ProgressEvent("token", content=event.content)
            elif event.type == "tool_calls":
                iter_tool_calls = event.tool_calls or []
            elif event.type == "done":
                iter_finish = event.finish_reason or "stop"
                iter_usage = event.usage

        llm_invoke_count += 1
        merge_usage(token_usage, iter_usage)
        finish_reason = iter_finish
        logger.debug(
            "[guide_loop] iter=%d tool_mode=%s elapsed=%.2fs finish_reason=%s",
            iteration, tool_mode, time.monotonic() - t0, iter_finish,
        )

        # malformed → NONE 모드 재 스트림
        if iter_finish == "malformed_function_call":
            logger.warning("[guide_loop] MALFORMED_FUNCTION_CALL → tool_mode=NONE 재호출")
            await llm.bind_tools([])
            retry_text: list[str] = []
            retry_finish = "stop"
            retry_usage = None
            async for event in stop_on_runaway(llm.stream(messages, tool_mode="NONE")):
                if is_cancelled():
                    break
                if event.type == "text":
                    if event.content:
                        retry_text.append(event.content)
                        yield ProgressEvent("token", content=event.content)
                elif event.type == "done":
                    retry_finish = event.finish_reason or "stop"
                    retry_usage = event.usage
            llm_invoke_count += 1
            merge_usage(token_usage, retry_usage)
            text_buf = retry_text
            finish_reason = retry_finish
            break

        # tool_calls 없음 → 최종 답변
        if iter_finish != "tool_calls" or not iter_tool_calls:
            text_buf = iter_text
            break

        # 어시스턴트 tool_call 메시지 추가
        messages.append(Message(
            role="assistant",
            content="",
            tool_calls=[tc.model_dump() for tc in iter_tool_calls],
        ))
        collected_tool_calls.extend(tc.model_dump() for tc in iter_tool_calls)

        # 사용자 중단 (체크포인트 c — 도구 실행 직전): KB 검색 전에 멈춘다.
        if is_cancelled():
            yield _stopped_result()
            return

        # 도구 순차 실행 (guide 모드는 단일 도구만 사용)
        for tc in iter_tool_calls:
            tool = find_tool(tc.name, all_tools)
            if tool is None:
                result_content = f"[ERROR] 도구를 찾을 수 없습니다: {tc.name}"
            else:
                result = await tool.execute(**tc.arguments, _categories=categories)
                if result.success:
                    meta = result.metadata or {}
                    if isinstance(result.output, list) and result.output:
                        guide_hits += len(result.output)
                    # 검색이 기권했는지 보존 — abstain gate 가 "가이드 밖 질문" 으로 판정한 신호.
                    last_abstain = bool(meta.get("abstain", False))
                    last_abstain_reason = meta.get("abstain_reason")
                result_content = serialize_result(result)
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

        # 정책 스냅샷을 화면 쪽(orchestrator 의 StreamSanitizer)에 먼저 알린다 — 도구가 돌아야
        # 정책이 켜지므로 최종 답변 토큰보다 항상 먼저 나간다.
        policy.observe_results(collected_tool_results)
        if policy.active:
            yield ProgressEvent("link_policy", content=policy.dumps())

    # 루프 종료 후 text_buf 가 비어있고 마지막이 tool_calls 였으면 NONE 강제 스트림
    if not text_buf and finish_reason == "tool_calls":
        await llm.bind_tools([])
        forced_text: list[str] = []
        forced_finish = "stop"
        forced_usage = None
        async for event in stop_on_runaway(llm.stream(messages, tool_mode="NONE")):
            if is_cancelled():
                break
            if event.type == "text":
                if event.content:
                    forced_text.append(event.content)
                    yield ProgressEvent("token", content=event.content)
            elif event.type == "done":
                forced_finish = event.finish_reason or "stop"
                forced_usage = event.usage
        llm_invoke_count += 1
        merge_usage(token_usage, forced_usage)
        text_buf = forced_text
        finish_reason = forced_finish

    # 비상 fallback: LLM 이 한 글자도 응답을 못 만든 극단 케이스만 처리한다.
    # 빈 KB 결과/도메인 밖/인사 같은 상황은 시스템 프롬프트의 안내 지시에 따라
    # LLM 이 직접 자연스러운 응답을 만들도록 위임 (중복 사과 문구 방지).
    final_content = "".join(text_buf)
    # 가이드 LLM이 "실제 데이터 조회·작업이 필요해 가이드로는 답할 수 없는 요청"이라 판단해
    # 마커를 붙였으면, abstain 신호로 승격(→ 상위에서 suggested_mode="agent")하고 마커는
    # 저장/표시 텍스트에서 제거한다. 스트리밍 표시 제거는 StreamSanitizer 가 담당한다.
    if GUIDE_AGENT_HINT in final_content:
        last_abstain = True
        last_abstain_reason = last_abstain_reason or "needs_agent"
        final_content = final_content.replace(GUIDE_AGENT_HINT, "").rstrip()
    # 사용자 중단: 토큰 스트림 break 로 빠져나온 경우 fallback 문구 없이 stopped 로 종결.
    if is_cancelled():
        yield _stopped_result()
        return
    if not final_content:
        yield ProgressEvent("token", content=_GUIDE_NO_RESPONSE_MSG)
        final_content = _GUIDE_NO_RESPONSE_MSG
    # 도구명 마스킹 — react 루프/그래프와 동일 계약을 맞춘다. 가이드 모드는 도구가 KB 검색
    # 뿐이라 실위험은 낮지만, 정화 없는 경로를 하나 남겨두면 나중에 그 경로로 샌다.
    # **반드시 위 GUIDE_AGENT_HINT 소비 이후**여야 한다 — sanitize 가 마커를 제거하므로
    # 앞에 두면 abstain(→ suggested_mode="agent") 승격이 조용히 죽는다.
    final_content = sanitize_tool_names(final_content, link_permits=policy.permits)
    # 가이드 참고 링크 각주 — 정화 **뒤**에, 이번 턴 검색 결과 청크 본문에 있는 URL 로만(코드 소유 섹션).
    # 화면에도 같은 각주를 토큰으로 흘린다(웹 출처 각주와 같은 방식) — 화면 = 히스토리.
    guide_footer = guide_links_footer(final_content, collected_tool_results)
    if guide_footer:
        yield ProgressEvent("token", content=guide_footer)
        final_content = final_content.rstrip() + guide_footer

    yield LoopResult(
        content=final_content,
        tool_calls=collected_tool_calls,
        tool_results=collected_tool_results,
        llm_invoke_count=llm_invoke_count,
        token_usage=token_usage,
        finish_reason=finish_reason,
        guide_hits=guide_hits,
        abstain=last_abstain,
        abstain_reason=last_abstain_reason,
    )
