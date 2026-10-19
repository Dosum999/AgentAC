"""도구 검색 및 실행 유틸리티.

LLM의 tool_call 요청을 실제 BaseTool 인스턴스와 연결하고,
병렬 실행 후 LLM 메시지 형식으로 직렬화한다.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from agent_arch.agents.tools.dynamic_api_tool import DynamicAPITool
from agent_arch.services.api_registry.loader import tool_loader
from agent_arch.llm.pii.audit import redact_for_log
from agent_arch.llm.pii.engine import find_tokens

logger = logging.getLogger(__name__)


def find_tool(name: str, all_tools: list):
    """이름으로 도구를 검색한다 (static + dynamic 통합).

    all_tools에서 찾지 못하면 llm_tools.json 캐시에서 fallback 조회하여
    DynamicAPITool을 즉석 생성한다. Qdrant top_k 검색 누락으로 인한
    "도구를 찾을 수 없습니다" 에러를 방지한다.

    Args:
        name: 검색할 도구 이름
        all_tools: 전체 도구 리스트

    Returns:
        BaseTool 인스턴스. 없으면 None
    """
    tool = next((t for t in all_tools if t.name == name), None)
    if tool is not None:
        return tool
    # fallback: 캐시된 전체 툴 목록에서 이름으로 직접 조회
    tool_def = tool_loader.get_by_name(name)
    if tool_def is not None:
        logger.info("[find_tool] fallback로 도구 생성: %s", name)
        return DynamicAPITool(tool_def)
    return None


def truncate_result(data: Any, max_items: int = 20) -> Any:
    """Tool 결과 데이터를 레코드 단위로 잘라 LLM context 크기를 제한한다.

    리스트가 max_items를 초과하면 앞에서 max_items개만 남기고
    잘린 사실을 메타 필드로 표기한다. JSON 구조는 유지된다.

    Args:
        data: 자를 데이터 (list 또는 dict)
        max_items: 최대 아이템 개수 (기본값: 20)

    Returns:
        잘린 데이터 또는 원본 데이터
    """
    if isinstance(data, list):
        if len(data) > max_items:
            return {"items": data[:max_items], "_truncated": f"{len(data)}건 중 {max_items}건만 표시"}
        return data
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(v, list) and len(v) > max_items:
                result[k] = v[:max_items]
                result[f"_truncated_{k}"] = f"{len(v)}건 중 {max_items}건만 표시"
            else:
                result[k] = v
        return result
    return data


def serialize_result(result) -> str:
    """ToolResult를 LLM 메시지용 문자열로 직렬화한다.

    성공 시 JSON 직렬화, 실패 시 [ERROR] 접두어 에러 메시지를 반환한다.

    Args:
        result: ToolResult 인스턴스

    Returns:
        LLM role="tool" 메시지에 삽입할 문자열
    """
    if result.success:
        output = truncate_result(result.output)
        return json.dumps(output, ensure_ascii=False)
    # 실패에 실린 데이터도 넘긴다(동명이인 후보 등). 에러 문구만 주고 "candidates 를 보여주고
    # 되물어라" 고 하면 모델은 목록을 **지어낸다**(2026-10-01 운영: 없는 부서 셋을 만들었다).
    # 접두어는 그대로 둔다 — `[ERROR]` 로 실패를 판정하는 곳(verify·budget·conversation)이 여럿이다.
    data = _failure_data(result.output)
    if data:
        return f"[ERROR] {result.error}\n{json.dumps(truncate_result(data), ensure_ascii=False)}"
    return f"[ERROR] {result.error}"


# 에러 문구와 같은 말을 되풀이할 뿐인 키 — 이것만 있으면 덧붙이지 않는다.
_ECHO_KEYS = frozenset({"error", "message", "success"})


def _failure_data(output):
    """실패 결과에서 모델에게 보여줄 **데이터**만 고른다(에러 문구의 반복은 뺀다)."""
    if isinstance(output, dict):
        rest = {k: v for k, v in output.items() if k not in _ECHO_KEYS}
        return rest or None
    return output or None


def unresolved_placeholder_error(tool_name: str, arguments: Any) -> Optional[str]:
    """인자에 가명화 자리표시 토큰이 남아 있으면 모델에 돌려줄 오류 문자열, 없으면 None.

    executor 와 HITL resume(`hitl/graph._execute_one`)이 같은 검사를 쓴다.
    """
    leftover = find_tokens(arguments)
    if not leftover:
        return None
    shown = ", ".join(dict.fromkeys(leftover))[:200]
    return (
        f"[ERROR] `{tool_name}` 인자에 자리표시 토큰이 남아 있습니다({shown}). "
        "토큰은 입력에 있던 꼴 그대로(대괄호·글자 변경 없이) 옮겨야 복원됩니다. "
        "입력에 없는 토큰을 지어내지 말고, 대상이 불분명하면 사용자에게 확인하세요."
    )


def _schedule_harvest(content: str) -> None:
    """가명화 어휘 — 도구 결과의 이름·이메일을 대화 범위에 남긴다(백그라운드, 가드 모드가 켜져 있을 때만)."""
    try:
        from agent_arch.llm.pii.policy import global_mode
        if global_mode() == "off":
            return
        from agent_arch.agents.context import get_conversation_id, get_user_context
        from agent_arch.llm.pii.sources import harvest_tool_result
        company = (get_user_context() or {}).get("company")
        asyncio.create_task(harvest_tool_result(get_conversation_id(), content, company))
    except Exception:  # noqa: BLE001
        logger.debug("[executor] 어휘 수확 예약 실패(무시)", exc_info=True)


async def execute_tools(
    tool_calls: list,
    all_tools: list,
    *,
    prior_calls: list | None = None,
    prior_results: list | None = None,
    web_rounds: int = 0,
) -> list[tuple]:
    """tool_call 목록을 병렬 실행하고 (tc, result_content) 튜플 리스트를 반환한다.

    asyncio.gather로 모든 tool_call을 동시에 실행한다.
    도구를 찾지 못하거나 실행 중 예외가 발생하면 [ERROR] 문자열을 반환하며
    다른 도구의 실행에는 영향을 주지 않는다.

    턴 예산(react/budget): 이번 턴에 이미 실행한 웹 호출(prior_calls/prior_results)과 이 배치를 함께 보고, 같은
    인자의 웹 호출은 실행하지 않고 "앞선 결과를 보라"를, 턴당 웹 호출 상한을 넘는 호출은 "지금 근거로 답하라"를
    돌려준다. 그래프와 옛 루프가 같은 실행기를 쓰므로 두 경로가 같은 가드를 지난다.

    Args:
        tool_calls: ToolCall 리스트 (LLMResponse.tool_calls)
        all_tools: 전체 도구 리스트 (static + dynamic)
        prior_calls: 이번 턴 앞선 iteration 의 tool_call dict 들(collected_tool_calls)
        prior_results: 그 결과들(collected_tool_results)
        web_rounds: 이번 턴 앞선 iteration 중 웹 호출을 실행한 스텝 수(단계 상한)

    Returns:
        (ToolCall, result_content_str) 튜플 리스트 (순서 보장)
    """
    from agent_arch.agents.context import withheld_tools
    from agent_arch.agents.react.budget import WebGuard

    guard = WebGuard(prior_calls, prior_results, rounds=web_rounds)
    blocked: dict[int, str] = {}
    for i, tc in enumerate(tool_calls):
        err = guard.check(tc.name, tc.arguments or {}, str(tc.id or ""))
        if err is not None:
            blocked[i] = err

    async def _exec(tc):
        """단일 tool_call을 실행하고 (tc, result_content) 튜플을 반환한다."""
        held = withheld_tools().get(tc.name)
        if held:
            # 공개 범위로 이번 턴에 막은 도구 — 바인딩에서 뺐지만 다른 경로(작업자 바인딩 등)로 와도 실행하지 않는다.
            logger.info("[executor] tool WITHHELD: %s", tc.name)
            return tc, f"[차단] {held}"
        tool = find_tool(tc.name, all_tools)
        # 진단 로그(INFO): 어떤 도구를 실제로 호출했는지 — "삭제/조회 실패"가 LLM측(도구
        # 미호출)인지 MCP측(호출했으나 실패)인지 구분하는 근거. args 는 PII 우려로 debug 유지.
        logger.info("[executor] call tool=%s", tc.name)
        logger.debug("[executor] tool=%s args=%s", tc.name, redact_for_log(tc.arguments))
        if tool is None:
            logger.warning("[executor] tool NOT FOUND: %s", tc.name)
            return tc, f"[ERROR] 도구를 찾을 수 없습니다: {tc.name}"
        leftover = unresolved_placeholder_error(tc.name, tc.arguments)
        if leftover is not None:
            # 가명화 래퍼가 복원하지 못한 자리표시가 인자에 남았다(모델이 토큰을 지어내거나 변형).
            # 그대로 실행하면 토큰이 수신자·결재선으로 나간다 — 막고 모델에 되돌려 다시 쓰게 한다.
            logger.warning("[executor] tool ARG PLACEHOLDER name=%s", tc.name)
            return tc, leftover
        try:
            result = await tool.execute(**tc.arguments)
        except TypeError as e:
            # 모델이 스키마를 어겨 인자를 빠뜨리거나 형을 틀린 경우다(실측 2026-09-23:
            # eapproval_create_document 를 title 없이 불러 TypeError → 턴 전체가 죽고
            # "답변을 생성하지 못했습니다"가 나갔다). 무엇이 잘못됐는지 돌려주면 모델이
            # 다음 호출에서 채운다 — 이 함수의 계약이 원래 그렇다(docstring).
            logger.warning("[executor] tool ARG ERROR name=%s: %s", tc.name, e)
            return tc, (
                f"[ERROR] `{tc.name}` 호출 인자가 잘못됐습니다: {e}. "
                "빠진 인자를 채워 다시 호출하세요."
            )
        except Exception as e:
            # 도구 내부 예외가 턴을 죽이지 않게 한다. 한 도구의 실패는 그 도구의 결과로만
            # 남고 다른 도구·나머지 대화는 그대로 이어진다.
            logger.exception("[executor] tool EXCEPTION name=%s", tc.name)
            return tc, f"[ERROR] `{tc.name}` 실행 중 오류가 발생했습니다: {e}"
        content = serialize_result(result)
        if result.success:
            _schedule_harvest(content)
            logger.debug(
                "[executor] tool_result name=%s success=True preview=%s",
                tc.name, content[:300],
            )
        else:
            # 진단(WARNING): 도구는 호출됐으나 실패 = MCP측 신호. 에러 텍스트로 원인 구분
            # (예: "team '회사' not found" → MCP 팀해석 문제).
            logger.warning("[executor] tool FAILED name=%s error=%s", tc.name, (result.error or "")[:300])
        return tc, content

    async def _blocked(tc, content):
        return tc, content

    return list(await asyncio.gather(*[
        _blocked(tc, blocked[i]) if i in blocked else _exec(tc) for i, tc in enumerate(tool_calls)
    ]))
