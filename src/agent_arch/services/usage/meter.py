"""벤더 호출 계측 — 돈이 나가는 호출마다 agent.usage_meters 에 한 행을 남긴다.

## 왜 벤더 경계인가

에이전트 청구는 오래 agent.messages 의 **대화 턴 토큰**만 셌다. 그러나 벤더 청구서에는 턴 밖의
호출도 찍힌다 — 대화 압축·메모리 요약·라우터·임베딩·영수증 OCR·도면 판독 본작업·게스트.
토큰이 아닌 비용도 있다 — Brave 검색 호출, OpenAI web_search 도구 호출, STT 오디오 길이.
담당자 지시(2026-09-29)로 이것들을 전부 에이전트 청구에 넣는다.

호출 지점을 하나씩 찾아 붙이면 새 기능이 생길 때마다 한 곳씩 빠진다. 그래서 **벤더 SDK 를
부르는 자리**(app/llm 프로바이더, 위탁 레인, 임베딩, STT, Brave 도구)에서 센다. 어떤 기능이
불렀든 그 자리를 지나야 돈이 나가므로 빠짐이 없고, 한 호출이 두 번 세어지지도 않는다.

## 귀속

회사·사용자·서비스는 요청 스코프 ContextVar(``get_user_context``)에서 읽는다 — 턴 저장
(app/memory/conversation.py)과 같은 출처라 두 원장이 같은 축으로 모인다. 사용자 컨텍스트가
세워지기 전에 도는 호출(STT 는 채팅 요청보다 먼저 전사한다)이나 따로 도는 경로(게스트)는
``meter_scope`` 또는 ``user_info`` 인자로 귀속을 넘긴다. 귀속이 없으면 NULL 로 남기고
common_app 이 '미매핑'으로 합산한다 — 총량에서 빠지지 않는다.

## 실패

기록 실패가 사용자 요청을 막아서는 안 된다. 쓰기는 백그라운드 태스크로 흘리고, 실패는
로그만 남긴다(표가 없는 환경 — migration_v42 미적용 — 에서도 요청은 정상이다).
"""
from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

from agent_arch.agents.context import get_user_context
from agent_arch.core.config import settings
from agent_arch.db.connection import db_pool

logger = logging.getLogger(__name__)

METER_LLM_TOKENS = "llm_tokens"
METER_WEB_SEARCH = "web_search_call"
METER_STT = "stt_seconds"

# 요청 스코프 귀속 덮어쓰기(게스트처럼 사용자 컨텍스트가 없는 경로용).
_scope: ContextVar[Optional[dict]] = ContextVar("usage_meter_scope", default=None)

# 진행 중인 쓰기 태스크 — 참조를 잡아 두지 않으면 GC 가 태스크를 거둘 수 있다.
_pending: set[asyncio.Task] = set()

_INSERT = """
INSERT INTO agent.usage_meters
    (user_sub, user_email, company, service, meter, provider, model_name,
     quantity, input_tokens, output_tokens, usage_missing, source,
     cached_input_tokens, cache_write_tokens)
VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
"""


@contextmanager
def meter_scope(**attrs: Any) -> Iterator[None]:
    """이 블록 안의 계측에 귀속(service·company·user_email·user_sub)을 덮어쓴다.

    Args:
        **attrs: 덮어쓸 귀속 값. None 인 값은 무시한다.
    """
    base = dict(_scope.get() or {})
    base.update({k: v for k, v in attrs.items() if v is not None})
    token = _scope.set(base)
    try:
        yield
    finally:
        _scope.reset(token)


def set_meter_scope(**attrs: Any) -> None:
    """요청 핸들러 첫 줄에서 귀속을 세운다(요청마다 컨텍스트가 따로라 되돌릴 필요가 없다).

    스트리밍 응답은 핸들러가 돌려준 뒤에도 같은 요청 컨텍스트에서 LLM 을 부르므로, with 블록
    (``meter_scope``)으로는 그 호출까지 덮지 못한다 — 그 경로용이다.
    """
    base = dict(_scope.get() or {})
    base.update({k: v for k, v in attrs.items() if v is not None})
    _scope.set(base)


def _attribution(user_info: Optional[dict]) -> dict:
    """계측 한 행의 귀속 — 명시 인자 > meter_scope > 요청 사용자 컨텍스트."""
    ctx = dict(get_user_context() or {})
    ctx.update(_scope.get() or {})
    if user_info:
        ctx.update({k: v for k, v in user_info.items() if v is not None})
    return {
        "user_sub": ctx.get("user_sub") or ctx.get("sub"),
        "user_email": ctx.get("user_email") or ctx.get("email"),
        "company": ctx.get("company"),
        "service": ctx.get("service"),
    }


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def cache_token_fields(details: Any) -> dict:
    """OpenAI usage 의 ``*_tokens_details`` 에서 캐시 적중·쓰기 토큰을 꺼낸다.

    Chat Completions 는 ``prompt_tokens_details``, Responses 는 ``input_tokens_details`` 에
    ``cached_tokens``(그리고 GPT-5.6 이상은 ``cache_write_tokens``)를 싣는다. SDK 버전·모델마다
    모양이 달라 getattr 로만 읽고, 정수가 아닌 값은 버린다.

    Returns:
        0 보다 큰 값만 담은 dict. 키가 없으면 0 으로 읽는다(``meter_llm_call``) — 캐시가 없는
        호출의 usage dict(LangSmith 메타데이터·턴 저장)는 지금 모양 그대로 둔다.
    """
    out = {}
    if details is None:
        return out
    for name in ("cached_tokens", "cache_write_tokens"):
        value = details.get(name) if isinstance(details, dict) else getattr(details, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            out[name] = value
    return out


async def _write(args: tuple) -> None:
    pool = db_pool.get_pool()
    if pool is None:
        return
    try:
        await pool.execute(_INSERT, *args)
    except Exception:
        logger.warning("사용량 계측 기록 실패(무시) meter=%s model=%s", args[4], args[6], exc_info=True)


def record_meter(
    meter: str,
    *,
    provider: Optional[str],
    model_name: Optional[str],
    quantity: float = 0,
    input_tokens: Any = 0,
    output_tokens: Any = 0,
    usage_missing: bool = False,
    source: Optional[str] = None,
    user_info: Optional[dict] = None,
    cached_input_tokens: Any = 0,
    cache_write_tokens: Any = 0,
) -> None:
    """계측 한 행을 남긴다. 이벤트 루프가 없거나 꺼져 있으면 조용히 건너뛴다.

    Args:
        meter: METER_* 중 하나.
        provider: 벤더(openai·gemini·vllm·brave …).
        model_name: 모델명. 검색 호출처럼 모델이 없으면 None.
        quantity: 호출 수(llm_tokens 는 1) · 검색 호출 수 · STT 초.
        input_tokens: 입력 토큰(llm_tokens 만).
        output_tokens: 출력 토큰(llm_tokens 만).
        usage_missing: 벤더가 usage 를 돌려주지 않았는지.
        source: 호출 지점 표식(디버깅용).
        user_info: 귀속 덮어쓰기(``build_user_info`` 결과 등).
        cached_input_tokens: input_tokens 중 캐시 적중분(llm_tokens 만, migration_v45).
        cache_write_tokens: input_tokens 중 캐시에 새로 쓴 몫(llm_tokens 만, migration_v45).
    """
    if not settings.enable_usage_meters:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    who = _attribution(user_info)
    args = (
        who["user_sub"], who["user_email"], who["company"], who["service"],
        meter, provider, model_name,
        float(quantity or 0), _as_int(input_tokens), _as_int(output_tokens),
        bool(usage_missing), source,
        # 맨 뒤에 붙인다 — _write 의 실패 로그가 args 위치(4·6)를 쓴다.
        _as_int(cached_input_tokens), _as_int(cache_write_tokens),
    )
    task = loop.create_task(_write(args))
    _pending.add(task)
    task.add_done_callback(_pending.discard)


def meter_llm_call(
    provider: Optional[str],
    model_name: Optional[str],
    usage: Optional[dict],
    *,
    source: Optional[str] = None,
) -> None:
    """LLM 한 호출의 토큰을 기록한다. usage 키는 prompt/completion 표준 키를 쓴다.

    Args:
        provider: 벤더.
        model_name: 모델명.
        usage: {prompt_tokens, completion_tokens, total_tokens[, cached_tokens,
            cache_write_tokens]}. None 이면 usage_missing. 캐시 두 키는 입력에 포함된 몫이고
            없으면 0 이다(vLLM·임베딩·MCP 등).
        source: 호출 지점 표식.
    """
    usage = usage or {}
    if model_name and model_name.startswith("models/"):
        # Gemini SDK 는 "models/<id>" 로 부른다 — 가중치표는 벤더 모델 id 로 등록된다.
        model_name = model_name[len("models/"):]
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = _as_int(usage.get("total_tokens"))
    if prompt is None and completion is None and total:
        # 입출력을 나눠 주지 않는 벤더 — 합계를 입력으로 본다(과소 집계보다 낫다).
        prompt = total
    elif total > _as_int(prompt) + _as_int(completion):
        # Gemini 2.5 는 생각 토큰(thoughts_token_count)을 candidates 에 넣지 않고 합계에만
        # 싣는다. 벤더는 그 몫을 **출력 단가로** 청구하므로 출력에 더한다.
        completion = total - _as_int(prompt)
    # 캐시 적중·쓰기는 입력의 서로 겹치지 않는 부분집합이다. 합이 입력을 넘으면 벤더 응답이
    # 이상한 것이라 숨기지 않고 경고를 남긴 뒤 입력 안으로 자른다(적중분을 먼저 채운다).
    input_n = _as_int(prompt)
    cached = _as_int(usage.get("cached_tokens"))
    cache_write = _as_int(usage.get("cache_write_tokens"))
    if cached + cache_write > input_n:
        logger.warning(
            "캐시 토큰이 입력을 넘음 — 입력으로 자름 provider=%s model=%s input=%d cached=%d write=%d",
            provider, model_name, input_n, cached, cache_write,
        )
        cached = min(cached, input_n)
        cache_write = min(cache_write, input_n - cached)
    record_meter(
        METER_LLM_TOKENS,
        provider=provider,
        model_name=model_name,
        quantity=1,
        input_tokens=prompt,
        output_tokens=completion,
        usage_missing=not usage,
        source=source,
        cached_input_tokens=cached,
        cache_write_tokens=cache_write,
    )


async def flush_pending(timeout: float = 5.0) -> None:
    """남은 쓰기를 기다린다(종료 훅·테스트용)."""
    if _pending:
        await asyncio.wait(list(_pending), timeout=timeout)


#: MCP 서버가 도구 호출 안에서 쓴 LLM 사용량을 싣는 예약 키(mattermost-mcp ``usage.USAGE_KEY``).
MCP_USAGE_KEY = "_usage"


def harvest_mcp_usage(tool_name: str, output: Any) -> Any:
    """MCP 도구 결과에서 ``_usage`` 를 떼어 계측에 적고, 떼어낸 결과를 돌려준다.

    메신저 KB 리랭킹·질의 임베딩·첨부 판독은 MCP 서버가 Gemini 를 직접 불러 agent 의 계측 지점을
    지나지 않는다. 서버는 회사·서비스를 모르므로 사용량만 돌려주고, 여기서 **이 요청의** 사용자
    컨텍스트로 귀속한다. 키는 모델이 보는 결과에서 지운다 — 청구 부수 정보다.
    """
    if not isinstance(output, dict) or MCP_USAGE_KEY not in output:
        return output
    calls = output.get(MCP_USAGE_KEY)
    cleaned = {k: v for k, v in output.items() if k != MCP_USAGE_KEY}
    try:
        for call in calls if isinstance(calls, list) else []:
            if not isinstance(call, dict):
                continue
            meter_llm_call(
                call.get("provider"),
                call.get("model_name"),
                None if call.get("usage_missing") else {
                    "prompt_tokens": call.get("input_tokens") or 0,
                    "completion_tokens": call.get("output_tokens") or 0,
                },
                source=f"mcp.{tool_name}.{call.get('source') or 'llm'}",
            )
    except Exception:
        logger.warning("MCP 사용량 기록 실패(무시) tool=%s", tool_name, exc_info=True)
    return cleaned
