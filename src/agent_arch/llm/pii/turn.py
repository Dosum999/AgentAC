"""턴 어휘 — 이번 요청의 컨텍스트(요청자·회사·디렉터리·도구 결과)로 어휘를 만든다.

구성: 요청자 본인(이름·이메일·계정) + 임직원 디렉터리(요청자별 캐시, 정식 이름·이메일만)
+ 이번 대화 도구 결과에서 수확한 이름(성 뗀 이름·영문 ID 까지) + 회사명(보호).
어느 소스가 실패해도 응답을 막지 않는다 — 그 소스만 비운다.

디렉터리는 ``prewarm()`` 이 턴 시작에 백그라운드 태스크로 채우기 시작하고,
``build_turn_lexicon()`` 은 LLM 호출 직전 2초 상한으로 기다린다. 초과하면 이번 턴은
부분 어휘로 가고 태스크는 끝까지 받아 다음 턴부터 완전해진다.
"""
from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from typing import Optional

from agent_arch.agents.context import get_conversation_id, get_user_context
from agent_arch.llm.pii import sources
from agent_arch.llm.pii.lexicon import Lexicon, build_lexicon

logger = logging.getLogger(__name__)

_WAIT_SECONDS = 2.0
_prewarm_task: ContextVar[Optional["asyncio.Task"]] = ContextVar("pii_prewarm_task", default=None)


def requester_from_context(ctx: Optional[dict] = None) -> dict:
    ctx = ctx if ctx is not None else (get_user_context() or {})
    return {
        "name": ctx.get("name"),
        "email": ctx.get("email"),
        "username": ctx.get("username"),
    }


def company_names_from_context(ctx: Optional[dict] = None) -> list[str]:
    ctx = ctx if ctx is not None else (get_user_context() or {})
    names: list[str] = []
    for key in ("company", "active_company"):
        v = ctx.get(key)
        if isinstance(v, str) and v.strip():
            names.append(v.strip())
    for c in ctx.get("companies") or []:
        if isinstance(c, dict):
            for key in ("name", "company_name", "company"):
                v = c.get(key)
                if isinstance(v, str) and v.strip():
                    names.append(v.strip())
        elif isinstance(c, str) and c.strip():
            names.append(c.strip())
    return names


async def _prewarm_if_enabled(sub: str, company: Optional[str]) -> None:
    from agent_arch.llm.pii.policy import company_enabled

    if await company_enabled(company):
        await sources.prewarm_directory(sub)


def prewarm() -> None:
    """턴 시작에 디렉터리 캐시 적재를 백그라운드로 시작한다(요청 컨텍스트를 상속). 회사가 켜지 않았으면 조회하지 않는다."""
    ctx = get_user_context() or {}
    sub = ctx.get("sub")
    if not sub:
        return
    try:
        task = asyncio.create_task(_prewarm_if_enabled(sub, ctx.get("company") or ctx.get("active_company")))
        _prewarm_task.set(task)
    except RuntimeError:  # 루프 밖
        pass


async def _directory(sub: Optional[str]) -> dict:
    task = _prewarm_task.get()
    if task is not None and not task.done():
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=_WAIT_SECONDS)
        except asyncio.TimeoutError:
            logger.info("[pii_guard] 디렉터리 예열 %.1fs 초과 — 이번 턴은 캐시 유무로 진행", _WAIT_SECONDS)
        except Exception:  # noqa: BLE001
            pass
    return await sources.directory_lexicon(sub, timeout=_WAIT_SECONDS)


async def build_turn_lexicon() -> Lexicon:
    """이번 턴의 어휘. 실패해도 예외를 내지 않는다(빈 어휘라도 정규식 치환은 돈다)."""
    try:
        ctx = get_user_context() or {}
        sub = ctx.get("sub")
        conv = get_conversation_id()
        directory, harvested = await asyncio.gather(_directory(sub), sources.harvested_lexicon(conv))
        persons = [(p[0], p[1]) for p in directory.get("persons", []) if isinstance(p, (list, tuple)) and len(p) == 2]
        companies = company_names_from_context(ctx) + list(directory.get("companies", []))
        return build_lexicon(
            persons, requester=requester_from_context(ctx), tool_persons=harvested, company_names=companies,
        )
    except Exception:  # noqa: BLE001
        logger.warning("[pii_guard] 턴 어휘 구성 실패 — 요청자 본인만으로 진행", exc_info=True)
        try:
            ctx = get_user_context() or {}
            return build_lexicon(requester=requester_from_context(ctx), company_names=company_names_from_context(ctx))
        except Exception:  # noqa: BLE001
            return Lexicon.empty()
