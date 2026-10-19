"""메모리 v2 LLM 레인 + 관대한 JSON 파서.

레인 분리: config/llm_providers.yaml 의 ``memory:`` 블록이 provider/model/max_tokens/
temperature 를 정한다(용도별 레인 선례). 대화 레인(vllm max_tokens 1024)을
그대로 쓰면 10턴 배치의 atom 배열이 잘려 파싱 실패로 조용히 빈 결과가 된다(v1 실측).

트레이싱 제외: 추출·판정·요약은 사용자 요청 처리가 아니라 백그라운드 유지보수라
LangSmith run/토큰 집계에서 뺀다(tracing_context(enabled=False)).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Any, Literal, Optional

from langsmith.run_helpers import tracing_context

from agent_arch.core.config import settings
from agent_arch.llm.base import Message
from agent_arch.llm.factory import LLMFactory

logger = logging.getLogger(__name__)

_LANE_KEYS = ("model", "max_tokens", "temperature", "top_p")
# provider·model 을 비워 둔다 — 배포가 DEFAULT_LLM_PROVIDER 로 정한다. 레인이 자기 값으로
# 갖는 것은 배치 JSON 생성에 필요한 토큰·온도뿐이다(대화 레인의 1024 로는 atom 배열이 잘린다).
_DEFAULT_LANE = {"max_tokens": 4096, "temperature": 0.2}


def memory_lane() -> dict:
    """``memory:`` 레인 설정(없으면 내장 기본값)."""
    lane = dict(settings.llm_providers.get("memory") or {})
    return {**_DEFAULT_LANE, **lane}


def _register_providers() -> None:
    """프로바이더 모듈 import 로 LLMFactory 등록을 보장한다(백그라운드 경로는 오케스트레이터를 안 거침)."""
    for mod in ("agent_arch.llm.vllm", "agent_arch.llm.gemini", "agent_arch.llm.openai_provider"):
        try:
            __import__(mod)
        except Exception:  # pragma: no cover - 선택 의존성 부재
            logger.debug("프로바이더 모듈 import 실패(무시): %s", mod)


def create_memory_llm():
    """메모리 레인 설정으로 LLM 인스턴스를 만든다."""
    lane = memory_lane()
    _register_providers()
    provider = lane.get("provider") or settings.default_llm_provider
    overrides = {k: lane[k] for k in _LANE_KEYS if lane.get(k) is not None}
    return LLMFactory.create(provider, **overrides)


def untraced():
    """이 컨텍스트 안의 provider.invoke 는 LangSmith run 을 만들지 않는다."""
    return tracing_context(enabled=False)


async def invoke_text(messages: list[Message]) -> tuple[str, str, int]:
    """레인 LLM 을 도구 없이 호출해 (본문, 모델명, 지연ms)를 돌려준다.

    finish_reason 이 length 면 잘린 것이라 경고를 남긴다 — 레인 max_tokens 을 올려야 한다.
    """
    llm = create_memory_llm()
    model = str(getattr(llm, "model", None) or memory_lane().get("model") or "?")
    t0 = time.monotonic()
    with untraced():
        resp = await llm.invoke(messages, tool_mode="NONE")
    latency_ms = int((time.monotonic() - t0) * 1000)
    content = (resp.content or "").strip()
    if getattr(resp, "finish_reason", "") == "length":
        logger.warning(
            "memory_v2 LLM 응답 잘림(finish_reason=length, model=%s, %d chars) — 레인 max_tokens 확인",
            model, len(content),
        )
    return content, model, latency_ms


def prompt_sha(text: str) -> str:
    """프롬프트 버전 식별자(sha256 앞 16자). 생성 로그에 남겨 어느 프롬프트가 만든 기억인지 추적."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


# ─── JSON 파서 ─────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?|\n?```\s*$", re.MULTILINE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
# "priority": <숫자·문자열·null 이 아닌 토큰> → 50 (약한 모델이 `priority: high` 류를 낸다)
# (?=\S) 로 \s* 의 역추적을 막는다 — 없으면 공백을 값의 일부로 먹어 "priority": 70 도 50 으로 바꿔 버린다.
_BAD_PRIORITY_RE = re.compile(r'("priority"\s*:\s*)(?=\S)(?![-\d"nt\[{])[^,}\]\n]+')
# "priority": "70" → 70
_QUOTED_PRIORITY_RE = re.compile(r'("priority"\s*:\s*)"(-?\d+)"')


def strip_fence(text: str) -> str:
    """```json ... ``` 펜스를 벗기고 본문만 남긴다."""
    t = (text or "").strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t).strip()
    return t


def repair_json(body: str) -> str:
    """흔한 JSON 오염을 국소 복구한다(후행 콤마, 비숫자 priority). 실패해도 원문은 유지."""
    b = _TRAILING_COMMA_RE.sub(r"\1", body)
    b = _QUOTED_PRIORITY_RE.sub(r"\g<1>\2", b)
    b = _BAD_PRIORITY_RE.sub(r"\g<1>50", b)
    return b


def _slice_outer(body: str, kind: Literal["object", "array"]) -> Optional[str]:
    open_ch, close_ch = ("{", "}") if kind == "object" else ("[", "]")
    start, end = body.find(open_ch), body.rfind(close_ch)
    if start == -1 or end <= start:
        return None
    return body[start : end + 1]


def extract_json(text: str, kind: Literal["object", "array"]) -> Any:
    """LLM 응답에서 JSON object/array 하나를 뽑는다. 못 뽑으면 None.

    순서: 펜스 제거 → 그대로 loads → 바깥 괄호 슬라이스 → 국소 복구 후 loads.
    kind 와 다른 타입이 나오면 None(호출부가 대안 kind 로 재시도).
    """
    body = strip_fence(text)
    if not body:
        return None
    want = dict if kind == "object" else list
    # 본문 전체가 유효 JSON 이면 그 타입이 답이다 — 다른 타입이면 바깥 괄호를 잘라 안쪽을
    # 꺼내지 않는다(배열 응답에서 첫 원소 객체를 "객체 응답"으로 오인하는 것을 막는다).
    try:
        whole = json.loads(body)
    except (ValueError, TypeError):
        whole = _MISSING
    if whole is not _MISSING:
        return whole if isinstance(whole, want) else None
    for candidate in (repair_json(body), _slice_outer(body, kind)):
        if not candidate:
            continue
        for attempt in (candidate, repair_json(candidate)):
            try:
                data = json.loads(attempt)
            except (ValueError, TypeError):
                continue
            if isinstance(data, want):
                return data
    return None


_MISSING = object()
