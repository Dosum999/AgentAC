"""한 턴의 웹 근거 총량 예산 — 더 찾게 하는 장치(충분성 루프·검증기 보강)가 모델 컨텍스트를 넘기지 않게 한다.

왜(2026-09-28 운영 서버 실측, vllm EPDM): 선실행 검색 + 충분성 2라운드 + 검증기 "더 찾아라" 보강이 겹쳐 검색 13건(건당 ~2만 자)·
페이지 8건이 한 대화에 쌓였다. 65,536 토큰 창을 넘겨 vLLM 이 max_tokens 를 떼고 재시도했지만 남은 자리를 추론이 다 써 빈 응답 →
ANY 재시도도 빈 응답 → "조회하지 못했습니다" 폴백. 근거는 가장 많이 모은 턴이 가장 나쁘게 끝났다.

두 가지를 코드가 한다:
- 쌓인 결과가 예산의 몇 배(``STOP_FACTOR``)를 넘기면 더 찾지 않는다 — 충분성 루프·검증기 보강을 멈추고 답으로(``over_budget``).
- 모델을 부르기 전에 웹 도구 결과를 예산 안으로 줄인다(``compact``). 줄이는 방식은 이력 저장과 같은
  ``trim_tool_results``(JSON 구조 유지, 큰 결과부터 공통 상한) — 확률적 요약이 아니라 결정론적 절단이다.
"""
from __future__ import annotations

import logging

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)


def _small_window(llm) -> bool:
    """컨텍스트 창이 작은 모델인가 — 지금은 자체 vLLM(65,536 토큰). gemini(1M)·그 밖의 클라우드 모델은 큰 창."""
    try:
        from agent_arch.llm.pii import unwrap_llm
        from agent_arch.llm.vllm import VllmLLM
        return isinstance(unwrap_llm(llm), VllmLLM)
    except Exception:
        return False


def budget(llm=None) -> int:
    """모델 창에 맞춘 웹 근거 총량(자). 실측(2026-09-28 운영 서버): 창 65k 에 맞춘 6만 자를 gemini 에도 걸자 근거가 18만→5만 자로
    잘려 답이 절반으로 짧아지고, 앞 턴에 찾던 엔화 값을 "찾을 수 없다"고 했다. 작은 창 모델만 좁게, 나머지는 넓게 둔다."""
    key, default = (("web_evidence_max_chars", 60_000) if _small_window(llm)
                    else ("web_evidence_max_chars_large", 400_000))
    try:
        return max(0, int(getattr(settings, key, default)))
    except Exception:
        return default


def _web_tool_messages(messages: list) -> list:
    from agent_arch.services.web.brave_output import WEB_TOOLS
    return [m for m in messages or [] if getattr(m, "role", None) == "tool" and getattr(m, "name", None) in WEB_TOOLS]


def web_chars(messages: list) -> int:
    return sum(len(m.content or "") for m in _web_tool_messages(messages))


#: 더 찾기를 멈추는 선 = 예산 × 이 배수. 모델에 넘기기 전엔 늘 ``compact`` 로 예산 안에 줄이므로, 루프를 예산에서 바로 멈추면
#: 선실행만으로 넘는 자료형 질문(EPDM: 선실행 9.6만 자)에서 재검색이 통째로 빠져 gemini 가 "확인되지 않는다"로 돌아갔다(2026-09-28).
#: 멈춤은 폭주(원 사고: 검색 13·페이지 8) 방지용으로만 둔다.
STOP_FACTOR = 3


def over_budget(messages: list, factor: int = STOP_FACTOR, llm=None) -> bool:
    cap = budget(llm)
    return bool(cap) and web_chars(messages) >= cap * factor


def compact(messages: list, cap: int | None = None, llm=None) -> int:
    """웹 도구 결과 메시지를 ``cap``(기본 예산) 안으로 제자리에서 줄인다. 줄인 글자 수를 돌려준다(0=그대로)."""
    from agent_arch.memory.conversation import MAX_TOOL_RESULT_CHARS, trim_tool_results

    cap = budget(llm) if cap is None else cap
    msgs = _web_tool_messages(messages)
    before = sum(len(m.content or "") for m in msgs)
    if not cap or before <= cap:
        return 0
    rows = trim_tool_results([{"tool_call_id": m.tool_call_id, "name": m.name, "content": m.content or ""} for m in msgs],
                             per_cap=min(MAX_TOOL_RESULT_CHARS, cap), total_cap=cap)
    for m, r in zip(msgs, rows):
        m.content = r["content"]
    after = sum(len(m.content or "") for m in msgs)
    logger.info("[evidence_budget] 웹 결과 %d건 %d자 → %d자(예산 %d)", len(msgs), before, after, cap)
    return before - after
