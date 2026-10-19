"""턴 예산 — 웹 도구 호출의 중복 억제와 상한.

문제(실측 2026-09-15): "Kim 관련된거 찾아봐" 한 턴에서 brave_web_search 14회, LLM 26회, 69.5만 토큰. 같은 검색어
서너 개를 서너 번씩 되풀이했다. 루프에는 "모델이 도구를 더 안 부르면 끝" 말고 멈출 장치가 없었다. Claude 연구 시스템도
같은 실패("없는 출처를 끝없이 긁기", "중복")를 겪고 예산과 상한으로 막았다고 공개했다.

해법은 두 가지 결정적 가드다. 판단(무엇을 검색할지)은 모델이 하고, 코드는 **자원**만 본다 — 의도 감지가 아니다.
1. 중복: 이번 턴에 같은 도구를 같은 인자로 이미 실행했으면 다시 부르지 않고 "앞선 결과를 보라"는 구조화 오류를 돌려준다.
   앞선 호출이 오류였으면 재시도는 허용한다.
2. 상한: 턴당 웹 호출 수가 WEB_MAX_CALLS_PER_TURN(기본 8, 0=무제한)에 닿으면 "지금 있는 근거로 답하라"를 돌려준다.
3. 단계 상한(2026-09-23 벤치): 호출 수 상한만으론 모델이 한 스텝에 하나씩 8번을 순차로 불렀다(LLM 11회·스텝당 5만 토큰 재전송,
   "오늘 환율" 13회). 웹 호출을 실행한 스텝 수가 WEB_MAX_ROUNDS_PER_TURN(기본 3)에 닿으면 그 뒤 스텝의 웹 호출을 막는다.
   한 스텝 안의 여러 호출은 한 단계로 센다 — 병렬로 넓게 부르는 쪽이 유리해진다.

두 가드는 그래프와 옛 루프가 공유하는 실행기(react/executor.execute_tools)에서 돈다. 웹 도구(brave_*)만 대상이다 —
사내 도구는 상태 변경 뒤 다시 읽는 반복이 정상이라 여기서 건드리지 않는다.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Optional

from agent_arch.agents.tools.base import ToolResult
from agent_arch.core.config import settings
from agent_arch.services.web.brave_output import WEB_TOOLS

logger = logging.getLogger(__name__)

DUPLICATE_CODE = "duplicate_call"
# 코드가 주도하는 호출(선실행·자동 열기·충분성 라운드)의 tool_call id 접두 — 모델 몫 상한(호출 수·단계)에서 뺀다.
# 그 호출들은 자체 예산(라우터 검색어 ≤3·자동 열기 ≤2·WEB_SUFFICIENCY_ROUNDS)으로 묶인다. 중복 차단은 그대로 받는다.
CODE_CALL_PREFIX = "prefetch_"
CAP_CODE = "web_budget_exceeded"
_GUARD_CODES = (DUPLICATE_CODE, CAP_CODE)


def round_cap() -> int:
    try:
        return max(0, int(getattr(settings, "web_max_rounds_per_turn", 3)))
    except Exception:
        return 3


def executed_web(pairs: Iterable[tuple]) -> bool:
    """실행 결과 (tc, content) 중 가드에 막히지 않은 웹 호출이 있었나 — 단계 수를 셀지 판단."""
    return any(getattr(tc, "name", None) in WEB_TOOLS and not is_guard_error(content) for tc, content in pairs)


def cap() -> int:
    try:
        return max(0, int(getattr(settings, "web_max_calls_per_turn", 8)))
    except Exception:
        return 8


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return " ".join(v.split()).casefold()
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items()) if x is not None}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    return v


def canonical_key(name: str, arguments: Optional[dict]) -> str:
    """도구 이름 + 정규화한 인자(키 정렬, 문자열 공백·대소문자 무시, None 제거)."""
    return f"{name}|{json.dumps(_norm(arguments or {}), ensure_ascii=False, sort_keys=True)}"


def is_guard_error(content: Any) -> Optional[str]:
    """실행기가 만든 가드 오류면 그 코드를, 아니면 None."""
    s = content if isinstance(content, str) else ""
    if not s.lstrip().startswith("[ERROR]"):
        return None
    for code in _GUARD_CODES:
        if f"{code}:" in s[:80]:
            return code
    return None


class WebGuard:
    """한 턴의 웹 호출 장부. 이전 iteration 의 호출·결과로 초기화하고, 이번 배치의 호출을 순서대로 판정한다."""

    def __init__(self, prior_calls: Optional[Iterable[dict]] = None, prior_results: Optional[Iterable[dict]] = None,
                 rounds: int = 0):
        results = {r.get("tool_call_id"): str(r.get("content") or "") for r in (prior_results or []) if isinstance(r, dict)}
        self.cap = cap()
        self.round_cap = round_cap()
        self.rounds = max(0, int(rounds or 0))
        self.web_calls = 0
        self.seen: dict[str, str] = {}
        self.duplicates = 0
        self.capped = 0
        for c in prior_calls or []:
            if not isinstance(c, dict) or c.get("name") not in WEB_TOOLS:
                continue
            content = results.get(c.get("id"), "")
            if is_guard_error(content):
                continue                                   # 가드가 막은 호출은 실행되지 않았다
            if not str(c.get("id") or "").startswith(CODE_CALL_PREFIX):
                self.web_calls += 1
            if not content.lstrip().startswith("[ERROR]"):
                self.seen.setdefault(canonical_key(c.get("name"), c.get("arguments")), str(c.get("id") or ""))

    def check(self, name: str, arguments: Optional[dict], call_id: str) -> Optional[str]:
        """실행해도 되면 None, 막으면 모델에게 줄 [ERROR] 문자열."""
        if name not in WEB_TOOLS:
            return None
        key = canonical_key(name, arguments)
        prev = self.seen.get(key)
        if prev:
            self.duplicates += 1
            logger.info("[budget] 중복 웹 호출 차단 tool=%s dup_of=%s", name, prev)
            return _error(DUPLICATE_CODE,
                          f"같은 검색({name}, 같은 인자)을 이번 턴에 이미 실행했다 — 앞선 결과(tool_call_id {prev})를 다시 보라. "
                          "반복하지 말고 다른 검색어·다른 관점으로 바꾸거나, 지금 있는 근거로 답하라.")
        if call_id.startswith(CODE_CALL_PREFIX):
            self.seen[key] = call_id                     # 코드 주도 호출: 중복만 막고 모델 몫 상한은 쓰지 않는다
            return None
        if self.round_cap and self.rounds >= self.round_cap:
            self.capped += 1
            logger.info("[budget] 웹 조회 단계 상한 도달 tool=%s rounds=%d", name, self.rounds)
            return _error(CAP_CODE,
                          f"이번 턴 웹 조회 단계 상한({self.round_cap}단계)에 닿았다. 더 검색하지 말고 지금 있는 근거로 답하라. "
                          "결과 중 가장 최근 시각의 값을 그 시각과 함께 쓰고, 확인하지 못한 부분은 그렇다고 말하라.")
        if self.cap and self.web_calls >= self.cap:
            self.capped += 1
            logger.info("[budget] 웹 호출 상한 도달 tool=%s cap=%d", name, self.cap)
            return _error(CAP_CODE,
                          f"이번 턴 웹 호출 상한({self.cap}회)에 닿았다. 더 검색하지 말고 지금 있는 근거로 답하라. "
                          "확인하지 못한 부분은 확인하지 못했다고 말하라.")
        self.web_calls += 1
        self.seen[key] = call_id
        return None

    def stats(self) -> dict:
        return {"web_calls": self.web_calls, "duplicates": self.duplicates, "capped": self.capped, "cap": self.cap,
                "rounds": self.rounds, "round_cap": self.round_cap}


def _error(code: str, message: str) -> str:
    from agent_arch.agents.react.executor import serialize_result
    return serialize_result(ToolResult(success=False, output={"error": code, "message": message}, error=f"{code}: {message}"))


def summarize(calls: Optional[Iterable[dict]], results: Optional[Iterable[dict]]) -> dict:
    """턴이 끝난 뒤 집계(step_meta 용): 실행된 웹 호출 수, 중복 차단 수, 상한 차단 수."""
    by_id = {r.get("tool_call_id"): str(r.get("content") or "") for r in (results or []) if isinstance(r, dict)}
    web_calls = duplicates = capped = 0
    for c in calls or []:
        if not isinstance(c, dict) or c.get("name") not in WEB_TOOLS:
            continue
        code = is_guard_error(by_id.get(c.get("id"), ""))
        if code == DUPLICATE_CODE:
            duplicates += 1
        elif code == CAP_CODE:
            capped += 1
        else:
            web_calls += 1
    return {"web_calls": web_calls, "duplicates": duplicates, "capped": capped, "cap": cap()}
