"""웹 근거 충분성 루프 — 답하기 전에 "이 근거로 질문에 명확히 답할 수 있나"를 판정하고, 부족하면 코드가 다시 찾는다.

왜(2026-09-28 운영 서버 실측): "오늘 유로 기준환율"에 선실행 검색 1회로 받은 결과엔 공식 매매기준율이 없었고, 모델(gemini)은
더 찾지 않고 "확인되지 않는다"로 답했다. "부족하면 더 찾아라"를 지침·도구 결과 안내·사후 검증기 어디에 넣어도 소극적인 모델은
따르지 않았다. 레퍼런스(Jina DeepSearch·LangChain Open Deep Research·Perplexity Pro)의 공통 설계는 답 생성과 분리된
**충분성 판정 → 부족하면 검색어를 바꿔 재검색 → 예산이 다하면 있는 근거로 답** 루프다. 검색 반복을 모델 성향이 아니라 코드가 보장한다.

한 라운드 = 판정 LLM 1회(가벼운 레인, 도구 없음) + 부족할 때 새 검색어 ≤3·열 URL ≤2 를 병렬 실행. 라운드 수는
WEB_SUFFICIENCY_ROUNDS(기본 2, 0=끔). 판정은 LLM 이, 실행·예산·중복 차단은 코드가 한다(budget.WebGuard 가 같은 검색 재실행을 막는다).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

_MAX_QUERIES = 3
_MAX_OPEN = 2
_DIGEST_ROWS = 28
_ROW_CHARS = 360
_PAGE_CHARS = 900
_ROWS_PER_CALL = 4
_DIGEST_CHARS = 11_000
_CONTEXT_TOKENS = 4096

SUFFICIENCY_PROMPT = """\
너는 웹 조사의 근거 충분성 판정기다. 답을 쓰지 말고, 지금 모은 근거로 사용자 질문에 **명확하고 최신으로** 답할 수 있는지만 판정한다.

충분(sufficient=true)의 조건 — 전부 만족해야 한다:
1. 질문이 묻는 바로 그 대상·수치·기간이 근거에 있다(비슷한 다른 대상·다른 통화쌍·다른 기간이면 부족).
2. '오늘·지금·현재'를 물었으면 한국 시각(KST) 오늘 날짜의 값이다 — 장중 값·최근 호가·오늘 새벽 해외 마감가도 오늘 값이다.
   해외 시각은 뒤의 [= … KST] 환산으로 판단한다.
   '현재 추이·요즘 단가'를 물었으면 이번 달 값이 있어야 한다. 지난달까지만 있으면 **첫 라운드는 부족**(이번 달 값을 찾는
   검색어·가격지수 원 페이지 열기를 제안), 마지막 라운드면 지난달까지로 충분으로 본다.
3. 공식·기준 값(예: 한국의 '기준환율' = 서울외국환중개 고시 매매기준율)을 물었으면 그 출처의 값이 있어야 한다. 없으면 **첫
   라운드는 부족**(고시 기관명·고시 페이지로 찾는 검색어 제안), 마지막 라운드에서 **같은 대상의 오늘 시장 값**이 있으면 충분으로
   보고 missing 에 "공식 고시값 미확인"이라고만 적는다(찾을 수 없는 값을 쫓아 검색을 되풀이하지 않는다).
4. 한 가지 사실을 묻는 질문이면 한 출처로 충분하다 — 조사·비교 질문이면 핵심 항목이 빠지지 않았다. 모든 월·모든 지역의
   빠짐없는 수치, 원인 분석의 완결성까지 요구하지 않는다.

부족하면:
- missing: 무엇이 빠졌는지 한 줄
- queries: 새 검색어 2~3개. **이미 쓴 검색어와 다르게** — 동의어·전문 용어·고시 기관명·언어(한/영)·날짜 표기·문서 유형
  (보도·고시·통계)을 바꾼다. 예) '유로 기준환율' → '유로 매매기준율 서울외국환중개', '외국환시세 9월 28일', 'EUR KRW 고시환율'
- open: 근거 목록에 있는 URL 중 원 페이지를 열어 확인할 것 최대 2개(조각이 오래됐거나 잘렸을 때). 목록에 없는 URL 금지.

출력은 JSON 하나만:
{"sufficient": true|false, "missing": "...", "queries": ["..."], "open": ["https://..."]}
"""


def rounds_budget() -> int:
    try:
        return max(0, int(getattr(settings, "web_sufficiency_rounds", 2)))
    except Exception:
        return 2


def _loads(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def build_digest(calls: list, results: list, question: str = "") -> tuple[str, list[str], list[str]]:
    """(근거 요약, 이미 쓴 검색어, 근거에 나온 URL). 결과 전문 대신 행마다 제목·URL·시점·본문 발췌(앞머리 + 수치·최근 시점 줄)."""
    from agent_arch.agents.react.provenance import _unwrap
    from agent_arch.services.web.brave_output import FETCH_TOOL, WEB_TOOLS
    from agent_arch.agents.context import web_url_key
    from agent_arch.services.web.periods import salient_excerpt

    by_id = {r.get("tool_call_id"): r for r in results or [] if isinstance(r, dict)}
    pages, rows_by_call, tried, urls = [], [], [], []
    for c in calls or []:
        if not isinstance(c, dict) or c.get("name") not in WEB_TOOLS:
            continue
        args = c.get("arguments") or {}
        if args.get("query"):
            tried.append(str(args["query"]))
        data = _loads((by_id.get(c.get("id")) or {}).get("content"))
        if not isinstance(data, dict) or "error" in data:
            continue
        if c.get("name") == FETCH_TOOL:
            pages.append((c.get("name"), {"url": data.get("final_url") or data.get("url"), "title": data.get("title"),
                                          "content": data.get("text"), "fetched_at": data.get("fetched_at"),
                                          "source": data.get("source")}))
            continue
        rows = data.get("results") if isinstance(data.get("results"), list) else []
        rows_by_call.append([(c.get("name"), r) for r in rows[:_ROWS_PER_CALL] if isinstance(r, dict)])
    # 연 페이지를 먼저, 검색 결과는 각 검색의 1위들 → 2위들 … 순(호출당 최대 _ROWS_PER_CALL) — 행 상한(_DIGEST_ROWS)을
    # 첫 검색 몇 개가 다 쓰면 뒤 라운드의 새 검색·연 페이지가 판정기에 안 보인다(2026-09-28 EPDM: 검색 13회, 앞 3회분만 보였다).
    ranked = [group[k] for k in range(_ROWS_PER_CALL) for group in rows_by_call if k < len(group)]
    lines, keys = [], set()
    for name, r in pages + ranked:
        if not r.get("url"):
            continue
        urls.append(r["url"])
        key = web_url_key(r["url"])
        if key in keys or len(lines) >= _DIGEST_ROWS:
            continue
        keys.add(key)
        meta = " | ".join(str(x) for x in (r.get("published"), r.get("latest_period") and f"최근표기 {r['latest_period']}",
                                           r.get("fetched_at") and f"열람 {r['fetched_at']}",
                                           r.get("source") == "search_index" and "인덱스 사본") if x)
        chars = _PAGE_CHARS if name == FETCH_TOOL else _ROW_CHARS
        body = salient_excerpt(_unwrap(r.get("content") or ""), chars, terms=question)
        lines.append(f"- [{name}] {str(r.get('title') or '')[:80]} | {r['url'][:120]}" + (f" | {meta}" if meta else "")
                     + f"\n  {body}")
    return "\n".join(lines)[:_DIGEST_CHARS], tried, urls


def parse_verdict(raw: str) -> dict:
    """관대한 파싱. 못 읽으면 충분(=루프 종료)으로 — 판정기 장애가 답을 막으면 안 된다."""
    from agent_arch.services.memory_v2.llm import extract_json
    data = extract_json(raw or "", "object")
    if not isinstance(data, dict):
        return {"sufficient": True, "parse_error": True, "queries": [], "open": []}
    suff = data.get("sufficient")
    suff = suff is True or str(suff).strip().lower() == "true"
    qs = [" ".join(str(q).split())[:120] for q in (data.get("queries") or []) if isinstance(q, str) and q.strip()]
    op = [str(u).strip() for u in (data.get("open") or []) if isinstance(u, str) and u.strip().startswith("http")]
    return {"sufficient": suff, "missing": str(data.get("missing") or "")[:200], "queries": qs[:_MAX_QUERIES], "open": op[:_MAX_OPEN]}


async def judge(question: str, digest: str, tried: list[str], round_no: int = 1, rounds: int = 1,
                provider: Optional[str] = None) -> dict:
    from agent_arch.agents.react.verify import create_verify_llm
    from agent_arch.llm.base import Message

    now = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M")
    body = (f"[현재 시각] {now} (KST)\n[판정 라운드] {round_no}/{rounds}" + (" — 마지막 라운드" if round_no >= rounds else " — 첫 라운드")
            + f"\n\n[사용자 질문]\n{(question or '').strip()[:600]}\n\n"
            f"[이미 쓴 검색어]\n" + ("\n".join(f"- {q}" for q in tried) or "(없음)") + f"\n\n[근거 요약]\n{digest or '(없음)'}")
    llm = create_verify_llm(provider)                       # 답하는 모델과 같은 제공자(verify.create_verify_llm)
    resp = await llm.invoke([Message(role="system", content=SUFFICIENCY_PROMPT), Message(role="user", content=body)], tool_mode="NONE")
    return parse_verdict(getattr(resp, "content", "") or "")


def plan_calls(verdict: dict, tried: list[str], seen_urls: list[str], known: set) -> list:
    """부족 판정 → 실행할 ToolCall(새 검색어는 brave_llm_context, 열 URL 은 web_fetch). 이미 쓴 검색어·목록 밖 URL 은 버린다."""
    from agent_arch.agents.react.budget import CODE_CALL_PREFIX
    from agent_arch.agents.context import web_url_key
    from agent_arch.llm.base import ToolCall
    from agent_arch.services.web.brave_output import CONTEXT_TOOL, FETCH_TOOL

    if verdict.get("sufficient"):
        return []
    done = {" ".join(q.split()).casefold() for q in tried}
    keys = {web_url_key(u) for u in seen_urls}
    out = []
    if CONTEXT_TOOL in known:
        for q in verdict.get("queries") or []:
            if q.casefold() in done:
                continue
            args = {"query": q, "maximum_number_of_tokens": _CONTEXT_TOKENS}
            if q.isascii():
                args.update(country="US", search_lang="en")
            out.append(ToolCall(id=f"{CODE_CALL_PREFIX}{CONTEXT_TOOL}__{uuid4().hex[:8]}", name=CONTEXT_TOOL, arguments=args))
    if FETCH_TOOL in known:
        for u in verdict.get("open") or []:
            if web_url_key(u) in keys:                       # 근거 목록에 있던 URL 만(지어낸 주소 금지)
                out.append(ToolCall(id=f"{CODE_CALL_PREFIX}{FETCH_TOOL}__{uuid4().hex[:8]}", name=FETCH_TOOL, arguments={"url": u}))
    return out
