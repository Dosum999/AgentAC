"""답변 검증기 — 모델이 최종 답을 낸 직후, `answer` 로 가기 전에 "요청을 충족했는가"를 본다.

문제(dev 실측 2026-09-15): react 루프의 종료 조건이 "모델이 도구를 더 안 부르면 끝"이라 첫 시도가 빈약해도
그대로 답이 됐다 — 검색 1회로 치과·TV·산업용 브라켓을 섞어 답하고, 산업용 브라켓을 못 찾자 산업용 배터리로
채우고, "최근 한 달 논문"에 2017년 위키백과를 내고, 기사 번호가 잘린 URL 을 열고도 상세 답을 냈다.

해법: 검증 노드. 입력은 사용자 요청·답 초안·도구 사용 요약·**코드가 뽑은 증거 사실**(결과 건수·오류·게시일
유무·본문 읽기 횟수·답에 적힌 URL 이 결과에 있는지·잘린 URL). 출력은 통과/보강 + 다음 행동 하나(재검색·본문
읽기·되묻기·한계 명시). 보강이면 지시를 메시지로 붙여 모델을 한 번 더 돌린다(횟수 상한).

도입 순서: off → **shadow**(판정만 기록, 흐름 불변) → on. 그림자 실측으로 판정 비율·정확도를 본 뒤 켠다.
비용은 대상 턴당 LLM 1회(전용 레인, 도구 없음). 대상은 도구를 쓴 턴과 웹 도구가 바인딩된 턴만.

의도 감지 정규식으로 도구를 강제하지 않는다 — 판단은 LLM 이, 코드는 **사실**만 댄다.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Optional

from agent_arch.agents.react.provenance import _unwrap as _unwrap_envelope
from agent_arch.services.web.brave_output import CONTEXT_TOOL, FETCH_TOOL, SEARCH_TOOLS, WEB_TOOLS as WEB_TOOL_NAMES
from agent_arch.services.web.periods import salient_excerpt
from agent_arch.core.config import settings
from agent_arch.llm.base import Message

logger = logging.getLogger(__name__)

MODE_OFF, MODE_SHADOW, MODE_ON = "off", "shadow", "on"
from agent_arch.agents.react.history import FEEDBACK_NAME  # noqa: E402 — 보강 지시 메시지 이름(history 가 걸러 낸다)
_MAX_DRAFT_CHARS = 3_000
_MAX_TOOL_SUMMARY_CHARS = 9_000
_SEARCH_ROWS_SHOWN = 8
_SNIPPET_CHARS = 200
_CONTEXT_BODY_CHARS = 500
_PAGE_BODY_CHARS = 900
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")
#: 웹이 아닌 도구 결과도 행 단위로 펼쳐 보여 준다 — 앞 400자만 자르면 첫 항목도 못 채워 검증기가
#: "증거에 없음·지어냄" 으로 오판한다(2026-09-17 실측: 멘션 3건 답변이 2라운드 연속 revise).
#: 행을 **도구 이름·키 이름으로 고르지 않는다.** 처음엔 접두어 목록 + `results` 봉투를 가정했는데, 목록에 없는 도구와
#: 키가 다른 도구가 그대로 구멍이 됐다 — search_colleagues(`colleagues`)를 "결과 0건" 으로 요약해 찾은 참석자를
#: "확인 안 됨" 으로 고쳐 쓰고(2026-10-01 stg), 가이드 검색(카테고리 그룹 → `chunks`)은 앞 400자에 첫 청크 하나만
#: 들어가 "공용 캘린더 만드는 방법" 답의 근거를 "도구 결과에 없음" 으로 깎았다(2026-10-07 운영). 그래서 결과 JSON 의
#: 어느 깊이에 있든 "dict 목록의 항목" 을 행으로 본다.
_ROW_MAX_DEPTH = 4                       # 봉투 → 그룹 → 행 → 하위 행이면 충분하다
_ROW_BODY_CHARS = 400                    # 가이드 청크는 절차가 본문이라 웹 스니펫(200)보다 길게
_FALLBACK_CHARS = 1_500                  # 행이 없는 결과(단건 dict·문자열)
#: 행에서 검증기가 볼 값 — 이름이 도구마다 달라 후보를 순서대로 본다. 못 찾으면 나머지 짧은 값들이 근거다(사람·파일·메일 머리글).
_ROW_TEXT_KEYS = ("message", "text", "snippet", "content", "purpose")
_ROW_TITLE_KEYS = ("channel_name", "title", "file_name", "name", "subject", "display_name", "author")
_ROW_LINK_KEYS = ("permalink", "doc_url", "url", "links")
_ROW_TIME_KEYS = ("created_at", "started_at", "date", "create_at", "modified")
#: 하위 행을 가진 항목의 이름표(가이드 카테고리·채널·팀) — 펼친 행 앞에 붙여 어디서 나온 행인지 남긴다.
_GROUP_LABEL_KEYS = ("category", "channel_name", "team_name", "display_name", "title", "name")
#: 행이 없을 때 "0건" 인지 가르는 건수 값 — 양수면 모르는 모양이니 원문을 보여 준다.
_ROW_COUNT_KEYS = ("count", "total", "total_matched", "total_mentions", "shown_channels", "result_count")
#: 본문 대신 나열할 때 뺄 값 — 식별자·점수·권한 표식은 근거가 아니다.
_ROW_SKIP_KEYS = ("id", "score", "version", "permission_required", "permission_blocked", "category_top_score")
# provider·model 은 비워 둔다 — 어느 모델로 돌릴지는 배포가 DEFAULT_LLM_PROVIDER 로 정한다.
# 여기 박아 두면 그 프로바이더가 죽었을 때 레인만 따로 죽는다(2026-09-18 vLLM 정지 때 실측).
# 레인이 자기 값으로 갖는 것은 "짧은 JSON 판정 1회"라는 성격뿐이다.
_LANE_DEFAULT = {"max_tokens": 512, "temperature": 0.0}
_VALID_ACTIONS = ("search", "fetch", "clarify", "state_limits", "rewrite", "none")


# ─── 설정 ──────────────────────────────────────────────────────────────────────

def mode() -> str:
    m = str(getattr(settings, "answer_verify_mode", MODE_OFF) or MODE_OFF).strip().lower()
    return m if m in (MODE_OFF, MODE_SHADOW, MODE_ON) else MODE_OFF


def max_rounds() -> int:
    try:
        return max(0, int(getattr(settings, "answer_verify_max_rounds", 1)))
    except Exception:
        return 1


def is_enabled() -> bool:
    return mode() != MODE_OFF


def should_verify(state: dict) -> bool:
    """검증 대상인가 — 도구를 쓴 턴, 또는 웹 도구가 손에 있는 턴. 승인 대기 초안이 있는 턴은 제외(카드가 답이다)."""
    if not is_enabled():
        return False
    if state.get("pending_drafts"):
        return False
    if state.get("used_tools"):
        return True
    names = {getattr(t, "name", "") for t in (state.get("all_tools") or [])}
    return bool(names & WEB_TOOL_NAMES)


def hold_needed(state: dict) -> bool:
    """on 모드의 대상 턴은 토큰을 붙든다 — 보강 판정이 나면 초안을 화면에 남기지 않기 위해."""
    return mode() == MODE_ON and should_verify(state)


# ─── 증거 사실(코드가 계산) ────────────────────────────────────────────────────

def _loads(s: Any) -> Any:
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s)
    except Exception:
        return None


def _rows(data: Any) -> list:
    if not isinstance(data, dict):
        return []
    rows = data.get("results")
    if isinstance(rows, dict) and isinstance(rows.get("items"), list):
        return rows["items"]
    return rows if isinstance(rows, list) else []


def _dict_list(v: Any) -> bool:
    return isinstance(v, list) and any(isinstance(x, dict) for x in v)


def _any_rows(data: Any, label: str = "", depth: int = 0) -> list[tuple[str, dict]]:
    """결과 JSON 어디에 있든 행을 꺼낸다 — (이름표, 행). 하위 목록을 가진 항목은 펼치고 이름표만 넘긴다."""
    if depth > _ROW_MAX_DEPTH:
        return []
    if isinstance(data, list):
        return [row for x in data if isinstance(x, dict) for row in _any_rows(x, label, depth + 1)]
    if not isinstance(data, dict):
        return []
    # dict 값을 펼치는 건 맨 위 봉투에서만(`results: {items}`·`requester`) — 행 안의 dict 필드까지 펼치면 행이 쪼개진다
    children = [v for v in data.values() if _dict_list(v) or (depth == 0 and isinstance(v, dict))]
    if not children:
        return [(label, data)] if depth else []          # 맨 위 단건 dict 는 행이 아니다(폴백이 보여 준다)
    own = next((str(data[k]) for k in _GROUP_LABEL_KEYS if data.get(k) and isinstance(data[k], (str, int))), "")
    inner = " / ".join(x for x in (label, own if depth else "") if x)
    rows = [(label, data)] if depth and any(data.get(k) for k in _ROW_TEXT_KEYS) else []
    for v in children:
        rows.extend(_any_rows(v, inner, depth + 1) if isinstance(v, list) else _any_rows([v], inner, depth))
    return rows


def _row_line(label: str, r: dict) -> str:
    title = next((str(r[k]) for k in _ROW_TITLE_KEYS if r.get(k) and not isinstance(r[k], (dict, list))), "")
    when = next((str(r[k]) for k in _ROW_TIME_KEYS if r.get(k)), "")
    link = next((str(r[k][0] if isinstance(r[k], list) else r[k])
                 for k in _ROW_LINK_KEYS if r.get(k)), "")
    body = next((str(r[k]) for k in _ROW_TEXT_KEYS if r.get(k) and isinstance(r[k], str)), "")
    if not body:
        used = set(_ROW_TITLE_KEYS + _ROW_TIME_KEYS + _ROW_LINK_KEYS + _ROW_SKIP_KEYS)
        body = " / ".join(str(v) for k, v in r.items()
                          if k not in used and not k.endswith("_id") and v not in (None, "")
                          and isinstance(v, (str, int)) and not isinstance(v, bool))
    head = f"[{label}] " if label else ""
    return f"  · {head}{title[:80]} | {when[:19]} | {link[:90]} | {' '.join(body.split())[:_ROW_BODY_CHARS]}"


def collect_facts(state: dict, draft: str) -> dict:
    calls = state.get("collected_tool_calls") or []
    results = state.get("collected_tool_results") or []
    by_id = {r.get("tool_call_id"): r for r in results if isinstance(r, dict)}
    tool_names = [c.get("name") for c in calls if isinstance(c, dict)]
    from agent_arch.agents.react.budget import CAP_CODE, DUPLICATE_CODE, is_guard_error
    guard_codes = [is_guard_error(r.get("content")) for r in results if isinstance(r, dict)]
    duplicates = sum(1 for g in guard_codes if g == DUPLICATE_CODE)
    capped = sum(1 for g in guard_codes if g == CAP_CODE)
    errors = sum(1 for r, g in zip([r for r in results if isinstance(r, dict)], guard_codes)
                 if g is None and str(r.get("content") or "").lstrip().startswith("[ERROR]"))

    web_searches, empty_searches, context_reads, freshness_requested, rows_total, rows_with_date = 0, 0, 0, False, 0, 0
    known_urls: set[str] = set()
    for c in calls:
        if not isinstance(c, dict) or c.get("name") not in WEB_TOOL_NAMES:
            continue
        args = c.get("arguments") or {}
        if args.get("freshness"):
            freshness_requested = True
        if c.get("name") == CONTEXT_TOOL:
            context_reads += 1
        else:
            web_searches += 1
        data = _loads((by_id.get(c.get("id")) or {}).get("content"))
        rows = _rows(data)
        if data is not None and not rows:
            empty_searches += 1
        for r in rows:
            if isinstance(r, dict):
                rows_total += 1
                if r.get("published"):
                    rows_with_date += 1
                if isinstance(r.get("url"), str):
                    known_urls.add(r["url"])

    # 웹 검색이 아닌 도구가 준 링크도 근거다 — SCMS 도구는 결과에 그 건·그 화면을 여는 주소('바로 가기')를 싣는다.
    # 웹 결과 행의 url 만 모으자 그 링크를 "도구 결과에 없는 URL" 로 잡아 지우게 했다(실측 2026-10-01: '출고 어떻게 해?'
    # 에 scms_guide 의 출고처리 링크를 붙였더니 state_limits 로 빠졌다).
    for r in results:
        if isinstance(r, dict) and isinstance(r.get("content"), str):
            known_urls.update(u.rstrip(".,)") for u in _URL_RE.findall(r["content"]))
    history = str(state.get("history_evidence") or "")
    if history:                                   # 앞 턴 답·도구 결과도 근거다(앞 턴에 확인한 공식값을 되돌리던 실측)
        known_urls.update(u.rstrip(".,)") for u in _URL_RE.findall(history))
    unsupported = (unsupported_numbers(draft, results + ([{"content": history}] if history else []))
                   if (web_searches or context_reads) else [])
    newest = newest_period(results) if (web_searches or context_reads) else None
    answer_urls = _URL_RE.findall(draft or "")
    unknown_urls = [u for u in answer_urls if u.rstrip(".,)") not in known_urls
                    and not any(k.startswith(u.rstrip(".,)")) for k in known_urls)]
    return {
        "tools": tool_names,
        "tool_calls": len(calls),
        "tool_errors": errors,
        "web_searches": web_searches,
        "empty_searches": empty_searches,
        "context_reads": context_reads,
        "freshness_requested": freshness_requested,
        "results_total": rows_total,
        "results_with_date": rows_with_date,
        "answer_urls_not_in_results": unknown_urls[:5],
        "numbers_not_in_results": unsupported,
        "newest_period_in_results": newest,
        "duplicate_calls": duplicates,
        "budget_capped": capped,
        "draft_chars": len(draft or ""),
    }


_MAX_UNSUPPORTED = 8


def _digits(s: str) -> str:
    return s.replace(",", "")


def unsupported_numbers(draft: Optional[str], results: list) -> list[str]:
    """답 초안의 특징적 수치(가격·비율·소수) 중 이번 턴 도구 결과 **전문** 어디에도 없는 것.

    벤치(2026-09-23): 심판 최다 감점이 "근거에 없는/시점이 다른 수치"였는데 검증기는 30문항 전부 pass 였다 — 검증기가 보는
    도구 요약은 행당 200~500자라 수치를 스스로 대조할 수 없다. 대조는 코드가 전문으로 하고, 판단(계산값·반올림인지)은
    검증기가 한다. 천 단위 쉼표는 무시하고 비교한다. URL 속 숫자는 보지 않는다.
    """
    from agent_arch.agents.react.provenance import _URL_IN_TEXT_RE, answer_numbers
    nums = answer_numbers(_URL_IN_TEXT_RE.sub(" ", draft or ""))
    if not nums:
        return []
    haystack = _digits(" ".join(str(r.get("content") or "") for r in results if isinstance(r, dict)))
    return [n for n in nums if _digits(n).rstrip("%") not in haystack and not _rounded_from(n, haystack)
            and not _reciprocal_of(n, haystack)][:_MAX_UNSUPPORTED]


def _reciprocal_of(n: str, haystack: str) -> bool:
    """답의 값이 결과에 적힌 작은 소수의 역수인가 — 환율 페이지가 "1 KRW = 0.000645998 EUR" 로만 적은 것을 1,548.00원으로 답한 경우.
    역수는 유효숫자가 줄어 정확히 같지 않으니 0.1% 안이면 같은 값으로 본다(2026-09-28 gemini 유로, 검증기가 근거 없음으로 잡았다)."""
    try:
        target = float(_digits(n).rstrip("%").strip())
    except ValueError:
        return False
    if target < 10:
        return False
    for h in _DEC_IN_TEXT_RE.finditer(haystack):
        v = float(h.group(0))
        if 0 < v < 0.1 and abs(1 / v - target) <= target * 1e-3:
            return True
    return False


_DEC_IN_TEXT_RE = re.compile(r"(?<![\d.])(\d+)\.(\d+)")


def _rounded_from(n: str, haystack: str) -> bool:
    """답의 소수가 결과의 더 긴 소수를 반올림한 값이거나 끝자리 0 만 다른 값인가 — OFX "1,540.207344" → 초안 "1,540.21",
    SMBS "diff5=-11.7" → 초안 "−11.70"(2026-09-28 vllm 환율: 이 차이로 state_limits 되돌림 1회, +14초).

    실측(2026-09-28 gemini 유로): 반올림 값을 "근거에 없는 수치"로 잡아 state_limits 로 되돌렸고, 모델은 그 값을 빼며
    "오늘 환율은 찾을 수 없다"로 답을 뒤집었다. 정수부가 같고 차이가 반올림 오차 안이면 같은 값으로 본다.
    """
    m = re.fullmatch(r"(\d+)\.(\d+)", _digits(n).rstrip("%").strip())
    if not m:
        return False
    whole, dec = m.group(1), m.group(2)
    tol = 0.5 * 10 ** -len(dec) + 1e-9
    target = float(f"{whole}.{dec}")
    for h in _DEC_IN_TEXT_RE.finditer(haystack):
        if h.group(1) != whole:
            continue
        v = float(h.group(0))
        # 결과가 더 긴 소수면 반올림, 같거나 짧으면 끝자리 0 차이(-11.7 ↔ -11.70)만 같은 값
        if (len(h.group(2)) > len(dec) and abs(v - target) <= tol) or v == target:
            return True
    return False


def newest_period(results: list) -> Optional[str]:
    """웹 결과 전체에서 가장 최근 연·월(오늘 이하)과 그 출처 호스트 — "2026-09 (imarcgroup.com)".

    실측(2026-09-23): 원 페이지에 9월 표가 있는데 모델이 8월 보도자료를 따라 "9월치 미발표"라고 답했고 검증기가 통과시켰다.
    검증기는 결과 요약만 봐서 9월 표기가 있었는지 알 수 없다 — 코드가 전문에서 찾아 사실로 댄다.
    """
    from urllib.parse import urlsplit

    from agent_arch.services.web.periods import latest_period
    best: tuple[str, str] = ("", "")
    for r in results or []:
        if not isinstance(r, dict) or r.get("name") not in WEB_TOOL_NAMES:
            continue
        data = _loads(r.get("content"))
        if not isinstance(data, dict):
            continue
        items = _rows(data) or ([data] if data.get("final_url") else [])
        for it in items:
            if not isinstance(it, dict):
                continue
            p = it.get("latest_period") or latest_period(f"{it.get('title') or ''}\n{it.get('content') or it.get('text') or ''}")
            if p and p > best[0]:
                best = (p, urlsplit(str(it.get("final_url") or it.get("url") or "")).hostname or "")
    return f"{best[0]} ({best[1]})" if best[0] else None


def _facts_lines(f: dict) -> str:
    lines = [
        f"- 도구 호출 {f['tool_calls']}회: {', '.join(f['tools']) or '(없음)'}",
        f"- 도구 오류 {f['tool_errors']}회",
    ]
    if f["web_searches"] or f["context_reads"]:
        lines.append(f"- 웹 검색 {f['web_searches']}회, 페이지 본문 조각 조회 {f['context_reads']}회(빈 결과 {f['empty_searches']}회), "
                     f"결과 {f['results_total']}건 중 게시일 있는 것 {f['results_with_date']}건"
                     + (", 최신성 필터 사용" if f["freshness_requested"] else ", 최신성 필터 없음"))
    if f["answer_urls_not_in_results"]:
        lines.append(f"- 답에 적힌 URL 중 도구 결과에 없는 것: {f['answer_urls_not_in_results']}")
    if f.get("newest_period_in_results"):
        lines.append(f"- 도구 결과에 적힌 가장 최근 연·월: {f['newest_period_in_results']} "
                     "(초안이 이보다 이른 시점을 '최신·확정치'라고 하거나 이 시점 자료가 '없다·미발표'라고 하면 그 출처를 놓친 것)")
    if f.get("numbers_not_in_results"):
        lines.append(f"- 답의 수치 중 이번 턴 도구 결과·앞 대화 어디에도 없는 것: {f['numbers_not_in_results']} "
                     "(결과 값으로 계산·환산·반올림한 것이면 정상, 그렇지 않으면 근거 없는 수치)")
    if f.get("duplicate_calls") or f.get("budget_capped"):
        lines.append(f"- 예산 가드: 중복 검색 차단 {f.get('duplicate_calls', 0)}회, 상한 차단 {f.get('budget_capped', 0)}회"
                     " (모델이 같은 검색을 반복했거나 예산을 다 썼다 — 남은 근거로 답했는지 보라)")
    return "\n".join(lines)


# ─── 최근 대화(검증기 입력) ────────────────────────────────────────────────────
# react/history.recent_history 를 쓴다(라우터와 공유). 이번 턴 문장만 주자 후속 질의를 "모호"로 뒤집던 실측(2026-09-15).
from agent_arch.agents.react.history import recent_history  # noqa: E402,F401 — 테스트·호출자가 V.recent_history 로 참조


# ─── 프롬프트 ──────────────────────────────────────────────────────────────────

VERIFY_SYSTEM_PROMPT = """\
당신은 AI 어시스턴트 답변의 검증기다. 최근 대화, 사용자 요청, 답 초안, 이번 턴에 쓴 도구 요약, 코드가 계산한 증거
사실을 보고 초안을 그대로 보낼지(pass) 한 번 더 보강할지(revise) 정한다.

요청은 반드시 최근 대화와 함께 읽는다. "가격 관련해서", "그건 어때" 같은 후속 질의는 직전 대화의 대상을 이어받는
것이며, 초안이 대화 맥락으로 대상을 정한 것은 정상이다 — 이를 "임의 설정"이나 "모호"로 보지 마라.

revise 로 볼 것(하나라도 뚜렷하면):
1. 요청의 핵심(대상·기간·유형 조건 포함)에 답하지 않았다. 예: "최근 한 달 논문"인데 오래된 백과 항목을 냈다.
2. 초안의 주장·수치가 도구 결과에 없다(지어낸 근거).
3. 조사·동향·비교 요청인데 검색 1회, 본문 조각 조회 0회로 끝났고 결과가 요청과 어긋난다.
4. 결과가 요청과 무관한데 그것으로 채웠다(예: 산업용 브라켓을 못 찾고 산업용 배터리로 채움).
5. 도구 오류·빈 결과가 있었는데 다른 검색어로 다시 시도하지 않았다.
6. 대상이 여러 뜻으로 갈리고 최근 대화로도 정해지지 않는데(예: 첫 질문의 브라켓=치과/산업/TV) 되묻지 않고 섞어 답했다.
7. 증거 사실의 "도구 결과 전문 어디에도 없는 수치"가 계산·환산·반올림으로 설명되지 않는 핵심 수치다 → state_limits 로
   그 수치를 빼거나 결과에 있는 값으로 고치게 한다(detail 에 수치를 적는다).
8. 웹 검색 도구가 있었는데 검색 0회로 최신·외부 사실(시세·일정·결과·최근 사건)을 단정했다(예: "아직 열리지 않았다") → search.
9. 증거 사실의 "도구 결과에 적힌 가장 최근 연·월"보다 초안이 이른 시점을 최신이라 하거나, 그 시점 자료가 없다·미발표라고 했다
   → state_limits(detail 에 그 출처 호스트와 시점: "결과의 ○○(2026-09) 자료를 반영하라").
   또는 초안이 "아직 공개되지 않았다·발표되지 않았다·존재하지 않는다"처럼 **존재 자체**를 단정했는데 근거는 "검색 결과에 없다"뿐이다
   → state_limits(“확인한 자료는 ○월까지”처럼 결과 범위로 바꿔 말하게). 검색 인덱스는 최신 갱신을 늦게 반영한다
   (실측 2026-09-23: 원 페이지엔 9월 가격이 있었는데 인덱스 조각이 8월까지라 "9월치는 미공개"라고 단정).
10. 초안이 요청한 기간·항목의 값을 "확인되지 않는다·찾을 수 없다·자료가 없다"고 했는데 이번 턴에 web_fetch 를 한 번도 쓰지
    않았고, 도구 결과에 그 값이 있을 법한 페이지 URL(가격지수·공고·통계·공식 발표)이 있다 → fetch(detail 에 그 URL 1~2개).
    검색 조각은 인덱스 스냅샷이라 원 페이지의 최신 갱신이 빠져 있다.

11. 근거는 도구 결과에 이미 있는데 초안이 그와 어긋난다 — 결과에 있는 값·시점을 "없다·확인되지 않는다"고 했거나, 더 최신
    값이 결과에 있는데 오래된 값을 앞세웠다 → rewrite(detail 에 써야 할 값·시점·출처: "Wise 9/28 1,548.29원을 첫 문장에").
    새 검색이 필요 없으니 search·fetch 가 아니다.

12. 서로 다른 출처(조사기관)의 값끼리 변동률·증감을 계산해 추이로 제시했다(예: A 기관 8월 2.45 → B 기관 9월 2.76 "+12.6%").
    기관마다 기준이 달라 틀린 추세다 → rewrite(detail: "기관별로 나눠 각 출처가 밝힌 증감률만 쓰라").

앞 턴 근거: 최근 대화의 앞 답·앞 도구 결과에 이미 나온 값·URL 은 근거가 있는 것이다 — "이번 턴 도구 결과에 없다"고 되돌리지 마라.
수치 대조: 결과 값을 반올림·자릿수 줄임·단위 환산한 수치는 같은 값이다(예: 1,540.207344 → 1,540.21) — 근거 없는 수치로 보지 마라.
판단 근거는 도구 결과와 증거 사실뿐이다. 초안의 URL·도메인·사실을 네 기억으로 추측해 의심하지 마라(예: "공식 도메인은 아마
다른 주소일 것") — 결과에 있는 값과 맞으면 맞는 것이다.
pass 로 볼 것: 요청에 맞게 답했고 근거가 결과에 있다. 문장 표현·길이·형식은 판단 대상이 아니다.
사소한 결함은 pass. 의심스러우면 pass — 보강은 비용이 든다.

action 은 revise 일 때 **하나만**:
- search: 다른 검색어로 재검색(detail 에 검색어, 필요하면 영어)
- fetch: 결과에 나온 페이지를 web_fetch 로 지금 열어 확인(detail 에 URL — 조각에 최신 값이 없을 때)
- clarify: 사용자에게 한 줄 되묻기(detail 에 질문) — 최근 대화로도 대상이 정해지지 않을 때만. 맥락으로 대상이
  정해지는데 근거가 부족하면 clarify 가 아니라 search 다.
- state_limits: 근거 없는 주장·무관한 내용만 빼기(detail 에 무엇을 뺄지) — 근거가 있는 최신 값까지 빼게 하지 마라
- rewrite: 새 도구 없이 결과에 있는 근거대로 고쳐 쓰기(detail 에 써야 할 값·시점·출처)

출력은 JSON 하나만. 다른 텍스트 금지.
{"verdict": "pass"|"revise", "issues": ["짧은 문장", ...최대 3], "action": {"type": "search"|"fetch"|"clarify"|"state_limits"|"rewrite"|"none", "detail": "..."}}
"""


def _summarize_result(name: Any, content: str, terms: str = "") -> str:
    """도구 결과를 검증기가 읽을 수 있는 모양으로.

    dev 실측(2026-09-15): 결과를 앞 400자로 자르자 검색 결과 5건 중 1건도 안 보였고, 검증기가 스니펫에 있는 수치(PP 4.0%,
    PVC 유럽, 에틸렌 계약가)를 "도구 결과에 전혀 없음·지어냄"으로 오판해 2라운드 연속 revise 를 냈다. 웹 도구 결과는 행 단위
    (제목·URL·게시일·본문 앞부분)로 펼친다. 본문 조각(context)은 더 길게 보여 준다.
    """
    if content.lstrip().startswith("[ERROR]"):
        return content[:300]
    data = _loads(content)
    if name in WEB_TOOL_NAMES and isinstance(data, dict):
        if "error" in data:
            return f"오류 {data.get('error')}: {str(data.get('message') or '')[:200]}"
        if name == FETCH_TOOL and data.get("text") is not None:
            # 연 페이지는 행이 없다 — 예전엔 "결과 0건"으로 요약돼, 페이지에서 읽은 정답(smbs 공식 매매기준율)을 검증기가
            # "도구 결과에 없는 주장"으로 두 번 되돌렸다(2026-09-28).
            when = " | ".join(str(x) for x in (data.get("fetched_at") and f"열람 {data['fetched_at']}",
                                               data.get("latest_period") and f"최근표기 {data['latest_period']}") if x)
            return (f"페이지 {str(data.get('title') or '')[:80]} | {str(data.get('final_url') or data.get('url') or '')[:90]}"
                    + (f" | {when}" if when else "") + f"\n  {salient_excerpt(_unwrap_envelope(data.get('text')), _PAGE_BODY_CHARS, terms=terms)}")
        rows = _rows(data)
        if not rows:
            return f"결과 0건 ({data.get('note') or ''})"[:200]
        body_chars = _CONTEXT_BODY_CHARS if data.get("kind") == "context" else _SNIPPET_CHARS
        lines = [f"결과 {len(rows)}건" + (" (페이지 본문 조각 포함)" if data.get("kind") == "context" else "")]
        for r in rows[:_SEARCH_ROWS_SHOWN]:
            if not isinstance(r, dict):
                continue
            lines.append(f"  · {str(r.get('title') or '')[:80]} | {str(r.get('url') or '')[:90]} | "
                         f"{r.get('published') or '날짜없음'} | {salient_excerpt(_unwrap_envelope(r.get('content')), body_chars, terms=terms)}")
        return "\n".join(lines)

    if isinstance(data, (dict, list)):
        rows = _any_rows(data)
        if rows:
            total = data.get("total") if isinstance(data, dict) else None
            lines = [f"결과 {len(rows)}건" + (f" (총 {total}건)" if total else "")]
            # 행 밖의 단일 값(가이드 본문·대상 직원·문서 번호)도 근거다 — 행만 펼치면 이것이 사라진다
            top = [f"{k}: {' '.join(str(v).split())[:_ROW_BODY_CHARS]}" for k, v in data.items()
                   if isinstance(v, (str, int)) and not isinstance(v, bool) and v != ""] if isinstance(data, dict) else []
            if top:
                lines.append("  " + " | ".join(top)[:_FALLBACK_CHARS])
            lines += [_row_line(label, r) for label, r in rows[:_SEARCH_ROWS_SHOWN]]
            return "\n".join(lines)
        counted = isinstance(data, dict) and any(
            isinstance(data.get(k), int) and not isinstance(data.get(k), bool) and data[k] > 0 for k in _ROW_COUNT_KEYS)
        empty = data == [] or (isinstance(data, dict) and not counted
                               and any(isinstance(v, list) and not v for v in data.values()))
        if empty:
            head = {k: data.get(k) for k in ("count", "total", "message", "scope_error",
                                             "retrieval_status") if data.get(k) is not None} if isinstance(data, dict) else {}
            return f"결과 0건 {head}"[:300]

    return content[:_FALLBACK_CHARS]


def _tool_summary(state: dict) -> str:
    calls = state.get("collected_tool_calls") or []
    results = {r.get("tool_call_id"): r for r in (state.get("collected_tool_results") or []) if isinstance(r, dict)}
    parts = []
    used = 0
    # 연 페이지를 먼저 — 요약 상한에 걸리면 뒤쪽 호출이 잘리는데, 답의 근거는 대개 연 페이지다.
    calls = sorted((c for c in calls if isinstance(c, dict)), key=lambda c: c.get("name") != FETCH_TOOL)
    for c in calls:
        args = json.dumps(c.get("arguments") or {}, ensure_ascii=False)[:200]
        res = _summarize_result(c.get("name"), str((results.get(c.get("id")) or {}).get("content") or ""),
                                state.get("input") or "")
        line = f"- {c.get('name')}({args})\n  → {res}"
        if used + len(line) > _MAX_TOOL_SUMMARY_CHARS:
            parts.append("- …(이하 생략)")
            break
        parts.append(line)
        used += len(line)
    return "\n".join(parts) or "(도구 호출 없음)"


def build_messages(user_request: str, draft: str, state: dict, facts: dict) -> list[Message]:
    # 오늘 날짜를 준다 — "최근 한 달" 같은 조건 판정과 재검색어 제안(연도)이 모델의 학습 시점에 묶이지 않게.
    from datetime import datetime
    history = recent_history(state.get("messages"))
    body = (
        f"[현재 날짜] {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"[최근 대화 — 이번 요청 이전]\n{history or '(없음 — 첫 질문)'}\n\n"
        f"[사용자 요청 — 이번 턴]\n{(user_request or '').strip()[:1000]}\n\n"
        f"[답 초안]\n{(draft or '').strip()[:_MAX_DRAFT_CHARS]}\n\n"
        f"[이번 턴 도구 사용]\n{_tool_summary(state)}\n\n"
        f"[증거 사실 — 코드가 계산함]\n{_facts_lines(facts)}"
    )
    return [Message(role="system", content=VERIFY_SYSTEM_PROMPT), Message(role="user", content=body)]


# ─── LLM 호출·파싱 ──────────────────────────────────────────────────────────────

def verify_lane() -> dict:
    lane = dict((getattr(settings, "llm_providers", {}) or {}).get("verify") or {})
    return {**_LANE_DEFAULT, **lane}


def _load_providers() -> None:
    for mod in ("agent_arch.llm.vllm", "agent_arch.llm.gemini", "agent_arch.llm.openai_provider", "agent_arch.llm.anthropic_provider"):
        try:
            __import__(mod)
        except Exception:  # pragma: no cover
            pass


def provider_of(llm: Any) -> Optional[str]:
    """이 턴 LLM 인스턴스의 제공자 이름(vllm·gemini …). 모르면 None."""
    from agent_arch.llm.factory import LLMFactory
    from agent_arch.llm.pii import unwrap_llm
    _load_providers()
    llm = unwrap_llm(llm)   # 가명화 래퍼(PIIGuardLLM)가 씌워져 있으면 벗기고 판별
    for name, cls in LLMFactory._providers.items():
        if isinstance(cls, type) and type(llm) is cls:
            return name
    return None


def create_verify_llm(provider: Optional[str] = None):
    """검증기·충분성 판정 레인. 제공자 우선순위: 레인 설정(verify.provider) > 이 턴 제공자 > DEFAULT_LLM_PROVIDER.

    예전엔 늘 DEFAULT_LLM_PROVIDER(vllm)였다 — 사용자가 gemini 로 묻는 턴도 판정은 vllm 이 했고, vllm 이 내려가자
    (2026-09-28) 판정기·검증기가 타임아웃으로 조용히 pass 가 돼 gemini 턴도 검증 없이 나갔다. 답하는 모델과 같은 제공자를 쓴다.
    """
    from agent_arch.llm.factory import LLMFactory
    _load_providers()
    lane = verify_lane()
    overrides = {k: lane[k] for k in ("model", "max_tokens", "temperature", "top_p") if lane.get(k) is not None}
    return LLMFactory.create(lane.get("provider") or provider or settings.default_llm_provider, **overrides)


def parse_verdict(raw: str) -> dict:
    """관대한 파싱. 못 읽으면 pass(보강은 확실할 때만)."""
    from agent_arch.services.memory_v2.llm import extract_json
    data = extract_json(raw or "", "object")
    if not isinstance(data, dict):
        return {"verdict": "pass", "issues": [], "action": {"type": "none", "detail": ""}, "parse_error": True}
    verdict = str(data.get("verdict") or "pass").strip().lower()
    verdict = verdict if verdict in ("pass", "revise") else "pass"
    issues = [str(i)[:160] for i in (data.get("issues") or []) if str(i).strip()][:3] if isinstance(data.get("issues"), list) else []
    action = data.get("action") if isinstance(data.get("action"), dict) else {}
    a_type = str(action.get("type") or "none").strip().lower()
    a_type = a_type if a_type in _VALID_ACTIONS else "none"
    detail = str(action.get("detail") or "").strip()[:400]
    if verdict == "revise" and a_type == "none":
        a_type = "state_limits"
    return {"verdict": verdict, "issues": issues, "action": {"type": a_type, "detail": detail}}


async def run_verifier(user_request: str, draft: str, state: dict) -> tuple[dict, dict, Optional[dict]]:
    """(판정, 증거 사실, 토큰 사용량). LLM 실패는 pass 로 흡수한다 — 검증기가 죽어 답이 막히면 안 된다."""
    facts = collect_facts(state, draft)
    t0 = time.monotonic()
    llm = create_verify_llm(provider_of(state.get("llm")))
    resp = await llm.invoke(build_messages(user_request, draft, state, facts), tool_mode="NONE")
    verdict = parse_verdict(getattr(resp, "content", "") or "")
    verdict["latency_ms"] = int((time.monotonic() - t0) * 1000)
    verdict["model"] = str(getattr(llm, "model", None) or verify_lane().get("model") or "?")
    return verdict, facts, getattr(resp, "usage", None)


def feedback_message(verdict: dict) -> Message:
    """보강 지시 — 화면에 보이지 않는 턴 내부 메시지(name 으로 식별). role 은 user: gemini 는 system 을 앞으로 끌어올려
    system_instruction 에 합치고(위치 상실+캐시 파괴), Gemma 템플릿은 중간 system 을 받지 않는다. 도구 결과 재주입과 같은 관례."""
    action = verdict.get("action") or {}
    a_type, detail = action.get("type", "none"), action.get("detail", "")
    what = {
        "search": f"다른 검색어로 다시 검색하라: {detail}",
        "fetch": f"web_fetch 로 다음 페이지를 지금 열어 요청한 값을 확인하라(검색 조각은 오래된 스냅샷일 수 있다): {detail}",
        "clarify": f"실행 대신 사용자에게 한 줄로 되물어라: {detail}",
        "state_limits": f"근거 없는 부분·무관한 내용만 빼거나 결과에 있는 값으로 고쳐라(근거가 있는 최신 값은 그대로 둔다): {detail}",
        "rewrite": f"새 도구를 쓰지 말고, 이미 받은 도구 결과대로 답을 고쳐 써라: {detail}",
    }.get(a_type, "부족한 부분을 도구로 보강하라.")
    issues = "\n".join(f"- {i}" for i in (verdict.get("issues") or [])) or "- (사유 없음)"
    return Message(
        role="user", name=FEEDBACK_NAME,
        content=("[답변 검증 — 사용자에게 보이지 않는 메시지] 위 초안은 요청을 충분히 충족하지 못했다.\n"
                 f"부족한 점:\n{issues}\n다음 행동: {what}\n"
                 "행동을 수행한 뒤 최종 답을 다시 작성하라. 도구 결과에 없는 내용은 넣지 말고, 답은 확인된 가장 최신 값으로 "
                 "시작하라. 확인하지 못한 것만 끝에 한 줄로 적어라. 이 메시지 자체를 언급하지 마라."),
    )
