"""브레이브 검색 MCP 서버 응답 → 모델·출처 각주·검증기가 같은 모양으로 읽는 **닫힌 결과 형식**.

MCP 서버(brave/brave-search-mcp-server)는 도구마다 응답 모양이 다르다(2026-09-15 실측):
- brave_web_search / brave_news_search: content 텍스트 항목 N개, 각각 결과 1건의 JSON 문자열(`{url,title,description,…}`).
  웹 검색은 결과 필터가 비면 FAQ·토론·뉴스·영상 항목이 섞여 들어오고 키가 다르다. 구조화 응답은 없다. 결과가 없으면
  isError 에 "No web results found".
- brave_llm_context: 구조화 응답 `{grounding:{generic:[{url,title,snippets[]}],map,poi}, sources:{url:{title,hostname,age[],…}}}`.
  스니펫은 평문·마크다운·JSON 문자열이 섞인다.

여기서 전부 한 모양으로 접는다: `{"kind": "results"|"context", "results": [{"title","url","published","content"}...], ...}`.
- title 은 정화한 평문(짧고 각주·판정에 쓴다), 본문(description/스니펫)은 **신뢰 불가 봉투** 안에 넣는다(services/web/envelope).
- 예산: 행 수·행당 본문·전체 본문 상한. 초과분은 잘라 내고 `truncated` 로 알린다.
"""
from __future__ import annotations

import html
import json
import re
from typing import Any, Iterable, Optional

from agent_arch.services.web.periods import annotate_timezones, latest_period
from agent_arch.services.web.envelope import sanitize_external_text, wrap_web_content as wrap_external_content

KIND_RESULTS = "results"
KIND_CONTEXT = "context"
CONTEXT_TOOL = "brave_llm_context"
SEARCH_TOOLS = frozenset({"brave_web_search", "brave_news_search"})
BRAVE_TOOLS = SEARCH_TOOLS | {CONTEXT_TOOL}
# 페이지 실시간 열기(app/agents/tools/web_fetch). 검색 인덱스 조각은 갱신이 늦어 가격지수 페이지의 9월치를 8월까지로 보여 줬다
# (2026-09-23) — 원 페이지를 직접 읽는 도구. 예산 가드·검증기·각주·이름 가드가 웹 호출로 함께 센다.
FETCH_TOOL = "web_fetch"
WEB_TOOLS = BRAVE_TOOLS | {FETCH_TOOL}

MAX_ROWS = 10
MAX_TITLE_CHARS = 200
MAX_DESC_CHARS = 700
MAX_CONTEXT_ROWS = 10
MAX_SNIPPET_CHARS = 2_000
MAX_CONTEXT_CHARS = 24_000
_NO_RESULTS_TEXT = "no web results found"
_TAG_RE = re.compile(r"</?(?:strong|em|b|i)>", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})")


def _plain(text: Any, limit: int) -> str:
    """제목 등 짧은 평문 — 태그·엔티티 제거 후 정화(특수 토큰·봉투 마커)."""
    s = html.unescape(_TAG_RE.sub("", str(text or "")))
    return sanitize_external_text(re.sub(r"\s+", " ", s))[:limit].strip()


def _http(url: Any) -> Optional[str]:
    return url if isinstance(url, str) and url.startswith(("http://", "https://")) and len(url) <= 2048 else None


def _published(entry: dict) -> Optional[str]:
    for k in ("page_age", "published", "date"):
        v = entry.get(k)
        if isinstance(v, str):
            m = _ISO_DATE_RE.search(v)
            if m:
                return m.group(1)
    return None


def _row_type(entry: dict) -> str:
    if "question" in entry and "answer" in entry:
        return "faq"
    if "page_age" in entry or "breaking" in entry:
        return "news"
    if "video" in entry or "duration" in entry:
        return "video"
    return "web"


def _search_row(entry: dict, source: str) -> Optional[dict]:
    url = _http(entry.get("url"))
    if not url:
        return None
    title = _plain(entry.get("title") or entry.get("question") or "", MAX_TITLE_CHARS)
    body_parts = [entry.get("description") or entry.get("answer") or ""]
    extra = entry.get("extra_snippets")
    if isinstance(extra, list):
        body_parts.extend(str(x) for x in extra[:3])
    body = _plain("\n".join(p for p in body_parts if p), MAX_DESC_CHARS)
    row = {"title": title, "url": url, "type": _row_type(entry)}
    pub = _published(entry)
    if pub:
        row["published"] = pub
    if entry.get("age") and isinstance(entry.get("age"), str):
        row["age"] = entry["age"][:40]
    period = latest_period(f"{title}\n{body}")
    if period:
        row["latest_period"] = period      # 본문에 적힌 가장 최근 연·월(services/web/periods) — 최신 출처 고르기용
        _drop_stale_published(row)
    row["content"] = wrap_external_content(annotate_timezones(body), source=source) if body else ""
    return row


def _drop_stale_published(row: dict) -> None:
    """게시일이 본문에 적힌 기간보다 이르면 뺀다 — 계속 갱신되는 가격지수 페이지의 게시일은 처음 만든 날이다.

    실측(2026-09-28 EPDM): Business Analytiq 행이 published 2025-11-19 · 본문 "September 2026" 이었고, gemini 가
    "2025-11-19 게시, 2026년 9월 데이터"라고 적었다(검증기·vllm 은 같은 날짜로 9월 자료를 오래됐다고 봤다). 기간은 latest_period 가 말한다.
    """
    pub = str(row.get("published") or "")
    if pub[:7] and row.get("latest_period") and pub[:7] < row["latest_period"]:
        row.pop("published", None)
        row.pop("age", None)


def parse_text_items(texts: Iterable[str]) -> list[dict]:
    """텍스트 항목마다 JSON 한 건. JSON 이 아니면 버린다(알림 문장 등)."""
    out = []
    for t in texts or []:
        try:
            d = json.loads(t)
        except Exception:
            continue
        if isinstance(d, dict):
            out.append(d)
        elif isinstance(d, list):
            out.extend(x for x in d if isinstance(x, dict))
    return out


_CREDIT_MARKERS = ("402 payment required", "credit_exhausted")
_RATE_MARKERS = ("429 too many requests", "rate_limited", "rate limit")


def provider_error(tool: str, query: str, texts: list[str], fallback: str) -> dict:
    """제공자 오류를 원인별 코드로. 실측(2026-09-23): 선불 크레딧 소진(402 CREDIT_EXHAUSTED)이 일반 provider_error 로 뭉개져
    모델은 "연결이 원활하지 않다, 잠시 후 다시"라고 답했고 운영에는 신호가 없었다 — 재시도로 풀리지 않는 오류다."""
    raw = " ".join(texts or [])
    low = raw.lower()
    if any(m in low for m in _CREDIT_MARKERS):
        return {"error": "provider_credit_exhausted", "tool": tool, "query": query,
                "message": "웹 검색 서비스의 이용 크레딧이 소진돼 지금은 검색할 수 없습니다(재시도로 풀리지 않음). "
                           "사용자에게 웹 검색이 일시 중단됐고 관리자 조치가 필요하다고 안내하세요."}
    if any(m in low for m in _RATE_MARKERS):
        return {"error": "provider_rate_limited", "tool": tool, "query": query,
                "message": "웹 검색 요청이 제공자 속도 제한에 걸렸습니다. 같은 검색을 되풀이하지 말고 이미 받은 결과로 답하세요."}
    return {"error": "provider_error", "tool": tool, "query": query,
            "message": sanitize_external_text(raw)[:500] or fallback}


def normalize_search(tool: str, query: str, texts: list[str], *, is_error: bool = False) -> dict:
    """brave_web_search / brave_news_search 응답 → results 형식. 결과 없음은 오류가 아니라 0건이다(모델이 검색어를 바꿔 다시 시도)."""
    joined = " ".join(texts or []).strip().lower()
    if is_error and _NO_RESULTS_TEXT in joined:
        return {"kind": KIND_RESULTS, "tool": tool, "query": query, "count": 0, "results": [],
                "note": "결과 없음 — 검색어를 바꾸거나(영어·핵심어) 다른 도구를 써 보세요."}
    if is_error:
        return provider_error(tool, query, texts, "검색 제공자 오류")
    rows = []
    seen: set[str] = set()
    for entry in parse_text_items(texts):
        row = _search_row(entry, source=tool)
        if row and row["url"] not in seen:
            seen.add(row["url"])
            rows.append(row)
    truncated = len(rows) > MAX_ROWS
    rows = rows[:MAX_ROWS]
    out = {"kind": KIND_RESULTS, "tool": tool, "query": query, "count": len(rows), "results": rows,
           "externalContent": {"untrusted": True, "wrapped": True}}
    if truncated:
        out["truncated"] = True
    if not rows:
        out["note"] = "결과 없음 — 검색어를 바꾸거나(영어·핵심어) 다른 도구를 써 보세요."
    return out


def normalize_context(query: str, structured: Any, texts: list[str], *, is_error: bool = False) -> dict:
    """brave_llm_context 응답 → context 형식. 페이지 본문 조각이 들어 있어 '본문 읽기'에 해당한다."""
    data = structured
    if not isinstance(data, dict) and texts:
        try:
            data = json.loads(texts[0])
        except Exception:
            data = None
    if is_error or not isinstance(data, dict):
        return provider_error(CONTEXT_TOOL, query, texts, "본문 조각 제공자 오류")
    grounding = data.get("grounding") if isinstance(data.get("grounding"), dict) else {}
    sources = data.get("sources") if isinstance(data.get("sources"), dict) else {}
    items = [x for x in (grounding.get("generic") or []) if isinstance(x, dict)]
    rows, total, truncated = [], 0, False
    for item in items:
        url = _http(item.get("url"))
        if not url:
            continue
        meta = sources.get(url) if isinstance(sources.get(url), dict) else {}
        snippets = []
        for s in item.get("snippets") or []:
            if not isinstance(s, str):
                continue
            clean = _plain(s, MAX_SNIPPET_CHARS) if len(s) <= MAX_SNIPPET_CHARS else _plain(s[:MAX_SNIPPET_CHARS], MAX_SNIPPET_CHARS)
            if not clean:
                continue
            if total + len(clean) > MAX_CONTEXT_CHARS:
                truncated = True
                break
            snippets.append(clean)
            total += len(clean)
        row = {"title": _plain(item.get("title") or meta.get("title") or "", MAX_TITLE_CHARS), "url": url,
               "hostname": str(meta.get("hostname") or "")[:100]}
        pub = None
        for a in meta.get("age") or []:
            if isinstance(a, str):
                m = _ISO_DATE_RE.search(a)
                if m:
                    pub = m.group(1)
                    break
        if pub:
            row["published"] = pub
        period = latest_period(row["title"] + "\n" + "\n".join(snippets))
        if period:
            row["latest_period"] = period
            _drop_stale_published(row)
        row["content"] = wrap_external_content(annotate_timezones("\n\n".join(snippets)), source=CONTEXT_TOOL) if snippets else ""
        rows.append(row)
        if truncated or len(rows) >= MAX_CONTEXT_ROWS:
            break
    out = {"kind": KIND_CONTEXT, "tool": CONTEXT_TOOL, "query": query, "count": len(rows), "results": rows,
           "chars": total, "externalContent": {"untrusted": True, "wrapped": True}}
    if truncated or len(items) > len(rows):
        out["truncated"] = True
    if not rows:
        out["note"] = "본문 조각 없음 — 검색어를 바꾸거나 brave_web_search 로 링크부터 찾아 보세요."
    else:
        # 결과 자리에 둔다 — 시스템 지침에 같은 말이 두 곳 있었는데도 gemini 는 web_fetch 를 한 번도 안 불렀다(2026-09-23,
        # EPDM·카본블랙 "9월 자료 없음"). 모델이 바로 앞에 읽는 도구 결과가 가장 잘 따르는 자리다.
        out["freshness_note"] = FRESHNESS_NOTE
    return out


FRESHNESS_NOTE = ("이 조각은 검색 인덱스의 스냅샷이라 페이지 최신 갱신이 빠져 있을 수 있다(각 행 published·latest_period 참고 — "
                  "latest_period 는 본문에 적힌 가장 최근 연·월이다. 출처끼리 다르면 더 최근 것을 우선 근거로 삼아라). "
                  "요청한 기간·항목의 값이 조각에 없으면 '없다'고 답하기 전에 그 값이 있을 법한 url 1~2개를 "
                  "web_fetch 로 열어 지금 페이지에서 확인하라.")


def normalize(tool: str, query: str, *, structured: Any, texts: list[str], is_error: bool) -> dict:
    if tool == CONTEXT_TOOL:
        return normalize_context(query, structured, texts, is_error=is_error)
    return normalize_search(tool, query, texts, is_error=is_error)
