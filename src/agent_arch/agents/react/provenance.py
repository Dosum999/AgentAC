"""답변의 출처를 **데이터로** 남긴다 — 모델의 기억력에 맡기지 않는다.

문제(실측 2026-09-15): 웹 검색으로 답한 턴에서 모델이 URL 을 적지 않았고, 합성기는 워커 문장을
다시 쓰며 출처를 떨어뜨렸다. 도구 결과는 한 턴만 이월되므로 두 턴 뒤엔 근거가 아무 데도 없었고,
"근거가 뭐야"에 모델은 **사내 메신저에서 찾았다고 출처를 지어냈다.** 웹 정보가 사내 자료로 둔갑하는
답이라 반드시 막아야 한다.

해법: 턴을 끝내는 코드 지점(react 그래프 최종 결과, 합성기 병합)에서 이번 턴의 도구 결과에서 웹 출처를
뽑아 답변 끝에 결정적으로 각주로 붙인다. 답변 본문에 들어가므로 대화 기록에 남고, 뒤 턴이 "웹 검색
결과였다"를 URL 과 함께 본다. 모델이 이미 같은 URL 을 적었으면 중복해서 붙이지 않는다.

각주는 답이 **참조한** 출처만 — 참조 판정은 (1) URL 이 본문에 있음, (2) 제목·호스트명 언급, (3) 답의 특징적 숫자
(가격·비율·수치)가 그 출처의 본문 조각에 있음. 셋 다 없으면 마지막 웹 도구 호출의 결과를 폴백으로 붙인다. 참조한 것이
있으면 참조 안 된 결과는 붙이지 않는다(버린 결과는 근거가 아니다).
실측(2026-09-15): 도구 순서대로 5개를 붙이자 첫 검색의 무관한 결과(코인·3D 프린팅)가 각주를 채우고 다음 턴 오해를 낳았다.
재실측(같은 날, 브레이브 MCP): brave_llm_context 결과 전체를 "읽은 페이지"로 쳐서 붙였더니 모델이 인라인 링크로 쓴 5건은
이미 본문에 있어 빠지고, 첫 한국어 질의의 무관한 6건(폴리매쉬 코인·폴리어학원·페로실리콘)이 각주가 됐다. llm_context 의
행은 검색 결과일 뿐 모델이 골라 읽은 페이지가 아니다 — "읽음" 계층은 틀린 개념이라 없앴다.
벤치(2026-09-23, 30문항): 모델이 본문에 결과 URL 을 평균 9.4개 인라인으로 달았는데도 각주가 4.9개를 더 붙였고, 그중 날짜 있는
항목의 38% 가 2026 이전(무관·옛 페이지)이었다 — 심판 최다 지적이 "본문 인용과 출처 목록 불일치". 원인은 호스트 판정이
본문 **URL 안의** 호스트명(yna.co.kr)에 걸려 같은 매체의 다른 결과를 전부 "참조"로 친 것. 그래서 (a) 모델이 결과 URL 을 하나라도
인라인으로 인용했으면 인용은 모델 몫이라 각주를 붙이지 않고, (b) 제목·호스트·숫자 판정은 URL 을 걷어낸 본문으로 한다.

웹 도구 결과의 모양은 services/web/brave_output 이 닫아 둔다: `{"kind": "results"|"context", "results": [{"title","url",
"published","content"}...]}`. 여기는 그 모양만 읽는다.
"""
from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from agent_arch.services.web.brave_output import CONTEXT_TOOL, FETCH_TOOL, KIND_CONTEXT, WEB_TOOLS  # noqa: F401 — 외부에서 이 이름으로 참조

MAX_SOURCES = 5
FOOTER_HEADER = "출처(웹):"

_ENVELOPE_BODY_RE = re.compile(r"---\n(.*?)\n<<<END_EXTERNAL_UNTRUSTED_CONTENT", re.S)
_TITLE_SPLIT_RE = re.compile(r"\s+[-|<>—–:]\s+|\s*\|\s*")
_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_MIN_TOKEN_HIT = 0.6
# 답의 특징적 숫자: 소수·천 단위 구분·백분율·세 자리 이상 정수. 연도(19xx/20xx)는 흔해서 뺀다.
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|\d+\.\d+%?|\d{3,}%?|\d{1,2}%")
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_BODY_KEEP = 8_000
_MIN_NUMBER_SCORE = 3


def _unwrap(text: Any) -> str:
    """봉투로 감싼 본문에서 내용만. 봉투가 없으면 그대로."""
    s = text if isinstance(text, str) else ""
    m = _ENVELOPE_BODY_RE.search(s)
    return (m.group(1) if m else s).strip()


def _http(url: Any) -> Optional[str]:
    return url if isinstance(url, str) and url.startswith(("http://", "https://")) else None


def _rows(data: dict) -> list:
    rows = data.get("results")
    if isinstance(rows, dict) and isinstance(rows.get("items"), list):   # truncate_result 가 20건 넘게 자른 모양
        return rows["items"]
    return rows if isinstance(rows, list) else []


def extract_sources(tool_results: Optional[Iterable[Any]]) -> list[dict]:
    """이번 턴 도구 결과에서 웹 출처를 뽑는다. [{"tool","url","title","site","published","batch","body"}], URL 기준 중복 제거.

    batch 는 그 URL 이 마지막으로 나온 웹 도구 호출 번호(0부터) — 마지막 호출만 고르는 폴백에 쓴다.
    body 는 참조 판정용 본문(봉투 벗긴 소문자, 앞 8천 자) — 각주에는 싣지 않는다.
    """
    out: list[dict] = []
    by_url: dict[str, dict] = {}
    batch = -1
    for tr in tool_results or []:
        name = tr.get("name") if isinstance(tr, dict) else getattr(tr, "name", None)
        content = tr.get("content") if isinstance(tr, dict) else getattr(tr, "content", None)
        if name not in WEB_TOOLS or not isinstance(content, str) or content.lstrip().startswith("[ERROR]"):
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue
        if not isinstance(data, dict) or "error" in data:
            continue
        batch += 1
        rows = _rows(data)
        if name == FETCH_TOOL and _http(data.get("final_url") or data.get("url")):
            # web_fetch 는 행이 아니라 한 페이지 — 실제로 읽은 최종 주소를 출처로.
            rows = [{"url": data.get("final_url") or data.get("url"), "title": data.get("title"), "content": data.get("text")}]
        for r in rows:
            if not isinstance(r, dict):
                continue
            url = _http(r.get("url"))
            if not url:
                continue
            body = _unwrap(r.get("content") or r.get("description") or "").lower()[:_BODY_KEEP]
            if url in by_url:
                by_url[url]["batch"] = batch                     # 마지막으로 나온 호출 기준(폴백 = "마지막 호출의 결과")
                if body and body not in by_url[url]["body"]:
                    by_url[url]["body"] = (by_url[url]["body"] + "\n" + body)[:_BODY_KEEP]
                continue
            src = {"tool": name, "url": url, "title": _unwrap(r.get("title"))[:80], "site": (urlsplit(url).hostname or ""),
                   "published": r.get("published") if isinstance(r.get("published"), str) else None,
                   "batch": batch, "body": body}
            by_url[url] = src
            out.append(src)
    return out


# ─── 참조 판정 ─────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _title_core(title: str) -> str:
    """'기사 제목 - 매체명' 꼴에서 제목 부분(가장 긴 조각)."""
    parts = [x.strip() for x in _TITLE_SPLIT_RE.split(title or "") if x and x.strip()]
    return max(parts, key=len) if parts else (title or "")


def answer_numbers(content: str) -> list[str]:
    """답에 적힌 특징적 숫자(연도 제외, 중복 제거). 근거 페이지 본문과 맞춰 보는 지문이다."""
    seen: list[str] = []
    for n in _NUM_RE.findall(content or ""):
        if _YEAR_RE.match(n) or n in seen:
            continue
        seen.append(n)
    return seen


def number_score(source: dict, numbers: list[str]) -> int:
    """출처 본문에 답의 숫자가 얼마나 있나. 소수·천 단위·백분율은 2점(우연히 겹치기 어렵다), 그 외 정수는 1점."""
    body = source.get("body") or ""
    if not body or not numbers:
        return 0
    score = 0
    for n in numbers:
        if n.lower() in body:
            score += 2 if any(ch in n for ch in ".,%") else 1
    return score


def is_referenced(source: dict, content_norm: str, numbers: Optional[list[str]] = None) -> bool:
    """답 본문이 이 출처를 참조했는가 — URL 포함, 제목(핵심부) 포함, 제목 토큰 다수 포함, 호스트명 언급, 또는 답의 숫자가 본문에 있음."""
    if not content_norm:
        return False
    url = (source.get("url") or "").lower()
    if url and url in content_norm:
        return True
    core = _norm(_title_core(source.get("title") or ""))
    if len(core) >= 10 and core in content_norm:
        return True
    toks = _TOKEN_RE.findall(core)
    if len(toks) >= 3 and sum(t in content_norm for t in toks) / len(toks) >= _MIN_TOKEN_HIT:
        return True
    site = (source.get("site") or "").lower().removeprefix("www.")
    if len(site) >= 6 and site in content_norm:
        return True
    return number_score(source, numbers or []) >= _MIN_NUMBER_SCORE


_URL_IN_TEXT_RE = re.compile(r"https?://[^\s)\]>\"'`]+")


def cites_inline(content: Optional[str], sources: list[dict]) -> bool:
    """답 본문이 이번 턴 웹 결과의 URL 을 하나라도 그대로 적었나(= 모델이 인용을 직접 했다)."""
    body = content or ""
    return any(s.get("url") and s["url"] in body for s in sources)


def select_footer_sources(content: Optional[str], sources: list[dict]) -> list[dict]:
    """각주에 올릴 출처 — 답이 참조한 것(URL·제목·호스트·숫자 일치). 하나도 없으면 마지막 웹 도구 호출의 결과.

    모델이 무엇이든 참조했다면 참조 안 된 검색 결과는 붙이지 않는다. 그건 근거가 아니라 버린 결과다.
    제목·호스트·숫자는 URL 을 걷어낸 본문에서 본다 — URL 속 호스트명·경로 숫자가 판정을 오염시킨다.
    """
    plain = _URL_IN_TEXT_RE.sub(" ", content or "")
    body_norm = _norm(plain)
    numbers = answer_numbers(plain)
    chosen = [s for s in sources if is_referenced(s, body_norm, numbers)]
    if not chosen:
        last = max((s["batch"] for s in sources if s.get("batch") is not None), default=None)
        chosen = [s for s in sources if s.get("batch") == last] if last is not None else []
    return chosen


def sources_footer(content: Optional[str], tool_results: Optional[Iterable[Any]]) -> str:
    """답에 아직 없는 웹 출처만 각주로(답이 참조한 것 우선). 붙일 게 없으면 빈 문자열."""
    sources = extract_sources(tool_results)
    if not sources:
        return ""
    body = content or ""
    if cites_inline(body, sources):
        return ""
    missing = [s for s in select_footer_sources(body, sources) if s["url"] not in body]
    if not missing:
        return ""
    lines = ["", "", FOOTER_HEADER]
    for s in missing[:MAX_SOURCES]:
        label = s["title"] or s["site"] or s["url"]
        date = f" ({s['published']})" if s.get("published") else ""
        lines.append(f"- {label}{date} — {s['url']}")
    return "\n".join(lines)


def append_sources_footer(content: Optional[str], tool_results: Optional[Iterable[Any]]) -> str:
    return (content or "") + sources_footer(content, tool_results)


def apply_sources_footer(result: Any) -> str:
    """LoopResult 의 content 에 각주를 붙이고(제자리) 붙인 각주를 돌려준다. 없으면 빈 문자열."""
    footer = sources_footer(getattr(result, "content", ""), getattr(result, "tool_results", None))
    if footer:
        result.content = (result.content or "") + footer
    return footer
