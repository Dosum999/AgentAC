"""HTML → 읽을 수 있는 본문(마크다운/텍스트).

세 단계: (1) 보이지 않는 요소 제거 — script/style 뿐 아니라 CSS·클래스·속성으로 숨긴 텍스트까지
(숨은 텍스트는 간접 프롬프트 주입의 단골 매체다), (2) 본문 블록 선택 — article/main/role=main 을
우선하고 없으면 단락 텍스트 밀도가 가장 높은 컨테이너, (3) 마크다운 변환 — 제목·목록·링크·코드·
인용·표만 살리고 장식 태그는 평문으로. 재귀 깊이는 상한을 두어 악성 중첩에서 시간이 폭주하지 않게 한다.
"""
from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup, Comment, NavigableString, Tag

try:  # lxml 이 있으면 빠르고 관대한 파서, 없으면 표준 html.parser (컨테이너 이미지에 lxml 이 빠져 있던 실측 2026-09-15)
    import lxml  # noqa: F401
    _PARSER = "lxml"
except Exception:  # pragma: no cover — 환경 의존
    _PARSER = "html.parser"


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, _PARSER)
    except Exception:
        return BeautifulSoup(html, "html.parser")

MAX_HTML_CHARS = 1_000_000
MAX_DEPTH = 64

_DROP_TAGS = ("script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed", "video", "audio", "map")
_BOILERPLATE_TAGS = ("nav", "footer", "header", "aside", "form", "button", "dialog", "menu")
_HIDDEN_CLASSES = frozenset({"sr-only", "visually-hidden", "d-none", "hidden", "invisible", "screen-reader-only", "offscreen"})
_HIDDEN_STYLE_PATTERNS = (
    (re.compile(r"(?:^|;)\s*display\s*:\s*([^;]+)", re.I), re.compile(r"^\s*none\s*$", re.I)),
    (re.compile(r"(?:^|;)\s*visibility\s*:\s*([^;]+)", re.I), re.compile(r"^\s*hidden\s*$", re.I)),
    (re.compile(r"(?:^|;)\s*opacity\s*:\s*([^;]+)", re.I), re.compile(r"^\s*0(?:\.0+)?\s*$")),
    (re.compile(r"(?:^|;)\s*font-size\s*:\s*([^;]+)", re.I), re.compile(r"^\s*0(px|em|rem|pt|%)?\s*$", re.I)),
    (re.compile(r"(?:^|;)\s*text-indent\s*:\s*([^;]+)", re.I), re.compile(r"^\s*-\d{4,}px\s*$")),
    (re.compile(r"(?:^|;)\s*color\s*:\s*([^;]+)", re.I), re.compile(r"^\s*transparent\s*$", re.I)),
    (re.compile(r"(?:^|;)\s*color\s*:\s*([^;]+)", re.I), re.compile(r"^\s*rgba\s*\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*0(?:\.0+)?\s*\)\s*$", re.I)),
    (re.compile(r"(?:^|;)\s*color\s*:\s*([^;]+)", re.I), re.compile(r"^\s*hsla\s*\(\s*[\d.]+\s*,\s*[\d.]+%?\s*,\s*[\d.]+%?\s*,\s*0(?:\.0+)?\s*\)\s*$", re.I)),
    (re.compile(r"(?:^|;)\s*transform\s*:\s*([^;]+)", re.I), re.compile(r"scale\s*\(\s*0\s*\)", re.I)),
)
_INVISIBLE_UNICODE_RE = re.compile("[​-‏ -‮⁠-⁯﻿­]")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_NL_RE = re.compile(r"\n{3,}")


def _style_hidden(style: str) -> bool:
    for prop_re, val_re in _HIDDEN_STYLE_PATTERNS:
        m = prop_re.search(style or "")
        if m and val_re.search(m.group(1)):
            return True
    return False


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden") or str(tag.get("aria-hidden", "")).lower() == "true":
        return True
    if tag.name == "input" and str(tag.get("type", "")).lower() == "hidden":
        return True
    classes = tag.get("class") or []
    if isinstance(classes, str):
        classes = classes.split()
    if any(c.lower() in _HIDDEN_CLASSES for c in classes):
        return True
    return _style_hidden(str(tag.get("style") or ""))


def strip_hidden(soup: BeautifulSoup) -> None:
    for t in soup.find_all(_DROP_TAGS):
        t.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    for t in list(soup.find_all(True)):
        if getattr(t, "decomposed", False):
            continue
        try:
            if _is_hidden(t):
                t.decompose()
        except Exception:
            continue


def select_main(soup: BeautifulSoup) -> Tag:
    for sel in ("article", "main", "[role=main]"):
        node = soup.select_one(sel)
        if node is not None and len(node.get_text(" ", strip=True)) >= 80:
            return node
    body = soup.body or soup
    best, best_score = body, 0
    for cand in body.find_all(["div", "section", "td", "li"]):
        score = sum(len(p.get_text(" ", strip=True)) for p in cand.find_all("p", recursive=True))
        if score > best_score:
            best, best_score = cand, score
    return best if best_score >= 200 else body


def _strip_boilerplate(node: Tag) -> None:
    for t in node.find_all(_BOILERPLATE_TAGS):
        t.decompose()


def _text(s: str) -> str:
    return _WS_RE.sub(" ", s)


def to_markdown(node: Tag) -> str:
    out: list[str] = []

    def walk(n, depth: int, list_ctx: Optional[tuple[str, int]] = None) -> None:
        if depth > MAX_DEPTH:
            return
        if isinstance(n, NavigableString):
            if not isinstance(n, Comment):
                out.append(_text(str(n)))
            return
        if not isinstance(n, Tag):
            return
        name = n.name.lower()
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            out.append("\n\n" + "#" * int(name[1]) + " " + n.get_text(" ", strip=True) + "\n\n")
            return
        if name == "p":
            out.append("\n\n")
            for c in n.children:
                walk(c, depth + 1, list_ctx)
            out.append("\n\n")
            return
        if name == "br":
            out.append("\n")
            return
        if name in ("ul", "ol"):
            out.append("\n")
            idx = 0
            for li in n.find_all("li", recursive=False):
                idx += 1
                marker = f"{idx}. " if name == "ol" else "- "
                out.append("\n" + marker)
                for c in li.children:
                    walk(c, depth + 1, (name, idx))
            out.append("\n")
            return
        if name == "a":
            label = n.get_text(" ", strip=True)
            href = str(n.get("href") or "").strip()
            if label and href.lower().startswith(("http://", "https://")):
                out.append(f"[{label}]({href})")
            else:
                out.append(label)
            return
        if name == "img":
            alt = str(n.get("alt") or "").strip()
            if alt:
                out.append(f"[이미지: {alt}]")
            return
        if name == "pre":
            out.append("\n\n```\n" + n.get_text() + "\n```\n\n")
            return
        if name == "code":
            out.append("`" + n.get_text() + "`")
            return
        if name == "blockquote":
            inner = n.get_text("\n", strip=True)
            out.append("\n\n" + "\n".join("> " + ln for ln in inner.splitlines()) + "\n\n")
            return
        if name == "table":
            out.append("\n\n")
            for tr in n.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if any(cells):
                    out.append(" | ".join(cells) + "\n")
            out.append("\n")
            return
        if name in ("hr",):
            out.append("\n\n---\n\n")
            return
        block = name in ("div", "section", "article", "main", "li", "tr", "dd", "dt", "dl", "figure", "figcaption", "summary", "details")
        if block:
            out.append("\n")
        for c in n.children:
            walk(c, depth + 1, list_ctx)
        if block:
            out.append("\n")

    walk(node, 0)
    text = "".join(out)
    text = "\n".join(ln.rstrip() for ln in text.splitlines())
    return _NL_RE.sub("\n\n", text).strip()


_INLINE_TAGS = ("strong", "em", "b", "i", "u", "span", "a", "code", "mark", "small", "sub", "sup", "abbr", "time", "font", "label")


def to_text(node: Tag) -> str:
    # 인라인 태그를 먼저 풀어야 "환율이 <strong>1,344원</strong>에" 가 줄바꿈으로 쪼개지지 않는다.
    for t in node.find_all(_INLINE_TAGS):
        t.unwrap()
    node.smooth()
    text = node.get_text("\n")
    text = "\n".join(_text(ln).strip() for ln in text.splitlines())
    return _NL_RE.sub("\n\n", text).strip()


_LD_KEYS = ("name", "price", "priceCurrency", "lowPrice", "highPrice", "priceValidUntil", "dateModified", "datePublished", "uploadDate")
_META_KEYS = ("product:price:amount", "product:price:currency", "og:price:amount", "og:price:currency",
              "article:published_time", "article:modified_time", "og:updated_time")
_MAX_FACTS_CHARS = 600


def _ld_fields(node, out: list, depth: int = 0) -> None:
    """JSON-LD 를 훑어 표준(schema.org) 필드만 모은다 — offers 안의 가격까지."""
    if depth > 4 or len(out) > 20:
        return
    if isinstance(node, list):
        for x in node:
            _ld_fields(x, out, depth + 1)
        return
    if not isinstance(node, dict):
        return
    for k in _LD_KEYS:
        v = node.get(k)
        if isinstance(v, (str, int, float)) and str(v).strip():
            out.append((k, str(v).strip()[:120]))
    for k in ("offers", "@graph", "mainEntity"):
        if k in node:
            _ld_fields(node[k], out, depth + 1)


def structured_facts(soup: BeautifulSoup) -> str:
    """페이지의 구조화 데이터(JSON-LD·메타 태그)에 적힌 가격·통화·게시/갱신 시각. 없으면 빈 문자열.

    실측(2026-09-28): TradingView USD/KRW 페이지는 화면 값이 스크립트로 그려져 본문 추출이 관련 기사 링크 596자뿐이었는데,
    원본 HTML 의 JSON-LD 에 "offers.price": "1355.28" 가 있었다. script 태그를 지우기 전에 표준 필드만 건진다.
    """
    import json

    found: list = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            _ld_fields(json.loads(tag.string or tag.get_text() or ""), found)
        except (ValueError, TypeError):
            continue
    for m in soup.find_all("meta"):
        key = (m.get("property") or m.get("name") or "").strip().lower()
        if key in _META_KEYS and (m.get("content") or "").strip():
            found.append((key, m["content"].strip()[:120]))
    if not any(k in ("price", "lowPrice", "highPrice", "product:price:amount", "og:price:amount") for k, _ in found):
        dates = [x for x in found if "date" in x[0] or "time" in x[0]]
        if not dates:
            return ""
        found = dates[:3]                      # 가격이 없으면 게시·갱신 시각만(시점 판단용)
    seen, parts = set(), []
    labels = {"priceValidUntil": "priceValidUntil(이 가격이 유효한 시각)", "dateModified": "dateModified(갱신 시각)",
              "datePublished": "datePublished(게시 시각)"}
    for k, v in found:
        if (k, v) not in seen:
            seen.add((k, v))
            parts.append(f"{labels.get(k, k)}: {v}")
    return ("[페이지 구조화 데이터 — 화면 값과 다를 수 있다, 시각을 보라] " + " · ".join(parts))[:_MAX_FACTS_CHARS]


def extract(html: str, *, mode: str = "markdown") -> tuple[str, str]:
    """(제목, 본문). 본문이 비면 빈 문자열 — 호출자가 다음 추출기로 넘어간다."""
    if not html:
        return "", ""
    if len(html) > MAX_HTML_CHARS:
        html = html[:MAX_HTML_CHARS]
    soup = _soup(html)
    title = ""
    if soup.title and soup.title.string:
        title = _text(soup.title.string).strip()
    facts = structured_facts(soup)            # script 를 지우기 전에(JSON-LD 는 script 태그다)
    strip_hidden(soup)
    main = select_main(soup)
    if not title:
        h1 = main.find("h1") or soup.find("h1")
        title = h1.get_text(" ", strip=True) if h1 else ""
    _strip_boilerplate(main)
    body = to_text(main) if mode == "text" else to_markdown(main)
    body = _INVISIBLE_UNICODE_RE.sub("", body)
    facts = _drop_stale_page_dates(facts, f"{title}\n{body}")
    if facts:
        body = f"{facts}\n\n{body}" if body.strip() else facts
    return _INVISIBLE_UNICODE_RE.sub("", title)[:256], body


_FACT_DATE_RE = re.compile(r"(?:date\w*|\w*_time|og:updated_time)[^:]*:\s*(20\d{2})-(\d{2})")


def _drop_stale_page_dates(facts: str, text: str) -> str:
    """가격 없이 게시·갱신 시각만 있는 구조화 데이터가 본문 표기보다 오래됐으면 버린다.

    실측(2026-09-28 EPDM): Business Analytiq 본문 제목은 "EPDM rubber price September 2026"인데 JSON-LD dateModified 는
    2025-11-19(CMS 가 페이지 틀 갱신 때만 바꾸는 값)였다. 이 줄을 "[구조화 데이터 — 시각을 보라]"로 앞세우자 gemini 는
    "2025년 11월 게시 자료"라 쓰고, vllm 은 9월 자료를 오래됐다며 뺐고, 검증기도 그 날짜로 되돌렸다. 가격이 함께 있는
    구조화 데이터(TradingView priceValidUntil)는 값의 시각이라 남긴다.
    """
    if not facts or "price" in facts.lower():
        return facts
    from agent_arch.services.web.periods import latest_period
    body_p = latest_period(text)
    meta = [f"{y}-{m}" for y, m in _FACT_DATE_RE.findall(facts)]
    if body_p and meta and max(meta) < body_p:
        return ""
    return facts


def basic_text_fallback(html: str) -> str:
    """추출기가 빈손일 때 마지막 수단 — 태그만 벗긴 평문(스크립트·스타일은 이미 제거)."""
    soup = _soup(html[:MAX_HTML_CHARS])
    for t in soup.find_all(_DROP_TAGS):
        t.decompose()
    return _INVISIBLE_UNICODE_RE.sub("", to_text(soup.body or soup))


# ─── 페이지가 불러오는 데이터 주소(정적 탐지) ───────────────────────────────────
# 실측(2026-09-28): 서울외국환중개 '오늘의 매매기준율' 페이지는 198KB HTML 인데 본문 추출은 490자 — 환율표를 인라인 스크립트의
# $.ajax({url: "/Flash/TodayExRate_flash.jsp", data: "tr_date=2026-09-28"}) 로 채운다. 그 주소를 부르면 오늘 고시값(EUR 1,538.04)이
# 그대로 온다. 업계 패턴(정적 추출 → 필요할 때만 렌더링)의 앞 단계로, 브라우저 없이 인라인 스크립트의 **리터럴** 주소만 찾는다.
# 동적으로 조립되는 주소·POST·외부 JS 파일은 대상이 아니다(그건 렌더링 몫).
# 블록 끝(`})`)은 찾지 않는다 — success 콜백 안의 중괄호가 여러 겹이라 끝을 찾으면 놓친다(실제 서울외국환중개 페이지에서 실패).
# `$.ajax({` 뒤 앞부분(url·data·type 는 보통 맨 앞)만 본다.
_AJAX_BLOCK_RE = re.compile(r"\$\.ajax\s*\(\s*\{(?P<body>.{0,400})", re.S)
_KV_RE = {k: re.compile(k + r"""\s*:\s*(["'])(?P<v>[^"']{1,300})\1""") for k in ("url", "data", "type", "method")}
_CALL_RES = (
    re.compile(r"""\$\.(?:get|getJSON)\s*\(\s*(["'])(?P<u>[^"']{1,300})\1"""),
    re.compile(r"""\bfetch\s*\(\s*(["'])(?P<u>[^"']{1,300})\1"""),
    re.compile(r"""\.open\s*\(\s*(["'])GET\1\s*,\s*(["'])(?P<u>[^"']{1,300})\2""", re.I),
)


def find_data_endpoints(html: str, base_url: str, *, limit: int = 2) -> list[str]:
    """인라인 스크립트에서 같은 사이트의 GET 데이터 주소(리터럴)를 찾는다. 쿼리 문자열 data 는 붙인다."""
    from urllib.parse import urljoin, urlsplit

    if not html:
        return []
    base_host = (urlsplit(base_url).hostname or "").lower()
    soup = _soup(html[:MAX_HTML_CHARS])
    scripts = "\n".join(s.get_text() for s in soup.find_all("script") if not s.get("src"))
    found: list[str] = []

    def add(u: str, data: str = "") -> None:
        u = u.strip()
        if not u or u.startswith(("javascript:", "#", "data:")) or "{" in u or "+" in u:
            return
        full = urljoin(base_url, u)
        if (urlsplit(full).hostname or "").lower() != base_host or urlsplit(full).scheme not in ("http", "https"):
            return
        if data and "=" in data and "{" not in data:
            full += ("&" if "?" in full else "?") + data.strip()
        if full not in found and full.split("#")[0] != base_url.split("#")[0]:
            found.append(full)

    for m in _AJAX_BLOCK_RE.finditer(scripts):
        body = m.group("body")
        body = body.split("function", 1)[0]                  # 콜백 본문의 다른 url 키에 끌려가지 않게
        kv = {k: (r.search(body).group("v") if r.search(body) else "") for k, r in _KV_RE.items()}
        if (kv["type"] or kv["method"] or "GET").upper() == "GET" and kv["url"]:
            add(kv["url"], kv["data"])
    for rx in _CALL_RES:
        for m in rx.finditer(scripts):
            add(m.group("u"))
    return found[:limit]
