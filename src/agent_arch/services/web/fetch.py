"""`web_fetch` 파이프라인 — URL 정리 → 호스트 정책·DNS 고정 → GET(리다이렉트 홉마다 재검사) →
스트리밍 크기 캡 → 문자셋 판별 → 콘텐츠 타입별 추출 → 예산·이어 읽기 → 봉투.

레퍼런스가 실사고 뒤 고정한 규칙(모두 그대로):
- URL 안 공백 복구(`https:// docs.x` — 모델이 만든 인자에 자주 섞인다). 경로·쿼리 공백은 그대로(퍼센트 인코딩).
- 리다이렉트는 수동으로 최대 3홉, **홉마다** 호스트 정책과 DNS 검증을 다시 한다(신뢰가 origin 을 넘어가지 않게).
- 본문은 스트리밍으로 읽고 750KB 를 넘으면 끊는다. 길이를 모르는 응답도 캡이 표시용이 아니라 실제 상한이다.
- 문자셋은 BOM → Content-Type charset → 첫 4KB 의 <meta charset> → 통계 추정 순(Shift_JIS/EUC-KR 깨짐 방지).
- Accept 는 text/markdown 을 우선한다(Cloudflare "Markdown for Agents" 등 서버가 주면 그대로 받는다).
- 운영자 추가 헤더는 프레이밍·식별 헤더(accept·user-agent·host·content-length…)를 걸러 낸다.
- 긴 페이지는 스필 파일 대신 `start` 로 이어 읽는다(우리 모델은 파일을 못 읽는다). 전문은 캐시에 두어
  이어 읽기가 재요청을 만들지 않게 한다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit

from agent_arch.services.web import html_extract
from agent_arch.services.web.common import ToolInputError, resolve_timeout_seconds
from agent_arch.services.web.envelope import wrap_web_content
from agent_arch.services.web.netguard import BlockedTarget, PinnedBackend, UnresolvableHost, check_hostname_policy, resolve_and_validate

logger = logging.getLogger(__name__)

DEFAULT_MAX_CHARS = 20_000
MAX_CHARS_CAP = 20_000
MIN_MAX_CHARS = 100
THIN_CONTENT_CHARS = 400          # 이보다 짧은 HTML 본문은 JS 렌더 페이지일 확률이 높다(dev 실측: weather.com 148자)
DEFAULT_MAX_BYTES = 750_000
MIN_MAX_BYTES, MAX_MAX_BYTES = 32_000, 10_000_000
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/122.0.0.0 Safari/537.36")
EXTRACT_MODES = ("markdown", "text")
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_BLOCKED_HEADER_NAMES = frozenset({
    "accept", "accept-language", "user-agent", "sec-fetch-mode", "connection", "content-length", "expect", "host",
    "keep-alive", "proxy-connection", "te", "trailer", "transfer-encoding", "upgrade", "cookie", "authorization",
})
_CACHE_PREFIX = "web:fetch:"
_META_CHARSET_RE = re.compile(rb'<meta[^>]+charset=["\']?\s*([a-zA-Z0-9_\-]+)', re.I)
_XML_ENC_RE = re.compile(rb'^\s*<\?xml[^>]*encoding=["\']([a-zA-Z0-9_\-]+)', re.I)

# requester(method, url, headers, pins) -> (status, headers(dict, lower keys), body_iter(bytes))
Requester = Callable[[str, str, dict, dict], Awaitable[tuple[int, dict, AsyncIterator[bytes]]]]


@dataclass
class FetchContext:
    timeout_seconds: int = 30
    max_bytes: int = DEFAULT_MAX_BYTES
    max_chars_cap: int = MAX_CHARS_CAP
    max_redirects: int = DEFAULT_MAX_REDIRECTS
    user_agent: str = DEFAULT_USER_AGENT
    blocked_hosts: tuple[str, ...] = ()
    allowed_hosts: tuple[str, ...] = ()
    operator_headers: dict[str, str] = field(default_factory=dict)
    cache_ttl_minutes: float = 15
    requester: Optional[Requester] = None            # 테스트 주입. None 이면 httpcore 고정 백엔드
    resolver: Optional[Callable[[str], Awaitable[list[str]]]] = None


class FetchError(Exception):
    def __init__(self, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.code, self.message, self.extra = code, message, extra

    def payload(self) -> dict:
        return {"error": self.code, "message": self.message, **self.extra}


# ─── URL ───────────────────────────────────────────────────────────────────────

def sanitize_url(raw: str) -> str:
    """스킴 뒤·호스트 안 공백만 복구한다. 경로·쿼리 공백은 WHATWG 규칙대로 뒤에서 인코딩된다."""
    s = (raw or "").strip()
    if not s:
        raise ToolInputError("url 은 비어 있지 않은 http(s) URL 이어야 합니다.")
    s = re.sub(r"^(https?):\s*//\s*", lambda m: f"{m.group(1).lower()}://", s, flags=re.I)
    m = re.match(r"^(https?://)([^/?#]*)(.*)$", s, flags=re.S)
    if not m:
        raise ToolInputError("url 은 http:// 또는 https:// 로 시작해야 합니다.")
    scheme, authority, rest = m.groups()
    authority = re.sub(r"\s+", "", authority)
    rest = rest.replace(" ", "%20")
    parts = urlsplit(scheme + authority + rest)
    if not parts.hostname:
        raise ToolInputError("url 에 호스트가 없습니다.")
    if parts.username or parts.password:
        raise ToolInputError("URL 에 자격증명(user:pass@)을 넣을 수 없습니다.")
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path or "/", parts.query, ""))


def _host_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


# ─── 요청 ──────────────────────────────────────────────────────────────────────

def build_request_headers(ctx: FetchContext) -> dict[str, str]:
    headers = {
        "Accept": "text/markdown, text/html;q=0.9, */*;q=0.1",
        "User-Agent": ctx.user_agent or DEFAULT_USER_AGENT,
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    }
    for k, v in (ctx.operator_headers or {}).items():
        if k.lower() in _BLOCKED_HEADER_NAMES:
            logger.warning("[web_fetch] 예약·프레이밍 헤더는 무시한다: %s", k)
            continue
        headers[k] = v
    return headers


async def _httpcore_requester(method: str, url: str, headers: dict, pins: dict, *, timeout: float):
    import httpcore

    pool = httpcore.AsyncConnectionPool(network_backend=PinnedBackend(pins=pins), http2=False, retries=0)
    # pool.request() 는 본문을 통째로 읽어 버린다(스트리밍 캡 무효). 저수준 handle_async_request 로 응답을 열고
    # 본문 반복자는 우리가 소유한다 — 캡에서 끊으면 나머지는 받지 않는다.
    parts = urlsplit(url)
    default_port = 443 if parts.scheme == "https" else 80
    host_header = parts.hostname or ""
    if parts.port and parts.port != default_port:
        host_header += f":{parts.port}"
    if ":" in host_header.split(":")[0] if parts.port else ":" in host_header:   # IPv6 리터럴은 대괄호
        host_header = f"[{parts.hostname}]" + (f":{parts.port}" if parts.port and parts.port != default_port else "")
    # 저수준 API 는 Host 를 자동으로 붙이지 않는다(운영자 헤더의 Host 는 build_request_headers 가 걸러 냈다).
    wire_headers = [(b"Host", host_header.encode())] + [(k.encode(), str(v).encode()) for k, v in headers.items()]
    req = httpcore.Request(method, url, headers=wire_headers,
                           extensions={"timeout": {"connect": timeout, "read": timeout, "write": timeout, "pool": timeout}})
    try:
        resp = await pool.handle_async_request(req)
    except Exception:
        await pool.aclose()
        raise
    hdrs = {k.decode().lower(): v.decode(errors="replace") for k, v in resp.headers}

    async def body() -> AsyncIterator[bytes]:
        try:
            async for chunk in resp.aiter_stream():
                yield chunk
        finally:
            await resp.aclose()
            await pool.aclose()

    return resp.status, hdrs, body()


async def guarded_get(url: str, ctx: FetchContext) -> tuple[int, dict, AsyncIterator[bytes], str]:
    """호스트 정책·DNS 검증을 통과한 GET. 리다이렉트를 수동으로 따라가며 홉마다 다시 검사한다.
    반환: (status, headers, body_iter, final_url)."""
    timeout = float(resolve_timeout_seconds(ctx.timeout_seconds))
    current = url
    for hop in range(ctx.max_redirects + 1):
        host = _host_of(current)
        check_hostname_policy(host, blocked=ctx.blocked_hosts, allowed=ctx.allowed_hosts)
        addrs = await resolve_and_validate(host, resolver=ctx.resolver)
        pins = {host: addrs[0]}
        headers = build_request_headers(ctx)
        if ctx.requester is not None:
            status, hdrs, body = await ctx.requester("GET", current, headers, pins)
        else:
            status, hdrs, body = await _httpcore_requester("GET", current, headers, pins, timeout=timeout)
        if status in _REDIRECT_STATUSES and hdrs.get("location"):
            async for _ in body:                      # 리다이렉트 본문은 버린다
                pass
            if hop >= ctx.max_redirects:
                raise FetchError("too_many_redirects", f"리다이렉트가 {ctx.max_redirects}회를 넘었습니다.")
            nxt = urljoin(current, hdrs["location"])
            current = sanitize_url(nxt)
            continue
        return status, hdrs, body, current
    raise FetchError("too_many_redirects", f"리다이렉트가 {ctx.max_redirects}회를 넘었습니다.")


# 본문이 이보다 짧은 HTML 은 화면 값을 스크립트로 채우는 페이지일 수 있다 — 인라인 스크립트의 데이터 주소를 찾아 부른다.
# 실측(2026-09-28): 서울외국환중개 매매기준율 198KB HTML → 본문 490자, 스크립트의 $.ajax 주소에 오늘 고시값.
DISCOVERY_MAX_BODY_CHARS = 2_000
_PAGE_DATA_MAX_BYTES = 200_000
_PAGE_DATA_MAX_CHARS = 4_000


async def _fetch_page_data(html: str, page_url: str, ctx: "FetchContext") -> str:
    """페이지가 불러오는 같은 사이트 데이터(GET 리터럴 주소, 최대 2)를 받아 텍스트로. 실패는 조용히 빈 문자열."""
    parts = []
    for u in html_extract.find_data_endpoints(html, page_url):
        try:
            status, hdrs, body, final = await guarded_get(u, ctx)       # 사설망 차단·리다이렉트 재검사 그대로
            if not (200 <= status < 300):
                async for _ in body:
                    pass
                continue
            raw, _cut = await read_body_capped(body, _PAGE_DATA_MAX_BYTES)
            data = decode_body(raw, hdrs.get("content-type", "")).strip()
        except Exception:  # noqa: BLE001 — 부가 데이터라 실패해도 본 페이지는 그대로
            continue
        if data and not data.lstrip().startswith("<"):                     # HTML(다른 페이지)은 데이터가 아니다
            parts.append(f"[페이지가 불러오는 데이터: {final}]\n{data[:_PAGE_DATA_MAX_CHARS]}")
    return "\n\n".join(parts)


async def read_body_capped(body: AsyncIterator[bytes], max_bytes: int) -> tuple[bytes, bool]:
    """캡을 넘기면 읽기를 끊는다. (bytes, truncated). 길이를 모르는 스트림도 실제로 여기서 막힌다."""
    buf = bytearray()
    truncated = False
    async for chunk in body:
        if not chunk:
            continue
        room = max_bytes - len(buf)
        if len(chunk) >= room:
            buf.extend(chunk[:room])
            truncated = True
            break
        buf.extend(chunk)
    return bytes(buf), truncated


# ─── 디코딩 ────────────────────────────────────────────────────────────────────

def _charset_from_header(content_type: str) -> Optional[str]:
    m = re.search(r"charset=\"?([a-zA-Z0-9_\-]+)", content_type or "", re.I)
    return m.group(1) if m else None


def decode_body(raw: bytes, content_type: str) -> str:
    """BOM → 헤더 charset → <meta charset>(첫 4KB) → 통계 추정 → utf-8(대체)."""
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw[3:].decode("utf-8", errors="replace")
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16", errors="replace")
    for cand in (_charset_from_header(content_type),
                 (m.group(1).decode() if (m := _META_CHARSET_RE.search(raw[:4096])) else None),
                 (m.group(1).decode() if (m := _XML_ENC_RE.search(raw[:4096])) else None)):
        if cand:
            try:
                return raw.decode(cand.strip().lower().replace("euc_kr", "euc-kr"), errors="replace")
            except LookupError:
                continue
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw[:200_000]).best()
        if best is not None and best.encoding:
            return raw.decode(best.encoding, errors="replace")
    except Exception:
        pass
    return raw.decode("utf-8", errors="replace")


# ─── 추출 ──────────────────────────────────────────────────────────────────────

def _media_type(content_type: str) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _is_json(media: str) -> bool:
    return media == "application/json" or media.endswith("+json")


_BINARY_PREFIXES = ("image/", "audio/", "video/", "font/", "application/octet-stream", "application/zip", "application/gzip",
                    "application/x-", "application/vnd.", "application/pdf", "application/msword")


def is_unsupported_media(media: str) -> bool:
    return bool(media) and media.startswith(_BINARY_PREFIXES) and not media.endswith(("+json", "+xml"))


def extract_content(text: str, media: str, *, mode: str) -> tuple[str, str, str]:
    """(추출기 이름, 제목, 본문)."""
    if is_unsupported_media(media):
        hint = " PDF 는 아직 읽지 못합니다." if media == "application/pdf" else ""
        raise FetchError("unsupported_content_type", f"텍스트가 아닌 응답({media})은 읽지 않습니다.{hint}", content_type=media)
    if media == "text/markdown":
        return "server-markdown", "", text
    if media in ("text/html", "application/xhtml+xml") or (not media and text.lstrip()[:1] == "<"):
        title, body = html_extract.extract(text, mode=mode)
        if body.strip():
            return "readability", title, body
        fallback = html_extract.basic_text_fallback(text)
        if fallback.strip():
            return "raw-html", title, fallback
        raise FetchError("empty_extraction", "페이지에서 읽을 수 있는 본문을 찾지 못했습니다(스크립트로만 그려지는 페이지일 수 있습니다).")
    if _is_json(media):
        try:
            return "json", "", json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except Exception:
            return "raw", "", text
    return "raw", "", text


# ─── 캐시 ──────────────────────────────────────────────────────────────────────

def _cache_key(url: str, mode: str, ctx: FetchContext) -> str:
    raw = ":".join(["fetch", url, mode, str(ctx.max_bytes), hashlib.sha256(json.dumps(sorted(ctx.operator_headers.items())).encode()).hexdigest()[:16]])
    return _CACHE_PREFIX + hashlib.sha1(raw.encode()).hexdigest()


async def _cache_get(key: str, ttl_ms: int) -> Optional[dict]:
    if ttl_ms <= 0:
        return None
    try:
        from agent_arch.memory.redis_store import redis_store
        entry = await redis_store.get(key)
    except Exception:
        return None
    if not isinstance(entry, dict) or "doc" not in entry:
        return None
    if time.time() * 1000 - float(entry.get("inserted_at") or 0) >= ttl_ms:
        return None
    return entry["doc"]


async def _cache_set(key: str, doc: dict, ttl_ms: int) -> None:
    if ttl_ms <= 0:
        return
    try:
        from agent_arch.memory.redis_store import redis_store
        await redis_store.set(key, {"doc": doc, "inserted_at": time.time() * 1000}, expire=max(1, ttl_ms // 1000))
    except Exception:
        logger.debug("web_fetch 캐시 기록 실패(무시)", exc_info=True)


# ─── 진입점 ────────────────────────────────────────────────────────────────────

def _read_params(args: dict, ctx: FetchContext) -> tuple[str, str, int, int]:
    url = sanitize_url(str(args.get("url") or ""))
    mode = str(args.get("extract_mode") or "markdown").strip().lower()
    if mode not in EXTRACT_MODES:
        raise ToolInputError("extract_mode 는 markdown 또는 text 여야 합니다.")
    raw_max = args.get("max_chars")
    max_chars = DEFAULT_MAX_CHARS if raw_max in (None, "") else int(raw_max)
    if max_chars < MIN_MAX_CHARS:
        raise ToolInputError(f"max_chars 는 {MIN_MAX_CHARS} 이상이어야 합니다.")
    max_chars = min(max_chars, ctx.max_chars_cap)
    raw_start = args.get("start")
    start = 0 if raw_start in (None, "") else int(raw_start)
    if start < 0:
        raise ToolInputError("start 는 0 이상이어야 합니다.")
    return url, mode, max_chars, start


async def fetch_web_payload(args: dict, ctx: FetchContext) -> dict:
    """한 URL 을 읽어 모델에 줄 자유 형식 dict 로. 정책 위반·HTTP 오류는 구조화 오류 dict."""
    url, mode, max_chars, start = _read_params(args, ctx)
    ttl_ms = int(max(0.0, float(ctx.cache_ttl_minutes)) * 60_000)
    key = _cache_key(url, mode, ctx)
    doc = await _cache_get(key, ttl_ms)
    cached = doc is not None
    if doc is None:
        started = time.time()
        try:
            status, hdrs, body, final_url = await guarded_get(url, ctx)
        except UnresolvableHost as e:
            return {"error": "unresolvable_host", "message": str(e), "url": url}
        except BlockedTarget as e:
            return {"error": "blocked_target", "message": str(e), "url": url}
        except FetchError as e:
            return {**e.payload(), "url": url}
        if status < 200 or status >= 300:
            async for _ in body:
                pass
            return {"error": "http_error", "message": f"HTTP {status} 응답입니다.", "status": status, "url": final_url}
        raw, truncated_bytes = await read_body_capped(body, max(MIN_MAX_BYTES, min(ctx.max_bytes, MAX_MAX_BYTES)))
        content_type = hdrs.get("content-type", "")
        media = _media_type(content_type)
        if is_unsupported_media(media):
            hint = " PDF 는 아직 읽지 못합니다." if media == "application/pdf" else ""
            return {"error": "unsupported_content_type", "message": f"텍스트가 아닌 응답({media})은 읽지 않습니다.{hint}",
                    "content_type": media, "url": final_url, "status": status}
        text = decode_body(raw, content_type)
        is_html = media in ("text/html", "application/xhtml+xml") or (not media and text.lstrip()[:1] == "<")
        try:
            extractor, title, content = extract_content(text, media, mode=mode)
        except FetchError as e:
            if not (is_html and e.code == "empty_extraction"):
                return {**e.payload(), "url": final_url, "status": status}
            extractor, title, content = "raw-html", "", ""
        if is_html and len(content) < DISCOVERY_MAX_BODY_CHARS:
            extra = await _fetch_page_data(text, final_url, ctx)
            if extra:
                content = f"{content}\n\n{extra}".strip()
                extractor = f"{extractor}+page-data"
        if not content.strip():
            return {"error": "empty_extraction", "message": "페이지에서 읽을 수 있는 본문을 찾지 못했습니다(스크립트로만 그려지는 페이지일 수 있습니다).",
                    "url": final_url, "status": status}
        doc = {"url": url, "final_url": final_url, "status": status, "content_type": content_type[:256],
               "extractor": extractor, "title": title[:256], "content": content, "truncated_bytes": truncated_bytes,
               "tookMs": int((time.time() - started) * 1000)}
        await _cache_set(key, doc, ttl_ms)

    content: str = doc["content"]
    total = len(content)
    if start >= total and total > 0:
        return {"error": "start_out_of_range", "message": f"start({start})가 본문 길이({total})를 넘습니다.", "url": doc["final_url"], "total_chars": total}
    piece = content[start:start + max_chars]
    end = start + len(piece)
    payload = {
        "url": doc["url"], "final_url": doc["final_url"], "status": doc["status"], "content_type": doc["content_type"],
        "extractor": doc["extractor"], "title": doc["title"],
        "text": wrap_web_content(piece, "web_fetch"),
        "chars": {"start": start, "end": end, "total": total},
        "truncated": bool(end < total or doc["truncated_bytes"]),
        "externalContent": {"untrusted": True, "source": "web_fetch", "wrapped": True},
    }
    if end < total:
        payload["next_start"] = end
        payload["hint"] = f"본문이 더 있습니다. 같은 url 에 start={end} 로 다시 부르면 이어서 읽습니다."
    if doc["truncated_bytes"]:
        payload["note"] = "응답이 크기 상한에서 잘렸습니다(페이지 뒷부분은 받지 않았습니다)."
    elif total < THIN_CONTENT_CHARS and doc["extractor"] in ("readability", "raw-html"):
        payload["note"] = ("본문이 거의 없습니다 — 스크립트로 그려지는 페이지(날씨·시세 위젯 등)일 수 있습니다. "
                           "검색 요약이나 다른 출처를 쓰세요.")
        payload["thin_content"] = True
    if cached:
        payload["cached"] = True
    if doc.get("tookMs") is not None and not cached:
        payload["tookMs"] = doc["tookMs"]
    return payload
