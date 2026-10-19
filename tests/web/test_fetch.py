"""web_fetch 파이프라인 — URL 정리, 리다이렉트 홉별 재검사, 크기 캡, 문자셋, 타입별 추출, 이어 읽기, 캐시, 헤더 필터."""
import json

import pytest

from agent_arch.services.web import fetch as F
from agent_arch.services.web.common import ToolInputError


class _Store:
    def __init__(self):
        self.data = {}
    async def get(self, k):
        return self.data.get(k)
    async def set(self, k, v, expire=None):
        self.data[k] = v
        return True


async def _aiter(*chunks):
    for c in chunks:
        yield c


def _requester(routes: dict, log: list):
    """routes: url → (status, headers, body_bytes|list[bytes])."""
    async def req(method, url, headers, pins):
        log.append({"url": url, "headers": headers, "pins": pins})
        status, hdrs, body = routes[url]
        chunks = body if isinstance(body, list) else [body]
        return status, {k.lower(): v for k, v in hdrs.items()}, _aiter(*chunks)
    return req


async def _public(host):
    return {"ex.com": ["93.184.216.34"], "next.ex.com": ["93.184.216.35"], "evil.ex.com": ["10.0.0.9"]}.get(host, [])


def _ctx(routes, log, **kw):
    base = dict(requester=_requester(routes, log), resolver=_public, cache_ttl_minutes=0, max_bytes=32_000)
    base.update(kw)
    return F.FetchContext(**base)


@pytest.fixture
def store(monkeypatch):
    import agent_arch.memory.redis_store as rs
    s = _Store(); monkeypatch.setattr(rs, "redis_store", s); return s


def test_sanitize_url_repairs_model_whitespace_and_rejects_bad_schemes():
    assert F.sanitize_url("https:// Ex.com /a b?q=1#frag") == "https://ex.com/a%20b?q=1"
    assert F.sanitize_url("HTTPS://ex.com") == "https://ex.com/"
    for bad in ("", "ftp://x", "javascript:alert(1)", "https://user:pw@ex.com/", "https:///nohost"):
        with pytest.raises(ToolInputError):
            F.sanitize_url(bad)


@pytest.mark.asyncio
async def test_html_happy_path_wraps_with_warning_and_reports_chars(store):
    html = "<html><head><title>기사</title><meta charset='utf-8'></head><body><article><h1>제목</h1>" + "<p>본문 문장입니다. </p>" * 20 + "</article></body></html>"
    log = []
    out = await F.fetch_web_payload({"url": "https://ex.com/a"}, _ctx({"https://ex.com/a": (200, {"Content-Type": "text/html; charset=utf-8"}, html.encode())}, log))
    assert out["status"] == 200 and out["extractor"] == "readability" and out["title"] == "기사"
    assert out["text"].startswith("보안 안내") and "Source: 웹 페이지" in out["text"] and "# 제목" in out["text"]
    assert out["chars"]["start"] == 0 and out["chars"]["end"] == out["chars"]["total"] and out["truncated"] is False
    assert log[0]["pins"] == {"ex.com": "93.184.216.34"} and log[0]["headers"]["Accept"].startswith("text/markdown")
    assert "Cookie" not in log[0]["headers"]


@pytest.mark.asyncio
async def test_redirects_revalidated_per_hop_and_capped(store):
    log = []
    routes = {
        "https://ex.com/r1": (302, {"Location": "/r2"}, b""),
        "https://ex.com/r2": (301, {"Location": "https://next.ex.com/final"}, b""),
        "https://next.ex.com/final": (200, {"Content-Type": "text/plain"}, b"done"),
        "https://ex.com/bad": (302, {"Location": "http://evil.ex.com/"}, b""),
        "https://ex.com/loop": (302, {"Location": "https://ex.com/loop"}, b""),
        "https://ex.com/meta": (302, {"Location": "http://169.254.169.254/latest"}, b""),
    }
    out = await F.fetch_web_payload({"url": "https://ex.com/r1"}, _ctx(routes, log))
    assert out["final_url"] == "https://next.ex.com/final" and out["extractor"] == "raw" and [l["url"] for l in log] == ["https://ex.com/r1", "https://ex.com/r2", "https://next.ex.com/final"]
    out = await F.fetch_web_payload({"url": "https://ex.com/bad"}, _ctx(routes, []))
    assert out["error"] == "blocked_target" and "내부망" in out["message"]          # 두 번째 홉에서 사설 IP → 차단
    out = await F.fetch_web_payload({"url": "https://ex.com/meta"}, _ctx(routes, []))
    assert out["error"] == "blocked_target"                                          # 메타데이터 리터럴
    out = await F.fetch_web_payload({"url": "https://ex.com/loop"}, _ctx(routes, []))
    assert out["error"] == "too_many_redirects"


@pytest.mark.asyncio
async def test_blocked_hosts_never_requested(store):
    log = []
    ctx = _ctx({}, log, blocked_hosts=("*.example.com",))
    assert (await F.fetch_web_payload({"url": "https://chat.example.com/x"}, ctx))["error"] == "blocked_target"
    assert (await F.fetch_web_payload({"url": "http://localhost:8000/"}, ctx))["error"] == "blocked_target"
    assert (await F.fetch_web_payload({"url": "http://127.1/"}, ctx))["error"] == "blocked_target"
    assert (await F.fetch_web_payload({"url": "http://[::ffff:10.0.0.1]/"}, ctx))["error"] == "blocked_target"
    assert log == []


@pytest.mark.asyncio
async def test_size_cap_truncates_stream_fail_closed(store):
    big = [b"x" * 10_000] * 10                                  # 100KB 를 10 청크로, 길이 헤더 없음
    log = []
    out = await F.fetch_web_payload({"url": "https://ex.com/big"}, _ctx({"https://ex.com/big": (200, {"Content-Type": "text/plain"}, big)}, log, max_bytes=32_000))
    assert out["truncated"] is True and "크기 상한" in out["note"] and out["chars"]["total"] == 32_000


@pytest.mark.asyncio
async def test_charset_sniffing_euc_kr_and_shift_jis(store):
    kr = "<html><head><meta charset=euc-kr><title>한글</title></head><body><article><p>" + "환율 기사 본문입니다. " * 10 + "</p></article></body></html>"
    out = await F.fetch_web_payload({"url": "https://ex.com/kr"}, _ctx({"https://ex.com/kr": (200, {"Content-Type": "text/html"}, kr.encode("euc-kr"))}, []))
    assert out["title"] == "한글" and "환율 기사 본문입니다" in out["text"]
    jp = "日本語の本文です。" * 10
    out = await F.fetch_web_payload({"url": "https://ex.com/jp"}, _ctx({"https://ex.com/jp": (200, {"Content-Type": "text/plain; charset=Shift_JIS"}, jp.encode("shift_jis"))}, []))
    assert "日本語の本文です" in out["text"]
    assert F.decode_body(b"\xef\xbb\xbfabc", "") == "abc"


@pytest.mark.asyncio
async def test_content_type_dispatch_json_markdown_and_http_error(store):
    routes = {
        "https://ex.com/j": (200, {"Content-Type": "application/json"}, json.dumps({"a": [1, 2], "b": "가"}).encode()),
        "https://ex.com/md": (200, {"Content-Type": "text/markdown"}, b"# Served markdown\n\nbody"),
        "https://ex.com/404": (404, {"Content-Type": "text/html"}, b"<html>nope</html>"),
        "https://ex.com/js": (200, {"Content-Type": "text/html"}, b"<html><body><div id=app></div><script>x()</script></body></html>"),
    }
    out = await F.fetch_web_payload({"url": "https://ex.com/j"}, _ctx(routes, []))
    assert out["extractor"] == "json" and '"a": [\n    1,' in out["text"]
    out = await F.fetch_web_payload({"url": "https://ex.com/md"}, _ctx(routes, []))
    assert out["extractor"] == "server-markdown" and "# Served markdown" in out["text"]
    out = await F.fetch_web_payload({"url": "https://ex.com/404"}, _ctx(routes, []))
    assert out["error"] == "http_error" and out["status"] == 404
    out = await F.fetch_web_payload({"url": "https://ex.com/js"}, _ctx(routes, []))
    assert out["error"] == "empty_extraction"


@pytest.mark.asyncio
async def test_pagination_and_cache_reuse(store):
    body = ("<html><body><article><p>" + "가나다라마바사아자차카타파하 " * 400 + "</p></article></body></html>").encode()
    log = []
    ctx = _ctx({"https://ex.com/long": (200, {"Content-Type": "text/html"}, body)}, log, cache_ttl_minutes=15)
    a = await F.fetch_web_payload({"url": "https://ex.com/long", "max_chars": 1000}, ctx)
    assert a["truncated"] is True and a["next_start"] == 1000 and "start=1000" in a["hint"] and a["chars"]["total"] > 5000
    b = await F.fetch_web_payload({"url": "https://ex.com/long", "max_chars": 1000, "start": 1000}, ctx)
    assert b["cached"] is True and b["chars"]["start"] == 1000 and len(log) == 1          # 이어 읽기는 재요청 없이 캐시에서
    c = await F.fetch_web_payload({"url": "https://ex.com/long", "start": 10**7}, ctx)
    assert c["error"] == "start_out_of_range"
    with pytest.raises(ToolInputError):
        await F.fetch_web_payload({"url": "https://ex.com/long", "max_chars": 10}, ctx)
    with pytest.raises(ToolInputError):
        await F.fetch_web_payload({"url": "https://ex.com/long", "extract_mode": "pdf"}, ctx)


def test_operator_headers_filtered():
    ctx = F.FetchContext(operator_headers={"X-Team": "agent", "Host": "evil", "Cookie": "sid=1", "Authorization": "Bearer x", "Accept": "*/*"})
    h = F.build_request_headers(ctx)
    assert h["X-Team"] == "agent" and h["Accept"].startswith("text/markdown")
    assert "Host" not in h and "Cookie" not in h and "Authorization" not in h


@pytest.mark.asyncio
async def test_binary_and_pdf_are_refused_before_decoding(store):
    routes = {
        "https://ex.com/img": (200, {"Content-Type": "image/png"}, b"\x89PNG\r\n" + b"\x00" * 100),
        "https://ex.com/doc.pdf": (200, {"Content-Type": "application/pdf"}, b"%PDF-1.7 ..."),
        "https://ex.com/feed": (200, {"Content-Type": "application/atom+xml"}, b"<feed><title>t</title></feed>"),
    }
    out = await F.fetch_web_payload({"url": "https://ex.com/img"}, _ctx(routes, []))
    assert out["error"] == "unsupported_content_type" and out["content_type"] == "image/png"
    out = await F.fetch_web_payload({"url": "https://ex.com/doc.pdf"}, _ctx(routes, []))
    assert out["error"] == "unsupported_content_type" and "PDF" in out["message"]
    out = await F.fetch_web_payload({"url": "https://ex.com/feed"}, _ctx(routes, []))
    assert out["extractor"] == "raw" and "<feed>" in out["text"]                    # +xml 은 텍스트로 통과


@pytest.mark.asyncio
async def test_unresolvable_host_has_its_own_code_and_thin_content_is_flagged(store):
    async def dead(host):
        return []
    out = await F.fetch_web_payload({"url": "https://dead.example/x"}, F.FetchContext(requester=_requester({}, []), resolver=dead, cache_ttl_minutes=0))
    assert out["error"] == "unresolvable_host" and "다른 결과" in out["message"]
    widget = b"<html><body><article><h1>Busan</h1><p>Loading...</p><div>Advertisement</div></article></body></html>"
    out = await F.fetch_web_payload({"url": "https://ex.com/w"}, _ctx({"https://ex.com/w": (200, {"Content-Type": "text/html"}, widget)}, []))
    assert out["thin_content"] is True and "스크립트" in out["note"]


def test_extract_falls_back_to_stdlib_parser(monkeypatch):
    from agent_arch.services.web import html_extract as H
    monkeypatch.setattr(H, "_PARSER", "no-such-parser")
    title, body = H.extract("<html><head><title>t</title></head><body><article><p>" + "본문 " * 50 + "</p></article></body></html>")
    assert title == "t" and "본문" in body


@pytest.mark.asyncio
async def test_thin_html_fetches_inline_script_data_endpoint():
    """실측(2026-09-28): 서울외국환중개 페이지 본문 490자, 인라인 $.ajax 주소에 오늘 매매기준율 — 그 데이터를 붙인다."""
    page = ("<html><body><p>환율 조회</p><script>$.ajax({ url: \"/Flash/TodayExRate_flash.jsp\", data: \"tr_date=2026-09-28\" });"
            "</script></body></html>").encode()
    data = b"?test0=test&USD=1,352.00&EUR=1,538.04&diff3=-18.89"
    log = []
    routes = {"https://ex.com/ExRate/TodayExRate.jsp": (200, {"Content-Type": "text/html; charset=utf-8"}, page),
              "https://ex.com/Flash/TodayExRate_flash.jsp?tr_date=2026-09-28": (200, {"Content-Type": "text/html"}, data)}
    out = await F.fetch_web_payload({"url": "https://ex.com/ExRate/TodayExRate.jsp"}, _ctx(routes, log))
    assert "EUR=1,538.04" in out["text"] and "페이지가 불러오는 데이터" in out["text"] and out["extractor"].endswith("+page-data")
    assert log[-1]["url"] == "https://ex.com/Flash/TodayExRate_flash.jsp?tr_date=2026-09-28"
