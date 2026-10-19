"""web_fetch 도구 — 등록·도메인, 게이트(웹 꺼짐·회사), 이번 턴에 본 URL 만 열기, 결과 갈래. 실측 배경은 도구 docstring(2026-09-23)."""
import json
from types import SimpleNamespace

import pytest

from agent_arch.agents import context as C
from agent_arch.agents.graphs.router import domain_from_tool_names
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.agents.tools.web_fetch import WebFetchTool
from agent_arch.core.config import settings
from agent_arch.services.web import brave_output


class _FakeRedis:
    def __init__(self):
        self.lists = {}
    async def lpush(self, k, v):
        self.lists.setdefault(k, []).insert(0, v)
    async def ltrim(self, k, a, b):
        pass
    async def get(self, k):
        return None
    async def set(self, *a, **k):
        return True


@pytest.fixture
def env(monkeypatch):
    import agent_arch.memory.redis_store as rs
    fake = _FakeRedis(); monkeypatch.setattr(rs, "redis_store", SimpleNamespace(redis=fake))
    monkeypatch.setattr(settings, "demo_mode", False, raising=False)
    monkeypatch.setattr(settings, "web_search_allowed_companies", "", raising=False)
    C.set_user_context({"sub": "u1", "email": "user@example.com", "company": "(주)회사"})
    C.set_web_search_enabled(True)                    # 턴 시작: 본 URL 집합도 새로 만든다
    return fake


def test_registered_retrieved_web_domain_and_counted_as_web():
    t = ToolRegistry.get("web_fetch")
    assert isinstance(t, WebFetchTool) and t.exposure == "retrieved" and t.get_required_params() == ["url"]
    assert domain_from_tool_names(["web_fetch"]) == "web" and "web_fetch" in brave_output.WEB_TOOLS
    assert "web_fetch" not in brave_output.BRAVE_TOOLS                              # 관문 생략 판정은 브레이브 3종만


@pytest.mark.asyncio
async def test_only_urls_seen_this_turn_can_be_opened(env, monkeypatch):
    import agent_arch.agents.tools.web_fetch as W
    async def fake_fetch(args, ctx):
        return {"url": args["url"], "final_url": args["url"], "status": 200, "extractor": "readability", "title": "EPDM", "text": "9월 가격"}
    monkeypatch.setattr(W, "fetch_web_payload", fake_fetch)
    t = WebFetchTool()
    res = await t.execute(url="https://www.imarcgroup.com/epdm-rubber-pricing-report")
    assert not res.success and res.output["error"] == "url_not_seen"                   # 검색 결과에 없던 주소
    C.remember_web_urls(["https://imarcgroup.com/epdm-rubber-pricing-report/?utm_source=x"])
    ok = await t.execute(url="https://www.imarcgroup.com/epdm-rubber-pricing-report")  # www·끝 슬래시·쿼리 차이는 같은 주소
    assert ok.success and ok.output["text"] == "9월 가격"
    assert any("url_not_seen" in v for v in env.lists.get("web:audit:(주)회사", []))  # 막힌 시도도 감사에 남는다


@pytest.mark.asyncio
async def test_web_off_blocks(env):
    C.set_web_search_enabled(False)
    res = await WebFetchTool().execute(url="https://a.com/")
    assert not res.success and res.output["error"] == "web_search_off"


def test_new_turn_resets_seen_urls():
    C.set_web_search_enabled(True); C.remember_web_urls(["https://a.com/x"])
    assert C.web_url_seen("https://a.com/x/")
    C.set_web_search_enabled(True)
    assert not C.web_url_seen("https://a.com/x")


@pytest.mark.asyncio
async def test_brave_results_register_urls(env, monkeypatch):
    from agent_arch.agents.tools.mcp.brave_web import BraveWebTool
    from agent_arch.agents.permissions import GateMode, PermissionRequirement
    t = BraveWebTool(server_url="http://brave/mcp", service_token="", name="brave_web_search", description="d",
                     input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                     permission=PermissionRequirement(service="web", mode=GateMode.PUBLIC))
    async def fake_call(arguments, headers):
        return SimpleNamespace(is_error=False, structured_content=None,
                               content=[SimpleNamespace(text=json.dumps({"url": "https://b.com/page", "title": "B", "description": "d"}))])
    monkeypatch.setattr(t, "_call_remote", fake_call)
    monkeypatch.setattr(settings, "web_search_daily_quota", 0, raising=False)
    assert (await t.execute(query="epdm price")).success
    assert C.web_url_seen("https://b.com/page")


def test_provenance_uses_fetched_final_url():
    from agent_arch.agents.react import provenance as P
    tr = [{"name": "web_fetch", "content": json.dumps({"url": "https://a.com/x", "final_url": "https://a.com/y", "title": "EPDM 9월",
                                                        "text": "본문"}, ensure_ascii=False)}]
    assert [s["url"] for s in P.extract_sources(tr)] == ["https://a.com/y"]


class _FakeCtxTool:
    def __init__(self, rows):
        self.rows, self.calls = rows, []
    async def execute(self, **kw):
        from agent_arch.agents.tools.base import ToolResult
        self.calls.append(kw)
        return ToolResult(success=True, output={"kind": "context", "results": self.rows})


def _patch(monkeypatch, fetch_result, rows):
    import agent_arch.agents.tools.web_fetch as W
    from agent_arch.agents.tools.registry import ToolRegistry
    async def fake_fetch(args, ctx):
        if isinstance(fetch_result, Exception):
            raise fetch_result
        return fetch_result
    monkeypatch.setattr(W, "fetch_web_payload", fake_fetch)
    ctx = _FakeCtxTool(rows)
    orig = ToolRegistry.get
    monkeypatch.setattr(ToolRegistry, "get", classmethod(lambda cls, n: ctx if n == "brave_llm_context" else orig(n)))
    return ctx


AP = "https://apnews.com/article/iran-us-strikes-hormuz-2026-cc891e"


@pytest.mark.asyncio
async def test_403_falls_back_to_search_index_copy_of_same_url(env, monkeypatch):
    """실측(2026-09-23): AP 403(Cloudflare) — '제목 site:호스트'로 인덱스 본문 조각을 받아 대체한다."""
    C.remember_web_urls([(AP, "Iran targets American allies in the Gulf")])
    ctx = _patch(monkeypatch, {"error": "http_error", "status": 403, "url": AP},
                 [{"url": "https://apnews.com/article/other", "content": "다른 기사"},
                  {"url": AP + "/", "title": "Iran targets", "content": "Iranian attacks have targeted U.S. allies", "published": "2026-09-02"}])
    res = await WebFetchTool().execute(url=AP)
    assert res.success and res.output["source"] == "search_index" and "Iranian attacks" in res.output["text"]
    assert "검색 인덱스 사본" in res.output["note"] and res.output["published"] == "2026-09-02"
    assert ctx.calls[0]["query"] == "Iran targets American allies in the Gulf site:apnews.com"


@pytest.mark.asyncio
async def test_challenge_page_with_200_also_falls_back(env, monkeypatch):
    C.remember_web_urls([AP])
    _patch(monkeypatch, {"url": AP, "final_url": AP, "status": 200, "title": "Just a moment...", "text": "..."},
           [{"url": AP, "title": "t", "content": "본문"}])
    res = await WebFetchTool().execute(url=AP)
    assert res.success and res.output["source"] == "search_index"


@pytest.mark.asyncio
async def test_no_index_copy_is_blocked_by_site_and_402_is_payment(env, monkeypatch):
    C.remember_web_urls([AP, "https://www.bloomberg.com/news/x"])
    _patch(monkeypatch, {"error": "http_error", "status": 403, "url": AP}, [{"url": "https://apnews.com/article/other", "content": "x"}])
    res = await WebFetchTool().execute(url=AP)
    assert not res.success and res.output["error"] == "blocked_by_site"                  # 다른 URL 로 대체하지 않는다
    _patch(monkeypatch, {"error": "http_error", "status": 402, "url": "https://www.bloomberg.com/news/x"}, [])
    paid = await WebFetchTool().execute(url="https://www.bloomberg.com/news/x")
    assert not paid.success and paid.output["error"] == "payment_required"


@pytest.mark.asyncio
async def test_connect_error_falls_back(env, monkeypatch):
    """koreaexim 인증서 체인 누락(ConnectError)도 인덱스로 대체."""
    u = "https://www.koreaexim.go.kr/wg/HPHKWG057M01"
    C.remember_web_urls([(u, "금리/환율정보")])
    _patch(monkeypatch, ConnectionError("ssl"), [{"url": u, "title": "금리/환율정보", "content": "환율표"}])
    res = await WebFetchTool().execute(url=u)
    assert res.success and res.output["text"] == "환율표"


@pytest.mark.asyncio
async def test_fetched_page_carries_fetched_at_and_kst_annotation(env, monkeypatch):
    import agent_arch.agents.tools.web_fetch as W
    u = "https://www.bloomberg.com/quote/KRWUSD:CUR"
    C.remember_web_urls([u])
    async def fake_fetch(args, ctx):
        return {"url": u, "final_url": u, "status": 200, "extractor": "readability", "title": "KRW-USD",
                "text": "0.0737 As of 7:33 PM EDT 09/27/26."}
    monkeypatch.setattr(W, "fetch_web_payload", fake_fetch)
    res = await WebFetchTool().execute(url=u)
    assert res.success and "[= 2026-09-28 08:33 KST]" in res.output["text"] and res.output["fetched_at"].endswith("KST")


def test_urls_in_previous_answers_can_be_opened():
    """후속 질의에서 앞 답의 씨티은행 URL 이 url_not_seen 으로 막혔다(2026-09-28) — 이전 답에 쓰인 주소는 연다."""
    from agent_arch.agents.graphs.react_graph import _remember_history_urls
    from agent_arch.llm.base import Message
    C.set_web_search_enabled(True)
    _remember_history_urls([Message(role="user", content="https://evil.example/x 열어"),
                            Message(role="assistant", content="출처: [씨티](https://www.citibank.co.kr/FxdExrt0100.act).")])
    assert C.web_url_seen("https://www.citibank.co.kr/FxdExrt0100.act") and not C.web_url_seen("https://evil.example/x")


@pytest.mark.asyncio
async def test_slow_page_hits_deadline_and_falls_back_to_index(env, monkeypatch):
    """Wise 가 가끔 27초 걸렸다(2026-09-28, 평소 1초) — 전체 마감을 넘기면 인덱스 사본으로."""
    import asyncio
    import agent_arch.agents.tools.web_fetch as W
    u = "https://wise.com/us/currency-converter/eur-to-krw-rate/history"
    C.remember_web_urls([(u, "Euro to KRW history")])
    ctx = _patch(monkeypatch, {}, [{"url": u, "title": "t", "content": "1 EUR = 1,548.29 KRW"}])
    async def slow(args, c):
        await asyncio.sleep(5)
    monkeypatch.setattr(W, "fetch_web_payload", slow)
    monkeypatch.setattr(settings, "web_fetch_deadline_seconds", 1.0, raising=False)
    res = await WebFetchTool().execute(url=u)
    assert res.success and res.output["source"] == "search_index" and "1,548.29" in res.output["text"]
