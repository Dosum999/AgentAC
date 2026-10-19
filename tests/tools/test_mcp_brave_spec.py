"""브레이브 MCP 서버 스펙·정책 — 토큰 없이 URL 만으로 등록, 도구 3종 정책, 어댑터 클래스."""
from agent_arch.agents.tools.mcp import provider as P
from agent_arch.agents.tools.mcp.brave_web import BraveWebTool
from agent_arch.agents.tools.mcp.remote_tool import _sanitize_parameters
from agent_arch.core.config import settings


def test_brave_spec_registered_without_token(monkeypatch):
    monkeypatch.setattr(settings, "brave_mcp_url", "http://agent-brave-mcp:8080/mcp", raising=False)
    monkeypatch.setattr(settings, "brave_mcp_token", None, raising=False)
    specs = {s.name: s for s in P.server_specs()}
    assert "brave" in specs and specs["brave"].token_required is False and specs["brave"].token == ""
    assert set(specs["brave"].policies) == {"brave_llm_context", "brave_web_search", "brave_news_search"}
    monkeypatch.setattr(settings, "brave_mcp_url", "", raising=False)
    assert "brave" not in {s.name for s in P.server_specs()}
    # 다른 서버는 여전히 토큰이 필요하다
    monkeypatch.setattr(settings, "mailcow_mcp_url", "http://m/mcp", raising=False)
    monkeypatch.setattr(settings, "mailcow_mcp_token", None, raising=False)
    assert "mailcow" not in {s.name for s in P.server_specs()}


def test_brave_policies_shape(monkeypatch):
    monkeypatch.setattr(settings, "web_search_default_country", "KR", raising=False)
    monkeypatch.setattr(settings, "web_search_default_language", "ko", raising=False)
    pol = P._brave_policies()
    for name, p in pol.items():
        assert p.tool_cls is BraveWebTool and p.exposure == "retrieved" and p.permission.service == "web"
        assert p.permission.mode.name == "PUBLIC" and "반드시 준수" in p.prompt_guidance and "사내" in p.prompt_guidance
        assert p.default_args["country"] == "KR" and p.default_args["search_lang"] == "ko"
    ctx = pol["brave_llm_context"]
    assert "x-loc-lat" in ctx.hidden_params and "api-version" in ctx.hidden_params and ctx.forced_args["enable_local"] is False
    assert ctx.default_args["maximum_number_of_urls"] == 10 and ctx.default_args["maximum_number_of_tokens"] == 8192
    assert "freshness" in ctx.hidden_params and "freshness" in pol["brave_web_search"].hidden_params      # 기간 필터는 뉴스만
    assert "freshness" not in pol["brave_news_search"].hidden_params
    assert pol["brave_web_search"].default_args["count"] == 10
    web = pol["brave_web_search"]
    assert web.forced_args["text_decorations"] is False and web.forced_args["safesearch"] == "moderate" and "summary" in web.hidden_params
    assert "query" not in web.hidden_params and "count" not in web.hidden_params


def test_sanitize_hides_params():
    schema = {"properties": {"query": {"type": "string", "title": "Q"}, "ui_lang": {"type": "string"}, "count": {"type": "integer"}}, "required": ["query"]}
    assert set(_sanitize_parameters(schema, {"ui_lang"})) == {"query", "count"} and "title" not in _sanitize_parameters(schema)["query"]
