"""BraveWebTool — 원격 호출 앞뒤의 문지기·정책 인자·결과 정규화. 전송 계층은 _call_remote 패치로 격리."""
import json
from types import SimpleNamespace

import pytest

from agent_arch.agents.context import set_user_context, set_web_search_enabled
from agent_arch.agents.permissions.spec import GateMode, PermissionRequirement
from agent_arch.agents.tools.mcp.brave_web import BraveWebTool
from agent_arch.core.config import settings

_PERM = PermissionRequirement(service="web", mode=GateMode.PUBLIC)
_SCHEMA = {"properties": {"query": {"type": "string"}, "country": {"type": "string", "default": "US"}, "count": {"type": "integer"},
                          "safesearch": {"type": "string"}, "ui_lang": {"type": "string", "title": "UI"}},
           "required": ["query", "safesearch"]}


class _FakeRedis:
    def __init__(self):
        self.kv, self.lists = {}, {}
    async def incr(self, k):
        self.kv[k] = self.kv.get(k, 0) + 1; return self.kv[k]
    async def expire(self, k, s):
        return True
    async def lpush(self, k, v):
        self.lists.setdefault(k, []).insert(0, v)
    async def ltrim(self, k, a, b):
        self.lists[k] = self.lists.get(k, [])[a:b + 1]


def _tool(name="brave_web_search", **kw):
    args = dict(server_url="http://brave:8080/mcp", service_token="", name=name, description="d", input_schema=_SCHEMA, permission=_PERM,
                exposure="retrieved", prompt_guidance="### 웹 검색 원칙", hidden_params={"safesearch", "ui_lang"},
                default_args={"country": "KR", "search_lang": "ko", "count": 8}, forced_args={"safesearch": "moderate", "text_decorations": False})
    args.update(kw)
    return BraveWebTool(**args)


def _result(texts=(), structured=None, is_error=False):
    return SimpleNamespace(is_error=is_error, structured_content=structured, content=[SimpleNamespace(text=t) for t in texts])


@pytest.fixture
def env(monkeypatch):
    import agent_arch.memory.redis_store as rs
    fake = _FakeRedis(); monkeypatch.setattr(rs, "redis_store", SimpleNamespace(redis=fake))
    monkeypatch.setattr(settings, "web_search_allowed_companies", "", raising=False)
    monkeypatch.setattr(settings, "web_search_daily_quota", 0, raising=False)
    set_user_context({"sub": "u1", "email": "user@example.com", "company": "(주)회사"})
    set_web_search_enabled(True)                     # 패널 토글 ∧ 회사 설정 통과한 요청
    return fake


def test_policy_shapes_exposed_parameters():
    t = _tool()
    assert set(t.parameters) == {"query", "country", "count"} and t.required_params == ["query"]      # 숨긴 인자는 노출·필수에서 빠진다
    assert t.prompt_guidance == "### 웹 검색 원칙" and t.exposure == "retrieved" and t.permission_required is _PERM
    assert t._merge_arguments({"query": "q", "count": 3, "safesearch": "off", "country": None}) == \
        {"country": "KR", "search_lang": "ko", "count": 3, "query": "q", "safesearch": "moderate", "text_decorations": False}


@pytest.mark.asyncio
async def test_search_call_merges_policy_args_and_normalizes(env, monkeypatch):
    t = _tool(); seen = {}
    async def fake_call(arguments, headers):
        seen.update(arguments); seen["headers"] = headers
        return _result(texts=[json.dumps({"url": "https://a.kr/1", "title": "A <strong>x</strong>", "description": "본문 &amp; 요약"})])
    monkeypatch.setattr(t, "_call_remote", fake_call)
    res = await t.execute(query="폴리머 가격", safesearch="off")
    assert res.success and res.output["kind"] == "results" and res.output["count"] == 1
    assert seen["country"] == "KR" and seen["safesearch"] == "moderate" and seen["count"] == 8 and seen["headers"].get("X-User-Email") == "user@example.com"
    assert res.output["results"][0]["title"] == "A x" and "본문 & 요약" in res.output["results"][0]["content"]
    assert res.metadata == {"tool": "brave_web_search", "kind": "results", "count": 1}
    audit = [json.loads(x) for x in env.lists["web:audit:(주)회사"]]
    assert audit[0]["outcome"] == "ok:1" and audit[0]["query"] == "폴리머 가격"


@pytest.mark.asyncio
async def test_context_call_uses_structured_content(env, monkeypatch):
    t = _tool(name="brave_llm_context")
    structured = {"grounding": {"generic": [{"url": "https://p.kr/x", "title": "PP 가격", "snippets": ["PP 4.0% 상승"]}], "map": []},
                  "sources": {"https://p.kr/x": {"title": "PP 가격", "hostname": "p.kr", "age": ["2026-09-01"]}}}
    async def fake_call(arguments, headers):
        return _result(texts=[json.dumps(structured)], structured=structured)
    monkeypatch.setattr(t, "_call_remote", fake_call)
    res = await t.execute(query="PP 가격")
    assert res.success and res.output["kind"] == "context" and res.output["results"][0]["published"] == "2026-09-01"
    assert "PP 4.0% 상승" in res.output["results"][0]["content"] and "보안 안내" in res.output["results"][0]["content"]


@pytest.mark.asyncio
async def test_gate_blocks_before_remote_call(env, monkeypatch):
    t = _tool(); called = []
    async def fake_call(arguments, headers):
        called.append(arguments); return _result()
    monkeypatch.setattr(t, "_call_remote", fake_call)
    res = await t.execute(query="user@example.com 메일")
    assert res.success is False and res.output["error"] == "blocked_identifier" and res.metadata["code"] == "blocked_identifier" and called == []
    monkeypatch.setattr(settings, "web_search_allowed_companies", "다른회사", raising=False)
    res = await t.execute(query="폴리머")
    assert res.success is False and res.output["error"] == "web_search_disabled" and called == []
    res = await t.execute(query="   ")
    assert res.success is False and res.metadata["reason"] == "input"


@pytest.mark.asyncio
async def test_no_results_is_zero_rows_and_transport_failure_is_structured(env, monkeypatch):
    t = _tool()
    async def none_call(arguments, headers):
        return _result(texts=["No web results found"], is_error=True)
    monkeypatch.setattr(t, "_call_remote", none_call)
    res = await t.execute(query="zzqx")
    assert res.success and res.output["count"] == 0 and "결과 없음" in res.output["note"]
    async def boom(arguments, headers):
        raise ConnectionError("down")
    monkeypatch.setattr(t, "_call_remote", boom)
    res = await t.execute(query="폴리머")
    assert res.success is False and res.output["error"] == "provider_unavailable" and res.metadata["code"] == "provider_unavailable"
    assert [json.loads(x)["outcome"] for x in env.lists["web:audit:(주)회사"]][:2] == ["transport_error", "ok:0"]



@pytest.mark.asyncio
async def test_off_request_refuses_before_remote_call(env, monkeypatch):
    """웹 검색 opt-in 이 꺼진 요청(기본)에서는 도구가 불려도 원격 호출 없이 거절한다 — sticky·직접 지정 경로의 마지막 방어선."""
    set_web_search_enabled(False)
    t = _tool(); called = []
    async def fake_call(arguments, headers):
        called.append(arguments); return _result()
    monkeypatch.setattr(t, "_call_remote", fake_call)
    res = await t.execute(query="폴리머 가격")
    assert res.success is False and res.output["error"] == "web_search_off" and "패널의 웹 검색" in res.error and called == []
    assert json.loads(env.lists["web:audit:(주)회사"][0])["outcome"] == "web_search_off"


def test_out_of_enum_value_falls_back_to_default():
    """실측(2026-09-23): search_lang="ko-hkr" 오타 하나로 서버가 호출 전체를 -32602 로 거절했다 → enum 밖 값은 비운 것으로."""
    schema = {"type": "object", "required": ["query"], "properties": {
        "query": {"type": "string"}, "search_lang": {"type": "string", "enum": ["ko", "en", "jp"]}, "count": {"type": "integer"}}}
    t = _tool(input_schema=schema)
    assert t._merge_arguments({"query": "q", "search_lang": "ko-hkr"})["search_lang"] == "ko"      # 기본값으로
    assert t._merge_arguments({"query": "q", "search_lang": "en"})["search_lang"] == "en"           # 허용값은 모델 선택 유지
    assert t._merge_arguments({"query": "q", "count": 3})["count"] == 3                             # enum 없는 인자는 그대로


# ─── 계측: Brave 는 성공한 요청 건수로 청구한다 ─────────────────────────────────

@pytest.fixture
def meters(monkeypatch):
    import agent_arch.agents.tools.mcp.brave_web as bw
    seen = []
    monkeypatch.setattr(bw, "record_meter", lambda meter, **kw: seen.append({"meter": meter, **kw}))
    return seen


@pytest.mark.asyncio
async def test_successful_search_is_metered_once(env, monkeypatch, meters):
    t = _tool()
    async def fake_call(arguments, headers):
        return _result(texts=[json.dumps({"url": "https://a.kr/1", "title": "A", "description": "d"})])
    monkeypatch.setattr(t, "_call_remote", fake_call)
    assert (await t.execute(query="환율")).success
    assert meters == [{"meter": "web_search_call", "provider": "brave", "model_name": "brave-search",
                       "quantity": 1, "source": "brave_mcp.brave_web_search"}]


@pytest.mark.asyncio
async def test_failed_or_blocked_search_is_not_metered(env, monkeypatch, meters):
    t = _tool()
    async def boom(arguments, headers):
        raise ConnectionError("down")
    monkeypatch.setattr(t, "_call_remote", boom)
    assert not (await t.execute(query="환율")).success
    set_web_search_enabled(False)                    # 토글이 꺼진 요청 — 요청 자체가 안 나간다
    assert not (await t.execute(query="환율")).success
    assert meters == []
