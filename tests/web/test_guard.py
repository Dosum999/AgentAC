"""웹 문지기 — 회사 허용·식별자 차단·일일 한도·감사 원장."""
import json
from types import SimpleNamespace

import pytest

from agent_arch.core.config import settings
from agent_arch.services.web import guard as G


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


def test_company_allowed_and_identifier_block(monkeypatch):
    monkeypatch.setattr(settings, "web_search_allowed_companies", "", raising=False)
    assert G.company_allowed(None) and G.company_allowed("아무회사")
    monkeypatch.setattr(settings, "web_search_allowed_companies", "(주)회사, 테스트 고객사", raising=False)
    assert G.company_allowed("(주)회사") and G.company_allowed("테스트 고객사") and not G.company_allowed("성호금속") and not G.company_allowed(None)
    user = {"email": "user@example.com", "name": "신현수", "username": "hshin"}
    assert G.blocked_identifier("user@example.com 메일 찾아", user) == "email"
    assert G.blocked_identifier("신현수 연락처", user) == "name"
    assert G.blocked_identifier("hshin 계정", user) in ("username", "account")
    assert G.blocked_identifier("폴리머 가격 동향", user) is None
    assert G.blocked_identifier("", {}) is None


@pytest.mark.asyncio
async def test_precheck_order_quota_and_audit(monkeypatch):
    import agent_arch.memory.redis_store as rs
    fake = _FakeRedis(); monkeypatch.setattr(rs, "redis_store", SimpleNamespace(redis=fake))
    monkeypatch.setattr(settings, "web_search_allowed_companies", "(주)회사", raising=False)
    monkeypatch.setattr(settings, "web_search_daily_quota", 2, raising=False)
    other = {"sub": "u2", "email": "x@a.kr", "company": "성호금속"}
    blocked = await G.precheck("폴리머", other, "brave_web_search")
    assert blocked["error"] == "web_search_disabled"
    user = {"sub": "u1", "email": "user@example.com", "company": "(주)회사"}
    blocked = await G.precheck("user@example.com", user, "brave_web_search")
    assert blocked["error"] == "blocked_identifier" and blocked["kind"] == "email"
    assert await G.precheck("폴리머 1", user, "brave_web_search") is None
    assert await G.precheck("폴리머 2", user, "brave_llm_context") is None
    over = await G.precheck("폴리머 3", user, "brave_web_search")
    assert over["error"] == "quota_exceeded"
    audit = [json.loads(x) for x in fake.lists["web:audit:(주)회사"]]
    assert [a["outcome"] for a in audit] == ["quota_exceeded", "blocked_identifier:email"]       # 통과한 호출은 어댑터가 결과와 함께 기록
    assert audit[0]["tool"] == "brave_web_search" and audit[0]["email"] == "user@example.com"
    assert [json.loads(x)["outcome"] for x in fake.lists["web:audit:성호금속"]] == ["company_disabled"]
    await G.audit("(주)회사", user, "brave_web_search", "폴리머 1", "ok:5")
    assert json.loads(fake.lists["web:audit:(주)회사"][0])["outcome"] == "ok:5"


@pytest.mark.asyncio
async def test_quota_fail_open_without_redis(monkeypatch):
    import agent_arch.memory.redis_store as rs
    monkeypatch.setattr(rs, "redis_store", SimpleNamespace(redis=None))
    monkeypatch.setattr(settings, "web_search_allowed_companies", "", raising=False)
    monkeypatch.setattr(settings, "web_search_daily_quota", 1, raising=False)
    user = {"sub": "u", "email": "a@b.kr", "company": "C"}
    assert await G.precheck("q1", user, "t") is None and await G.precheck("q2", user, "t") is None   # Redis 없으면 한도 미적용
    monkeypatch.setattr(settings, "web_search_daily_quota", "bad", raising=False)
    assert G.daily_quota() == 0


@pytest.mark.asyncio
async def test_resolve_web_search_requires_toggle_and_company_setting(monkeypatch):
    calls = []
    async def fake_features(company):
        calls.append(company); return {"web_search": company == "켠회사", "answer_mode": False}
    monkeypatch.setattr(G, "fetch_company_agent_features", fake_features)
    monkeypatch.setattr(settings, "web_search_allowed_companies", "", raising=False)
    assert await G.resolve_web_search(None, "켠회사") is False and calls == []              # 토글 없으면 조회조차 않는다
    assert await G.resolve_web_search(False, "켠회사") is False
    assert await G.resolve_web_search(True, "켠회사") is True and await G.resolve_web_search(True, "끈회사") is False
    assert await G.resolve_web_search(True, None) is False
    # .env 허용 목록이 있으면 그것이 회사 설정을 대신한다(운영 강제 스위치)
    monkeypatch.setattr(settings, "web_search_allowed_companies", "끈회사", raising=False)
    calls.clear()
    assert await G.resolve_web_search(True, "끈회사") is True and await G.resolve_web_search(True, "켠회사") is False and calls == []


@pytest.mark.asyncio
async def test_fetch_company_agent_features_calls_common_app_and_caches(monkeypatch):
    import agent_arch.memory.redis_store as rs
    store = {}
    class _Store:
        async def get(self, k): return store.get(k)
        async def set(self, k, v, expire=None): store[k] = v; store["_ttl"] = expire; return True
    monkeypatch.setattr(rs, "redis_store", _Store())
    monkeypatch.setattr(settings, "frappe_api_key", "k", raising=False)
    monkeypatch.setattr(settings, "frappe_api_secret", "s", raising=False)
    monkeypatch.setattr(settings, "common_app_url", "https://common.test", raising=False)
    seen = {}
    class _Resp:
        def raise_for_status(self): return None
        def json(self): return {"message": {"web_search": 1, "answer_mode": 0}}
    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url, params=None, headers=None):
            seen.update(url=url, params=params, headers=headers); return _Resp()
    monkeypatch.setattr(G.httpx, "AsyncClient", _Client)
    f = await G.fetch_company_agent_features("(주)회사")
    assert f == {"web_search": True, "answer_mode": False}
    assert seen["url"].endswith("/api/method/" + G._FEATURES_METHOD) and seen["params"] == {"company": "(주)회사"} and seen["headers"]["Authorization"] == "token k:s"
    assert store["company_agent_features:v2:(주)회사"] == f and store["_ttl"] == settings.company_providers_cache_ttl
    seen.clear()
    assert await G.fetch_company_agent_features("(주)회사") == f and seen == {}                  # 캐시 적중 — 재호출 없음
    assert await G.fetch_company_agent_features(None) == {}
    # 키 미설정 → 닫힘
    monkeypatch.setattr(settings, "frappe_api_key", None, raising=False)
    assert await G.fetch_company_agent_features("다른회사") == {}
