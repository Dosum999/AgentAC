"""웹 문지기 × 가명화 — 어휘(임직원·수확 이름)가 검색어에 있으면 막고, 감사 원장의 검색어는 토큰으로 남긴다."""
import json
from types import SimpleNamespace

import pytest

from agent_arch.agents.context import set_conversation_id, set_user_context
from agent_arch.core.config import settings
from agent_arch.llm.pii import sources
from agent_arch.services.web import guard as G


class _FakeRedis:
    def __init__(self):
        self.lists = {}

    async def incr(self, k):
        return 1

    async def expire(self, k, s):
        return True

    async def lpush(self, k, v):
        self.lists.setdefault(k, []).insert(0, v)

    async def ltrim(self, k, a, b):
        self.lists[k] = self.lists.get(k, [])[a:b + 1]


@pytest.fixture
def env(monkeypatch):
    import agent_arch.memory.redis_store as rs
    fake = _FakeRedis()
    monkeypatch.setattr(rs, "redis_store", SimpleNamespace(redis=fake))
    monkeypatch.setattr(settings, "web_search_allowed_companies", "(주)회사", raising=False)
    monkeypatch.setattr(settings, "web_search_daily_quota", 0, raising=False)
    monkeypatch.setattr(settings, "pii_guard_secret", "s", raising=False)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "(주)회사", raising=False)

    async def fake_dir(sub, timeout=2.0):
        return {"persons": [["김보라", "user@example.com"]], "companies": ["(주)회사"]}

    async def fake_harv(conv):
        return [("박미래", "")]

    monkeypatch.setattr(sources, "directory_lexicon", fake_dir)
    monkeypatch.setattr(sources, "harvested_lexicon", fake_harv)
    set_user_context({"sub": "u1", "email": "user@example.com", "name": "백지선", "company": "(주)회사"})
    set_conversation_id("c1")
    yield fake
    set_user_context({})
    set_conversation_id(None)


USER = {"sub": "u1", "email": "user@example.com", "name": "백지선", "company": "(주)회사"}


@pytest.mark.asyncio
async def test_lexicon_names_blocked_only_when_guard_on(env, monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_mode", "off", raising=False)
    assert await G.precheck("김보라 연락처", USER, "brave_web_search") is None          # 기존 동작: 제3자 이름은 통과
    monkeypatch.setattr(settings, "pii_guard_mode", "shadow", raising=False)
    blocked = await G.precheck("김보라 연락처", USER, "brave_web_search")
    assert blocked["error"] == "blocked_identifier" and blocked["kind"] == "person"
    blocked = await G.precheck("박미래 대리 이력", USER, "brave_web_search")
    assert blocked and blocked["kind"] == "person"
    assert await G.precheck("폴리머 가격 동향", USER, "brave_web_search") is None
    assert await G.precheck("김보라색 페인트", USER, "brave_web_search") is None        # 이름 경계


@pytest.mark.asyncio
async def test_audit_query_is_tokenized_when_guard_on(env, monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_mode", "shadow", raising=False)
    await G.audit("(주)회사", USER, "brave_web_search", "user@example.com 010-1234-5678 검색", "blocked_identifier:email")
    rec = json.loads(env.lists["web:audit:(주)회사"][0])
    assert "user@example.com" not in rec["query"] and "1234" not in rec["query"]
    assert rec["query"].count("[[") == 2 and rec["query"].endswith(" 검색")
    monkeypatch.setattr(settings, "pii_guard_mode", "off", raising=False)
    await G.audit("(주)회사", USER, "brave_web_search", "user@example.com 검색", "x")
    assert json.loads(env.lists["web:audit:(주)회사"][0])["query"] == "user@example.com 검색"   # off 면 종전 그대로
