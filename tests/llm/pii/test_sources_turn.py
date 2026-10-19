"""어휘 소스 — 디렉터리 페이징·요청자별 암호화 캐시·시간 상한, 도구 결과 수확 규칙, 턴 어휘 조립, 예열 대기."""
from agent_arch.extensions import extensions
import asyncio
from types import SimpleNamespace

import pytest

from agent_arch.agents.context import set_conversation_id, set_user_context
from agent_arch.core.config import settings
from agent_arch.llm.pii import sources, turn
from agent_arch.llm.pii.tokens import Kind


class _Store:
    def __init__(self):
        self.kv, self.ttl = {}, {}

    async def get(self, k):
        return self.kv.get(k)

    async def set(self, k, v, expire=None):
        self.kv[k] = v
        self.ttl[k] = expire
        return True

    async def delete(self, k):
        self.kv.pop(k, None)
        return True


@pytest.fixture
def store(monkeypatch):
    import agent_arch.memory.redis_store as rs
    s = _Store()
    monkeypatch.setattr(rs, "redis_store", s)
    monkeypatch.setattr(settings, "pii_guard_secret", "unit-secret", raising=False)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "(주)회사", raising=False)
    return s


class _Client:
    enabled = True
    _company = "(주)회사"

    def __init__(self, people_by_company, companies=None, delay=0.0):
        self.people, self.companies, self.delay, self.calls = people_by_company, companies, delay, []

    async def list_companies(self):
        return self.companies if self.companies is not None else [{"company": c, "company_name": f"{c}명"} for c in self.people]

    async def search_colleagues(self, query="", limit=20, department=None, company=None, offset=0):
        self.calls.append((company, offset))
        if self.delay:
            await asyncio.sleep(self.delay)
        rows = self.people.get(company, [])
        return [{"name": n, "email": e, "company": company} for n, e in rows[offset:offset + limit]]


# ── 암호화 ───────────────────────────────────────────────────────────────────
def test_encrypt_roundtrip_and_ciphertext_hides_values(monkeypatch):
    monkeypatch.setattr(settings, "pii_guard_secret", "k1", raising=False)
    token = sources.encrypt_json({"persons": [["김보라", "user@example.com"]]})
    assert "김보라" not in token and "bkim" not in token
    assert sources.decrypt_json(token) == {"persons": [["김보라", "user@example.com"]]}
    monkeypatch.setattr(settings, "pii_guard_secret", "k2", raising=False)
    with pytest.raises(Exception):
        sources.decrypt_json(token)                                     # 다른 비밀로는 못 푼다


# ── 디렉터리 ──────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fetch_directory_pages_every_company():
    people = {"A": [(f"가{i:03d}", f"a{i}@x.kr") for i in range(120)], "B": [("김보라", "bkim@x.kr"), ("", "noname@x.kr")]}
    client = _Client(people)
    data = await sources.fetch_directory(client)
    assert len(data["persons"]) == 122 and ["김보라", "bkim@x.kr"] in data["persons"] and ["", "noname@x.kr"] in data["persons"]
    assert data["companies"] == ["A", "A명", "B", "B명"]
    assert client.calls == [("A", 0), ("A", 50), ("A", 100), ("B", 0)]  # 50 미만이 오면 다음 회사


@pytest.mark.asyncio
async def test_directory_lexicon_caches_encrypted_per_user(store, monkeypatch):
    client = _Client({"A": [("김보라", "bkim@x.kr")]})
    monkeypatch.setattr(extensions, "directory_client", lambda: client)
    data = await sources.directory_lexicon("u1")
    assert data["persons"] == [["김보라", "bkim@x.kr"]]
    key = sources.USER_KEY.format(sub="u1")
    assert key in store.kv and "김보라" not in store.kv[key] and store.ttl[key] == settings.pii_guard_lexicon_ttl
    calls = len(client.calls)
    assert await sources.directory_lexicon("u1") == data and len(client.calls) == calls     # 캐시 적중
    assert await sources.directory_lexicon(None) == {"persons": [], "companies": []}


@pytest.mark.asyncio
async def test_directory_lexicon_timeout_and_failure_are_soft(store, monkeypatch):
    slow = _Client({"A": [("김보라", "bkim@x.kr")]}, delay=0.2)
    monkeypatch.setattr(extensions, "directory_client", lambda: slow)
    assert await sources.directory_lexicon("u1", timeout=0.05) == {"persons": [], "companies": []}
    assert sources.USER_KEY.format(sub="u1") not in store.kv

    class _Boom:
        enabled = True

        async def list_companies(self):
            raise RuntimeError("directory down")

    monkeypatch.setattr(extensions, "directory_client", lambda: _Boom())
    assert await sources.directory_lexicon("u2") == {"persons": [], "companies": []}
    monkeypatch.setattr(extensions, "directory_client", lambda: SimpleNamespace(enabled=False))
    assert await sources.directory_lexicon("u3") == {"persons": [], "companies": []}


# ── 수확 ─────────────────────────────────────────────────────────────────────
def test_harvest_persons_rules():
    payload = {
        "colleagues": [{"name": "김보라", "email": "user@example.com", "company": "(주)회사"}, {"display_name": "John Smith"}],
        "messages": [{"sender": "박미래", "sender_email": "user@example.com", "file_name": "보고서.pdf", "tool_name": "mail_list"}],
        "assignee": {"name": "이수", "mail": "user@example.com"},
        "count": 3, "company_name": "김보라식품", "channel_name": "영업팀", "name": "결재 요청서", "url": "https://x.com/u/jsbaek",
    }
    got = set(sources.harvest_persons(payload))
    assert got == {("김보라", "user@example.com"), ("John Smith", ""), ("박미래", "user@example.com"), ("이수", "user@example.com")}
    assert sources.harvest_from_text("[ERROR] 실패") == [] and sources.harvest_from_text('"그냥 문자열"') == []


@pytest.mark.asyncio
async def test_remember_and_read_harvest_merges_and_caps(store, monkeypatch):
    monkeypatch.setattr(sources, "_MAX_HARVEST", 3)
    await sources.remember_harvest("c1", [("김보라", "bkim@x.kr")])
    await sources.remember_harvest("c1", [("김보라", "bkim@x.kr"), ("박미래", ""), ("이수", ""), ("최정", "")])
    got = await sources.harvested_lexicon("c1")
    assert got == [("박미래", ""), ("이수", ""), ("최정", "")]                   # 최근 3명만, 중복 제거
    assert store.ttl[sources.CONV_KEY.format(conv="c1")] == sources.CONV_TTL
    assert await sources.harvested_lexicon(None) == [] and await sources.harvested_lexicon("none") == []
    await sources.harvest_tool_result("c2", '{"colleagues":[{"name":"김보라","email":"bkim@x.kr"}]}', "(주)회사")
    assert await sources.harvested_lexicon("c2") == [("김보라", "bkim@x.kr")]
    await sources.harvest_tool_result("c3", '{"colleagues":[{"name":"김보라","email":"bkim@x.kr"}]}', "안켠회사")
    assert await sources.harvested_lexicon("c3") == []                     # 회사가 안 켰으면 남기지 않는다


# ── 턴 어휘 조립 ─────────────────────────────────────────────────────────────
@pytest.fixture
def ctx():
    set_user_context({"sub": "u1", "name": "백지선", "email": "user@example.com", "username": "jsbaek", "company": "(주)회사"})
    set_conversation_id("c1")
    yield
    set_user_context({})
    set_conversation_id(None)


@pytest.mark.asyncio
async def test_build_turn_lexicon_composes_sources(ctx, monkeypatch):
    async def fake_dir(sub, timeout=2.0):
        assert sub == "u1"
        return {"persons": [["김보라", "user@example.com"], ["이수", "user@example.com"]], "companies": ["(주)회사", "회사SB"]}

    async def fake_harv(conv):
        assert conv == "c1"
        return [("박미래", "user@example.com")]

    monkeypatch.setattr(sources, "directory_lexicon", fake_dir)
    monkeypatch.setattr(sources, "harvested_lexicon", fake_harv)
    lex = await turn.build_turn_lexicon()
    forms = {(a.form, a.text, a.source) for a in lex.aliases}
    assert ("full_name", "김보라", "directory") in forms and ("given_name", "보라", "directory") not in forms
    assert ("full_name", "박미래", "tool") in forms and ("given_name", "미래", "tool") in forms
    assert ("given_name", "지선", "requester") in forms and ("handle", "jsbaek", "requester") in forms
    assert lex.company_names == {"(주)회사", "회사SB"}
    assert lex.pattern.search("김보라님") and lex.pattern.search("user@example.com")


@pytest.mark.asyncio
async def test_build_turn_lexicon_falls_back_to_requester_on_failure(ctx, monkeypatch):
    async def boom(sub, timeout=2.0):
        raise RuntimeError("redis down")

    monkeypatch.setattr(sources, "directory_lexicon", boom)
    lex = await turn.build_turn_lexicon()
    assert {a.text for a in lex.aliases if a.kind is Kind.PERSON} == {"백지선", "지선"}


@pytest.mark.asyncio
async def test_prewarm_task_is_awaited_briefly_then_cache_used(ctx, monkeypatch, store):
    client = _Client({"A": [("김보라", "bkim@x.kr")]}, delay=0.05)
    monkeypatch.setattr(extensions, "directory_client", lambda: client)
    turn.prewarm()
    task = turn._prewarm_task.get()
    assert task is not None
    lex = await turn.build_turn_lexicon()
    assert task.done() and "김보라" in {a.text for a in lex.aliases}
    assert len(client.calls) == 1                                          # 예열이 받은 캐시를 그대로 씀


@pytest.mark.asyncio
async def test_prewarm_slow_does_not_block_turn(ctx, monkeypatch, store):
    monkeypatch.setattr(turn, "_WAIT_SECONDS", 0.05)
    client = _Client({"A": [("김보라", "bkim@x.kr")]}, delay=0.3)
    monkeypatch.setattr(extensions, "directory_client", lambda: client)
    turn.prewarm()
    lex = await turn.build_turn_lexicon()
    assert "김보라" not in {a.text for a in lex.aliases} and "백지선" in {a.text for a in lex.aliases}
    await turn._prewarm_task.get()                                           # 예열은 끝까지 돌아 캐시를 채운다
    assert sources.USER_KEY.format(sub="u1") in store.kv


# ── 회사 설정 게이트 ─────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_company_setting_enabled_with_last_known(store, monkeypatch):
    from agent_arch.llm.pii import policy
    from agent_arch.services.web import guard as G

    answers = {"(주)회사": {"pii_guard": True, "web_search": False}}

    async def fake_features(company):
        return answers.get(company, {})

    monkeypatch.setattr(G, "fetch_company_agent_features", fake_features)
    monkeypatch.setattr(settings, "pii_guard_mode", "enforce", raising=False)
    monkeypatch.setattr(settings, "pii_guard_force_companies", "", raising=False)
    assert await policy.resolve_mode("gemini", "(주)회사") == "enforce"
    key = policy.LAST_KNOWN_KEY.format(company="(주)회사")
    assert store.kv[key] is True and store.ttl[key] == policy.LAST_KNOWN_TTL
    answers["(주)회사"] = {}                                                # 조회 실패 → 마지막 켬 유지
    assert await policy.resolve_mode("gemini", "(주)회사") == "enforce"
    answers["(주)회사"] = {"pii_guard": False}                              # 명시 끔 → 즉시 반영 + 기억 삭제
    assert await policy.resolve_mode("gemini", "(주)회사") == "off" and key not in store.kv
    answers["(주)회사"] = {}
    assert await policy.resolve_mode("gemini", "(주)회사") == "off"
    assert await policy.resolve_mode("gemini", "처음회사") == "off"          # 키 없음 + 기억 없음 = 꺼짐(opt-in)


def test_rules_for_settings_adds_empno(monkeypatch):
    from agent_arch.llm.pii import policy
    from agent_arch.llm.pii.patterns import RULES

    monkeypatch.setattr(settings, "pii_guard_empno_regex", "", raising=False)
    assert policy.rules_for_settings() is RULES
    monkeypatch.setattr(settings, "pii_guard_empno_regex", r"E-\d{4}", raising=False)
    rules = policy.rules_for_settings()
    assert len(rules) == len(RULES) + 1 and rules[-1].kind is Kind.EMPNO
    monkeypatch.setattr(settings, "pii_guard_empno_regex", r"E-(\d{4}", raising=False)
    assert policy.rules_for_settings() is RULES


@pytest.mark.asyncio
async def test_prewarm_skips_companies_not_opted_in(ctx, monkeypatch, store):
    monkeypatch.setattr(settings, "pii_guard_force_companies", "다른회사", raising=False)
    client = _Client({"A": [("김보라", "bkim@x.kr")]})
    monkeypatch.setattr(extensions, "directory_client", lambda: client)
    turn.prewarm()
    await turn._prewarm_task.get()
    assert client.calls == [] and sources.USER_KEY.format(sub="u1") not in store.kv
