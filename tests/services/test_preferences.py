"""사용자 개인설정 (Redis 대화 이력 정책) 단위 테스트.

대상:
    - app.models.preferences.UserPreferencesUpdate 검증 규칙
    - app.services.preferences.service: Redis 캐시 hit/miss + negative cache + invalidate
    - app.memory.conversation.append_turn: prefs가 있을 때/없을 때 LTRIM/EXPIRE
    - app.api.v1.preferences: 4개 엔드포인트 동작 (행 없음 fallback, merge-patch, 401)

Repository(asyncpg)는 monkeypatch로 대체한다. Redis도 in-memory fake로 교체해
실제 외부 의존을 일으키지 않는다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from agent_arch.memory import conversation as conv_module
from agent_arch.models.preferences import UserPreferencesUpdate
from agent_arch.services.preferences import (
    UserPreferences,
    service as pref_service,
)


# ── 공용 fake 인프라 ──────────────────────────────────────────────────────────


class _FakeRedis:
    """in-memory dict 기반 redis_store 대체.

    실제 redis_store와 동일 인터페이스 중 본 모듈이 사용하는 것만 구현한다.
    """

    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}
        self.lists: Dict[str, List[str]] = {}
        self.expires: Dict[str, int] = {}

    async def get(self, key: str) -> Any:
        return self.kv.get(key)

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        # 실제 redis_store는 json.dumps 후 저장 → 동일하게 흉내내 직렬화 가능성 검증
        self.kv[key] = json.loads(json.dumps(value))
        if expire is not None:
            self.expires[key] = expire
        return True

    async def delete(self, key: str) -> bool:
        self.kv.pop(key, None)
        self.lists.pop(key, None)
        self.expires.pop(key, None)
        return True

    async def rpush(self, key: str, *values: Any) -> bool:
        self.lists.setdefault(key, []).extend(json.dumps(v) for v in values)
        return True

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        lst = self.lists.get(key, [])
        # Python 슬라이싱과 redis ltrim end-inclusive 차이 보정
        end_idx = None if end == -1 else end + 1
        self.lists[key] = lst[start:end_idx]
        return True

    async def expire(self, key: str, seconds: int) -> bool:
        self.expires[key] = seconds
        return True

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list:
        lst = self.lists.get(key, [])
        end_idx = None if end == -1 else end + 1
        return [json.loads(v) for v in lst[start:end_idx]]


@pytest.fixture
def fake_redis(monkeypatch):
    """redis_store를 in-memory fake로 교체한다.

    service.py, conversation.py 양쪽에서 import한 동일 객체를 둘 다 갈아끼운다.
    """
    fake = _FakeRedis()
    monkeypatch.setattr(pref_service, "redis_store", fake)
    monkeypatch.setattr(conv_module, "redis_store", fake)
    return fake


def _make_prefs(
    user_sub: str = "u-1",
    max_history: int = 40,
    session_ttl_seconds: int = 43200,
) -> UserPreferences:
    """테스트용 UserPreferences 인스턴스."""
    now = datetime(2026, 5, 13, 10, tzinfo=timezone.utc)
    return UserPreferences(
        user_sub=user_sub,
        max_history=max_history,
        session_ttl_seconds=session_ttl_seconds,
        created_at=now,
        updated_at=now,
    )


# ── 1. Pydantic 검증 ──────────────────────────────────────────────────────────


class TestUserPreferencesUpdate:
    """UserPreferencesUpdate의 검증 규칙."""

    def test_full_payload_ok(self):
        body = UserPreferencesUpdate(max_history=40, session_ttl_seconds=43200)
        assert body.max_history == 40
        assert body.session_ttl_seconds == 43200

    def test_partial_payload_ok(self):
        """merge-patch — 한 필드만 보내도 OK."""
        body = UserPreferencesUpdate(max_history=30)
        assert body.max_history == 30
        assert body.session_ttl_seconds is None

    def test_empty_payload_ok(self):
        """두 필드 모두 누락도 허용 (의미 있는 갱신은 없지만 422는 아님)."""
        body = UserPreferencesUpdate()
        assert body.max_history is None
        assert body.session_ttl_seconds is None

    @pytest.mark.parametrize("bad", [1, 3, 5, 99])
    def test_odd_max_history_rejected(self, bad):
        with pytest.raises(ValidationError):
            UserPreferencesUpdate(max_history=bad)

    @pytest.mark.parametrize("bad", [0, 1, 101, 102, -2])
    def test_out_of_range_max_history_rejected(self, bad):
        with pytest.raises(ValidationError):
            UserPreferencesUpdate(max_history=bad)

    @pytest.mark.parametrize("bad", [0, 599, 604801, -1])
    def test_out_of_range_ttl_rejected(self, bad):
        with pytest.raises(ValidationError):
            UserPreferencesUpdate(session_ttl_seconds=bad)


# ── 2. service 캐시 동작 ─────────────────────────────────────────────────────


class TestPreferenceCache:
    """service.get_for_user / invalidate."""

    @pytest.mark.asyncio
    async def test_db_miss_caches_negative_marker(self, fake_redis, monkeypatch):
        """미등록 사용자: DB miss → sentinel을 60s 캐시 → None 반환."""
        async def _find_none(_sub):
            return None
        monkeypatch.setattr(pref_service, "find_preferences", _find_none)

        result = await pref_service.get_for_user("u-missing")

        assert result is None
        cached = fake_redis.kv["pref:v1:u-missing"]
        assert cached == pref_service._NEGATIVE_MARKER
        assert fake_redis.expires["pref:v1:u-missing"] == 60

    @pytest.mark.asyncio
    async def test_db_hit_caches_payload(self, fake_redis, monkeypatch):
        """등록 사용자: DB 조회 결과 직렬화해 캐시 + 그 값을 반환."""
        prefs = _make_prefs(user_sub="u-1", max_history=40)
        async def _find_ok(_sub):
            return prefs
        monkeypatch.setattr(pref_service, "find_preferences", _find_ok)

        result = await pref_service.get_for_user("u-1")

        assert result == prefs
        assert isinstance(fake_redis.kv["pref:v1:u-1"], dict)
        assert fake_redis.kv["pref:v1:u-1"]["max_history"] == 40

    @pytest.mark.asyncio
    async def test_redis_hit_skips_db(self, fake_redis, monkeypatch):
        """Redis hit 시 repository를 건드리지 않는다."""
        prefs = _make_prefs(user_sub="u-2", session_ttl_seconds=600)
        await fake_redis.set("pref:v1:u-2", pref_service._serialize(prefs))

        called = {"count": 0}
        async def _find_should_not_run(_sub):
            called["count"] += 1
            return None
        monkeypatch.setattr(pref_service, "find_preferences", _find_should_not_run)

        result = await pref_service.get_for_user("u-2")

        assert result == prefs
        assert called["count"] == 0

    @pytest.mark.asyncio
    async def test_negative_marker_hit_returns_none(self, fake_redis, monkeypatch):
        """sentinel이 캐시돼 있으면 DB 조회 없이 즉시 None."""
        await fake_redis.set("pref:v1:u-3", pref_service._NEGATIVE_MARKER)

        async def _find_should_not_run(_sub):
            raise AssertionError("미등록 캐시가 있으면 DB를 안 건드려야 한다")
        monkeypatch.setattr(pref_service, "find_preferences", _find_should_not_run)

        result = await pref_service.get_for_user("u-3")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_user_sub_returns_none(self, fake_redis):
        """게스트/익명: user_sub이 비어 있으면 즉시 None."""
        assert await pref_service.get_for_user(None) is None
        assert await pref_service.get_for_user("") is None
        assert fake_redis.kv == {}  # 캐시 키도 만들지 않음

    @pytest.mark.asyncio
    async def test_invalidate_deletes_cache(self, fake_redis):
        await fake_redis.set("pref:v1:u-4", {"foo": "bar"})
        ok = await pref_service.invalidate("u-4")
        assert ok is True
        assert "pref:v1:u-4" not in fake_redis.kv


# ── 3. conversation.append_turn 자연 적용 ─────────────────────────────────────


class TestAppendTurnNaturalApply:
    """append_turn이 prefs를 자연 적용하는지 검증."""

    @pytest.mark.asyncio
    async def test_uses_user_prefs_when_present(self, fake_redis, monkeypatch):
        """prefs가 있으면 그 값으로 LTRIM/EXPIRE를 적용한다.

        append_turn은 raw Redis 키를 첫 인자로 받고 user_sub로 prefs를 조회한다.
        """
        prefs = _make_prefs(user_sub="u-1", max_history=4, session_ttl_seconds=600)
        async def _get_prefs(_sub):
            return prefs
        monkeypatch.setattr(conv_module, "get_user_preferences", _get_prefs)

        key = "conversation:c-1"
        for i in range(3):
            await conv_module.append_turn(key, f"q{i}", f"a{i}", user_sub="u-1")

        assert len(fake_redis.lists[key]) == 4
        assert fake_redis.expires[key] == 600

    @pytest.mark.asyncio
    async def test_falls_back_to_global_defaults(self, fake_redis, monkeypatch):
        """prefs가 없으면 전역 MAX_HISTORY=20, SESSION_TTL=86400 적용."""
        async def _no_prefs(_sub):
            return None
        monkeypatch.setattr(conv_module, "get_user_preferences", _no_prefs)

        key = "conversation:c-2"
        for i in range(11):
            await conv_module.append_turn(key, f"q{i}", f"a{i}", user_sub="u-2")

        assert len(fake_redis.lists[key]) == 20
        assert fake_redis.expires[key] == conv_module.SESSION_TTL


# ── 4. 라우터 ─────────────────────────────────────────────────────────────────


def _make_request(user_sub: Optional[str] = "u-1") -> MagicMock:
    """FastAPI Request mock — request.state.user["sub"]만 채운다."""
    req = MagicMock()
    req.state.user = {"sub": user_sub} if user_sub else None
    return req


class TestPreferencesRouter:
    """agent_arch.api.v1.preferences의 4개 엔드포인트 동작.

    repository 함수를 monkeypatch로 mock해서 DB 없이 검증한다.
    """

    @pytest.mark.asyncio
    async def test_get_returns_default_when_no_row(self, monkeypatch):
        """행이 없는 사용자 GET → is_default=true + 전역 기본값."""
        from agent_arch.api.v1 import preferences as router_module

        async def _find_none(_sub):
            return None
        monkeypatch.setattr(router_module, "find_preferences", _find_none)

        resp = await router_module.get_preferences(_make_request("u-new"))

        assert resp.user_sub == "u-new"
        assert resp.is_default is True
        assert resp.max_history == conv_module.MAX_HISTORY
        assert resp.session_ttl_seconds == conv_module.SESSION_TTL
        assert resp.updated_at is None

    @pytest.mark.asyncio
    async def test_get_returns_row_when_present(self, monkeypatch):
        """행이 있는 사용자 GET → is_default=false + 실제 값."""
        from agent_arch.api.v1 import preferences as router_module

        async def _find_ok(_sub):
            return _make_prefs(user_sub="u-1", max_history=40, session_ttl_seconds=43200)
        monkeypatch.setattr(router_module, "find_preferences", _find_ok)

        resp = await router_module.get_preferences(_make_request("u-1"))

        assert resp.is_default is False
        assert resp.max_history == 40
        assert resp.session_ttl_seconds == 43200
        assert resp.updated_at is not None

    @pytest.mark.asyncio
    async def test_put_invokes_upsert_and_invalidate(self, monkeypatch):
        """PUT은 upsert 호출 + 캐시 invalidate를 수행한다."""
        from agent_arch.api.v1 import preferences as router_module

        captured: Dict[str, Any] = {}

        async def _upsert(*, user_sub, max_history, session_ttl_seconds):
            captured["upsert"] = (user_sub, max_history, session_ttl_seconds)
            return _make_prefs(user_sub=user_sub, max_history=max_history or 20,
                               session_ttl_seconds=session_ttl_seconds or 86400)

        async def _invalidate(user_sub):
            captured["invalidate"] = user_sub
            return True

        monkeypatch.setattr(router_module, "upsert_preferences", _upsert)
        monkeypatch.setattr(router_module, "invalidate", _invalidate)

        body = UserPreferencesUpdate(max_history=30)
        resp = await router_module.update_preferences(_make_request("u-1"), body)

        # 누락된 ttl은 None 그대로 repository로 흘러간다 (COALESCE 책임은 SQL)
        assert captured["upsert"] == ("u-1", 30, None)
        assert captured["invalidate"] == "u-1"
        assert resp.is_default is False
        assert resp.max_history == 30

    @pytest.mark.asyncio
    async def test_delete_invokes_delete_and_invalidate(self, monkeypatch):
        from agent_arch.api.v1 import preferences as router_module

        calls: List[Tuple[str, str]] = []

        async def _delete(user_sub):
            calls.append(("delete", user_sub))
            return True

        async def _invalidate(user_sub):
            calls.append(("invalidate", user_sub))
            return True

        monkeypatch.setattr(router_module, "delete_preferences", _delete)
        monkeypatch.setattr(router_module, "invalidate", _invalidate)

        resp = await router_module.delete_preferences_route(_make_request("u-1"))

        assert resp.status_code == 204
        assert calls == [("delete", "u-1"), ("invalidate", "u-1")]

    @pytest.mark.asyncio
    async def test_history_clear_invokes_clear_user_conversations(self, monkeypatch):
        """본인 모든 대화방 키를 일괄 삭제하므로 clear_user_conversations를 호출."""
        from agent_arch.api.v1 import preferences as router_module

        captured: Dict[str, str] = {}

        async def _clear(user_sub):
            captured["clear"] = user_sub
            return 0
        monkeypatch.setattr(router_module, "clear_user_conversations", _clear)

        resp = await router_module.clear_user_history(_make_request("u-1"))

        assert resp.status_code == 204
        assert captured["clear"] == "u-1"

    @pytest.mark.asyncio
    async def test_missing_user_sub_raises_401(self):
        """request.state.user에 sub가 없으면 401."""
        from fastapi import HTTPException
        from agent_arch.api.v1 import preferences as router_module

        with pytest.raises(HTTPException) as exc:
            await router_module.get_preferences(_make_request(None))
        assert exc.value.status_code == 401
