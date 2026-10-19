"""사용자 프로필 캐시 서비스 단위 테스트.

대상:
    - app.services.user_profile.service.compute_profile_hash 결정성
    - upsert_profile: 신규 / 동일 hash / 변경 케이스의 updated 플래그와 DB write 분기
    - get_profile: Redis hit, Redis miss + DB hit (backfill), miss + DB miss
    - invalidate: Redis 키 삭제

Repository(asyncpg)와 redis_store는 mock으로 대체해 실제 외부 의존을 일으키지 않는다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from agent_arch.services.user_profile import (
    UserProfileRow,
    compute_profile_hash,
    service as profile_service,
)


# ── 공용 fake 인프라 ──────────────────────────────────────────────────────────


class _FakeRedis:
    """in-memory dict 기반 redis_store 대체."""

    def __init__(self) -> None:
        self.kv: Dict[str, Any] = {}
        self.expires: Dict[str, int] = {}

    async def get(self, key: str) -> Any:
        return self.kv.get(key)

    async def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        # 실제 redis_store는 json.dumps 후 저장 — 동일하게 흉내내 직렬화 가능성도 검증
        self.kv[key] = json.loads(json.dumps(value, ensure_ascii=False))
        if expire is not None:
            self.expires[key] = expire
        return True

    async def delete(self, key: str) -> bool:
        existed = key in self.kv
        self.kv.pop(key, None)
        self.expires.pop(key, None)
        return existed


@pytest.fixture
def fake_redis(monkeypatch):
    """service.redis_store를 in-memory fake로 교체."""
    fake = _FakeRedis()
    monkeypatch.setattr(profile_service, "redis_store", fake)
    return fake


def _make_profile(**overrides: Any) -> dict:
    """Frappe get_current_user 응답을 닮은 dict."""
    base = {
        "user": "user@example.com",
        "full_name": "사용자명",
        "is_guest": False,
        "company": "ACME",
        "companies": ["ACME", "ACME_HK"],
        "role_type": "EMPLOYEE",
        "service_access": [],
        "assigned_roles": ["agent_user"],
        "permissions": {"hr_service": ["read"]},
    }
    base.update(overrides)
    return base


def _make_row(profile: dict, profile_hash: str) -> UserProfileRow:
    now = datetime(2026, 5, 13, 10, tzinfo=timezone.utc)
    return UserProfileRow(
        user_sub="u-1",
        email=profile.get("user"),
        is_guest=bool(profile.get("is_guest")),
        profile=profile,
        profile_hash=profile_hash,
        created_at=now,
        updated_at=now,
    )


# ── 1. compute_profile_hash 결정성 ────────────────────────────────────────────


class TestComputeProfileHash:
    """sha256 canonical hash가 결정적이며 의미 있는 변화에만 반응함을 검증."""

    def test_same_input_same_hash(self):
        profile = _make_profile()
        assert compute_profile_hash(profile) == compute_profile_hash(profile)

    def test_key_order_irrelevant(self):
        """키 순서만 다른 두 dict는 동일 hash."""
        a = {"user": "u@e.com", "company": "ACME", "is_guest": False}
        b = {"is_guest": False, "company": "ACME", "user": "u@e.com"}
        assert compute_profile_hash(a) == compute_profile_hash(b)

    def test_value_change_changes_hash(self):
        original = _make_profile()
        changed = _make_profile(company="ACME_HK")
        assert compute_profile_hash(original) != compute_profile_hash(changed)

    def test_unicode_preserved(self):
        """한글이 포함된 값도 정상적으로 해시된다."""
        profile = _make_profile(full_name="홍길동", company="대티씨오")
        hash_value = compute_profile_hash(profile)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # sha256 hex digest


# ── 2. upsert_profile 분기 동작 ──────────────────────────────────────────────


class TestUpsertProfile:
    """변경 감지로 DB write를 절약하는지, Redis는 항상 갱신되는지 검증."""

    @pytest.mark.asyncio
    async def test_new_user_writes_db(self, fake_redis, monkeypatch):
        """신규 sub: 기존 hash 없음 → INSERT 호출 + Redis SET + updated=True."""
        profile = _make_profile()
        new_hash = compute_profile_hash(profile)
        write_called: Dict[str, Any] = {}

        async def _find_hash(_sub):
            return None

        async def _upsert(**kwargs):
            write_called.update(kwargs)
            return _make_row(profile, new_hash)

        monkeypatch.setattr(profile_service, "find_profile_hash", _find_hash)
        monkeypatch.setattr(profile_service, "upsert_profile_row", _upsert)

        updated = await profile_service.upsert_profile("u-1", profile)

        assert updated is True
        assert write_called["user_sub"] == "u-1"
        assert write_called["email"] == "user@example.com"
        assert write_called["is_guest"] is False
        assert write_called["profile_hash"] == new_hash
        # Redis는 변경 여부와 무관하게 갱신
        assert fake_redis.kv["user_profile:u-1"]["user"] == "user@example.com"
        assert fake_redis.expires["user_profile:u-1"] == profile_service.REDIS_PROFILE_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_rest_v1_profile_writes_normalized_email_and_permissions(
        self, fake_redis, monkeypatch
    ):
        """REST v1 MyContext는 DB/Redis에 Agent 평면 프로필로 저장된다."""
        profile = {
            "user": {
                "id": "user@example.com",
                "email": "user@example.com",
                "full_name": "박정훈",
                "image": "",
            },
            "selected_company_id": "ACME",
            "employee": None,
            "companies": [],
            "authorization": {
                "company_id": "ACME",
                "role_type": "EMPLOYEE",
                "permissions": {"eapproval": ["eapproval.access"]},
                "service_access": [],
                "role_assignments": [],
            },
        }
        captured: Dict[str, Any] = {}

        async def _find_hash(_sub):
            return None

        async def _upsert(**kwargs):
            captured.update(kwargs)
            return None

        monkeypatch.setattr(profile_service, "find_profile_hash", _find_hash)
        monkeypatch.setattr(profile_service, "upsert_profile_row", _upsert)

        updated = await profile_service.upsert_profile("u-rest", profile)

        assert updated is True
        assert captured["email"] == "user@example.com"
        assert captured["profile"]["user"] == "user@example.com"
        assert captured["profile"]["full_name"] == "박정훈"
        assert captured["profile"]["permissions"] == {
            "eapproval": ["eapproval.access"]
        }
        assert fake_redis.kv["user_profile:u-rest"]["user"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_same_hash_skips_db_write(self, fake_redis, monkeypatch):
        """동일 profile: hash가 같으면 DB write 생략, Redis만 갱신, updated=False."""
        profile = _make_profile()
        existing_hash = compute_profile_hash(profile)

        async def _find_hash(_sub):
            return existing_hash

        def _upsert_should_not_run(**kwargs):
            raise AssertionError("동일 hash인데 DB write가 호출됨")

        monkeypatch.setattr(profile_service, "find_profile_hash", _find_hash)
        monkeypatch.setattr(profile_service, "upsert_profile_row", _upsert_should_not_run)

        updated = await profile_service.upsert_profile("u-1", profile)

        assert updated is False
        # Redis는 여전히 갱신됨 (TTL 슬라이딩)
        assert fake_redis.kv["user_profile:u-1"]["user"] == "user@example.com"

    @pytest.mark.asyncio
    async def test_changed_profile_triggers_db_write(self, fake_redis, monkeypatch):
        """profile 내용이 바뀌면 hash가 달라져 DB write 발생, updated=True."""
        old_profile = _make_profile(company="ACME")
        new_profile = _make_profile(company="ACME_HK")
        old_hash = compute_profile_hash(old_profile)
        new_hash = compute_profile_hash(new_profile)
        assert old_hash != new_hash

        async def _find_hash(_sub):
            return old_hash

        captured: Dict[str, Any] = {}

        async def _upsert(**kwargs):
            captured.update(kwargs)
            return _make_row(new_profile, new_hash)

        monkeypatch.setattr(profile_service, "find_profile_hash", _find_hash)
        monkeypatch.setattr(profile_service, "upsert_profile_row", _upsert)

        updated = await profile_service.upsert_profile("u-1", new_profile)

        assert updated is True
        assert captured["profile_hash"] == new_hash
        assert captured["profile"]["company"] == "ACME_HK"
        # Redis는 신규 profile로 갱신
        assert fake_redis.kv["user_profile:u-1"]["company"] == "ACME_HK"

    @pytest.mark.asyncio
    async def test_is_guest_propagates(self, fake_redis, monkeypatch):
        """profile.is_guest=true가 is_guest 컬럼/Redis 양쪽에 그대로 반영."""
        profile = _make_profile(is_guest=True)
        captured: Dict[str, Any] = {}

        async def _find_hash(_sub):
            return None

        async def _upsert(**kwargs):
            captured.update(kwargs)
            return _make_row(profile, compute_profile_hash(profile))

        monkeypatch.setattr(profile_service, "find_profile_hash", _find_hash)
        monkeypatch.setattr(profile_service, "upsert_profile_row", _upsert)

        await profile_service.upsert_profile("u-guest", profile)

        assert captured["is_guest"] is True
        assert fake_redis.kv["user_profile:u-guest"]["is_guest"] is True


# ── 3. get_profile (Redis 우선 + DB fallback + backfill) ─────────────────────


class TestGetProfile:
    """Redis hit / miss → DB / miss → None 경로."""

    @pytest.mark.asyncio
    async def test_redis_hit_skips_db(self, fake_redis, monkeypatch):
        """Redis에 캐시된 profile이 있으면 DB를 건드리지 않는다."""
        profile = _make_profile()
        await fake_redis.set("user_profile:u-1", profile)

        async def _find_should_not_run(_sub):
            raise AssertionError("Redis hit인데 DB가 호출됨")

        monkeypatch.setattr(profile_service, "find_profile_row", _find_should_not_run)

        result = await profile_service.get_profile("u-1")
        assert result == profile

    @pytest.mark.asyncio
    async def test_redis_miss_db_hit_backfills(self, fake_redis, monkeypatch):
        """Redis miss → DB에서 가져와 Redis에 backfill."""
        profile = _make_profile()
        row = _make_row(profile, compute_profile_hash(profile))

        async def _find_row(_sub):
            return row

        monkeypatch.setattr(profile_service, "find_profile_row", _find_row)

        result = await profile_service.get_profile("u-1")

        assert result == profile
        assert fake_redis.kv["user_profile:u-1"] == profile
        assert fake_redis.expires["user_profile:u-1"] == profile_service.REDIS_PROFILE_TTL_SECONDS

    @pytest.mark.asyncio
    async def test_redis_miss_db_miss_returns_none(self, fake_redis, monkeypatch):
        """완전히 미등록: 모두 miss → None, Redis backfill 없음."""
        async def _find_none(_sub):
            return None

        monkeypatch.setattr(profile_service, "find_profile_row", _find_none)

        result = await profile_service.get_profile("u-missing")
        assert result is None
        assert "user_profile:u-missing" not in fake_redis.kv

    @pytest.mark.asyncio
    async def test_empty_user_sub_returns_none(self, fake_redis):
        """None/빈 문자열: 즉시 None, 캐시 키도 만들지 않음."""
        assert await profile_service.get_profile(None) is None
        assert await profile_service.get_profile("") is None
        assert fake_redis.kv == {}


# ── 4. invalidate ────────────────────────────────────────────────────────────


class TestInvalidate:
    @pytest.mark.asyncio
    async def test_deletes_redis_key(self, fake_redis):
        await fake_redis.set("user_profile:u-1", _make_profile())
        ok = await profile_service.invalidate("u-1")
        assert ok is True
        assert "user_profile:u-1" not in fake_redis.kv

    @pytest.mark.asyncio
    async def test_empty_user_sub_returns_false(self, fake_redis):
        assert await profile_service.invalidate("") is False
        assert await profile_service.invalidate(None) is False  # type: ignore[arg-type]
