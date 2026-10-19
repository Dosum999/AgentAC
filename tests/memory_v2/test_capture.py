"""캡처 훅 — 플래그 게이트, 즉시 트리거 마커, fire-and-forget 스케줄."""
from __future__ import annotations

import asyncio

import pytest

from agent_arch.services.memory_v2 import capture


@pytest.fixture
def flags(monkeypatch):
    def _set(master: bool, mode: str):
        monkeypatch.setattr(capture.settings, "enable_user_memory", master, raising=False)
        monkeypatch.setattr(capture.settings, "memory_v2_mode", mode, raising=False)
    return _set


def test_gates(flags):
    flags(True, "off")
    assert not capture.enabled() and not capture.is_active()
    flags(True, "shadow")
    assert capture.enabled() and not capture.is_active()
    flags(True, "ACTIVE ")
    assert capture.enabled() and capture.is_active()
    flags(False, "active")                       # 마스터 스위치 우선
    assert not capture.enabled() and not capture.is_active()


def test_is_immediate_markers():
    assert capture.is_immediate("앞으로 표 쓰지 마")
    assert capture.is_immediate("이거 기억해줘")
    assert not capture.is_immediate("안 읽은 메일 있어?")
    assert not capture.is_immediate("")
    assert not capture.is_immediate(None)


@pytest.mark.asyncio
async def test_on_turn_saved_disabled_returns_false(flags):
    flags(True, "off")
    assert capture.on_turn_saved("u1", "m1", "hi") is False


@pytest.mark.asyncio
async def test_on_turn_saved_schedules_bump(flags, monkeypatch):
    flags(True, "shadow")
    calls = []

    async def fake_bump_pending(user_sub, message_id, *, immediate, idle_seconds, default_threshold):
        calls.append((user_sub, message_id, immediate, idle_seconds, default_threshold))
        return {"pending_count": 1}

    import agent_arch.db.repository.memory_v2 as repo
    monkeypatch.setattr(repo, "bump_pending", fake_bump_pending)
    assert capture.on_turn_saved("u1", "m1", "앞으로 표 쓰지 마") is True
    await asyncio.sleep(0)
    assert calls == [("u1", "m1", True, capture.IDLE_SECONDS, capture.DEFAULT_THRESHOLD)]


@pytest.mark.asyncio
async def test_on_turn_saved_requires_ids(flags):
    flags(True, "shadow")
    assert capture.on_turn_saved(None, "m1", "x") is False
    assert capture.on_turn_saved("u1", None, "x") is False


@pytest.mark.asyncio
async def test_bump_swallows_repository_errors(monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("db down")

    import agent_arch.db.repository.memory_v2 as repo
    monkeypatch.setattr(repo, "bump_pending", boom)
    assert await capture.bump("u1", "m1", "x") is None
