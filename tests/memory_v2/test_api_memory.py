"""/me/memory 라우터 — 인증 sub 강제, v2 게이트(409), 조회·삭제·초기화 위임."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from agent_arch.api.v1 import memory as api
from agent_arch.core.config import settings


def _req(sub="u1"):
    return SimpleNamespace(state=SimpleNamespace(user={"sub": sub} if sub else {}))


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "enable_user_memory", True, raising=False)
    monkeypatch.setattr(settings, "memory_v2_mode", "shadow", raising=False)


@pytest.mark.asyncio
async def test_requires_user_and_flag(monkeypatch):
    with pytest.raises(HTTPException) as e:
        await api.get_my_memory(_req(sub=None), limit=10)
    assert e.value.status_code == 401
    monkeypatch.setattr(settings, "enable_user_memory", True, raising=False)
    monkeypatch.setattr(settings, "memory_v2_mode", "off", raising=False)
    with pytest.raises(HTTPException) as e:
        await api.get_my_memory(_req(), limit=10)
    assert e.value.status_code == 409


@pytest.mark.asyncio
async def test_get_delegates_export(enabled, monkeypatch):
    async def export(sub, *, limit):
        return {"atoms": [], "sub": sub, "limit": limit}

    monkeypatch.setattr(api.memory_admin, "export_user_memory", export)
    assert await api.get_my_memory(_req(), limit=7) == {"atoms": [], "sub": "u1", "limit": 7}


@pytest.mark.asyncio
async def test_forget_atom_paths(enabled, monkeypatch):
    async def forget(sub, atom_id):
        return {"id": atom_id} if atom_id.startswith("aaaaaaaa") else None

    monkeypatch.setattr(api.memory_explicit, "forget_atom", forget)
    ok = await api.forget_my_atom(_req(), "aaaaaaaa-0000-0000-0000-000000000001")
    assert ok.status_code == 204
    with pytest.raises(HTTPException) as e:
        await api.forget_my_atom(_req(), "bbbbbbbb-0000-0000-0000-000000000001")
    assert e.value.status_code == 404
    with pytest.raises(HTTPException) as e:
        await api.forget_my_atom(_req(), "not-a-uuid")
    assert e.value.status_code == 400


@pytest.mark.asyncio
async def test_reset_delegates(enabled, monkeypatch):
    async def reset(sub):
        return {"atoms": 3, "scenes": 1, "persona": True, "qdrant": True, "sub": sub}

    monkeypatch.setattr(api.memory_admin, "reset_user_memory", reset)
    assert (await api.reset_my_memory(_req()))["atoms"] == 3
