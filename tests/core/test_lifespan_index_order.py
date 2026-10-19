"""부팅 인덱싱 순서 — 레지스트리 최종 패스는 MCP 등록 완료 후 (2026-07-09 설계).

핵심 불변식: 증분 인덱싱(부분 목록)은 고아 삭제로 남의 포인트를 지울 수 있으므로,
전체 스냅샷 패스가 **마지막**에 돌아 자가치유해야 한다. 여기서는 그 순서 보장
(gather 후 index_tools)과 게이트(unified off 시 스킵), MCP 미설정 시에도 정적
인덱싱이 도는 것을 검증한다.
"""
import asyncio

import pytest

from agent_arch.core import lifespan as ls


@pytest.mark.asyncio
async def test_registry_final_pass_runs_after_mcp_tasks(monkeypatch):
    """MCP 태스크가 전부 끝난 뒤에야 defs_from_registry 스냅샷·index_tools 가 호출된다."""
    order: list[str] = []

    async def slow_mcp():
        await asyncio.sleep(0.05)
        order.append("mcp_done")

    async def fake_index_tools(defs):
        order.append("registry_indexed")
        return {"total_tools": len(defs), "changed": 0, "orphans_deleted": 0, "skipped": True}

    monkeypatch.setattr(ls.settings, "unified_tool_indexing", True)
    monkeypatch.setattr(ls.api_indexer, "index_tools", fake_index_tools)
    monkeypatch.setattr(
        "agent_arch.services.api_registry.registry_loader.defs_from_registry",
        lambda only_names=None: (order.append("snapshot"), [])[1],
    )

    tasks = [asyncio.create_task(slow_mcp()), asyncio.create_task(slow_mcp())]
    await ls._index_registry_after_mcp(tasks)

    assert order == ["mcp_done", "mcp_done", "snapshot", "registry_indexed"]  # gather → 스냅샷 → 인덱싱


@pytest.mark.asyncio
async def test_registry_final_pass_runs_even_without_mcp(monkeypatch):
    """MCP 서버 미설정(빈 태스크)이어도 정적 스킬 인덱싱은 수행된다."""
    called = {"n": 0}

    async def fake_index_tools(defs):
        called["n"] += 1
        return {}

    monkeypatch.setattr(ls.settings, "unified_tool_indexing", True)
    monkeypatch.setattr(ls.api_indexer, "index_tools", fake_index_tools)
    monkeypatch.setattr(
        "agent_arch.services.api_registry.registry_loader.defs_from_registry",
        lambda only_names=None: [],
    )
    await ls._index_registry_after_mcp([])
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_registry_final_pass_skipped_when_unified_off(monkeypatch):
    """UNIFIED_TOOL_INDEXING=false 면 최종 패스를 돌지 않는다 (기존 게이트 유지)."""
    called = {"n": 0}

    async def fake_index_tools(defs):
        called["n"] += 1
        return {}

    monkeypatch.setattr(ls.settings, "unified_tool_indexing", False)
    monkeypatch.setattr(ls.api_indexer, "index_tools", fake_index_tools)
    await ls._index_registry_after_mcp([])
    assert called["n"] == 0
