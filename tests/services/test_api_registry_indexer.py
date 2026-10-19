"""APIRegistryIndexer.index_tools 병렬 임베딩 동작 테스트.

Qdrant 실 연결·실 임베딩 없이, 변경 감지/병렬성/실패 내성/payload 무결성만 검증한다.
임베딩(self._embed)·upsert(upsert_in_batches)·기존 해시 조회는 Mock으로 대체한다.
"""
import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.services.api_registry.indexer import (
    APIRegistryIndexer,
    _tool_hash,
    _tool_id,
)
from agent_arch.services.api_registry.models import ExternalToolDef


def _mk_tool(name: str) -> ExternalToolDef:
    """테스트용 ExternalToolDef 팩토리."""
    return ExternalToolDef(
        tool_name=name,
        description=f"{name} 설명",
        is_external=True,
        external_service="hr_service",
        http_method="GET",
        endpoint_path=f"/{name}/",
        parameters_schema={"type": "object", "properties": {}},
        enabled=True,
    )


def _make_indexer(existing_hashes: dict) -> APIRegistryIndexer:
    """_client·_fetch_existing_meta를 Mock으로 채운 인덱서를 만든다.

    existing_hashes({id: hash})를 _fetch_existing_meta 형태({id: {hash, origin}})로
    변환해 주입한다 (origin은 기존 dynamic_api 포인트로 간주).
    """
    indexer = APIRegistryIndexer()
    indexer._client = MagicMock()
    indexer._client.delete = AsyncMock()
    meta = {pid: {"hash": h, "origin": "dynamic_api"} for pid, h in existing_hashes.items()}
    indexer._fetch_existing_meta = AsyncMock(return_value=meta)
    return indexer


class TestIndexToolsParallel:
    """index_tools 병렬 임베딩."""

    @pytest.mark.asyncio
    async def test_embeds_all_changed_and_builds_points(self):
        """전부 신규면 모든 도구를 임베딩하고 _hash가 payload에 들어간다."""
        tools = [_mk_tool(f"t{i}") for i in range(5)]
        indexer = _make_indexer({})
        indexer._embed = AsyncMock(return_value=[0.1] * 3072)

        with patch(
            "agent_arch.services.api_registry.indexer.upsert_in_batches", AsyncMock()
        ) as mock_upsert:
            result = await indexer.index_tools(tools)

        assert result["changed"] == 5
        assert result["skipped"] is False
        assert indexer._embed.await_count == 5

        points = mock_upsert.call_args.args[2]
        assert len(points) == 5
        for tool, point in zip(tools, points):
            assert point.id == _tool_id(tool)
            assert point.payload["_hash"] == _tool_hash(tool)

    @pytest.mark.asyncio
    async def test_skips_unchanged_tools(self):
        """해시가 동일한 도구는 임베딩하지 않는다(변경 감지 유지)."""
        tools = [_mk_tool("a"), _mk_tool("b")]
        existing = {_tool_id(_mk_tool("a")): _tool_hash(tools[0])}
        indexer = _make_indexer(existing)
        indexer._embed = AsyncMock(return_value=[0.2] * 3072)

        with patch(
            "agent_arch.services.api_registry.indexer.upsert_in_batches", AsyncMock()
        ) as mock_upsert:
            result = await indexer.index_tools(tools)

        assert result["changed"] == 1
        assert indexer._embed.await_count == 1
        points = mock_upsert.call_args.args[2]
        assert len(points) == 1
        assert points[0].id == _tool_id(_mk_tool("b"))

    @pytest.mark.asyncio
    async def test_runs_embeddings_concurrently(self):
        """여러 도구 임베딩이 직렬이 아니라 병렬로 실행된다(동시 in-flight > 1)."""
        tools = [_mk_tool(f"t{i}") for i in range(6)]
        indexer = _make_indexer({})

        state = {"current": 0, "max": 0}

        async def fake_embed(text, task_type="RETRIEVAL_DOCUMENT"):
            state["current"] += 1
            state["max"] = max(state["max"], state["current"])
            await asyncio.sleep(0.01)
            state["current"] -= 1
            return [0.1] * 3072

        indexer._embed = fake_embed

        with patch("agent_arch.services.api_registry.indexer.upsert_in_batches", AsyncMock()):
            await indexer.index_tools(tools)

        assert state["max"] > 1

    @pytest.mark.asyncio
    async def test_one_failure_does_not_abort_batch(self):
        """한 도구 임베딩이 실패해도 나머지는 인덱싱된다(부분 성공)."""
        tools = [_mk_tool(f"t{i}") for i in range(4)]
        indexer = _make_indexer({})

        async def flaky_embed(text, task_type="RETRIEVAL_DOCUMENT"):
            if text.startswith("t2:"):  # "t2: t2 설명"
                raise RuntimeError("embedding 429 exhausted")
            return [0.1] * 3072

        indexer._embed = flaky_embed

        with patch(
            "agent_arch.services.api_registry.indexer.upsert_in_batches", AsyncMock()
        ) as mock_upsert:
            result = await indexer.index_tools(tools)

        assert result["changed"] == 3
        points = mock_upsert.call_args.args[2]
        assert len(points) == 3
        assert all(p.id != _tool_id(_mk_tool("t2")) for p in points)
