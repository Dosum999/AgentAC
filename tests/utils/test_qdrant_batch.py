"""agent_arch.utils.qdrant.upsert_in_batches 단위 테스트.

대량 포인트를 단일 upsert로 보내면 발생하던 httpx.ReadError 회귀를 막기 위해,
포인트가 batch_size 단위로 나뉘어 여러 번 upsert 되는지 검증한다.
"""
import pytest
from unittest.mock import AsyncMock

from agent_arch.utils.qdrant import upsert_in_batches, UPSERT_BATCH_SIZE


def _points(n: int):
    """간단한 더미 포인트 리스트 (PointStruct 대신 정수로 충분 — 분할 로직만 검증)."""
    return list(range(n))


class TestUpsertInBatches:
    @pytest.mark.asyncio
    async def test_splits_into_expected_batches(self):
        """250개 + batch 100 → 100/100/50 으로 3회 호출된다."""
        client = AsyncMock()
        pts = _points(250)
        await upsert_in_batches(client, "col", pts, batch_size=100)

        sizes = [len(c.kwargs["points"]) for c in client.upsert.await_args_list]
        assert sizes == [100, 100, 50]
        assert all(c.kwargs["collection_name"] == "col" for c in client.upsert.await_args_list)

    @pytest.mark.asyncio
    async def test_single_batch_when_under_limit(self):
        """포인트 수가 batch_size 이하이면 1회만 호출된다."""
        client = AsyncMock()
        await upsert_in_batches(client, "col", _points(30), batch_size=100)
        assert client.upsert.await_count == 1
        assert len(client.upsert.await_args_list[0].kwargs["points"]) == 30

    @pytest.mark.asyncio
    async def test_empty_points_no_call(self):
        """빈 리스트면 upsert를 호출하지 않는다."""
        client = AsyncMock()
        await upsert_in_batches(client, "col", [], batch_size=100)
        assert client.upsert.await_count == 0

    @pytest.mark.asyncio
    async def test_preserves_order_and_coverage(self):
        """모든 포인트가 순서대로 정확히 한 번씩 전달된다."""
        client = AsyncMock()
        pts = _points(105)
        await upsert_in_batches(client, "col", pts, batch_size=50)

        sent = []
        for c in client.upsert.await_args_list:
            sent.extend(c.kwargs["points"])
        assert sent == pts

    @pytest.mark.asyncio
    async def test_default_batch_size_used(self):
        """batch_size 미지정 시 UPSERT_BATCH_SIZE 가 적용된다."""
        client = AsyncMock()
        await upsert_in_batches(client, "col", _points(UPSERT_BATCH_SIZE + 1))
        assert client.upsert.await_count == 2
        assert len(client.upsert.await_args_list[0].kwargs["points"]) == UPSERT_BATCH_SIZE
        assert len(client.upsert.await_args_list[1].kwargs["points"]) == 1
