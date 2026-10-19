"""메모리 v2 Qdrant 인덱서 — L1 atom 의 벡터 half.

컬렉션 ``agent_memory_v2``(v1 ``agent_memory`` 와 분리, 플래그 병행 기간 충돌 방지).
point id = atom id(1:1). update/merge 로 atom 이 superseded 되면 **쓰기 경로에서 즉시**
포인트를 지운다 — v1 이 soft-delete 후 포인트를 남겨 stale 회상이 났던 결함을 구조로 닫는다.

소유자 격리: search 는 Qdrant 필터에 user_sub 매칭을 강제한다.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from agent_arch.core.config import settings
from agent_arch.db.repository.memory_v2 import list_unindexed_atoms, mark_atoms_indexed
from agent_arch.llm.embeddings import embed_text
from agent_arch.utils.qdrant import upsert_in_batches

logger = logging.getLogger(__name__)

COLLECTION = "agent_memory_v2"
EMBEDDING_DIM = 3072
_EMBED_CONCURRENCY = 8
class MemoryV2Indexer:
    def __init__(self) -> None:
        self.COLLECTION = COLLECTION
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self) -> None:
        self._client = AsyncQdrantClient(url=settings.qdrant_url)
        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]
        if self.COLLECTION in names:
            info = await self._client.get_collection(self.COLLECTION)
            if info.config.params.vectors.size != EMBEDDING_DIM:
                await self._client.delete_collection(self.COLLECTION)
                logger.warning("memory_v2 컬렉션 재생성(차원 변경) → %d", EMBEDDING_DIM)
                names.remove(self.COLLECTION)
        if self.COLLECTION not in names:
            await self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info("Qdrant 컬렉션 '%s' 생성(dim=%d)", self.COLLECTION, EMBEDDING_DIM)

    def _require(self) -> AsyncQdrantClient:
        if not self._client:
            raise RuntimeError("indexer.connect()를 먼저 호출하세요")
        return self._client

    @staticmethod
    def _payload(atom: dict) -> dict:
        return {
            "user_sub": atom["user_sub"],
            "atom_type": atom["atom_type"],
            "scene_name": atom.get("scene_name"),
            "company_scope": atom.get("company_scope"),
            "priority": atom.get("priority"),
            "content": atom["content"],
            "status": "active",
        }

    async def index_atoms(self, atoms: list[dict]) -> int:
        """atom content 를 임베딩해 upsert 하고 PG 플래그를 세운다. 성공 건수 반환."""
        client = self._require()
        if not atoms:
            return 0
        sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

        async def _one(atom: dict) -> Optional[PointStruct]:
            async with sem:
                vec = await embed_text(atom["content"], task_type="RETRIEVAL_DOCUMENT", dimensions=EMBEDDING_DIM)
            return PointStruct(id=atom["id"], vector=vec, payload=self._payload(atom))

        results = await asyncio.gather(*[_one(a) for a in atoms], return_exceptions=True)
        points = [p for p in results if isinstance(p, PointStruct)]
        if len(points) < len(results):
            logger.warning("memory_v2 인덱싱 %d건 임베딩 실패 — 다음 sync 에서 재시도", len(results) - len(points))
        if not points:
            return 0
        await upsert_in_batches(client, self.COLLECTION, points)
        await mark_atoms_indexed([str(p.id) for p in points])
        return len(points)

    async def sync_pending(self, limit: int = 200) -> int:
        pending = await list_unindexed_atoms(limit=limit)
        return await self.index_atoms(pending) if pending else 0

    async def search(
        self,
        user_sub: str,
        query: str,
        *,
        top_k: int = 5,
        atom_type: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """본인 atom 중 의미 유사 상위 top_k. [{"id","score", ...payload}]."""
        client = self._require()
        must: list[Any] = [
            FieldCondition(key="user_sub", match=MatchValue(value=user_sub)),
            FieldCondition(key="status", match=MatchValue(value="active")),
        ]
        if atom_type:
            must.append(FieldCondition(key="atom_type", match=MatchValue(value=atom_type)))
        vector = await embed_text(query, task_type="RETRIEVAL_QUERY", dimensions=EMBEDDING_DIM)
        result = await client.query_points(
            collection_name=self.COLLECTION,
            query=vector,
            query_filter=Filter(must=must),
            limit=top_k,
            score_threshold=score_threshold or None,
        )
        return [{"id": str(h.id), "score": h.score, **(h.payload or {})} for h in result.points]

    async def delete_points(self, ids: list[str]) -> None:
        if not ids:
            return
        await self._require().delete(
            collection_name=self.COLLECTION, points_selector=PointIdsList(points=list(ids)),
        )

    async def delete_user_points(self, user_sub: str) -> None:
        await self._require().delete(
            collection_name=self.COLLECTION,
            points_selector=Filter(must=[FieldCondition(key="user_sub", match=MatchValue(value=user_sub))]),
        )


_indexer: Optional[MemoryV2Indexer] = None
_indexer_lock = asyncio.Lock()


async def get_v2_indexer() -> MemoryV2Indexer:
    """연결된 싱글톤(최초 1회 connect). 연결 실패는 예외로 올린다 — 호출부가 fail-open 처리."""
    global _indexer
    if _indexer is not None:
        return _indexer
    async with _indexer_lock:
        if _indexer is None:
            idx = MemoryV2Indexer()
            await idx.connect()
            _indexer = idx
    return _indexer
