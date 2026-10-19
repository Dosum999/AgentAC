import asyncio
import hashlib
import json
import uuid
from typing import Dict, List, Optional

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from agent_arch.core.config import settings
from agent_arch.llm.embeddings import embed_text
from agent_arch.utils.qdrant import upsert_in_batches
from agent_arch.services.api_registry.models import ExternalToolDef

# gemini-embedding-001 기본 출력 차원
EMBEDDING_DIM = 3072
# 임베딩 동시 호출 상한 (guide_kb 인덱서와 동일 — rate limit 보호용)
_EMBED_CONCURRENCY = 20


def _tool_id(tool: ExternalToolDef) -> str:
    """도구의 결정론적 포인트 UUID를 생성한다.

    dynamic_api는 기존 키(이름만)를 유지해 재인덱싱(301개 재임베딩)을
    피하고, registry는 origin 접두로 이름 충돌을 격리한다.

    Args:
        tool: 도구 정의

    Returns:
        포인트 UUID 문자열
    """
    key = tool.tool_name if tool.origin == "dynamic_api" else f"{tool.origin}:{tool.tool_name}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def _tool_hash(tool: ExternalToolDef) -> str:
    """도구 내용(description + parameters_schema)의 MD5 해시를 반환한다.

    서버 재시작 시 변경된 도구만 재인덱싱하기 위한 변경 감지에 사용한다.

    Args:
        tool: 해시를 계산할 도구 정의 객체

    Returns:
        도구 내용의 MD5 해시 문자열
    """
    content = json.dumps(tool.model_dump(), sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


class APIRegistryIndexer:
    """Qdrant 기반 API 도구 레지스트리.

    llm_tools.json의 도구들을 임베딩하여 Qdrant에 저장하고,
    사용자 쿼리와 의미적으로 유사한 도구를 검색한다.
    """

    COLLECTION = "api_registry"
    TOP_K_DEFAULT = 7

    def __init__(self):
        self._client: Optional[AsyncQdrantClient] = None

    async def connect(self) -> None:
        """Qdrant 클라이언트를 초기화하고 컬렉션을 준비한다.

        컬렉션이 없거나 벡터 차원이 다르면 재생성한다.

        Raises:
            RuntimeError: Qdrant 연결 실패 시
        """
        self._client = AsyncQdrantClient(url=settings.qdrant_url)

        existing = await self._client.get_collections()
        names = [c.name for c in existing.collections]

        if self.COLLECTION in names:
            # 차원 불일치 시 재생성 (임베딩 모델 변경 대응)
            info = await self._client.get_collection(self.COLLECTION)
            existing_dim = info.config.params.vectors.size
            if existing_dim != EMBEDDING_DIM:
                await self._client.delete_collection(self.COLLECTION)
                print(f"ℹ️  Qdrant 컬렉션 재생성 (차원 변경: {existing_dim} → {EMBEDDING_DIM})")
                names.remove(self.COLLECTION)

        if self.COLLECTION not in names:
            await self._client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            print(f"✅ Qdrant 컬렉션 '{self.COLLECTION}' 생성 완료 (dim={EMBEDDING_DIM})")

    async def index_tools(self, tools: List[ExternalToolDef]) -> Dict[str, int | bool]:
        """변경된 도구만 임베딩하여 Qdrant에 upsert한다.

        각 도구의 내용 해시를 Qdrant payload의 '_hash' 필드에 저장한다.
        서버 재시작 시 해시가 동일한 도구는 건너뛰어 불필요한 임베딩 API 호출을 방지한다.

        Args:
            tools: 인덱싱할 도구 정의 리스트

        Returns:
            인덱싱 결과 dict:
                - total_tools: 입력 도구 총 개수
                - changed: 실제로 임베딩 + upsert된 도구 수
                - orphans_deleted: 삭제된 고아 포인트 수
                - skipped: 변경 없음으로 임베딩을 전부 건너뛴 경우 True

        Raises:
            RuntimeError: connect() 호출 전일 경우
        """
        if not self._client:
            raise RuntimeError("indexer.connect()를 먼저 호출하세요")

        # 1. Qdrant에서 기존 포인트의 해시·origin 조회
        existing = await self._fetch_existing_meta()
        existing_hashes = {pid: meta["hash"] for pid, meta in existing.items()}

        # 2. 입력에서 사라진 고아 포인트 제거 — **입력 배치에 포함된 origin만**
        #    범위로 한다. 동적/레지스트리를 따로 인덱싱해도 서로를 고아로
        #    오인해 지우지 않도록 (origin 없는 구버전 포인트는 dynamic_api 취급).
        batch_origins = {t.origin for t in tools}
        current_ids = {_tool_id(t) for t in tools}
        orphan_ids = {
            pid for pid, meta in existing.items()
            if meta["origin"] in batch_origins and pid not in current_ids
        }
        if orphan_ids:
            await self._client.delete(
                collection_name=self.COLLECTION,
                points_selector=list(orphan_ids),
            )
            print(f"ℹ️  API Registry: {len(orphan_ids)}개 고아 포인트 삭제")

        # 3. 변경/신규 도구 필터링
        changed = [
            t for t in tools
            if existing_hashes.get(_tool_id(t)) != _tool_hash(t)
        ]

        if not changed:
            if not orphan_ids:
                print(f"API Registry: 변경된 도구 없음 - 재인덱싱 생략 (총 {len(tools)}개)")
            return {
                "total_tools": len(tools),
                "changed": 0,
                "orphans_deleted": len(orphan_ids),
                "skipped": len(orphan_ids) == 0,
            }

        print(f"ℹ️  API Registry: {len(changed)}/{len(tools)}개 도구 변경 감지 → 재인덱싱 중...")

        # 3. 변경된 도구만 세마포어로 동시성 제한하며 병렬 임베딩 후 upsert
        # (직렬 대비 재인덱싱 시간 대폭 단축. 임베딩은 도구 간 독립적이라 결과는 동일)
        sem = asyncio.Semaphore(_EMBED_CONCURRENCY)

        async def _embed_tool(tool: ExternalToolDef) -> PointStruct:
            text = f"{tool.tool_name}: {tool.description}"
            async with sem:
                vector = await self._embed(text, task_type="RETRIEVAL_DOCUMENT")
            payload = tool.model_dump()
            payload["_hash"] = _tool_hash(tool)  # 변경 감지용 해시 저장
            # dev의 병렬 임베딩 구조(return) + 본 브랜치의 _tool_id(tool) 시그니처
            # (origin별 ID 격리 — registry/dynamic 충돌 방지) 결합
            return PointStruct(
                id=_tool_id(tool),
                vector=vector,
                payload=payload,
            )

        # return_exceptions=True: 한 도구의 임베딩 실패가 전체 재인덱싱을 중단시키지 않도록 한다.
        # (실패 도구는 _hash가 저장되지 않으므로 다음 재인덱싱에서 자동 재시도된다)
        results = await asyncio.gather(
            *[_embed_tool(t) for t in changed], return_exceptions=True
        )
        points = [r for r in results if isinstance(r, PointStruct)]
        failed = len(results) - len(points)
        if failed:
            print(f"⚠️  API Registry: {failed}개 도구 임베딩 실패 — 다음 재인덱싱에서 재시도")

        await upsert_in_batches(self._client, self.COLLECTION, points)
        print(f"✅ API Registry: {len(points)}개 도구 인덱싱 완료")
        return {
            "total_tools": len(tools),
            "changed": len(points),
            "orphans_deleted": len(orphan_ids),
            "skipped": False,
        }

    async def _fetch_existing_meta(self) -> Dict[str, Dict[str, str]]:
        """기존 포인트 ID → {hash, origin} 매핑을 반환한다.

        origin 필드가 없는 구버전 포인트는 dynamic_api로 취급한다.
        """
        meta: Dict[str, Dict[str, str]] = {}
        offset = None

        while True:
            records, next_offset = await self._client.scroll(
                collection_name=self.COLLECTION,
                limit=200,
                offset=offset,
                with_payload=["_hash", "origin"],
                with_vectors=False,
            )
            for record in records:
                payload = record.payload or {}
                meta[str(record.id)] = {
                    "hash": payload.get("_hash", ""),
                    "origin": payload.get("origin") or "dynamic_api",
                }

            if next_offset is None:
                break
            offset = next_offset

        return meta

    async def search(
        self,
        query: str,
        top_k: int = TOP_K_DEFAULT,
        service_filter: Optional[str] = None,
        include_registry: bool = False,
        domain: Optional[str] = None,
    ) -> List[ExternalToolDef]:
        """쿼리와 의미적으로 유사한 도구를 반환한다.

        Args:
            query: 사용자 입력 텍스트
            top_k: 반환할 최대 도구 수
            service_filter: 특정 서비스로 필터링 (e.g. "hr_service")
            include_registry: origin="registry" 포인트(정적 스킬·MCP) 포함 여부.
                False면 기존 동작(동적 API만) — UNIFIED_TOOL_INDEXING off 상태에서
                잔존 registry 포인트가 섞이지 않도록 하는 안전장치.
            domain: 정규화된 도메인 키(hr/talk/mail 등)로 제한.
                tool_search의 도메인 스코프 탐색용.

        Returns:
            관련 도구 리스트

        Raises:
            RuntimeError: connect() 호출 전일 경우
        """
        if not self._client:
            raise RuntimeError("indexer.connect()를 먼저 호출하세요")

        vector = await self._embed(query, task_type="RETRIEVAL_QUERY")

        must = []
        must_not = []
        if service_filter:
            must.append(
                FieldCondition(
                    key="external_service",
                    match=MatchValue(value=service_filter),
                )
            )
        if domain:
            must.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        if not include_registry:
            # origin 없는 구버전 포인트는 dynamic_api이므로 영향 없음
            must_not.append(
                FieldCondition(key="origin", match=MatchValue(value="registry"))
            )
        query_filter = Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None

        results = await self._client.query_points(
            collection_name=self.COLLECTION,
            query=vector,
            limit=top_k,
            query_filter=query_filter,
        )

        return [ExternalToolDef(**hit.payload) for hit in results.points]

    async def _embed(self, text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
        """텍스트를 임베딩한다 (프로바이더는 settings.embedding_model 로 결정).

        Args:
            text: 임베딩할 텍스트
            task_type: RETRIEVAL_DOCUMENT(인덱싱) / RETRIEVAL_QUERY(검색).
                Gemini 에서 사용되고 OpenAI 에서는 무시된다.

        Returns:
            텍스트의 임베딩 벡터 리스트 (길이 3072)
        """
        return await embed_text(text, task_type=task_type, dimensions=EMBEDDING_DIM)


api_indexer = APIRegistryIndexer()
