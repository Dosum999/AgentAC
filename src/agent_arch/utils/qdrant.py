"""Qdrant 공통 유틸.

대량 포인트를 한 번의 ``upsert`` 요청으로 보내면 요청 본문이 수십 MB에 달해
Qdrant가 응답 전에 커넥션을 끊어 ``httpx.ReadError`` → ``ResponseHandlingException``
이 발생한다(특히 3072차원 벡터 수백~수천 개). 이를 막기 위해 포인트를 일정 크기로
나눠 순차 upsert 한다.
"""
from typing import List

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import PointStruct

# 한 번의 upsert 요청에 담을 최대 포인트 수.
# 3072차원 float 기준 100개 ≈ 수 MB 수준으로, Qdrant REST 본문 한도/처리 시간에 안전.
UPSERT_BATCH_SIZE = 100


async def upsert_in_batches(
    client: AsyncQdrantClient,
    collection_name: str,
    points: List[PointStruct],
    *,
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """포인트 리스트를 batch_size 단위로 나눠 순차 upsert 한다.

    Args:
        client: Qdrant 비동기 클라이언트.
        collection_name: 대상 컬렉션 이름.
        points: upsert 할 PointStruct 리스트 (빈 리스트면 아무 작업도 하지 않음).
        batch_size: 배치당 최대 포인트 수.
    """
    for i in range(0, len(points), batch_size):
        await client.upsert(
            collection_name=collection_name,
            points=points[i:i + batch_size],
        )
