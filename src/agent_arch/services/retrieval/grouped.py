"""Qdrant ``query_points_groups`` 기반 도메인 독립 그룹 검색 유틸.

지정한 payload 필드(예: ``"category"``, ``"external_service"``)로 검색 결과를
그룹화해 반환하고, 그룹 점수 분포로 ``"single" | "multi" | "none"`` 답변 모드를
판정한다.

본 모듈은 abstain / 도메인 정책에 무지하다 — 호출자가 자기 도메인 모델
(``ConfluenceChunk``, ``ExternalToolDef`` 등)과 임계값을 주입한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, List, Literal, Optional, Tuple, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchAny

T = TypeVar("T")
AnswerMode = Literal["single", "multi", "none"]


@dataclass
class CategoryGroup(Generic[T]):
    """그룹 키 값별 검색 결과 묶음.

    Attributes:
        group_key: 그룹 키 값 (예: ``"Messenger"``, ``"hr_service"``).
        pairs: ``(도메인 모델 인스턴스, score)`` 리스트, score 내림차순 보장.
        top_score: ``pairs[0][1]`` 편의 필드. 빈 그룹은 표현하지 않는다.
    """

    group_key: str
    pairs: List[Tuple[T, float]]
    top_score: float


async def search_grouped(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: List[float],
    *,
    group_by: str,
    group_size: int,
    limit: int,
    payload_factory: Callable[[dict], T],
    score_threshold: float = 0.0,
    categories: Optional[List[str]] = None,
) -> List[CategoryGroup[T]]:
    """Qdrant ``query_points_groups`` 결과를 ``CategoryGroup`` 리스트로 반환한다.

    그룹 간은 ``top_score`` 내림차순, 그룹 내 ``pairs``는 score 내림차순으로
    정렬되도록 보장한다. payload가 없거나 hits가 비어 있는 그룹은 제외한다.

    Args:
        client: 연결된 ``AsyncQdrantClient``.
        collection_name: 검색할 컬렉션 이름.
        query_vector: 쿼리 임베딩 벡터.
        group_by: 그룹화에 사용할 payload 필드. KEYWORD 타입 인덱스가 필요하다.
        group_size: 그룹당 반환할 최대 hits 수.
        limit: 반환할 최대 그룹 수.
        payload_factory: raw payload dict를 도메인 모델로 변환하는 함수.
        score_threshold: hit 단위 최소 유사도 임계값. abstain 등 도메인 정책은
            상위 호출자가 별도로 판정한다.
        categories: ``group_by`` 필드 값의 화이트리스트. ``None`` 또는 빈 리스트면
            필터 미적용(전체 검색). 지정하면
            ``Filter(must=[FieldCondition(key=group_by, match=MatchAny(any=categories))])``
            를 적용해 해당 값을 가진 청크만 검색한다. ``group_by`` 필드를 필터 키로
            재사용하는 패턴이므로, 현 호출처는 ``group_by="category"`` 단일이라
            안전하다. 다른 ``group_by``로 호출하는 경로가 생기면 인자명을
            일반화하거나 가드를 추가할 것.

    Returns:
        ``CategoryGroup`` 리스트. 결과가 없으면 빈 리스트를 반환한다.
    """
    query_filter = (
        Filter(must=[FieldCondition(key=group_by, match=MatchAny(any=categories))])
        if categories
        else None
    )

    result = await client.query_points_groups(
        collection_name=collection_name,
        query=query_vector,
        group_by=group_by,
        group_size=group_size,
        limit=limit,
        score_threshold=score_threshold,
        with_payload=True,
        query_filter=query_filter,
    )

    groups: List[CategoryGroup[T]] = []
    for g in result.groups:
        pairs = [
            (payload_factory(hit.payload), float(hit.score))
            for hit in g.hits
            if hit.payload is not None
        ]
        if not pairs:
            continue
        pairs.sort(key=lambda x: x[1], reverse=True)
        groups.append(
            CategoryGroup(
                group_key=str(g.id),
                pairs=pairs,
                top_score=pairs[0][1],
            )
        )

    groups.sort(key=lambda x: x.top_score, reverse=True)
    return groups


def analyze_dominance(
    groups: List[CategoryGroup[T]],
    *,
    dominance_threshold: float,
    runner_up_floor: float,
    max_multi: int = 4,
) -> Tuple[AnswerMode, List[CategoryGroup[T]]]:
    """그룹 점수 분포로 답변 모드를 결정한다 (순수 함수, 도메인 독립).

    판정 규칙:
        - 빈 입력 → ``("none", [])``
        - 그룹 1개 → ``("single", groups[:1])``
        - ``top1 - top2 ≥ dominance_threshold`` 그리고 ``top2 < runner_up_floor`` →
          ``("single", [groups[0]])``
        - 그 외 → ``("multi", runner_up_floor 이상 그룹 [:max_multi])``.
          해당 그룹이 2개 미만이면 ``groups[:2]`` fallback.

    Args:
        groups: ``top_score`` 내림차순으로 정렬된 그룹 리스트.
        dominance_threshold: 단일 분기 판정에 필요한 ``top1 - top2`` 점수 차이.
        runner_up_floor: 2위 이하 그룹이 multi에 포함되기 위한 최소 ``top_score``.
        max_multi: multi 분기 시 포함할 그룹 최대 수.

    Returns:
        ``(answer_mode, 답변 컨텍스트로 넘길 그룹 리스트)``.
    """
    if not groups:
        return "none", []
    if len(groups) == 1:
        return "single", groups[:1]

    g0, g1 = groups[0], groups[1]
    if (g0.top_score - g1.top_score) >= dominance_threshold and g1.top_score < runner_up_floor:
        return "single", [g0]

    multi = [g for g in groups if g.top_score >= runner_up_floor][:max_multi]
    if len(multi) < 2:
        multi = groups[:2]
    return "multi", multi


def merge_groups(
    group_lists: List[List[CategoryGroup[T]]],
    *,
    key: Callable[[T], str],
    group_size: int,
    limit: int,
) -> List[CategoryGroup[T]]:
    """여러 검색 결과(그룹 리스트)를 하나로 합친다 (순수 함수, 도메인 독립).

    같은 항목(``key`` 동일)은 **가장 높은 score** 하나만 남기고, 그룹 키별로 다시 묶어
    ``group_size`` 개씩 ``limit`` 그룹까지 자른다 — 결과 모양(슬롯 수)은 단일 검색과
    같으므로 하류(LLM 컨텍스트·abstain 임계값)는 입력이 늘어난 것을 모른다.
    점수는 코사인 원값을 그대로 쓴다(RRF 같은 순위 융합은 abstain 임계값을 깨뜨린다).

    Args:
        group_lists: 검색어별 ``CategoryGroup`` 리스트. 비어 있는 리스트는 무시한다.
        key: 항목 동일성 키 (예: ``lambda c: c.chunk_id``).
        group_size: 합친 뒤 그룹당 남길 최대 항목 수.
        limit: 합친 뒤 남길 최대 그룹 수.

    Returns:
        그룹 간 ``top_score`` 내림차순, 그룹 내 score 내림차순인 ``CategoryGroup`` 리스트.
    """
    best: dict = {}   # group_key -> {item_key: (item, score)}
    for groups in group_lists:
        for g in groups:
            bucket = best.setdefault(g.group_key, {})
            for item, score in g.pairs:
                k = key(item)
                if k not in bucket or score > bucket[k][1]:
                    bucket[k] = (item, score)

    merged: List[CategoryGroup[T]] = []
    for group_key, bucket in best.items():
        pairs = sorted(bucket.values(), key=lambda x: x[1], reverse=True)[:group_size]
        if pairs:
            merged.append(CategoryGroup(group_key=group_key, pairs=pairs, top_score=pairs[0][1]))
    merged.sort(key=lambda x: x.top_score, reverse=True)
    return merged[:limit]


def flatten_groups(groups: List[CategoryGroup[T]]) -> List[Tuple[T, float]]:
    """그룹들을 평탄화해 score 내림차순 ``(payload, score)`` 리스트로 반환한다.

    abstain gate, 평가 하니스 등 단일 점수 기준 판정에 사용한다.

    Args:
        groups: ``CategoryGroup`` 리스트.

    Returns:
        모든 그룹의 ``pairs``를 score 내림차순으로 평탄화한 리스트.
    """
    flat: List[Tuple[T, float]] = []
    for g in groups:
        flat.extend(g.pairs)
    flat.sort(key=lambda x: x[1], reverse=True)
    return flat
