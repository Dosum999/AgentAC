"""``app.services.retrieval.grouped`` 단위 테스트.

Qdrant ``query_points_groups`` 호출은 모킹하고, 결과 변환·정렬·평탄화를 검증한다.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from qdrant_client.http.models import FieldCondition, Filter, MatchAny

from agent_arch.services.retrieval.grouped import (
    CategoryGroup,
    flatten_groups,
    search_grouped,
)


def _hit(payload, score):
    """``ScoredPoint`` 흉내 — payload/score 속성만 갖도록 한다."""
    return MagicMock(payload=payload, score=score)


def _group(group_id, hits):
    """``PointGroup`` 흉내 — id/hits 속성만 갖도록 한다."""
    return MagicMock(id=group_id, hits=hits)


def _fake_client(groups):
    """``query_points_groups`` 만 모킹된 가짜 client."""
    client = MagicMock()
    client.query_points_groups = AsyncMock(return_value=MagicMock(groups=groups))
    return client


@pytest.mark.asyncio
async def test_search_grouped_orders_groups_and_pairs():
    """그룹은 top_score 내림차순, 그룹 내 pairs는 score 내림차순으로 정렬된다."""
    g1 = _group("A", [_hit({"v": "a1"}, 0.6), _hit({"v": "a2"}, 0.8)])
    g2 = _group("B", [_hit({"v": "b1"}, 0.9)])
    g3 = _group("C", [_hit({"v": "c1"}, 0.7)])
    client = _fake_client([g1, g2, g3])

    groups = await search_grouped(
        client,
        "col",
        [0.0] * 3,
        group_by="cat",
        group_size=2,
        limit=5,
        payload_factory=lambda p: p["v"],
    )

    assert [g.group_key for g in groups] == ["B", "A", "C"]
    assert [g.top_score for g in groups] == [0.9, 0.8, 0.7]
    # 그룹 A 내부 정렬 확인
    a = next(g for g in groups if g.group_key == "A")
    assert a.pairs == [("a2", 0.8), ("a1", 0.6)]


@pytest.mark.asyncio
async def test_search_grouped_empty():
    """결과가 없으면 빈 리스트를 반환한다."""
    client = _fake_client([])
    groups = await search_grouped(
        client,
        "col",
        [0.0],
        group_by="cat",
        group_size=1,
        limit=1,
        payload_factory=lambda p: p,
    )
    assert groups == []


@pytest.mark.asyncio
async def test_search_grouped_skips_none_payload():
    """payload가 None인 hit은 제외, 모두 None인 그룹은 그룹 자체가 제외된다."""
    g1 = _group("A", [_hit(None, 0.9)])
    g2 = _group("B", [_hit({"v": "b1"}, 0.7), _hit(None, 0.6)])
    client = _fake_client([g1, g2])

    groups = await search_grouped(
        client,
        "col",
        [0.0],
        group_by="cat",
        group_size=2,
        limit=5,
        payload_factory=lambda p: p["v"],
    )

    assert [g.group_key for g in groups] == ["B"]
    assert groups[0].pairs == [("b1", 0.7)]


@pytest.mark.asyncio
async def test_search_grouped_passes_qdrant_kwargs():
    """Qdrant 클라이언트에 group_by/group_size/limit/score_threshold 가 그대로 전달된다."""
    client = _fake_client([])
    await search_grouped(
        client,
        "user_guide_kb",
        [0.1, 0.2],
        group_by="category",
        group_size=3,
        limit=4,
        payload_factory=lambda p: p,
        score_threshold=0.25,
    )
    client.query_points_groups.assert_awaited_once()
    kwargs = client.query_points_groups.await_args.kwargs
    assert kwargs["collection_name"] == "user_guide_kb"
    assert kwargs["group_by"] == "category"
    assert kwargs["group_size"] == 3
    assert kwargs["limit"] == 4
    assert kwargs["score_threshold"] == 0.25
    assert kwargs["with_payload"] is True


@pytest.mark.asyncio
async def test_search_grouped_no_filter_when_categories_none():
    """``categories=None`` → ``query_filter=None`` (필터 미적용)."""
    client = _fake_client([])
    await search_grouped(
        client,
        "user_guide_kb",
        [0.1],
        group_by="category",
        group_size=1,
        limit=1,
        payload_factory=lambda p: p,
    )
    kwargs = client.query_points_groups.await_args.kwargs
    assert kwargs["query_filter"] is None


@pytest.mark.asyncio
async def test_search_grouped_no_filter_when_categories_empty():
    """``categories=[]`` (빈 리스트) → ``query_filter=None`` (전체 검색)."""
    client = _fake_client([])
    await search_grouped(
        client,
        "user_guide_kb",
        [0.1],
        group_by="category",
        group_size=1,
        limit=1,
        payload_factory=lambda p: p,
        categories=[],
    )
    kwargs = client.query_points_groups.await_args.kwargs
    assert kwargs["query_filter"] is None


@pytest.mark.asyncio
async def test_search_grouped_single_category_filter():
    """``categories=["Messenger"]`` → ``MatchAny(any=["Messenger"])`` 필터."""
    client = _fake_client([])
    await search_grouped(
        client,
        "user_guide_kb",
        [0.1],
        group_by="category",
        group_size=1,
        limit=1,
        payload_factory=lambda p: p,
        categories=["Messenger"],
    )
    kwargs = client.query_points_groups.await_args.kwargs
    expected = Filter(
        must=[FieldCondition(key="category", match=MatchAny(any=["Messenger"]))]
    )
    assert kwargs["query_filter"] == expected


@pytest.mark.asyncio
async def test_search_grouped_multi_category_filter():
    """``categories=["Messenger", "Drive"]`` → ``MatchAny`` 다중 값 필터."""
    client = _fake_client([])
    await search_grouped(
        client,
        "user_guide_kb",
        [0.1],
        group_by="category",
        group_size=1,
        limit=1,
        payload_factory=lambda p: p,
        categories=["Messenger", "Drive"],
    )
    kwargs = client.query_points_groups.await_args.kwargs
    expected = Filter(
        must=[
            FieldCondition(
                key="category", match=MatchAny(any=["Messenger", "Drive"])
            )
        ]
    )
    assert kwargs["query_filter"] == expected


def test_flatten_groups_sorted_desc():
    """flatten_groups은 모든 pairs를 score 내림차순으로 평탄화한다."""
    groups = [
        CategoryGroup("B", [("b1", 0.9)], 0.9),
        CategoryGroup("A", [("a2", 0.8), ("a1", 0.6)], 0.8),
        CategoryGroup("C", [("c1", 0.7)], 0.7),
    ]
    flat = flatten_groups(groups)
    assert [s for _, s in flat] == [0.9, 0.8, 0.7, 0.6]


def test_flatten_groups_empty():
    """빈 입력 → 빈 리스트."""
    assert flatten_groups([]) == []


# ── merge_groups (원문 병합용 순수 함수) ──────────────────────────────────────


def _cg(key, pairs):
    from agent_arch.services.retrieval.grouped import CategoryGroup
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return CategoryGroup(group_key=key, pairs=pairs, top_score=pairs[0][1])


class TestMergeGroups:
    def test_same_item_keeps_max_score_and_regroups(self):
        from agent_arch.services.retrieval.grouped import merge_groups
        a = [_cg("Messenger", [("m1", 0.70), ("m2", 0.66)]), _cg("Drive", [("d1", 0.62)])]
        b = [_cg("Messenger", [("m1", 0.76), ("m3", 0.71)]), _cg("CS", [("c1", 0.74)])]
        merged = merge_groups([a, b], key=lambda s: s, group_size=2, limit=4)
        by = {g.group_key: g for g in merged}
        # m1 은 두 결과에 다 있음 → 큰 점수 하나. 그룹당 2개 → m2(0.66) 탈락
        assert [k for k, _ in by["Messenger"].pairs] == ["m1", "m3"]
        assert by["Messenger"].pairs[0][1] == pytest.approx(0.76)
        # 그룹 순서 = top_score 내림차순
        assert [g.group_key for g in merged] == ["Messenger", "CS", "Drive"]

    def test_limit_caps_groups_and_empty_lists_ignored(self):
        from agent_arch.services.retrieval.grouped import merge_groups
        a = [_cg("A", [("a", 0.9)]), _cg("B", [("b", 0.8)]), _cg("C", [("c", 0.7)])]
        merged = merge_groups([[], a, []], key=lambda s: s, group_size=2, limit=2)
        assert [g.group_key for g in merged] == ["A", "B"]

    def test_single_list_is_identity_shape(self):
        from agent_arch.services.retrieval.grouped import merge_groups
        a = [_cg("A", [("a1", 0.9), ("a2", 0.5)]), _cg("B", [("b", 0.8)])]
        merged = merge_groups([a], key=lambda s: s, group_size=2, limit=6)
        assert [(g.group_key, [k for k, _ in g.pairs]) for g in merged] == [("A", ["a1", "a2"]), ("B", ["b"])]
