"""``analyze_dominance`` 순수 함수 단위 테스트."""
from agent_arch.services.retrieval.grouped import CategoryGroup, analyze_dominance


def _g(key: str, top: float) -> CategoryGroup:
    """단일 hit을 가진 헬퍼 그룹 생성."""
    return CategoryGroup(key, [(f"{key}_chunk", top)], top)


def test_empty_groups_returns_none():
    """빈 입력 → ('none', [])."""
    mode, sel = analyze_dominance(
        [], dominance_threshold=0.05, runner_up_floor=0.6
    )
    assert mode == "none"
    assert sel == []


def test_single_group_returns_single():
    """그룹 1개 → ('single', groups[:1])."""
    groups = [_g("A", 0.7)]
    mode, sel = analyze_dominance(
        groups, dominance_threshold=0.05, runner_up_floor=0.6
    )
    assert mode == "single"
    assert sel == groups


def test_dominant_top_returns_single():
    """top1-top2 ≥ threshold 이고 top2 < floor → single."""
    groups = [_g("A", 0.85), _g("B", 0.50), _g("C", 0.45)]
    mode, sel = analyze_dominance(
        groups, dominance_threshold=0.05, runner_up_floor=0.6
    )
    assert mode == "single"
    assert [g.group_key for g in sel] == ["A"]


def test_runner_up_above_floor_forces_multi():
    """top2 ≥ floor → 압도 차이 충족돼도 multi로 빠진다."""
    groups = [_g("A", 0.90), _g("B", 0.65), _g("C", 0.40)]
    mode, sel = analyze_dominance(
        groups, dominance_threshold=0.05, runner_up_floor=0.6
    )
    assert mode == "multi"
    # B만 floor 이상 → A,B 선택, C는 floor 미만이라 제외
    assert [g.group_key for g in sel] == ["A", "B"]


def test_close_scores_returns_multi():
    """top1-top2 < threshold → multi."""
    groups = [_g("A", 0.72), _g("B", 0.71), _g("C", 0.65)]
    mode, sel = analyze_dominance(
        groups, dominance_threshold=0.05, runner_up_floor=0.6
    )
    assert mode == "multi"
    assert [g.group_key for g in sel] == ["A", "B", "C"]


def test_multi_clamped_by_max_multi():
    """max_multi 만큼만 포함된다."""
    groups = [_g(c, 0.7) for c in "ABCDEF"]
    mode, sel = analyze_dominance(
        groups, dominance_threshold=0.05, runner_up_floor=0.6, max_multi=4
    )
    assert mode == "multi"
    assert [g.group_key for g in sel] == ["A", "B", "C", "D"]


def test_multi_fallback_when_none_above_floor():
    """모든 그룹이 floor 미만이고 single dominance도 아닐 때 groups[:2] fallback."""
    # top1-top2 = 0.04 < 0.05 → single 조건 불충족, 모두 floor 미만
    groups = [_g("A", 0.55), _g("B", 0.51), _g("C", 0.45)]
    mode, sel = analyze_dominance(
        groups, dominance_threshold=0.05, runner_up_floor=0.6
    )
    assert mode == "multi"
    assert [g.group_key for g in sel] == ["A", "B"]
