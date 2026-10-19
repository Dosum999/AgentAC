"""source_routing 모듈 단위 테스트.

resolve_defaults: prefix 매칭 / 빈 입력 / 매핑 미스 경로 검증.
apply_source_defaults: Form 명시값 > X-Source-Path 매핑 > None 우선순위 검증.

_PREFIX_MAP 이 비어 있을 때(어떤 source_path 도 (None, None) 으로 폴백)와,
매핑이 채워졌을 때(우선순위가 깨지지 않는지)를 monkeypatch 로 함께 검증한다.
"""
import pytest

from agent_arch.agents import source_routing
from agent_arch.agents.source_routing import apply_source_defaults, resolve_defaults


@pytest.fixture
def empty_map(monkeypatch):
    """_PREFIX_MAP 이 빈 상태를 강제한다."""
    monkeypatch.setattr(source_routing, "_PREFIX_MAP", {})


@pytest.fixture
def sample_map(monkeypatch):
    """샘플 매핑을 주입한다.

    guide_categories 값은 Qdrant user_guide_kb 의 실제 category 값과 일치한다
    (app/services/guide_kb/ingestion/chunker.py:25-30 참조).
    """
    monkeypatch.setattr(source_routing, "_PREFIX_MAP", {
        "/hr": {"agent_mode": "guide", "guide_categories": "HR System"},
        "/hr/leave": {"agent_mode": "guide", "guide_categories": "HR System,General"},
        "/logistics": {"agent_mode": "guide", "guide_categories": "Logistics"},
    })


class TestResolveDefaults:
    """resolve_defaults: source_path 입력 → 매핑 dict 출력."""

    def test_none_returns_empty_dict(self, empty_map):
        assert resolve_defaults(None) == {}

    def test_empty_string_returns_empty_dict(self, empty_map):
        assert resolve_defaults("") == {}

    def test_miss_returns_empty_dict(self, sample_map):
        assert resolve_defaults("/unknown/path") == {}

    def test_exact_prefix_match(self, sample_map):
        assert resolve_defaults("/hr") == {
            "agent_mode": "guide",
            "guide_categories": "HR System",
        }

    def test_prefix_match_with_trailing_segment(self, sample_map):
        assert resolve_defaults("/hr/employee") == {
            "agent_mode": "guide",
            "guide_categories": "HR System",
        }

    def test_longest_prefix_wins(self, sample_map):
        # "/hr/leave" 가 "/hr" 보다 더 긴 매칭이므로 우선
        assert resolve_defaults("/hr/leave/new") == {
            "agent_mode": "guide",
            "guide_categories": "HR System,General",
        }

    def test_similar_prefix_is_not_matched(self, sample_map):
        # "/hr2" 는 "/hr" 의 segment prefix 가 아님
        # (startswith("/hr/") False, == "/hr" False)
        assert resolve_defaults("/hr2") == {}


class TestApplySourceDefaults:
    """apply_source_defaults: 우선순위 Form > 매핑 > None."""

    def test_all_inputs_none_returns_pair_of_none(self, empty_map):
        assert apply_source_defaults(None, None, None) == (None, None)

    def test_empty_map_provides_no_fallback(self, empty_map):
        # _PREFIX_MAP 비어있으면 source_path 가 있어도 fallback 없음
        assert apply_source_defaults(None, None, "/hr/leave") == (None, None)

    def test_form_value_wins_over_mapping(self, sample_map):
        # Form 명시값은 항상 매핑을 덮는다 (기존 호출 100% 호환 보장)
        mode, cats = apply_source_defaults("demo", "Custom", "/hr/leave")
        assert mode == "demo"
        assert cats == "Custom"

    def test_mapping_fills_when_form_omitted(self, sample_map):
        mode, cats = apply_source_defaults(None, None, "/logistics")
        assert mode == "guide"
        assert cats == "Logistics"

    def test_partial_form_override_keeps_mapping_for_unset_field(self, sample_map):
        # agent_mode 만 Form 으로 명시, guide_categories 는 매핑 fallback
        mode, cats = apply_source_defaults("agent", None, "/hr/leave")
        assert mode == "agent"
        assert cats == "HR System,General"

    def test_no_source_path_no_mapping(self, sample_map):
        assert apply_source_defaults(None, None, None) == (None, None)

    def test_empty_form_string_treated_as_unset(self, sample_map):
        # FastAPI Form 이 빈 문자열을 보낼 수도 있음 — falsy 이므로 매핑으로 떨어져야 함
        mode, cats = apply_source_defaults("", "", "/hr")
        assert mode == "guide"
        assert cats == "HR System"


class TestProductionPrefixMap:
    """실제 시드된 _PREFIX_MAP (monkeypatch 없음) 동작 검증.

    HR 시드 매핑이 의도대로 동작하는지 smoke 테스트한다.
    _PREFIX_MAP 이 확장되면 본 테스트도 함께 보강.
    """

    def test_hr_seed_maps_to_guide_with_hr_system(self):
        # /hr 단일 prefix 가 모든 하위 path 를 커버해야 한다
        result = resolve_defaults("/hr/leave/new")
        assert result.get("agent_mode") == "guide"
        assert result.get("guide_categories") == "HR System"

    def test_hr_seed_exact_prefix_match(self):
        # /hr 자체도 매칭되어야 한다
        result = resolve_defaults("/hr")
        assert result.get("agent_mode") == "guide"
        assert result.get("guide_categories") == "HR System"

    def test_non_seeded_prefix_returns_empty(self):
        # 시드되지 않은 prefix 는 fallback 없음 (settings 디폴트로 떨어지도록)
        assert resolve_defaults("/logistics/order") == {}
        assert resolve_defaults("/dmp/dashboard") == {}


# ── 사용량 관측용 서비스 축 ────────────────────────────────────────────────────
class TestResolveService:
    """요청 경로 → 서비스 축. 사용량이 어느 면에 청구될지를 정하는 판정이라 경계가 중요하다."""

    @pytest.mark.parametrize(
        "path,expected",
        [
            # 실제 요청 경로 — 라우터가 전부 settings.api_v1_prefix 아래 걸린다.
            ("/api/v1/agent/stream", "browser_panel"),
            ("/api/v1/agent", "browser_panel"),
            ("/api/v1/audio/transcribe", "browser_panel"),
            ("/api/v1/guest/run", "guest"),
            ("/api/v1/agent/app/run", "app"),
        ],
    )
    def test_maps_real_request_paths(self, path, expected):
        """`request.url.path` 원본(= prefix 포함)을 그대로 넣어도 판정돼야 한다.

        회귀 방지: 매핑 키만 보고 prefix 없는 경로로 시험하면 코드와 테스트가 같은 착각을
        공유해 통과한다. 실제로 운영에서 신규 트래픽이 전량 미분류로 적재됐던 원인이다.
        """
        assert source_routing.resolve_service(path) == expected

    @pytest.mark.parametrize(
        "path,expected",
        [
            ("/agent/stream", "browser_panel"),
            ("/agent", "browser_panel"),
            ("/audio/transcribe", "browser_panel"),
            ("/guest/run", "guest"),
        ],
    )
    def test_maps_known_prefixes(self, path, expected):
        """prefix 가 안 붙은 형태도 받는다 — 내부 호출이 경로를 어디서 얻든 동작해야 한다."""
        assert source_routing.resolve_service(path) == expected

    def test_prefix_itself_is_unmatched(self):
        """prefix 만 온 경우("/api/v1")는 어떤 면도 아니다 — 지어내면 남의 청구에 섞인다."""
        assert source_routing.resolve_service("/api/v1") is None
        assert source_routing.resolve_service("/api/v1/") is None

    def test_longest_prefix_wins(self):
        """'/agent/app' 가 '/agent' 보다 먼저 잡혀야 전용 앱 사용량이 브라우저 패널로 안 섞인다."""
        assert source_routing.resolve_service("/agent/app/run") == "app"
        assert source_routing.resolve_service("/agent/app") == "app"

    def test_prefix_must_be_a_path_segment(self):
        """'/agentic' 이 '/agent' 로 잡히면 남의 사용량이 브라우저 패널 청구에 들어간다."""
        assert source_routing.resolve_service("/agentic/run") is None
        assert source_routing.resolve_service("/api/v1/agentic/run") is None

    @pytest.mark.parametrize("path", [None, "", "/unknown", "/api/v1/unknown"])
    def test_returns_none_when_unmatched(self, path):
        """판정 실패 시 기본 서비스를 지어내지 않는다 — 미분류로 남겨야 화면에서 발견된다."""
        assert source_routing.resolve_service(path) is None


class TestMemorySurface:
    """기억의 자리 — 사용량 서비스 축과 따로 판정한다(/audio 가 두 축에서 갈린다)."""

    def test_mes_app_chat_and_voice_share_one_window(self):
        from agent_arch.agents.source_routing import resolve_memory_surface

        assert resolve_memory_surface("/api/v1/agent/app/run") == "app"
        assert resolve_memory_surface("/api/v1/agent/app/stream") == "app"
        assert resolve_memory_surface("/api/v1/audio/run") == "app"
        assert resolve_memory_surface("/audio/stream") == "app"      # prefix 없는 형태도 받는다

    def test_other_surfaces_have_no_place(self):
        from agent_arch.agents.source_routing import resolve_memory_surface

        assert resolve_memory_surface("/api/v1/agent/run") is None   # 브라우저 패널
        assert resolve_memory_surface(None) is None

    def test_usage_axis_is_unchanged(self):
        """서비스 축은 그대로 — /audio 는 사용량으로는 패널이고 자리로는 전용 앱 창이다."""
        from agent_arch.agents.source_routing import resolve_service

        assert resolve_service("/api/v1/audio/run") == "browser_panel"
        assert resolve_service("/api/v1/agent/app/run") == "app"
