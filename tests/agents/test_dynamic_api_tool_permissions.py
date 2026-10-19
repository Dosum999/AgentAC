"""DynamicAPITool 생성 시 권한 메타가 PermissionIndex에 올바르게 등록되는지 검증.

이 테스트는 DynamicAPITool의 권한 메타 자동 부여 로직만 검증한다 (HTTP 호출,
sanitize 등 다른 책임은 별도 테스트에서 다룬다).

google.genai dev dep 누락 우회:
    api_registry/__init__.py 가 indexer를 top-level import 하기 때문에
    부모 패키지 import 시점에 ``google.genai``가 필요하다. 운영에선 설치돼
    있지만 일부 dev 환경에선 누락될 수 있어, 테스트 콜렉션이 깨지지 않도록
    최소 stub만 주입한다.
"""
from __future__ import annotations

import sys
from types import ModuleType


def _ensure_genai_stub() -> None:
    """google.genai 미설치 환경에서 indexer import가 깨지지 않도록 stub 주입.

    real SDK가 설치된 환경에서는 실제 모듈을 import 해 ``sys.modules``에 캐시한다.
    이렇게 하면 빈 stub 이 ``google.genai.types`` 를 덮어써 다른 테스트
    (예: test_gemini.py)의 ``GenerateContentConfig``/``FunctionCall`` 접근을
    깨뜨리는 오염을 import 순서와 무관하게 방지한다.
    """
    if "google.genai" in sys.modules:
        return
    try:
        import google.genai  # noqa: F401  # real SDK 존재 → 캐시만 하고 stub 미주입
        return
    except ImportError:
        pass
    google_mod = sys.modules.setdefault("google", ModuleType("google"))
    genai_mod = ModuleType("google.genai")
    genai_mod.Client = type("Client", (), {})
    types_mod = ModuleType("google.genai.types")
    types_mod.EmbedContentConfig = type("EmbedContentConfig", (), {})
    genai_mod.types = types_mod
    google_mod.genai = genai_mod
    sys.modules["google.genai"] = genai_mod
    sys.modules["google.genai.types"] = types_mod


_ensure_genai_stub()

import pytest

from agent_arch.agents.permissions import GateMode, permission_index
from agent_arch.agents.tools.dynamic_api_tool import DynamicAPITool
from agent_arch.services.api_registry.models import ExternalToolDef


def _make_def(
    tool_name: str,
    *,
    external_service: str | None = "logistics_service",
    http_method: str = "GET",
    required_permissions: dict | None = None,
) -> ExternalToolDef:
    """간결한 ExternalToolDef 빌더."""
    payload: dict = {
        "tool_name": tool_name,
        "description": "test",
        "is_external": 1,
        "external_service": external_service,
        "http_method": http_method,
        "endpoint_path": "/test/",
        "parameters_schema": {"type": "object", "properties": {}},
        "enabled": 1,
    }
    if required_permissions is not None:
        payload["required_permissions"] = required_permissions
    return ExternalToolDef(**payload)


@pytest.fixture(autouse=True)
def clean_index():
    """각 테스트마다 PermissionIndex 초기화 (테스트 간 격리)."""
    snapshot = dict(permission_index._index)
    permission_index.clear()
    yield
    permission_index._index = snapshot


# ── required_permissions 명시된 경우 ────────────────────────────────────────

def test_required_permissions_registered_as_is():
    """tool_def에 required_permissions가 있으면 그대로 등록."""
    DynamicAPITool(
        _make_def(
            "logistics_service_inventory_get_one",
            required_permissions={
                "service": "logistics",
                "mode": "permission",
                "permissions_all": ["logistics.inventory_view"],
            },
        )
    )
    req = permission_index.get("logistics_service_inventory_get_one")
    assert req is not None
    assert req.mode == GateMode.PERMISSION
    assert req.service == "logistics"
    assert req.permissions_all == ("logistics.inventory_view",)


def test_required_permissions_access_only_mode():
    DynamicAPITool(
        _make_def(
            "hr_service_employees_list",
            external_service="hr_service",
            required_permissions={"service": "hr", "mode": "access_only"},
        )
    )
    req = permission_index.get("hr_service_employees_list")
    assert req is not None
    assert req.mode == GateMode.ACCESS_ONLY
    assert req.service == "hr"


# ── required_permissions 없을 때 fallback ──────────────────────────────────

def test_fallback_access_only_from_external_service():
    """required_permissions 없으면 external_service로 ACCESS_ONLY fallback.

    logistics_service 는 전용 도메인 "mespoc" 에 매핑된다(SERVICE_ALIAS, 2026-08-24).
    폴백은 ACCESS_ONLY 하드코딩 — mespoc.access 는 없는 키라 전원 deny(fail-safe,
    231건 전부 명시 메타 보유라 실경로 없음).
    """
    DynamicAPITool(_make_def("mespoc_service_new_tool", external_service="logistics_service"))
    req = permission_index.get("mespoc_service_new_tool")
    assert req is not None
    assert req.mode == GateMode.ACCESS_ONLY
    assert req.service == "mespoc"


def test_fallback_normalizes_service_alias():
    """SERVICE_ALIAS 매핑이 fallback에도 적용된다 (hr_service → hr, mattermost_service → talk)."""
    DynamicAPITool(_make_def("hr_service_x", external_service="hr_service"))
    DynamicAPITool(_make_def("mattermost_service_y", external_service="mattermost_service"))
    assert permission_index.get("hr_service_x").service == "hr"
    assert permission_index.get("mattermost_service_y").service == "talk"


def test_fallback_unknown_service_passthrough():
    """SERVICE_ALIAS에 없는 external_service는 그대로 사용 (운영자가 새 서비스 추가 시 무리 없음)."""
    DynamicAPITool(_make_def("some_tool", external_service="brand_new_service"))
    req = permission_index.get("some_tool")
    assert req is not None
    assert req.service == "brand_new_service"


# ── 등록 안 됨 (게이트가 deny) ─────────────────────────────────────────────

def test_no_external_service_not_registered():
    """required_permissions도 external_service도 없으면 등록 안 함 → 게이트 deny."""
    DynamicAPITool(_make_def("orphan_tool", external_service=None))
    assert permission_index.get("orphan_tool") is None


def test_empty_external_service_not_registered():
    DynamicAPITool(_make_def("orphan_tool_2", external_service=""))
    assert permission_index.get("orphan_tool_2") is None


# ── 잘못된 required_permissions ────────────────────────────────────────────

def test_malformed_required_permissions_falls_back():
    """required_permissions 파싱 실패 → external_service fallback으로 진행."""
    DynamicAPITool(
        _make_def(
            "logistics_service_bad_meta",
            required_permissions={"mode": "permission"},  # service 키 누락
        )
    )
    req = permission_index.get("logistics_service_bad_meta")
    assert req is not None
    assert req.mode == GateMode.ACCESS_ONLY  # fallback
    assert req.service == "mespoc"  # logistics_service → "mespoc" (2026-08-24 PUBLIC 전환·도메인 이전)


def test_malformed_meta_no_external_service_not_registered():
    """잘못된 메타 + external_service 없음 → 등록 안 됨."""
    DynamicAPITool(
        _make_def(
            "fully_broken",
            external_service=None,
            required_permissions={"mode": "invalid_mode_string"},
        )
    )
    assert permission_index.get("fully_broken") is None
