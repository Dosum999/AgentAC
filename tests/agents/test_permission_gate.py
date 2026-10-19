"""권한 게이트 단위 테스트.

평가 순서·모드별 분기·메타 미등록 deny default를 모두 커버한다. 도구는
실제로 안 만들고 ``permission_index.register()``로 가짜 메타만 주입해 게이트
로직만 단독 검증한다.
"""
from __future__ import annotations

import json
import logging

import pytest

from agent_arch.agents.permissions import (
    DenyReason,
    GateMode,
    PermissionRequirement,
    evaluate,
    is_allowed,
    permission_index,
)


@pytest.fixture
def clean_index():
    """각 테스트마다 글로벌 인덱스를 초기화하고 종료 시 복원."""
    snapshot = dict(permission_index._index)
    permission_index.clear()
    yield permission_index
    permission_index._index = snapshot


@pytest.fixture
def profile_employee() -> dict:
    """일반 직원 — logistics view 권한 + hr 접근만."""
    return {
        "is_guest": False,
        "permissions": {
            "logistics": ["logistics.access", "logistics.inventory_view"],
            "hr": ["hr.access"],
        },
        "assigned_roles": [],
    }


@pytest.fixture
def profile_admin() -> dict:
    """관리자 — logistics/talk 어드민 + 회사 관리자 role."""
    return {
        "is_guest": False,
        "permissions": {
            "logistics": [
                "logistics.access",
                "logistics.admin",
                "logistics.inventory_view",
                "logistics.inventory_edit",
                "logistics.settings",
            ],
            "talk": ["talk.access", "talk.admin"],
        },
        "assigned_roles": ["회사 관리자"],
    }


@pytest.fixture
def profile_guest() -> dict:
    return {"is_guest": True}


# ── 메타 미등록 / profile 없음 ──────────────────────────────────────────────

def test_unknown_tool_denied(clean_index, profile_employee):
    """인덱스에 없는 도구는 deny (fail-safe default)."""
    assert is_allowed("nonexistent_tool", profile_employee) is False


def test_no_profile_denied(clean_index):
    clean_index.register(
        "any_tool",
        PermissionRequirement(service="logistics", mode=GateMode.ACCESS_ONLY),
    )
    assert is_allowed("any_tool", None) is False


# ── guest ──────────────────────────────────────────────────────────────────

def test_guest_denied_by_default(clean_index, profile_guest):
    clean_index.register(
        "tool_a",
        PermissionRequirement(service="logistics", mode=GateMode.ACCESS_ONLY),
    )
    assert is_allowed("tool_a", profile_guest) is False


def test_guest_allowed_when_opt_in(clean_index, profile_guest):
    clean_index.register(
        "guest_ok",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.ACCESS_ONLY,
            allow_guest=True,
        ),
    )
    assert is_allowed("guest_ok", profile_guest) is True


# ── 공통 1차: {svc}.access ─────────────────────────────────────────────────

def test_no_service_permissions_key_denied(clean_index, profile_employee):
    """permissions에 서비스 키 자체가 없음 → 차단."""
    clean_index.register(
        "proc_tool",
        PermissionRequirement(service="proc", mode=GateMode.ACCESS_ONLY),
    )
    assert is_allowed("proc_tool", profile_employee) is False


def test_empty_service_permissions_denied(clean_index):
    """권한 없는 서비스는 빈 리스트로 옴 → 차단."""
    profile = {
        "is_guest": False,
        "permissions": {"hr": []},
    }
    clean_index.register(
        "hr_tool",
        PermissionRequirement(service="hr", mode=GateMode.ACCESS_ONLY),
    )
    assert is_allowed("hr_tool", profile) is False


def test_missing_access_key_denied(clean_index):
    """세분 권한이 있어도 ``{svc}.access`` 가 없으면 1차 게이트에서 차단."""
    profile = {
        "is_guest": False,
        "permissions": {"hr": ["hr.settings"]},
    }
    clean_index.register(
        "hr_tool",
        PermissionRequirement(service="hr", mode=GateMode.ACCESS_ONLY),
    )
    assert is_allowed("hr_tool", profile) is False


# ── PERMISSION 모드 ────────────────────────────────────────────────────────

def test_permission_all_pass(clean_index, profile_employee):
    clean_index.register(
        "logistics_view",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.PERMISSION,
            permissions_all=("logistics.inventory_view",),
        ),
    )
    assert is_allowed("logistics_view", profile_employee) is True


def test_permission_all_missing(clean_index, profile_employee):
    clean_index.register(
        "logistics_edit",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.PERMISSION,
            permissions_all=("logistics.inventory_edit",),
        ),
    )
    assert is_allowed("logistics_edit", profile_employee) is False


def test_permission_any_pass(clean_index, profile_employee):
    clean_index.register(
        "logistics_view_or_prod",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.PERMISSION,
            permissions_any=(
                "logistics.inventory_view",
                "logistics.production_view",
            ),
        ),
    )
    assert is_allowed("logistics_view_or_prod", profile_employee) is True


def test_permission_any_missing(clean_index, profile_employee):
    clean_index.register(
        "logistics_strict",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.PERMISSION,
            permissions_any=("logistics.production_edit", "logistics.settings"),
        ),
    )
    assert is_allowed("logistics_strict", profile_employee) is False


# ── ROLE 모드 ──────────────────────────────────────────────────────────────

def test_role_match(clean_index, profile_admin):
    clean_index.register(
        "talk_admin",
        PermissionRequirement(
            service="talk",
            mode=GateMode.ROLE,
            roles_any=("회사 관리자",),
        ),
    )
    assert is_allowed("talk_admin", profile_admin) is True


def test_role_no_match(clean_index, profile_employee):
    """``{svc}.access`` 는 있지만 assigned_roles에 매칭 없음 → 차단."""
    clean_index.register(
        "logistics_role",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.ROLE,
            roles_any=("회사 관리자",),
        ),
    )
    assert is_allowed("logistics_role", profile_employee) is False


def test_role_empty_roles_any_denied(clean_index, profile_admin):
    """ROLE 모드에서 roles_any가 비면 deny (실수 등록 방지)."""
    clean_index.register(
        "bad_role",
        PermissionRequirement(service="talk", mode=GateMode.ROLE),
    )
    assert is_allowed("bad_role", profile_admin) is False


# ── SERVICE_ADMIN 모드 ─────────────────────────────────────────────────────

def test_service_admin_pass(clean_index, profile_admin):
    clean_index.register(
        "logistics_settings",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.SERVICE_ADMIN,
        ),
    )
    assert is_allowed("logistics_settings", profile_admin) is True


def test_service_admin_fail(clean_index, profile_employee):
    clean_index.register(
        "logistics_settings",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.SERVICE_ADMIN,
        ),
    )
    assert is_allowed("logistics_settings", profile_employee) is False


# ── ACCESS_ONLY 모드 ───────────────────────────────────────────────────────

def test_access_only_pass(clean_index, profile_employee):
    clean_index.register(
        "hr_dynamic",
        PermissionRequirement(service="hr", mode=GateMode.ACCESS_ONLY),
    )
    assert is_allowed("hr_dynamic", profile_employee) is True


# ── PUBLIC 모드 ────────────────────────────────────────────────────────────

def test_public_pass_without_service_access(clean_index):
    """PUBLIC은 사용권 검사 건너뜀 — permissions 비어 있어도 통과."""
    profile = {"is_guest": False, "permissions": {}}
    clean_index.register(
        "calculator",
        PermissionRequirement(service="", mode=GateMode.PUBLIC),
    )
    assert is_allowed("calculator", profile) is True


def test_public_guest_denied_by_default(clean_index, profile_guest):
    clean_index.register(
        "calculator",
        PermissionRequirement(service="", mode=GateMode.PUBLIC),
    )
    assert is_allowed("calculator", profile_guest) is False


def test_public_guest_allowed_when_opt_in(clean_index, profile_guest):
    clean_index.register(
        "open_tool",
        PermissionRequirement(service="", mode=GateMode.PUBLIC, allow_guest=True),
    )
    assert is_allowed("open_tool", profile_guest) is True


# ── 인덱스 로드 ────────────────────────────────────────────────────────────

def test_load_from_json(tmp_path, clean_index):
    spec_file = tmp_path / "skill_permissions.json"
    spec_file.write_text(
        json.dumps(
            {
                "tool_a": {
                    "service": "logistics",
                    "mode": "permission",
                    "permissions_all": ["logistics.inventory_view"],
                },
                "tool_b": {
                    "service": "talk",
                    "mode": "role",
                    "roles_any": ["회사 관리자"],
                },
                "tool_c": {
                    "service": "hr",
                    "mode": "access_only",
                    "allow_guest": True,
                },
            }
        ),
        encoding="utf-8",
    )
    size = clean_index.load_from_json(str(spec_file))
    assert size == 3
    assert clean_index.get("tool_a").mode == GateMode.PERMISSION
    assert clean_index.get("tool_a").permissions_all == ("logistics.inventory_view",)
    assert clean_index.get("tool_b").roles_any == ("회사 관리자",)
    assert clean_index.get("tool_c").allow_guest is True


def test_load_missing_file_resets_index(tmp_path, clean_index):
    """파일이 없으면 빈 인덱스로 초기화 (부팅 차단 없음)."""
    clean_index.register(
        "stale_tool",
        PermissionRequirement(service="logistics", mode=GateMode.ACCESS_ONLY),
    )
    size = clean_index.load_from_json(str(tmp_path / "nonexistent.json"))
    assert size == 0
    assert clean_index.get("stale_tool") is None


def test_load_skips_malformed_entry(tmp_path, clean_index):
    """잘못된 항목은 무시하고 정상 항목만 로드 (deny default가 fail-safe)."""
    spec_file = tmp_path / "skill_permissions.json"
    spec_file.write_text(
        json.dumps(
            {
                "good": {"service": "logistics", "mode": "access_only"},
                "bad_mode": {"service": "logistics", "mode": "unknown_mode"},
                "missing_service": {"mode": "permission"},
            }
        ),
        encoding="utf-8",
    )
    size = clean_index.load_from_json(str(spec_file))
    assert size == 1
    assert clean_index.get("good") is not None
    assert clean_index.get("bad_mode") is None
    assert clean_index.get("missing_service") is None


# ── evaluate(): 거부 사유 구별 ──────────────────────────────────────────────
# "권한이 없다"와 "그런 도구가 없다"가 같은 False 로 뭉개지면 사용자에게 잘못된
# 안내("기능이 없습니다")가 나가므로, 사유가 정확히 갈리는지를 고정한다.


@pytest.fixture
def reset_no_meta_warnings():
    """NO_META 경고 dedup 집합을 비운다 (모듈 레벨 전역 상태)."""
    from agent_arch.agents.permissions import gate

    snapshot = set(gate._warned_no_meta)
    gate._warned_no_meta.clear()
    yield gate._warned_no_meta
    gate._warned_no_meta.clear()
    gate._warned_no_meta.update(snapshot)


def test_evaluate_no_meta(clean_index, profile_employee):
    """메타 미등록 → NO_META. 등록 누락(버그) 신호이며 서비스 키를 알 수 없다."""
    decision = evaluate("nonexistent_tool", profile_employee)
    assert decision.allowed is False
    assert decision.reason is DenyReason.NO_META
    assert decision.service == ""


def test_evaluate_no_profile(clean_index):
    clean_index.register(
        "any_tool",
        PermissionRequirement(service="logistics", mode=GateMode.ACCESS_ONLY),
    )
    decision = evaluate("any_tool", None)
    assert decision.reason is DenyReason.NO_PROFILE
    assert decision.service == "logistics"


def test_evaluate_no_profile_allows_opted_in_public_tool(clean_index):
    """PUBLIC + allow_no_profile 은 프로필 없이도 통과한다 (PROC 처럼 dmp 권한 체계 밖 호스트).

    프로필을 그 호스트가 대신 등록하는 길은 막혀 있다 — 프로필은 sub 단위 전체 대체라
    같은 사용자의 dmp 프로필을 덮어쓴다. 그래서 도구가 스스로 선언한다.
    """
    clean_index.register(
        "proc_tool",
        PermissionRequirement(service="", mode=GateMode.PUBLIC, allow_no_profile=True),
    )
    decision = evaluate("proc_tool", None)
    assert decision.allowed is True


def test_evaluate_no_profile_still_denies_plain_public_tool(clean_index):
    """opt-in 안 한 PUBLIC 도구는 그대로 막힌다 — 플래그가 전역 완화가 아님을 못박는다."""
    clean_index.register(
        "plain_public",
        PermissionRequirement(service="", mode=GateMode.PUBLIC),
    )
    decision = evaluate("plain_public", None)
    assert decision.allowed is False
    assert decision.reason is DenyReason.NO_PROFILE


def test_evaluate_no_profile_denies_opted_in_non_public_tool(clean_index):
    """allow_no_profile 은 PUBLIC 모드에서만 유효하다 — 권한 검사를 우회하지 못한다."""
    clean_index.register(
        "svc_tool",
        PermissionRequirement(
            service="logistics", mode=GateMode.ACCESS_ONLY, allow_no_profile=True
        ),
    )
    decision = evaluate("svc_tool", None)
    assert decision.allowed is False
    assert decision.reason is DenyReason.NO_PROFILE


def test_evaluate_guest_blocked(clean_index, profile_guest):
    clean_index.register(
        "tool_a",
        PermissionRequirement(service="logistics", mode=GateMode.ACCESS_ONLY),
    )
    assert evaluate("tool_a", profile_guest).reason is DenyReason.GUEST_BLOCKED


def test_evaluate_no_service_access(clean_index, profile_employee):
    """서비스 사용권 자체가 없음 — 실사용 거부 사유의 거의 전부."""
    clean_index.register(
        "proc_tool",
        PermissionRequirement(service="proc", mode=GateMode.ACCESS_ONLY),
    )
    decision = evaluate("proc_tool", profile_employee)
    assert decision.reason is DenyReason.NO_SERVICE_ACCESS
    assert decision.service == "proc"


@pytest.mark.parametrize(
    "req",
    [
        PermissionRequirement(
            service="logistics",
            mode=GateMode.PERMISSION,
            permissions_all=("logistics.inventory_edit",),
        ),
        PermissionRequirement(
            service="logistics", mode=GateMode.ROLE, roles_any=("회사 관리자",)
        ),
        PermissionRequirement(service="logistics", mode=GateMode.SERVICE_ADMIN),
    ],
    ids=["permission", "role", "service_admin"],
)
def test_evaluate_mode_not_satisfied(clean_index, profile_employee, req):
    """1차 게이트는 통과하고 mode별 세분 조건만 미충족 → MODE_NOT_SATISFIED.

    세 모드를 하나의 사유로 묶은 것은 의도다 — 현재 이 모드를 쓰는 도구가
    없어, 사유를 쪼개면 테스트만 가능한 죽은 분기가 된다.
    """
    clean_index.register("gated", req)
    decision = evaluate("gated", profile_employee)
    assert decision.reason is DenyReason.MODE_NOT_SATISFIED
    assert decision.service == "logistics"


def test_evaluate_allowed_has_no_reason(clean_index, profile_employee):
    clean_index.register(
        "hr_dynamic",
        PermissionRequirement(service="hr", mode=GateMode.ACCESS_ONLY),
    )
    decision = evaluate("hr_dynamic", profile_employee)
    assert decision.allowed is True
    assert decision.reason is None
    assert decision.is_permission_issue is False


@pytest.mark.parametrize(
    "reason, expected",
    [
        (DenyReason.NO_SERVICE_ACCESS, True),
        (DenyReason.MODE_NOT_SATISFIED, True),
        # 아래 셋은 권한을 부여해도 풀리지 않으므로 "관리자에게 권한 요청" 안내가 틀린다.
        (DenyReason.NO_META, False),
        (DenyReason.NO_PROFILE, False),
        (DenyReason.GUEST_BLOCKED, False),
    ],
)
def test_is_permission_issue_classification(reason, expected):
    from agent_arch.agents.permissions import GateDecision

    assert GateDecision(allowed=False, reason=reason).is_permission_issue is expected


def test_is_allowed_agrees_with_evaluate(clean_index, profile_employee):
    """하위호환 — bool wrapper 가 evaluate 와 어긋나지 않는다."""
    clean_index.register(
        "ok_tool", PermissionRequirement(service="hr", mode=GateMode.ACCESS_ONLY)
    )
    clean_index.register(
        "denied_tool", PermissionRequirement(service="proc", mode=GateMode.ACCESS_ONLY)
    )
    for name in ("ok_tool", "denied_tool", "unregistered_tool"):
        assert is_allowed(name, profile_employee) is evaluate(name, profile_employee).allowed


def test_no_meta_warns_once_per_tool(clean_index, profile_employee, reset_no_meta_warnings, caplog):
    """NO_META 는 WARNING 이지만 도구당 1회만 — 게이트는 턴마다 반복 호출된다."""
    with caplog.at_level(logging.WARNING, logger="agent_arch.agents.permissions.gate"):
        for _ in range(3):
            evaluate("ghost_tool", profile_employee)
    warnings = [r for r in caplog.records if "ghost_tool" in r.getMessage()]
    assert len(warnings) == 1


# ── services_any (서비스 횡단 OR 1차 관문 — ACCESS_ONLY 전용) ──────────────
# 물류 MCP 공통 도구(생산·재고 조회)가 쓰는 축: logistics.access 또는 mes.access
# 어느 한쪽만 있어도 통과해야 한다. 근거: logistics-mcp/docs/tool-exposure-plan.md §3.

def _profile_with(*services: str) -> dict:
    return {
        "is_guest": False,
        "permissions": {s: [f"{s}.access"] for s in services},
        "assigned_roles": [],
    }


@pytest.mark.parametrize(
    "services, expect_allowed, expect_matched",
    [
        (("logistics",), True, "logistics"),
        (("mes",), True, "mes"),
        (("logistics", "mes"), True, "logistics"),  # 순서대로 첫 매칭
        (("hr",), False, None),
    ],
)
def test_services_any_or_gate(clean_index, services, expect_allowed, expect_matched):
    clean_index.register(
        "logi_common_tool",
        PermissionRequirement(
            service="logistics",
            mode=GateMode.ACCESS_ONLY,
            services_any=("logistics", "mes"),
        ),
    )
    decision = evaluate("logi_common_tool", _profile_with(*services))
    assert decision.allowed is expect_allowed
    if expect_allowed:
        # 매칭된 서비스가 로그 진단용으로 기록된다 (도메인 축은 req.service 별도)
        assert decision.service == expect_matched
    else:
        assert decision.reason == DenyReason.NO_SERVICE_ACCESS


def test_services_any_empty_falls_back_to_single_service(clean_index):
    """services_any 미지정이면 종전과 동일 — 주 서비스 단일 검사."""
    clean_index.register(
        "logi_only_tool",
        PermissionRequirement(service="logistics", mode=GateMode.ACCESS_ONLY),
    )
    assert evaluate("logi_only_tool", _profile_with("logistics")).allowed is True
    assert evaluate("logi_only_tool", _profile_with("mes")).allowed is False


@pytest.mark.parametrize(
    "mode", [GateMode.PERMISSION, GateMode.ROLE, GateMode.SERVICE_ADMIN, GateMode.PUBLIC]
)
def test_services_any_rejects_non_access_only_modes(mode):
    """세분 권한 키는 서비스 네임스페이스 귀속이라 횡단 조합은 등록 시점에 거부."""
    with pytest.raises(ValueError):
        PermissionRequirement(service="logistics", mode=mode, services_any=("logistics", "mes"))


def test_services_any_from_dict_roundtrip():
    req = PermissionRequirement.from_dict(
        {"service": "logistics", "mode": "access_only", "services_any": ["logistics", "mes"]}
    )
    assert req.services_any == ("logistics", "mes")
