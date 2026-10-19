"""권한 게이트 — 도구 호출 가능 여부 평가.

평가 순서:
    1. ``permission_index`` 에 메타 없음 → **deny** (fail-safe, 명시 등록 강제).
    2. ``profile`` 없음 → deny.
    3. ``profile.is_guest=True`` → ``allow_guest`` 인 경우만 통과.
    4. ``mode == PUBLIC`` → 서비스 사용권 검사 건너뛰고 통과 (calculator 등 범용).
    5. 공통 1차 게이트: ``permissions[svc]`` 에 ``"{svc}.access"`` 포함.
       ``services_any`` 가 있으면(ACCESS_ONLY 전용) 그 목록 중 **어느 한 서비스**
       의 access 라도 있으면 통과 — 물류/MES 공용 도구의 서비스 횡단 OR.
    6. mode별 평가 (PERMISSION/ROLE/SERVICE_ADMIN/ACCESS_ONLY).

권한 raw는 ``profile.permissions`` (서비스 키 → 권한 키 리스트) 하나로 통일됐다.
서비스 사용권은 ``"{svc}.access"``, 서비스 어드민은 ``"{svc}.admin"`` 키로
표현되며, 권한이 없는 서비스는 리스트가 비어 있거나 키 자체가 없다 —
둘 다 deny로 동일하게 처리한다.

profile은 Frappe ``get_current_user`` 응답 dict 그대로. ``user_info``에서
권한 raw가 분리되어 있는 정책(미들웨어 ``_profile_to_user_dict``)을 유지하기
위해, 호출자(orchestrator)가 ``app.services.user_profile.get_profile()``로
별도 fetch한 profile을 인자로 넘긴다.

**거부 사유 구별**: ``evaluate()`` 는 ``GateDecision`` 으로 사유까지 돌려준다.
"권한이 없다"와 "그런 도구가 없다"는 사용자에게 전혀 다른 안내가 필요한데,
bool 하나로는 호출자가 구별할 수 없어 권한 부족도 "기능이 없습니다"로 나갔다.
``is_allowed()`` 는 bool만 필요한 호출자를 위한 하위호환 wrapper다.
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_arch.agents.permissions.index import permission_index
from agent_arch.agents.permissions.spec import (
    DenyReason,
    GateDecision,
    GateMode,
    PermissionRequirement,
)

logger = logging.getLogger(__name__)

# NO_META 경고를 낸 도구 이름. 게이트는 요청마다 도구 수만큼 호출되므로 dedup 없이는
# 같은 경고가 로그를 뒤덮는다. 프로세스 생애 1회만 알리면 진단 목적은 충족된다.
_warned_no_meta: set[str] = set()


def evaluate(tool_name: str, profile: Optional[dict]) -> GateDecision:
    """도구가 현재 사용자에게 허용되는지 평가하고 사유까지 반환한다.

    Args:
        tool_name: ``ToolRegistry`` / ``DynamicAPITool``의 도구 이름.
        profile: Frappe ``get_current_user`` 응답 dict. 미인증 또는 fetch
            실패 시 None.

    Returns:
        ``GateDecision``. ``allowed=False`` 면 ``reason`` 에 거부 사유가 담긴다.
        호출자는 ``decision.is_permission_issue`` 로 "권한 안내를 해야 하는
        거부"인지 판별할 수 있다.
    """
    if _blocked_unattended(tool_name):
        return _deny(tool_name, DenyReason.UNATTENDED, "")

    req = permission_index.get(tool_name)
    if req is None:
        # 메타 미등록 = 명시적 deny. 신규 도구 추가 시 매핑 강제.
        # 다만 이건 정상 동작이 아니라 등록 누락이다 — 이 도구는 아무에게도
        # 노출되지 않는데 사용자에겐 "기능 없음"으로 보인다. 반드시 알린다.
        if tool_name not in _warned_no_meta:
            _warned_no_meta.add(tool_name)
            logger.warning(
                "[gate] 권한 메타 미등록 — 전원 deny (도구가 아무에게도 노출되지 않음): %s. "
                "정적 도구면 클래스에 permission_required 를 선언해야 한다.",
                tool_name,
            )
        return GateDecision(allowed=False, reason=DenyReason.NO_META)

    if profile is None:
        # PUBLIC + allow_no_profile 도구는 dmp 권한 프로필 없이도 통과한다
        # (근거는 PermissionRequirement.allow_no_profile docstring).
        if req.mode == GateMode.PUBLIC and req.allow_no_profile:
            return GateDecision(allowed=True, service=req.service)
        return _deny(tool_name, DenyReason.NO_PROFILE, req.service)

    if bool(profile.get("is_guest")):
        if req.allow_guest:
            return GateDecision(allowed=True, service=req.service)
        return _deny(tool_name, DenyReason.GUEST_BLOCKED, req.service)

    if req.mode == GateMode.PUBLIC:
        return GateDecision(allowed=True, service=req.service)

    # 1차 관문 — services_any 가 있으면(공통 도구, ACCESS_ONLY 전용 — spec 에서
    # 강제) 그 목록이 후보 전체이고 어느 한 서비스의 사용권이면 통과한다.
    # 매칭된 서비스를 GateDecision.service 에 기록해 로그에서 어느 축으로
    # 통과했는지 진단 가능하게 한다 (도메인 축은 req.service 가 별도 담당).
    candidates = req.services_any or (req.service,)
    matched = next(
        (svc for svc in candidates
         if f"{svc}.access" in _service_permissions(profile, svc)),
        None,
    )
    if matched is None:
        return _deny(tool_name, DenyReason.NO_SERVICE_ACCESS, req.service)
    user_perms = _service_permissions(profile, matched)

    if not _mode_satisfied(req, profile, user_perms):
        return _deny(tool_name, DenyReason.MODE_NOT_SATISFIED, matched)

    return GateDecision(allowed=True, service=matched)


def _blocked_unattended(tool_name: str) -> bool:
    """예약 실행(무인) 중인 승인 필요 도구인가. 이 게이트를 지나야 바인딩되므로 모델은 그 도구를 보지 못한다."""
    from agent_arch.agents.context import is_unattended
    if not is_unattended():
        return False
    from agent_arch.agents.hitl.policy import requires_approval
    return requires_approval(tool_name)


def is_allowed(tool_name: str, profile: Optional[dict]) -> bool:
    """도구가 현재 사용자에게 허용되는지 평가한다 (bool만 필요한 호출자용).

    사유가 필요하면 ``evaluate()`` 를 쓴다.

    Args:
        tool_name: ``ToolRegistry`` / ``DynamicAPITool``의 도구 이름.
        profile: Frappe ``get_current_user`` 응답 dict. 미인증 또는 fetch
            실패 시 None.

    Returns:
        True면 LLM에 bind 허용. False면 도구 목록에서 제외해야 한다.
    """
    return evaluate(tool_name, profile).allowed


def _deny(tool_name: str, reason: DenyReason, service: str) -> GateDecision:
    """거부 결정을 만들며 진단 로그를 남긴다.

    NO_META 를 제외한 사유는 모두 **정상 동작**(권한 없는 사용자에게 안 보이는 것)
    이므로 debug 레벨이다. 특정 사용자에게 도구가 왜 안 보이는지 추적할 때만 켠다.
    """
    logger.debug("[gate] deny tool=%s reason=%s service=%s", tool_name, reason.value, service)
    return GateDecision(allowed=False, reason=reason, service=service)


def _service_permissions(profile: dict, service: str) -> set[str]:
    """profile.permissions(dict)에서 해당 서비스의 권한 키 집합을 꺼낸다.

    권한이 없는 서비스는 키가 아예 없거나 빈 리스트로 오므로, 두 경우 모두
    빈 집합으로 통일한다 (게이트에서 deny).
    """
    return set((profile.get("permissions") or {}).get(service) or [])


def _mode_satisfied(
    req: PermissionRequirement,
    profile: dict,
    user_perms: set[str],
) -> bool:
    """1차 게이트(``{svc}.access``) 통과 후 mode별 세분 평가.

    PERMISSION 모드에서 permissions_all/any가 모두 비어 있으면 access 키
    통과만으로 True (= ACCESS_ONLY와 동치). 도구 메타에서 명시적으로
    ACCESS_ONLY를 쓰는 것이 가독성상 권장.

    ※ 현재 PERMISSION/ROLE/SERVICE_ADMIN 모드를 쓰는 도구는 없다 (실사용은
      ACCESS_ONLY/PUBLIC 뿐). 그래서 미충족 사유를 세분하지 않고 호출자가
      ``MODE_NOT_SATISFIED`` 하나로 받는다.
    """
    match req.mode:
        case GateMode.ACCESS_ONLY:
            return True
        case GateMode.SERVICE_ADMIN:
            return f"{req.service}.admin" in user_perms
        case GateMode.ROLE:
            if not req.roles_any:
                return False
            roles = set(profile.get("assigned_roles") or [])
            return any(r in roles for r in req.roles_any)
        case GateMode.PERMISSION:
            if req.permissions_all and not set(req.permissions_all).issubset(user_perms):
                return False
            if req.permissions_any and not (set(req.permissions_any) & user_perms):
                return False
            return True
    return False
