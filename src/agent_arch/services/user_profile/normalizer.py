"""외부 사용자 프로필을 Agent 내부 canonical 형태로 정규화한다.

지원하는 입력 계약:

* 레거시 Frappe ``get_current_user``: ``profile.user``가 이메일 문자열
* REST v1 ``GET /api/v1/me/context``: ``profile.user``가 사용자 객체이고
  권한은 ``profile.authorization`` 아래에 위치

Agent 내부 소비자는 기존 평면 키(``user``, ``full_name``, ``permissions`` 등)를
계속 사용한다. REST v1의 최상위 context는 보존하고, 중첩된 사용자·권한·직원
필드는 기존 소비자가 읽는 호환 키로 평탄화한다.
"""
from __future__ import annotations

from typing import Any


class InvalidUserProfileError(ValueError):
    """지원하지 않거나 사용자 식별자가 없는 프로필 입력."""


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _first_text(*values: Any) -> str:
    for value in values:
        text = _text(value)
        if text:
            return text
    return ""


def _normalize_roles(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []

    roles: list[str] = []
    for item in raw:
        if isinstance(item, dict):
            role = _first_text(item.get("label"), item.get("id"))
        else:
            role = _text(item)
        if role:
            roles.append(role)
    return roles


def _merge_service_access(permissions: Any, service_access: Any) -> dict[str, Any]:
    """``service_access`` 사용권을 권한 게이트가 읽는 키로 합성해 병합한다.

    REST v1 계약은 권한을 두 축으로 나눈다 — ``authorization.permissions``에는
    세분 권한이 관리되는 서비스만 오고, 단순 사용권·서비스 관리자 여부는
    ``service_access[].has_access / is_service_admin``으로 분리됐다(게이트웨이
    확인: 의도된 계약). 권한 게이트(app/agents/permissions/gate.py)는
    ``permissions[svc]``의 ``"{svc}.access"``/``"{svc}.admin"`` 키만 보므로,
    여기서 합성하지 않으면 access-only 서비스(hr·talk·calendar·drive 등)가
    일괄 deny된다

    Args:
        permissions: 서비스 키 → 권한 키 리스트 dict (없으면 빈 dict 취급)
        service_access: ``{"service", "has_access", "is_service_admin"}`` 리스트

    Returns:
        입력을 변형하지 않은 새 permissions dict. ``has_access=True`` 서비스에
        ``.access``, ``is_service_admin=True``면 ``.admin``까지 중복 없이 추가.
    """
    merged: dict[str, Any] = {}
    if isinstance(permissions, dict):
        for svc, keys in permissions.items():
            merged[svc] = list(keys) if isinstance(keys, list) else keys

    for entry in service_access if isinstance(service_access, list) else []:
        if not isinstance(entry, dict):
            continue
        svc = _text(entry.get("service"))
        if not svc or not entry.get("has_access"):
            continue
        keys = merged.get(svc)
        if not isinstance(keys, list):
            keys = []
            merged[svc] = keys
        for key in (f"{svc}.access",) + (
            (f"{svc}.admin",) if entry.get("is_service_admin") else ()
        ):
            if key not in keys:
                keys.append(key)
    return merged


def _normalize_companies(raw: Any) -> Any:
    """REST 회사 키를 레거시 소비자가 읽는 별칭과 함께 보존한다."""
    if not isinstance(raw, list):
        return raw

    companies: list[Any] = []
    for item in raw:
        if not isinstance(item, dict):
            companies.append(item)
            continue
        company = dict(item)
        if "company_group" not in company:
            company["company_group"] = item.get("company_group_id")
        companies.append(company)
    return companies


def _normalize_employee(raw: Any) -> Any:
    """REST v1의 중첩 Employee를 기존 Agent 평면 키로 변환한다."""
    if raw is None or not isinstance(raw, dict):
        return raw

    employee = dict(raw)
    department = raw.get("department")
    designation = raw.get("designation")
    position = raw.get("position")
    employment_type = raw.get("employment_type")
    contacts = raw.get("contacts")
    work_address = raw.get("work_address")
    personal = raw.get("personal")

    if isinstance(department, dict):
        employee["department"] = department.get("id") or ""
        employee["departmentName"] = department.get("name") or ""
        employee["parentDepartment"] = department.get("parent_name") or ""
    if isinstance(designation, dict):
        employee["designation"] = designation.get("id") or ""
        employee["designationName"] = designation.get("name") or ""
    if isinstance(position, dict):
        employee["position"] = position.get("id") or ""
        employee["positionName"] = position.get("name") or ""
    if isinstance(employment_type, dict):
        employee["employmentType"] = employment_type.get("id") or ""
        employee["isRegular"] = employment_type.get("is_regular")
    if isinstance(contacts, dict):
        employee["workEmail"] = contacts.get("work_email") or ""
        employee["workPhone"] = contacts.get("work_phone") or ""
        employee["extension"] = contacts.get("extension") or ""
        employee["workFax"] = contacts.get("work_fax") or ""
        employee["phone"] = contacts.get("mobile_no") or ""
        employee["personalEmail"] = contacts.get("alternative_email") or ""
        employee["subEmail"] = contacts.get("alternative_email") or ""
    if isinstance(work_address, dict):
        employee["workAddress"] = work_address.get("address_1") or ""
        employee["workAddressDetail"] = work_address.get("address_2") or ""
    if isinstance(personal, dict):
        employee["birth"] = personal.get("birth_date") or ""
        employee["address"] = personal.get("address") or ""
        employee["bio"] = personal.get("bio") or ""

    employee["company"] = raw.get("company_id") or employee.get("company") or ""
    employee["joinDate"] = raw.get("date_of_joining") or employee.get("joinDate") or ""
    return employee


def normalize_user_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """외부 프로필을 Agent 내부 평면 계약으로 변환한다.

    레거시 프로필은 복사 후 그대로 유지한다. REST v1 프로필은 중첩 사용자와 권한을
    평탄화한다. 두 계약 모두 사용자 식별자는 비어 있지 않아야 한다.
    """
    if not isinstance(profile, dict):
        raise InvalidUserProfileError("profile must be an object")

    raw_user = profile.get("user")
    if isinstance(raw_user, str):
        email = _text(raw_user)
        if not email:
            raise InvalidUserProfileError("profile.user is required")
        normalized = dict(profile)
        normalized["user"] = email
        # 이미 평탄화된(v1 산출) 프로필이 재정규화되는 경로 — 사용권 합성은
        # service_access가 실제로 있을 때만 수행해 레거시 원형을 보존한다.
        service_access = profile.get("service_access")
        if isinstance(service_access, list) and service_access:
            normalized["permissions"] = _merge_service_access(
                profile.get("permissions"), service_access
            )
        return normalized

    if not isinstance(raw_user, dict):
        raise InvalidUserProfileError(
            "profile.user must be a string or an object with email/id"
        )

    email = _first_text(raw_user.get("email"), raw_user.get("id"))
    if not email:
        raise InvalidUserProfileError(
            "profile.user.email or profile.user.id is required"
        )

    authorization = profile.get("authorization")
    authorization = authorization if isinstance(authorization, dict) else {}

    permissions = authorization.get("permissions")
    if not isinstance(permissions, dict):
        existing_permissions = profile.get("permissions")
        permissions = (
            existing_permissions if isinstance(existing_permissions, dict) else {}
        )

    service_access = authorization.get("service_access")
    if not isinstance(service_access, list):
        existing_access = profile.get("service_access")
        service_access = existing_access if isinstance(existing_access, list) else []

    if service_access:
        permissions = _merge_service_access(permissions, service_access)

    if "role_assignments" in authorization:
        assigned_roles = _normalize_roles(authorization.get("role_assignments"))
    else:
        assigned_roles = _normalize_roles(profile.get("assigned_roles"))

    normalized = dict(profile)
    normalized.update(
        {
            "user": email,
            "full_name": _first_text(
                raw_user.get("full_name"), profile.get("full_name"), email
            ),
            "user_image": _first_text(raw_user.get("image"), profile.get("user_image")),
            "is_guest": bool(profile.get("is_guest", False)),
            "company": _first_text(
                profile.get("company"),
                profile.get("selected_company_id"),
                authorization.get("company_id"),
            )
            or None,
            "role_type": _first_text(
                authorization.get("role_type"), profile.get("role_type")
            )
            or None,
            "permissions": permissions,
            "service_access": service_access,
            "assigned_roles": assigned_roles,
            "employee": _normalize_employee(profile.get("employee")),
            "companies": _normalize_companies(profile.get("companies")),
        }
    )
    return normalized
