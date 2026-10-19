"""레거시 Frappe 및 REST v1 사용자 프로필 정규화 테스트."""
from __future__ import annotations

from copy import deepcopy

import pytest

from agent_arch.services.user_profile.normalizer import (
    InvalidUserProfileError,
    normalize_user_profile,
)


def _rest_profile(**overrides) -> dict:
    profile = {
        "user": {
            "id": "user@example.com",
            "email": "user@example.com",
            "full_name": "박정훈",
            "image": "/files/profile.png",
            "enabled": True,
        },
        "companies": [
            {
                "id": "ACME",
                "name": "회사",
                "company_group_id": "ACME-GROUP",
            }
        ],
        "selected_company_id": "ACME",
        "selection_persisted": True,
        "employee": {
            "id": "EMP-001",
            "name": "박정훈",
            "company_id": "ACME",
            "department": {
                "id": "DEPT-DEV",
                "name": "개발팀",
                "parent_id": "",
                "parent_name": "",
            },
            "designation": {"id": "DES-01", "name": "매니저"},
            "position": {"id": "POS-01", "name": "팀원"},
            "employment_type": {"id": "FULL_TIME", "is_regular": True},
            "contacts": {
                "work_email": "user@example.com",
                "work_phone": "02-1234-5678",
                "extension": "123",
                "work_fax": "",
                "mobile_no": "010-0000-0000",
                "alternative_email": "personal@example.com",
            },
            "work_address": {"address_1": "서울", "address_2": "강남구"},
            "personal": {
                "birth_date": "1990-01-01",
                "address": "서울",
                "bio": "소개",
            },
            "date_of_joining": "2024-01-02",
        },
        "authorization": {
            "company_id": "ACME",
            "role_type": "EMPLOYEE",
            "permissions": {
                "eapproval": ["eapproval.access"],
                "talk": ["talk.access"],
            },
            "service_access": [
                {
                    "service": "eapproval",
                    "has_access": True,
                    "is_service_admin": False,
                }
            ],
            "role_assignments": [
                {"id": "ROLE-1", "label": "전자결재 담당자"},
            ],
        },
        # dmp-frontend가 선택 회사를 명시적으로 덧붙인다.
        "company": "ACME",
        "is_guest": False,
    }
    profile.update(overrides)
    return profile


class TestNormalizeUserProfile:
    def test_legacy_profile_is_preserved(self):
        profile = {
            "user": " user@example.com ",
            "full_name": "테스트",
            "permissions": {"eapproval": ["eapproval.access"]},
        }

        normalized = normalize_user_profile(profile)

        assert normalized == {
            "user": "user@example.com",
            "full_name": "테스트",
            "permissions": {"eapproval": ["eapproval.access"]},
        }
        assert profile["user"] == " user@example.com "

    def test_rest_v1_profile_is_flattened_for_datv_consumers(self):
        profile = _rest_profile()
        original = deepcopy(profile)

        normalized = normalize_user_profile(profile)

        assert normalized["user"] == "user@example.com"
        assert normalized["full_name"] == "박정훈"
        assert normalized["user_image"] == "/files/profile.png"
        assert normalized["company"] == "ACME"
        assert normalized["role_type"] == "EMPLOYEE"
        assert normalized["permissions"] == {
            "eapproval": ["eapproval.access"],
            "talk": ["talk.access"],
        }
        assert normalized["service_access"][0]["service"] == "eapproval"
        assert normalized["assigned_roles"] == ["전자결재 담당자"]
        assert normalized["authorization"] == profile["authorization"]
        assert normalized["selection_persisted"] is True
        assert profile == original

    def test_rest_v1_employee_is_flattened(self):
        employee = normalize_user_profile(_rest_profile())["employee"]

        assert employee["id"] == "EMP-001"
        assert employee["department"] == "DEPT-DEV"
        assert employee["departmentName"] == "개발팀"
        assert employee["designation"] == "DES-01"
        assert employee["designationName"] == "매니저"
        assert employee["position"] == "POS-01"
        assert employee["positionName"] == "팀원"
        assert employee["employmentType"] == "FULL_TIME"
        assert employee["isRegular"] is True
        assert employee["workEmail"] == "user@example.com"
        assert employee["workAddressDetail"] == "강남구"
        assert employee["joinDate"] == "2024-01-02"

    def test_rest_v1_company_alias_is_added(self):
        company = normalize_user_profile(_rest_profile())["companies"][0]
        assert company["company_group_id"] == "ACME-GROUP"
        assert company["company_group"] == "ACME-GROUP"

    def test_rest_v1_falls_back_to_user_id(self):
        profile = _rest_profile(
            user={"id": "user@example.com", "email": "", "full_name": "Fallback"}
        )
        assert normalize_user_profile(profile)["user"] == "user@example.com"

    def test_rest_v1_access_only_service_is_merged_into_permissions(self):
        # calendar·hr처럼 세분 권한 없이 사용권(service_access)만 오는 서비스가
        # 게이트 키({svc}.access/{svc}.admin)로 합성되는지 — 2026-07-31 QA 회귀.
        profile = _rest_profile()
        profile["authorization"] = deepcopy(profile["authorization"])
        profile["authorization"]["service_access"] = [
            {"service": "eapproval", "has_access": True, "is_service_admin": False},
            {"service": "calendar", "has_access": True, "is_service_admin": True},
            {"service": "hr", "has_access": False, "is_service_admin": True},
            {"service": "", "has_access": True},
            "garbage",
        ]
        original = deepcopy(profile)

        permissions = normalize_user_profile(profile)["permissions"]

        # 사용권만 있는 서비스: access 합성, 서비스 관리자면 admin까지
        assert permissions["calendar"] == ["calendar.access", "calendar.admin"]
        # has_access=False면 admin이 True여도 합성하지 않는다
        assert "hr" not in permissions
        # 기존 세분 권한 키는 중복 없이 보존
        assert permissions["eapproval"] == ["eapproval.access"]
        assert permissions["talk"] == ["talk.access"]
        # 입력 불변
        assert profile == original

    def test_rest_v1_service_access_fills_missing_permissions(self):
        # authorization.permissions가 아예 없어도 사용권만으로 게이트 키가 생긴다.
        profile = _rest_profile()
        profile["authorization"] = deepcopy(profile["authorization"])
        profile["authorization"].pop("permissions")
        profile["authorization"]["service_access"] = [
            {"service": "drive", "has_access": True, "is_service_admin": False},
        ]

        permissions = normalize_user_profile(profile)["permissions"]

        assert permissions == {"drive": ["drive.access"]}

    def test_flattened_profile_renormalization_merges_service_access(self):
        # 이미 평탄화된(user=문자열) v1 산출 프로필이 재정규화되는 경로에서도
        # service_access 사용권이 합성된다.
        profile = {
            "user": "user@example.com",
            "permissions": {"mail": ["mail.access"]},
            "service_access": [
                {"service": "drive", "has_access": True, "is_service_admin": False},
            ],
        }

        normalized = normalize_user_profile(profile)

        assert normalized["permissions"] == {
            "mail": ["mail.access"],
            "drive": ["drive.access"],
        }

    @pytest.mark.parametrize(
        "profile",
        [
            {},
            {"user": ""},
            {"user": None},
            {"user": {}},
            {"user": {"id": "", "email": ""}},
        ],
    )
    def test_missing_user_identity_is_rejected(self, profile):
        with pytest.raises(InvalidUserProfileError):
            normalize_user_profile(profile)
