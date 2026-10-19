"""권한 요구사항 자료형.

도구 하나가 "어느 서비스에 속하고 어떤 권한이 있어야 호출 가능한가"를
선언하는 불변 자료구조(``PermissionRequirement``)와, 게이트 평가 결과를
담는 ``GateDecision`` / ``DenyReason`` 을 정의한다.

``PermissionRequirement`` 는 도구 등록 시점에 생성되어 ``PermissionIndex``에
저장된다 (등록 경로 3가지는 ``app.agents.permissions.index`` docstring 참조).

게이트 평가 흐름은 ``app.agents.permissions.gate`` 참조.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class GateMode(str, Enum):
    """권한 평가 모드.

    Attributes:
        PUBLIC: 서비스 매핑 없는 범용 도구용. 서비스 사용권 검사를 건너뛰고 로그인
            사용자면 통과 (게스트는 ``allow_guest``로 별도 토글). calculator,
            search 같은 권한 무관 도구가 사용.
        ACCESS_ONLY: ``permissions[svc]``에 ``"{svc}.access"`` 포함만 요구. 동적
            도구의 default 정책이며, 세분 권한 없이도 서비스 사용권만 보면 충분한 경우.
        PERMISSION: ``permissions[svc]``에 명시 권한 키가 포함되어야 함. 일반적인
            logistics/mes/hr 도구가 사용.
        ROLE: ``assigned_roles``에 매칭 role 문자열이 있어야 함. "회사 관리자" 같은
            서비스 횡단 role 게이트.
        SERVICE_ADMIN: ``permissions[svc]``에 ``"{svc}.admin"`` 포함 필요. 서비스
            설정 변경 류 도구.
    """

    PUBLIC = "public"
    ACCESS_ONLY = "access_only"
    PERMISSION = "permission"
    ROLE = "role"
    SERVICE_ADMIN = "service_admin"


@dataclass(frozen=True)
class PermissionRequirement:
    """도구 한 개의 권한 요구사항.

    Attributes:
        service: 정규화된 서비스 키 (``permissions`` dict의 키와 일치).
            ``mattermost`` 같은 prefix는 ``SERVICE_ALIAS``로 사전 정규화한다.
            PUBLIC 모드에서는 무의미하므로 빈 문자열 권장.
        mode: 평가 방식.
        permissions_all: PERMISSION 모드에서 AND 조건으로 요구할 권한 키들.
        permissions_any: PERMISSION 모드에서 OR 조건으로 요구할 권한 키들.
            all/any 모두 비어 있으면 ``"{svc}.access"`` 통과만으로 OK
            (= ACCESS_ONLY와 동치).
        roles_any: ROLE 모드에서 OR 조건으로 매칭할 role 문자열들.
        allow_guest: ``is_guest=True`` 사용자에게도 노출할지 여부. 기본 False.
        services_any: 서비스 횡단 OR 1차 관문 (ACCESS_ONLY 전용, 2026-08-26).
            비어 있지 않으면 이 목록의 **어느 한 서비스라도** ``"{svc}.access"`` 가
            있으면 1차 관문 통과 (예: 물류 MCP 공통 도구 ``("logistics", "mes")``
            — 생산·재고 조회는 물류 사무실과 MES 현장이 함께 쓴다).
            ``service`` 필드는 그대로 유지한다 — 검색 도메인 도출
            (registry_loader._domain_of)·로그 진단용 주 서비스.

            **ACCESS_ONLY 한정** (`__post_init__` 에서 강제): PERMISSION/
            SERVICE_ADMIN 의 세분 권한 키는 ``permissions[svc]`` 네임스페이스에
            귀속되므로 서비스가 복수면 키 탐색 버킷·조합 의미가 모호해지고
            (fail-safe 원칙과 충돌), ``{svc}.admin`` 횡단은 권한 상승 통로가
            된다. ROLE 은 원래 서비스 무관이라 필요 없음.
            근거 상세: logistics-mcp/docs/tool-exposure-plan.md §3·§6.
        allow_no_profile: dmp 권한 프로필이 없는 사용자에게도 노출할지 여부(PUBLIC 모드에서만
            유효). 기본 False.

            왜 필요한가: PUBLIC 의 정의는 "로그인 사용자면 통과"인데, 게이트는 프로필이
            없으면 로그인 여부와 무관하게 막는다. SCMS 처럼 dmp 권한 체계 **밖**에 있는
            호스트는 게이트웨이가 검증한 sub 만 갖고 오고 프로필은 dmp 앱이 등록하므로,
            그 사용자에게는 계산기 같은 권한 무관 도구까지 전부 막힌다(채널이 도구 0개).

            프로필을 SCMS 가 대신 등록하는 길은 쓰지 않는다 — 프로필은 sub 단위 **전체
            대체**라 같은 사용자의 dmp 프로필(권한 포함)을 덮어써 두 앱이 서로 밀어낸다.
            그래서 "이 도구는 프로필 없이도 된다"를 도구가 스스로 선언한다.
    """

    service: str
    mode: GateMode = GateMode.PERMISSION
    permissions_all: tuple[str, ...] = ()
    permissions_any: tuple[str, ...] = ()
    roles_any: tuple[str, ...] = ()
    services_any: tuple[str, ...] = ()
    allow_guest: bool = False
    allow_no_profile: bool = False

    def __post_init__(self) -> None:
        # services_any 는 ACCESS_ONLY 전용 — 다른 모드와 조합되면 등록 시점에
        # 즉시 실패시킨다(잘못된 조합이 게이트에서 조용히 이상 동작하는 것 방지).
        if self.services_any and self.mode != GateMode.ACCESS_ONLY:
            raise ValueError(
                f"services_any 는 ACCESS_ONLY 모드에서만 쓸 수 있다 (mode={self.mode.value})"
            )

    @classmethod
    def from_dict(cls, spec: dict) -> "PermissionRequirement":
        """dict 형태(JSON 또는 LLM Tool 메타) → PermissionRequirement.

        ``service`` 필드는 필수. 잘못된 mode 문자열은 ValueError 전파.
        """
        return cls(
            service=spec["service"],
            mode=GateMode(spec.get("mode", "permission")),
            permissions_all=tuple(spec.get("permissions_all", ())),
            permissions_any=tuple(spec.get("permissions_any", ())),
            roles_any=tuple(spec.get("roles_any", ())),
            services_any=tuple(spec.get("services_any", ())),
            allow_guest=bool(spec.get("allow_guest", False)),
            allow_no_profile=bool(spec.get("allow_no_profile", False)),
        )


class DenyReason(str, Enum):
    """게이트 거부 사유.

    권한 유무와 도구 유무를 호출자가 구별할 수 있게 하는 것이 목적이다.
    ``is_allowed`` 가 bool 하나만 돌려주면 사유들이 모두 False로 동일해져,
    권한 문제임에도 기능이 없다고 안내됐다.

    ``GateMode`` 는 5종이지만 현재 등록된 도구가 쓰는 모드는 ``ACCESS_ONLY``/``PUBLIC`` 둘뿐이라,
    PERMISSION/ROLE/SERVICE_ADMIN 미충족은 ``MODE_NOT_SATISFIED`` 하나로 묶었다 
    그 모드를 쓰는 도구가 실제로 생기면 그때 분할한다.

    Attributes:
        NO_META: 권한 메타 미등록. **버그 신호** — 도구가 아무에게도 노출되지
            않는데 사용자에겐 "기능 없음"으로 보인다. gate 에서 WARNING 로깅.
        NO_PROFILE: 미인증 또는 profile fetch 실패 (profile=None).
        GUEST_BLOCKED: 게스트 사용자 + ``allow_guest=False``.
        NO_SERVICE_ACCESS: ``"{svc}.access"`` 없음 = 서비스 사용권 자체가 없음.
            실사용 거부 사유의 거의 전부.
        MODE_NOT_SATISFIED: mode별 세분 조건(권한 키/role/admin) 미충족.
        UNATTENDED: 예약 실행(무인) 중이라 승인 필요 도구를 막았다. 권한 문제가 아니다.
    """

    NO_META = "no_meta"
    NO_PROFILE = "no_profile"
    GUEST_BLOCKED = "guest_blocked"
    NO_SERVICE_ACCESS = "no_service_access"
    MODE_NOT_SATISFIED = "mode_not_satisfied"
    UNATTENDED = "unattended"


# 사용자에게 "권한이 없다"로 안내해야 하는 사유. 나머지는 안내 문구가 다르다 —
# NO_META 는 개발 문제(권한을 줘도 안 풀림), NO_PROFILE/GUEST_BLOCKED 는 로그인 문제.
_PERMISSION_REASONS = frozenset(
    {DenyReason.NO_SERVICE_ACCESS, DenyReason.MODE_NOT_SATISFIED}
)


@dataclass(frozen=True)
class GateDecision:
    """게이트 평가 결과 — 허용 여부 + 거부 사유.

    Attributes:
        allowed: True면 LLM에 bind 허용.
        reason: 거부 사유. ``allowed=True`` 면 None.
        service: 평가에 쓰인 서비스 키. **로그 진단용**이며 LLM 응답에는 넣지
            않는다. 메타 미등록(NO_META)이면 빈 문자열.
    """

    allowed: bool
    reason: Optional[DenyReason] = None
    service: str = ""

    @property
    def is_permission_issue(self) -> bool:
        """사용자에게 '권한 없음'으로 안내해야 하는 거부인가.

        True일 때만 "담당 관리자에게 권한 부여를 요청" 안내.
        """
        return self.reason in _PERMISSION_REASONS
