"""권한 게이트 패키지.

도구 호출 가능 여부를, 도구 등록 시점에 채워진 ``PermissionIndex``(tool_name →
``PermissionRequirement``)를 통해 평가한다. 인덱스를 채우는 경로는
``app.agents.permissions.index`` docstring 참조.

``evaluate()`` 는 거부 사유(``DenyReason``)까지 돌려주므로 호출자가 "권한이
없다"와 "그런 도구가 없다"를 구별할 수 있다. bool만 필요하면 ``is_allowed()``.
자세한 흐름은 각 모듈 docstring 참조.
"""
from agent_arch.agents.permissions.gate import evaluate, is_allowed
from agent_arch.agents.permissions.index import permission_index
from agent_arch.agents.permissions.services import SERVICE_ALIAS, normalize_service
from agent_arch.agents.permissions.spec import (
    DenyReason,
    GateDecision,
    GateMode,
    PermissionRequirement,
)

__all__ = [
    "DenyReason",
    "GateDecision",
    "GateMode",
    "PermissionRequirement",
    "SERVICE_ALIAS",
    "evaluate",
    "is_allowed",
    "normalize_service",
    "permission_index",
]
