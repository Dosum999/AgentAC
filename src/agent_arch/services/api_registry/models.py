from typing import Any, Dict, List, Optional
from pydantic import BaseModel, model_validator


class ExternalToolDef(BaseModel):
    """llm_tools.json 한 항목의 파싱 모델.

    외부 서비스 API 도구 정의를 표현하는 Pydantic 모델입니다.
    JSON 파일에서 로드된 도구 정보를 타입 안전하게 다룰 수 있습니다.

    Attributes:
        tool_name: 도구 이름
        description: 도구 설명
        is_external: 외부 서비스 여부
        external_service: 외부 서비스 이름
        http_method: HTTP 메서드
        endpoint_path: 엔드포인트 경로
        parameters_schema: 파라미터 스키마
        enabled: 활성화 여부
        required_permissions: 권한 게이트 메타. 없으면 런타임에 ACCESS_ONLY
            fallback(permissions의 "{service}.access" 키만 검사). 자세한 스키마는
            ``app.agents.permissions.spec.PermissionRequirement.from_dict`` 참조.
    """

    tool_name: str
    description: str
    is_external: bool
    external_service: Optional[str] = None
    # origin="registry"(정적 스킬·MCP — 인덱스엔 이름+설명만, 실행은 ToolRegistry
    # 실물 인스턴스가 담당)는 http 필드가 없으므로 Optional. dynamic_api는 필수
    # (validator에서 강제 — 기존 안전성 유지).
    origin: str = "dynamic_api"         # "dynamic_api" | "registry"
    # 도메인 = 정규화된 권한 서비스 키 (logistics/hr/talk/mail/...). 검색·
    # tool_search의 도메인 필터 축이며 권한 게이트와 같은 키 체계를 쓴다.
    # 동적은 external_service 정규화, registry는 permission.service에서 도출.
    domain: Optional[str] = None
    http_method: Optional[str] = None   # GET/POST/PUT/DELETE/PATCH
    endpoint_path: Optional[str] = None  # "/attendance-advanced/logs/"
    parameters_schema: Dict[str, Any]  # JSON Schema (파싱된 상태)
    enabled: bool
    required_permissions: Optional[Dict[str, Any]] = None

    model_config = {"extra": "ignore"}

    @model_validator(mode="after")
    def require_http_fields_for_dynamic(self) -> "ExternalToolDef":
        """dynamic_api 항목은 실행 정보(http_method/endpoint_path)가 필수다."""
        if self.origin == "dynamic_api" and (not self.http_method or not self.endpoint_path):
            raise ValueError(
                f"dynamic_api 도구 '{self.tool_name}'에 http_method/endpoint_path 누락"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def coerce_booleans(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Frappe 스타일의 0/1 정수를 bool로 변환한다.

        JSON에서 숫자 0/1로 표현된 불리언 값을 Python bool 타입으로 변환합니다.

        Args:
            values (Dict[str, Any]): 원본 필드 값 딕셔너리

        Returns:
            Dict[str, Any]: 불리언 변환이 적용된 딕셔너리
        """
        for field in ("is_external", "enabled"):
            if field in values and not isinstance(values[field], bool):
                values[field] = bool(values[field])
        return values

    @property
    def properties(self) -> Dict[str, Any]:
        """파라미터 스키마의 properties 필드를 반환한다.

        Returns:
            Dict[str, Any]: 파라미터 속성 딕셔너리
        """
        return self.parameters_schema.get("properties", {})

    @property
    def required_params(self) -> List[str]:
        """파라미터 스키마의 required 필드를 반환한다.

        Returns:
            List[str]: 필수 파라미터 이름 리스트
        """
        return self.parameters_schema.get("required", [])
