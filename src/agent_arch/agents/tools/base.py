from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, ClassVar
from pydantic import BaseModel
from langsmith import traceable
from agent_arch.agents.permissions.spec import PermissionRequirement


class ToolResult(BaseModel):
    """도구 실행 결과 모델.

    Attributes:
        success: 실행 성공 여부
        output: 실행 결과 데이터
        error: 오류 메시지 (실패 시)
        metadata: 추가 메타데이터
    """

    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """에이전트 도구 기본 인터페이스.

    모든 도구는 이 클래스를 상속하여 구현한다.

    Attributes:
        permission_required: 권한 게이트 메타. 도구가 호출 가능한지 평가하는
            기준이다. None이면 ``@register_tool`` 가 PermissionIndex에 등록하지
            않으므로 게이트가 deny default를 적용한다(fail-safe).
            동적 도구(DynamicAPITool)는 클래스 속성이 아니라 인스턴스마다
            ``llm_tools.json.required_permissions`` 에서 가져와 register한다.
        config: 도구 설정 딕셔너리
        name: 도구 이름
        description: 도구 설명
        parameters: 도구 파라미터 스키마
        required_params: 필수 파라미터 리스트
    """

    permission_required: ClassVar[Optional[PermissionRequirement]] = None

    # 노출 티어 (문자열 — Enum 정의는 app.agents.tools.exposure, 순환 import 회피).
    # "core"=상시 바인딩(기본), "retrieved"=질의 검색 시만. tools_config.yaml의
    # exposure 키가 이 값을 오버라이드한다. 해석은 exposure.resolve_exposure().
    exposure: ClassVar[str] = "core"

    # 이 도구를 "언제·어떻게 써야 하는가"를 LLM 에 알리는 시스템 프롬프트 규칙 문단.
    # 비어 있지 않고 도구가 enabled + core 로 바인딩되면, build_system_prompt() 가
    # 프롬프트에 자동 포함한다(도구가 꺼지면 규칙도 함께 사라짐). 개별 사용법이 아니라
    # "반드시 이 도구로 처리하라" 같은 전역 규칙에만 쓴다. 렌더는 app.agents.prompts.system.
    prompt_guidance: ClassVar[str] = ""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """BaseTool 인스턴스를 초기화한다.

        Args:
            config (Optional[Dict[str, Any]]): 도구 설정 딕셔너리
        """
        self.config = config or {}
        self.name = self.get_name()
        self.description = self.get_description()
        self.parameters = self.get_parameters()
        self.required_params = self.get_required_params()

    @abstractmethod
    def get_name(self) -> str:
        """도구 이름을 반환한다.

        Returns:
            str: 도구 이름 문자열
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """도구 설명을 반환한다.

        Returns:
            str: 도구 설명 문자열
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """도구 파라미터 스키마를 반환한다 (JSON Schema 형식).

        Returns:
            Dict[str, Any]: 파라미터 스키마 딕셔너리.

        Example:
            {
                "query": {
                    "type": "string",
                    "description": "검색 쿼리"
                },
                "limit": {
                    "type": "integer",
                    "description": "결과 개수",
                    "default": 5
                }
            }
        """
        pass

    @abstractmethod
    def get_required_params(self) -> List[str]:
        """필수 파라미터 리스트를 반환한다.

        Returns:
            List[str]: 필수 파라미터 이름 리스트
        """
        pass

    @traceable(run_type="tool")
    async def execute(self, **kwargs) -> ToolResult:
        """도구를 실행한다.

        Args:
            **kwargs: 도구 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        return await self._real_execute(**kwargs)

    async def _real_execute(self, **kwargs) -> ToolResult:
        """실제 모드 실행 - 실제 외부 API를 호출한다.

        하위 클래스에서 실제 API 연동이 준비되면 이 메서드를 재정의한다.

        Args:
            **kwargs: 도구 파라미터

        Returns:
            ToolResult: 실제 실행 결과
        """
        return ToolResult(
            success=False,
            output=None,
            error=f"'{self.name}' 도구의 실제 API 구현이 아직 완료되지 않았습니다."
        )

    async def __call__(self, **kwargs) -> ToolResult:
        """도구를 호출한다 (execute의 별칭).

        Args:
            **kwargs: 도구 파라미터

        Returns:
            ToolResult: 실행 결과
        """
        return await self.execute(**kwargs)

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """파라미터 유효성을 검증한다.

        Args:
            params (Dict[str, Any]): 검증할 파라미터 딕셔너리

        Returns:
            bool: 유효성 검증 결과

        Raises:
            ValueError: 필수 파라미터가 누락된 경우
        """
        for required in self.required_params:
            if required not in params:
                raise ValueError(f"Missing required parameter: {required}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """도구 정보를 딕셔너리로 변환한다.

        Returns:
            Dict[str, Any]: 도구 정보 딕셔너리
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "required": self.required_params
        }

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환한다.

        Returns:
            str: 클래스 이름과 도구 이름을 포함한 문자열
        """
        return f"{self.__class__.__name__}(name={self.name})"
