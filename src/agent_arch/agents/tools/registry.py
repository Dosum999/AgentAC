from typing import Dict, List, Optional, Any
from agent_arch.agents.tools.base import BaseTool
from agent_arch.core.config import settings


def _invalidate_sanitize_whitelist() -> None:
    """도구 목록이 바뀌었음을 응답 정화기(sanitize)에 알린다.

    ``sanitize`` 의 도구명 마스킹 화이트리스트는 lazy 빌드 후 프로세스 생애 동안 캐시된다.
    MCP 원격 도구는 부팅 후 백그라운드 태스크로 **뒤늦게** 등록되므로, 무효화가 없으면
    등록 전에 들어온 첫 요청이 MCP 없는 스냅샷을 굳혀 그 프로세스 내내 메일·일정 도구명이
    마스킹되지 않는다(2026-08 실제 누출 원인).

    import 를 함수 안에 두는 이유: ``sanitize._collect_tool_names()`` 가 거꾸로 이 모듈의
    ``ToolRegistry`` 를 읽는 **상호 의존**이다. 어느 한쪽이라도 모듈 레벨로 올리면 부팅 시
    ImportError 가 난다.
    """
    from agent_arch.utils.sanitize import reset_whitelist_cache

    reset_whitelist_cache()


class ToolRegistry:
    """도구 레지스트리.

    도구를 등록하고 관리한다. LLM 공급자별로 도구를 적절한 포맷으로 변환하여 제공한다.

    Attributes:
        _tools: 등록된 도구들의 딕셔너리
    """

    _tools: Dict[str, BaseTool] = {}

    @classmethod
    def register(cls, tool: BaseTool):
        """도구를 등록한다.

        Args:
            tool (BaseTool): 등록할 BaseTool 인스턴스

        Raises:
            ValueError: 동일한 이름의 도구가 이미 등록된 경우
        """
        tool_name = tool.name
        if tool_name in cls._tools:
            raise ValueError(f"Tool '{tool_name}' is already registered")

        cls._tools[tool_name] = tool
        _invalidate_sanitize_whitelist()

    @classmethod
    def unregister(cls, tool_name: str):
        """도구 등록을 해제한다.

        Args:
            tool_name (str): 등록 해제할 도구 이름
        """
        if tool_name in cls._tools:
            del cls._tools[tool_name]
            _invalidate_sanitize_whitelist()

    @classmethod
    def get(cls, tool_name: str) -> Optional[BaseTool]:
        """이름으로 도구를 조회한다.

        Args:
            tool_name (str): 조회할 도구 이름

        Returns:
            Optional[BaseTool]: 도구 객체. 없으면 None 반환
        """
        return cls._tools.get(tool_name)

    @classmethod
    def get_all(cls) -> Dict[str, BaseTool]:
        """등록된 모든 도구를 반환한다.

        Returns:
            Dict[str, BaseTool]: 모든 등록된 도구의 복사본 딕셔너리
        """
        return cls._tools.copy()

    @classmethod
    def get_enabled_tools(cls) -> List[BaseTool]:
        """활성화된 도구만 반환한다.

        tools_config.yaml에서 enabled=true로 설정된 도구만 반환한다.

        Returns:
            List[BaseTool]: 활성화된 도구 리스트
        """
        enabled_tools = []

        for tool_name, tool in cls._tools.items():
            tool_config = settings.get_tool_config(tool_name)
            if tool_config.get("enabled", True):
                enabled_tools.append(tool)

        return enabled_tools

    @classmethod
    def get_tools_by_category(cls, category: str) -> List[BaseTool]:
        """카테고리별 도구를 조회한다.

        tools_config.yaml의 combinations에서 정의된 도구 조합을 반환한다.

        Args:
            category (str): 카테고리 이름 (예: research, data_analysis)

        Returns:
            List[BaseTool]: 해당 카테고리의 도구 리스트
        """
        combinations = settings.tools_config.get("combinations", {})
        tool_names = combinations.get(category, [])

        tools = []
        for name in tool_names:
            tool = cls.get(name)
            if tool:
                tools.append(tool)

        return tools

    @classmethod
    def get_tools_for_llm(
        cls,
        provider: str,
        tool_names: Optional[List[str]] = None
    ) -> List[BaseTool]:
        """특정 LLM 공급자용 도구 리스트를 반환한다.

        Args:
            provider (str): LLM 공급자 이름
            tool_names (Optional[List[str]]): 특정 도구만 선택. None이면 활성화된 모든 도구 반환.

        Returns:
            List[BaseTool]: 해당 공급자에서 사용 가능한 도구 리스트
        """
        if tool_names:
            tools = [cls.get(name) for name in tool_names if cls.get(name)]
        else:
            tools = cls.get_enabled_tools()

        # 공급자별 필터링 (필요시)
        # 예: 특정 공급자에서 지원하지 않는 도구 제외
        return tools

    @classmethod
    def list_tool_names(cls) -> List[str]:
        """등록된 모든 도구 이름 리스트를 반환한다.

        Returns:
            List[str]: 등록된 도구 이름 리스트
        """
        return list(cls._tools.keys())

    @classmethod
    def clear(cls):
        """모든 도구 등록을 해제한다 (주로 테스트용).

        Note:
            이 메서드는 주로 단위 테스트에서 사용되며,
            운영 환경에서는 사용하지 않는 것을 권장합니다.
        """
        cls._tools.clear()
        _invalidate_sanitize_whitelist()


# 도구 자동 등록을 위한 데코레이터
def register_tool(tool_class: type) -> type:
    """도구를 자동 등록하는 데코레이터.

    ``ToolRegistry`` 에 도구를 등록하고, 클래스에 ``permission_required`` 가
    선언돼 있으면 ``PermissionIndex`` 에도 함께 register한다. 정적 Skill은
    이 한 번의 import-time 호출로 권한 메타까지 자기서술적으로 등록된다.

    동적 도구(DynamicAPITool)는 인스턴스마다 ``llm_tools.json`` 의
    ``required_permissions`` 필드를 보고 직접 register하므로 이 경로를 안 탄다.

    Example:
        @register_tool
        class HrSkillRequestLeave(BaseTool):
            permission_required = PermissionRequirement(
                service="hr", mode=GateMode.PERMISSION,
                permissions_all=("hr.leave_request",),
            )

    Returns:
        type: 원본 클래스 (데코레이터 패턴)
    """
    tool_instance = tool_class()
    ToolRegistry.register(tool_instance)

    if tool_instance.permission_required is not None:
        from agent_arch.agents.permissions import permission_index
        permission_index.register(tool_instance.name, tool_instance.permission_required)

    return tool_class
