from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from agent_arch.llm.base import BaseLLM
from agent_arch.agents.tools.base import BaseTool


class BaseAgent(ABC):
    """에이전트 기본 클래스.

    모든 에이전트는 이 클래스를 상속한다.

    Attributes:
        llm: LLM 인스턴스
        tools: 등록된 도구 리스트
        config: 에이전트 설정 딕셔너리
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[BaseTool]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """BaseAgent 인스턴스를 초기화한다.

        Args:
            llm (BaseLLM): LLM 인스턴스
            tools (Optional[List[BaseTool]]): 도구 리스트 (선택사항)
            config (Optional[Dict[str, Any]]): 에이전트 설정 딕셔너리 (선택사항)
        """
        self.llm = llm
        self.tools = tools or []
        self.config = config or {}

        # 도구를 LLM에 바인딩
        if self.tools:
            self.llm.bind_tools(self.tools)

    @abstractmethod
    async def run(self, input: str, **kwargs) -> Dict[str, Any]:
        """에이전트를 실행한다.

        Args:
            input (str): 사용자 입력
            **kwargs: 추가 파라미터

        Returns:
            Dict[str, Any]: 실행 결과
        """
        pass

    async def stream(self, input: str, **kwargs):
        """스트리밍 방식으로 실행한다 (선택적 구현).

        Args:
            input (str): 사용자 입력
            **kwargs: 추가 파라미터

        Yields:
            Any: 중간 결과

        Raises:
            NotImplementedError: 스트리밍을 지원하지 않는 경우
        """
        raise NotImplementedError("Streaming not supported for this agent")

    def get_tools(self) -> List[BaseTool]:
        """등록된 도구 리스트를 반환한다.

        Returns:
            List[BaseTool]: 등록된 도구 리스트
        """
        return self.tools

    def add_tool(self, tool: BaseTool):
        """도구를 추가한다.

        Args:
            tool (BaseTool): 추가할 도구
        """
        self.tools.append(tool)
        self.llm.bind_tools(self.tools)

    def __repr__(self) -> str:
        """객체의 문자열 표현을 반환한다.

        Returns:
            str: 클래스 이름, LLM 정보, 도구 개수를 포함한 문자열
        """
        return f"{self.__class__.__name__}(llm={self.llm}, tools={len(self.tools)})"
