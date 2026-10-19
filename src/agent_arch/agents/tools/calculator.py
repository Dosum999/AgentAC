from typing import Dict, Any, List
from agent_arch.agents.permissions import GateMode, PermissionRequirement
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.agents.tools.registry import register_tool


@register_tool
class CalculatorTool(BaseTool):
    """수학 계산 도구.

    기본 산술 연산과 수학 표현식을 지원하는 계산기 도구입니다.
    외부 API 의존성이 없어 데모/실제 모드 모두 동일하게 동작합니다.
    """

    # 권한 무관 순수 계산 — 프로필 없는 호스트(SCMS 채널)에서도 노출.
    permission_required = PermissionRequirement(
        service="", mode=GateMode.PUBLIC, allow_no_profile=True
    )

    def get_name(self) -> str:
        """도구 이름을 반환한다.

        Returns:
            str: 도구 이름 'calculator'
        """
        return "calculator"

    def get_description(self) -> str:
        """도구 설명을 반환한다.

        Returns:
            str: 도구에 대한 영문 설명
        """
        return "Perform mathematical calculations. Supports basic arithmetic and mathematical expressions."

    def get_parameters(self) -> Dict[str, Any]:
        """도구 파라미터 스키마를 반환한다 (JSON Schema 형식).

        Returns:
            Dict[str, Any]: 파라미터 스키마 딕셔너리
                - expression (str): 계산할 수학 표현식
        """
        return {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2', '10 * 5 / 2')"
            }
        }

    def get_required_params(self) -> List[str]:
        """필수 파라미터 리스트를 반환한다.

        Returns:
            List[str]: 필수 파라미터 이름 리스트
        """
        return ["expression"]

    async def _demo_execute(self, **kwargs) -> ToolResult:
        """데모 모드 계산 실행 - 실제와 동일한 수식 계산을 수행한다.

        계산기는 외부 API 의존성이 없으므로 데모/실제 모드 모두 동일하게 동작한다.

        Args:
            **kwargs: 도구 파라미터 (expression)

        Returns:
            ToolResult: 계산 결과
        """
        return await self._calculate(**kwargs)

    async def _real_execute(self, **kwargs) -> ToolResult:
        """실제 모드 계산 실행 - 실제와 동일한 수식 계산을 수행한다.

        Args:
            **kwargs: 도구 파라미터 (expression)

        Returns:
            ToolResult: 계산 결과
        """
        return await self._calculate(**kwargs)

    async def _calculate(self, **kwargs) -> ToolResult:
        """수식을 계산한다.

        안전한 계산을 위해 제한된 네임스페이스에서 eval을 실행합니다.

        Args:
            **kwargs: 도구 파라미터 (expression: 수학 표현식)

        Returns:
            ToolResult: 계산 결과 또는 오류 정보

        Raises:
            ZeroDivisionError: 0으로 나누기 시 발생
            Exception: 기타 계산 오류 시 발생
        """
        try:
            self.validate_params(kwargs)

            expression = kwargs["expression"]

            # 안전한 계산을 위해 제한된 네임스페이스 사용
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
            }

            # eval 대신 더 안전한 방법 사용 (간단한 구현)
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            return ToolResult(
                success=True,
                output=str(result),
                metadata={
                    "expression": expression,
                    "result_type": type(result).__name__
                }
            )

        except ZeroDivisionError:
            return ToolResult(
                success=False,
                output=None,
                error="Division by zero"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Calculation error: {str(e)}"
            )
