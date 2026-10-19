"""BaseTool.execute() 테스트.

@traceable 데코레이터 적용 여부, _real_execute 위임,
ToolResult 반환 구조를 검증한다.
"""
import pytest
from agent_arch.agents.tools.base import BaseTool, ToolResult


class _DummyTool(BaseTool):
    """테스트용 최소 구현 도구."""

    def get_name(self) -> str:
        return "dummy_tool"

    def get_description(self) -> str:
        return "테스트용 도구"

    def get_parameters(self):
        return {"query": {"type": "string"}}

    def get_required_params(self):
        return ["query"]

    async def _real_execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="real_result")


class TestBaseToolTraceable:
    """@traceable 적용 여부 검증."""

    def test_execute_is_wrapped(self):
        """`execute`는 @traceable 래핑으로 __wrapped__ 속성을 갖는다."""
        assert hasattr(BaseTool.execute, "__wrapped__"), (
            "execute()에 @traceable이 적용되지 않았습니다."
        )

    def test_execute_name_preserved(self):
        """래핑 후에도 함수 이름이 보존된다."""
        assert BaseTool.execute.__name__ == "execute"


class TestBaseToolExecuteRouting:
    """execute() 는 _real_execute() 로 위임한다."""

    @pytest.mark.asyncio
    async def test_execute_calls_real_execute(self):
        result = await _DummyTool().execute(query="테스트")

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.output == "real_result"

    @pytest.mark.asyncio
    async def test_real_execute_default_not_implemented(self):
        """_real_execute()를 재정의하지 않으면 success=False를 반환한다."""

        class _MinimalTool(BaseTool):
            def get_name(self): return "minimal"
            def get_description(self): return "최소 도구"
            def get_parameters(self): return {}
            def get_required_params(self): return []
            # _real_execute는 재정의하지 않음 (BaseTool 기본 구현 사용)

        result = await _MinimalTool().execute()

        assert result.success is False
        assert result.error is not None


class TestToolResult:
    """ToolResult 모델 검증."""

    def test_success_result(self):
        r = ToolResult(success=True, output={"key": "value"})
        assert r.success is True
        assert r.error is None

    def test_error_result(self):
        r = ToolResult(success=False, output=None, error="API 오류")
        assert r.success is False
        assert r.error == "API 오류"

    def test_metadata_optional(self):
        r = ToolResult(success=True, output="ok", metadata={"elapsed": 0.5})
        assert r.metadata == {"elapsed": 0.5}
