"""agent_arch.agents.react.executor 단위 테스트.

find_tool, truncate_result, serialize_result, execute_tools의
핵심 동작과 경계 조건을 검증한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.react.executor import (
    execute_tools,
    find_tool,
    serialize_result,
    truncate_result,
)
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import ToolCall


# ── 테스트 헬퍼 ──────────────────────────────────────────────────────────────

def _make_tool(name: str, output="결과"):
    """이름과 실행 결과가 고정된 모의 도구를 생성한다."""
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(return_value=ToolResult(success=True, output=output))
    return tool


def _tc(name: str) -> ToolCall:
    """테스트용 ToolCall을 생성한다."""
    return ToolCall(id=f"id_{name}", name=name, arguments={})


# ── find_tool ─────────────────────────────────────────────────────────────────

class TestFindTool:
    """find_tool() 동작 검증."""

    def test_finds_tool_by_name(self):
        """이름이 일치하는 도구를 반환한다."""
        t1 = _make_tool("alpha")
        t2 = _make_tool("beta")
        assert find_tool("beta", [t1, t2]) is t2

    def test_returns_none_when_not_found(self):
        """이름이 없으면 None을 반환한다."""
        assert find_tool("ghost", [_make_tool("alpha")]) is None

    def test_returns_none_for_empty_list(self):
        """도구 리스트가 비어 있으면 None을 반환한다."""
        assert find_tool("alpha", []) is None


# ── truncate_result ───────────────────────────────────────────────────────────

class TestTruncateResult:
    """truncate_result() 동작 검증."""

    def test_list_over_max_is_sliced(self):
        """max_items 초과 리스트는 앞에서 잘리고 _truncated 메타 필드가 추가된다."""
        data = list(range(25))
        result = truncate_result(data, max_items=20)
        assert result["items"] == list(range(20))
        assert "20건만 표시" in result["_truncated"]

    def test_list_within_max_is_unchanged(self):
        """max_items 이하 리스트는 그대로 반환된다."""
        data = [1, 2, 3]
        assert truncate_result(data, max_items=10) == [1, 2, 3]

    def test_dict_with_long_list_value_is_truncated(self):
        """dict 값 중 max_items 초과 리스트는 잘리고 _truncated_{key} 필드가 추가된다."""
        data = {"items": list(range(30)), "meta": "ok"}
        result = truncate_result(data, max_items=20)
        assert len(result["items"]) == 20
        assert "_truncated_items" in result
        assert result["meta"] == "ok"

    def test_dict_with_short_list_value_unchanged(self):
        """dict 값의 리스트가 max_items 이하이면 변경되지 않는다."""
        data = {"items": [1, 2, 3]}
        assert truncate_result(data, max_items=10) == {"items": [1, 2, 3]}

    def test_scalar_passthrough(self):
        """list/dict가 아닌 값은 그대로 반환된다."""
        assert truncate_result("문자열") == "문자열"
        assert truncate_result(42) == 42
        assert truncate_result(None) is None


# ── serialize_result ──────────────────────────────────────────────────────────

class TestSerializeResult:
    """serialize_result() 동작 검증."""

    def test_success_serialized_as_json(self):
        """성공 결과는 JSON 문자열로 직렬화된다."""
        result = ToolResult(success=True, output={"key": "값"})
        serialized = serialize_result(result)
        assert '"key"' in serialized
        assert '"값"' in serialized

    def test_failed_result_prefixed_with_error(self):
        """실패 결과는 [ERROR] 접두어와 함께 에러 메시지가 반환된다."""
        result = ToolResult(success=False, output=None, error="API 오류")
        serialized = serialize_result(result)
        assert serialized.startswith("[ERROR]")
        assert "API 오류" in serialized

    def test_failed_result_keeps_its_data(self):
        """실패에 실린 데이터(동명이인 후보)도 모델에 간다 — 없으면 모델이 목록을 지어낸다(2026-10-01)."""
        result = ToolResult(success=False, error="3명입니다",
                            output={"ambiguous": True, "candidates": [{"name": "김가나"}]})
        serialized = serialize_result(result)
        assert serialized.startswith("[ERROR] 3명입니다")
        assert '"candidates"' in serialized and "김가나" in serialized

    def test_failed_result_does_not_echo_error_only_payload(self):
        """에러 문구를 되풀이할 뿐인 output 은 덧붙이지 않는다(budget 차단 메시지 등)."""
        result = ToolResult(success=False, error="E: m", output={"error": "E", "message": "m"})
        assert serialize_result(result) == "[ERROR] E: m"

    def test_success_output_is_truncated_before_serializing(self):
        """성공 결과의 리스트 output은 직렬화 전에 잘린다."""
        result = ToolResult(success=True, output=list(range(30)))
        serialized = serialize_result(result)
        # truncate_result가 적용되어 "items"와 "_truncated" 키가 있어야 한다
        assert "_truncated" in serialized


# ── execute_tools ─────────────────────────────────────────────────────────────

class TestExecuteTools:
    """execute_tools() 동작 검증."""

    @pytest.mark.asyncio
    async def test_returns_tc_content_tuples(self):
        """각 tool_call에 대해 (ToolCall, result_content) 튜플을 반환한다."""
        tool_a = _make_tool("tool_a", output="a 결과")
        results = await execute_tools([_tc("tool_a")], [tool_a])
        assert len(results) == 1
        tc, content = results[0]
        assert tc.name == "tool_a"
        assert "a 결과" in content

    @pytest.mark.asyncio
    async def test_missing_tool_returns_error_string(self):
        """도구를 찾지 못하면 [ERROR] 문자열이 결과로 반환된다."""
        results = await execute_tools([_tc("ghost_tool")], [])
        _, content = results[0]
        assert "[ERROR]" in content
        assert "ghost_tool" in content

    @pytest.mark.asyncio
    async def test_multiple_tools_executed_and_order_preserved(self):
        """여러 tool_call을 병렬 실행하고 입력 순서 그대로 결과를 반환한다."""
        tools = [_make_tool("t1", "r1"), _make_tool("t2", "r2")]
        results = await execute_tools([_tc("t1"), _tc("t2")], tools)
        assert results[0][0].name == "t1"
        assert results[1][0].name == "t2"


# ── 턴 예산(react/budget) — 실행기에서 중복·상한을 막는다 ─────────────────────────

class TestExecuteToolsBudget:
    @pytest.mark.asyncio
    async def test_duplicate_web_call_is_not_executed_again(self, monkeypatch):
        from agent_arch.core.config import settings
        monkeypatch.setattr(settings, "web_max_calls_per_turn", 8, raising=False)
        tool = _make_tool("brave_web_search", output={"kind": "results", "results": []})
        prior_calls = [{"id": "p1", "name": "brave_web_search", "arguments": {"query": "kim pse"}}]
        prior_results = [{"tool_call_id": "p1", "name": "brave_web_search", "content": '{"kind": "results", "results": []}'}]
        tcs = [ToolCall(id="n1", name="brave_web_search", arguments={"query": "Kim PSE"}),          # 앞선 호출과 같음(대소문자만 다름)
               ToolCall(id="n2", name="brave_web_search", arguments={"query": "kim publications"}),
               ToolCall(id="n3", name="brave_web_search", arguments={"query": "kim  publications"})]  # 같은 배치 안 중복
        results = await execute_tools(tcs, [tool], prior_calls=prior_calls, prior_results=prior_results)
        assert [tc.id for tc, _ in results] == ["n1", "n2", "n3"]                                   # 순서 보장
        assert results[0][1].startswith("[ERROR] duplicate_call:") and "tool_call_id p1" in results[0][1]
        assert results[1][1] == '{"kind": "results", "results": []}'
        assert results[2][1].startswith("[ERROR] duplicate_call:") and "tool_call_id n2" in results[2][1]
        assert tool.execute.await_count == 1                                                          # 실제 실행은 한 번

    @pytest.mark.asyncio
    async def test_cap_blocks_beyond_budget_but_not_internal_tools(self, monkeypatch):
        from agent_arch.core.config import settings
        monkeypatch.setattr(settings, "web_max_calls_per_turn", 1, raising=False)
        web = _make_tool("brave_llm_context", output={"kind": "context", "results": []})
        cal = _make_tool("calendar_list", output=[])
        tcs = [ToolCall(id="1", name="brave_llm_context", arguments={"query": "a"}),
               ToolCall(id="2", name="brave_llm_context", arguments={"query": "b"}),
               ToolCall(id="3", name="calendar_list", arguments={})]
        results = await execute_tools(tcs, [web, cal])
        assert results[0][1].startswith('{"kind"') and results[1][1].startswith("[ERROR] web_budget_exceeded:") and results[2][1] == "[]"
        assert web.execute.await_count == 1 and cal.execute.await_count == 1

    @pytest.mark.asyncio
    async def test_without_prior_lists_behaves_as_before(self):
        tool = _make_tool("calc", output=1)
        results = await execute_tools([_tc("calc"), _tc("calc")], [tool])                          # 사내 도구 반복은 그대로 실행
        assert [c for _, c in results] == ["1", "1"] and tool.execute.await_count == 2


class TestExecuteToolsSurvivesToolErrors:
    """도구가 예외를 던져도 턴이 죽지 않는다.

    실측 2026-09-23: 모델이 ``eapproval_create_document`` 를 필수 인자 ``title`` 없이
    불러 TypeError 가 났고, 실행기에 방어가 없어 예외가 gather 를 타고 올라가 스트림
    전체가 끝났다 — 사용자에게는 "답변을 생성하지 못했습니다"만 남았다.
    """

    @pytest.mark.asyncio
    async def test_missing_required_argument_returns_actionable_error(self):
        """인자 누락은 무엇이 빠졌는지 돌려준다 — 모델이 다음 호출에서 채우게."""
        tool = MagicMock()
        tool.name = "create_doc"
        tool.execute = AsyncMock(
            side_effect=TypeError(
                "_real_execute() missing 1 required positional argument: 'title'"
            )
        )

        results = await execute_tools([_tc("create_doc")], [tool])

        assert len(results) == 1
        _tc_out, content = results[0]
        assert content.startswith("[ERROR]")
        assert "title" in content

    @pytest.mark.asyncio
    async def test_unexpected_exception_does_not_kill_other_tools(self):
        """한 도구가 터져도 나머지 도구 결과는 그대로 돌아온다."""
        boom = MagicMock()
        boom.name = "boom"
        boom.execute = AsyncMock(side_effect=RuntimeError("내부 폭발"))
        fine = _make_tool("fine", output="정상")

        results = await execute_tools([_tc("boom"), _tc("fine")], [boom, fine])

        contents = {tc.name: content for tc, content in results}
        assert contents["boom"].startswith("[ERROR]")
        assert "정상" in contents["fine"]
