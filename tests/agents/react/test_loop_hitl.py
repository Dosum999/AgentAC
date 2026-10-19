"""agent_arch.agents.react.loop HITL 게이트 단위 테스트 (Slice 2).

USE_LANGGRAPH_HITL 플래그에 따라:
    ON  + 승인필요 도구 호출 → confirm=False 강제 + pending_action + 턴 종결(action_pending)
    ON  + 읽기 도구          → 게이트 미적용(평소 경로)
    OFF + 승인필요 도구 호출 → 평소 경로(LLM 이 준 confirm 그대로, pending_action 없음)
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.react import loop as loop_mod
from agent_arch.agents.react.loop import react_loop_events
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import Message, StreamEvent, ToolCall


_MSGS = [Message(role="user", content="메일 보내줘")]


def _text(content="완료했습니다", finish_reason="stop"):
    async def _gen(*args, **kwargs):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason=finish_reason, usage=None)
    return _gen


def _tool_calls(tool_calls):
    async def _gen(*args, **kwargs):
        yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
        yield StreamEvent(type="done", finish_reason="tool_calls", usage=None)
    return _gen


def _llm(*per_iter_gens):
    iter_gens = list(per_iter_gens)

    def _next_gen(*args, **kwargs):
        return iter_gens.pop(0)(*args, **kwargs)

    llm = MagicMock()
    llm.stream = _next_gen
    llm.bind_tools = AsyncMock()
    return llm


def _action_tool(name="mail_send_message"):
    """confirm 값을 기록하고 draft view 를 반환하는 모의 액션 도구."""
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(
        return_value=ToolResult(
            success=True,
            output={"view": {"app": "mail", "kind": "draft", "label": "메일 초안"}},
        )
    )
    return tool


def _read_tool(name="search_colleagues"):
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(return_value=ToolResult(success=True, output="결과"))
    return tool


async def _run(llm, tools, max_iterations=5):
    events = [
        e
        async for e in react_loop_events(
            llm, "메일 보내줘", list(_MSGS), tools, AsyncMock(return_value=[]), max_iterations
        )
    ]
    results = [e for e in events if isinstance(e, LoopResult)]
    assert len(results) == 1
    return [e for e in events if isinstance(e, ProgressEvent)], results[0]


class TestHitlGate:
    @pytest.mark.asyncio
    async def test_flag_on_forces_draft_and_pauses(self, monkeypatch):
        """flag ON + 승인필요 도구 → confirm=False 강제, pending_action, action_pending."""
        monkeypatch.setattr(loop_mod.settings, "use_langgraph_hitl", True)
        tool = _action_tool("mail_send_message")
        tc = ToolCall(
            id="id1",
            name="mail_send_message",
            arguments={"to": ["a@b.c"], "subject": "s", "body": "b", "confirm": True},
        )
        _, result = await _run(_llm(_tool_calls([tc])), [tool])

        assert result.finish_reason == "action_pending"
        assert result.pending_action is not None
        assert result.pending_action["tool_name"] == "mail_send_message"
        # confirm 은 코드가 False 로 강제 (LLM 이 True 를 줘도 무시)
        _, kwargs = tool.execute.call_args
        assert kwargs["confirm"] is False
        # pending_action.arguments 에는 confirm 미포함
        assert "confirm" not in result.pending_action["arguments"]
        assert result.pending_action["arguments"]["subject"] == "s"
        # draft view 추출
        assert result.pending_action["draft_view"]["kind"] == "draft"

    @pytest.mark.asyncio
    async def test_flag_on_does_not_gate_read_tools(self, monkeypatch):
        """flag ON 이라도 읽기 도구는 게이트 미적용 — 평소처럼 루프 진행."""
        monkeypatch.setattr(loop_mod.settings, "use_langgraph_hitl", True)
        tool = _read_tool("search_colleagues")
        _, result = await _run(
            _llm(_tool_calls([ToolCall(id="r1", name="search_colleagues", arguments={})]), _text("결과입니다")),
            [tool],
        )
        assert result.pending_action is None
        assert result.finish_reason != "action_pending"
        assert result.content == "결과입니다"

    @pytest.mark.asyncio
    async def test_flag_on_collects_all_approval_drafts_into_queue(self, monkeypatch):
        """한 턴에 승인필요 액션 2개면 둘 다 pending_queue 에 담고 첫 draft에서 return 안 함."""
        monkeypatch.setattr(loop_mod.settings, "use_langgraph_hitl", True)
        mail = _action_tool("mail_send_message")
        dm = _action_tool("mattermost_skill_send_dm")
        tc1 = ToolCall(id="t1", name="mail_send_message", arguments={"to": ["a@b.c"], "subject": "s", "body": "b", "confirm": True})
        tc2 = ToolCall(id="t2", name="mattermost_skill_send_dm", arguments={"user_name": "u", "message": "m", "confirm": True})
        _, result = await _run(_llm(_tool_calls([tc1, tc2])), [mail, dm])

        assert result.finish_reason == "action_pending"
        assert result.pending_queue is not None and len(result.pending_queue) == 2
        assert result.pending_action == result.pending_queue[0]  # 단수 = 큐[0] 미러
        assert {p["tool_name"] for p in result.pending_queue} == {"mail_send_message", "mattermost_skill_send_dm"}
        # 둘 다 confirm=False 강제(조기 발송 0)
        assert mail.execute.call_args.kwargs["confirm"] is False
        assert dm.execute.call_args.kwargs["confirm"] is False
        for p in result.pending_queue:
            assert "confirm" not in p["arguments"]

    @pytest.mark.asyncio
    async def test_flag_on_no_draft_view_does_not_pause(self, monkeypatch):
        """승인필요 도구가 실행됐어도 draft view 가 없으면(미발견/에러) 멈추지 않고
        루프를 이어 LLM 이 복구하게 둔다 — 멈춤·빈응답 버그 회귀 방지."""
        monkeypatch.setattr(loop_mod.settings, "use_langgraph_hitl", True)
        tool = MagicMock()
        tool.name = "mail_send_message"
        tool.execute = AsyncMock(return_value=ToolResult(success=False, output=None, error="도구 미발견"))
        tc = ToolCall(id="id1", name="mail_send_message", arguments={"confirm": True})
        _, result = await _run(_llm(_tool_calls([tc]), _text("복구 후 답변")), [tool])

        assert result.pending_action is None  # draft view 없어 일시정지 안 함
        assert result.finish_reason != "action_pending"
        assert result.content == "복구 후 답변"
        # 그래도 confirm 은 False 로 강제됨(조기 실행 차단은 유지)
        _, kwargs = tool.execute.call_args
        assert kwargs["confirm"] is False

    @pytest.mark.asyncio
    async def test_flag_off_keeps_legacy_path(self, monkeypatch):
        """flag OFF → 게이트 없음. LLM 이 준 confirm 그대로, pending_action 없음."""
        monkeypatch.setattr(loop_mod.settings, "use_langgraph_hitl", False)
        tool = _action_tool("mail_send_message")
        tc = ToolCall(id="id1", name="mail_send_message", arguments={"confirm": True})
        _, result = await _run(_llm(_tool_calls([tc]), _text("발송했습니다")), [tool])

        assert result.pending_action is None
        assert result.finish_reason != "action_pending"
        _, kwargs = tool.execute.call_args
        assert kwargs.get("confirm") is True  # 강제하지 않음 (기존 동작 보존)
