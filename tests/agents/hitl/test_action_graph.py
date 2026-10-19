"""agent_arch.agents.hitl 단위 테스트 — 정책 + interrupt/checkpoint/resume.

앱을 기동하지 않고 MemorySaver + mock 도구로 다음을 검증한다:
    (a) 첫 ainvoke 는 interrupt(draft)만 내고 실제 실행은 0회 (코드 강제 일시정지)
    (b) 동일 thread_id 재개(Command resume)는 confirm=True 로 정확히 1회 실행
    (c) 서로 다른 thread_id 는 독립 (한쪽 재개가 다른 쪽에 영향 없음)
"""
from contextlib import contextmanager

import pytest
from unittest.mock import AsyncMock, MagicMock

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_arch.agents.hitl.graph import build_action_graph, interrupt_payload
from agent_arch.agents.hitl.policy import APPROVAL_REQUIRED_TOOLS, requires_approval
from agent_arch.agents.tools.base import ToolResult
from agent_arch.agents.tools.registry import ToolRegistry


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _make_action_tool(name: str, view: dict | None = None):
    """confirm=True 호출 시 결과 view 를 반환하는 모의 액션 도구."""
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(
        return_value=ToolResult(success=True, output={"view": view or {"kind": "result"}})
    )
    return tool


@contextmanager
def _swap_registered(*tools):
    """모의 도구를 레지스트리에 **갈아 끼운다**(같은 이름이 이미 있으면 치웠다 되돌린다).

    ⚠️ 이름이 이미 등록돼 있을 수 있다. ``mattermost_skill_send_dm`` 은 예전엔 MCP 원격
    이름이라 레지스트리에 없었지만, 생성 파일 첨부 래퍼가 생기면서 로컬 도구가 됐다
    (``tools/skills/messenger``). 그냥 ``register`` 하면 ValueError 로 죽고, 먼저 등록한
    도구가 정리되지 않은 채 남아 **뒤따르는 테스트가 줄줄이** 깨진다.
    """
    saved = {}
    try:
        for tool in tools:
            existing = ToolRegistry.get(tool.name)
            if existing is not None:
                saved[tool.name] = existing
                ToolRegistry.unregister(tool.name)
            ToolRegistry.register(tool)
        yield tools
    finally:
        for tool in tools:
            if ToolRegistry.get(tool.name) is not None:
                ToolRegistry.unregister(tool.name)
        for tool in saved.values():
            ToolRegistry.register(tool)


def _pending(tool_name: str = "mail_send_message") -> dict:
    return {
        "action_id": "act-1",
        "tool_name": tool_name,
        "arguments": {"to": ["a@b.c"], "subject": "원래제목", "body": "본문"},
        "draft_view": {"app": "mail", "kind": "draft", "label": "메일 초안"},
    }


@pytest.fixture
def mail_tool():
    """mail_send_message 모의 도구를 레지스트리에 등록/해제한다."""
    tool = _make_action_tool(
        "mail_send_message", view={"app": "mail", "kind": "result", "label": "발송완료"}
    )
    with _swap_registered(tool):
        yield tool


# ── 정책 ──────────────────────────────────────────────────────────────────────

class TestPolicy:
    def test_core_approval_set_covers_collaboration_writes(self):
        # 코어는 일반 협업 도메인(일정·메일·메신저·인사)의 쓰기 도구만 안다.
        assert len(APPROVAL_REQUIRED_TOOLS) == 10
        assert requires_approval("mail_send_message")
        assert requires_approval("mail_save_contact")  # 연락처 저장(쓰기)도 승인 필요
        assert requires_approval("calendar_update_event")  # 일정 수정도 승인 필요
        assert requires_approval("calendar_delete_event")
        assert requires_approval("hr_request_attendance_correction")

    def test_extension_adds_domain_write_tools(self, monkeypatch):
        """업무 도메인의 쓰기 도구는 확장 슬롯으로 승인 목록에 더한다."""
        from agent_arch.extensions import extensions

        monkeypatch.setattr(extensions, "approval_required", {"orders_create"})
        monkeypatch.setattr(extensions, "action_done_messages", {
            "orders_create": lambda args: "임시저장했어요." if args.get("submit") is False else "등록했어요.",
        })
        from agent_arch.agents.hitl.policy import action_done_message

        assert requires_approval("orders_create")
        assert not requires_approval("orders_list")
        assert action_done_message("orders_create", {"submit": False}) == "임시저장했어요."
        assert action_done_message("orders_create", {}) == "등록했어요."

    def test_read_tools_do_not_require_approval(self):
        assert not requires_approval("search_colleagues")
        assert not requires_approval("calendar_list_events")  # 조회는 승인 불요


# ── interrupt / checkpoint / resume ───────────────────────────────────────────

class TestActionGraph:
    @pytest.mark.asyncio
    async def test_first_invoke_interrupts_without_executing(self, mail_tool):
        """첫 ainvoke 는 interrupt(draft)만 — 실제 실행 0회."""
        graph = build_action_graph(MemorySaver())
        config = {"configurable": {"thread_id": "conv-1"}}

        result = await graph.ainvoke({"pending_action": _pending()}, config)

        mail_tool.execute.assert_not_awaited()  # 사람 승인 전 실행 금지
        payload = interrupt_payload(result)
        assert payload is not None
        assert payload["tool_name"] == "mail_send_message"
        assert payload["draft_view"]["kind"] == "draft"
        assert "result" not in result or result.get("result") is None

    @pytest.mark.asyncio
    async def test_resume_executes_confirm_true_exactly_once(self, mail_tool):
        """재개 시 confirm=True + (수정 반영) 승인값으로 정확히 1회 실행."""
        graph = build_action_graph(MemorySaver())
        config = {"configurable": {"thread_id": "conv-2"}}

        await graph.ainvoke({"pending_action": _pending()}, config)
        approved = {"to": ["a@b.c"], "subject": "수정된제목", "body": "본문"}
        final = await graph.ainvoke(Command(resume=approved), config)

        mail_tool.execute.assert_awaited_once()
        _, kwargs = mail_tool.execute.call_args
        assert kwargs["confirm"] is True
        assert kwargs["subject"] == "수정된제목"  # 승인 시 수정값 반영
        assert "confirm" not in approved  # 원본 approved 오염 없음
        assert final["result"]["success"] is True
        assert final["result"]["view"]["label"] == "발송완료"

    @pytest.mark.asyncio
    async def test_llm_supplied_confirm_true_is_ignored(self, mail_tool):
        """초안 인자에 confirm=True 가 섞여 있어도 첫 호출은 절대 실행하지 않는다."""
        graph = build_action_graph(MemorySaver())
        config = {"configurable": {"thread_id": "conv-3"}}
        pending = _pending()
        pending["arguments"]["confirm"] = True  # LLM 이 위조했다고 가정

        result = await graph.ainvoke({"pending_action": pending}, config)

        mail_tool.execute.assert_not_awaited()
        assert interrupt_payload(result) is not None

    @pytest.mark.asyncio
    async def test_thread_isolation(self, mail_tool):
        """서로 다른 thread_id 는 독립 — A만 재개하면 실행은 1회뿐."""
        graph = build_action_graph(MemorySaver())
        cfg_a = {"configurable": {"thread_id": "conv-A"}}
        cfg_b = {"configurable": {"thread_id": "conv-B"}}

        await graph.ainvoke({"pending_action": _pending()}, cfg_a)
        await graph.ainvoke({"pending_action": _pending()}, cfg_b)
        # A만 재개
        await graph.ainvoke(Command(resume={"to": ["a@b.c"], "subject": "A", "body": "b"}), cfg_a)

        mail_tool.execute.assert_awaited_once()  # B는 여전히 일시정지 상태

    @pytest.mark.asyncio
    async def test_resume_reports_failure_when_tool_unregistered(self):
        """재개 시 도구가 미등록이면 실패 결과를 반환한다(예외로 죽지 않음)."""
        graph = build_action_graph(MemorySaver())
        config = {"configurable": {"thread_id": "conv-4"}}

        await graph.ainvoke({"pending_action": _pending("ghost_tool")}, config)
        final = await graph.ainvoke(Command(resume={"x": 1}), config)

        assert final["result"]["success"] is False
        assert "미등록" in final["result"]["error"]


class TestActionGraphQueue:
    """멀티-액션 큐: 항목마다 interrupt, resume 한 번에 정확히 1건 실행(이중실행 없음)."""

    @staticmethod
    def _item(action_id, tool_name, args):
        return {"action_id": action_id, "tool_name": tool_name, "arguments": args,
                "draft_view": {"app": "x", "kind": "draft"}}

    @pytest.mark.asyncio
    async def test_queue_resumes_each_once_in_order(self):
        t1 = _make_action_tool("mail_send_message", view={"app": "mail", "kind": "result", "label": "메일완료"})
        t2 = _make_action_tool("mattermost_skill_send_dm", view={"app": "mattermost", "kind": "result", "label": "DM완료"})
        with _swap_registered(t1, t2):
            graph = build_action_graph(MemorySaver())
            cfg = {"configurable": {"thread_id": "q-1"}}
            queue = [
                self._item("a1", "mail_send_message", {"to": ["x"], "subject": "s", "body": "b"}),
                self._item("a2", "mattermost_skill_send_dm", {"user_name": "u", "message": "m"}),
            ]
            r0 = await graph.ainvoke({"pending_queue": queue}, cfg)
            assert "__interrupt__" in r0
            t1.execute.assert_not_awaited()
            t2.execute.assert_not_awaited()
            p0 = interrupt_payload(r0)
            assert p0["tool_name"] == "mail_send_message" and p0["queue_total"] == 2 and p0["is_last"] is False

            r1 = await graph.ainvoke(Command(resume={"to": ["x"], "subject": "s", "body": "b"}), cfg)
            t1.execute.assert_awaited_once()      # 1번만 실행
            t2.execute.assert_not_awaited()
            p1 = interrupt_payload(r1)
            assert p1["tool_name"] == "mattermost_skill_send_dm" and p1["is_last"] is True

            r2 = await graph.ainvoke(Command(resume={"user_name": "u", "message": "m"}), cfg)
            t1.execute.assert_awaited_once()      # 여전히 1회 — 이중실행 없음(핵심)
            t2.execute.assert_awaited_once()
            assert interrupt_payload(r2) is None  # 큐 소진
            assert r2["result"]["success"] is True
            assert len(r2["results"]) == 2

    @pytest.mark.asyncio
    async def test_reject_skips_action_without_executing(self, mail_tool):
        from agent_arch.agents.hitl.graph import REJECT_SENTINEL
        graph = build_action_graph(MemorySaver())
        cfg = {"configurable": {"thread_id": "q-reject"}}
        await graph.ainvoke(
            {"pending_queue": [self._item("a1", "mail_send_message", {"to": ["x"]})]}, cfg
        )
        final = await graph.ainvoke(Command(resume=REJECT_SENTINEL), cfg)

        mail_tool.execute.assert_not_awaited()       # 거절 → 실행 안 함
        assert final["result"]["rejected"] is True
        assert final["result"]["success"] is False

    @pytest.mark.asyncio
    async def test_thread_reuse_across_turns_resets_state(self, mail_tool):
        """같은 thread_id 를 다음 턴에 재사용해도 리셋(index=0,results=[])하면 충돌 안 함.

        멀티-턴 버그 회귀: 이전 턴이 소비한 index 가 남아 새 짧은 큐와 충돌(IndexError)하던 것.
        """
        graph = build_action_graph(MemorySaver())
        cfg = {"configurable": {"thread_id": "reuse-1"}}
        # 턴 1: 큐 2개 모두 소비 → index=2 로 종료
        q1 = [self._item("a1", "mail_send_message", {"to": ["x"]}),
              self._item("a2", "mail_send_message", {"to": ["y"]})]
        await graph.ainvoke({"pending_queue": q1, "index": 0, "results": [], "result": None}, cfg)
        await graph.ainvoke(Command(resume={"to": ["x"]}), cfg)
        await graph.ainvoke(Command(resume={"to": ["y"]}), cfg)
        # 턴 2: 같은 thread_id 에 큐 1개 + 리셋 → 크래시 없이 새 큐 첫 항목 interrupt
        q2 = [self._item("b1", "mail_send_message", {"to": ["z"]})]
        r = await graph.ainvoke({"pending_queue": q2, "index": 0, "results": [], "result": None}, cfg)
        assert "__interrupt__" in r
        assert interrupt_payload(r)["queue_total"] == 1

    @pytest.mark.asyncio
    async def test_reject_then_reuse_thread_shows_new_card(self, mail_tool):
        """QA-05 회귀: 한 액션을 '거절'한 뒤 같은 대화(thread)에 새 액션을 적재하면
        새 카드(interrupt)가 정상으로 떠야 한다(거절 후 stale state 로 멈추지 않음).

        증상이던 것: 일정 삭제 거절 → 다음 삭제 요청 시 확인 카드가 안 뜨고 텍스트만.
        """
        from agent_arch.agents.hitl.graph import REJECT_SENTINEL
        graph = build_action_graph(MemorySaver())
        cfg = {"configurable": {"thread_id": "reject-reuse"}}
        # 턴 1: 삭제 A 적재 → 거절(그래프 END)
        await graph.ainvoke(
            {"pending_queue": [self._item("a1", "calendar_delete_event", {"uid": "A"})],
             "index": 0, "results": [], "result": None}, cfg
        )
        await graph.ainvoke(Command(resume=REJECT_SENTINEL), cfg)
        # 턴 2: 같은 thread 에 새 삭제 B 적재(리셋) → 새 카드가 떠야 함
        r = await graph.ainvoke(
            {"pending_queue": [self._item("b1", "calendar_delete_event", {"uid": "B"})],
             "index": 0, "results": [], "result": None}, cfg
        )
        assert "__interrupt__" in r, "거절 후 재사용 시 새 카드가 안 뜸(QA-05)"
        p = interrupt_payload(r)
        assert p["queue_total"] == 1
        assert p["arguments"]["uid"] == "B"  # 옛 A 가 아닌 새 B 가 떠야
