"""agent_arch.agents.hitl.resume 단위 테스트 — load_pending / 검증 / resume_action.

프로세스 단일 그래프(get_action_graph, MemorySaver)를 실제로 사용하되 thread_id 를
테스트마다 유니크하게 둬 격리한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.hitl import (
    StalePendingAction,
    get_action_graph,
    load_pending,
    resume_action,
    validate_resume_values,
)
from agent_arch.agents.tools.base import ToolResult
from agent_arch.agents.tools.registry import ToolRegistry


def _pending() -> dict:
    return {
        "action_id": "a1",
        "tool_name": "mail_send_message",
        "arguments": {"to": ["a@b.c"], "subject": "s", "body": "b"},
        "draft_view": {
            "app": "mail",
            "kind": "draft",
            "form": {"values": {"to": ["a@b.c"], "subject": "s", "body": "b"}},
        },
    }


async def _checkpoint(conversation_id: str) -> None:
    """interrupt 상태로 만들어 둔다(초안 적재)."""
    graph = get_action_graph()
    await graph.ainvoke(
        {"pending_action": _pending()}, {"configurable": {"thread_id": conversation_id}}
    )


@pytest.fixture
def mail_tool():
    tool = MagicMock()
    tool.name = "mail_send_message"
    tool.execute = AsyncMock(
        return_value=ToolResult(
            success=True, output={"view": {"app": "mail", "kind": "result", "label": "발송완료"}}
        )
    )
    ToolRegistry.register(tool)
    try:
        yield tool
    finally:
        ToolRegistry.unregister("mail_send_message")


class TestLoadPending:
    @pytest.mark.asyncio
    async def test_none_when_no_checkpoint(self):
        assert await load_pending("res-never-seen") is None

    @pytest.mark.asyncio
    async def test_returns_pending_after_interrupt(self):
        await _checkpoint("res-load-1")
        pending = await load_pending("res-load-1")
        assert pending is not None
        assert pending["tool_name"] == "mail_send_message"

    @pytest.mark.asyncio
    async def test_none_after_consumed(self, mail_tool):
        await _checkpoint("res-consumed-1")
        await resume_action("res-consumed-1", {"to": ["a@b.c"], "subject": "s", "body": "b"})
        # 재개되어 소비됨 → 재-resume 차단(409)을 위해 None 반환
        assert await load_pending("res-consumed-1") is None


class TestValidate:
    def test_rejects_unknown_field(self):
        with pytest.raises(ValueError):
            validate_resume_values(_pending(), {"to": ["x"], "evil_field": "1"})

    def test_allows_known_and_confirm(self):
        validate_resume_values(_pending(), {"to": ["x"], "subject": "y", "body": "z", "confirm": True})


class TestResumeAction:
    @pytest.mark.asyncio
    async def test_resume_executes_confirm_true_with_edits(self, mail_tool):
        await _checkpoint("res-exec-1")
        result = await resume_action("res-exec-1", {"to": ["a@b.c"], "subject": "수정됨", "body": "b"})

        assert result["success"] is True
        assert result["view"]["label"] == "발송완료"
        assert result["next_pending"] is None  # 단일 액션 → 다음 없음
        mail_tool.execute.assert_awaited_once()
        _, kwargs = mail_tool.execute.call_args
        assert kwargs["confirm"] is True
        assert kwargs["subject"] == "수정됨"

    @pytest.mark.asyncio
    async def test_resume_with_empty_values_still_executes(self, mail_tool):
        """값 무수정 승인(빈 values) 회귀 가드 — Command(resume={}) 는 falsy 라 노드가
        재실행되지 않는 langgraph 특성 때문에, 빈 승인은 진공 값({"confirm": True})으로
        재개해야 한다(2026-08-13 accounting E2E 실측 버그, af0cc87 재적용)."""
        await _checkpoint("res-empty-1")
        result = await resume_action("res-empty-1", {})

        assert result["success"] is True  # 무동작·502 가 아니라 draft 인자 그대로 실행
        mail_tool.execute.assert_awaited_once()
        _, kwargs = mail_tool.execute.call_args
        assert kwargs["confirm"] is True
        assert kwargs["subject"] == "s"  # draft 원본 인자 유지


def _qitem(action_id: str, tool_name: str, args: dict) -> dict:
    return {"action_id": action_id, "tool_name": tool_name, "arguments": args,
            "draft_view": {"app": "x", "kind": "draft", "form": {"values": args}}}


async def _checkpoint_queue(conversation_id: str, queue: list) -> None:
    graph = get_action_graph()
    await graph.ainvoke({"pending_queue": queue}, {"configurable": {"thread_id": conversation_id}})


@pytest.fixture
def two_tools():
    """승인 도구 2종을 모의로 갈아 끼운다.

    ⚠️ **이름이 이미 등록돼 있을 수 있다.** ``mattermost_skill_send_dm`` 은 예전엔 MCP
    원격 이름이라 레지스트리에 없었지만, 생성 파일 첨부 래퍼가 생기면서 로컬 도구가 됐다
    (``tools/skills/messenger``). 그때 이 픽스처가 ``register`` 에서 예외로 죽었고,
    **먼저 등록한 mail 이 정리되지 않은 채** 남아 뒤따르는 테스트가 줄줄이 깨졌다
    (setup 예외라 finally 가 돌지 않는다). 그래서 기존 도구를 치워 두고 끝나면 되돌린다.
    """
    tools, saved = {}, {}
    for name, label in [("mail_send_message", "메일완료"), ("mattermost_skill_send_dm", "DM완료")]:
        existing = ToolRegistry.get(name)
        if existing is not None:
            saved[name] = existing
            ToolRegistry.unregister(name)
        t = MagicMock()
        t.name = name
        t.execute = AsyncMock(
            return_value=ToolResult(success=True, output={"view": {"app": name, "kind": "result", "label": label}})
        )
        ToolRegistry.register(t)
        tools[name] = t
    try:
        yield tools
    finally:
        for name in tools:
            ToolRegistry.unregister(name)
        for name, tool in saved.items():
            ToolRegistry.register(tool)


class TestResumeQueue:
    @pytest.mark.asyncio
    async def test_resume_advances_queue_with_next_pending(self, two_tools):
        queue = [
            _qitem("a1", "mail_send_message", {"to": ["x"], "subject": "s", "body": "b"}),
            _qitem("a2", "mattermost_skill_send_dm", {"user_name": "u", "message": "m"}),
        ]
        await _checkpoint_queue("rq-1", queue)
        p0 = await load_pending("rq-1")
        assert p0["tool_name"] == "mail_send_message" and p0["is_last"] is False

        r1 = await resume_action("rq-1", {"to": ["x"], "subject": "s", "body": "b"})
        assert r1["success"] is True
        assert r1["next_pending"]["tool_name"] == "mattermost_skill_send_dm"
        two_tools["mail_send_message"].execute.assert_awaited_once()
        two_tools["mattermost_skill_send_dm"].execute.assert_not_awaited()

        p1 = await load_pending("rq-1")
        assert p1["tool_name"] == "mattermost_skill_send_dm" and p1["is_last"] is True

        r2 = await resume_action("rq-1", {"user_name": "u", "message": "m"})
        assert r2["success"] is True and r2["next_pending"] is None
        two_tools["mattermost_skill_send_dm"].execute.assert_awaited_once()
        assert await load_pending("rq-1") is None

    @pytest.mark.asyncio
    async def test_reject_skips_and_advances(self, two_tools):
        queue = [
            _qitem("a1", "mail_send_message", {"to": ["x"]}),
            _qitem("a2", "mattermost_skill_send_dm", {"user_name": "u", "message": "m"}),
        ]
        await _checkpoint_queue("rq-reject", queue)
        r1 = await resume_action("rq-reject", {}, reject=True)
        assert r1.get("rejected") is True
        assert r1["next_pending"]["tool_name"] == "mattermost_skill_send_dm"
        two_tools["mail_send_message"].execute.assert_not_awaited()

        r2 = await resume_action("rq-reject", {"user_name": "u", "message": "m"})
        assert r2["success"] is True and r2["next_pending"] is None
        two_tools["mattermost_skill_send_dm"].execute.assert_awaited_once()


# ── 재개 잠금(Redis 경로) ─────────────────────────────────────────────────────


class _FakeRedis:
    """SETNX/GET/DELETE 만 흉내 내는 최소 Redis — 여러 워커를 가정한 경로를 덮는다."""

    def __init__(self):
        self.store: dict = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    async def get(self, key):
        return self.store.get(key)

    async def delete(self, key):
        self.store.pop(key, None)


@pytest.mark.asyncio
async def test_resume_guard_uses_redis_and_releases(monkeypatch):
    """Redis 가 있으면 그쪽으로 잠근다 — 프로세스가 여럿이어도 승인은 한 번만 통과해야 한다."""
    import agent_arch.memory.redis_store as rs
    from agent_arch.agents.hitl import ResumeInProgress, resume_guard

    fake = _FakeRedis()
    monkeypatch.setattr(rs.redis_store, "redis", fake, raising=False)

    async with resume_guard("conv-lock-1"):
        assert fake.store                                    # 잠겼다
        with pytest.raises(ResumeInProgress):
            async with resume_guard("conv-lock-1"):
                pass
        async with resume_guard("conv-lock-2"):              # 다른 대화는 막지 않는다
            pass
    assert fake.store == {}                                  # 빠져나오면 푼다


@pytest.mark.asyncio
async def test_resume_guard_does_not_block_when_redis_is_down(monkeypatch):
    """Redis 장애로 승인을 막지는 않는다 — 프로세스 내 잠금으로 내려간다."""
    import agent_arch.memory.redis_store as rs
    from agent_arch.agents.hitl import resume_guard

    class _Broken(_FakeRedis):
        async def set(self, *a, **k):
            raise RuntimeError("connection refused")

    monkeypatch.setattr(rs.redis_store, "redis", _Broken(), raising=False)
    async with resume_guard("conv-lock-3"):
        pass


class _SnapRedis:
    def __init__(self):
        self.d = {}

    async def set(self, k, v, ex=None, nx=False):
        if nx and k in self.d:
            return False
        self.d[k] = v
        return True

    async def get(self, k):
        return self.d.get(k)

    async def delete(self, k):
        self.d.pop(k, None)


class TestSurvivesRestart:
    """파드 재시작(MemorySaver 유실) 뒤에도 승인 카드가 살아 있다 — Redis 사본으로 복원(2026-10-13 운영)."""

    @staticmethod
    def _restart(monkeypatch):
        import agent_arch.agents.hitl.graph as g

        monkeypatch.setattr(g, "_ACTION_GRAPH", None)  # 새 프로세스 = 빈 체크포인터

    @pytest.mark.asyncio
    async def test_pending_restored_and_consumed_once(self, monkeypatch, mail_tool):
        import agent_arch.agents.hitl.resume as r

        fake = _SnapRedis()
        monkeypatch.setattr(r, "_redis", lambda: fake)
        cid = "res-restart-1"
        await _checkpoint(cid)
        await r.save_queue_snapshot(cid, [_pending()])

        self._restart(monkeypatch)
        pending = await load_pending(cid)
        assert pending is not None and pending["tool_name"] == "mail_send_message"

        out = await resume_action(cid, {"to": ["a@b.c"], "subject": "s", "body": "b"})
        assert out["success"] is True
        assert mail_tool.execute.await_count == 1
        assert f"hitl:queue:{cid}" not in fake.d  # 소비된 항목은 사본에서도 빠진다

        self._restart(monkeypatch)
        assert await load_pending(cid) is None  # 실제로 처리된 카드는 여전히 '이미 처리됨'

    @pytest.mark.asyncio
    async def test_no_snapshot_keeps_old_behavior(self, monkeypatch):
        import agent_arch.agents.hitl.resume as r

        monkeypatch.setattr(r, "_redis", lambda: _SnapRedis())
        await _checkpoint("res-restart-2")
        self._restart(monkeypatch)
        assert await load_pending("res-restart-2") is None


async def _checkpoint_with_id(conversation_id: str, pending_id: str) -> None:
    graph = get_action_graph()
    await graph.ainvoke(
        {"pending_action": {**_pending(), "pending_id": pending_id}},
        {"configurable": {"thread_id": conversation_id}},
    )


class TestPendingIdMatch:
    """카드의 pending_id 가 현재 대기 항목과 같을 때만 재개한다 — 앞 카드가 새 초안을 실행하지 않게."""

    @pytest.mark.asyncio
    async def test_matching_id_executes(self, mail_tool):
        await _checkpoint_with_id("pid-ok", "p-1")
        result = await resume_action("pid-ok", {}, pending_id="p-1")
        assert result["success"] is True
        mail_tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mismatched_id_is_refused_and_nothing_runs(self, mail_tool):
        await _checkpoint_with_id("pid-stale", "p-new")
        with pytest.raises(StalePendingAction):
            await resume_action("pid-stale", {}, pending_id="p-old")
        mail_tool.execute.assert_not_awaited()
        assert (await load_pending("pid-stale"))["pending_id"] == "p-new"  # 대기 상태 그대로

    @pytest.mark.asyncio
    async def test_missing_id_is_refused_when_item_has_one(self, mail_tool):
        await _checkpoint_with_id("pid-missing", "p-1")
        with pytest.raises(StalePendingAction):
            await resume_action("pid-missing", {})
        mail_tool.execute.assert_not_awaited()
