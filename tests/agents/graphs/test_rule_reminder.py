"""도구 결과 뒤 사용자 규칙 리마인더 — 도구 무관 공용 지점(react_graph._execute_node).

이 턴에 회상된 사용자 규칙이 있으면 도구 결과 묶음의 마지막 결과 뒤에 붙어 **다음 LLM 호출**이 본다.
저장되는 도구 결과(tool_ctx·이력)에는 들어가지 않는다. 규칙이 없으면 아무것도 붙지 않는다.
"""
import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.context import set_memory_recall
from agent_arch.agents.graphs.react_graph import run_react_graph
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import Message, StreamEvent, ToolCall
from agent_arch.services.memory_v2.recall import RecallResult

RULE = "점심 상신 시 결재자는 김부장, 결제수단은 법인카드(2857)로 한다"


def _llm(seen):
    calls = [
        [StreamEvent(type="tool_calls", tool_calls=[
            ToolCall(id="t1", name="a", arguments={}), ToolCall(id="t2", name="b", arguments={})]),
         StreamEvent(type="done", finish_reason="tool_calls")],
        [StreamEvent(type="text", content="완료"), StreamEvent(type="done", finish_reason="stop")],
    ]

    def stream(messages, **kw):
        seen.append([(m.role, m.content) for m in messages])
        events = calls.pop(0)

        async def gen():
            for e in events:
                yield e
        return gen()

    llm = MagicMock()
    llm.stream = stream
    llm.bind_tools = AsyncMock()
    return llm


def _tool(name):
    t = MagicMock()
    t.name = name
    t.execute = AsyncMock(return_value=ToolResult(success=True, output={"required_next": ["결제수단"]}))
    return t


async def _run(recall):
    set_memory_recall(recall)
    seen: list = []
    try:
        res = await run_react_graph(_llm(seen), "점심 상신", [Message(role="user", content="점심 상신")],
                                    [_tool("a"), _tool("b")], AsyncMock(return_value=[]), 3)
    finally:
        set_memory_recall(None)
    tool_msgs = [c for role, c in seen[1] if role == "tool"]
    return res, tool_msgs


@pytest.mark.asyncio
async def test_reminder_after_last_tool_result_only_and_not_persisted(caplog):
    caplog.set_level("INFO", logger="agent_arch.agents.graphs.react_graph")
    res, tool_msgs = await _run(RecallResult(atoms=[
        {"atom_type": "instruction", "content": RULE},
        {"atom_type": "episodic", "content": "9/22 김부장에게 상신했다"},
    ]))
    assert len(tool_msgs) == 2
    assert "[사용자가 정한 규칙" not in tool_msgs[0]
    shown = json.loads(tool_msgs[1])                     # 결과 구조는 그대로(JSON 객체에 키 하나 추가)
    assert shown["required_next"] == ["결제수단"] and shown["user_rules"].startswith("[사용자가 정한 규칙")
    assert RULE in tool_msgs[1] and "9/22" not in tool_msgs[1]      # 기록은 규칙이 아니다
    assert all("[사용자가 정한 규칙" not in str(r.get("content")) for r in res.tool_results)
    assert "[memory_rules] 리마인더 부착 tool=b rules=1 form=json" in caplog.text   # 운영 관측 로그


@pytest.mark.asyncio
async def test_no_rules_no_reminder():
    _, tool_msgs = await _run(RecallResult(atoms=[{"atom_type": "persona", "content": "녹차를 좋아한다"}]))
    assert all("[사용자가 정한 규칙" not in c for c in tool_msgs)
    _, tool_msgs = await _run(None)
    assert all("[사용자가 정한 규칙" not in c for c in tool_msgs)


def test_attach_keeps_non_json_readable():
    from agent_arch.agents.graphs.react_graph import _attach_rule_reminder
    assert _attach_rule_reminder("[ERROR] 실패", "R") == "[ERROR] 실패\n\nR"
    assert json.loads(_attach_rule_reminder('{"a": 1}', "R")) == {"a": 1, "user_rules": "R"}
    assert _attach_rule_reminder("[1, 2]", "R") == "[1, 2]\n\nR"
