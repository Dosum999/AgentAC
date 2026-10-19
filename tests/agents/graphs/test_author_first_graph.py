"""author-first 그래프 배선 — 첫 도구 고정 + `must_contain_link` 시스템 주입.

배경(2026-09-16 운영): "홍길동 이사님이 최근 언급한 에이전트의 역할과 권한 문서라고 공유한
링크 찾아줘" 에 에이전트가 작성자 검색을 부르지 않고 일반 지식검색으로 가서, AI팀 채널에
업로드된 첨부파일(TD4 화면설계서.hwp)을 답으로 내놨다. 정답은 그 사람이 올린 게시글이다.

`system.py` 의 「반드시 작성자 검색」 지시와 「`must_contain_link=true` 를 넣어라」 지시가
**둘 다 지켜지지 않았다.** 프롬프트는 확률적이라, 둘 다 코드로 내린다.

⚠️ LangGraph 경로 전용이다(`USE_LANGGRAPH_REACT=true`). 구 `react/loop.py` 에는 강제 지점이 없다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.graphs.react_graph import run_react_graph
from agent_arch.agents.react.author_first import AUTHOR_TOOL_NAME
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import Message, StreamEvent, ToolCall

_SEARCH_TOOL = "mattermost_search_knowledge"

# 이번 사고를 일으킨 실제 문장
_Q = "홍길동 이사님이 최근 언급한 에이전트의 역할과 권한 문서라고 공유한 링크 찾아줘"


def _stream_tool_calls(tool_calls):
    async def _gen(*a, **k):
        yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
        yield StreamEvent(type="done", finish_reason="tool_calls", usage=None)
    return _gen


def _stream_text(content):
    async def _gen(*a, **k):
        yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason="stop", usage=None)
    return _gen


def _make_llm(specs, calls):
    llm = MagicMock()
    gens = list(specs)

    def _next(*a, **k):
        calls.append(k)
        return gens.pop(0)(*a, **k)

    llm.stream = _next
    llm.bind_tools = AsyncMock()
    return llm


def _tool(name, output='{"count": 1, "results": [{"permalink": "https://talk/x"}]}'):
    t = MagicMock()
    t.name = name
    t.execute = AsyncMock(return_value=ToolResult(success=True, output=output))
    return t


@pytest.mark.asyncio
async def test_first_call_is_pinned_to_the_author_tool():
    calls = []
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=AUTHOR_TOOL_NAME,
                                     arguments={"author": "홍길동", "query": "역할과 권한"})]),
        _stream_text("찾았습니다."),
    ], calls)
    await run_react_graph(
        llm, _Q, [Message(role="user", content=_Q)],
        [_tool(AUTHOR_TOOL_NAME), _tool(_SEARCH_TOOL)],
        AsyncMock(return_value=[]), 5,
    )
    assert calls[0]["tool_mode"] == "ANY"
    assert calls[0]["allowed_tools"] == [AUTHOR_TOOL_NAME]
    # 두 번째부터는 평소대로
    assert calls[1]["tool_mode"] == "AUTO"


@pytest.mark.asyncio
async def test_original_intent_is_preserved_without_forcing_body_urls(monkeypatch):
    """모델이 `must_contain_link` 를 빠뜨려도 시스템이 채운다."""
    seen = {}

    async def _capture(tool_calls, all_tools, **_kw):
        seen["args"] = dict(tool_calls[0].arguments or {})
        return [(tool_calls[0], '{"count": 1, "results": [{"permalink": "https://talk/x"}]}')]

    monkeypatch.setattr("agent_arch.agents.graphs.react_graph.execute_tools", _capture)
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=AUTHOR_TOOL_NAME,
                                     arguments={"author": "홍길동", "query": "역할과 권한"})]),
        _stream_text("찾았습니다."),
    ], [])
    await run_react_graph(
        llm, _Q, [Message(role="user", content=_Q)],
        [_tool(AUTHOR_TOOL_NAME)], AsyncMock(return_value=[]), 5,
    )
    assert "must_contain_link" not in seen["args"]
    assert seen["args"]["search_intent"] == _Q


@pytest.mark.asyncio
async def test_model_supplied_value_is_respected(monkeypatch):
    """모델이 이미 채웠으면 덮어쓰지 않는다."""
    seen = []

    async def _capture(tool_calls, all_tools, **_kw):
        seen.append(dict(tool_calls[0].arguments or {}))
        return [(tool_calls[0], '{"count": 1, "results": [{"x": 1}]}')]

    monkeypatch.setattr("agent_arch.agents.graphs.react_graph.execute_tools", _capture)
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=AUTHOR_TOOL_NAME,
                                     arguments={"author": "홍길동", "must_contain_link": True})]),
        _stream_text("네."),
    ], [])
    await run_react_graph(
        llm, _Q, [Message(role="user", content=_Q)],
        [_tool(AUTHOR_TOOL_NAME)], AsyncMock(return_value=[]), 5,
    )
    assert seen[0]["must_contain_link"] is True
    assert len(seen) == 1                      # 0건이 아니므로 재검색 없음


@pytest.mark.asyncio
async def test_zero_hits_retries_without_the_link_filter(monkeypatch):
    """링크 한정으로 0건이면 조건을 풀어 한 번 더 부른다 — 판정이 틀려도 답을 잃지 않는다."""
    seen = []

    async def _capture(tool_calls, all_tools, **_kw):
        args = dict(tool_calls[0].arguments or {})
        seen.append(args)
        body = ('{"count": 0, "results": []}' if args.get("must_contain_link")
                else '{"count": 1, "results": [{"permalink": "https://talk/x"}]}')
        return [(tool_calls[0], body)]

    monkeypatch.setattr("agent_arch.agents.graphs.react_graph.execute_tools", _capture)
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=AUTHOR_TOOL_NAME,
                                     arguments={"author": "홍길동", "must_contain_link": True})]),
        _stream_text("찾았습니다."),
    ], [])
    await run_react_graph(
        llm, _Q, [Message(role="user", content=_Q)],
        [_tool(AUTHOR_TOOL_NAME)], AsyncMock(return_value=[]), 5,
    )
    assert len(seen) == 2, "0건이면 조건을 풀어 재검색해야 한다"
    assert seen[0].get("must_contain_link") is True
    assert "must_contain_link" not in seen[1]


@pytest.mark.asyncio
async def test_ordinary_question_is_not_pinned():
    """묶으면 죽는 문장 — 전사 검색이 정답이다."""
    calls = []
    q = "지난주 품질 이슈 찾아줘"
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=_SEARCH_TOOL, arguments={"query": q})]),
        _stream_text("정리했습니다."),
    ], calls)
    await run_react_graph(
        llm, q, [Message(role="user", content=q)],
        [_tool(AUTHOR_TOOL_NAME), _tool(_SEARCH_TOOL)],
        AsyncMock(return_value=[]), 5,
    )
    assert calls[0].get("allowed_tools") in (None, [])
