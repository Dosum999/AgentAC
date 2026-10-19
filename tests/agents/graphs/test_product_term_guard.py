"""지식검색 질의에서 지워진 **제품 이름**을 시스템이 되살린다.

배경(2026-09-16 실사용): "에이전트의 권한과 역할 문서를 언급한 링크 찾아줘" 에 모델이
`query="권한 역할 문서"` 를 보냈다 — **`에이전트` 가 빠졌다.** 그러자 권한(RBAC) 얘기가 든
기술제안서가 통째로 걸려 엉뚱한 문서를 답했다.

직전 다섯 턴은 전부 `에이전트` 를 남겨 정답을 찾았다:

    02:36  에이전트 역할 권한
    04:53  에이전트 역할 권한
    05:03  에이전트 역할 권한 문서
    07:58  에이전트 권한 역할 문서   (×3)
    08:16  권한 역할 문서        ← 여기서 빠졌다

원인은 프롬프트의 「회사명·조직단위명을 `query` 에 남기지 마라」 규칙이 **제품 이름까지
먹은 것**이다. 프롬프트에 예외를 적었지만 확률적이라 코드로 받친다.

⚠️ `@에이전트`·"에이전트야" 처럼 **부르는 말**은 검색어가 아니다 — 그건 되살리지 않는다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.graphs.react_graph import run_react_graph
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import Message, StreamEvent, ToolCall

_SEARCH = "mattermost_search_knowledge"


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


def _make_llm(specs):
    llm = MagicMock()
    gens = list(specs)
    llm.stream = lambda *a, **k: gens.pop(0)(*a, **k)
    llm.bind_tools = AsyncMock()
    return llm


def _tool(name):
    t = MagicMock()
    t.name = name
    t.execute = AsyncMock(return_value=ToolResult(success=True, output='{"count":1,"results":[{}]}'))
    return t


async def _run(monkeypatch, user_input, sent_query):
    """모델이 `sent_query` 를 보냈을 때 **실제로 실행된** 인자를 돌려준다."""
    seen = {}

    async def _capture(tool_calls, all_tools, **_kw):
        seen["args"] = dict(tool_calls[0].arguments or {})
        return [(tool_calls[0], '{"count":1,"results":[{}]}')]

    monkeypatch.setattr("agent_arch.agents.graphs.react_graph.execute_tools", _capture)
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=_SEARCH, arguments={"query": sent_query})]),
        _stream_text("정리했습니다."),
    ])
    await run_react_graph(
        llm, user_input, [Message(role="user", content=user_input)],
        [_tool(_SEARCH)], AsyncMock(return_value=[]), 5,
    )
    return seen["args"]


@pytest.mark.asyncio
async def test_dropped_product_name_is_restored(monkeypatch):
    """사고를 일으킨 그 턴 그대로."""
    args = await _run(monkeypatch, "에이전트의 권한과 역할 문서를 언급한 링크 찾아줘", "권한 역할 문서")
    assert args["query"] == "에이전트 권한 역할 문서"


@pytest.mark.asyncio
async def test_query_that_already_has_it_is_untouched(monkeypatch):
    """직전 다섯 턴처럼 모델이 남겼으면 그대로 둔다 — 중복으로 붙이면 안 된다."""
    args = await _run(monkeypatch, "에이전트 권한과 역할 문서 찾아줘", "에이전트 권한 역할 문서")
    assert args["query"] == "에이전트 권한 역할 문서"


@pytest.mark.asyncio
async def test_vocative_is_not_a_search_term(monkeypatch):
    """"에이전트야 폴리머 동향 알려줘" 의 `에이전트` 는 부르는 말이다 — 검색어가 아니다."""
    args = await _run(monkeypatch, "에이전트야 폴리머 동향 알려줘", "폴리머 동향")
    assert args["query"] == "폴리머 동향"


@pytest.mark.asyncio
async def test_mention_prefix_is_not_a_search_term(monkeypatch):
    """메신저 봇 호출은 `@에이전트` 로 시작한다 — 매 턴 검색어가 되면 안 된다."""
    args = await _run(monkeypatch, "@에이전트 폴리머 동향 알려줘", "폴리머 동향")
    assert args["query"] == "폴리머 동향"


@pytest.mark.asyncio
async def test_mention_prefix_does_not_block_a_real_topic(monkeypatch):
    """부르면서 동시에 주제로 쓸 수 있다 — `@에이전트 에이전트의 권한 문서`."""
    args = await _run(monkeypatch, "@에이전트 에이전트의 권한과 역할 문서 찾아줘", "권한 역할 문서")
    assert args["query"] == "에이전트 권한 역할 문서"


@pytest.mark.asyncio
async def test_unrelated_query_is_untouched(monkeypatch):
    args = await _run(monkeypatch, "지난주 품질 이슈 찾아줘", "품질 이슈")
    assert args["query"] == "품질 이슈"


@pytest.mark.asyncio
async def test_empty_query_is_left_alone(monkeypatch):
    """주제어 없는 활동형 질문은 최근순 경로다 — 여기에 검색어를 만들어 넣으면 안 된다."""
    args = await _run(monkeypatch, "에이전트 채널 최근 무슨 얘기 있었어", "")
    assert args["query"] == ""
