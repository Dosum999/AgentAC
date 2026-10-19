"""소스 우선순위 — 메신저 먼저, 드라이브는 신호가 있을 때.

왜 이 규칙이 생겼나 (2026-09-15 실측)
-------------------------------------
"이영희 이사님이 에이전트 정책 문서라고 공유한거 찾아줘" 에 라우터가 talk·drive 를 **둘 다**
골랐고, 병렬 실행 뒤 합성 단계에서 드라이브가 물어온 ``TD4 화면설계서.hwp`` 가 답변 본문을
차지했다. 메신저 워커는 정답 글을 찾아냈는데도 각주로 밀렸다. 사용자 눈에는 "메신저라고
말해야만 메신저를 본다" 로 보인다.

규칙은 셋이다.
    1. 대화·논의·공유 성격이면 talk 단독 (라우터 프롬프트 — 여기서는 검증 대상 아님)
    2. talk+drive 조회쌍이면 **drive 를 미루고**, 메신저가 근거를 못 찾았을 때만 돌린다
    3. 합성할 때 메신저 결과를 **먼저** 놓는다(본문 ↔ 참고 자료)

2번은 **좁게** 적용해야 한다. 드라이브가 실행을 맡았거나 사용자가 파일·폴더를 직접
가리켰으면 미루지 않는다 — 미루면 사용자가 요청한 일을 안 한 것이 되고, "근거 없음" 판정이
틀리면 결과가 영영 나오지 않는다. 그 경계가 이 파일의 대부분이다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.graphs import supervisor as _sup
from agent_arch.agents.graphs.supervisor import run_supervisor
from agent_arch.llm.base import LLMResponse, Message, StreamEvent
from agent_arch.agents.tools.base import ToolResult
from agent_arch.agents.graphs.router import RoutePlan
from agent_arch.agents.react.types import LoopResult


def _plan(domains, subqueries=None, actions=None) -> RoutePlan:
    return RoutePlan(domains=list(domains), subqueries=subqueries or {},
                     actions=actions or {}, confidence=1.0)


# ── 미루기 판정 ──────────────────────────────────────────────────────────────

def test_drive_is_deferred_for_conversational_question():
    plan = _plan(["talk", "drive"], {"talk": "공유한 링크 찾기", "drive": "관련 문서 찾기"})
    now, later = _sup._split_deferred(plan, "이영희 이사님이 에이전트 정책 문서라고 공유한거 찾아줘")
    assert now == ["talk"] and later == ["drive"]


@pytest.mark.parametrize("utterance", [
    "드라이브에서 품질 보고서 찾아줘",
    "전사 공용 문서 폴더 보여줘",
    "지난달 정산 xlsx 어디 있어?",
    "그 피피티 요약해줘",
    "계약서 양식 좀 줘",
])
def test_explicit_file_signals_keep_drive_in_the_first_wave(utterance):
    """사용자가 파일·폴더를 직접 가리켰으면 미루지 않는다 — 그건 드라이브 질문이다."""
    plan = _plan(["talk", "drive"])
    now, later = _sup._split_deferred(plan, utterance)
    assert set(now) == {"talk", "drive"} and later == []


def test_document_word_alone_is_not_a_drive_signal():
    """'문서'·'자료' 는 메신저에 공유된 링크에도 붙는 말이라 신호로 치지 않는다."""
    assert _sup._has_drive_signal("관련 문서 있어?") is False
    assert _sup._has_drive_signal("자료 좀 찾아줘") is False
    assert _sup._has_drive_signal("드라이브 자료") is True


def test_drive_action_is_never_deferred():
    """드라이브가 실행(공유링크 생성 등)을 맡았으면 미루면 요청을 안 한 것이 된다."""
    plan = _plan(["talk", "drive"], {"talk": "논의 정리"}, {"drive": "공유 링크 만들어줘"})
    now, later = _sup._split_deferred(plan, "그 논의 정리하고 공유 링크도 만들어줘")
    assert set(now) == {"talk", "drive"} and later == []


def test_other_domain_pairs_are_untouched():
    """메일·일정처럼 성격이 다른 도메인은 동시에 보는 것이 정상이다."""
    plan = _plan(["mail", "calendar"])
    now, later = _sup._split_deferred(plan, "메일 확인하고 일정도 알려줘")
    assert now == ["mail", "calendar"] and later == []


def test_drive_alone_is_untouched():
    plan = _plan(["drive"])
    now, later = _sup._split_deferred(plan, "그 파일 요약해줘")
    assert now == ["drive"] and later == []


# ── 근거 판정 ────────────────────────────────────────────────────────────────

def test_found_something_requires_tool_evidence():
    """문장만 보면 안 된다 — 도구가 0건이어도 모델은 '없습니다' 라는 문장을 만든다."""
    assert _sup._worker_found_something(
        LoopResult(content="찾았습니다", tool_results=[{"x": 1}], finish_reason="stop")) is True
    assert _sup._worker_found_something(
        LoopResult(content="관련 내용을 찾지 못했습니다", tool_results=[], finish_reason="stop")) is False
    assert _sup._worker_found_something(
        LoopResult(content="", tool_results=[{"x": 1}], finish_reason="stop")) is False
    assert _sup._worker_found_something(None) is False


# ── 합성 순서 ────────────────────────────────────────────────────────────────

def test_messenger_result_leads_the_synthesis():
    """드라이브 결과가 앞에 오면 합성기가 그걸 본문으로 삼는다(실측 사고)."""
    ok = [("drive", LoopResult(content="드라이브 파일 목록", finish_reason="stop")),
          ("talk", LoopResult(content="메신저에서 찾은 글", finish_reason="stop"))]
    msgs = _sup._synth_messages("공유한 문서 찾아줘", ok)
    body = msgs[-1].content
    assert body.index("메신저에서 찾은 글") < body.index("드라이브 파일 목록")
    assert "참고 자료" in msgs[0].content


# ── 실행 흐름(run_supervisor) ────────────────────────────────────────────────

def _stream_text(text: str, tool_calls=None):
    """워커 LLM 스트림 더블 — 도구를 부르고 답을 내거나, 바로 답만 낸다."""
    async def g(*a, **k):
        if tool_calls:
            yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
            yield StreamEvent(type="done", finish_reason="tool_calls", usage=None)
        else:
            yield StreamEvent(type="text", content=text)
            yield StreamEvent(type="done", finish_reason="stop", usage=None)
    return g


def _worker_llm(per_call_specs):
    calls = {"n": 0, "domains": []}

    def make():
        calls["n"] += 1
        gens = list(per_call_specs())
        llm = MagicMock()
        llm.stream = lambda *a, **k: gens.pop(0)(*a, **k)
        llm.bind_tools = AsyncMock()
        return llm

    make.calls = calls
    return make


def _router(json_text: str):
    m = MagicMock()
    m.invoke = AsyncMock(return_value=LLMResponse(content=json_text, finish_reason="stop"))
    return m


def _tool(name: str, output="찾은 결과"):
    t = MagicMock()
    t.name = name
    t.execute = AsyncMock(return_value=ToolResult(success=True, output=output))
    return t


@pytest.mark.asyncio
async def test_drive_worker_is_skipped_when_messenger_found_evidence():
    """메신저가 근거를 가져오면 드라이브는 **아예 돌지 않는다** — 답이 깨끗해지고 호출도 준다."""
    router = _router('{"domains":["talk","drive"],'
                     '"subqueries":{"talk":"공유한 글 찾기","drive":"관련 문서"},"confidence":0.9}')
    resolved: list = []

    async def resolve(subquery, domain):
        resolved.append(domain)
        return [_tool(f"{domain}_search")]

    def specs():
        return [_stream_text("", tool_calls=[{"id": "1", "name": "talk_search", "arguments": {}}]),
                _stream_text("메신저에서 찾았습니다")]

    main = MagicMock()
    main.invoke = AsyncMock()          # synth 는 불릴 일이 없다(단일 결과)
    main.bind_tools = AsyncMock()
    r = await run_supervisor(
        main, "이영희 이사님이 공유한거 찾아줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_worker_llm(specs),
        fallback_fn=AsyncMock(), router_llm=router, max_iterations=3,
    )
    assert "메신저에서 찾았습니다" in r.content
    assert "drive" not in resolved      # 드라이브 워커 자체가 안 떴다
    main.invoke.assert_not_called()


@pytest.mark.asyncio
async def test_drive_runs_when_messenger_found_nothing():
    """메신저가 빈손이면 드라이브로 넓힌다 — 재현율을 포기하지 않는다."""
    router = _router('{"domains":["talk","drive"],'
                     '"subqueries":{"talk":"공유한 글 찾기","drive":"관련 문서"},"confidence":0.9}')
    resolved: list = []

    async def resolve(subquery, domain):
        resolved.append(domain)
        return [_tool(f"{domain}_search")]

    def specs():
        # 메신저 워커: 도구 없이 "못 찾았다" 만 답한다(tool_results 가 비어 근거 없음).
        return [_stream_text("관련 내용을 찾지 못했습니다"),
                _stream_text("", tool_calls=[{"id": "1", "name": "drive_search", "arguments": {}}]),
                _stream_text("드라이브에서 찾았습니다")]

    main = MagicMock()
    main.invoke = AsyncMock(return_value=LLMResponse(content="통합 답변", finish_reason="stop",
                                                     usage={"total_tokens": 1}))
    main.bind_tools = AsyncMock()
    await run_supervisor(
        main, "그 내용 찾아줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_worker_llm(specs),
        fallback_fn=AsyncMock(), router_llm=router, max_iterations=3,
    )
    assert resolved == ["talk", "drive"]   # 순서대로 — 메신저 먼저, 그다음 드라이브


# ── 대화를 찾는 질문에서 '드라이브'는 주제어 (2026-10-01 실사용) ──────────────
# "confluence → DMP 드라이브 전환 얘기한 것 찾아줘" 가 드라이브와 병렬로 돌아, 합성 단계에서
# 메신저 후보 목록이 한 문단으로 뭉개졌다.

_ORIGINAL = "내가 이전에 confluence 관련해서 DMP 드라이브 로 전환에 관련해서 이야기 한 것이 있으면 찾아줘"


def test_drive_as_topic_of_a_conversation_is_deferred():
    plan = _plan(["talk", "drive"], {"talk": "DMP 드라이브 전환 논의", "drive": "DMP 드라이브 전환 문서"})
    now, later = _sup._split_deferred(plan, _ORIGINAL)
    assert now == ["talk"] and later == ["drive"]


@pytest.mark.parametrize("utterance", [
    "드라이브에서 찾고 메신저에서 얘기한 것도 정리해줘",       # 위치
    "그때 얘기한 정산 xlsx 찾아줘",                           # 확장자
])
def test_real_file_signal_still_wins_in_a_conversation_question(utterance):
    plan = _plan(["talk", "drive"])
    now, later = _sup._split_deferred(plan, utterance)
    assert set(now) == {"talk", "drive"} and later == []


def test_synth_keeps_numbered_list():
    """합성기는 담당 결과의 번호 목록(건별 출처)을 문단으로 풀지 않는다."""
    msgs = _sup._synth_messages("찾아줘", [("talk", LoopResult(content="1. a"))])
    assert "그 목록을 문단으로 풀지 말고" in msgs[0].content
