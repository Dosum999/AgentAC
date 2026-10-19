"""agent_arch.agents.react.guide_loop (guide_loop_events) 단위 테스트.

LLM 응답이 있으면 그대로 신뢰하고, 빈 응답 케이스에서만 비상 fallback 메시지가
발동되는지, AUTO tool_mode 정책과 ProgressEvent 순서를 검증한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.react.guide_loop import guide_loop_events, _GUIDE_NO_RESPONSE_MSG
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.agents.tools.base import ToolResult
from agent_arch.llm.base import Message, StreamEvent, ToolCall


# ── 테스트 헬퍼 ──────────────────────────────────────────────────────────────

_MSGS = [Message(role="user", content="질문")]


def _text(content: str = "답변", usage: dict | None = None, finish_reason: str = "stop"):
    """텍스트 답변 회차 stream 시퀀스 (text → done) 팩토리."""
    async def _gen(*args, **kwargs):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason=finish_reason, usage=usage)
    return _gen


def _tool_calls(tool_calls: list[ToolCall], usage: dict | None = None):
    """도구 호출 회차 stream 시퀀스 (tool_calls → done(tool_calls)) 팩토리."""
    async def _gen(*args, **kwargs):
        yield StreamEvent(type="tool_calls", tool_calls=tool_calls)
        yield StreamEvent(type="done", finish_reason="tool_calls", usage=usage)
    return _gen


def _llm(*per_iter_gens) -> MagicMock:
    """각 stream() 호출마다 다음 generator 가 사용되도록 side_effect 로 주입한 LLM."""
    iter_gens = list(per_iter_gens)

    def _next_gen(*args, **kwargs):
        gen = iter_gens.pop(0)
        return gen(*args, **kwargs)

    llm = MagicMock()
    llm.stream = _next_gen
    llm.bind_tools = AsyncMock()
    return llm


def _make_guide_tool(name: str = "guide_kb", hits: list | None = None):
    """execute가 hits 리스트를 출력하는 모의 가이드 도구를 생성한다."""
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(
        return_value=ToolResult(success=True, output=hits if hits is not None else [])
    )
    return tool


def _make_guide_tool_meta(
    name: str = "guide_kb",
    hits: list | None = None,
    *,
    abstain: bool = False,
    abstain_reason: str | None = None,
):
    """abstain 신호를 metadata 에 담아 반환하는 모의 가이드 도구."""
    tool = MagicMock()
    tool.name = name
    tool.execute = AsyncMock(
        return_value=ToolResult(
            success=True,
            output=hits if hits is not None else [],
            metadata={"abstain": abstain, "abstain_reason": abstain_reason},
        )
    )
    return tool


def _tc(name: str = "guide_kb") -> ToolCall:
    return ToolCall(id=f"id_{name}", name=name, arguments={})


async def _run(llm, guide_tool=None, categories=None):
    """guide_loop_events 를 소비하여 (progress_events, loop_result) 튜플을 반환한다."""
    if guide_tool is None:
        guide_tool = _make_guide_tool()
    events = [
        e
        async for e in guide_loop_events(
            llm, "질문", list(_MSGS), guide_tool, categories=categories
        )
    ]
    progress = [e for e in events if isinstance(e, ProgressEvent)]
    results = [e for e in events if isinstance(e, LoopResult)]
    assert len(results) == 1, "LoopResult는 정확히 한 번 yield 되어야 한다"
    return progress, results[0]


# ── 최종 content 결정 ────────────────────────────────────────────────────────

class TestGuideLoopContent:
    """LLM 응답 유무에 따른 최종 content 결정 검증.

    KB 검색 결과 0건이어도 LLM 응답이 있으면 그대로 사용 (중복 사과 문구 방지).
    LLM 이 한 글자도 응답을 못 만든 경우에만 비상 fallback 발동.
    """

    @pytest.mark.asyncio
    async def test_inquiry_table_html_break_is_normalized_after_kb_search(self):
        name = "search_user_guide_kb"
        answer = (
            "| 문의 경로 | 안내 |\n| --- | --- |\n"
            "| 회원 문의 폼 | [문의 폼 바로가기](https://example.com/inquiry)"
            "<br>물음표 → 문의하기 메뉴 선택 |"
        )
        # 히트 본문에 그 주소가 실제로 있어야 링크가 살아남는다 — 가이드 턴에서는 "모델이 실제로
        # 본 텍스트에 있던 주소"만 답변에 남기기 때문(지어낸 가이드 링크 차단, react/link_policy.py).
        tool = _make_guide_tool(name, hits=["고객센터 문의 방법: https://example.com/inquiry"])
        llm = _llm(_tool_calls([_tc(name)]), _text(answer))
        _, result = await _run(llm, tool)
        assert result.content == answer.replace("<br>", " ")
        assert result.guide_hits == 1
        tool.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_guide_hits_positive_uses_llm_content(self):
        """KB 검색 히트가 있으면 LLM이 생성한 content를 반환한다."""
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool("guide_kb", hits=["문서1", "문서2"])
        llm = _llm(
            _tool_calls([tc]),
            _text("안내 내용입니다."),
        )
        _, result = await _run(llm, guide_tool)
        assert result.content == "안내 내용입니다."
        assert result.guide_hits == 2

    @pytest.mark.asyncio
    async def test_guide_hits_zero_with_llm_content_uses_llm_content(self):
        """KB 히트가 0건이어도 LLM 응답이 있으면 그대로 반환한다.

        시스템 프롬프트가 빈 결과 대응을 이미 가르치므로 (prompts/system.py
        L369-383), 그 위에 사과 문구를 덧붙이지 않는다.
        """
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool("guide_kb", hits=[])
        llm = _llm(
            _tool_calls([tc]),
            _text("LLM이 생성한 안내 내용"),
        )
        _, result = await _run(llm, guide_tool)
        assert result.content == "LLM이 생성한 안내 내용"
        assert _GUIDE_NO_RESPONSE_MSG not in result.content
        assert result.guide_hits == 0

    @pytest.mark.asyncio
    async def test_no_tool_calls_uses_llm_content(self):
        """LLM이 도구를 호출하지 않고 바로 답해도 그 응답을 그대로 반환한다.

        "안녕" 같은 trivial 질문 시나리오. tool_mode=AUTO 로 검색 강제 해제됨.
        """
        llm = _llm(_text("안녕하세요! 무엇을 도와드릴까요?"))
        _, result = await _run(llm)
        assert result.content == "안녕하세요! 무엇을 도와드릴까요?"
        assert _GUIDE_NO_RESPONSE_MSG not in result.content
        assert result.guide_hits == 0

    @pytest.mark.asyncio
    async def test_empty_llm_content_falls_back(self):
        """LLM 텍스트가 끝까지 비면 비상 fallback 메시지로 대체된다.

        malformed function call 반복, tool_call 만 반복 등 극단 케이스에서
        NONE 강제 호출까지 빈 응답이면 _GUIDE_NO_RESPONSE_MSG 발동.
        """
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool("guide_kb", hits=["문서1"])
        # iter 0: tool_calls → guide_hits=1
        # iter 1: 빈 content + stop → 루프 break, text_buf 빈 상태
        # finish_reason="stop" 이라 NONE 강제 호출 진입 안 함 → final_content == ""
        llm = _llm(
            _tool_calls([tc]),
            _text("", finish_reason="stop"),
        )
        _, result = await _run(llm, guide_tool)
        assert result.guide_hits == 1
        assert result.content == _GUIDE_NO_RESPONSE_MSG

    @pytest.mark.asyncio
    async def test_token_usage_accumulated(self):
        """여러 번 stream 호출한 token_usage가 누적된다."""
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool("guide_kb", hits=["doc"])
        usage_1 = {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12}
        usage_2 = {"prompt_tokens": 2, "completion_tokens": 5, "total_tokens": 7}
        llm = _llm(
            _tool_calls([tc], usage=usage_1),
            _text("답변", usage=usage_2),
        )
        _, result = await _run(llm, guide_tool)
        assert result.token_usage["prompt_tokens"] == 6
        assert result.token_usage["completion_tokens"] == 13


# ── abstain(에이전트 모드 전환 제안) 신호 전파 검증 ──────────────────────────

class TestGuideLoopAbstain:
    """KB 검색의 abstain 신호가 LoopResult 로 전파되는지 검증.

    가이드로 답할 수 없는 질문(abstain=True)이면 상위(orchestrator)가
    suggested_mode="agent" 로 승격해 "AI 에이전트 모드로 전환할까요?" 를 제안한다.
    """

    @pytest.mark.asyncio
    async def test_abstain_true_propagates_to_loop_result(self):
        """검색이 기권하면 LoopResult.abstain=True + 사유가 전파된다."""
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool_meta(
            "guide_kb", hits=[], abstain=True, abstain_reason="llm_judged_unrelated"
        )
        llm = _llm(
            _tool_calls([tc]),
            _text("죄송하지만 가이드로는 안내가 어려워요."),
        )
        _, result = await _run(llm, guide_tool)
        assert result.abstain is True
        assert result.abstain_reason == "llm_judged_unrelated"

    @pytest.mark.asyncio
    async def test_abstain_false_when_search_confident(self):
        """검색이 기권하지 않으면 LoopResult.abstain=False."""
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool_meta(
            "guide_kb", hits=["문서1"], abstain=False, abstain_reason=None
        )
        llm = _llm(
            _tool_calls([tc]),
            _text("안내드릴게요."),
        )
        _, result = await _run(llm, guide_tool)
        assert result.abstain is False
        assert result.abstain_reason is None

    @pytest.mark.asyncio
    async def test_no_search_leaves_abstain_default_false(self):
        """검색을 아예 하지 않은(인사 등) 턴은 abstain 기본값 False 로 남는다."""
        llm = _llm(_text("안녕하세요! 무엇을 도와드릴까요?"))
        _, result = await _run(llm)
        assert result.abstain is False
        assert result.abstain_reason is None

    @pytest.mark.asyncio
    async def test_agent_hint_marker_promotes_to_abstain_and_is_stripped(self):
        """LLM이 응답 끝에 NEED_AGENT 마커를 붙이면 abstain=True + 마커는 content에서 제거된다.

        검색이 기권하지 않아도(가이드에 관련 안내가 있어도) 실제 데이터/작업 요청이면
        마커로 에이전트 모드 전환을 제안한다.
        """
        from agent_arch.utils.sanitize import GUIDE_AGENT_HINT

        tc = _tc("guide_kb")
        # 검색은 자신있게 매칭(abstain=False)이지만, LLM이 데이터 요청으로 판단해 마커 부착
        guide_tool = _make_guide_tool_meta(
            "guide_kb", hits=["연차 가이드"], abstain=False, abstain_reason=None
        )
        llm = _llm(
            _tool_calls([tc]),
            _text(f"남은 연차는 HR 서비스에서 확인하실 수 있어요.\n{GUIDE_AGENT_HINT}"),
        )
        _, result = await _run(llm, guide_tool)
        assert result.abstain is True
        assert result.abstain_reason == "needs_agent"
        assert GUIDE_AGENT_HINT not in result.content
        assert result.content == "남은 연차는 HR 서비스에서 확인하실 수 있어요."


# ── ProgressEvent 순서 검증 ───────────────────────────────────────────────────

class TestGuideLoopProgressEvents:
    """guide_loop_events 의 ProgressEvent 순서 검증."""

    @pytest.mark.asyncio
    async def test_thinking_emitted_first(self):
        """첫 번째 이벤트는 항상 ProgressEvent("thinking")다."""
        llm = _llm(_text("답변"))
        progress, _ = await _run(llm)
        assert len(progress) >= 1
        assert progress[0].event_type == "thinking"

    @pytest.mark.asyncio
    async def test_searching_emitted_on_first_iteration(self):
        """iter=0에서 ProgressEvent("searching")가 yield된다."""
        llm = _llm(_text("답변"))
        progress, _ = await _run(llm)
        event_types = [e.event_type for e in progress]
        assert "searching" in event_types

    @pytest.mark.asyncio
    async def test_token_events_emitted_during_answer(self):
        """텍스트 답변 회차에서 ProgressEvent('token') 이 발생한다."""
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool("guide_kb", hits=["doc"])
        llm = _llm(
            _tool_calls([tc]),
            _text("안내 내용"),
        )
        progress, result = await _run(llm, guide_tool)
        token_events = [e for e in progress if e.event_type == "token"]
        assert token_events
        assert "안내 내용" in "".join(e.content for e in token_events)
        assert result.content == "안내 내용"

    @pytest.mark.asyncio
    async def test_categories_passed_to_tool_execute(self):
        """``categories`` 인자가 ``tool.execute`` 에 ``_categories`` 키로 전달된다."""
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool("guide_kb", hits=["doc"])
        llm = _llm(
            _tool_calls([tc]),
            _text("안내 내용"),
        )
        _, _ = await _run(llm, guide_tool, categories=["Drive"])
        guide_tool.execute.assert_awaited_once()
        assert guide_tool.execute.await_args.kwargs == {"_categories": ["Drive"]}

    @pytest.mark.asyncio
    async def test_categories_none_passed_when_omitted(self):
        """``categories`` 미전송 시 ``tool.execute`` 에 ``_categories=None`` 으로 전달."""
        tc = _tc("guide_kb")
        guide_tool = _make_guide_tool("guide_kb", hits=["doc"])
        llm = _llm(
            _tool_calls([tc]),
            _text("안내"),
        )
        _, result = await _run(llm, guide_tool)
        guide_tool.execute.assert_awaited_once()
        assert guide_tool.execute.await_args.kwargs == {"_categories": None}
        assert isinstance(result, LoopResult)


# ── 최종 content 정화 (도구명 마스킹 + 마커 소비 순서) ───────────────────────

class TestGuideLoopSanitize:
    """가이드 최종 content 도 react 루프와 동일하게 도구명이 마스킹되어야 한다.

    정화 위치는 GUIDE_AGENT_HINT 소비 **이후**여야 한다 — sanitize 가 마커를 제거하므로
    앞에 두면 abstain(→ suggested_mode="agent") 승격이 조용히 죽는다.
    """

    @pytest.mark.asyncio
    async def test_tool_name_masked_in_final_content(self):
        llm = _llm(_text("hr_service_attendance_read 로 확인하세요."))
        _, result = await _run(llm)
        assert "hr_service_attendance_read" not in result.content
        assert "내부 도구" in result.content

    @pytest.mark.asyncio
    async def test_marker_consumed_before_sanitize_strips_it(self):
        """마커가 붙어도 abstain 승격이 살아 있고, 표시 텍스트에는 마커가 없다."""
        from agent_arch.utils.sanitize import GUIDE_AGENT_HINT

        llm = _llm(_text("실제 데이터 조회가 필요합니다." + GUIDE_AGENT_HINT))
        _, result = await _run(llm)
        assert result.abstain is True
        assert result.abstain_reason == "needs_agent"
        assert GUIDE_AGENT_HINT not in result.content
        assert "실제 데이터 조회가 필요합니다." in result.content


# ── 근거 없는 가이드 링크 차단(화면 = 저장본) ─────────────────────────────────

class TestGuideLinkPolicy:
    """2026-09-21 신고: 가이드 모드 답변 말미에 **지어낸 주소**의 가이드 문서 링크가 붙었다."""

    _FORM = "https://example.com/main/support/inquiry/write"
    _FAKE = "https://example.com/user-guide/agent-deep-dive"

    @pytest.mark.asyncio
    async def test_fabricated_links_and_orphan_leadin_are_dropped(self):
        name = "search_user_guide_kb"
        answer = ("에이전트 사용 방법입니다.\n\n"
                  f"문의는 [문의 폼]({self._FORM}) 에서 가능합니다.\n\n"
                  f"더 자세한 내용은 아래 가이드를 참고해 주세요. [Agent 사용 심층 가이드]({self._FAKE})\n")
        tool = _make_guide_tool(name, hits=[f"회원 문의 폼( {self._FORM} )을 이용하세요."])
        llm = _llm(_tool_calls([_tc(name)]), _text(answer))
        events, result = await _run(llm, tool)

        assert self._FAKE not in result.content
        assert "가이드를 참고해 주세요" not in result.content
        assert self._FORM in result.content                 # 검색 결과에 있던 실무 주소는 유지

        # 화면(정책을 먹인 스트리밍 정화) == 저장본. 정책 이벤트는 토큰보다 먼저 나간다.
        from agent_arch.agents.react.link_policy import TurnLinkPolicy
        from agent_arch.utils.sanitize import StreamSanitizer

        kinds = [e.event_type for e in events]
        assert kinds.index("link_policy") < kinds.index("token")
        policy = TurnLinkPolicy()
        sanitizer = StreamSanitizer(link_permits=policy.permits)
        screen = []
        for e in events:
            if e.event_type == "link_policy":
                policy.update_from(e.content)
            elif e.event_type == "token":
                screen.append(sanitizer.feed(e.content))
        screen.append(sanitizer.flush())
        assert "".join(screen).rstrip() == result.content.rstrip()
        assert self._FAKE not in "".join(screen)
