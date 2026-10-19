"""방법·설정 질문은 첫 iteration 에 가이드 검색을 강제한다 (guide-first).

배경(2026-09-02 실측): 같은 대화의 후속 질문("07:40 출근 연장 20분 없애는 방법은?")에서
Gemini 가 도구를 한 번도 부르지 않고(iterations=0) 앞 턴 맥락만으로 답해 근태 설정
안내를 놓쳤다. 검색만 했으면 근태 설정 청크가 상위에 잡혔다.

방어: 사용자 입력이 방법·절차·설정·기능 유무·증상 질문이고 ``search_user_guide_kb`` 가
바인딩돼 있으면, 첫 call_model 만 tool_mode=ANY + allowed_tools=[가이드 도구] 로 호출해
검색을 건너뛸 수 없게 한다. 실행 요청("~해줘")은 강제하지 않는다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.llm.base import Message, StreamEvent, ToolCall
from agent_arch.agents.tools.base import ToolResult
from agent_arch.agents.react.howto import is_howto_question, pick_forced_tool, GUIDE_TOOL_NAME
from agent_arch.agents.graphs.react_graph import run_react_graph


# ── 1) 질문 유형 판별 ────────────────────────────────────────────────────────

REAL_MISSED = [
    "그럼 시급제의 경우 정상 출근이 08:00 인데 07:40분에 출근해서 연장근무수당이 20분 잡히는데 그걸 없애는방법은?",
    "그럼 개개인별 하루 하루 선택해서 수정을 해야하나요? ",
    "출퇴근은 조기출근으로 찍히되 근무요약을 고정으로 시간 지정을 할수있는 방법은 없나요?",
    "8시부터 근무인데 7시55분 출근으로 찍히면 5분 연장으로 급여가 계산되는데 그걸 없애는 방법",
    "인사 사원별 근무현황에서 미승인근태에서 승인된 근태로 넘어가는건 자동으로 이뤄지는게 아니라 개개인별 승인을 해야하는가요?",
]


@pytest.mark.parametrize("q", REAL_MISSED)
def test_howto_detects_real_missed_questions(q):
    assert is_howto_question(q) is True


@pytest.mark.parametrize("q", [
    "[Mattermost 메시지] 요청자: 신희섬  근무조 변경 신청은 어떻게 해?",
    # 개발서버 실측(2026-09-04): 이 질문은 find_hr_tools(화면 열기)로 새어 DMP 권한 설정을 안내했다
    "근태 설정은 어디서 해?",
    "연장근무 가산율은 어디서 설정해?",
    "급여명세서에 찍히는 법인 인감은 어디서 등록해?",
    "휴가 신청 어떻게 해?",
    "메일 서명은 어디서 설정해요?",
    "연차 규정이 뭐예요?",
    "개인 채널이 갑자기 회사 채널 탭으로 옮겨졌어요 왜 이렇게 돼요?",
])
def test_howto_detects_method_policy_symptom(q):
    assert is_howto_question(q) is True


@pytest.mark.parametrize("q", [
    "휴가 신청해줘",
    "김철수한테 회의 자료 보내줘",
    "이번 달 내 근태 조회해줘",
    "응 고마워",
    "네",
    "오늘 날씨",
    '{"goto": [], "graph": null, "resume": {"title": "경비청구 어떻게"}, "update": null}',
])
def test_howto_rejects_actions_and_smalltalk(q):
    assert is_howto_question(q) is False


def test_pick_forced_tool_requires_guide_tool_bound():
    assert pick_forced_tool("연장 없애는 방법은?", {GUIDE_TOOL_NAME, "hr_x"}) == GUIDE_TOOL_NAME
    assert pick_forced_tool("연장 없애는 방법은?", {"hr_x"}) is None
    assert pick_forced_tool("휴가 신청해줘", {GUIDE_TOOL_NAME}) is None


# ── 2) 그래프: 첫 call_model 만 강제, 이후는 AUTO ─────────────────────────────

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


def _tool(name):
    t = MagicMock()
    t.name = name
    t.execute = AsyncMock(return_value=ToolResult(success=True, output="가이드 결과"))
    return t


@pytest.mark.asyncio
async def test_graph_forces_guide_tool_on_first_call_for_howto():
    calls = []
    q = "7시55분 출근으로 찍히면 5분 연장으로 계산되는데 그걸 없애는 방법"
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=GUIDE_TOOL_NAME, arguments={"query": q})]),
        _stream_text("근태 설정에서 체크하세요."),
    ], calls)
    result = await run_react_graph(
        llm, q, [Message(role="user", content=q)],
        [_tool(GUIDE_TOOL_NAME), _tool("hr_attendance_query")],
        AsyncMock(return_value=[]), 5,
    )
    assert result.content == "근태 설정에서 체크하세요."
    assert calls[0]["tool_mode"] == "ANY"
    assert calls[0]["allowed_tools"] == [GUIDE_TOOL_NAME]
    # 두 번째 호출(검색 결과 반영 후)은 평소대로 AUTO, 도구 제한 없음
    assert calls[1]["tool_mode"] == "AUTO"
    assert "allowed_tools" not in calls[1] or calls[1]["allowed_tools"] is None
    assert result.step_meta["guide_first"] is True


@pytest.mark.asyncio
async def test_graph_does_not_force_for_action_request():
    calls = []
    q = "휴가 신청해줘"
    llm = _make_llm([_stream_text("신청 도와드릴게요.")], calls)
    result = await run_react_graph(
        llm, q, [Message(role="user", content=q)],
        [_tool(GUIDE_TOOL_NAME), _tool("leave_apply")],
        AsyncMock(return_value=[]), 5,
    )
    assert calls[0]["tool_mode"] == "AUTO"
    assert not calls[0].get("allowed_tools")
    assert result.step_meta["guide_first"] is False


@pytest.mark.asyncio
async def test_graph_does_not_force_when_guide_tool_absent():
    calls = []
    q = "연장 없애는 방법은?"
    llm = _make_llm([_stream_text("답")], calls)
    await run_react_graph(
        llm, q, [Message(role="user", content=q)], [_tool("hr_only")],
        AsyncMock(return_value=[]), 5,
    )
    assert calls[0]["tool_mode"] == "AUTO"
    assert not calls[0].get("allowed_tools")


@pytest.mark.asyncio
async def test_forced_guide_search_uses_user_question_as_query():
    """운영 실측(2026-09-04): LLM 이 '시급제 연장근무수당 조정' 같은 짧은 검색어를 만들면 급여 문단만
    잡히고, 질문 문장을 그대로 검색하면 근태 설정 문단이 1위였다. 강제 턴에서는 질문 문장을 검색어로 쓴다."""
    calls = []
    q = "그럼 시급제의 경우 정상 출근이 08:00 인데 07:40분에 출근해서 연장근무수당이 20분 잡히는데 그걸 없애는방법은?"
    guide = _tool(GUIDE_TOOL_NAME)
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=GUIDE_TOOL_NAME, arguments={"query": "시급제 연장근무수당 조정"})]),
        _stream_text("근태 설정에서 체크하세요."),
    ], calls)
    await run_react_graph(llm, q, [Message(role="user", content=q)], [guide], AsyncMock(return_value=[]), 5)
    assert guide.execute.await_args.kwargs["query"] == q


@pytest.mark.asyncio
async def test_non_forced_guide_search_keeps_llm_query():
    """강제가 아닌 일반 턴(실행 요청 등)에서는 LLM 검색어를 건드리지 않는다."""
    calls = []
    q = "휴가 신청해줘"
    guide = _tool(GUIDE_TOOL_NAME)
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name=GUIDE_TOOL_NAME, arguments={"query": "휴가 신청 절차"})]),
        _stream_text("답"),
    ], calls)
    await run_react_graph(llm, q, [Message(role="user", content=q)], [guide], AsyncMock(return_value=[]), 5)
    assert guide.execute.await_args.kwargs["query"] == "휴가 신청 절차"


# ── 2026-10-13: 서브쿼리에서 유실된 guide 신호를 턴 원문으로 복구 (turn_input_guide) ─────────


@pytest.mark.asyncio
async def test_graph_forces_guide_by_turn_input_when_subquery_lost_signal():
    """라우터 재작성 서브쿼리엔 신호가 없어도 턴 원문이 howto 면 강제하고, 검색어도 원문을 쓴다."""
    from agent_arch.agents import context as ctx
    calls = []
    sub = "회사 채널 검색 및 목록 조회"                       # 9/28 실측 재작성 — 정규식 미매치
    raw = "채널 탐색에서 회사 채널이 검색이 안 돼요"            # g014 원문 — 매치
    ctx.set_turn_input(raw)
    tools = [_tool(GUIDE_TOOL_NAME), _tool("mattermost_skill_find_channels")]
    try:
        llm = _make_llm([
            _stream_tool_calls([ToolCall(id="c1", name=GUIDE_TOOL_NAME, arguments={"query": sub})]),
            _stream_text("회사 채널은 탐색에 안 나와요."),
        ], calls)
        result = await run_react_graph(llm, sub, [Message(role="user", content=sub)], tools,
                                       AsyncMock(return_value=[]), 5, turn_input_guide=True)
    finally:
        ctx.set_turn_input(None)
    assert calls[0]["tool_mode"] == "ANY"
    assert calls[0]["allowed_tools"] == [GUIDE_TOOL_NAME]
    assert result.step_meta["guide_first"] is True
    assert tools[0].execute.await_args.kwargs["query"] == raw


@pytest.mark.asyncio
async def test_graph_raw_fallback_replaces_query_even_when_original_is_shorter():
    """원문 폴백 턴은 원문이 서브쿼리·모델 검색어보다 짧아도 검색어를 원문으로 바꾼다 (길이 비교 우회)."""
    from agent_arch.agents import context as ctx
    calls = []
    sub = "회사 채널 검색 및 목록 조회"           # 16자, 정규식 미매치
    raw = "회사 채널 안 보여요"                  # 11자, 매치
    ctx.set_turn_input(raw)
    tools = [_tool(GUIDE_TOOL_NAME), _tool("mattermost_skill_find_channels")]
    try:
        llm = _make_llm([
            _stream_tool_calls([ToolCall(id="c1", name=GUIDE_TOOL_NAME, arguments={"query": sub})]),
            _stream_text("회사 채널은 탐색에 안 나와요."),
        ], calls)
        result = await run_react_graph(llm, sub, [Message(role="user", content=sub)], tools,
                                       AsyncMock(return_value=[]), 5, turn_input_guide=True)
    finally:
        ctx.set_turn_input(None)
    assert result.step_meta["guide_first"] is True
    assert tools[0].execute.await_args.kwargs["query"] == raw


@pytest.mark.asyncio
async def test_graph_subquery_match_keeps_existing_query_rule():
    """서브쿼리가 스스로 howto 면 종전 규칙 그대로 — 원문은 보지 않고, 검색어는 '더 길면 서브쿼리'."""
    from agent_arch.agents import context as ctx
    calls = []
    sub = "회사 채널이 검색이 안 되는 원인 확인 방법"
    ctx.set_turn_input("회사 채널 안 보여요")
    tools = [_tool(GUIDE_TOOL_NAME)]
    try:
        llm = _make_llm([
            _stream_tool_calls([ToolCall(id="c1", name=GUIDE_TOOL_NAME, arguments={"query": "채널 검색"})]),
            _stream_text("답"),
        ], calls)
        result = await run_react_graph(llm, sub, [Message(role="user", content=sub)], tools,
                                       AsyncMock(return_value=[]), 5, turn_input_guide=True)
    finally:
        ctx.set_turn_input(None)
    assert result.step_meta["guide_first"] is True
    assert tools[0].execute.await_args.kwargs["query"] == sub


@pytest.mark.asyncio
async def test_graph_turn_input_does_not_force_action_requests():
    """원문이 실행 요청이면 턴 원문 경로로도 강제하지 않는다."""
    from agent_arch.agents import context as ctx
    calls = []
    sub = "휴가 신청"
    ctx.set_turn_input("휴가 신청해줘")
    try:
        llm = _make_llm([_stream_text("신청 도와드릴게요.")], calls)
        result = await run_react_graph(llm, sub, [Message(role="user", content=sub)],
                                       [_tool(GUIDE_TOOL_NAME), _tool("leave_apply")],
                                       AsyncMock(return_value=[]), 5, turn_input_guide=True)
    finally:
        ctx.set_turn_input(None)
    assert calls[0]["tool_mode"] == "AUTO"
    assert result.step_meta["guide_first"] is False


@pytest.mark.asyncio
async def test_graph_turn_input_ignored_unless_caller_allows():
    """호출자가 turn_input_guide 를 안 켜면(기본) 원문을 보지 않는다 — 복합 발화의 메일 워커 보호."""
    from agent_arch.agents import context as ctx
    calls = []
    sub = "김철수에게 메일 보내기"
    ctx.set_turn_input("회사 채널 찾는 방법 알려주고 김철수에게 메일 보내줘")
    try:
        llm = _make_llm([_stream_text("메일 초안을 준비했어요.")], calls)
        result = await run_react_graph(llm, sub, [Message(role="user", content=sub)],
                                       [_tool(GUIDE_TOOL_NAME), _tool("mail_send_message")],
                                       AsyncMock(return_value=[]), 5)
    finally:
        ctx.set_turn_input(None)
    assert calls[0]["tool_mode"] == "AUTO"
    assert result.step_meta["guide_first"] is False


@pytest.mark.asyncio
async def test_guide_first_flag_false_for_thread_or_author_forcing(monkeypatch):
    """thread-first/author-first 강제는 guide_first 로 집계되지 않는다 (2026-10-13 교정)."""
    import agent_arch.agents.graphs.react_graph as rg
    monkeypatch.setattr(rg, "pick_thread_tool", lambda text, known, root: "mattermost_skill_read_thread")
    calls = []
    q = "이 스레드 요약해줘"
    llm = _make_llm([
        _stream_tool_calls([ToolCall(id="c1", name="mattermost_skill_read_thread", arguments={})]),
        _stream_text("요약입니다."),
    ], calls)
    result = await run_react_graph(llm, q, [Message(role="user", content=q)],
                                   [_tool(GUIDE_TOOL_NAME), _tool("mattermost_skill_read_thread")],
                                   AsyncMock(return_value=[]), 5)
    assert calls[0]["allowed_tools"] == ["mattermost_skill_read_thread"]
    assert result.step_meta["guide_first"] is False


@pytest.mark.parametrize("single_domain,read_q,action_q,expected", [
    (True, "회사 채널 검색", None, True),            # 단일 도메인 · read 만
    (True, None, "메시지 보내기", True),             # 단일 도메인 · write 만
    (True, "회사 채널 찾는 방법", "메시지 보내기", False),  # 같은 도메인 read+write — 서브런 2개
    (False, "회사 채널 검색", None, False),          # 복합 발화(도메인 2개 이상)
    (False, None, None, False),
])
def test_turn_input_guide_allowed_only_for_single_domain_single_subrun(single_domain, read_q, action_q, expected):
    from agent_arch.agents.graphs.supervisor import _turn_input_guide_allowed
    assert _turn_input_guide_allowed(single_domain, read_q, action_q) is expected


@pytest.mark.asyncio
async def test_supervisor_treats_deferred_domain_as_multi_domain(monkeypatch):
    """talk+drive 에서 drive 가 뒤로 미뤄져 지금 도는 도메인이 하나여도 single_domain=False 로 넘긴다
    (deferred 워커도 같은 원문을 다시 보므로 계획 전체 기준)."""
    from agent_arch.agents.graphs import supervisor as sv
    from agent_arch.agents.react.types import LoopResult
    from agent_arch.llm.base import LLMResponse
    seen = []

    async def fake_worker(domain, read_q, action_q, messages, **kw):
        seen.append((domain, kw.get("single_domain")))
        return domain, None
    monkeypatch.setattr(sv, "_run_worker", fake_worker)
    # talk+drive 조회쌍, 발화에 파일·폴더 신호 없음 → _split_deferred 가 drive 를 미룬다
    router = MagicMock()
    router.invoke = AsyncMock(return_value=LLMResponse(
        content='{"domains":["talk","drive"],"subqueries":{"talk":"회사 채널 공지 확인","drive":"회사 채널 관련 문서"},"confidence":0.9}',
        finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}))
    main = MagicMock(); main.invoke = AsyncMock(return_value=LLMResponse(content="", finish_reason="stop", usage={}))
    main.bind_tools = AsyncMock()
    await sv.run_supervisor(
        main, "회사 채널 공지 어떻게 확인해?", [Message(role="user", content="회사 채널 공지 어떻게 확인해?")],
        resolve_tools_fn=AsyncMock(return_value=[]), make_worker_llm_fn=lambda *a, **k: MagicMock(),
        fallback_fn=AsyncMock(return_value=LoopResult(content="fb", tool_results=[], step_meta={})),
        router_llm=router, max_iterations=3,
    )
    assert seen, "워커가 호출되지 않음"
    assert all(flag is False for _, flag in seen), seen
    assert seen[0][0] == "talk"
