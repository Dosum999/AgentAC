"""멀티 오케스트레이션 supervisor 테스트 (Phase 2b).

라우팅 분기(단일/멀티/fallback) + synthesizer + **병렬 워커 격리(핵심)**를 검증한다.
격리 = 각 워커가 fresh llm·messages 복사본·독립 ContextVar(asyncio.gather Task copy)를
가져 서로 오염되지 않음. 특히 병렬 워커의 발견도구(_discovered)가 도메인 간 안 섞임을
실측한다(mock 아닌 실제 gather 실행 경로).
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.llm.base import Message, StreamEvent, ToolCall, LLMResponse
from agent_arch.agents.tools.base import ToolResult
from agent_arch.agents.react.types import LoopResult, ProgressEvent
from agent_arch.agents.graphs.supervisor import run_supervisor, stream_supervisor, _collect_pending, _merge_pending
from agent_arch.agents.graphs.router import RoutePlan
import agent_arch.agents.graphs.supervisor as _sup


# ── 헬퍼 ──────────────────────────────────────────────────────────────────────

def _stream_text(content, usage=None):
    async def g(*a, **k):
        if content:
            yield StreamEvent(type="text", content=content)
        yield StreamEvent(type="done", finish_reason="stop", usage=usage)
    return g


def _stream_tc(tcs, usage=None):
    async def g(*a, **k):
        yield StreamEvent(type="tool_calls", tool_calls=tcs)
        yield StreamEvent(type="done", finish_reason="tool_calls", usage=usage)
    return g


def _make_tool(name, output="결과"):
    t = MagicMock()
    t.name = name
    t.execute = AsyncMock(return_value=ToolResult(success=True, output=output))
    return t


def _llm_json(content):
    m = MagicMock()
    m.invoke = AsyncMock(return_value=LLMResponse(content=content, finish_reason="stop"))
    return m


def _wllm_factory(specs_builder):
    """make_worker_llm_fn: 호출마다 fresh mock llm(fresh stream specs). .calls['n']=호출수."""
    calls = {"n": 0}

    def make():
        calls["n"] += 1
        llm = MagicMock()
        gens = list(specs_builder())

        def nxt(*a, **k):
            return gens.pop(0)(*a, **k)

        llm.stream = nxt
        llm.bind_tools = AsyncMock()
        return llm

    make.calls = calls
    return make


def _single_answer_specs():
    return [_stream_text("도메인 답변")]


# ── 라우팅 분기 ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_domain_no_synth():
    """단일 도메인 → 워커 1개, synthesizer 미호출(워커 결과 그대로)."""
    router = _llm_json('{"domains":["hr"],"subqueries":{"hr":"연차"},"confidence":0.95}')
    main = MagicMock()
    main.invoke = AsyncMock()   # synth 호출되면 안 됨
    resolve = AsyncMock(return_value=[_make_tool("hr_leave")])
    make_wllm = _wllm_factory(_single_answer_specs)

    r = await run_supervisor(
        main, "내 연차", [Message(role="user", content="내 연차")],
        resolve_tools_fn=resolve, make_worker_llm_fn=make_wllm,
        fallback_fn=AsyncMock(), router_llm=router, max_iterations=3,
    )
    assert r.content == "도메인 답변"
    main.invoke.assert_not_called()          # synth 없음
    resolve.assert_awaited_once_with("연차", "hr")
    assert make_wllm.calls["n"] == 1


@pytest.mark.asyncio
async def test_multi_domain_synthesized():
    """멀티 도메인 → 병렬 워커 + synthesizer 1콜 병합."""
    router = _llm_json('{"domains":["hr","talk"],"subqueries":{"hr":"연차","talk":"공유"},"confidence":0.9}')
    main = MagicMock()
    main.invoke = AsyncMock(return_value=LLMResponse(content="통합 답변", finish_reason="stop",
                                                     usage={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}))
    main.bind_tools = AsyncMock()

    async def resolve(subquery, domain):
        return [_make_tool(f"{domain}_tool")]

    make_wllm = _wllm_factory(_single_answer_specs)
    r = await run_supervisor(
        main, "연차 확인하고 공유해줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=make_wllm,
        fallback_fn=AsyncMock(), router_llm=router, max_iterations=3,
    )
    assert r.content == "통합 답변"
    main.invoke.assert_awaited_once()        # synth 1회
    assert make_wllm.calls["n"] == 2         # 워커 2개 각자 fresh llm
    assert r.step_meta["synthesized"] is True
    assert {w["domain"] for w in r.step_meta["workers"]} == {"hr", "talk"}


@pytest.mark.asyncio
async def test_fallback_on_no_route():
    """라우터 파싱 실패 → fallback_fn 호출."""
    router = _llm_json("JSON 아님")
    sentinel = LoopResult(content="fallback 답변", finish_reason="stop")
    fb = AsyncMock(return_value=sentinel)
    r = await run_supervisor(
        MagicMock(), "뭔가", [Message(role="user", content="q")],
        resolve_tools_fn=AsyncMock(), make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=fb, router_llm=router,
    )
    assert r is sentinel
    fb.assert_awaited_once()


@pytest.mark.asyncio
async def test_fallback_on_low_confidence():
    """저신뢰 → fallback(임계값 0.6 미만)."""
    router = _llm_json('{"domains":["hr"],"confidence":0.3}')
    fb = AsyncMock(return_value=LoopResult(content="fb", finish_reason="stop"))
    await run_supervisor(
        MagicMock(), "q", [Message(role="user", content="q")],
        resolve_tools_fn=AsyncMock(), make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=fb, router_llm=router, confidence_threshold=0.6,
    )
    fb.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_empty_still_runs_worker_via_finder():
    """resolve 가 도메인 도구 0개여도 find_<domain>_tools 로 워커가 돌아 fallback 안 감(자체복구)."""
    router = _llm_json('{"domains":["hr"],"subqueries":{"hr":"연차"},"confidence":0.9}')
    fb = AsyncMock(return_value=LoopResult(content="fb", finish_reason="stop"))
    r = await run_supervisor(
        MagicMock(), "연차", [Message(role="user", content="연차")],
        resolve_tools_fn=AsyncMock(return_value=[]),   # 도메인 도구 0 → find_ 로 복구
        make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=fb, router_llm=router,
    )
    fb.assert_not_called()               # 무스코프 fallback 아님
    assert r.content == "도메인 답변"     # find_ 만으로도 워커 실행


# ── 멀티도메인 + HITL 승인 (버그픽스: pending_queue 보존) ────────────────────

def test_collect_pending_aggregates():
    r_pending = LoopResult(content="", finish_reason="action_pending",
                           pending_queue=[{"tool_name": "mail_send_message", "action_id": "a1"}])
    r_done = LoopResult(content="일정 답변", finish_reason="stop")
    q = _collect_pending([("mail", r_pending), ("calendar", r_done)])
    assert len(q) == 1 and q[0]["tool_name"] == "mail_send_message"


def test_collect_pending_empty_when_none():
    assert _collect_pending([("hr", LoopResult(content="a", finish_reason="stop"))]) == []


def test_merge_pending_preserves_queue_and_done_content():
    r_pending = LoopResult(content="메일 초안", finish_reason="action_pending",
                           pending_queue=[{"tool_name": "mail_send_message", "action_id": "a1"}],
                           tool_calls=[{"id": "m1"}])
    r_done = LoopResult(content="일정 답변", finish_reason="stop", tool_calls=[{"id": "c1"}])
    merged = _merge_pending([("mail", r_pending), ("calendar", r_done)],
                            [{"tool_name": "mail_send_message", "action_id": "a1"}],
                            RoutePlan(domains=["mail", "calendar"], confidence=0.9))
    assert merged.finish_reason == "action_pending"
    assert merged.pending_queue[0]["tool_name"] == "mail_send_message"
    assert merged.pending_action["tool_name"] == "mail_send_message"
    assert "일정 답변" in merged.content          # 완료 워커 content 포함
    assert "메일 초안" in merged.content            # 승인대기 워커 content(읽기 요약 등)도 보존
    assert len(merged.tool_calls) == 2              # 전 워커 tool_calls 병합


@pytest.mark.asyncio
async def test_read_write_subruns_scope_tools(monkeypatch):
    """분해 강제: read 서브런=비-승인 도구+find_ / write 서브런=도메인 전체(조회+승인)+find_.
    content=read요약+카드 고정문구 (초안 있으면 write 텍스트는 버림)."""
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        names = [t.name for t in tools]
        captured.append({"query": next((m.content for m in reversed(messages) if m.role == "user"), None),
                         "tools": names})
        if any("send" in n for n in names):    # write 서브런(승인 도구 포함) → 초안
            return LoopResult(content="초안텍스트는버려짐", finish_reason="action_pending",
                              pending_queue=[{"tool_name": "mail_send_message", "action_id": "a1", "draft_view": {}}])
        return LoopResult(content="받은 메일 요약", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json('{"domains":["mail"],"subqueries":{"mail":"받은 메일 확인"},'
                       '"actions":{"mail":"테스트 메일 발송"},"confidence":0.9}')

    async def resolve(query, domain):
        # 실제 스코프 검색처럼 write 질의에도 조회 도구가 섞여 나온다 — write 서브런이
        # 승인 도구와 함께 조회 도구도 bind 하는 새 동작을 검증하기 위함.
        if "발송" in query:
            return [_make_tool("mail_send_message"), _make_tool("mail_list_messages")]
        return [_make_tool("mail_list_messages")]

    r = await run_supervisor(
        MagicMock(), "받은 메일 확인하고 테스트 발송", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
    )
    # WRITE 서브런에는 READ 결과 인계 블록이 뒤에 붙는다 — 서브런 식별은 앞부분으로 한다.
    read_run = next(c for c in captured if (c["query"] or "").startswith("받은 메일 확인"))
    assert "mail_list_messages" in read_run["tools"] and "find_mail_tools" in read_run["tools"]
    assert "mail_send_message" not in read_run["tools"]     # 승인 도구는 read 서브런에서 제외(초안 물리적 불가)
    write_run = next(c for c in captured if (c["query"] or "").startswith("테스트 메일 발송"))
    # write 서브런 = 도메인 전체(조회+승인) + find_ — 실행 전 조회(수신자·uid 확인)를 스스로 수행 가능.
    assert set(write_run["tools"]) == {"mail_send_message", "mail_list_messages", "find_mail_tools"}
    assert "받은 메일 요약" in r.content and "초안텍스트는버려짐" not in r.content  # 초안 有 → write텍스트 폐기 유지
    assert r.finish_reason == "action_pending" and r.pending_queue[0]["tool_name"] == "mail_send_message"


def test_combine_worker_preserves_no_draft_write_text():
    """무초안 write(필수값 되물음·실패 안내) 텍스트 보존 — 버리면 빈 응답/프론트 무한 로딩(실측).

    초안이 있을 때의 폐기 규칙은 test_read_write_subruns_scope_tools 가 계속 보증한다.
    """
    # write 단독(라우터가 subqueries 없이 actions 만 준 케이스) — 재질문이 content 에 살아남는다.
    w = LoopResult(content="휴가 시작일과 종료일을 알려주세요.", finish_reason="stop")
    r = _sup._combine_worker(None, w)
    assert "시작일과 종료일" in r.content
    assert r.finish_reason == "stop" and not r.pending_queue

    # read 요약과 공존 시 둘 다 보존(이어붙임).
    rd = LoopResult(content="잔여 연차 3일", finish_reason="stop")
    both = _sup._combine_worker(rd, w)
    assert "잔여 연차 3일" in both.content and "시작일과 종료일" in both.content


def test_combine_worker_drops_write_restatement():
    """WRITE 가 READ 결과를 인계받아 **같은 말을 다시 한** 경우엔 이어붙이지 않는다.

    실측 2026-09-15: "이영희 이사님이 메신저에서 공유한 ... 링크 찾아줘" 에 READ 서브런이
    답을 만들고(find_posts_by_author), 실행할 것이 없던 WRITE 서브런이 그 인계 내용을
    그대로 다시 서술했다. 둘을 이어붙여 **같은 답이 두 번** 나갔다(DB answer 917자에 같은
    제목 2회). 사용자에게는 "답변이 반복된다"로 보인다.
    """
    read_text = (
        "이영희 이사님이 공유하신 '에이전트의 역할과 권한' 관련 링크입니다.\n"
        "- 에이전트의 역할과 권한 (2026-08-28): Confluence 링크\n"
        "- 공유 채널: 회사 scrum"
    )
    write_text = (
        "이영희 이사님이 메신저에서 공유하신 '에이전트의 역할과 권한' 관련 링크입니다.\n"
        "- 에이전트의 역할과 권한 (2026-08-28): Confluence 링크\n"
        "- 공유 채널: 회사 scrum"
    )
    r = _sup._combine_worker(
        LoopResult(content=read_text, finish_reason="stop"),
        LoopResult(content=write_text, finish_reason="stop"),
    )
    assert r.content.count("에이전트의 역할과 권한 (2026-08-28)") == 1


def test_combine_worker_keeps_the_longer_restatement():
    """재서술이라도 더 많이 담은 쪽을 남긴다 — 어느 쪽이 자세한지는 턴마다 다르다."""
    short, long_ = "잔여 연차는 3일입니다.", "잔여 연차는 3일입니다. 사용 기한은 12월 31일입니다."
    r = _sup._combine_worker(
        LoopResult(content=short, finish_reason="stop"),
        LoopResult(content=long_, finish_reason="stop"),
    )
    assert r.content == long_


def test_restatement_judgement_separates_followup_questions():
    """되물음·실패 안내는 재서술이 아니다 — 버리면 빈 응답이 된다(그 회귀를 지킨다)."""
    read_text = "잔여 연차 3일"
    assert _sup._is_restatement(read_text, "잔여 연차는 3일입니다.") is True
    assert _sup._is_restatement(read_text, "휴가 시작일과 종료일을 알려주세요.") is False
    assert _sup._is_restatement("", "무언가") is False
    assert _sup._is_restatement("무언가", "") is False


def test_scope_messages_replaces_user_and_adds_domain_directive():
    """워커 메시지 = 마지막 user 를 서브쿼리로 교체 + system 에 도메인 전담 지침 추가, 원본 불변."""
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="이전질문"),
        Message(role="assistant", content="이전답변"),
        Message(role="user", content="메일 확인하고 일정도 알려줘"),
    ]
    out = _sup._scope_messages(msgs, "메일만", "mail")
    assert out[-1].role == "user" and out[-1].content == "메일만"        # 서브쿼리로 교체
    assert out[0].content.startswith("sys") and "mail" in out[0].content and "도메인 전담" in out[0].content  # 지침 추가
    assert msgs[-1].content == "메일 확인하고 일정도 알려줘" and msgs[0].content == "sys"  # 원본 불변
    assert out[1].content == "이전질문"  # 히스토리 유지


@pytest.mark.asyncio
async def test_run_supervisor_multi_pending_skips_synth(monkeypatch):
    """멀티도메인에서 워커 하나가 승인대기 → synth 스킵 + pending_queue 전파(409 버그 방지)."""
    # mail=발송(action), calendar=조회(read)
    router = _llm_json('{"domains":["mail","calendar"],"subqueries":{"calendar":"일정"},'
                       '"actions":{"mail":"메일 발송"},"confidence":0.9}')
    main = MagicMock()
    main.invoke = AsyncMock()      # synth 호출되면 안 됨
    main.bind_tools = AsyncMock()

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        if any("send" in t.name for t in tools):   # write 서브런 → 승인대기 초안
            return LoopResult(content="", finish_reason="action_pending",
                              pending_queue=[{"tool_name": "mail_send_message", "action_id": "a1", "draft_view": {"k": 1}}])
        return LoopResult(content="일정 답변", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)

    async def resolve(query, domain):
        return [_make_tool("mail_send_message")] if domain == "mail" else [_make_tool("calendar_list_events")]

    r = await run_supervisor(
        main, "메일 보내고 일정 확인", [Message(role="user", content="q")],
        resolve_tools_fn=resolve,
        make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
    )
    assert r.finish_reason == "action_pending"
    assert r.pending_queue[0]["tool_name"] == "mail_send_message"   # 승인 큐 보존 → 체크포인트 가능
    main.invoke.assert_not_called()                                 # synth 스킵


# ── ★ 병렬 워커 격리 (2b 핵심) ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parallel_worker_contextvar_isolation():
    """병렬 워커의 _discovered 격리: hr 워커는 hr_disc만, talk 워커는 talk_disc만 발견.

    gather 가 각 워커를 Task 로 감싸며 context 를 copy → 워커별 발견도구 대기열이 격리.
    누출되면 한 워커가 다른 도메인의 발견도구까지 보게 되어 assert 실패.
    """
    from agent_arch.agents.tools.tool_search import _enqueue

    def resolve_finder(domain):
        disc = _make_tool(f"{domain}_disc")
        finder = MagicMock()
        finder.name = "find_x"

        async def _ex(**kw):
            _enqueue([disc])   # 이 워커의 도메인 발견도구를 대기열에 적재
            return ToolResult(success=True, output={"found": 1})

        finder.execute = AsyncMock(side_effect=_ex)
        return finder

    async def resolve(subquery, domain):
        return [resolve_finder(domain)]

    # 워커 llm: iter0 find_x 호출 → iter1 답변
    def worker_specs():
        return [_stream_tc([ToolCall(id="f", name="find_x", arguments={})]),
                _stream_text("답변")]

    router = _llm_json('{"domains":["hr","talk"],"subqueries":{"hr":"h","talk":"t"},"confidence":0.9}')
    main = MagicMock()
    main.invoke = AsyncMock(return_value=LLMResponse(content="통합", finish_reason="stop"))
    main.bind_tools = AsyncMock()

    r = await run_supervisor(
        main, "hr랑 talk 둘 다", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(worker_specs),
        fallback_fn=AsyncMock(), router_llm=router, max_iterations=3,
    )
    workers = {w["domain"]: w["step_meta"] for w in r.step_meta["workers"]}
    assert workers["hr"]["discovered_tools"] == ["hr_disc"], \
        f"hr 워커 발견도구 오염: {workers['hr']['discovered_tools']}"
    assert workers["talk"]["discovered_tools"] == ["talk_disc"], \
        f"talk 워커 발견도구 오염: {workers['talk']['discovered_tools']}"


# ── 스트리밍 ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sticky_fallback_to_prior_domain():
    """라우터 punt(참조 질의) + prior_domain → 그 도메인으로 스코프 유지(무스코프 fallback 아님)."""
    router = _llm_json("JSON 아님")   # 라우터 punt
    resolve = AsyncMock(return_value=[_make_tool("hr_tool")])
    fb = AsyncMock(return_value=LoopResult(content="무스코프 fallback", finish_reason="stop"))
    r = await run_supervisor(
        MagicMock(), "그 중 제일 큰 거", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=fb, router_llm=router, prior_domain="hr",
    )
    fb.assert_not_called()                                    # sticky → 무스코프 fallback 안 감
    resolve.assert_awaited_once_with("그 중 제일 큰 거", "hr")  # 직전 도메인으로 스코프
    assert r.content == "도메인 답변"


@pytest.mark.asyncio
async def test_sticky_fallback_preserves_prior_action_and_binds_action_tool(monkeypatch):
    """거절 후 정정 턴: 라우터 punt 여도 직전 승인도구(create)를 WRITE 로 복원해 초안 재생성.

    회귀(실측 2026-07-27~31): sticky fallback 이 도메인만 복원해 정정 요청이 READ 기본값으로
    떨어졌고, READ 의 requires_approval 필터가 eapproval_create_document 를 제거해 NOT FOUND
    → LLM 이 '상신 준비했습니다'를 환각(answer_guard 가 문구만 제거, 초안 0건).
    """
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        names = [t.name for t in tools]
        captured.append({"query": query, "tools": names})
        return LoopResult(
            content="수정 초안",
            finish_reason="action_pending",
            pending_queue=[{
                "tool_name": "eapproval_create_document",
                "action_id": "a-revised",
                "draft_view": {"kind": "draft"},
            }],
        )

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    # 라우터 punt: actions 를 쓰다 JSON 이 잘린 실측 패턴.
    router = _llm_json('{"domains":["eapproval"],"actions":{"eapproval":"결')
    resolve = AsyncMock(return_value=[])   # retrieval 이 놓쳐도 sticky 로 create 가 복원돼야 함
    fallback = AsyncMock(return_value=LoopResult(content="fallback", finish_reason="stop"))
    sticky = [_make_tool("eapproval_create_document")]  # 직전 턴 = 승인(실행) 도구 사용
    user_input = "결제수단 법인카드(0993)으로 바꿔줘"

    r = await run_supervisor(
        MagicMock(), user_input, [Message(role="user", content=user_input)],
        resolve_tools_fn=resolve,
        make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=fallback,
        router_llm=router,
        prior_domain="eapproval",
        sticky_tools=sticky,
    )

    fallback.assert_not_called()
    # 정정 발화가 READ(기본)가 아니라 WRITE(action) 질의로 갔고,
    resolve.assert_awaited_once_with(user_input, "eapproval")
    assert captured[0]["query"] == user_input
    # retrieval 이 비어도 sticky create 가 WRITE 에 바인딩됐다(NOT FOUND 차단).
    assert "eapproval_create_document" in captured[0]["tools"]
    # 초안이 그대로 전파된다(카드+버튼의 원료).
    assert r.finish_reason == "action_pending"
    assert r.pending_queue[0]["draft_view"]["kind"] == "draft"


@pytest.mark.asyncio
async def test_sticky_fallback_without_prior_action_stays_read(monkeypatch):
    """직전 턴이 조회뿐이면 sticky fallback 은 기존대로 READ — 스푸리어스 WRITE 방지."""
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        captured.append([t.name for t in tools])
        return LoopResult(content="조회 답", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json("JSON 아님")   # punt
    resolve = AsyncMock(return_value=[_make_tool("eapproval_list_documents")])
    sticky = [_make_tool("eapproval_list_documents")]  # 승인도구 아님 → prior_was_action=False

    r = await run_supervisor(
        MagicMock(), "그거 다시 보여줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve,
        make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
        prior_domain="eapproval", sticky_tools=sticky,
    )
    assert r.content == "조회 답"
    # READ 서브런 1회만 돌았다(WRITE 스푸리어스 없음).
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_no_sticky_without_prior_domain():
    """prior_domain 없으면 라우터 punt → 무스코프 fallback."""
    router = _llm_json("아님")
    fb = AsyncMock(return_value=LoopResult(content="fb", finish_reason="stop"))
    await run_supervisor(
        MagicMock(), "그 중", [Message(role="user", content="q")],
        resolve_tools_fn=AsyncMock(), make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=fb, router_llm=router, prior_domain=None,
    )
    fb.assert_awaited_once()


@pytest.mark.asyncio
async def test_stream_single_domain_streams_worker():
    """단일 도메인 스트림 → 워커 토큰이 그대로 흐른다."""
    router = _llm_json('{"domains":["calendar"],"subqueries":{"calendar":"내일 회의"},"confidence":0.9}')
    resolve = AsyncMock(return_value=[_make_tool("cal")])
    events = []
    async for ev in stream_supervisor(
        MagicMock(), "내일 회의", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(lambda: [_stream_text("일정 답변")]),
        fallback_stream_fn=None, router_llm=router,
    ):
        events.append(ev)
    tokens = "".join(e.content for e in events if isinstance(e, ProgressEvent) and e.event_type == "token")
    assert tokens == "일정 답변"
    assert any(isinstance(e, LoopResult) for e in events)


@pytest.mark.asyncio
async def test_stream_multi_domain_streams_synth():
    """멀티 도메인 스트림 → 진행이벤트 + synthesizer 토큰만 스트림, 마지막 LoopResult."""
    router = _llm_json('{"domains":["hr","talk"],"confidence":0.9}')
    main = MagicMock()
    main.bind_tools = AsyncMock()
    main.stream = _stream_text("통합 스트림 답변")   # synth 스트림

    async def resolve(subquery, domain):
        return [_make_tool(f"{domain}_tool")]

    events = []
    async for ev in stream_supervisor(
        main, "hr talk 둘 다", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(lambda: [_stream_text("w")]),
        fallback_stream_fn=None, router_llm=router,
    ):
        events.append(ev)
    tokens = "".join(e.content for e in events if isinstance(e, ProgressEvent) and e.event_type == "token")
    assert tokens == "통합 스트림 답변"
    assert isinstance(events[-1], LoopResult)
    assert events[-1].step_meta["synthesized"] is True


@pytest.mark.asyncio
async def test_stream_fallback():
    """라우팅 실패 → fallback_stream_fn 이벤트가 흐른다."""
    router = _llm_json("아님")

    async def fb_stream():
        yield ProgressEvent("token", content="fb 토큰")
        yield LoopResult(content="fb 토큰", finish_reason="fallback")

    events = [ev async for ev in stream_supervisor(
        MagicMock(), "q", [Message(role="user", content="q")],
        resolve_tools_fn=AsyncMock(), make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_stream_fn=fb_stream, router_llm=router,
    )]
    assert any(isinstance(e, LoopResult) and e.finish_reason == "fallback" for e in events)


# ── 워커 스코프: 글로벌 유틸리티 + sticky 이전턴 도구(모놀리식 패리티) ────────────

@pytest.mark.asyncio
async def test_worker_scope_global_read_only_and_sticky(monkeypatch):
    """글로벌 유틸리티는 **read 서브런에만** 얹힌다(write 는 도메인 전체+find_ 스코프[#105]라
    글로벌 없이도 도메인 내부 조회 가능 — 막연한 실행에서 글로벌로 헤매는 지연/블록 회피).
    도메인 sticky 는 read 에 복원되고 다른 도메인 sticky 는 격리."""
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), None)
        names = [t.name for t in tools]
        captured.append({"query": last_user, "tools": names})
        if any("send" in n for n in names):     # write 서브런(승인 도구) → 초안
            return LoopResult(content="draft", finish_reason="action_pending",
                              pending_queue=[{"tool_name": "mail_send_message", "action_id": "a1", "draft_view": {}}])
        return LoopResult(content="read ok", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json('{"domains":["mail"],"subqueries":{"mail":"받은 메일"},'
                       '"actions":{"mail":"메일 발송"},"confidence":0.9}')

    async def resolve(query, domain):
        return [_make_tool("mail_send_message")] if "발송" in query else [_make_tool("mail_list_messages")]

    global_tools = [_make_tool("calculator"), _make_tool("search_colleagues")]
    # mail 도메인 sticky(prefix mail_) + hr 도메인 sticky(prefix hr_) — hr 은 mail 워커에서 격리돼야.
    sticky = [_make_tool("mail_draft_reply"), _make_tool("hr_leave_balance")]

    await run_supervisor(
        MagicMock(), "받은 메일 확인하고 발송", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
        global_tools=global_tools, sticky_tools=sticky,
    )
    read = next(c for c in captured if (c["query"] or "").startswith("받은 메일"))
    write = next(c for c in captured if (c["query"] or "").startswith("메일 발송"))
    # 글로벌 유틸리티는 read 에만 — 단 사람 검색은 write 에도(_WRITE_GLOBAL_ALLOWLIST 예외)
    assert {"calculator", "search_colleagues"} <= set(read["tools"])
    assert "calculator" not in write["tools"]        # 지연 회귀 원인 글로벌은 write 에서 여전히 배제
    # 사람 해석(이름→email)은 write 에 필수 — actions-only 턴은 read 서브런이 없어
    # 수신자·결재선이 이름만으로 진행됨(실측: SMTP 502 / 알림 멘션 빈칸).
    assert "search_colleagues" in write["tools"]
    assert "mail_send_message" in write["tools"]     # write 는 도메인 승인 도구 포함(#105 스코프)
    # 도메인 sticky 복원(read/write) + 다른 도메인 sticky 격리 — write 에도 이어받아야
    # 정정 턴에서 retrieval 이 직전 승인도구를 놓쳐도 NOT FOUND 가 안 난다.
    assert "mail_draft_reply" in read["tools"]
    assert "mail_draft_reply" in write["tools"]
    assert "hr_leave_balance" not in read["tools"] and "hr_leave_balance" not in write["tools"]


@pytest.mark.asyncio
async def test_attachment_reader_reaches_write_only_worker(monkeypatch):
    """첨부 읽기는 도메인 워커의 WRITE 서브런에도 얹힌다 — 2026-10-08 운영 실측 회귀.

    "이 발주서 수주 등록해줘" 는 라우터가 actions 만 준다(read 서브런 없음). 물류 WRITE 워커가
    첨부 PDF 를 열려고 mattermost_read_file 을 불렀지만 스코프에 없어 `tool NOT FOUND` 가 났고,
    사용자에게 "파일 내용을 텍스트로 알려 달라"고 답했다. 다른 글로벌은 여전히 write 에서 배제.
    """
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        captured.append([t.name for t in tools])
        return LoopResult(content="draft", finish_reason="action_pending",
                          pending_queue=[{"tool_name": "logistics_create_sales_order",
                                          "action_id": "a1", "draft_view": {}}])

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json('{"domains":["logistics"],"subqueries":{},'
                       '"actions":{"logistics":"발주서를 읽어 수주서 등록"},"confidence":1.0}')
    await run_supervisor(
        MagicMock(), "이 발주서 수주 등록해줘", [Message(role="user", content="q")],
        resolve_tools_fn=AsyncMock(return_value=[_make_tool("logistics_create_sales_order")]),
        make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
        global_tools=[_make_tool("calculator"), _make_tool("mattermost_read_file")],
    )
    assert len(captured) == 1
    assert "mattermost_read_file" in captured[0]
    assert "calculator" not in captured[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("text, promoted", [
    ("브라켓 A-10 발주요청건이 변경되었어.. 발주수량이 5000개로 바뀌고 납기일도 일주일 앞당겨졌어 수정해줘", True),
    ("이 발주서로 수주 등록해 주세요", True),
    ("납기 바꿔줘", True),
    ("H사에서 긴급 증량 요청이 왔습니다. 브라켓 A-10 3,000 → 5,000, 납기도 1주 당겨 달라고 합니다. 발주서 첨부합니다.", True),
    ("H사가 수량을 늘려 달랍니다", True),
    ("H사에서 발주서 왔는데 내용 좀 봐줘", False),
    ("변경된 발주서 내용 확인해줘", False),
    ("수정된 수주서 목록 보여줘", False),
])
async def test_imperative_write_request_gets_action_even_if_router_says_read(monkeypatch, text, promoted):
    """첨부 + '…수정해줘' 를 라우터가 조회로만 분류해도 실행 워커가 돈다(2026-10-12 운영 회귀). 조회만이면 그대로."""
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        captured.append([t.name for t in tools])
        return LoopResult(content="ok", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json('{"domains":["logistics"],"subqueries":{"logistics":"첨부 발주서 변경 내용 확인"},'
                       '"confidence":1.0}')

    async def resolve(query, domain):
        return [_make_tool("logistics_update_sales_order"), _make_tool("logistics_list_sales_orders")]

    await run_supervisor(
        MagicMock(), text, [Message(role="user", content=text)],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
    )
    write_runs = [c for c in captured if "logistics_update_sales_order" in c]
    assert bool(write_runs) is promoted


@pytest.mark.asyncio
@pytest.mark.parametrize("text, migrated", [
    ("물류 마이그레이션에 있는 엑셀파일들 물류로 마이그레이션 해줘", True),
    ("드라이브 자재입고 엑셀 ERP 입고처리로 이관해줘", True),
    ("물류 마이그레이션 폴더안에 어떤 파일이 있어??", False),
    ("자재입고 엑셀 파일 내용 뭐야?", False),
    ("물류 마이그레이션 폴더 엑셀 이관 좀 해 주세요", True),
])
async def test_drive_excel_migration_goes_to_logistics_action(monkeypatch, text, migrated):
    """'폴더·엑셀파일' 때문에 라우터가 drive 조회로 보내도 이관 요청은 logistics 실행 하나로 간다(2026-10-13).

    drive 조회 워커에는 이관 도구가 없어 "ERP 에서 직접 입력하라" 로 끝났다. drive 워커는 같이 돌리지 않는다.
    """
    ran = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        ran.append([t.name for t in tools])
        return LoopResult(content="ok", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json('{"domains":["drive"],"subqueries":{"drive":"물류 마이그레이션 폴더 엑셀 확인"},'
                       '"confidence":1.0}')

    async def resolve(query, domain):
        if domain == "drive":
            return [_make_tool("drive_list_files")]
        return [_make_tool("logistics_migrate_drive_excel"), _make_tool("logistics_list_goods_receipts")]

    await run_supervisor(
        MagicMock(), text, [Message(role="user", content=text)],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
    )
    assert any("logistics_migrate_drive_excel" in r for r in ran) is migrated
    assert any("drive_list_files" in r for r in ran) is (not migrated)


def test_attachment_reader_survives_domain_global_narrowing():
    """web 처럼 글로벌을 좁힌 도메인도 첨부 읽기는 남는다."""
    tools = [_make_tool("calculator"), _make_tool("search_colleagues"), _make_tool("mattermost_read_file")]
    names = {t.name for t in _sup._domain_globals(tools, "web")}
    assert "mattermost_read_file" in names and "search_colleagues" not in names


@pytest.mark.asyncio
async def test_worker_scope_defaults_no_global_when_unset(monkeypatch):
    """global_tools/sticky_tools 미주입(기본) 시 스코프에 아무것도 안 얹힘(무회귀)."""
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        captured.append([t.name for t in tools])
        return LoopResult(content="ok", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json('{"domains":["hr"],"subqueries":{"hr":"연차"},"confidence":0.95}')
    await run_supervisor(
        MagicMock(), "내 연차", [Message(role="user", content="q")],
        resolve_tools_fn=AsyncMock(return_value=[_make_tool("hr_leave")]),
        make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
    )
    # hr_leave + find_hr_tools 만(글로벌 없음)
    assert "calculator" not in captured[0] and "search_colleagues" not in captured[0]
    assert "hr_leave" in captured[0]


# ── 거절 초안 정정 턴: read → action 승격 + 주입 블록 이월 ────────────────────

@pytest.fixture
def rejected_draft(monkeypatch):
    """요청 스코프에 '직전 거절된 초안' 을 심는다(orchestrator 가 하는 일의 테스트판)."""
    from agent_arch.agents.context import set_rejected_draft_context
    draft = {
        "tool_name": "calendar_create_event",
        "arguments": {"summary": "테스트 회의", "dtstart": "2026-08-13T09:00:00+09:00"},
        "fields": [],
    }
    set_rejected_draft_context(draft)
    yield draft
    set_rejected_draft_context(None)


def test_bias_promotes_read_to_action_after_reject(rejected_draft):
    """거절 stash 가 있으면 확신 read 플랜도 WRITE 로 승격하고 그 도메인 subquery 는 뗀다.

    회귀: "10시로 변경해줘" 는 유창해서 라우터가 높은 신뢰도로 subqueries(조회)에 넣고,
    _decide 는 조기 return 해 sticky 보정을 건너뛴다 → READ 서브런이 requires_approval
    도구를 제거해 초안을 다시 만들 수단이 사라진다.
    """
    plan = RoutePlan(domains=["calendar"], subqueries={"calendar": "9시 일정 찾아 10시로"}, confidence=0.9)
    out = _sup._bias_to_action_for_pending_draft(plan, "calendar", "9시 말고 10시로 변경해줘")

    assert out.action_query("calendar") == "9시 일정 찾아 10시로"   # WRITE 로 승격
    assert out.read_query("calendar") is None                       # read/write 동시 실행 방지
    assert plan.subqueries == {"calendar": "9시 일정 찾아 10시로"}   # 원본 불변


def test_bias_noop_without_rejected_draft():
    """거절 stash 가 없으면 플랜을 건드리지 않는다 — 승인 후 정상 후속 조회가 WRITE 로 새면 안 된다."""
    from agent_arch.agents.context import set_rejected_draft_context
    set_rejected_draft_context(None)
    plan = RoutePlan(domains=["calendar"], subqueries={"calendar": "방금 만든 일정 언제였지"}, confidence=0.9)
    out = _sup._bias_to_action_for_pending_draft(plan, "calendar", "방금 만든 일정 언제였지?")
    assert out is plan and out.action_query("calendar") is None


def test_bias_keeps_existing_action_and_other_domains(rejected_draft):
    """이미 action 으로 분류됐으면 그대로 두고, 다른 도메인 subquery 는 유지한다."""
    plan = RoutePlan(
        domains=["calendar", "mail"],
        subqueries={"mail": "받은 메일 확인"},
        actions={"calendar": "일정 다시 만들어"},
        confidence=0.9,
    )
    out = _sup._bias_to_action_for_pending_draft(plan, "calendar", "q")
    assert out is plan                                   # 손대지 않음
    assert out.read_query("mail") == "받은 메일 확인"


def test_bias_noop_when_prior_domain_not_routed(rejected_draft):
    """라우터가 다른 도메인으로 갔으면 승격하지 않는다(엉뚱한 도메인에 카드 생성 방지)."""
    plan = RoutePlan(domains=["mail"], subqueries={"mail": "메일 확인"}, confidence=0.9)
    out = _sup._bias_to_action_for_pending_draft(plan, "calendar", "q")
    assert out is plan and out.action_query("mail") is None


def test_scope_messages_carries_rejected_draft_note(rejected_draft):
    """워커 메시지에 초안 인자 원본이 이월돼야 한다 — 없으면 주입 전체가 무의미하다.

    _scope_messages 는 마지막 user content 를 도메인 서브쿼리로 갈아끼우므로,
    orchestrator 가 augmented_input 에 붙인 블록이 워커에는 도달하지 않는다.
    """
    msgs = [Message(role="system", content="sys"), Message(role="user", content="원본 발화")]
    out = _sup._scope_messages(msgs, "일정 다시 만들어", "calendar")

    assert out[-1].content.startswith("일정 다시 만들어")
    assert "[직전 거절된 초안]" in out[-1].content
    assert "2026-08-13T09:00:00+09:00" in out[-1].content   # ISO 인자 원본이 살아 있음
    assert msgs[-1].content == "원본 발화"                    # 원본 불변


def test_scope_messages_no_note_without_rejected_draft():
    """거절 stash 가 없으면 블록을 붙이지 않는다(평상시 프롬프트 오염 방지)."""
    from agent_arch.agents.context import set_rejected_draft_context
    set_rejected_draft_context(None)
    out = _sup._scope_messages([Message(role="user", content="q")], "서브쿼리", "calendar")
    assert out[-1].content == "서브쿼리"


def test_bias_derives_domain_from_stashed_tool_when_prior_domain_missing(rejected_draft):
    """prior_domain 이 없어도(tool_ctx 만료 등) stash 의 도구명으로 도메인을 유도해 승격한다."""
    plan = RoutePlan(domains=["calendar"], subqueries={"calendar": "10시로 변경"}, confidence=0.9)
    out = _sup._bias_to_action_for_pending_draft(plan, None, "10시로 변경해줘")
    assert out.action_query("calendar") == "10시로 변경"
    assert out.read_query("calendar") is None


# ── 시점 표현 유실 방지 (라우터가 '다음주'를 떨어뜨려도 워커에 도달) ──────────

def test_lost_time_note_reattaches_dropped_week_marker():
    """실측 회귀: "다음주 금요일 …잡아줘" 의 action 질의에서 '다음주'가 빠져
    워커가 이번주 금요일(08-14)에 등록했다. 같은 발화의 조회 경로는 정상이었으므로
    모델이 아니라 라우터 유실이 원인 — 코드가 결정적으로 되붙인다."""
    note = _sup._lost_time_note("다음주 금요일 오전 9시 테스트 미팅 잡아줘", "테스트 미팅 오전 9시 등록")
    assert "다음주" in note


def test_lost_time_note_noop_when_preserved():
    """라우터가 이미 보존했으면 덧붙이지 않는다(서브쿼리 노이즈 방지)."""
    assert _sup._lost_time_note("다음주 금요일 미팅 잡아줘", "다음주 금요일 미팅 등록") == ""
    assert _sup._lost_time_note("회의 잡아줘", "회의 등록") == ""


def test_lost_time_note_matches_across_spacing():
    """'다음 주'(띄어쓰기)와 '다음주'를 같은 표현으로 본다."""
    assert "다음 주" in _sup._lost_time_note("다음 주 수요일 회의", "수요일 회의 등록")
    assert _sup._lost_time_note("다음 주 수요일 회의", "다음주 수요일 회의 등록") == ""


def test_lost_time_note_dedupes_overlapping_markers():
    """'다다음주'가 '다음주'를 포함해 두 번 나열되지 않는다."""
    note = _sup._lost_time_note("다다음주 화요일 회의", "화요일 회의 등록")
    assert "다다음주" in note and note.count("다음주") == 1


def test_scope_messages_carries_lost_time_marker():
    """_scope_messages 가 서브쿼리 교체 시 유실된 시점 표현을 함께 실어 보낸다."""
    msgs = [Message(role="system", content="sys"),
            Message(role="user", content="다음주 금요일 오전 9시 테스트 미팅 잡아줘")]
    out = _sup._scope_messages(msgs, "테스트 미팅 오전 9시 등록", "calendar")
    assert out[-1].content.startswith("테스트 미팅 오전 9시 등록")
    assert "다음주" in out[-1].content
    assert msgs[-1].content == "다음주 금요일 오전 9시 테스트 미팅 잡아줘"   # 원본 불변


# ── 원문 보존(A안, SUPERVISOR_PRESERVE_ORIGINAL) ──────────────────────────────
# 교체 기인 유실 계급(이미지 컨텍스트 4507889 · 주차 표현 4b31ed7 · 거절 초안 0a38065)을
# 구조적으로 차단하는 경로. 서브쿼리는 워커 입력이 아니라 "[담당 지시]" 라벨로만 쓰인다.
# 상세: reports/supervisor_subquery_replacement_analysis.md


@pytest.fixture
def preserve_original(monkeypatch):
    """A안 플래그 ON — 워커 입력 원문 보존 경로로 전환."""
    monkeypatch.setattr(_sup.settings, "supervisor_preserve_original", True)


def test_scope_messages_preserve_keeps_original_and_labels_read(preserve_original):
    """A안 READ: 마지막 user 원문 유지 + 조회 담당 지시만 덧붙임(교체 아님). 원본 불변."""
    msgs = [Message(role="system", content="sys"),
            Message(role="user", content="메일 확인하고 다음주 금요일 미팅 잡아줘")]
    out = _sup._scope_messages(msgs, "받은 메일 확인", "mail", mode="read")
    c = out[-1].content
    assert c.startswith("메일 확인하고 다음주 금요일 미팅 잡아줘")           # 원문이 본문
    assert "[담당 지시]" in c and "'받은 메일 확인'" in c and "조회 부분" in c
    assert "도메인 전담" in out[0].content                                    # 워커 지침 유지
    assert msgs[-1].content == "메일 확인하고 다음주 금요일 미팅 잡아줘"      # 원본 불변


def test_scope_messages_preserve_write_mode_directive(preserve_original):
    """A안 WRITE: 실행 + 선행 조회 허용, 독립 조회만 금지 — 문구가 코드 정책과 일치.

    WRITE 서브런은 조회+승인 도구를 함께 바인딩한다(598670e: UID·원본 메일 등 선행
    조회 없이는 NOT FOUND). "조회를 다루지 마라"로만 쓰면 모델이 선행 조회까지 금지로
    해석해 그 장애를 프롬프트로 재도입한다 — 선행 조회 허용 문구를 계약으로 잠근다.
    """
    msgs = [Message(role="user", content="메일 확인하고 김보라한테 회신 보내줘")]
    out = _sup._scope_messages(msgs, "김보라에게 회신 발송", "mail", mode="write")
    c = out[-1].content
    assert c.startswith("메일 확인하고 김보라한테 회신 보내줘")
    assert "실행 부분" in c and "'김보라에게 회신 발송'" in c
    assert "선행 조회" in c                                  # 실행에 필요한 조회는 허용
    assert "독립적인 조회" in c                               # 금지는 독립 조회로 한정


def test_scope_messages_preserve_keeps_time_marker_without_note(preserve_original):
    """A안: '다음주'가 서브쿼리에 없어도 원문에 남는다 — 4b31ed7 회귀의 구조적 해소.

    보상 노트("원문의 시점 표현")는 붙지 않아야 한다 — 원문이 이미 워커에 도달하므로
    재부착하면 중복 노이즈다.
    """
    msgs = [Message(role="user", content="다음주 금요일 오전 9시 테스트 미팅 잡아줘")]
    out = _sup._scope_messages(msgs, "테스트 미팅 오전 9시 등록", "calendar", mode="write")
    assert "다음주" in out[-1].content
    assert "원문의 시점 표현" not in out[-1].content


def test_scope_messages_preserve_skips_notes(preserve_original, rejected_draft, monkeypatch):
    """A안: 이미지/거절초안 컨텍스트가 있어도 보상 노트를 재부착하지 않는다.

    블록은 orchestrator 가 원문(augmented_input)에 이미 실었으므로 노트를 또 붙이면
    같은 내용이 두 번 들어간다. 컨텍스트를 실제로 심어 두고 노트 부재를 확인한다.
    """
    monkeypatch.setattr("agent_arch.agents.context.get_image_context", lambda: [object()])
    msgs = [Message(role="user", content="원본 발화 [직전 거절된 초안] 인자원본")]
    out = _sup._scope_messages(msgs, "일정 다시 만들어", "calendar", mode="write")
    c = out[-1].content
    assert c.count("[직전 거절된 초안]") == 1              # 원문 것 하나뿐(재부착 없음)
    assert "이미지가 첨부돼 있습니다" not in c              # img_note 미부착


def test_scope_messages_preserve_skips_directive_when_subquery_is_original(preserve_original):
    """서브쿼리=원문 경로(sticky fallback·빈 플랜 기본값·거절 승격 폴백): 지시문 생략.

    이 경로들은 라우터 재작성이 없어 서브쿼리 자리에 user_input 이 그대로 온다.
    담당 범위가 요청 전체이므로 라벨이 무의미하고, 덧붙이면 원문이 본문+라벨로
    두 번 실린다(원문 2회 주입 — 리뷰 지적).
    """
    msgs = [Message(role="system", content="sys"),
            Message(role="user", content="어제 그거 다시 보여줘")]
    out = _sup._scope_messages(msgs, "어제 그거 다시 보여줘", "mail", mode="read")
    assert out[-1].content == "어제 그거 다시 보여줘"       # 원문 그대로 — 지시문 없음
    assert "[담당 지시]" not in out[-1].content
    assert "도메인 전담" in out[0].content                  # 워커 지침(격리)은 여전히 적용


def test_scope_messages_legacy_unchanged_when_flag_off():
    """플래그 OFF(기본): 기존 교체+보상 노트 동작 그대로 — 무회귀 확인."""
    from agent_arch.agents.context import set_rejected_draft_context
    set_rejected_draft_context(None)
    msgs = [Message(role="user", content="다음주 금요일 미팅 잡아줘")]
    out = _sup._scope_messages(msgs, "미팅 등록", "calendar", mode="write")
    assert out[-1].content.startswith("미팅 등록")          # 교체 유지
    assert "다음주" in out[-1].content                       # 레거시 보상 노트 경로 유지


# ── synthesizer 출력 정화 (도구명 누출 봉합) ──────────────────────────────────

@pytest.mark.asyncio
async def test_synth_output_sanitized_nonstream():
    """비스트리밍 synth 가 도구명을 뱉어도 LoopResult.content 에서 마스킹된다."""
    router = _llm_json('{"domains":["hr","talk"],"subqueries":{"hr":"연차","talk":"공유"},"confidence":0.9}')
    main = MagicMock()
    main.invoke = AsyncMock(return_value=LLMResponse(
        content="hr_service_attendance_read 로 조회했습니다.", finish_reason="stop"))
    main.bind_tools = AsyncMock()

    async def resolve(subquery, domain):
        return [_make_tool(f"{domain}_tool")]

    r = await run_supervisor(
        main, "연차 확인하고 공유해줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router, max_iterations=3,
    )
    assert "hr_service_attendance_read" not in r.content
    assert "내부 도구" in r.content


@pytest.mark.asyncio
async def test_synth_output_sanitized_stream_saved_text():
    """스트리밍 synth 의 **저장본**(LoopResult.content)도 정화된다.

    화면 토큰은 orchestrator 의 StreamSanitizer 가 막지만, buf 를 그대로 LoopResult 로
    넘기면 DB/히스토리에만 원문이 남아 다음 턴 맥락으로 누출이 되돌아온다.
    단언은 도구명 마스킹으로 한정한다 — head 단위 정화(StreamSanitizer)와 전문 정화는
    수식 스팬 경계에서 결과가 미세하게 다를 수 있어 "화면 == 저장본"은 보장 대상이 아니다.
    """
    router = _llm_json('{"domains":["hr","talk"],"subqueries":{"hr":"연차","talk":"공유"},"confidence":0.9}')
    main = MagicMock()
    main.stream = _stream_text("hr_service_attendance_read 로 조회했습니다.")
    main.bind_tools = AsyncMock()

    async def resolve(subquery, domain):
        return [_make_tool(f"{domain}_tool")]

    final = None
    async for ev in stream_supervisor(
        main, "연차 확인하고 공유해줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(lambda: [_stream_text("w")]),
        fallback_stream_fn=None, router_llm=router, max_iterations=3,
    ):
        if isinstance(ev, LoopResult):
            final = ev
    assert final is not None
    assert "hr_service_attendance_read" not in final.content
    assert "내부 도구" in final.content


# ── 수집 중 초안: 짧은 답변 턴을 실행으로 태운다 ──────────────────────────────────

@pytest.fixture
def collecting_draft(monkeypatch):
    """요청 스코프에 '되물을 값이 남은 초안' 포인터를 심는다(orchestrator 가 하는 일의 테스트판)."""
    from agent_arch.agents.context import set_collecting_draft_context
    pointer = {"tool_name": "eapproval_create_document", "template_code": "expense-report"}
    set_collecting_draft_context(pointer)
    yield pointer
    set_collecting_draft_context(None)


def test_bias_promotes_read_to_action_while_collecting(collecting_draft):
    """수집 중이면 확신 read 플랜도 WRITE 로 승격한다.

    실측 2026-08-18 00:05: 되묻기에 "둘 다" 라고 답한 턴을 라우터가 conf=0.60 으로
    subqueries(조회)에 넣어 _decide 가 조기 return → READ 서브런이 create 를 제거 →
    모델이 "[승인] 버튼을 눌러주세요" 문구만 쓰고 초안 0건(answer_guard 가 문구 제거).
    """
    plan = RoutePlan(domains=["eapproval"], subqueries={"eapproval": "둘 다"}, confidence=0.60)
    out = _sup._bias_to_action_for_pending_draft(plan, "eapproval", "둘 다")

    assert out.action_query("eapproval") == "둘 다"      # WRITE 로 승격
    assert out.read_query("eapproval") is None           # read/write 동시 실행 방지


def test_bias_leaves_other_domains_untouched_while_collecting(collecting_draft):
    """경비 수집 중이라도 **다른 도메인** 플랜은 손대지 않는다 (크로스 도메인 오염 방지)."""
    plan = RoutePlan(domains=["calendar"], subqueries={"calendar": "오늘 일정"}, confidence=0.9)
    out = _sup._bias_to_action_for_pending_draft(plan, "calendar", "오늘 일정 뭐 있어")

    assert out is plan                                   # 사본조차 만들지 않는다
    assert out.action_query("calendar") is None


def test_bias_noop_without_any_pending_draft():
    """거절도 수집도 없으면 기존 동작 — 조회는 조회로 남는다."""
    plan = RoutePlan(domains=["eapproval"], subqueries={"eapproval": "지난달 경비"}, confidence=0.9)
    out = _sup._bias_to_action_for_pending_draft(plan, "eapproval", "지난달 경비 뭐 올렸지?")

    assert out is plan
    assert out.read_query("eapproval") == "지난달 경비"






@pytest.mark.asyncio
async def test_sticky_fallback_other_domain_unaffected_by_collecting(monkeypatch, collecting_draft):
    """경비 수집 중에 **다른 도메인** punt 턴이 오면 기존대로 READ 다 (스푸리어스 WRITE 방지)."""
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        captured.append([t.name for t in tools])
        return LoopResult(content="조회 답", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    resolve = AsyncMock(return_value=[_make_tool("calendar_list_events")])

    await run_supervisor(
        MagicMock(), "그거 다시 보여줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve,
        make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=_llm_json("JSON 아님"),
        prior_domain="calendar",
        sticky_tools=[_make_tool("calendar_list_events")],
    )

    # 캘린더는 수집 포인터와 도메인이 달라 승격 대상이 아니다 — 승인 도구가 붙지 않는다.
    assert "calendar_create_event" not in captured[0]


# ── 거절 stash 와 수집 포인터가 공존할 때 도메인이 가려지지 않는다 ────────────────

@pytest.fixture
def calendar_rejected_and_expense_collecting(monkeypatch):
    """캘린더 초안을 거절한 상태 + 경비 수집 진행 중 — 두 stash 공존 경계조건."""
    from agent_arch.agents.context import (
        set_collecting_draft_context,
        set_rejected_draft_context,
    )
    set_rejected_draft_context({
        "tool_name": "calendar_create_event",
        "arguments": {"summary": "회의"},
        "fields": [],
    })
    set_collecting_draft_context({
        "tool_name": "eapproval_create_document", "template_code": "expense-report",
    })
    yield
    set_rejected_draft_context(None)
    set_collecting_draft_context(None)


def test_both_stashes_are_visible(calendar_rejected_and_expense_collecting):
    """예전엔 `rejected or collecting` 이라 캘린더가 이겨 경비가 가려졌다."""
    assert set(_sup._stash_domains()) == {"calendar", "eapproval"}
    assert _sup._has_pending_draft("calendar")
    assert _sup._has_pending_draft("eapproval")
    assert not _sup._has_pending_draft("mail")


def test_bias_promotes_expense_even_with_calendar_rejected(
    calendar_rejected_and_expense_collecting,
):
    """경비 수집 승격이 캘린더 거절 stash 에 가려지지 않아야 한다."""
    plan = RoutePlan(domains=["eapproval"], subqueries={"eapproval": "둘 다"}, confidence=0.6)
    out = _sup._bias_to_action_for_pending_draft(plan, "eapproval", "둘 다")
    assert out.action_query("eapproval") == "둘 다"
    assert out.read_query("eapproval") is None


def test_bias_promotes_only_the_routed_domain(calendar_rejected_and_expense_collecting):
    """두 stash 가 다 있어도 라우터가 고른 도메인만 승격한다(다른 도메인 오염 없음)."""
    plan = RoutePlan(
        domains=["mail"], subqueries={"mail": "받은 메일"}, confidence=0.9,
    )
    assert _sup._bias_to_action_for_pending_draft(plan, "mail", "받은 메일") is plan




# ── _domain_queries: 직전 턴 write 의도 이어받기 ─────────────────────────────
#
# 실측(2026-08-31): 동명이인 되묻기에 사용자가 "hshin" 이라고 짧게 답하자 라우터가 read 로
# 분류했고, READ 서브런이 requires_approval 도구를 제외해 직전 턴에 쓰던 send_dm 이 사라져
# `tool NOT FOUND` 로 끝났다. 되묻기 답변 턴은 write 의 연속이므로 write 로 태운다.


def _tool(name):
    t = MagicMock()
    t.name = name
    return t


_STICKY_WRITE = [_tool("mattermost_skill_send_dm")]      # talk · 승인 필요
_STICKY_READ = [_tool("mattermost_skill_get_my_channels")]  # talk · 승인 불필요


def test_domain_queries_carries_write_intent_from_sticky():
    """되묻기 답변 턴(동사 없는 짧은 답)이 write 로 태워진다."""
    plan = RoutePlan(domains=["talk"], subqueries={}, confidence=0.9)
    read_q, action_q = _sup._domain_queries(plan, "talk", "hshin", sticky_tools=_STICKY_WRITE)

    assert action_q == "hshin"
    assert read_q is None        # 폴백 read 는 접는다(write 가 도메인 조회까지 포함)


def test_domain_queries_no_carry_without_sticky_write():
    """직전 턴에 write 도구가 없었으면 기존대로 read 다 — 조회 턴이 write 로 새면 안 된다."""
    plan = RoutePlan(domains=["talk"], subqueries={}, confidence=0.9)
    read_q, action_q = _sup._domain_queries(plan, "talk", "hshin", sticky_tools=_STICKY_READ)

    assert action_q is None
    assert read_q == "hshin"


def test_domain_queries_no_carry_without_sticky_at_all():
    plan = RoutePlan(domains=["talk"], subqueries={}, confidence=0.9)
    assert _sup._domain_queries(plan, "talk", "hshin") == ("hshin", None)


def test_domain_queries_carry_is_domain_scoped():
    """sticky 가 다른 도메인 도구면 이어받지 않는다(교차 도메인 오염 방지)."""
    plan = RoutePlan(domains=["talk"], subqueries={}, confidence=0.9)
    other = [_tool("eapproval_create_document")]
    read_q, action_q = _sup._domain_queries(plan, "talk", "hshin", sticky_tools=other)

    assert action_q is None and read_q == "hshin"


def test_domain_queries_keeps_explicit_read_alongside_carry():
    """라우터가 명시한 read 질의는 유지한다 — 폴백만 접는다."""
    plan = RoutePlan(domains=["talk"], subqueries={"talk": "최근 DM 확인"}, confidence=0.9)
    read_q, action_q = _sup._domain_queries(plan, "talk", "hshin", sticky_tools=_STICKY_WRITE)

    assert read_q == "최근 DM 확인"
    assert action_q == "hshin"


def test_domain_queries_existing_action_untouched():
    """이미 action 으로 분류됐으면 그대로 둔다."""
    plan = RoutePlan(domains=["talk"], actions={"talk": "DM 보내"}, confidence=0.9)
    read_q, action_q = _sup._domain_queries(plan, "talk", "hshin", sticky_tools=_STICKY_WRITE)

    assert action_q == "DM 보내"
    assert read_q is None



# ── 후속 질의 상속 · 출처 각주 · retrieved 도구 지침 ─────────────────────────────

@pytest.mark.asyncio
async def test_decide_follow_up_inherits_prior_domains(monkeypatch):
    """라우터가 후속(follow_up)이라 하면 도메인을 새로 추측하지 않고 직전 실행 도메인들을 상속한다."""
    from agent_arch.agents.context import set_web_search_enabled
    monkeypatch.setattr(_sup, "route", AsyncMock(return_value=RoutePlan(domains=["talk"], confidence=0.7, follow_up=True)))
    plan, ok = await _sup._decide(MagicMock(), "근거가 어떻게 되는지 알려줘", 0.6, "web", prior_domains=["web", "talk"])
    assert ok and plan.domains == ["talk"]                                          # 웹이 꺼진 요청은 직전 web 턴을 잇지 않는다
    set_web_search_enabled(True)
    plan, ok = await _sup._decide(MagicMock(), "근거가 어떻게 되는지 알려줘", 0.6, "web", prior_domains=["web", "talk"])
    assert ok and plan.domains == ["web", "talk"] and plan.follow_up and plan.confidence == 1.0
    assert plan.read_query("web") == "근거가 어떻게 되는지 알려줘" and plan.read_query("talk") == "근거가 어떻게 되는지 알려줘"
    # 도메인 모름 + 후속 → 직전 단일 도메인이라도 상속
    monkeypatch.setattr(_sup, "route", AsyncMock(return_value=RoutePlan(domains=[], confidence=0.2, follow_up=True)))
    plan, ok = await _sup._decide(MagicMock(), "그거 자세히", 0.6, "web", prior_domains=[])
    assert ok and plan.domains == ["web"]
    # 직전 도메인이 없으면(무스코프 폴백 뒤) 상속할 것이 없어 종전 punt 경로
    plan, ok = await _sup._decide(MagicMock(), "그거 자세히", 0.6, None, prior_domains=[])
    assert ok is False


@pytest.mark.asyncio
async def test_decide_scms_order_is_not_overridden_by_follow_up(monkeypatch):
    """메신저 발주요청은 켜진 요청에만 있는 도메인 — 앞 턴(그땐 없던 선택지) 도메인 상속이 덮지 않는다."""
    from agent_arch.agents.context import set_scms_order_enabled
    monkeypatch.setattr(_sup, "route", AsyncMock(return_value=RoutePlan(
        domains=["scms_order"], confidence=1.0, follow_up=True)))
    set_scms_order_enabled(True)
    try:
        plan, ok = await _sup._decide(MagicMock(), "발주요청해줘", 0.6, "eapproval", prior_domains=["eapproval"])
        assert ok and plan.domains == ["scms_order"]
        assert plan.action_query("scms_order") == "발주요청해줘"     # 승인 도구 하나뿐 — 조회로 태우면 도구가 빠진다
        assert plan.read_query("scms_order") is None
    finally:
        set_scms_order_enabled(False)
    # 꺼진 요청이면 라우터가 그렇게 내도 종전 상속 경로
    plan, ok = await _sup._decide(MagicMock(), "발주요청해줘", 0.6, "eapproval", prior_domains=["eapproval"])
    assert ok and plan.domains == ["eapproval"]


@pytest.mark.asyncio
async def test_decide_follow_up_prior_domain_is_a_hint_not_override(monkeypatch):
    """운영 서버 사용자(2026-09-29): 웹을 끈 채 물은 턴이 memory 로 처리된 뒤 웹을 켜고 '웹 검색으로 오늘 기준 환율'을 물었는데,
    라우터 web(0.95~1.00)을 follow-up 상속이 세 번 memory 로 덮었다. 확신 있는 라우터 판단이 직전 도메인보다 우선한다."""
    from agent_arch.agents.context import set_web_search_enabled
    set_web_search_enabled(True)
    async def decide(plan, prior):
        monkeypatch.setattr(_sup, "route", AsyncMock(return_value=plan))
        return await _sup._decide(MagicMock(), "아뇨 웹 검색으로 오늘 기준 환율 알려주세요", 0.6, prior[0], prior_domains=prior)
    plan, ok = await decide(RoutePlan(domains=["web"], subqueries={"web": "오늘 기준 환율"}, confidence=1.0, follow_up=True,
                                      web_queries=["원달러 환율 오늘"]), ["memory"])
    assert ok and plan.domains == ["web"] and plan.web_queries == ["원달러 환율 오늘"]
    plan, ok = await decide(RoutePlan(domains=["web", "memory"], confidence=0.95, follow_up=True), ["memory"])
    assert ok and plan.domains == ["web", "memory"]                                   # 전엔 web 이 빠지고 memory 만 남았다
    # 확신이 기준(0.85) 아래면 종전대로 직전 도메인으로 채운다(9/15 '근거가 뭐야' → talk 0.70 오분류 보호)
    plan, ok = await decide(RoutePlan(domains=["talk"], confidence=0.7, follow_up=True), ["web"])
    assert ok and plan.domains == ["web"]
    from agent_arch.core.config import settings
    monkeypatch.setattr(settings, "router_followup_override_confidence", 0.65, raising=False)
    plan, ok = await decide(RoutePlan(domains=["talk"], confidence=0.7, follow_up=True), ["web"])
    assert ok and plan.domains == ["talk"]                                            # 기준은 설정으로 조정


@pytest.mark.asyncio
async def test_decide_non_follow_up_unchanged(monkeypatch):
    monkeypatch.setattr(_sup, "route", AsyncMock(return_value=RoutePlan(domains=["hr"], subqueries={"hr": "연차"}, confidence=0.9)))
    plan, ok = await _sup._decide(MagicMock(), "내 연차 며칠 남았어", 0.6, "web", prior_domains=["web"])
    assert ok and plan.domains == ["hr"]                                        # 새 요청은 직전 도메인에 끌려가지 않는다


@pytest.mark.asyncio
async def test_last_route_roundtrip(monkeypatch):
    import agent_arch.memory.conversation as conv
    store = {}
    class _R:
        async def set(self, k, v, expire=None):
            store[k] = v; return True
        async def get(self, k):
            return store.get(k)
    monkeypatch.setattr(conv, "redis_store", _R())
    await conv.save_last_route("cid1", ["web", "talk"])
    assert await conv.load_last_route("cid1") == ["web", "talk"]
    await conv.save_last_route("cid1", [])
    assert await conv.load_last_route("cid1") == [] and await conv.load_last_route(None) == [] and await conv.load_last_route("nope") == []


def test_scope_messages_appends_retrieved_tool_guidance_only(monkeypatch):
    from agent_arch.agents.tools.exposure import Exposure
    core = MagicMock(); core.name = "search_colleagues"; core.prompt_guidance = "### 코어 지침"
    web = MagicMock(); web.name = "brave_web_search"; web.prompt_guidance = "### 웹 검색(brave_web_search) 사용 원칙"
    monkeypatch.setattr(_sup, "settings", MagicMock(supervisor_preserve_original=True))
    import agent_arch.agents.tools.exposure as exp
    monkeypatch.setattr(exp, "resolve_exposure", lambda t: Exposure.CORE if t.name == "search_colleagues" else Exposure.RETRIEVED)
    msgs = [Message(role="system", content="sys"), Message(role="user", content="q")]
    out = _sup._scope_messages(msgs, "q", "web", tools=[core, web])
    assert "웹 검색(brave_web_search) 사용 원칙" in out[0].content and "코어 지침" not in out[0].content
    assert "너는 'web' 도메인 전담" in out[0].content and "근거를 확보" not in out[0].content
    assert "출처' 표기로" in out[0].content or "출처" in out[0].content


def test_merge_appends_web_sources_footer():
    import json
    tr = {"tool_call_id": "c", "name": "brave_web_search", "content": json.dumps({"results": [{"title": "T", "url": "https://t.com/1"}]})}
    web = LoopResult(content="웹 워커 답", tool_results=[tr])
    talk = LoopResult(content="사내엔 없음")
    plan = RoutePlan(domains=["web", "talk"], confidence=0.9)
    merged = _sup._merge([("web", web), ("talk", talk)], "합성된 답변(URL 없음)", None, plan)
    assert merged.content.endswith("출처(웹):\n- T — https://t.com/1")
    merged2 = _sup._merge([("web", web)], "이미 https://t.com/1 를 적은 답", None, plan)
    assert "출처(웹)" not in merged2.content


def test_worker_directive_web_addendum_only_for_web():
    """웹 워커에만 조사 절차(검색어 2개+·본문 확인·무관 결과 버림·되묻기)가 붙고 격리 원칙은 공통."""
    web = _sup._worker_directive("web")
    mail = _sup._worker_directive("mail")
    assert "'web' 도메인 전담" in web and "노력은 질문에 맞춰라" in web and "brave_llm_context" in web and "되물어라" in web
    assert "'mail' 도메인 전담" in mail and "노력은 질문에 맞춰라" not in mail
    assert web.startswith(mail.split("'mail'")[0])   # 공통 앞부분 동일


@pytest.mark.asyncio
async def test_supervisor_passes_recent_history_to_router(monkeypatch):
    """라우터는 이번 문장만 보고 'Kim 관련된거'를 사내 사람 검색으로 보냈다(실측 2026-09-16). 이제 앞 턴 발췌를 함께 받는다."""
    captured = {}
    async def fake_route(llm, query, *, prior_domain=None, prior_domains=None, history="", **kw):
        captured["history"] = history; captured["query"] = query
        return RoutePlan(domains=["web"], subqueries={"web": "PSE 연구자 Kim"}, confidence=0.9, follow_up=True)
    monkeypatch.setattr(_sup, "route", fake_route)
    monkeypatch.setattr(_sup, "_load_prior_domains", AsyncMock(return_value=["web"]))
    monkeypatch.setattr(_sup, "_save_route", AsyncMock())
    worker = MagicMock(); worker.stream = MagicMock(); worker.bind_tools = AsyncMock()
    async def _gen(*a, **k):
        yield StreamEvent(type="text", content="Kim 교수의 PSE 연구 요약")
        yield StreamEvent(type="done", finish_reason="stop")
    worker.stream = _gen
    msgs = [Message(role="user", content="PSE 에서 데이터 없는 기계 파라미터 추정 연구 있어?"),
            Message(role="assistant", content="선형 응답 기반 추정 등 4건이 있습니다. " + "x" * 700),
            Message(role="user", content="Kim 관련된거 찾아봐")]
    r = await run_supervisor(MagicMock(), "Kim 관련된거 찾아봐", msgs, resolve_tools_fn=AsyncMock(return_value=[]), make_worker_llm_fn=lambda: worker,
                             fallback_fn=AsyncMock(), router_llm=MagicMock(), max_iterations=3)
    assert captured["query"] == "Kim 관련된거 찾아봐"
    assert captured["history"].startswith("사용자: PSE 에서 데이터 없는 기계 파라미터 추정 연구 있어?\n어시스턴트: 선형 응답 기반")
    assert "Kim 관련된거 찾아봐" not in captured["history"] and len(captured["history"]) < 700          # 이번 턴 제외·발화 600자 상한
    assert r.content.startswith("Kim 교수의 PSE 연구 요약")
class TestWriteSubrunCanBuildWhatItSends:
    """"만들어서 보내줘" 는 만드는 것까지가 그 실행이다 — 2026-09-11 실측 회귀.

    글로벌 유틸은 원칙상 READ 서브런 전용이다(막연한 실행 요청에 웹검색·계산기를 얹으면
    초안 대신 엉뚱한 도구를 맴돈다). 그래서 WRITE 는 `_WRITE_GLOBAL_ALLOWLIST` 만 받는데,
    `create_file` 이 빠져 있어 "분석 내용을 워드 파일로 생성하여 전송" 이 WRITE 로
    라우팅된 순간 워커가 **붙일 파일을 만들 수단 없이** 전송 도구만 쥐게 됐다:

        [supervisor] worker domain=talk WRITE q='분석 내용을 워드 파일로 생성하여 전송'
        [executor] call tool=create_file
        [executor] tool NOT FOUND: create_file

    모델은 정확히 호출했다. 스코프에만 없었다.
    """

    def test_create_file_is_allowed_in_write_subrun(self):
        from agent_arch.agents.graphs.supervisor import _WRITE_GLOBAL_ALLOWLIST

        assert "create_file" in _WRITE_GLOBAL_ALLOWLIST, (
            "WRITE 서브런에서 파일을 만들 수 없으면 '만들어서 보내줘' 가 완수 불가다"
        )

    def test_recipient_resolution_stays_allowed(self):
        """대조군 — 기존 예외가 사라지면 수신자 해석·가이드 근거가 끊긴다."""
        from agent_arch.agents.graphs.supervisor import _WRITE_GLOBAL_ALLOWLIST

        assert {"search_colleagues", "search_user_guide_kb"} <= _WRITE_GLOBAL_ALLOWLIST

    def test_write_allowlist_stays_small(self):
        """글로벌을 통째로 열면 WRITE 가 초안 대신 도구를 맴돈다 — 예외는 예외로 둔다."""
        from agent_arch.agents.graphs.supervisor import _WRITE_GLOBAL_ALLOWLIST

        assert len(_WRITE_GLOBAL_ALLOWLIST) <= 5



# ── READ → WRITE 인계 (2026-09-15 회귀) ──────────────────────────────────────
#
# 두 서브런은 같은 messages 를 받는다 — WRITE 는 READ 가 무엇을 찾았는지 몰랐다.
# 실측: READ 가 "에이전트_요금제_안내.xlsx" 를 찾아냈는데 WRITE 는 "찾은 파일을 전송하기"
# 라는 지시만 받고 같은 조회 도구를 다시 부르다 전송에 도달하지 못했다(post_message 0회).

def test_handoff_carries_what_read_found():
    from agent_arch.agents.graphs.supervisor import _read_handoff
    from agent_arch.agents.react.loop import LoopResult

    block = _read_handoff(LoopResult(
        content="이 스레드에서 요약된 엑셀 파일은 에이전트_요금제_안내.xlsx 입니다.",
        finish_reason="stop"))

    assert "[조회 결과]" in block
    assert "에이전트_요금제_안내.xlsx" in block
    assert "다시 조회하지 말고" in block


def test_handoff_is_empty_without_read():
    """READ 서브런이 없으면(actions 만 준 턴) 붙일 것이 없다."""
    from agent_arch.agents.graphs.supervisor import _read_handoff
    from agent_arch.agents.react.loop import LoopResult

    assert _read_handoff(None) == ""
    assert _read_handoff(LoopResult(content="   ", finish_reason="stop")) == ""


def test_handoff_is_capped():
    """조회 답변 전문을 그대로 넣으면 WRITE 프롬프트가 부푼다."""
    from agent_arch.agents.graphs.supervisor import _HANDOFF_MAX_CHARS, _read_handoff
    from agent_arch.agents.react.loop import LoopResult

    block = _read_handoff(LoopResult(content="가" * 5000, finish_reason="stop"))

    assert len(block) < _HANDOFF_MAX_CHARS + 200
    assert block.rstrip().endswith("…")


@pytest.mark.asyncio
async def test_write_subrun_receives_read_result(monkeypatch):
    """WRITE 워커의 마지막 user 메시지에 READ 가 찾은 내용이 실려야 한다."""
    captured = []

    async def fake_worker(llm, query, messages, tools, search_fn, max_iter, **kwargs):
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        captured.append({"q": query, "user": last_user})
        if any("send" in t.name for t in tools):
            return LoopResult(content="전송했습니다", finish_reason="stop")
        return LoopResult(content="에이전트_요금제_안내.xlsx 를 찾았습니다", finish_reason="stop")

    monkeypatch.setattr(_sup, "run_react_graph", fake_worker)
    router = _llm_json('{"domains":["mail"],"subqueries":{"mail":"파일 찾기"},'
                       '"actions":{"mail":"찾은 파일 전송"},"confidence":0.9}')

    async def resolve(query, domain):
        if "전송" in query:
            return [_make_tool("mail_send_message"), _make_tool("mail_list_messages")]
        return [_make_tool("mail_list_messages")]

    await run_supervisor(
        MagicMock(), "파일 찾아서 전송해줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_single_answer_specs),
        fallback_fn=AsyncMock(), router_llm=router,
    )

    write = next(c for c in captured if "전송" in c["q"])
    read = next(c for c in captured if "찾기" in c["q"])
    assert "에이전트_요금제_안내.xlsx" in write["user"]     # READ 결과가 WRITE 로 넘어갔다
    assert "에이전트_요금제_안내.xlsx" not in read["user"]  # READ 자신에게는 붙지 않는다


# ── 합성 답변의 근거 없는 가이드 링크 차단 ────────────────────────────────────

_FORM_URL = "https://example.com/main/support/inquiry/write"
_FAKE_GUIDE_URL = "https://example.com/user-guide/agent-deep-dive"


@pytest.mark.asyncio
async def test_synth_drops_fabricated_guide_links_and_emits_policy_first():
    """합성은 워커 결과를 **새로 쓰는** LLM 이라 제목만 아는 가이드 문서에 주소를 붙일 수 있다.

    병렬 워커는 ainvoke 라 자기 정책을 흘리지 못하므로 supervisor 가 synth 직전에 대신 보낸다 —
    그 이벤트가 합성 토큰보다 먼저 나가야 화면 정화기가 제때 물린다.
    """
    router = _llm_json('{"domains":["guide","talk"],"confidence":0.9}')
    main = MagicMock()
    main.bind_tools = AsyncMock()
    main.stream = _stream_text(
        f"문의는 [문의 폼]({_FORM_URL}) 에서 가능합니다. "
        f"더 자세한 내용은 아래 가이드를 참고해 주세요. [Agent 사용 심층 가이드]({_FAKE_GUIDE_URL})")

    async def resolve(subquery, domain):
        return [_make_tool("search_user_guide_kb",
                           output=[{"category": "General", "category_top_score": 0.8, "chunks": [
                               {"title": "고객 문의", "content": f"회원 문의 폼( {_FORM_URL} )을 이용하세요.",
                                "score": 0.8}]}])]

    def _worker_specs():
        return [_stream_tc([ToolCall(id="t1", name="search_user_guide_kb", arguments={"query": "q"})]),
                _stream_text("워커 답변")]

    events = []
    async for ev in stream_supervisor(
        main, "에이전트 사용방법 안내해줘", [Message(role="user", content="q")],
        resolve_tools_fn=resolve, make_worker_llm_fn=_wllm_factory(_worker_specs),
        fallback_stream_fn=None, router_llm=router,
    ):
        events.append(ev)

    final = events[-1]
    assert isinstance(final, LoopResult)
    assert _FAKE_GUIDE_URL not in final.content
    assert "가이드를 참고해 주세요" not in final.content
    assert _FORM_URL in final.content

    kinds = [e.event_type for e in events if isinstance(e, ProgressEvent)]
    assert "link_policy" in kinds and kinds.index("link_policy") < kinds.index("token")


def test_web_worker_globals_are_narrowed_and_finder_skipped_when_bound():
    """벤치(2026-09-23): 웹 워커에 사내 디렉터리·가이드가 얹혀 오호출, 바인딩된 brave_* 를 두고 관문부터 부르는 헛걸음."""
    from types import SimpleNamespace as NS
    from agent_arch.agents.graphs import supervisor as S
    g = [NS(name=n) for n in ("calculator", "create_file", "search_colleagues", "search_user_guide_kb", "memory_search")]
    assert [t.name for t in S._domain_globals(g, "web")] == ["calculator", "create_file"]
    assert len(S._domain_globals(g, "mail")) == 5                                    # 다른 도메인은 그대로
    web = [NS(name=n) for n in ("brave_llm_context", "brave_web_search", "brave_news_search")]
    assert S._with_domain_finder(web, "web") == web                                  # 셋 다 손에 있으면 관문 없음


def test_make_worker_passes_domain_only_to_factories_that_take_it():
    from agent_arch.agents.graphs import supervisor as S
    seen = []
    assert S._make_worker(lambda: "plain", "web") == "plain"                          # 옛 팩토리(테스트 대역)
    def f(domain=None):
        seen.append(domain); return "d"
    assert S._make_worker(f, "web") == "d" and seen == ["web"]


def test_orchestrator_worker_llm_applies_domain_overrides(monkeypatch):
    from agent_arch.agents import orchestrator as O
    from agent_arch.core.config import settings
    got = {}
    monkeypatch.setattr(settings, "llm_providers", {**settings.llm_providers,
                        "worker_overrides": {"web": {"vllm": {"chat_template_kwargs": {"enable_thinking": True}}}}}, raising=False)
    monkeypatch.setattr(O.LLMFactory, "create", classmethod(lambda cls, p=None, **kw: got.update(p=p, **kw) or "llm"))
    orch = O.AgentOrchestrator.__new__(O.AgentOrchestrator); orch.llm_provider = "vllm"
    assert orch._make_worker_llm("web") == "llm" and got["chat_template_kwargs"] == {"enable_thinking": True}
    got.clear(); orch._make_worker_llm("mail")
    assert "chat_template_kwargs" not in got                                          # 항목 없는 도메인은 대화 레인 그대로


def test_web_directive_quote_rule_and_no_market_hours_assumption():
    """실측(2026-09-28 08:34): '개장 전' 문구 + 옛 지식(9시 개장)으로 '최신은 9/26' — 원·달러는 2026-07 부터 24시간."""
    from agent_arch.agents.graphs.supervisor import _DOMAIN_ADDENDA
    web = _DOMAIN_ADDENDA["web"]
    assert "개장 전" not in web and "가장 최근 시각의 값" in web and "네 기억으로 판단하지 말고" in web


def test_followup_inheritance_keeps_web_plan_and_fallback_gets_query():
    """운영 서버(2026-09-28): 후속 '오늘 유로 기준환율'에서 상속 경로가 라우터 검색어를 버려 선실행·충분성 루프가 빠졌다."""
    from agent_arch.agents.graphs import supervisor as S
    from agent_arch.agents.graphs.router import RoutePlan
    p = RoutePlan(domains=["web"], web_queries=["유로 원 기준환율"], web_recency="day", web_open=True)
    assert S._web_plan(p) == {"web_queries": ["유로 원 기준환율"], "web_recency": "day", "web_open": True}
    assert S._web_plan(None, "오늘  유로 기준환율 확인해줘")["web_queries"] == ["오늘 유로 기준환율 확인해줘"]
    assert S._web_plan(RoutePlan(domains=["web"]), "")["web_queries"] == []


def test_web_worker_prompt_drops_internal_answer_structure_sections():
    """'오늘 환율'에 묻지 않은 통화 표 + 인사이트 + 제안(2026-09-28) — 사내용 답변 구조·인사이트 의무를 웹 워커에서 뺀다."""
    from agent_arch.agents.graphs import supervisor as S
    prompt = "머리말\n## 답변 구조\n① 핵심\n④ 제안\n## 데이터 해석 원칙\n- 유지\n## 에이전틱 답변 원칙 (반드시 준수)\n### 인사이트 도출 의무\n- x\n### 금지 패턴\n- 인사이트·제안 섹션이 1줄 이하인 답변\n"
    out = S._strip_internal_answer_sections(prompt)
    assert "답변 구조" not in out and "인사이트 도출 의무" not in out and "데이터 해석 원칙" in out and "머리말" in out


def test_draft_note_uses_card_note_when_given():
    """초안 카드가 자기 안내(draft_note)를 주면 공통 '발송/등록 초안' 대신 그것을 쓴다(물류 이관 2026-10-13)."""
    own = [{"tool_name": "logistics_migrate_drive_excel", "draft_view": {"draft_note": "엑셀 2개를 확인했어요."}}]
    assert _sup._draft_note(own) == "엑셀 2개를 확인했어요."
    assert _sup._draft_note([{"tool_name": "mail_send_message", "draft_view": {}}]) == _sup._DRAFT_NOTE
    assert _sup._draft_note([{"tool_name": "x", "draft_view": None}]) == _sup._DRAFT_NOTE
