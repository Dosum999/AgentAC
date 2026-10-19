"""답변 검증기 — 증거 사실 계산·판정 파싱·보강 지시·최근 대화·도구 요약·대상 판정. LLM 은 부르지 않는다."""
import json
from types import SimpleNamespace

from agent_arch.agents.react import verify as V
from agent_arch.core.config import settings
from agent_arch.llm.base import Message


def _env(t):
    return f'<<<EXTERNAL_UNTRUSTED_CONTENT id="ab">>>\nSource: 웹 검색\n---\n{t}\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id="ab">>>'


def _call(cid, name, **args):
    return {"id": cid, "name": name, "arguments": args}


def _result(cid, name, rows, kind="results"):
    return {"tool_call_id": cid, "name": name, "content": json.dumps({"kind": kind, "tool": name, "count": len(rows), "results": rows}, ensure_ascii=False)}


def _state(calls, results, *, tools=(), used=None, drafts=None):
    return {
        "collected_tool_calls": calls, "collected_tool_results": results,
        "all_tools": [SimpleNamespace(name=t) for t in tools],
        "used_tools": bool(calls) if used is None else used, "pending_drafts": drafts or [],
    }


# ─── 증거 사실 ─────────────────────────────────────────────────────────────────

def test_collect_facts_counts_searches_context_dates_and_unknown_urls():
    calls = [
        _call("c1", "brave_web_search", query="브라켓 시장", freshness="pm"),
        _call("c2", "brave_web_search", query="industrial bracket market"),
        _call("c3", "brave_llm_context", query="industrial bracket market trends"),
        _call("c4", "mattermost_find_channels"),
    ]
    results = [
        _result("c1", "brave_web_search", [{"url": "https://a.com/1", "title": "A", "published": "2026-09-01", "content": _env("x")}, {"url": "https://a.com/2", "title": "B", "content": ""}]),
        _result("c2", "brave_web_search", []),
        _result("c3", "brave_llm_context", [{"url": "https://c.com/1", "title": "C", "content": _env("본문")}], kind="context"),
        {"tool_call_id": "c4", "name": "mattermost_find_channels", "content": "[ERROR] 401"},
    ]
    draft = "요약. 근거: https://a.com/1 그리고 https://zzz.com/made-up 참고."
    f = V.collect_facts(_state(calls, results), draft)
    assert f["tool_calls"] == 4 and f["tool_errors"] == 1
    assert f["web_searches"] == 2 and f["context_reads"] == 1 and f["empty_searches"] == 1 and f["freshness_requested"] is True
    assert f["results_total"] == 3 and f["results_with_date"] == 1
    assert f["answer_urls_not_in_results"] == ["https://zzz.com/made-up"]          # 결과에 있는 a.com/1 은 빠진다
    assert f["duplicate_calls"] == 0 and f["budget_capped"] == 0
    lines = V._facts_lines(f)
    assert "웹 검색 2회, 페이지 본문 조각 조회 1회(빈 결과 1회)" in lines and "최신성 필터 사용" in lines and "zzz.com" in lines


def test_collect_facts_no_tools_is_quiet():
    f = V.collect_facts(_state([], []), "그냥 답")
    assert f["tool_calls"] == 0 and f["web_searches"] == 0 and f["context_reads"] == 0 and f["answer_urls_not_in_results"] == []
    assert "웹 검색" not in V._facts_lines(f)


# ─── 판정 파싱·보강 지시 ─────────────────────────────────────────────────────────

def test_parse_verdict_lenient_and_defaults():
    v = V.parse_verdict('```json\n{"verdict": "REVISE", "issues": ["최근 논문 요청에 2017년 위키 항목", "", 3], "action": {"type": "search", "detail": "attention mechanism 2026 arxiv"}}\n```')
    assert v["verdict"] == "revise" and v["issues"] == ["최근 논문 요청에 2017년 위키 항목", "3"]
    assert v["action"] == {"type": "search", "detail": "attention mechanism 2026 arxiv"}
    assert V.parse_verdict('{"verdict": "revise"}')["action"]["type"] == "state_limits"          # 행동 없는 revise 는 한계 명시로
    assert V.parse_verdict("판단: 괜찾아요")["verdict"] == "pass" and V.parse_verdict("x").get("parse_error") is True
    assert V.parse_verdict('{"verdict": "maybe", "action": {"type": "dance"}}') == {"verdict": "pass", "issues": [], "action": {"type": "none", "detail": ""}}


def test_feedback_message_is_user_role_with_action():
    m = V.feedback_message({"verdict": "revise", "issues": ["산업용 브라켓 대신 배터리 시장으로 채움"], "action": {"type": "state_limits", "detail": "배터리 문단 제거"}})
    assert m.role == "user" and m.name == V.FEEDBACK_NAME
    assert "배터리 문단 제거" in m.content and "산업용 브라켓 대신" in m.content and "이 메시지 자체를 언급하지 마라" in m.content
    assert "되물어라" in V.feedback_message({"action": {"type": "clarify", "detail": "치과용인가요, 산업용인가요?"}}).content
    assert "web_fetch" in V.feedback_message({"action": {"type": "fetch", "detail": "PP 가격"}}).content


# ─── 최근 대화·본문 ─────────────────────────────────────────────────────────────

def test_recent_history_keeps_prior_turns_and_drops_tool_and_feedback_noise():
    msgs = [
        Message(role="system", content="sys"),
        Message(role="user", content="폴리머 최근 동향 알려줘"),
        Message(role="assistant", content="이번 주 폴리머 동향입니다. " + "x" * 800 + "\n\n출처(웹):\n- 폴리메시 코인 — https://tokenpost.kr/1"),
        Message(role="user", content="[도구 결과] brave_web_search: {...}"),
        Message(role="user", content="가격 관련해서 말이야"),
        Message(role="assistant", content="", tool_calls=[{"id": "t", "name": "brave_web_search"}]),
        Message(role="tool", content="{}", tool_call_id="t"),
        Message(role="assistant", content="폴리머 가격 초안"),
        V.feedback_message({"issues": ["x"], "action": {"type": "search", "detail": "d"}}),   # 보강 라운드: 초안+지시가 끝에 있다
    ]
    h = V.recent_history(msgs)
    assert h.startswith("사용자: 폴리머 최근 동향 알려줘\n어시스턴트: 이번 주 폴리머 동향입니다.")
    assert "출처(웹)" not in h and "tokenpost" not in h                       # 답은 앞 500자만 → 각주 탈락
    assert "[도구 결과]" not in h and "초안" not in h and V.FEEDBACK_NAME not in h and "가격 관련해서" not in h
    assert V.recent_history([Message(role="user", content="첫 질문")]) == "" and V.recent_history(None) == ""


def test_build_messages_shape_history_and_truncation():
    st = _state([_call("c1", "brave_web_search", query="q")], [_result("c1", "brave_web_search", [{"url": "https://a.com/1", "title": "x" * 900, "content": _env("y" * 900)}])])
    st["messages"] = [Message(role="user", content="폴리머 동향"), Message(role="assistant", content="동향 답"), Message(role="user", content="가격 관련해서 말이야")]
    msgs = V.build_messages("가격 관련해서 말이야", "초안" * 5000, st, V.collect_facts(st, ""))
    assert msgs[0].role == "system" and msgs[0].content == V.VERIFY_SYSTEM_PROMPT and "최근 대화" in msgs[0].content
    body = msgs[1].content
    assert body.startswith("[현재 날짜] 20") and "[최근 대화 — 이번 요청 이전]\n사용자: 폴리머 동향\n어시스턴트: 동향 답" in body
    assert "[사용자 요청 — 이번 턴]\n가격 관련해서 말이야" in body and "[답 초안]" in body and "[이번 턴 도구 사용]" in body and "[증거 사실" in body
    assert len(body) < 7_000
    assert "(없음 — 첫 질문)" in V.build_messages("q", "d", _state([], []), V.collect_facts(_state([], []), ""))[1].content


def test_tool_summary_expands_rows_and_context_and_errors():
    rows = [{"title": f"결과 {i}", "url": f"https://r.com/{i}", "published": "2026-09-01" if i == 4 else None,
             "content": _env("PP 가격 4.0% 상승, 프로필렌 원료비 인상" if i == 4 else "x" * 900)} for i in range(5)]
    calls = [_call("c1", "brave_web_search", query="polypropylene price"), _call("c2", "brave_llm_context", query="PVC"),
             _call("c3", "brave_news_search", query="빈 검색"), _call("c4", "calc"), _call("c5", "brave_web_search", query="막힘")]
    results = [_result("c1", "brave_web_search", rows),
               _result("c2", "brave_llm_context", [{"title": "PVC 가격 리포트", "url": "https://p.com/a", "content": _env("유럽 PVC 가격은 1분기 견조한 상승세 " + "y" * 2000)}], kind="context"),
               _result("c3", "brave_news_search", []),
               {"tool_call_id": "c4", "name": "calc", "content": "[ERROR] 나눗셈 0"},
               {"tool_call_id": "c5", "name": "brave_web_search", "content": json.dumps({"error": "blocked_identifier", "message": "사내 식별자"})}]
    summary = V._tool_summary(_state(calls, results))
    assert "결과 5건" in summary and "4.0% 상승" in summary and "https://r.com/4 | 2026-09-01" in summary   # 5번째 행의 본문까지 보인다
    assert "EXTERNAL_UNTRUSTED" not in summary and "x" * 300 not in summary                                     # 봉투 제거·본문 200자
    assert "(페이지 본문 조각 포함)" in summary and "PVC 가격 리포트 | https://p.com/a" in summary and "유럽 PVC 가격은" in summary and "y" * 600 not in summary
    assert "결과 0건" in summary and "[ERROR] 나눗셈 0" in summary and "오류 blocked_identifier" in summary
    assert len(summary) <= V._MAX_TOOL_SUMMARY_CHARS + 50


# ─── 대상 판정·모드·레인 ──────────────────────────────────────────────────────────

def test_should_verify_and_hold(monkeypatch):
    monkeypatch.setattr(settings, "answer_verify_mode", "off", raising=False)
    assert V.mode() == "off" and not V.should_verify(_state([], [], tools=("brave_web_search",)))
    monkeypatch.setattr(settings, "answer_verify_mode", "shadow", raising=False)
    assert V.should_verify(_state([], [], tools=("brave_llm_context",)))                # 웹 도구가 손에 있으면 대상
    assert V.should_verify(_state([_call("c", "calendar_list")], [], used=True))
    assert not V.should_verify(_state([], [], tools=("calendar_list",)))                # 도구 안 쓴 사내 턴은 제외
    assert not V.should_verify(_state([_call("c", "brave_web_search")], [], drafts=[{"x": 1}]))   # 승인 카드 턴 제외
    assert not V.hold_needed(_state([], [], tools=("brave_web_search",)))               # shadow 는 토큰 안 붙든다
    monkeypatch.setattr(settings, "answer_verify_mode", "on", raising=False)
    assert V.hold_needed(_state([], [], tools=("brave_web_search",)))
    monkeypatch.setattr(settings, "answer_verify_mode", "weird", raising=False)
    assert V.mode() == "off"
    monkeypatch.setattr(settings, "answer_verify_max_rounds", "2", raising=False)
    assert V.max_rounds() == 2


def test_verify_lane_defaults_merge(monkeypatch):
    monkeypatch.setattr(settings, "llm_providers", {"verify": {"provider": "gemini", "model": "g"}}, raising=False)
    lane = V.verify_lane()
    assert lane["provider"] == "gemini" and lane["model"] == "g" and lane["max_tokens"] == 512
    # 레인이 provider 를 안 적으면 배포 기본값을 따른다 — 여기 박아 두면 그 프로바이더가
    # 죽었을 때 검증기만 따로 죽는다(2026-09-18 vLLM 정지 때 실측).
    monkeypatch.setattr(settings, "llm_providers", {}, raising=False)
    lane = V.verify_lane()
    assert "provider" not in lane and "model" not in lane and lane["max_tokens"] == 512

    captured = {}
    from agent_arch.llm.factory import LLMFactory
    monkeypatch.setattr(LLMFactory, "create", staticmethod(lambda p, **kw: captured.update({"p": p, **kw})))
    monkeypatch.setattr(settings, "default_llm_provider", "gemini", raising=False)
    V.create_verify_llm()
    assert captured["p"] == "gemini" and "model" not in captured


def test_collect_facts_counts_budget_guard_blocks():
    from agent_arch.agents.react import budget as B
    calls = [_call("c1", "brave_web_search", query="q"), _call("c2", "brave_web_search", query="q"), _call("c3", "brave_web_search", query="r")]
    results = [_result("c1", "brave_web_search", []),
               {"tool_call_id": "c2", "name": "brave_web_search", "content": B._error(B.DUPLICATE_CODE, "x")},
               {"tool_call_id": "c3", "name": "brave_web_search", "content": B._error(B.CAP_CODE, "y")}]
    f = V.collect_facts(_state(calls, results), "답")
    assert f["duplicate_calls"] == 1 and f["budget_capped"] == 1 and f["tool_errors"] == 0            # 가드 차단은 도구 오류로 세지 않는다
    assert "예산 가드: 중복 검색 차단 1회, 상한 차단 1회" in V._facts_lines(f)



def test_numbers_not_in_results_uses_full_result_text():
    """벤치(2026-09-23): 근거에 없는 수치가 최다 감점인데 검증기는 전부 pass — 행당 200~500자 요약으론 대조가 안 된다."""
    long_body = "가" * 3000 + " 9월 22일 종가 1358.2원, 전일 대비 22.8원 하락 "
    calls = [_call("c1", "brave_llm_context", query="원달러 환율")]
    results = [_result("c1", "brave_llm_context", [{"url": "https://yna.co.kr/v/1", "title": "환율", "content": _env(long_body)}], kind="context")]
    draft = "종가 **1,358.2원**(▼22.8원), 장중 저가 1,352.9원 ([연합](https://yna.co.kr/v/574680))"
    f = V.collect_facts(_state(calls, results), draft)
    assert f["numbers_not_in_results"] == ["1,352.9"]                                   # 쉼표 무시·전문 대조·URL 속 숫자 제외
    assert "1,352.9" in V._facts_lines(f) and "계산·환산·반올림" in V._facts_lines(f)
    assert V.collect_facts(_state([], []), draft)["numbers_not_in_results"] == []         # 웹 조회가 없는 턴은 대조하지 않는다


def test_prompt_has_grounding_and_no_search_rules():
    assert "도구 결과 전문 어디에도 없는 수치" in V.VERIFY_SYSTEM_PROMPT and "검색 0회로 최신·외부 사실" in V.VERIFY_SYSTEM_PROMPT


def test_prompt_forbids_asserting_nonexistence_from_absence():
    """실측(2026-09-23): 인덱스 조각이 8월까지라 '9월치 미공개'라고 단정 — 원 페이지엔 9월 가격이 있었다."""
    assert "존재 자체" in V.VERIFY_SYSTEM_PROMPT and "결과 범위로 바꿔" in V.VERIFY_SYSTEM_PROMPT


def test_fetch_action_points_to_web_fetch_and_rule_exists():
    """gemini 가 지침·결과 안내를 무시하고 web_fetch 0회로 '9월 자료 없음'(2026-09-23) — 검증기가 URL 을 지정해 열게 한다."""
    msg = V.feedback_message({"verdict": "revise", "issues": ["9월 단가 미확인"], "action": {"type": "fetch", "detail": "https://a.com/p"}})
    assert "web_fetch" in msg.content and "https://a.com/p" in msg.content
    assert "web_fetch 를 한 번도 쓰지" in V.VERIFY_SYSTEM_PROMPT


def test_newest_period_fact_points_to_freshest_source():
    """모델이 9월 표(IMARC)를 두고 8월 보도자료를 따라 '9월치 미발표' — 결과의 최신 시점과 출처를 사실로 댄다."""
    calls = [_call("c1", "brave_llm_context", query="epdm"), _call("c2", "web_fetch", url="https://www.imarcgroup.com/epdm")]
    results = [_result("c1", "brave_llm_context", [{"url": "https://openpr.com/a", "title": "EPDM August 2026", "content": _env("August 2026 prices")}], kind="context"),
               {"tool_call_id": "c2", "name": "web_fetch", "content": json.dumps({"final_url": "https://www.imarcgroup.com/epdm", "title": "EPDM",
                                                                                  "text": "EPDM Rubber Prices September 2026 table"})}]
    f = V.collect_facts(_state(calls, results), "8월이 최신이고 9월치는 미발표")
    assert f["newest_period_in_results"].startswith("2026-09 (www.imarcgroup.com)") or f["newest_period_in_results"] == "2026-09 (www.imarcgroup.com)"
    assert "가장 최근 연·월: 2026-09" in V._facts_lines(f)


def test_links_from_non_web_tools_count_as_known():
    """PROC 도구가 준 '바로 가기' 링크는 근거다 — 웹 결과만 보자 지어낸 URL 로 잡아 지우게 했다(실측 2026-10-01)."""
    calls = [{"id": "g1", "name": "proc_guide", "arguments": {"topic": "입고처리·출고처리 7단계"}}]
    results = [{"tool_call_id": "g1", "name": "proc_guide",
                "content": '{"메뉴": "출고처리", "바로 가기": ["https://procurement.example.com/supplier/logistics"]}'}]
    draft = "[출고처리](https://procurement.example.com/supplier/logistics) 에서 진행하세요. 참고 https://made.up/x"
    f = V.collect_facts(_state(calls, results), draft)
    assert f["answer_urls_not_in_results"] == ["https://made.up/x"]
