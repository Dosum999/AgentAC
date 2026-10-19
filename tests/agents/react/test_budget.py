"""턴 예산 — 웹 호출 중복 억제·상한. 판단은 코드가 자원만 본다(의도 감지 아님)."""
import json

from agent_arch.agents.react import budget as B
from agent_arch.core.config import settings


def test_canonical_key_normalizes_args():
    a = B.canonical_key("brave_web_search", {"query": "  Polymer  Price ", "count": 8, "country": None})
    b = B.canonical_key("brave_web_search", {"count": 8, "query": "polymer price"})
    assert a == b
    assert B.canonical_key("brave_web_search", {"query": "polymer price"}) != B.canonical_key("brave_news_search", {"query": "polymer price"})
    assert B.canonical_key("x", None) == "x|{}"


def _dup_err(cid="c1"):
    return B._error(B.DUPLICATE_CODE, f"x {cid}")


def test_guard_dedups_prior_calls_but_allows_retry_after_error(monkeypatch):
    monkeypatch.setattr(settings, "web_max_calls_per_turn", 8, raising=False)
    prior_calls = [{"id": "c1", "name": "brave_llm_context", "arguments": {"query": "폴리머 가격"}},
                   {"id": "c2", "name": "brave_web_search", "arguments": {"query": "kim pse"}},
                   {"id": "c3", "name": "brave_web_search", "arguments": {"query": "kim pse"}},          # 이미 중복 차단됐던 호출
                   {"id": "c4", "name": "brave_news_search", "arguments": {"query": "실패한 검색"}},
                   {"id": "c5", "name": "mattermost_find_channels", "arguments": {"q": "x"}}]
    prior_results = [{"tool_call_id": "c1", "content": json.dumps({"kind": "context", "results": []})},
                     {"tool_call_id": "c2", "content": json.dumps({"kind": "results", "results": []})},
                     {"tool_call_id": "c3", "content": _dup_err("c2")},
                     {"tool_call_id": "c4", "content": "[ERROR] provider_error: 서버 오류"},
                     {"tool_call_id": "c5", "content": "[]"}]
    g = B.WebGuard(prior_calls, prior_results)
    assert g.web_calls == 3 and g.duplicates == 0                                            # 가드 차단(c3)은 실행된 호출이 아니다
    err = g.check("brave_llm_context", {"query": " 폴리머   가격"}, "n1")
    assert err and err.startswith("[ERROR] duplicate_call:") and "tool_call_id c1" in err and "다른 검색어" in err
    assert g.check("brave_news_search", {"query": "실패한 검색"}, "n2") is None            # 오류였던 호출은 재시도 허용
    assert g.check("mattermost_find_channels", {"q": "x"}, "n3") is None                     # 사내 도구는 대상 아님
    assert g.check("brave_web_search", {"query": "new query"}, "n4") is None
    assert g.check("brave_web_search", {"query": "NEW QUERY"}, "n5").startswith("[ERROR] duplicate_call:")   # 같은 배치 안 중복
    st = g.stats()
    assert {k: st[k] for k in ("web_calls", "duplicates", "capped", "cap")} == {"web_calls": 5, "duplicates": 2, "capped": 0, "cap": 8}


def test_guard_caps_web_calls_per_turn(monkeypatch):
    monkeypatch.setattr(settings, "web_max_calls_per_turn", 2, raising=False)
    g = B.WebGuard([], [])
    assert g.check("brave_web_search", {"query": "a"}, "1") is None and g.check("brave_web_search", {"query": "b"}, "2") is None
    err = g.check("brave_web_search", {"query": "c"}, "3")
    assert err.startswith("[ERROR] web_budget_exceeded:") and "상한(2회)" in err and "지금 있는 근거로 답하라" in err
    assert g.check("calendar_list", {}, "4") is None                                           # 상한은 웹 도구에만
    assert g.stats()["capped"] == 1 and g.stats()["web_calls"] == 2
    monkeypatch.setattr(settings, "web_max_calls_per_turn", 0, raising=False)                   # 0 = 무제한
    g2 = B.WebGuard([], [])
    assert all(g2.check("brave_web_search", {"query": str(i)}, str(i)) is None for i in range(20))
    monkeypatch.setattr(settings, "web_max_calls_per_turn", "bad", raising=False)
    assert B.cap() == 8


def test_is_guard_error_and_summarize():
    assert B.is_guard_error(_dup_err()) == "duplicate_call" and B.is_guard_error(B._error(B.CAP_CODE, "m")) == "web_budget_exceeded"
    assert B.is_guard_error("[ERROR] provider_error: x") is None and B.is_guard_error('{"kind": "results"}') is None and B.is_guard_error(None) is None
    calls = [{"id": "1", "name": "brave_web_search"}, {"id": "2", "name": "brave_web_search"}, {"id": "3", "name": "brave_llm_context"}, {"id": "4", "name": "calc"}]
    results = [{"tool_call_id": "1", "content": "{}"}, {"tool_call_id": "2", "content": _dup_err()}, {"tool_call_id": "3", "content": B._error(B.CAP_CODE, "m")}, {"tool_call_id": "4", "content": "1"}]
    s = B.summarize(calls, results)
    assert s["web_calls"] == 1 and s["duplicates"] == 1 and s["capped"] == 1 and s["cap"] == B.cap()


def test_guard_caps_rounds_but_allows_parallel_calls_in_a_round(monkeypatch):
    """벤치(2026-09-23): 호출 수 상한만으론 한 스텝에 하나씩 순차로 8번 — 단계(스텝) 수로도 막는다. 한 단계 안 병렬은 허용."""
    monkeypatch.setattr(settings, "web_max_calls_per_turn", 8, raising=False)
    monkeypatch.setattr(settings, "web_max_rounds_per_turn", 3, raising=False)
    g = B.WebGuard([], [], rounds=2)
    assert all(g.check("brave_llm_context", {"query": f"q{i}"}, f"c{i}") is None for i in range(4))   # 3번째 단계: 4개 병렬 OK
    g3 = B.WebGuard([], [], rounds=3)
    err = g3.check("brave_web_search", {"query": "more"}, "x")
    assert err and err.startswith("[ERROR] web_budget_exceeded:") and "3단계" in err and "가장 최근 시각의 값" in err
    assert g3.check("calculator", {"expression": "1+1"}, "y") is None                                # 사내·계산 도구는 대상 아님
    monkeypatch.setattr(settings, "web_max_rounds_per_turn", 0, raising=False)
    assert B.WebGuard([], [], rounds=99).check("brave_web_search", {"query": "q"}, "z") is None       # 0 = 무제한


def test_executed_web_ignores_guard_blocks_and_internal_tools():
    from types import SimpleNamespace as NS
    blocked = B._error(B.CAP_CODE, "x")
    assert B.executed_web([(NS(name="brave_web_search"), '{"kind":"results"}')]) is True
    assert B.executed_web([(NS(name="brave_web_search"), blocked), (NS(name="calculator"), "2")]) is False
    assert B.executed_web([(NS(name="brave_news_search"), "[ERROR] provider_error: x")]) is True    # 실패도 외부 호출은 했다


def test_code_driven_calls_are_exempt_from_model_caps_but_deduped(monkeypatch):
    """선실행·충분성 라운드(prefetch_ id)는 모델 몫 호출 상한을 쓰지 않는다 — 쓰면 코드가 더 찾자마자 모델이 한 번도 못 찾는다."""
    monkeypatch.setattr(settings, "web_max_calls_per_turn", 2, raising=False)
    monkeypatch.setattr(settings, "web_max_rounds_per_turn", 3, raising=False)
    prior = [{"id": f"prefetch_brave_llm_context__{i}", "name": "brave_llm_context", "arguments": {"query": f"q{i}"}} for i in range(5)]
    res = [{"tool_call_id": c["id"], "content": json.dumps({"kind": "context", "results": []})} for c in prior]
    g = B.WebGuard(prior, res)
    assert g.web_calls == 0 and g.check("brave_web_search", {"query": "model"}, "m1") is None       # 모델 몫은 그대로 남았다
    assert g.check("brave_llm_context", {"query": "q1"}, "prefetch_x").startswith("[ERROR] duplicate_call:")   # 코드 호출도 중복은 막힌다
    assert g.check("brave_llm_context", {"query": "new"}, "prefetch_y") is None
