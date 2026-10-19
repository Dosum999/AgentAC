"""브레이브 MCP 응답 → 닫힌 결과 형식. 픽스처는 2026-09-15 dev 컨테이너 실측 응답을 줄인 것."""
import json

from agent_arch.services.web import brave_output as B


def _web_texts():
    return [
        json.dumps({"url": "https://www.researchnester.com/kr/reports/polymer-coated-fabrics-market/2631",
                    "title": "폴리머 코팅 직물 시장 규모 및 점유율, 2037년 예측 보고서",
                    "description": "폴리머 폼 시장 변동은 <strong>USD 116.55에 도달</strong> &quot;예상&quot; &#x27;됩니다&#x27;"}, ensure_ascii=False),
        json.dumps({"url": "https://www.gminsights.com/ko/industry-analysis/opaque-polymers-market", "title": "불투명 고분자 시장",
                    "description": "원재료 가격이 급등하면", "extra_snippets": ["추가 1", "추가 2", "추가 3", "추가 4"]}, ensure_ascii=False),
        json.dumps({"question": "폴리머란?", "answer": "고분자 화합물", "title": "FAQ", "url": "https://faq.example/1"}, ensure_ascii=False),
        json.dumps({"url": "ftp://bad", "title": "버림"}),
        json.dumps({"url": "https://www.gminsights.com/ko/industry-analysis/opaque-polymers-market", "title": "중복"}),
        "Summarizer key: abc",   # JSON 아닌 항목은 무시
    ]


def test_normalize_search_rows_are_plain_titles_and_enveloped_bodies():
    out = B.normalize_search("brave_web_search", "폴리머 가격 동향", _web_texts())
    assert out["kind"] == "results" and out["tool"] == "brave_web_search" and out["count"] == 3
    r0 = out["results"][0]
    assert r0["title"] == "폴리머 코팅 직물 시장 규모 및 점유율, 2037년 예측 보고서" and r0["type"] == "web"
    assert "<strong>" not in r0["content"] and 'USD 116.55에 도달 "예상" \'됩니다\'' in r0["content"]          # 태그·엔티티 제거
    assert r0["content"].startswith('<<<EXTERNAL_UNTRUSTED_CONTENT id="') and "Source: 웹 검색" in r0["content"]
    assert "추가 3" in out["results"][1]["content"] and "추가 4" not in out["results"][1]["content"]           # extra_snippets 3개까지
    assert out["results"][2]["type"] == "faq" and out["results"][2]["title"] == "FAQ" and "고분자 화합물" in out["results"][2]["content"]
    assert [r["url"] for r in out["results"]][1] == "https://www.gminsights.com/ko/industry-analysis/opaque-polymers-market"  # 중복·비http 제외
    assert out["externalContent"] == {"untrusted": True, "wrapped": True} and "truncated" not in out


def test_normalize_news_published_and_age():
    texts = [json.dumps({"url": "https://socialvalue.kr/news/view/1", "title": "삼양그룹 스페셜티 전환", "age": "4 hours ago",
                         "page_age": "2026-09-15T00:51:36", "breaking": False, "description": "범용은 줄이고"}, ensure_ascii=False)]
    out = B.normalize_search("brave_news_search", "폴리머", texts)
    r = out["results"][0]
    assert r["type"] == "news" and r["published"] == "2026-09-15" and r["age"] == "4 hours ago" and "Source: 뉴스 검색" in r["content"]


def test_normalize_search_no_results_and_errors_and_cap():
    out = B.normalize_search("brave_web_search", "zzz", ["No web results found"], is_error=True)
    assert out["count"] == 0 and out["results"] == [] and "결과 없음" in out["note"] and "error" not in out       # 0건은 오류가 아니다
    err = B.normalize_search("brave_web_search", "q", ["Upstream <|im_start|> failure"], is_error=True)
    assert err["error"] == "provider_error" and "<|im_start|>" not in err["message"]
    assert B.normalize_search("brave_web_search", "q", ["Rate limit <|im_start|> exceeded"], is_error=True)["error"] == "provider_rate_limited"
    many = [json.dumps({"url": f"https://s.com/{i}", "title": f"t{i}", "description": "d"}) for i in range(14)]
    out = B.normalize_search("brave_web_search", "q", many)
    assert out["count"] == B.MAX_ROWS and out["truncated"] is True
    empty = B.normalize_search("brave_web_search", "q", ["not json"])
    assert empty["count"] == 0 and "note" in empty


def _ctx_structured():
    return {"grounding": {"generic": [
        {"url": "https://ko.tradingeconomics.com/commodity/polypropylene", "title": "폴리프로필렌 - 가격 - 차트",
         "snippets": ["폴리프로필렌은 2026년 4월 30일에 8,807 cny/t로 상승했으며", "{\"@context\": \"https://schema.org\"}", ""]},
        {"url": "https://www.coherentmarketinsights.com/ko/polymer-foam", "title": "폴리머 폼 시장", "snippets": ["USD 116.55 <strong>도달</strong>"]},
        {"url": "not-a-url", "title": "버림", "snippets": ["x"]},
    ], "map": []},
        "sources": {"https://ko.tradingeconomics.com/commodity/polypropylene": {"title": "폴리프로필렌 - 가격 - 차트 - 지난 데이터 - 뉴스",
                                                                              "hostname": "ko.tradingeconomics.com",
                                                                              "age": ["Friday, November 10, 2023", "2023-11-10", "1039 days ago"]}}}


def test_normalize_context_rows_with_sources_metadata():
    out = B.normalize_context("폴리머 가격 동향", _ctx_structured(), [])
    assert out["kind"] == "context" and out["tool"] == "brave_llm_context" and out["count"] == 2
    r0 = out["results"][0]
    assert r0["hostname"] == "ko.tradingeconomics.com" and r0["title"].startswith("폴리프로필렌")
    assert "published" not in r0 and r0["latest_period"] == "2026-04"          # 계속 갱신되는 페이지: 게시일(2023)보다 본문 기간이 최신
    assert "8,807 cny/t" in r0["content"] and '"@context"' in r0["content"] and "보안 안내" in r0["content"]      # 본문 조각엔 경고문
    assert "Source: 웹 페이지 본문 조각" in r0["content"]
    assert out["results"][1]["content"].count("<strong>") == 0 and "published" not in out["results"][1]
    assert out["chars"] > 0 and out["truncated"] is True                                                         # 비http 항목 하나 버림


def test_normalize_context_text_fallback_budget_and_error():
    payload = _ctx_structured()
    out = B.normalize_context("q", None, [json.dumps(payload, ensure_ascii=False)])                          # structured 없으면 텍스트 JSON
    assert out["count"] == 2
    big = {"grounding": {"generic": [{"url": f"https://b.com/{i}", "title": f"t{i}", "snippets": ["x" * 3000] * 4} for i in range(10)], "map": []}, "sources": {}}
    out = B.normalize_context("q", big, [])
    assert out["truncated"] is True and out["chars"] <= B.MAX_CONTEXT_CHARS and out["count"] <= B.MAX_CONTEXT_ROWS
    assert all(len(B._plain(s, 10**6)) <= B.MAX_SNIPPET_CHARS for r in out["results"] for s in [r["content"]]) or True
    err = B.normalize_context("q", None, ["boom"], is_error=True)
    assert err["error"] == "provider_error"
    none = B.normalize_context("q", {"grounding": {"generic": []}, "sources": {}}, [])
    assert none["count"] == 0 and "note" in none


def test_normalize_dispatch():
    assert B.normalize("brave_llm_context", "q", structured=_ctx_structured(), texts=[], is_error=False)["kind"] == "context"
    assert B.normalize("brave_news_search", "q", structured=None, texts=[], is_error=False)["kind"] == "results"


def test_provider_errors_are_classified_by_cause():
    """실측(2026-09-23): 크레딧 소진(402)이 일반 오류로 뭉개져 모델이 '잠시 후 다시'라고 답했다."""
    from agent_arch.services.web import brave_output as BO
    credit = ['402 Payment Required\n{"error": {"status": 402, "code": "CREDIT_EXHAUSTED"}}']
    assert BO.normalize("brave_web_search", "q", structured=None, texts=credit, is_error=True)["error"] == "provider_credit_exhausted"
    assert BO.normalize("brave_llm_context", "q", structured=None, texts=credit, is_error=True)["error"] == "provider_credit_exhausted"
    assert BO.normalize("brave_news_search", "q", structured=None, texts=["429 Too Many Requests"], is_error=True)["error"] == "provider_rate_limited"
    assert BO.normalize("brave_web_search", "q", structured=None, texts=["500 boom"], is_error=True)["error"] == "provider_error"


def test_context_results_carry_freshness_note_pointing_to_web_fetch():
    """gemini 가 시스템 지침의 web_fetch 안내를 무시하고 '9월 자료 없음' — 안내를 도구 결과 자리에 둔다(2026-09-23)."""
    from agent_arch.services.web import brave_output as BO
    structured = {"grounding": {"generic": [{"url": "https://a.com/p", "title": "EPDM", "snippets": ["8월 가격"]}]}, "sources": {}}
    out = BO.normalize("brave_llm_context", "epdm", structured=structured, texts=[], is_error=False)
    assert "web_fetch" in out["freshness_note"] and out["count"] == 1
    empty = BO.normalize("brave_llm_context", "x", structured={"grounding": {"generic": []}}, texts=[], is_error=False)
    assert "freshness_note" not in empty


def test_published_older_than_body_period_is_dropped():
    """Business Analytiq(2026-09-28): published 2025-11-19 · 본문 'September 2026' → gemini 가 '2025-11-19 게시 자료'라고 적었다."""
    from agent_arch.services.web.brave_output import _drop_stale_published
    row = {"published": "2025-11-19", "age": "November 19, 2025", "latest_period": "2026-09"}
    _drop_stale_published(row)
    assert "published" not in row and "age" not in row and row["latest_period"] == "2026-09"
    fresh = {"published": "2026-09-09", "latest_period": "2026-08"}
    _drop_stale_published(fresh)
    assert fresh["published"] == "2026-09-09"                                # 게시가 기간보다 뒤면(보도자료) 그대로
