"""출처 각주 — 웹 도구(브레이브 MCP) 결과에서 URL 을 뽑아 답변 끝에 결정적으로 붙인다. 답이 참조한 출처부터, 참조 안 된 결과는 붙이지 않는다."""
import json

from agent_arch.agents.react import provenance as P
from agent_arch.agents.react.types import LoopResult


def _env(t):
    return f'<<<EXTERNAL_UNTRUSTED_CONTENT id="abcd">>>\nSource: 웹 검색\n---\n{t}\n<<<END_EXTERNAL_UNTRUSTED_CONTENT id="abcd">>>'


def _search(cid, *rows, tool="brave_web_search"):
    return {"tool_call_id": cid, "name": tool, "content": json.dumps({"kind": "results", "tool": tool, "results": [
        {"title": t, "url": u, "content": _env("요약"), **({"published": d} if d else {})} for t, u, d in rows]}, ensure_ascii=False)}


def _context(cid, *rows):
    return {"tool_call_id": cid, "name": "brave_llm_context", "content": json.dumps({"kind": "context", "tool": "brave_llm_context", "results": [
        {"title": t, "url": u, "hostname": "h", "content": _env("본문 조각"), **({"published": d} if d else {})} for t, u, d in rows]}, ensure_ascii=False)}


def test_extract_from_search_and_context_dedup_and_flags():
    err = {"tool_call_id": "c3", "name": "brave_web_search", "content": "[ERROR] boom"}
    structured_err = {"tool_call_id": "c5", "name": "brave_news_search", "content": json.dumps({"error": "quota_exceeded", "message": "x"})}
    internal = {"tool_call_id": "c4", "name": "mattermost_search_knowledge", "content": json.dumps({"results": [{"url": "https://talk/x"}]})}
    srcs = P.extract_sources([_search("c1", ("A 기사", "https://a.com/1", "2026-09-01"), ("A 기사 2", "https://a.com/1", None), ("bad", "ftp://x", None)),
                              _context("c2", ("B 기사", "https://b.com/x", None), ("A 다시", "https://a.com/1", None)), err, structured_err, internal])
    assert [s["url"] for s in srcs] == ["https://a.com/1", "https://b.com/x"]           # 중복 URL·비http·오류·사내 도구 제외
    assert srcs[0]["title"] == "A 기사" and srcs[0]["site"] == "a.com" and srcs[0]["published"] == "2026-09-01"
    assert srcs[0]["batch"] == 1 and "요약" in srcs[0]["body"] and "본문 조각" in srcs[0]["body"]   # 같은 URL: 본문 합침, batch 는 마지막 호출
    assert srcs[1]["batch"] == 1 and srcs[1]["tool"] == "brave_llm_context" and srcs[1]["body"] == "본문 조각"


def test_footer_lists_only_missing_urls_and_caps():
    rows = [(f"제목{i}", f"https://s.com/{i}", None) for i in range(8)]
    tr = [_search("c1", *rows)]
    footer = P.sources_footer("아무 출처도 언급하지 않은 답", tr)                       # 참조 없음 → 마지막 호출 결과, 상한 5
    assert footer.startswith("\n\n출처(웹):") and footer.count("\n- ") == P.MAX_SOURCES
    assert P.sources_footer("본문에 https://s.com/0 은 이미 적었다", tr) == ""            # 하나를 인라인으로 적었으면 그게 근거 — 나머지는 붙이지 않는다
    distinct = [_search("c2", ("서울 지하철 요금 인상 확정", "https://n.kr/subway", None), ("부산 항만 물동량 증가", "https://n.kr/port", None))]
    assert P.sources_footer("서울 지하철 요금 인상 확정 기사에 따르면", distinct) == "\n\n출처(웹):\n- 서울 지하철 요금 인상 확정 — https://n.kr/subway"   # 제목으로 참조한 것만
    assert P.sources_footer("아무 도구 없음", []) == "" and P.sources_footer("x", [{"name": "calculator", "content": "1"}]) == ""


def test_apply_on_loop_result_mutates_content_once_with_date():
    r = LoopResult(content="답변", tool_results=[_search("c1", ("T", "https://t.com/1", "2026-09-12"))])
    footer = P.apply_sources_footer(r)
    assert footer and r.content.endswith("- T (2026-09-12) — https://t.com/1")
    assert P.apply_sources_footer(r) == "" and r.content.count("https://t.com/1") == 1    # 두 번 붙지 않는다


def test_truncated_results_shape_and_bad_json():
    tr = [{"name": "brave_web_search", "content": json.dumps({"kind": "results", "results": {"items": [{"title": "x", "url": "https://x.com/"}], "_truncated": "30건 중 20건"}})},
          {"name": "brave_web_search", "content": "{not json"}]
    assert [s["url"] for s in P.extract_sources(tr)] == ["https://x.com/"]


# ─── 각주는 답이 참조한 출처부터 ────────────────────────────────────────────────

def _polymer_turn():
    ko = _search("s1", ("폴리메시 30일 16.5% 반등…장기 약세는 여전 - TokenPost", "https://www.tokenpost.kr/news/405765", None),
                 ("현대차, 자동차 부품도 '3D 프린팅'으로 직접 제조 | 더구루", "https://view.mk.co.kr/car-tech/271050/", None),
                 ("폴리하이드록시알카노에이트 시장 규모 및 성장 분석 (2036년)", "https://www.researchnester.com/kr/pha/4627", None),
                 ("2022년 국내 최초로 화학적 재활용 상용화-국민일보", "https://www.kmib.co.kr/article/view.asp?arcid=9000011045", None),
                 ("폐섬유 폴리에스터사, 상업생산 돌입 < 지구촌 Global + < 뉴스 - 국제섬유신문", "https://www.itnk.co.kr/news/articleView.html?idxno=81845", None),
                 ("여섯째 무관", "https://junk.com/6", None))
    en = _search("s2", ("Rearranged polymer chains create degradable materials with potential for stronger, tougher packaging", "https://phys.org/news/2026-09-rearranged-polymer-chains.html", "2026-09-08"),
                 ("Biobased Oligo- and Polyesters from Fatty Acids with Demonstrated Biodegradability or Recyclability - PubMed", "https://pubmed.ncbi.nlm.nih.gov/42658061/", None),
                 ("Chemical recycling of post-consumer plastics - LCC CNRS Toulouse", "https://www.lcc-toulouse.fr/en/chemical-recycling/", None),
                 ("Unrelated english result about polymer clay crafts", "https://crafts.example/clay", None))
    ctx = _context("f1", ("Biobased...", "https://pubmed.ncbi.nlm.nih.gov/42658061/", None))
    return [ko, en, ctx]


def test_footer_prefers_sources_the_answer_referenced_and_drops_unreferenced():
    content = ("이번 주 폴리머 동향입니다.\n"
               "1) 버지니아 공대 연구진의 생분해성 폴리머 개발(2026-09-08)\n"
               "2) 지방산 기반 바이오 폴리머의 생분해성 및 재활용성(2026-09-14)\n"
               "3) 프랑스 CNRS 연구진의 폐플라스틱 화학적 재활용 촉매(2026-09-09)\n\n"
               "🔗 참고 링크\n"
               "Rearranged polymer chains create degradable materials with potential for stronger, tougher packaging\n"
               "Biobased Oligo- and Polyesters from Fatty Acids with Demonstrated Biodegradability or Recyclability for Industrial Applications - PubMed\n"
               "Chemical recycling of post-consumer plastics - LCC CNRS Toulouse")
    footer = P.sources_footer(content, _polymer_turn())
    urls = [l.split(" — ")[-1] for l in footer.split("\n") if l.startswith("- ")]
    assert urls == ["https://phys.org/news/2026-09-rearranged-polymer-chains.html",
                    "https://pubmed.ncbi.nlm.nih.gov/42658061/",
                    "https://www.lcc-toulouse.fr/en/chemical-recycling/"]                 # 참조한 3건만, 제목 순서
    assert "(2026-09-08)" in footer                                                       # 게시일이 있으면 함께
    assert "tokenpost" not in footer and "junk.com" not in footer and "crafts.example" not in footer   # 버린 결과는 근거가 아니다


def test_footer_falls_back_to_last_call_only_when_nothing_referenced():
    tr = _polymer_turn()
    footer = P.sources_footer("요약만 적은 답", tr)                                       # 제목·URL·숫자 어느 것도 없다 → 마지막 호출(context 1건)
    assert footer.count("\n- ") == 1 and "pubmed.ncbi.nlm.nih.gov/42658061" in footer
    footer2 = P.sources_footer("요약만 적은 답", tr[:2])                                  # 마지막 호출 = 영어 검색 → 첫 한국어 검색의 무관 결과 제외
    urls = [l.split(" — ")[-1] for l in footer2.split("\n") if l.startswith("- ")]
    assert len(urls) == 4 and "tokenpost" not in footer2


def _context_rows(cid, *rows):
    return {"tool_call_id": cid, "name": "brave_llm_context", "content": json.dumps({"kind": "context", "results": [
        {"title": t, "url": u, "content": _env(body)} for t, u, body in rows]}, ensure_ascii=False)}


def test_context_rows_are_not_read_pages_inline_citations_leave_no_footer():
    """재실측 재현: 첫 질의는 무관(코인·학원), 둘째 질의가 근거. 모델이 근거 5건을 인라인 링크로 적었으면 각주는 없어야 한다."""
    junk = _context_rows("c1", ("폴리매쉬(POLYX) 56원선 강세", "https://job-post.co.kr/news/1", "폴리매쉬 56원 5.03% 상승 51.4원"),
                         ("폴리어학원 교습비", "https://brandcontent.co.kr/poly-english", "1,089,000원 131만 9천 원"))
    used = _context_rows("c2", ("PVC Resin Prices Q3 2026", "https://www.sam-uk.com/2026-pvc", "China spot PVC 4,890 CNY, up 9.69% month on month, 2.28% above last year"),
                         ("PET Prices August 2026", "https://www.openpr.com/news/463", "PET price Africa USD 1.09/Kg rising 5.8%"),
                         ("Unused resin page", "https://unused.example/x", "HDPE spot 1,222 USD"))
    content = ("PVC 현물 4,890 CNY 로 9.69% 상승 ([SAM-UK](https://www.sam-uk.com/2026-pvc))\n"
               "PET 아프리카 1.09달러/Kg, 5.8%↑ ([OpenPR](https://www.openpr.com/news/463))")
    assert P.sources_footer(content, [junk, used]) == ""                                # 참조한 2건은 본문에 URL 이 있고, 나머지는 붙이지 않는다
    # 인라인 링크 없이 숫자만 적은 답 → 숫자 일치로 근거 2건만 각주, 코인·학원·미사용 페이지는 제외
    plain = "PVC 현물 4,890 CNY 로 9.69% 상승. PET 아프리카 1.09달러/Kg, 5.8% 상승."
    footer = P.sources_footer(plain, [junk, used])
    urls = [l.split(" — ")[-1] for l in footer.split("\n") if l.startswith("- ")]
    assert urls == ["https://www.sam-uk.com/2026-pvc", "https://www.openpr.com/news/463"]
    assert P.answer_numbers("2026년 3분기 4,890 CNY 9.69% 644달러 20년래 5% 2025") == ["4,890", "9.69%", "644", "5%"]   # 연도 제외·중복 제거
    assert P.number_score({"body": "pvc 4,890 cny 9.69%"}, ["4,890", "9.69%", "644"]) == 4 and P.number_score({"body": ""}, ["1"]) == 0


def test_is_referenced_by_title_core_tokens_or_host():
    body = P._norm("연구 결과는 phys.org 기사와 '폐섬유 폴리에스터사 상업생산 돌입' 보도를 참고했습니다. 폴리머 시장은 성장 중.")
    assert P.is_referenced({"title": "아무 제목 - 매체", "site": "phys.org"}, body)                                   # 호스트명 언급
    assert P.is_referenced({"title": "폐섬유 폴리에스터사, 상업생산 돌입 < 지구촌 Global + < 뉴스 - 국제섬유신문", "site": "x.kr"}, body)  # 제목 핵심부 토큰
    assert not P.is_referenced({"title": "폴리머 시장 규모 및 성장 분석 (2036년)", "site": "researchnester.com"}, body)    # '폴리머' 한 토큰만 겹침
    assert not P.is_referenced({"title": "kr", "site": "a.kr"}, body) and not P.is_referenced({"title": "t", "site": "s"}, "")
    assert P.is_referenced({"title": "t", "site": "s", "url": "https://q.com/a"}, P._norm("근거: https://q.com/a 참고"))            # URL 포함
    assert P._title_core("짧은 제목 | 아주아주 긴 매체 이름이 뒤에 붙는 경우") == "아주아주 긴 매체 이름이 뒤에 붙는 경우"


def test_inline_citation_suppresses_footer_even_with_same_host_results():
    """벤치 재현(2026-09-23): 본문 URL 속 호스트명이 같은 매체의 다른 결과를 '참조'로 끌어와 무관·옛 기사를 각주에 붙였다."""
    tr = [_search("c1", ("원/달러 환율 22.8원 내린 1,358.2원", "https://yna.co.kr/view/A1", "2026-09-22"),
                  ("2023년 환율 결산", "https://yna.co.kr/view/OLD", "2023-12-29"),
                  ("다른 매체 기사", "https://news1.kr/x/574680", None))]
    body = "전일 종가 **1,358.2원** ([연합뉴스](https://yna.co.kr/view/A1))"
    assert P.sources_footer(body, tr) == ""                                              # 인용은 모델이 했다 — 같은 호스트의 옛 기사를 끌어오지 않는다


def test_host_and_numbers_inside_urls_do_not_count_as_reference():
    srcs = [{"title": "무관한 제목", "url": "https://yna.co.kr/view/OLD", "site": "yna.co.kr", "body": "574680 1,358"},
            {"title": "다른 제목", "url": "https://b.com/1", "site": "b.com", "body": ""}]
    body = "근거 https://yna.co.kr/view/ZZZ?idxno=574680 참고, 결과 요약만 적음"               # 결과에 없는 URL — 인라인 인용 아님
    assert P.select_footer_sources(body, srcs) == []                                     # 호스트·숫자는 URL 밖 본문에서만 본다(폴백 batch 없음)
