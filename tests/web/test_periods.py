"""본문의 가장 최근 연·월 — 최신 출처 고르기·검증기 대조(실측 2026-09-23: 9월 표를 두고 8월 보도자료를 따름)."""
from agent_arch.services.web.periods import latest_period


def test_latest_period_formats_and_future_excluded():
    t = "EPDM Rubber Prices September 2026 | Last Updated: Sep 18, 2026. Forecast to December 2035. 2026년 8월 대비, 2026-07-15 기준"
    assert latest_period(t, today=(2026, 9)) == "2026-09"                 # 2035 전망은 오늘 이후라 제외
    assert latest_period("2026년 8월 가격, August 2026", today=(2026, 9)) == "2026-08"
    assert latest_period("no dates here", today=(2026, 9)) is None and latest_period(None) is None
    assert latest_period("May 2026 and 2026/13", today=(2026, 9)) == "2026-05"   # 13월은 무시


def test_annotate_timezones_converts_dated_foreign_times_to_kst():
    """실측(2026-09-28): Bloomberg '20:02 EDT 09/27' = KST 9/28 09:02 인데 모델이 '9/27 자료'로 오판."""
    from agent_arch.services.web.periods import annotate_timezones as A
    assert A("As of7:33 PM EDT 09/27/26.") == "As of7:33 PM EDT 09/27/26 [= 2026-09-28 08:33 KST]."
    assert A("Sep 27, 2026, 16:31 UTC") .endswith("[= 2026-09-28 01:31 KST]")
    assert A("Last Updated: 2026-09-27 20:02 EDT").endswith("[= 2026-09-28 09:02 KST]")
    assert A("마감 16:31 UTC") == "마감 16:31 UTC"                                 # 날짜 없으면 손대지 않는다
    assert A("2026-09-28 08:44 KST") == "2026-09-28 08:44 KST" and A(None) is None


def test_iso_timestamps_are_annotated_except_kst():
    """TradingView priceValidUntil(금요일 종가 시각)을 모델이 '오늘 09:26 기준'으로 답한 재현(2026-09-28)."""
    from agent_arch.services.web.periods import annotate_timezones as A
    assert A("priceValidUntil: 2026-09-25T20:59:45Z").endswith("[= 2026-09-26 05:59 KST]")
    assert A("t 2026-09-28T08:30:00+09:00") == "t 2026-09-28T08:30:00+09:00"
    assert A("t 2026-09-27T20:02:00-04:00").endswith("[= 2026-09-28 09:02 KST]")


def test_salient_excerpt_keeps_head_and_late_numeric_lines():
    from agent_arch.services.web.periods import salient_excerpt
    t = "EPDM price index. " + "Methodology and background text. " * 30 + "\nSeptember 2026 North America 2.99 USD/kg, up 4.9%.\nNo numbers here."
    out = salient_excerpt(t, 300, terms="EPDM 9월", today=(2026, 9))
    assert out.startswith("EPDM price index.") and "2.99 USD/kg" in out and "No numbers" not in out and len(out) <= 300
    assert salient_excerpt("짧은 본문", 300) == "짧은 본문"


def test_wib_is_annotated_to_kst():
    from agent_arch.services.web.periods import annotate_timezones
    assert "[= 2026-09-27 09:00 KST]" in annotate_timezones("Updated 27 Sep 2026 07:00 WIB")


def test_stale_page_meta_dates_are_dropped_when_body_is_newer():
    """Business Analytiq(2026-09-28): 본문 'September 2026' 인데 dateModified 2025-11-19 를 앞세워 모델·검증기가 오판."""
    from agent_arch.services.web.html_extract import extract
    ld = '<script type="application/ld+json">{"dateModified":"2025-11-19T15:12:39+00:00"}</script>'
    body = ("<article><h1>EPDM rubber price index</h1><h3>EPDM rubber price September 2026 and outlook</h3>"
            "<p>North America:US$2.99/KG, 4.9% up. More text for the extractor to keep this block as main content.</p></article>")
    assert "dateModified" not in extract(f"<html><head>{ld}</head><body>{body}</body></html>")[1]
    old = body.replace("September 2026", "the latest month")
    assert "dateModified" in extract(f"<html><head>{ld}</head><body>{old}</body></html>")[1]   # 본문에 더 최신 표기가 없으면 남긴다
