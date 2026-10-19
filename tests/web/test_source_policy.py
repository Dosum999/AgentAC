"""출처 정책(Brave Goggles) — 설정 파일 → 강제 인자. 벤치(2026-09-23): 리포트팜이 '시장 동향' 상위를 채워 출처 품질 2.87/5."""
from agent_arch.core.config import settings
from agent_arch.services.web import source_policy as SP


def test_parse_goggle_drops_comments_and_blanks():
    assert SP.parse_goggle("! 설명\n\n$discard,site=a.com\n  $boost=2,site=go.kr  \n") == "$discard,site=a.com\n$boost=2,site=go.kr"
    assert SP.parse_goggle("! only comments\n") is None


def test_default_file_blocks_report_farms_and_boosts_public(monkeypatch):
    monkeypatch.setattr(settings, "web_source_goggle_path", "", raising=False)
    SP.load_goggle.cache_clear()
    g = SP.load_goggle()
    assert "$discard,site=researchnester.com" in g and "$boost=3,site=go.kr" in g and "!" not in g


def test_missing_file_means_no_rerank(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "web_source_goggle_path", str(tmp_path / "nope.txt"), raising=False)
    SP.load_goggle.cache_clear()
    try:
        assert SP.load_goggle() is None
    finally:
        SP.load_goggle.cache_clear()


def test_brave_policies_force_goggles_on_all_tools(monkeypatch, tmp_path):
    from agent_arch.agents.tools.mcp import provider
    f = tmp_path / "g.txt"; f.write_text("$discard,site=x.com\n", encoding="utf-8")
    monkeypatch.setattr(settings, "web_source_goggle_path", str(f), raising=False)
    SP.load_goggle.cache_clear()
    try:
        pol = provider._brave_policies()
        assert all(p.forced_args.get("goggles") == "$discard,site=x.com" for p in pol.values())
        assert all("goggles" in p.hidden_params for p in pol.values())                  # 모델은 못 바꾼다
    finally:
        SP.load_goggle.cache_clear()
