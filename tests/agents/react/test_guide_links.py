"""가이드 참고 링크 각주 — search_user_guide_kb 결과 청크 본문의 URL 만, 코드가 `- URL` 로 나열한다.

모델이 지어낸 참고 링크(`[제목](None)`·제목 영어 슬러그·무의미 반복) 실측(2026-09-16~17)이 배경.
"""
import json

from agent_arch.agents.react import guide_links as G
from agent_arch.agents.react.types import LoopResult

_FORM = "https://example.com/main/support/inquiry/write"
_CONTACT = "https://www.example.com/contact"
_WIKI = "https://example.atlassian.net/wiki/spaces/GUIDE/pages/1/x"


def _chunk(title, content, score=0.7, blocked=False):
    return {"title": title, "content": content, "score": score, "permission_required": "system_user", "permission_blocked": blocked}


def _tr(groups, name=G.GUIDE_TOOL, cid="c1"):
    return {"tool_call_id": cid, "name": name, "content": json.dumps(groups, ensure_ascii=False)}


# stg Qdrant 실측 청크(페이지 371589121 "고객 문의 접수 및 대응") 모양 그대로.
_FAQ_GROUP = {"category": "General", "category_top_score": 0.7865, "chunks": [
    _chunk("고객 문의 접수 및 대응 - 자주 묻는 질문", "## 자주 묻는 질문", 0.7865),
    _chunk("고객 문의 접수 및 대응 - Q. 문의는 어디로 하면 되나요?",
           f"회원은 회원 문의 폼(\n{_FORM}\n) 또는 메신저의 \"회사-문의하기\" 채널을, 비회원은 비회원 문의 폼(\n{_CONTACT}\n)을 이용해 주세요.", 0.77),
    _chunk("고객 문의 접수 및 대응 - Q. 회원이 아니어도 문의할 수 있나요?",
           f"가능합니다. 비회원은 비회원 문의 폼(\n{_CONTACT}\n)에서 문의를 남기실 수 있습니다.", 0.76),   # 앞 문장은 라벨에 안 붙는다
    _chunk("고객 문의 접수 및 대응 - 문의하는 방법",
           f"| 회원 | 회원 문의 폼 ( {_FORM} ) — 플랫폼 우측 상단 물음표(?)의 「문의하기」 메뉴( {_FORM} )로 접속 |\n| 비회원 | 비회원 문의 폼 ( {_CONTACT} ) |", 0.75),
]}
_MSG_GROUP = {"category": "Messenger", "category_top_score": 0.70, "chunks": [
    _chunk("메신저 채널 설정 및 운영 - 회사-문의하기 채널", f"채널 안내는 [가이드]({_WIKI}) 참고", 0.70),
]}


def test_extract_urls_dedup_and_filters():
    hik = {"category": "HR System", "category_top_score": 0.5, "chunks": [
        _chunk("[근무관리] 안면인식기 - 방법 B", "단말기 웹( https://단말기IP )에서 업로드", 0.5),      # 플레이스홀더 호스트
        _chunk("비밀 문서 - 관리자", f"관리자 폼( {_FORM}/admin )", 0.9, blocked=True),               # 권한 차단 청크
        _chunk("DMP 사용 가이드 - 접속", "포털은 [DMP 메인](https://app.example.com/main) 에서.", 0.4),   # 마크다운 링크
        _chunk("그림 - 캡처", "![캡처](https://img.example.com/a.png)", 0.4),                             # 이미지 링크 제외
    ]}
    links = G.extract_guide_links([_tr([_FAQ_GROUP, _MSG_GROUP, hik])])
    assert [l["url"] for l in links] == [_FORM, _CONTACT, "https://app.example.com/main"]     # 위키·플레이스홀더·차단·이미지 제외, 중복 제거
    assert links[0]["group_score"] == 0.7865 and links[0]["title"].startswith("고객 문의 접수 및 대응")


def test_extract_ignores_other_tools_errors_abstain_and_truncated_shapes():
    other = {"tool_call_id": "x", "name": "brave_web_search", "content": json.dumps({"results": [{"url": "https://a.com"}]})}
    err = {"tool_call_id": "e", "name": G.GUIDE_TOOL, "content": "[ERROR] boom"}
    wrapped = {"tool_call_id": "w", "name": G.GUIDE_TOOL, "content": json.dumps({"items": [_FAQ_GROUP], "_truncated": "…"}, ensure_ascii=False)}
    inner = {"category": "General", "category_top_score": 0.9, "chunks": {"items": [_chunk("페이지 - 절", f"폼( {_CONTACT} )")], "_truncated_chunks": "…"}}
    assert G.extract_guide_links([other, err, _tr([])]) == []          # 다른 도구·오류·abstain(빈 결과)
    assert [l["url"] for l in G.extract_guide_links([wrapped])] == [_FORM, _CONTACT]
    assert [l["url"] for l in G.extract_guide_links([_tr([inner])])] == [_CONTACT]


def test_select_referenced_page_first_then_top_group_fallback():
    links = G.extract_guide_links([_tr([_MSG_GROUP, _FAQ_GROUP])])
    assert [l["url"] for l in G.select_guide_links("고객 문의는 두 가지 방법으로 접수하고 대응할 수 있습니다.", links)] == [_FORM, _CONTACT]
    assert [l["url"] for l in G.select_guide_links("전혀 다른 이야기", links)] == [_FORM, _CONTACT]   # 최상위 그룹 폴백
    assert G.select_guide_links("아무 말", []) == []


def test_footer_format_missing_only_cap_and_none_without_urls():
    tr = _tr([_FAQ_GROUP])
    assert G.guide_links_footer("고객 문의는 접수·대응 절차가 있습니다.", [tr]) == f"\n\n🔗 참고 링크\n- {_FORM}\n- {_CONTACT}"
    assert G.guide_links_footer(f"고객 문의 접수 대응은 [회원 문의 폼]({_FORM}) 에서.", [tr]) == f"\n\n🔗 참고 링크\n- {_CONTACT}"
    no_url = {"category": "DMP", "category_top_score": 0.8, "chunks": [_chunk("DMP 표시 언어 가이드 - 개요", "설정 > 언어에서 바꿉니다.")]}
    assert G.guide_links_footer("DMP 표시 언어는 설정 > 언어에서", [_tr([no_url])]) == ""    # 실측: 언어 설정 다섯 페이지 — URL 없음 → 섹션 없음
    many = {"category": "G", "category_top_score": 0.9, "chunks": [_chunk(f"페이지{i} - 절", f"폼{i}( https://s{i}.example.com/x )") for i in range(8)]}
    assert G.guide_links_footer("무관한 답", [_tr([many])]).count("\n- ") == G.MAX_LINKS


def test_apply_footer_in_place_and_survives_sanitize():
    from agent_arch.utils.sanitize import sanitize_tool_names
    r = LoopResult(content="고객 문의는 접수하고 대응합니다.\n\n", tool_results=[_tr([_FAQ_GROUP])])
    footer = G.apply_guide_links_footer(r)
    assert r.content == "고객 문의는 접수하고 대응합니다." + footer
    assert sanitize_tool_names(r.content) == r.content              # 코드 각주는 정화기(모델 섹션 제거)를 통과한다
    assert G.apply_guide_links_footer(LoopResult(content="답", tool_results=[])) == ""
