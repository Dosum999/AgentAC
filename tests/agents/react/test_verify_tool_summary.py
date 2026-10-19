"""검증기에 넘기는 도구 결과 요약 (`verify._summarize_result`).

왜 있는가 — 2026-09-17 실사용: "오늘 나를 언급한 링크 찾아주고 내용도 보여줘" 에 에이전트가
`find_my_mentions` 로 **정답 3건을 이미 찾아 놓고도** 검증기가 두 라운드 연속 revise 를 냈다.

    round=1 issues=['초안의 구체적인 내용과 URL들이 도구 결과(증거)에 포함되어 있지 않음']
    round=2 issues=['…', '존재하지 않는 링크와 내용을 지어내어 답변함']

답변은 **맞았다.** 검증기가 증거를 못 본 것이다 — 사내 도구 결과가 `content[:400]` 로
잘려 첫 항목의 본문도 다 안 들어갔기 때문이다. 그 사이 불필요한 `read_thread` 3회가
더 돌았고, 상한에 걸려서야 답이 나갔다.

⚠️ **웹 도구는 2026-09-15 에 같은 함정을 이미 고쳤다**(행 단위로 펼침). 그때 사내 도구는
빠져 있었다. 검증기가 늘 "지어냈다"를 외치면 **진짜 환각을 놓친다** — 그게 이 테스트의 값이다.
"""
from __future__ import annotations

import json

import pytest

from agent_arch.agents.react.verify import _summarize_result

_LINK = "https://chat.example.com/team/pl/aaaaaaaaaaaaaaaaaaaaaaaaaa"
# 실사용 답변에 실제로 실린 문장
_BODY = "에이전트의 NemoClaw 사용 전략에 대한 내용으로, 에이전트 + 선택적 NemoClaw/OpenShell 전략을 정리했습니다."


def _mentions(n: int = 3) -> str:
    return json.dumps(
        {"count": n, "results": [
            {"channel_name": f"채널{i}", "created_at": f"2026-09-17T1{i}:32:00+09:00",
             "message": _BODY, "permalink": _LINK, "author": "이영희"}
            for i in range(n)
        ]}, ensure_ascii=False,
    )


@pytest.mark.parametrize(
    "tool",
    [
        "mattermost_skill_find_my_mentions",
        "mattermost_skill_find_posts_by_author",
        "mattermost_search_knowledge",
        "drive_search_files",
        "search_colleagues",
    ],
)
def test_internal_tools_are_expanded_row_by_row(tool):
    out = _summarize_result(tool, _mentions())
    assert "결과 3건" in out
    assert out.count("·") >= 3, "행마다 한 줄씩 나와야 한다"


def test_body_and_link_reach_the_verifier():
    """이게 안 되면 검증기가 '증거에 없음 → 지어냄' 으로 오판한다."""
    out = _summarize_result("mattermost_skill_find_my_mentions", _mentions())
    assert _LINK in out, "링크가 증거에 있어야 한다"
    assert "NemoClaw" in out, "본문이 증거에 있어야 한다"


def test_old_truncation_would_have_hidden_the_evidence():
    """회귀 감시 — 앞 400자 방식이었다면 마지막 행은 보이지도 않았다."""
    raw = _mentions()
    assert _LINK not in raw[400:][:1] or raw[:400].count(_LINK) < 3
    out = _summarize_result("mattermost_skill_find_my_mentions", raw)
    assert out.count(_LINK) == 3, "모든 행의 링크가 보여야 한다"


def test_empty_result_keeps_the_reason():
    """0건일 때 **왜** 0건인지가 남아야 한다 — 검증기가 '검색 실패' 와 '정말 없음' 을 가른다."""
    out = _summarize_result(
        "mattermost_skill_find_posts_by_author",
        json.dumps({"count": 0, "results": [], "retrieval_status": "unverified"}, ensure_ascii=False),
    )
    assert "0건" in out and "unverified" in out


def test_rows_are_capped():
    out = _summarize_result("mattermost_skill_find_my_mentions", _mentions(30))
    assert out.count("·") <= 8, "검증기 입력이 무한정 커지면 안 된다"


def test_result_without_rows_falls_back_to_head():
    """행이 없는 결과(단건 dict·문자열)는 앞부분을 보여 준다 — 상한은 있다."""
    raw = json.dumps({"whatever": "x" * 3000}, ensure_ascii=False)
    out = _summarize_result("some_other_tool", raw)
    assert out.startswith('{"whatever"') and len(out) <= 1500


def test_error_payload_is_passed_through():
    assert _summarize_result("mattermost_search_knowledge", "[ERROR] 권한 없음").startswith("[ERROR]")


def test_fetched_page_is_summarized_with_its_numbers():
    """smbs 공식 매매기준율(2026-09-28): web_fetch 결과가 '결과 0건'으로 요약돼 검증기가 정답을 두 번 되돌렸다."""
    page = "서울외국환중개 환율조회 " + "메뉴 " * 400 + "\n?USD=1,352.00&updown1=1&&diff1=-8&EUR=1,538.04&updown3=1&&diff3=-18.89"
    out = _summarize_result("web_fetch", json.dumps({"final_url": "http://www.smbs.biz/ExRate/TodayExRate.jsp", "title": "환율조회",
                                                      "text": page, "fetched_at": "2026-09-28 10:35 KST"}, ensure_ascii=False),
                            "오늘 유로 기준환율")
    assert "결과 0건" not in out and "EUR=1,538.04" in out and "smbs.biz" in out


def test_verify_lane_follows_turn_provider(monkeypatch):
    """vllm 이 내려가자(2026-09-28) gemini 턴의 판정·검증도 vllm 타임아웃으로 pass 가 됐다 — 답하는 제공자를 따른다."""
    from agent_arch.agents.react import verify as V
    from agent_arch.core.config import settings
    from agent_arch.llm.factory import LLMFactory
    from agent_arch.llm.gemini import GeminiLLM
    from agent_arch.llm.vllm import VllmLLM
    assert V.provider_of(object.__new__(GeminiLLM)) == "gemini" and V.provider_of(object.__new__(VllmLLM)) == "vllm"
    assert V.provider_of(object()) is None
    made = []
    monkeypatch.setattr(LLMFactory, "create", classmethod(lambda cls, p=None, **kw: made.append(p) or object()))
    monkeypatch.setattr(settings, "default_llm_provider", "vllm", raising=False)
    V.create_verify_llm("gemini"); V.create_verify_llm(None)
    assert made == ["gemini", "vllm"]


# ─── 실제 결과 모양 (2026-10-01) ──────────────────────────────────────────────
# 위 테스트는 멘션 모양({count, results}) 데이터에 도구 이름만 바꿔 넣어서, `results` 키가 아닌 도구는 실제로는
# "결과 0건 {'count': N}" 으로 요약되는 걸 못 잡았다. stg 실측: 경비청구 참석자 3명을 search_colleagues 로 찾았는데
# 검증기가 "검색 0건인데 확인 완료라고 주장(환각)" 으로 revise → 답이 "소속 확인 안 됨" 으로 고쳐져 나갔다.
# 아래 모양은 stg 배포본(directory.py · nextcloud-mcp · mailcow-mcp · mattermost-mcp) 소스에서 옮겼다.

_POST = {"id": "p1", "root_id": "p1", "author": "박수미", "message": "입고검사 일정 공유드립니다",
         "links": [], "create_at": 1790800000000, "reply_count": 0}

_REAL_SHAPES = {
    "search_colleagues": (
        {"colleagues": [{"employee": "e1", "name": "박수미", "email": "smpark@example.com", "designation": "선임",
                         "department": "플랫폼개발팀", "company": "(주)회사"}], "count": 1},
        ["박수미", "플랫폼개발팀", "(주)회사"]),
    "drive_search_files": (
        {"query": "견적", "count": 1, "files": [{"path": "/Shared/견적서.pdf", "name": "견적서.pdf",
                                                "modified": "2026-09-30T10:00:00"}]},
        ["견적서.pdf", "/Shared/견적서.pdf"]),
    "drive_list_files": (
        {"path": "/Shared", "count": 1, "entries": [{"path": "/Shared/회의록", "name": "회의록"}]},
        ["회의록"]),
    "mail_list_messages": (
        {"folder": "INBOX", "count": 1, "total_matched": 1, "has_more": False,
         "messages": [{"uid": 7, "from": "jpark@example.com", "subject": "주간 보고", "date": "2026-09-30"}]},
        ["주간 보고", "jpark@example.com"]),
    "mattermost_read_thread": (
        {"root_id": "p1", "channel_id": "c1", "count": 1, "posts": [_POST]},
        ["박수미", "입고검사 일정"]),
    "mattermost_skill_get_direct_messages": (
        {"dm_with": "박수미", "channel_id": "c1", "posts": [_POST]},
        ["입고검사 일정"]),
    # 리스트를 반환하는 스킬은 MCP 가 {"result": [...]} 로 감싼다 — 채널 블록 안의 posts 까지 펼쳐야 본문이 보인다
    "mattermost_skill_get_channel_messages": (
        {"result": [{"channel_id": "c1", "team_id": "t1", "posts": [_POST]}]},
        ["입고검사 일정"]),
    "mattermost_skill_find_channels": (
        {"result": [{"channel_name": "scrum", "channel_display_name": "스크럼", "purpose": "회사 스크럼", "score": 1.0}]},
        ["scrum", "회사 스크럼"]),
    "mattermost_skill_get_unread_messages": (
        {"total_mention_channels": 1, "total_mentions": 1, "shown_channels": 1,
         "channels": [{"channel_name": "scrum", "unread_count": 1, "mention_count": 1, "posts": [_POST]}]},
        ["scrum", "입고검사 일정"]),
    "mattermost_skill_get_my_channels": (
        {"requester": {"display_name": "김보라"},
         "teams": [{"team_id": "t1", "team_name": "회사", "channels": [{"display_name": "scrum", "purpose": "스크럼"}]}]},
        ["회사", "scrum"]),
}


@pytest.mark.parametrize("tool", sorted(_REAL_SHAPES))
def test_real_result_shapes_are_not_reported_as_empty(tool):
    data, must_show = _REAL_SHAPES[tool]
    out = _summarize_result(tool, json.dumps(data, ensure_ascii=False))
    assert "0건" not in out, f"{tool}: 행이 있는데 0건으로 보이면 검증기가 정상 답을 환각으로 되돌린다\n{out}"
    for s in must_show:
        assert s in out, f"{tool}: 검증기가 대조할 값 {s!r} 이 요약에 없다\n{out}"


def test_colleague_rows_carry_department_and_company():
    """답의 "○○팀 소속" 을 대조할 근거 — 사람 행엔 본문이 없어 상세 값이 본문 자리에 와야 한다."""
    data, _ = _REAL_SHAPES["search_colleagues"]
    out = _summarize_result("search_colleagues", json.dumps(data, ensure_ascii=False))
    assert "결과 1건" in out and "smpark@example.com" in out


def test_unknown_row_key_is_still_expanded():
    """새 도구가 모르는 키로 행을 실어도 펼친다 — 도구 이름·키 목록에 기대지 않는다."""
    raw = json.dumps({"count": 2, "widgets": [{"name": "A"}, {"name": "B"}]}, ensure_ascii=False)
    out = _summarize_result("brand_new_tool", raw)
    assert "결과 2건" in out and "· A" in out and "· B" in out


def test_truly_empty_internal_result_still_says_zero():
    out = _summarize_result("search_colleagues", json.dumps({"colleagues": [], "count": 0}, ensure_ascii=False))
    assert "결과 0건" in out


def test_nested_rows_are_capped():
    data = {"channels": [{"channel_name": f"ch{i}", "posts": [_POST] * 5} for i in range(5)]}
    out = _summarize_result("mattermost_skill_get_unread_messages", json.dumps(data, ensure_ascii=False))
    assert out.count("·") <= 8, "펼친 뒤에도 검증기 입력 상한은 그대로다"


# ─── 가이드 검색 (2026-10-07 운영) ─────────────────────────────────────────────
# "DMP에서 회사 공용 캘린더를 만들려면?" — search_user_guide_kb 는 웹 도구도 접두어 목록에도 없어 `content[:400]` 로
# 잘렸다. 결과(카테고리 그룹 → chunks, 실측 6,576자)의 앞 400자엔 General 의 첫 청크 하나뿐이라, 검증기가
# "캘린더 생성·공유 방법이 도구 결과에 없음" 으로 state_limits 를 냈다(15:07 메신저·15:16 웹 같은 판정).
# 아래는 그 결과 모양(로컬 user_guide_kb 실측)을 줄인 것이다.

def _chunk(title: str, content: str, page_id: str = "220889097") -> dict:
    return {"page_id": page_id, "title": title, "content": content, "keywords": ["전체 직원"], "version": 6,
            "score": 0.75, "permission_required": "system_user", "permission_blocked": False}


_FAQ = ("**Q: 팀 전체 일정을 공유하려면 어떻게 하나요?** **A:** 팀 공용 달력을 생성한 후 달력 수정 화면에서 "
        "공유 링크를 구성원에게 전달하면, 전원이 동일한 달력에서 일정을 확인하고 등록할 수 있습니다.")
_GUIDE = [
    {"category": "General", "category_top_score": 0.8092, "chunks": [
        _chunk("DMP 사용 가이드 - 캘린더", "## 캘린더\n월간 캘린더 형태로 일정을 확인할 수 있습니다. " * 3, "220790789"),
        _chunk("DMP 사용 가이드 - 회사 설정 페이지로 이동하는 방법", "1. 상단 헤더 맨 오른쪽의 프로필 아이콘을 클릭합니다.", "220790789")]},
    {"category": "Calendar", "category_top_score": 0.7725, "chunks": [
        _chunk("캘린더 사용 가이드 - Calendar 앱 진입 방법", "1. 브라우저에서 DMP 메인 페이지에 접속합니다. 2. 왼쪽 패널에서 [그룹웨어]를 클릭합니다."),
        _chunk("캘린더 사용 가이드 - 자주 묻는 질문", _FAQ)]},
    {"category": "HR System", "category_top_score": 0.7013, "chunks": [
        _chunk("[직원관리] 직원 등록은 DMP에서 합니다", "신규 직원 등록은 DMP에서 진행합니다.", "352124939")]},
]


def test_guide_groups_reach_the_verifier():
    raw = json.dumps(_GUIDE, ensure_ascii=False)
    assert "공유 링크" not in raw[:400], "전제: 예전 400자 절단이면 FAQ 청크가 안 보였다"
    out = _summarize_result("search_user_guide_kb", raw)
    assert "공유 링크를 구성원에게 전달" in out, out
    assert "[Calendar]" in out and "캘린더 사용 가이드 - 자주 묻는 질문" in out, "어느 카테고리의 어느 문서인지 남아야 한다"
    assert "결과 5건" in out


def test_guide_abstain_is_zero():
    """abstain 이면 output=[] — 검증기가 '가이드에 없음' 을 알아야 한다."""
    assert "결과 0건" in _summarize_result("search_user_guide_kb", "[]")


def test_dict_field_inside_a_row_does_not_split_the_row():
    raw = json.dumps({"count": 1, "results": [
        {"title": "출고 처리", "content": "출고 화면에서 처리합니다", "author": {"name": "박수미"}}]}, ensure_ascii=False)
    out = _summarize_result("scms_some_tool", raw)
    assert "결과 1건" in out and "출고 화면에서 처리합니다" in out


def test_top_level_values_are_kept_next_to_rows():
    """행 밖의 단일 값(대상 직원·연도)도 근거다 — 행만 펼치면 사라진다."""
    raw = json.dumps({"employee_name": "김보라", "year": 2026,
                      "balances": [{"leave_type": "연차", "remaining": 7}]}, ensure_ascii=False)
    out = _summarize_result("hr_leave_balance", raw)
    assert "김보라" in out and "2026" in out and "연차" in out and "7" in out
