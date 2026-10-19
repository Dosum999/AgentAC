"""action 도구 view → 워크스페이스 artifact 합성/추출 단위 테스트.

dmp-frontend 데모 패널이 이미 소비하는 app-embed 스키마로 합성되는지 고정한다.
"""
from __future__ import annotations

import json

import pytest

from agent_arch.agents.tools.action_view import build_action_artifact, extract_action_views


def _draft_view():
    return {
        "app": "calendar",
        "kind": "draft",
        "url": "https://drive.example.com/apps/calendar/dayGridMonth/now/new/popover/1/1781758800/1781762400",
        "label": "새 일정 미리보기 — 팀 회의",
        "resource_id": "",
        "preview": {"title": "일정 등록 (승인 대기)", "summary": "등록할까요?",
                    "fields": [{"label": "제목", "value": "팀 회의"}]},
        "form": {
            "kind": "calendar_event",
            "endpoint": "/api/v1/workspace/calendar/event",
            "values": {"summary": "팀 회의", "dtstart": "2026-06-18T14:00:00",
                       "dtend": "2026-06-18T15:00:00", "attendees": []},
        },
    }


def test_build_artifact_app_embed_schema():
    art = build_action_artifact(_draft_view(), "conv-1", "msg-1")
    assert art is not None
    for key in ("id", "tenantId", "conversationId", "messageId", "sourceService",
                "type", "schemaVersion", "freshness", "title", "createdAt"):
        assert key in art, f"필수 필드 누락: {key}"
    assert art["type"] == "app-embed"
    assert art["freshness"] == "live"
    assert art["sourceService"] == "schedule"          # calendar → schedule
    assert art["resourceUrl"].startswith("https://drive.example.com/apps/calendar/")
    assert art["actionKind"] == "draft"
    assert art["payload"]["display"]["title"] == "일정 등록 (승인 대기)"
    assert art["id"].startswith("artifact:default:schedule:app-embed:live:")
    # 편집 폼 계약 전파 (프론트가 이걸로 편집 폼 + 승인 POST)
    assert art["form"]["endpoint"] == "/api/v1/workspace/calendar/event"
    assert art["form"]["values"]["summary"] == "팀 회의"


def test_build_artifact_drops_non_https_keeps_card():
    v = {"app": "calendar", "kind": "draft", "url": "http://drive.example.com/x",
         "preview": {"title": "x"}}
    art = build_action_artifact(v, "c", "m")
    assert art is not None
    assert "resourceUrl" not in art          # http → 제외
    assert art["payload"]["display"]["title"] == "x"


def test_build_artifact_none_when_nothing_to_show():
    assert build_action_artifact(None, "c", "m") is None
    assert build_action_artifact({}, "c", "m") is None
    assert build_action_artifact({"url": "http://x/y"}, "c", "m") is None   # https 아니고 preview 없음


def test_extract_action_views_only_action_results():
    tool_results = [
        {"name": "calendar_create_event",
         "content": json.dumps({"draft": True, "view": _draft_view()}, ensure_ascii=False)},
        {"name": "mail_list_messages", "content": json.dumps({"messages": []}, ensure_ascii=False)},
        {"name": "broken", "content": "[ERROR] 실패"},
    ]
    views = extract_action_views(tool_results)
    assert len(views) == 1
    assert views[0]["app"] == "calendar"


def test_extract_action_views_accepts_card_only_view():
    tr = [{"name": "x", "content": json.dumps({"view": {"app": "y", "preview": {"title": "t"}}})}]
    assert len(extract_action_views(tr)) == 1


def test_extract_action_views_empty():
    assert extract_action_views(None) == []
    assert extract_action_views([]) == []


@pytest.mark.parametrize(
    "app,expected",
    [("mail", "mail"), ("mattermost", "messenger"), ("hr", "hr"),
     ("calendar", "schedule"), ("drive", "drive")],
)
def test_app_source_mapping_card_only_draft(app, expected):
    """비-캘린더 도메인(메일/메신저/휴가): url 없는 카드형 draft + form 계약 고정."""
    view = {
        "app": app, "kind": "draft",
        "preview": {"title": "확인", "fields": [{"label": "x", "value": "y"}]},
        "form": {"kind": "k", "endpoint": "/api/v1/workspace/x", "values": {"a": 1}},
    }
    art = build_action_artifact(view, "c", "m")
    assert art is not None
    assert art["sourceService"] == expected
    assert art["actionKind"] == "draft"
    assert "resourceUrl" not in art                       # 카드형(임베드 URL 없음)
    assert art["payload"]["display"]["title"] == "확인"
    assert art["form"]["endpoint"] == "/api/v1/workspace/x"
    assert art["form"]["values"] == {"a": 1}


# ── 범위 오류(scope_error) 결과는 "내용"이 아니다 ────────────────────────────────
#
# 지식검색이 팀·채널 이름을 해석하지 못하면 **검색을 아예 하지 않고** scope_error +
# 되묻기 후보를 돌려준다. 그 후보 목록이 _CONTENT_KEYS(`channels` 등)에 실리면
# "히트 있음"으로 오판돼 티어 3(지식검색 근거)을 받고, 답변 근거가 없는데도 메신저
# iframe 이 워크스페이스 패널을 차지한다 — 되묻는 답변 옆에 엉뚱한 화면이 열리는 건
# 지금 고치는 버그(2026-08-26 다른 회사 채널 답변)의 시각적 재현이다.

def _scope_error_result():
    return {
        "count": 0, "results": [],
        "scope_error": {"reason": "team_not_found", "message": "…"},
        "channels": [{"team_name": "CustomerA", "channel_name": "개발팀"},
                     {"team_name": "성호금속", "channel_name": "개발팀"}],
    }


def test_scope_error_result_is_not_contentful():
    from agent_arch.agents.tools.action_view import _contribution_tier

    assert _contribution_tier("mattermost_search_knowledge", _scope_error_result()) < 3


def test_contentful_search_still_wins_tier3():
    """가드가 과잉 발동해 정상 지식검색 히트까지 강등되면 안 된다."""
    from agent_arch.agents.tools.action_view import _contribution_tier

    ok = {"count": 3, "results": [{"channel_name": "개발팀"}]}
    assert _contribution_tier("mattermost_search_knowledge", ok) == 3


def test_scope_error_view_does_not_win_workspace_panel():
    """프론트는 *마지막* view 로 이동한다 — 범위 오류 view 가 뒤에 오면 안 된다."""
    mail_hit = {"count": 2, "messages": [{"id": 1}, {"id": 2}],
                "view": {"app": "mail", "preview": {"title": "메일"}}}
    scope_err = dict(_scope_error_result(),
                     view={"app": "mattermost", "preview": {"title": "메신저"}})
    views = extract_action_views([
        {"name": "mail_list_messages", "content": json.dumps(mail_hit, ensure_ascii=False)},
        {"name": "mattermost_search_knowledge", "content": json.dumps(scope_err, ensure_ascii=False)},
    ])
    assert views[-1]["app"] == "mail", "근거 없는 메신저 화면이 패널을 차지했다"
