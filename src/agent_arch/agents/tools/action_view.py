"""실제 action 도구 결과 → 워크스페이스 패널용 artifact (workspace.* 이벤트).

dmp-frontend의 기존 워크스페이스 패널(demo-agent-workspace: ArtifactStore +
app-embed iframe 렌더)을 **그대로 재활용**한다. 즉 백엔드는 그 패널이 이미
소비하는 ``workspace.open/update/complete`` 시퀀스 + artifact 스키마로 내보내고,
프론트는 변경 없이(또는 최소 수정으로) 렌더한다.

action 도구(예: nextcloud calendar/drive)는 성공 출력에 ``view`` 디스크립터를
동봉하고, orchestrator가 루프 종료 후 이를 뽑아 artifact로 합성·방출한다.

view 계약 (도구 출력에 동봉):
    { "app": "calendar"|"drive"|...,        # 소스 앱 → sourceService 매핑
      "kind": "draft"|"result"|...,          # draft=승인대기 / result=완료
      "url": "<https 딥링크>",               # app-embed resourceUrl (iframe 대상)
      "auth_url": "<https SSO 세션확립 URL>", # 선택 → artifact.authUrl (프론트 1차 로드용)
      "label": "...",                        # artifact.title
      "resource_id": "...",                  # 선택
      "preview": { "title","summary","fields":[{label,value}] } }  # payload.display
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# view.app → artifact.sourceService 매핑 (프론트 패널이 아이콘/라벨에 사용).
_APP_SOURCE: Dict[str, str] = {
    "mattermost": "messenger",
    "talk": "messenger",
    "nextcloud": "drive",
    "drive": "drive",
    "calendar": "schedule",
    "mailcow": "mail",
    "mail": "mail",
    "hr": "hr",
    "eapproval": "eapproval",
}
_TENANT = "default"
_RESOURCE_KEY_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_key(value: str) -> str:
    cleaned = _RESOURCE_KEY_RE.sub("-", value or "").strip("-")
    return cleaned or "unknown"


# 도구 결과가 "내용"을 반환했는지 판단할 때 확인하는 리스트형 payload 키들.
_CONTENT_KEYS = ("results", "messages", "items", "posts", "channels", "files", "events")


def _is_contentful(data: dict) -> bool:
    """도구 결과가 실질 내용(검색/조회 히트)을 담고 있는지 — count>0 또는 비지 않은 리스트.

    ⚠️ **범위 오류(`scope_error`)는 내용이 아니다.** 지식검색이 팀·채널 이름을 해석하지
    못하면 검색을 아예 하지 않고 `scope_error` + 되묻기 후보를 돌려주는데, 그 후보 목록이
    `_CONTENT_KEYS`(`channels`·`items` 등)에 실리면 "히트 있음"으로 오판돼 티어 3을 받고
    **답변 근거가 없는데도 메신저 iframe이 워크스페이스 패널을 차지한다.** 되묻는 답변
    옆에 엉뚱한 화면이 열리는 건 지금 고치는 버그의 시각적 재현이다.
    """
    if not isinstance(data, dict):
        return False
    if data.get("scope_error"):
        return False
    if isinstance(data.get("count"), int) and data["count"] > 0:
        return True
    return any(isinstance(data.get(k), list) and data.get(k) for k in _CONTENT_KEYS)


def _contribution_tier(name: str, data: dict) -> int:
    """view 우선순위용 기여도 티어. 높을수록 '답변 기여'가 크다.

    0 = 순수 화면 열기(``*_open_screen``, 내용 기여 없음) → 가장 낮음.
    1 = 내용 도구지만 빈 결과.
    2 = 내용 도구 + 실질 결과(일반 조회/읽기 히트).
    3 = 지식검색(``*_search_knowledge``) 히트 — RAG 답변의 명시적 근거라 가장 강함.
    """
    name = name if isinstance(name, str) else ""
    if name.endswith("_open_screen"):
        return 0
    contentful = _is_contentful(data)
    if name.endswith("_search_knowledge") and contentful:
        return 3
    return 2 if contentful else 1


def extract_action_views(tool_results: Optional[list]) -> List[dict]:
    """LoopResult.tool_results 에서 ``view`` 디스크립터를 **답변 기여도 순으로** 뽑는다.

    각 결과 content는 executor.serialize_result 가 만든 JSON 문자열이며, action 도구는
    성공 출력에 ``view`` 키를 동봉한다. 실패([ERROR] …)·view 없는 결과는 건너뛴다.

    **정렬(우선순위):** 프론트 워크스페이스 패널은 *마지막에 열린* iframe 으로 이동하므로,
    답변에 실제로 기여한 도메인의 view 가 **마지막에** 오도록 정렬한다. 기준은
    ``(기여 티어, 그 도메인의 총 기여점수, 원래 순서)`` 오름차순 — 예컨대 메신저 지식검색이
    히트를 냈고 메일은 화면만 열었다면, 메일 view 를 먼저·메신저 view 를 나중에 방출해
    왼쪽 패널이 메일이 아니라 근거가 된 메신저 대화로 열린다.
    """
    try:
        from agent_arch.agents.graphs.router import domain_from_tool_names
    except Exception:  # 임포트 실패 시 도메인 점수 없이 티어+순서로만 정렬(안전 폴백)
        domain_from_tool_names = None

    # 1) 모든 결과를 한 번 파싱: (원래 인덱스, 도구명, 파싱 data)
    parsed: List[tuple] = []
    for idx, tr in enumerate(tool_results or []):
        if not isinstance(tr, dict):
            continue
        name = tr.get("name")
        content = tr.get("content")
        data = None
        if isinstance(content, str) and content.lstrip()[:1] in ("{", "["):
            try:
                data = json.loads(content)
            except (ValueError, TypeError):
                data = None
        parsed.append((idx, name, data))

    # 2) 도메인별 기여 점수(open_screen 은 제외) — view 없는 내용 도구도 도메인 기여로 집계.
    domain_score: Dict[str, int] = {}
    if domain_from_tool_names is not None:
        for _idx, name, data in parsed:
            if not isinstance(data, dict):
                continue
            tier = _contribution_tier(name or "", data)
            if tier <= 0:
                continue
            try:
                dom = domain_from_tool_names([name]) if name else None
            except Exception:
                dom = None
            if dom:
                domain_score[dom] = domain_score.get(dom, 0) + tier

    # 3) view 를 가진 결과만 모아 (티어, 도메인점수, 원래순서) 로 정렬.
    ranked: List[tuple] = []
    for idx, name, data in parsed:
        if not isinstance(data, dict):
            continue
        view = data.get("view")
        if not (isinstance(view, dict) and (view.get("url") or view.get("preview"))):
            continue
        tier = _contribution_tier(name or "", data)
        dom = None
        if domain_from_tool_names is not None and name:
            try:
                dom = domain_from_tool_names([name])
            except Exception:
                dom = None
        ranked.append(((tier, domain_score.get(dom, 0), idx), view))

    ranked.sort(key=lambda x: x[0])
    return [view for _key, view in ranked]


def build_action_artifact(
    view: Optional[dict],
    conversation_id: Optional[str],
    message_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """``view`` → dmp-frontend 워크스페이스 패널이 렌더하는 artifact dict로 합성한다.

    데모 패널이 이미 소비하는 app-embed 스키마(필수 필드 + resourceUrl + payload.display)
    를 따른다. ``url``은 https만 통과(임베드 허용 origin 판단은 프론트의
    AGENT_APP_EMBED_ALLOWED_ORIGINS가 담당). url·preview 둘 다 없으면 None.

    Args:
        view: 도구 출력의 ``view`` dict.
        conversation_id / message_id: artifact 식별 컨텍스트.

    Returns:
        artifact dict (workspace.update 동봉용) 또는 None.
    """
    if not isinstance(view, dict):
        return None
    raw_url = view.get("url")
    url = raw_url if isinstance(raw_url, str) and raw_url.startswith("https://") else None
    preview = view.get("preview") if isinstance(view.get("preview"), dict) else None
    if url is None and preview is None:
        return None

    app = str(view.get("app") or "app")
    source_service = _APP_SOURCE.get(app, _safe_key(app))
    kind = str(view.get("kind") or "result")
    resource_key = _safe_key(str(view.get("resource_id") or kind or "result"))

    # dmp-frontend artifact 검증(assertString)은 conversationId/messageId 가 비어있지
    # 않은 문자열일 것을 요구한다. 신규 대화 등 conversation_id/message_id 가 아직 비어
    # 있을 때 "" 를 넣으면 "Artifact 필수 필드가 없습니다: conversationId" 로 거부되므로
    # 비어있지 않은 fallback(소스+리소스 키)을 보장한다.
    conv_id_val = conversation_id or message_id or f"{source_service}-{resource_key}"
    msg_id_val = message_id or conv_id_val

    # url 있는 view(예: 캘린더 새 일정 팝오버)는 app-embed 로 — host 가 iframe 미리보기
    # 화면을 띄우고, 동시에 @org/ui 가 같은 update 에서 actionKind=draft+form 으로
    # 승인/거절 카드를 렌더한다(미리보기 iframe + 카드 둘 다). url 없는 draft 만
    # 카드 전용(approval-draft), 그 외(완료 등)는 approval-result.
    if url is not None:
        art_type, freshness = "app-embed", "live"
    elif kind == "draft":
        art_type, freshness = "approval-draft", "snapshot"
    else:
        art_type, freshness = "approval-result", "snapshot"

    artifact: Dict[str, Any] = {
        "id": f"artifact:{_TENANT}:{source_service}:{art_type}:{freshness}:{resource_key}:1.0",
        "tenantId": _TENANT,
        "conversationId": conv_id_val,
        "messageId": msg_id_val,
        "sourceService": source_service,
        "type": art_type,
        "schemaVersion": "1.0",
        "freshness": freshness,
        "title": str(view.get("label") or (preview or {}).get("title") or "실행 결과"),
        "createdAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        # draft(승인대기) / result(완료) — 프론트가 상태 배지·승인 힌트에 사용 가능.
        "actionKind": kind,
    }
    if url is not None:
        artifact["resourceId"] = str(view.get("resource_id") or resource_key)
        artifact["resourceUrl"] = url
        # 선택: SSO 세션 확립용 1차 로드 URL. 프론트가 iframe 을 authUrl 로 먼저 띄워
        # (무음 로그인으로 세션 확립) 로드 완료 후 src 를 resourceUrl 로 바꿔 해당
        # 리소스로 정확히 이동한다. 일부 앱(MM)이 SSO-wrap 의 redirect_to 를 무시해
        # 딥링크가 깨지는 걸 우회하기 위함. 없으면 프론트는 resourceUrl 단독 로드.
        raw_auth = view.get("auth_url")
        if isinstance(raw_auth, str) and raw_auth.startswith("https://"):
            artifact["authUrl"] = raw_auth
    if preview is not None:
        artifact["payload"] = {"display": preview}
    # 편집 폼 계약(values + 승인 POST endpoint) — 프론트가 네이티브 편집 폼을
    # 렌더하고 '승인' 시 endpoint로 수정값을 보내 실제 생성하게 한다.
    form = view.get("form")
    if isinstance(form, dict):
        artifact["form"] = form
    return artifact
