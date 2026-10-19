"""대화방 / 메시지 히스토리 라우터.

v1 노출 범위:
- ``GET  /api/v1/me/conversation``                                       — 대화방 목록 (집계 포함)
- ``GET  /api/v1/me/conversation/search``                                — 대화 전체 키워드 검색
- ``GET  /api/v1/me/conversation/feedback``                              — 내가 평가한 turn 모아보기
- ``GET  /api/v1/me/conversation/{id}``                                  — 대화방 단건 메타
- ``GET    /api/v1/me/conversation/{id}/messages``                       — 대화방 안 메시지 목록
- ``GET    /api/v1/me/conversation/{id}/messages/{message_id}``          — 단건 메시지 상세
- ``DELETE /api/v1/me/conversation/{id}``                                — 대화방 아카이브(소프트 삭제)

생성은 여전히 lazy-create 경로만 사용한다(POST 미노출). rename(PATCH title)도 아직
미노출 — 제목은 첫 질의에서 자동 파생된다. DELETE는 목록/드로어의 삭제 UX용으로만 archive
(``archived_at`` 세팅)한다. 하드 삭제는 하지 않는다.

응답 노출 정책:
LLM 내부 필드(llm_provider, model_name, *_tokens, llm_invoke_count, tool_calls,
langsmith_run_id, session_id, metadata)는 본 라우터 응답에 절대 포함하지 않는다.
admin 라우터에서 별도 다룬다.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, List, Optional
from urllib.parse import quote

import httpx
from fastapi import APIRouter, HTTPException, Query, Request, Response, status
from fastapi.responses import StreamingResponse

from agent_arch.services import file_service

from agent_arch.db.repository import (
    derive_conversation_title,
    get_conversation,
    get_message_in_conversation,
    list_conversations_for_user,
    list_messages_in_conversation,
    list_rated_messages_for_user,
    search_messages_for_user,
    update_conversation,
)
from agent_arch.models.agent import (
    ConversationDetail,
    ConversationListItem,
    ConversationListResponse,
    ConversationMessage,
    ConversationMessageListResponse,
    ConversationSearchItem,
    ConversationSearchResponse,
    FeedbackItem,
    FeedbackListResponse,
)
from agent_arch.utils.rating import classify_rating

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/me/conversation", tags=["conversation"])


# 라우터 단의 rating 화이트리스트. repository는 ``"any"``를 None으로 정규화한 값을 받는다.
_ALLOWED_RATING_FILTERS: frozenset[str] = frozenset(
    {"positive", "negative", "rated", "unrated", "any"}
)

# 피드백 모아보기 전용 화이트리스트. ``unrated``/``any``는 이 화면에서 의미가 없을 뿐
# 아니라, 무심코 넘기면 "전 대화 전문 덤프"가 되므로 아예 받지 않는다.
_ALLOWED_FEEDBACK_RATING_FILTERS: frozenset[str] = frozenset(
    {"positive", "negative", "rated"}
)


# ---------------------------------------------------------------------------
# 공통 helper
# ---------------------------------------------------------------------------


def _require_user_sub(request: Request) -> str:
    """``request.state.user["sub"]``를 강제 추출 (preferences 라우터와 동일 패턴)."""
    user = getattr(request.state, "user", None) or {}
    sub = user.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="user_sub missing")
    return sub


def _iso_z(value) -> Optional[str]:
    """asyncpg naive datetime을 ISO8601 UTC(Z 접미)로 직렬화한다."""
    if value is None:
        return None
    return value.isoformat() + "Z"


# 날짜만 온 입력을 하루로 펼칠 때 쓰는 기준 시간대. 사용자가 화면에서 고르는 "9월 3일"은
# KST 하루지 UTC 하루가 아니다.
_LOCAL_TZ = timezone(timedelta(hours=9))

_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _parse_iso(value: Optional[str], field: str) -> Optional[datetime]:
    """ISO8601 입력을 naive UTC datetime으로 정규화한다.

    **날짜만(``YYYY-MM-DD``) 오면 그 날 하루로 펼친다** — ``since``는 KST 00:00:00,
    ``until``은 KST 23:59:59.999999. 그냥 자정으로 읽으면 ``until=2026-09-03``이
    "9월 3일 00시까지"가 되어 **종료일 하루가 통째로 빠진다**. 화면에서 날짜를 고른
    사용자에겐 설명할 수 없는 결과다. 시각까지 준 입력은 그대로 존중한다.

    Args:
        value: 사용자 입력 문자열 또는 None.
        field: 필드명 (``"since"`` / ``"until"``). 에러 메시지 + 날짜만 왔을 때
            하루의 어느 끝으로 펼칠지 결정에 쓰인다.

    Returns:
        파싱된 datetime (UTC, tzinfo=None). 입력이 None이면 None.

    Raises:
        HTTPException(400): ISO 파싱 실패.
    """
    if value is None:
        return None

    text = value.strip()
    if _DATE_ONLY_RE.match(text):
        try:
            day = date.fromisoformat(text)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid {field} (ISO8601 expected)")
        bound = (
            datetime.combine(day, time.max, tzinfo=_LOCAL_TZ)
            if field == "until"
            else datetime.combine(day, time.min, tzinfo=_LOCAL_TZ)
        )
        return bound.astimezone(timezone.utc).replace(tzinfo=None)

    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"invalid {field} (ISO8601 expected)")
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ---------------------------------------------------------------------------
# Cursor + ctx
# ---------------------------------------------------------------------------
# cursor payload 구조 (base64.urlsafe_b64encode 후 padding 제거):
#   {"ts": "<iso>", "id": "<uuid>", "ctx": {<필터·정렬 스냅샷>}}
#
# 필터(rating, order, since, until)를 바꾼 채 cursor를 다시 보내면 일관성이 깨지므로
# ctx 불일치를 400으로 차단한다. 프론트는 새 쿼리에 cursor를 동봉하지 않아야 한다.


def _decode_cursor(
    cursor: Optional[str],
    expected_ctx: dict,
    legacy_defaults: Optional[dict] = None,
) -> Optional[tuple[datetime, str]]:
    """opaque cursor를 디코딩하고 ctx 일치를 검증한다.

    Args:
        cursor: 프론트가 echo 한 opaque cursor. None이면 첫 페이지.
        expected_ctx: 이번 요청의 필터 스냅샷.
        legacy_defaults: ctx에 **나중에 추가된** 키의 기본값. 배포 직전에 발급된 cursor는
            그 키가 없어 그대로 비교하면 전부 400이 된다("더 보기"가 죽는다). 없는 키만
            이 기본값으로 채워 비교하므로, 기본값과 다른 값을 고른 요청은 여전히 걸린다.

    Returns:
        ``(timestamp, id)`` 또는 cursor가 None일 때 None.

    Raises:
        HTTPException(400): cursor 변조 또는 ctx 불일치.
    """
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        obj = json.loads(raw)
        ts = datetime.fromisoformat(obj["ts"].replace("Z", "+00:00"))
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        cursor_ctx = obj.get("ctx", {})
        item_id = obj["id"]
    except Exception:
        raise HTTPException(status_code=400, detail="invalid cursor")

    for key, value in (legacy_defaults or {}).items():
        cursor_ctx.setdefault(key, value)

    if cursor_ctx != expected_ctx:
        raise HTTPException(
            status_code=400,
            detail="cursor_context_mismatch: filter/order changed; restart pagination",
        )
    return ts, item_id


def _encode_cursor(sort_key: datetime, item_id: str, ctx: dict) -> str:
    """다음 페이지 cursor를 생성한다. ``sort_keys=True``로 ctx 비교가 안정되도록 보장."""
    blob = json.dumps(
        {"ts": sort_key.isoformat() + "Z", "id": item_id, "ctx": ctx},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return base64.urlsafe_b64encode(blob).rstrip(b"=").decode()


def _build_message_ctx(
    rating: Optional[str],
    order: str,
    since: Optional[str],
    until: Optional[str],
) -> dict:
    """메시지 목록 cursor에 묶을 필터 스냅샷."""
    return {"rating": rating or "any", "order": order, "since": since, "until": until}


def _build_conversation_ctx(
    since: Optional[str], until: Optional[str], order: str = "desc"
) -> dict:
    """대화방 목록 cursor에 묶을 필터 스냅샷."""
    return {"since": since, "until": until, "order": order}


def _build_feedback_ctx(
    rating: str, since: Optional[str], until: Optional[str], order: str = "desc"
) -> dict:
    """피드백 모아보기 cursor에 묶을 필터 스냅샷.

    ``_build_message_ctx``를 재사용하지 않는 이유: 이 목록엔 ``order``가 없어 스냅샷 모양이
    다르고, 그 덕에 ``/messages``에서 받은 cursor를 여기에 넣으면 400으로 정확히 걸린다.
    기존 두 빌더의 키는 절대 건드리지 않는다 — 바꾸면 유통 중인 cursor가 전부 400이 된다.
    """
    return {"rating": rating, "since": since, "until": until, "order": order}


def _parse_jsonb(value) -> Optional[Any]:
    """asyncpg 기본 코덱상 JSONB 가 문자열로 올 수 있어 파이썬 객체로 정규화한다."""
    if not value:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def _row_to_message(row: dict) -> ConversationMessage:
    """repository row → 응답 모델. rating polarity/kind는 단일 출처에서 분류."""
    polarity, kind = classify_rating(row.get("rating"))
    action_view = _parse_jsonb(row.get("action_view"))
    if not isinstance(action_view, dict):
        action_view = None
    attachments = _parse_jsonb(row.get("attachments"))
    if not isinstance(attachments, list):
        attachments = None
    return ConversationMessage(
        message_id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        query=row["query"],
        answer=row["answer"],
        rating=row.get("rating"),
        rating_polarity=polarity,
        rating_kind=kind,
        response_time_sec=row.get("response_time"),
        status_code=row.get("status_code"),
        answer_length=row.get("answer_length"),
        created_at=_iso_z(row["created_at"]),
        action_view=action_view,
        attachments=attachments,
    )


def _row_to_feedback_item(row: dict) -> FeedbackItem:
    """repository row → 피드백 모아보기 항목.

    ``_row_to_message``와 필드 집합이 달라(대화방 제목이 있고 첨부/실행 메타가 없다) 따로
    둔다. rating 분류는 같은 ``classify_rating`` 단일 출처를 쓴다.
    """
    polarity, kind = classify_rating(row.get("rating"))
    return FeedbackItem(
        message_id=str(row["id"]),
        conversation_id=str(row["conversation_id"]),
        conversation_title=row.get("conversation_title"),
        query=row["query"],
        # 검색어가 없으므로 앞부분 발췌로 떨어진다(_snippet 의 미매칭 폴백 경로).
        answer_snippet=_snippet(row.get("answer"), ""),
        rating=row.get("rating"),
        rating_polarity=polarity,
        rating_kind=kind,
        created_at=_iso_z(row["created_at"]),
    )


async def _enrich_attachment_urls(messages: List[ConversationMessage]) -> None:
    """메시지들의 첨부에 presigned 다운로드 URL 을 일괄 발급해 주입한다 (in-place).

    메시지에는 file_id 만 저장돼 있다(URL 은 만료되므로 저장 금지 원칙). 조회 시점에
    file-service 로 일괄 발급받아 각 항목에 ``download_url`` 을 붙인다. file-service
    미설정/실패 시 메타만 내려간다(조용히 생략) — 히스토리 조회 자체는 막지 않는다.
    """
    file_ids = [
        a["file_id"]
        for m in messages
        if m.attachments
        for a in m.attachments
        if isinstance(a, dict) and a.get("file_id")
    ]
    if not file_ids:
        return
    urls = await file_service.download_urls(file_ids)
    if not urls:
        return
    for m in messages:
        if not m.attachments:
            continue
        for a in m.attachments:
            if isinstance(a, dict) and a.get("file_id") in urls:
                a["download_url"] = urls[a["file_id"]]


def _snippet(text: Optional[str], keyword: str, width: int = 200) -> str:
    """검색어가 매칭된 부위를 중심으로 답변을 발췌한다.

    답변이 길면 매칭 부위가 뒤쪽(예: 5000자 지점)에 있을 수 있어, 앞부분을 그냥
    자르면 정작 왜 걸렸는지가 안 보인다. 대소문자 무시로 위치를 잡아 앞뒤로 잘라내고
    잘린 쪽에 말줄임표를 붙인다. 못 찾으면(형태소 차이 등으로 리터럴 매칭 실패)
    앞부분 ``width``자로 폴백한다.

    Args:
        text: 원본 답변. None/빈 문자열이면 빈 문자열 반환.
        keyword: 사용자가 입력한 검색어.
        width: 발췌 길이(문자 수).

    Returns:
        발췌된 평문. 하이라이팅 마크업은 넣지 않는다(클라이언트 몫).
    """
    body = (text or "").strip()
    if not body:
        return ""
    if len(body) <= width:
        return body

    pos = body.lower().find((keyword or "").strip().lower())
    if pos < 0:
        return body[:width].rstrip() + "…"

    # 매칭 위치를 발췌 구간의 대략 중앙에 놓는다.
    start = max(0, pos - width // 2)
    end = min(len(body), start + width)
    start = max(0, end - width)  # 끝에 붙었을 때 앞쪽으로 되밀어 width를 채운다
    return (
        ("…" if start > 0 else "")
        + body[start:end].strip()
        + ("…" if end < len(body) else "")
    )


# ---------------------------------------------------------------------------
# 라우트
# ---------------------------------------------------------------------------


@router.get("", response_model=ConversationListResponse)
async def list_my_conversations(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None, description="다음 페이지 cursor (opaque)"),
    since: Optional[str] = Query(None, description="마지막 활동 시각 하한 (ISO8601)"),
    until: Optional[str] = Query(None, description="마지막 활동 시각 상한 (ISO8601)"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="desc=최신순 | asc=과거순"),
) -> ConversationListResponse:
    """본인 대화방 목록을 마지막 활동 시각 역순으로 조회한다.

    archived 방은 항상 제외. ``message_count`` / ``rated_count`` / ``positive_count`` /
    ``negative_count`` 집계 컬럼이 함께 내려간다(목록의 피드백 배지용).
    ``order``로 최신순(기본)/과거순을 고른다. 정렬 키는 마지막 활동 시각이며, 활동이 없는
    방은 생성 시각으로 대신한다(빈 방이 따로 밀려나지는 않는다).
    cursor는 opaque — 프론트는 그대로 echo만 한다. 필터(``since``/``until``)를 바꾼 채
    cursor를 다시 보내면 400 ``cursor_context_mismatch``.
    """
    user_sub = _require_user_sub(request)
    ctx = _build_conversation_ctx(since, until, order)
    # ``order``는 ctx에 나중에 추가된 키다 — 그 전에 발급된 cursor 를 400 으로 죽이지 않는다.
    cur = _decode_cursor(cursor, expected_ctx=ctx, legacy_defaults={"order": "desc"})
    since_dt = _parse_iso(since, "since")
    until_dt = _parse_iso(until, "until")

    rows = await list_conversations_for_user(
        user_sub=user_sub,
        limit=limit,
        cursor=cur,
        since=since_dt,
        until=until_dt,
        order=order,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [
        ConversationListItem(
            # 저장된 title 이 없으면(과거 대화/미생성) 첫 질의에서 즉석 유도해 목록에
            # 의미 있는 제목이 뜨게 한다. 둘 다 없으면 None → 프론트가 폴백 표시.
            conversation_id=str(r["id"]),
            title=r.get("title") or derive_conversation_title(r.get("first_query")),
            created_at=_iso_z(r["created_at"]),
            last_message_at=_iso_z(r.get("last_message_at")),
            message_count=r.get("message_count", 0),
            rated_count=r.get("rated_count", 0),
            positive_count=r.get("positive_count", 0),
            negative_count=r.get("negative_count", 0),
        )
        for r in rows
    ]

    next_cursor: Optional[str] = None
    if has_more and rows:
        last = rows[-1]
        sort_key = last.get("last_message_at") or last["created_at"]
        next_cursor = _encode_cursor(sort_key, str(last["id"]), ctx)

    return ConversationListResponse(items=items, next_cursor=next_cursor, has_more=has_more)


# 주의: 이 라우트는 ``/{conversation_id}`` 보다 **먼저** 선언돼야 한다. FastAPI는 선언
# 순서대로 매칭하므로 뒤에 두면 "search"가 conversation_id로 잡혀 400(invalid uuid)이 난다.
@router.get("/search", response_model=ConversationSearchResponse)
async def search_my_conversations(
    request: Request,
    q: str = Query(..., min_length=2, description="검색어 (2자 이상)"),
    limit: int = Query(20, ge=1, le=50),
    offset: int = Query(0, ge=0, le=500, description="건너뛸 건수 (관련도 정렬이라 cursor 대신 offset)"),
    rating: Optional[str] = Query(
        None,
        description="평가 필터: positive | negative | rated | unrated | any",
    ),
    since: Optional[str] = Query(None, description="생성 시각 하한 (ISO8601)"),
    until: Optional[str] = Query(None, description="생성 시각 상한 (ISO8601)"),
) -> ConversationSearchResponse:
    """본인 대화 전체를 가로질러 키워드로 검색한다.

    pg_trgm 유사도 순으로 정렬되며, 아카이브(삭제)한 대화방은 결과에서 빠진다.
    답변은 전문이 아니라 매칭 부위 중심 발췌(``answer_snippet``)로 내려간다.

    기간(``since``/``until``)과 평가(``rating``) 필터를 함께 받는다 — "지난주 대화 중에"
    또는 "내가 별로예요 준 것 중에" 로 좁히는 게 검색창에서 가장 흔한 후속 동작이다.
    offset 방식이라 cursor ctx 정합 문제는 없다(필터를 바꿔도 400이 나지 않는다).
    """
    user_sub = _require_user_sub(request)

    if rating is not None and rating not in _ALLOWED_RATING_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"rating must be one of {sorted(_ALLOWED_RATING_FILTERS)}",
        )
    since_dt = _parse_iso(since, "since")
    until_dt = _parse_iso(until, "until")

    rows = await search_messages_for_user(
        user_sub,
        q,
        limit=limit,
        offset=offset,
        rating_filter=rating if rating != "any" else None,
        since=since_dt,
        until=until_dt,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]

    items = [
        ConversationSearchItem(
            message_id=str(r["id"]),
            conversation_id=str(r["conversation_id"]),
            conversation_title=r.get("conversation_title"),
            query=r["query"],
            answer_snippet=_snippet(r.get("answer"), q),
            created_at=_iso_z(r["created_at"]),
            score=float(r.get("score") or 0.0),
        )
        for r in rows
    ]

    return ConversationSearchResponse(items=items, total_hint=None, has_more=has_more)


# 주의: 이 라우트도 ``/{conversation_id}`` 보다 **먼저** 선언돼야 한다. 뒤에 두면
# "feedback"이 conversation_id로 잡혀 400(invalid conversation_id)이 난다.
@router.get("/feedback", response_model=FeedbackListResponse)
async def list_my_feedback(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None, description="다음 페이지 cursor (opaque)"),
    rating: str = Query("rated", description="평가 필터: positive | negative | rated"),
    since: Optional[str] = Query(None, description="생성 시각 하한 (ISO8601)"),
    until: Optional[str] = Query(None, description="생성 시각 상한 (ISO8601)"),
    order: str = Query("desc", pattern="^(asc|desc)$", description="desc=최신순 | asc=과거순"),
) -> FeedbackListResponse:
    """본인이 평가를 남긴 turn을 대화방을 가로질러 최신순으로 모아 본다.

    "내가 별로예요 준 게 어디 있었지"를 대화방마다 열어보지 않고 한 목록에서 훑기 위한
    화면이다. 기본값이 ``rated``(전체 평가)인 이유는, 부정만 내려보내면 "내 피드백 전체"를
    보려면 파라미터를 알아야만 하기 때문 — 프론트가 기본 탭을 ``negative``로 잡는다.

    ``unrated``/``any``는 받지 않는다(400). 미평가까지 열면 전 대화 덤프가 된다.
    관련도가 아니라 시간 정렬이라 검색과 달리 opaque cursor를 쓴다.
    """
    user_sub = _require_user_sub(request)

    if rating not in _ALLOWED_FEEDBACK_RATING_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"rating must be one of {sorted(_ALLOWED_FEEDBACK_RATING_FILTERS)}",
        )

    ctx = _build_feedback_ctx(rating, since, until, order)
    cur = _decode_cursor(cursor, expected_ctx=ctx, legacy_defaults={"order": "desc"})
    since_dt = _parse_iso(since, "since")
    until_dt = _parse_iso(until, "until")

    rows = await list_rated_messages_for_user(
        user_sub,
        limit=limit,
        cursor=cur,
        rating_filter=rating,
        since=since_dt,
        until=until_dt,
        order=order,
    )
    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [_row_to_feedback_item(r) for r in rows]

    next_cursor: Optional[str] = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = _encode_cursor(last["created_at"], str(last["id"]), ctx)

    return FeedbackListResponse(items=items, next_cursor=next_cursor, has_more=has_more)


@router.get("/{conversation_id}", response_model=ConversationDetail)
async def get_my_conversation(
    request: Request,
    conversation_id: str,
) -> ConversationDetail:
    """본인 소유 대화방 단건 메타. 타인 소유 또는 비-UUID는 404 / 400."""
    user_sub = _require_user_sub(request)
    try:
        uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid conversation_id")

    row = await get_conversation(conversation_id, user_sub)
    if row is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    return ConversationDetail(
        conversation_id=str(row["id"]),
        title=row.get("title"),
        created_at=_iso_z(row["created_at"]),
        last_message_at=_iso_z(row.get("last_message_at")),
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_my_conversation(
    request: Request,
    conversation_id: str,
) -> Response:
    """본인 소유 대화방을 아카이브(소프트 삭제)한다.

    ``archived_at``을 세팅해 목록(``list_conversations_for_user``)에서 즉시 제외한다.
    하드 삭제가 아니라 감사/복구 여지를 남긴다. 타인 소유 또는 존재하지 않으면 404,
    비-UUID는 400. 이미 아카이브된 방을 다시 호출하면 갱신 행이 0이라 404가 될 수 있어
    호출 전 존재를 확인한다(멱등 삭제 UX).
    """
    user_sub = _require_user_sub(request)
    try:
        uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid conversation_id")

    # 소유·존재 확인(타인 방 404). 이미 아카이브 상태여도 update가 0행일 수 있어 선확인.
    if await get_conversation(conversation_id, user_sub) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    await update_conversation(conversation_id, user_sub, archived=True)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{conversation_id}/messages",
    response_model=ConversationMessageListResponse,
)
async def list_messages(
    request: Request,
    conversation_id: str,
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None, description="다음 페이지 cursor (opaque)"),
    rating: Optional[str] = Query(
        None,
        description="평가 필터: positive | negative | rated | unrated | any",
    ),
    since: Optional[str] = Query(None, description="생성 시각 하한 (ISO8601)"),
    until: Optional[str] = Query(None, description="생성 시각 상한 (ISO8601)"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
) -> ConversationMessageListResponse:
    """본인 소유 대화방의 메시지를 페이지네이션 조회한다.

    빈 응답 + cursor 없음이면 대화방 존재 여부를 한 번 더 확인해 404로 변환한다 —
    "타인 대화방"과 "본인의 빈 방"을 구분하기 위함. cursor가 있는 후속 페이지는
    빈 응답이어도 단순 종결로 처리한다.
    """
    user_sub = _require_user_sub(request)

    try:
        uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid conversation_id")

    if rating is not None and rating not in _ALLOWED_RATING_FILTERS:
        raise HTTPException(
            status_code=400,
            detail=f"rating must be one of {sorted(_ALLOWED_RATING_FILTERS)}",
        )

    ctx = _build_message_ctx(rating, order, since, until)
    cur = _decode_cursor(cursor, expected_ctx=ctx)
    since_dt = _parse_iso(since, "since")
    until_dt = _parse_iso(until, "until")

    rows = await list_messages_in_conversation(
        conversation_id=conversation_id,
        user_sub=user_sub,
        limit=limit,
        cursor=cur,
        rating_filter=rating if rating != "any" else None,
        since=since_dt,
        until=until_dt,
        order=order,
    )

    # 빈 결과이고 cursor도 없으면 "대화방 자체가 본인 소유가 아님" 가능성을 따로 확인.
    if not rows and cursor is None:
        exists = await get_conversation(conversation_id, user_sub)
        if exists is None:
            raise HTTPException(status_code=404, detail="conversation not found")

    has_more = len(rows) > limit
    rows = rows[:limit]
    items = [_row_to_message(r) for r in rows]
    # 첨부 이미지 복원 — file_id 로 presigned 다운로드 URL 을 조회 시점에 일괄 발급
    await _enrich_attachment_urls(items)

    next_cursor: Optional[str] = None
    if has_more and rows:
        last = rows[-1]
        next_cursor = _encode_cursor(last["created_at"], str(last["id"]), ctx)

    return ConversationMessageListResponse(
        items=items, next_cursor=next_cursor, has_more=has_more
    )


@router.get(
    "/{conversation_id}/messages/{message_id}",
    response_model=ConversationMessage,
)
async def get_message(
    request: Request,
    conversation_id: str,
    message_id: str,
) -> ConversationMessage:
    """본인 소유 대화방 안 단건 메시지 상세."""
    user_sub = _require_user_sub(request)
    for label, val in (("conversation_id", conversation_id), ("message_id", message_id)):
        try:
            uuid.UUID(val)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"invalid {label}")

    row = await get_message_in_conversation(conversation_id, message_id, user_sub)
    if row is None:
        raise HTTPException(status_code=404, detail="message not found")
    msg = _row_to_message(row)
    await _enrich_attachment_urls([msg])
    return msg


# 첨부 소유 확인 시 훑을 메시지 수. 한 대화의 첨부는 앞쪽 몇 턴에 몰려 있고(도면을 올리고
# 대화가 이어진다), 전량 스캔은 긴 대화에서 비싸다. 여기 안에 없으면 404 로 돌려 사용자가
# 다시 올리게 한다 — 넉넉히 잡되 무한은 아니게.
_ATTACHMENT_SCAN_LIMIT = 200


@router.get("/{conversation_id}/attachments/{file_id}/download")
async def download_conversation_attachment(
    request: Request,
    conversation_id: str,
    file_id: str,
):
    """본인 대화방의 첨부 원본을 스트리밍한다.

    **왜 presigned URL 을 그냥 주지 않고 흘려주는가.** 히스토리 조회는 이미 첨부마다
    ``download_url``(file-service presigned)을 내려준다. 그건 `<img src>` 로 그리기엔
    충분하지만, 다른 앱이 브라우저에서 ``fetch`` 로 **바이트를 가져다 쓰려면** 다른
    오리진이라 CORS 에 막힌다. 게이트웨이/BFF 를 지나는 이 경로는 호출 앱과 같은 오리진이
    되므로 그 문제가 없다.

    **소유 검증을 여기서 한다.** file_id 만으로 파일을 내주면 남의 대화 첨부를 file_id 추측
    으로 긁을 수 있다. 그래서 (a) 대화방이 요청자 것인지, (b) 그 file_id 가 **그 대화의**
    첨부인지 둘 다 확인한다. 하나라도 아니면 404 — 403 은 "그 파일은 존재한다" 를 알려준다.

    Args:
        request: X-User-Sub 를 담은 요청.
        conversation_id: 대화방 UUID.
        file_id: file-service 파일 id.

    Returns:
        원본 바이트 스트림. Content-Disposition 에 원래 파일명을 싣는다.

    Raises:
        HTTPException: 비-UUID 400, 남의 방·없는 첨부 404, file-service 미설정 503.
    """
    user_sub = _require_user_sub(request)
    try:
        uuid.UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid conversation_id")

    # 소유 확인 — list_messages_in_conversation 도 JOIN 으로 검증하지만, 방이 아예 없을 때
    # "첨부 없음(404)" 과 "방 없음(404)" 을 같은 응답으로 맞추려면 여기서 먼저 본다.
    if await get_conversation(conversation_id, user_sub) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    rows = await list_messages_in_conversation(
        conversation_id, user_sub, limit=_ATTACHMENT_SCAN_LIMIT
    )
    meta: Optional[dict] = None
    for row in rows:
        for att in _parse_jsonb(row.get("attachments")) or []:
            if isinstance(att, dict) and att.get("file_id") == file_id:
                meta = att
                break
        if meta:
            break
    if meta is None:
        raise HTTPException(status_code=404, detail="attachment not found")

    if not file_service.enabled():
        raise HTTPException(status_code=503, detail="file service not configured")
    urls = await file_service.download_urls([file_id])
    source = urls.get(file_id)
    if not source:
        # 스토리지에서 사라졌거나 발급이 실패했다 — 호출부가 "직접 첨부하라" 로 안내한다.
        raise HTTPException(status_code=404, detail="attachment not available")

    file_name = str(meta.get("filename") or "attachment")
    media_type = str(meta.get("content_type") or "application/octet-stream")

    async def _body():
        # 클라이언트를 제너레이터 안에서 열고 닫는다 — 밖에서 열면 응답이 흐르기 전에 닫힌다.
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("GET", source) as upstream:
                upstream.raise_for_status()
                async for chunk in upstream.aiter_bytes():
                    yield chunk

    # 한글 파일명은 RFC 5987 로 싣는다 — filename= 만 쓰면 헤더가 latin-1 인코딩에서 깨진다.
    disposition = f"attachment; filename*=UTF-8''{quote(file_name)}"
    return StreamingResponse(
        _body(), media_type=media_type, headers={"Content-Disposition": disposition}
    )
