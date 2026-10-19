"""conversation 라우터 핸들러 단위 테스트.

- ``_require_user_sub`` / cursor encode·decode 동작
- 4개 라우트(목록/단건/메시지목록/메시지단건) 정상 / 예외 분기
- 응답 노출 정책: LLM 내부 필드 키 절대 부재
- Cursor ctx 불일치 시 400

repository 호출은 monkeypatch로 in-memory fake 함수로 대체한다.
"""
from __future__ import annotations

import json
import base64
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException

from agent_arch.api.v1 import conversation as conv_router
from agent_arch.api.v1.conversation import (
    _build_conversation_ctx,
    _build_message_ctx,
    _decode_cursor,
    _encode_cursor,
    _parse_iso,
    _row_to_message,
    get_message,
    get_my_conversation,
    list_messages,
    list_my_conversations,
)


# ---------------------------------------------------------------------------
# 공통 helper
# ---------------------------------------------------------------------------


def _make_request(sub: Optional[str] = "u-1") -> Any:
    """``request.state.user["sub"]``만 흉내내는 Request 대체.

    sub=None이면 user 없음 — 401 분기 트리거.
    """
    state = SimpleNamespace(user={"sub": sub} if sub else None)
    return SimpleNamespace(state=state)


# 핸들러를 직접 호출할 때 FastAPI Query(...) 기본값 객체가 그대로 인자로 들어가
# truthy로 평가되는 문제 회피용 헬퍼. 단위 테스트는 TestClient를 거치지 않으므로
# 호출자가 모든 Query 파라미터를 명시적으로 None / 기본값으로 채워준다.


async def _call_list_conv(req, *, limit=20, cursor=None, since=None, until=None, order="desc"):
    return await list_my_conversations(
        req, limit=limit, cursor=cursor, since=since, until=until, order=order
    )


async def _call_list_messages(
    req,
    conv_id,
    *,
    limit=20,
    cursor=None,
    rating=None,
    since=None,
    until=None,
    order="desc",
):
    return await list_messages(
        req,
        conv_id,
        limit=limit,
        cursor=cursor,
        rating=rating,
        since=since,
        until=until,
        order=order,
    )


def _row(
    *,
    id: str = "m1",
    conversation_id: str = "c1",
    rating: Optional[str] = None,
    created_at: datetime = datetime(2026, 5, 14, 10, 0, 0),
    **extra: Any,
) -> dict:
    base = {
        "id": id,
        "conversation_id": conversation_id,
        "query": "q",
        "answer": "a",
        "rating": rating,
        "response_time": 1.23,
        "status_code": 200,
        "answer_length": 1,
        "created_at": created_at,
    }
    base.update(extra)
    return base


# ---------------------------------------------------------------------------
# cursor encode/decode
# ---------------------------------------------------------------------------


class TestCursorRoundTrip:
    """cursor encode 후 decode가 동일 (timestamp, id)로 복원되는지."""

    def test_round_trip(self):
        ctx = {"rating": "any", "order": "desc", "since": None, "until": None}
        ts = datetime(2026, 5, 14, 10, 30, 45)
        token = _encode_cursor(ts, "id-1", ctx)
        decoded = _decode_cursor(token, expected_ctx=ctx)
        assert decoded == (ts, "id-1")

    def test_decode_none_returns_none(self):
        assert _decode_cursor(None, expected_ctx={}) is None

    def test_decode_garbage_raises_400(self):
        with pytest.raises(HTTPException) as exc:
            _decode_cursor("not-base64-at-all!@#", expected_ctx={})
        assert exc.value.status_code == 400
        assert "invalid cursor" in exc.value.detail

    def test_decode_ctx_mismatch_raises_400(self):
        """cursor 생성 시 ctx와 다른 ctx로 디코드 → cursor_context_mismatch."""
        ctx1 = {"rating": "negative", "order": "desc", "since": None, "until": None}
        ctx2 = {"rating": "positive", "order": "desc", "since": None, "until": None}
        token = _encode_cursor(datetime(2026, 5, 14), "id-1", ctx1)

        with pytest.raises(HTTPException) as exc:
            _decode_cursor(token, expected_ctx=ctx2)
        assert exc.value.status_code == 400
        assert "cursor_context_mismatch" in exc.value.detail


# ---------------------------------------------------------------------------
# _row_to_message — classify_rating 단일 출처 활용
# ---------------------------------------------------------------------------


class TestRowToMessage:
    """row → ConversationMessage 변환 시 rating 분류 결과가 함께 채워진다."""

    def test_thumbs_up_classified_positive_preset(self):
        msg = _row_to_message(_row(rating="thumbs_up"))
        assert msg.rating == "thumbs_up"
        assert msg.rating_polarity == "positive"
        assert msg.rating_kind == "preset"

    def test_freeform_korean_classified_negative_freeform(self):
        msg = _row_to_message(_row(rating="표현이 어색함"))
        assert msg.rating_polarity == "negative"
        assert msg.rating_kind == "freeform"

    def test_null_rating(self):
        msg = _row_to_message(_row(rating=None))
        assert msg.rating is None
        assert msg.rating_polarity is None
        assert msg.rating_kind is None


# ---------------------------------------------------------------------------
# list_my_conversations
# ---------------------------------------------------------------------------


class TestListMyConversations:
    """``GET /me/conversation`` 핸들러."""

    @pytest.mark.asyncio
    async def test_missing_user_sub_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            await _call_list_conv(_make_request(sub=None))
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_returns_items_with_aggregates(self, monkeypatch):
        """repository row의 집계 4종이 응답에 그대로 전달된다."""
        rows = [
            {
                "id": "c1",
                "title": None,
                "created_at": datetime(2026, 5, 13, 9, 0),
                "last_message_at": datetime(2026, 5, 14, 9, 0),
                "message_count": 5,
                "rated_count": 2,
                "positive_count": 1,
                "negative_count": 1,
            }
        ]

        async def _fake(**kwargs):
            return rows

        monkeypatch.setattr(conv_router, "list_conversations_for_user", _fake)

        resp = await _call_list_conv(_make_request(), limit=20)
        assert len(resp.items) == 1
        assert resp.items[0].conversation_id == "c1"
        assert resp.items[0].message_count == 5
        assert resp.items[0].rated_count == 2
        assert resp.items[0].positive_count == 1
        assert resp.items[0].negative_count == 1
        assert resp.items[0].title is None
        assert resp.has_more is False
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_order_forwarded_to_repository(self, monkeypatch):
        """정렬 선택이 repository 로 그대로 넘어간다 (기본은 최신순)."""
        calls = []

        async def _fake(**kwargs):
            calls.append(kwargs)
            return []

        monkeypatch.setattr(conv_router, "list_conversations_for_user", _fake)

        await _call_list_conv(_make_request())
        assert calls[0]["order"] == "desc"

        await _call_list_conv(_make_request(), order="asc")
        assert calls[1]["order"] == "asc"

    @pytest.mark.asyncio
    async def test_has_more_when_repo_returns_limit_plus_one(self, monkeypatch):
        """repo가 limit+1개 반환 → has_more=True, items는 limit만큼만, next_cursor 생성."""
        rows = [
            {
                "id": f"c{i}",
                "title": None,
                "created_at": datetime(2026, 5, 13, 9, 0),
                "last_message_at": datetime(2026, 5, 14, 9, i),
                "message_count": 1,
                "rated_count": 0,
            }
            for i in range(3)
        ]

        async def _fake(**kwargs):
            return rows  # limit=2일 때 3개 반환 → has_more

        monkeypatch.setattr(conv_router, "list_conversations_for_user", _fake)

        resp = await _call_list_conv(_make_request(), limit=2)
        assert resp.has_more is True
        assert len(resp.items) == 2
        assert resp.next_cursor is not None


# ---------------------------------------------------------------------------
# get_my_conversation
# ---------------------------------------------------------------------------


class TestGetMyConversation:
    """``GET /me/conversation/{id}`` 핸들러."""

    @pytest.mark.asyncio
    async def test_invalid_uuid_400(self):
        with pytest.raises(HTTPException) as exc:
            await get_my_conversation(_make_request(), "not-a-uuid")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_not_found_404(self, monkeypatch):
        async def _none(*a, **kw):
            return None
        monkeypatch.setattr(conv_router, "get_conversation", _none)

        with pytest.raises(HTTPException) as exc:
            await get_my_conversation(
                _make_request(), "00000000-0000-0000-0000-000000000001"
            )
        assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# list_messages
# ---------------------------------------------------------------------------


_VALID_UUID = "00000000-0000-0000-0000-000000000001"


class TestListMessages:
    """``GET /me/conversation/{id}/messages`` 핸들러."""

    @pytest.mark.asyncio
    async def test_invalid_conv_id_400(self):
        with pytest.raises(HTTPException) as exc:
            await _call_list_messages(_make_request(), "bad-uuid")
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_rating_invalid_value_400(self):
        with pytest.raises(HTTPException) as exc:
            await _call_list_messages(_make_request(), _VALID_UUID, rating="up")
        assert exc.value.status_code == 400
        assert "rating must be one of" in exc.value.detail

    @pytest.mark.asyncio
    async def test_empty_owned_conversation_returns_empty_items(self, monkeypatch):
        """본인 소유 + 메시지 0건 → 200 + 빈 items (404 아님)."""
        async def _empty(**kw):
            return []
        async def _exists(*a, **kw):
            return {"id": _VALID_UUID}

        monkeypatch.setattr(conv_router, "list_messages_in_conversation", _empty)
        monkeypatch.setattr(conv_router, "get_conversation", _exists)

        resp = await _call_list_messages(_make_request(), _VALID_UUID)
        assert resp.items == []
        assert resp.has_more is False

    @pytest.mark.asyncio
    async def test_alien_conversation_returns_404(self, monkeypatch):
        """타인 conv_id → repo 빈 리스트 + exists None → 404."""
        async def _empty(**kw):
            return []
        async def _none(*a, **kw):
            return None

        monkeypatch.setattr(conv_router, "list_messages_in_conversation", _empty)
        monkeypatch.setattr(conv_router, "get_conversation", _none)

        with pytest.raises(HTTPException) as exc:
            await _call_list_messages(_make_request(), _VALID_UUID)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_rows_mapped_to_message_model(self, monkeypatch):
        """repo row → ConversationMessage 필드 매핑."""
        async def _rows(**kw):
            return [_row(id="m1", conversation_id=_VALID_UUID, rating="thumbs_up")]
        monkeypatch.setattr(conv_router, "list_messages_in_conversation", _rows)

        resp = await _call_list_messages(_make_request(), _VALID_UUID)
        assert len(resp.items) == 1
        item = resp.items[0]
        assert item.message_id == "m1"
        assert item.rating == "thumbs_up"
        assert item.rating_polarity == "positive"
        assert item.rating_kind == "preset"

    @pytest.mark.asyncio
    async def test_response_dump_omits_llm_internal_fields(self, monkeypatch):
        """JSON 직렬화 후에도 LLM 내부 필드는 키 자체가 부재 (노출 정책)."""
        async def _rows(**kw):
            return [_row(id="m1", conversation_id=_VALID_UUID, rating="thumbs_up")]
        monkeypatch.setattr(conv_router, "list_messages_in_conversation", _rows)

        resp = await _call_list_messages(_make_request(), _VALID_UUID)
        item_keys = set(resp.items[0].model_dump().keys())
        forbidden = {
            "llm_provider", "model_name",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "llm_invoke_count", "tool_calls", "langsmith_run_id",
            "session_id", "metadata",
        }
        leaked = forbidden & item_keys
        assert not leaked, f"응답에 LLM 내부 필드 누출: {leaked}"

    @pytest.mark.asyncio
    async def test_cursor_ctx_mismatch_400(self, monkeypatch):
        """rating=negative로 발급된 cursor를 rating=positive로 사용 → 400."""
        # 라우터 ctx 빌더로 같은 값을 생성 (sort_keys=True 정합성 필요)
        ctx_neg = _build_message_ctx("negative", "desc", None, None)
        cursor = _encode_cursor(datetime(2026, 5, 14), "m1", ctx_neg)

        async def _rows(**kw):
            return []
        monkeypatch.setattr(conv_router, "list_messages_in_conversation", _rows)

        with pytest.raises(HTTPException) as exc:
            await _call_list_messages(
                _make_request(), _VALID_UUID, cursor=cursor, rating="positive"
            )
        assert exc.value.status_code == 400
        assert "cursor_context_mismatch" in exc.value.detail


# ---------------------------------------------------------------------------
# get_message
# ---------------------------------------------------------------------------


class TestGetMessage:
    """단건 메시지 상세."""

    @pytest.mark.asyncio
    async def test_invalid_conv_id_400(self):
        with pytest.raises(HTTPException) as exc:
            await get_message(_make_request(), "bad", _VALID_UUID)
        assert exc.value.status_code == 400
        assert "invalid conversation_id" in exc.value.detail

    @pytest.mark.asyncio
    async def test_invalid_message_id_400(self):
        with pytest.raises(HTTPException) as exc:
            await get_message(_make_request(), _VALID_UUID, "bad")
        assert exc.value.status_code == 400
        assert "invalid message_id" in exc.value.detail

    @pytest.mark.asyncio
    async def test_not_found_404(self, monkeypatch):
        async def _none(*a, **kw):
            return None
        monkeypatch.setattr(conv_router, "get_message_in_conversation", _none)

        with pytest.raises(HTTPException) as exc:
            await get_message(_make_request(), _VALID_UUID, _VALID_UUID)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_ok_returns_message(self, monkeypatch):
        valid_msg_id = "00000000-0000-0000-0000-000000000002"

        async def _row_fn(*a, **kw):
            return _row(id=valid_msg_id, conversation_id=_VALID_UUID, rating=None)
        monkeypatch.setattr(conv_router, "get_message_in_conversation", _row_fn)

        msg = await get_message(_make_request(), _VALID_UUID, valid_msg_id)
        assert msg.message_id == valid_msg_id
        assert msg.rating_polarity is None


# ---------------------------------------------------------------------------
# OpenAPI / 통합 라우트 노출
# ---------------------------------------------------------------------------


class TestCursorLegacyCompat:
    """ctx에 키가 **나중에 추가**됐을 때, 그 전에 발급된 cursor 를 죽이지 않는다.

    배포 직전에 "더 보기" 를 누르려던 사용자가 전부 400 을 맞는 걸 막는 장치다.
    기본값과 다른 값을 고른 요청은 여전히 걸려야 한다(그건 진짜 불일치다).
    """

    def _legacy_cursor(self, ctx: Dict[str, Any]) -> str:
        blob = json.dumps(
            {"ts": "2026-09-03T15:33:00Z", "id": "c1", "ctx": ctx},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return base64.urlsafe_b64encode(blob).rstrip(b"=").decode()

    def test_missing_key_filled_with_default(self):
        """order 가 없던 옛 cursor + 기본값(desc) 요청 → 통과."""
        stale = self._legacy_cursor({"since": None, "until": None})
        out = _decode_cursor(
            stale,
            expected_ctx=_build_conversation_ctx(None, None, "desc"),
            legacy_defaults={"order": "desc"},
        )
        assert out is not None

    def test_missing_key_still_rejected_for_non_default(self):
        """옛 cursor 를 과거순 요청에 그대로 쓰면 400 — 실제로 구간이 어긋난다."""
        stale = self._legacy_cursor({"since": None, "until": None})
        with pytest.raises(HTTPException) as exc:
            _decode_cursor(
                stale,
                expected_ctx=_build_conversation_ctx(None, None, "asc"),
                legacy_defaults={"order": "desc"},
            )
        assert exc.value.status_code == 400
        assert "cursor_context_mismatch" in exc.value.detail

    def test_other_filter_mismatch_still_rejected(self):
        """기간이 다르면 여전히 400 — 보정은 '없는 키' 에만 적용된다."""
        stale = self._legacy_cursor({"since": "2026-09-01", "until": None})
        with pytest.raises(HTTPException) as exc:
            _decode_cursor(
                stale,
                expected_ctx=_build_conversation_ctx(None, None, "desc"),
                legacy_defaults={"order": "desc"},
            )
        assert exc.value.status_code == 400


class TestParseIso:
    """``_parse_iso`` — 날짜만 온 입력을 하루로 펼치는 규칙.

    화면에서 "9월 1일 ~ 9월 4일"을 고른 사용자가 기대하는 건 KST 하루 단위다. 자정으로
    읽으면 종료일이 통째로 빠지고, UTC 로 읽으면 시작일 앞 9시간이 빠진다.
    """

    def test_date_only_since_is_kst_midnight(self):
        assert _parse_iso("2026-09-01", "since") == datetime(2026, 8, 31, 15, 0, 0)

    def test_date_only_until_covers_whole_day(self):
        """종료일 하루가 통째로 빠지면 안 된다 — KST 23:59:59.999999."""
        assert _parse_iso("2026-09-04", "until") == datetime(2026, 9, 4, 14, 59, 59, 999999)

    def test_explicit_time_is_respected(self):
        """시각까지 준 입력은 펼치지 않는다."""
        assert _parse_iso("2026-09-04T23:59:59+09:00", "until") == datetime(
            2026, 9, 4, 14, 59, 59
        )
        assert _parse_iso("2026-09-01T00:00:00Z", "since") == datetime(2026, 9, 1, 0, 0, 0)

    def test_none_passes_through(self):
        assert _parse_iso(None, "since") is None

    def test_invalid_raises_400_with_field_name(self):
        for bad in ["어제", "2026-13-01", "2026/09/01", ""]:
            with pytest.raises(HTTPException) as exc:
                _parse_iso(bad, "until")
            assert exc.value.status_code == 400
            assert "until" in exc.value.detail


class TestRouterRegistration:
    """라우터에 4개 라우트가 정확히 노출되는지."""

    def test_routes_present(self):
        paths = {r.path for r in conv_router.router.routes}
        assert "/me/conversation" in paths
        assert "/me/conversation/feedback" in paths
        assert "/me/conversation/{conversation_id}" in paths
        assert "/me/conversation/{conversation_id}/messages" in paths
        assert "/me/conversation/{conversation_id}/messages/{message_id}" in paths


# ---------------------------------------------------------------------------
# download_conversation_attachment
# ---------------------------------------------------------------------------


class TestDownloadConversationAttachment:
    """``GET /me/conversation/{id}/attachments/{file_id}/download`` 핸들러.

    PROC 가 채팅 첨부(도면)를 발주 첨부로 옮길 때 쓴다 — presigned URL 을 브라우저에서
    바로 fetch 하면 다른 오리진이라 CORS 에 막히므로 이쪽을 지난다.
    """

    FID = "file-abc"

    @staticmethod
    def _msgs(file_id="file-abc", filename="brk.pdf"):
        async def _rows(*a, **kw):
            return [
                {"attachments": [{"file_id": file_id, "filename": filename,
                                  "content_type": "application/pdf"}]}
            ]
        return _rows

    @staticmethod
    def _exists():
        async def _row(*a, **kw):
            return {"id": _VALID_UUID}
        return _row

    @pytest.mark.asyncio
    async def test_invalid_uuid_400(self):
        with pytest.raises(HTTPException) as exc:
            await conv_router.download_conversation_attachment(
                _make_request(), "bad-uuid", self.FID
            )
        assert exc.value.status_code == 400

    @pytest.mark.asyncio
    async def test_alien_conversation_404(self, monkeypatch):
        """남의 방 첨부는 404 — 403 은 그 방이 존재한다는 사실을 알려준다."""
        async def _none(*a, **kw):
            return None
        monkeypatch.setattr(conv_router, "get_conversation", _none)

        with pytest.raises(HTTPException) as exc:
            await conv_router.download_conversation_attachment(
                _make_request(), _VALID_UUID, self.FID
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_file_id_from_another_conversation_404(self, monkeypatch):
        """file_id 만으로 파일을 내주면 남의 첨부를 추측으로 긁을 수 있다."""
        monkeypatch.setattr(conv_router, "get_conversation", self._exists())
        monkeypatch.setattr(
            conv_router, "list_messages_in_conversation", self._msgs(file_id="other-file")
        )

        with pytest.raises(HTTPException) as exc:
            await conv_router.download_conversation_attachment(
                _make_request(), _VALID_UUID, self.FID
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_file_service_disabled_503(self, monkeypatch):
        """미설정은 404(없음)가 아니라 503(지금 못 준다) — 원인이 구분돼야 한다."""
        monkeypatch.setattr(conv_router, "get_conversation", self._exists())
        monkeypatch.setattr(conv_router, "list_messages_in_conversation", self._msgs())
        monkeypatch.setattr(conv_router.file_service, "enabled", lambda: False)

        with pytest.raises(HTTPException) as exc:
            await conv_router.download_conversation_attachment(
                _make_request(), _VALID_UUID, self.FID
            )
        assert exc.value.status_code == 503

    @pytest.mark.asyncio
    async def test_missing_presigned_url_404(self, monkeypatch):
        """스토리지에서 사라진 파일 — 호출부가 '직접 첨부하라' 로 안내할 수 있어야 한다."""
        monkeypatch.setattr(conv_router, "get_conversation", self._exists())
        monkeypatch.setattr(conv_router, "list_messages_in_conversation", self._msgs())
        monkeypatch.setattr(conv_router.file_service, "enabled", lambda: True)

        async def _no_urls(ids):
            return {}
        monkeypatch.setattr(conv_router.file_service, "download_urls", _no_urls)

        with pytest.raises(HTTPException) as exc:
            await conv_router.download_conversation_attachment(
                _make_request(), _VALID_UUID, self.FID
            )
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_streams_with_original_filename(self, monkeypatch):
        """한글 파일명은 RFC 5987 로 실어야 latin-1 헤더에서 깨지지 않는다."""
        monkeypatch.setattr(conv_router, "get_conversation", self._exists())
        monkeypatch.setattr(
            conv_router, "list_messages_in_conversation", self._msgs(filename="브라켓 도면.pdf")
        )
        monkeypatch.setattr(conv_router.file_service, "enabled", lambda: True)

        async def _urls(ids):
            return {self.FID: "https://files.example/obj?sig=1"}
        monkeypatch.setattr(conv_router.file_service, "download_urls", _urls)

        resp = await conv_router.download_conversation_attachment(
            _make_request(), _VALID_UUID, self.FID
        )
        assert resp.media_type == "application/pdf"
        disposition = resp.headers["content-disposition"]
        assert disposition.startswith("attachment; filename*=UTF-8''")
        # 한글이 percent-encoding 되어 있어야 한다(raw 로 나가면 헤더 인코딩에서 깨진다).
        assert "%EB%B8%8C%EB%9D%BC%EC%BC%93" in disposition
