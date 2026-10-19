"""``GET /me/conversation/feedback`` 핸들러 단위 테스트.

민원 대응 화면 — "자신이 피드백 준 것에 대하여 볼 수 있어야 한다". 대화방을 가로질러
내가 평가한 turn만 모아 본다.

- 매핑 / 401 / rating 화이트리스트 / cursor ctx / 라우트 선언 순서
- 응답 노출 정책: LLM 내부 필드 키 절대 부재

repository 호출은 monkeypatch로 in-memory fake 함수로 대체한다.
"""
from __future__ import annotations

import base64
import json
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from fastapi import HTTPException

from agent_arch.api.v1 import conversation as conv_router
from agent_arch.api.v1.conversation import _build_feedback_ctx, list_my_feedback

_CONV_UUID = "11111111-1111-1111-1111-111111111111"


def _make_request(sub: Optional[str] = "u-1") -> Any:
    state = SimpleNamespace(user={"sub": sub} if sub else None)
    return SimpleNamespace(state=state)


# Query(...) 기본값 객체가 인자로 새지 않도록 모든 파라미터를 명시 전달한다.
async def _call_feedback(
    req, *, limit=20, cursor=None, rating="rated", since=None, until=None, order="desc"
):
    return await list_my_feedback(
        req, limit=limit, cursor=cursor, rating=rating, since=since, until=until, order=order
    )


def _row(**overrides) -> dict:
    base = {
        "id": "m1",
        "conversation_id": _CONV_UUID,
        "conversation_title": "휴가 문의",
        "query": "연차 며칠 남았어?",
        "answer": "3일 남았습니다.",
        "rating": "답변이 질문과 맞지 않음",
        "created_at": datetime(2026, 9, 3, 15, 33, 0),
    }
    base.update(overrides)
    return base


@pytest.fixture
def feedback_calls(monkeypatch):
    """repository 함수를 캡처용 fake로 교체. ``state["rows"]``로 반환값 제어."""
    state: dict = {"calls": [], "rows": [_row()]}

    async def _list(user_sub, *, limit, cursor=None, rating_filter="rated",
                    since=None, until=None, order="desc"):
        state["calls"].append(
            {
                "user_sub": user_sub,
                "limit": limit,
                "cursor": cursor,
                "rating_filter": rating_filter,
                "since": since,
                "until": until,
                "order": order,
            }
        )
        return list(state["rows"])

    monkeypatch.setattr(conv_router, "list_rated_messages_for_user", _list)
    return state


class TestFeedbackHandler:
    @pytest.mark.asyncio
    async def test_success_maps_row_to_item(self, feedback_calls):
        """'별로예요' 자유 텍스트가 사유 그대로 + negative/freeform 으로 내려간다."""
        resp = await _call_feedback(_make_request())

        assert feedback_calls["calls"][0]["user_sub"] == "u-1"
        item = resp.items[0]
        assert item.message_id == "m1"
        assert item.conversation_id == _CONV_UUID
        assert item.conversation_title == "휴가 문의"
        assert item.query == "연차 며칠 남았어?"
        assert item.answer_snippet == "3일 남았습니다."
        assert item.rating == "답변이 질문과 맞지 않음"
        assert item.rating_polarity == "negative"
        assert item.rating_kind == "freeform"
        assert item.created_at == "2026-09-03T15:33:00Z"
        assert resp.has_more is False
        assert resp.next_cursor is None

    @pytest.mark.asyncio
    async def test_thumbs_up_is_positive_preset(self, feedback_calls):
        feedback_calls["rows"] = [_row(rating="thumbs_up")]
        resp = await _call_feedback(_make_request(), rating="positive")
        assert resp.items[0].rating_polarity == "positive"
        assert resp.items[0].rating_kind == "preset"

    @pytest.mark.asyncio
    async def test_missing_user_sub_raises_401(self, feedback_calls):
        with pytest.raises(HTTPException) as exc:
            await _call_feedback(_make_request(sub=None))
        assert exc.value.status_code == 401
        assert feedback_calls["calls"] == []

    @pytest.mark.asyncio
    async def test_default_rating_is_rated(self, feedback_calls):
        await _call_feedback(_make_request())
        assert feedback_calls["calls"][0]["rating_filter"] == "rated"

    @pytest.mark.parametrize("bad", ["unrated", "any", "zzz", ""])
    @pytest.mark.asyncio
    async def test_rejected_rating_values(self, feedback_calls, bad):
        """미평가/전체는 이 화면에서 받지 않는다 — 전 대화 덤프가 된다."""
        with pytest.raises(HTTPException) as exc:
            await _call_feedback(_make_request(), rating=bad)
        assert exc.value.status_code == 400
        assert feedback_calls["calls"] == []

    @pytest.mark.asyncio
    async def test_since_until_parsed(self, feedback_calls):
        await _call_feedback(
            _make_request(), since="2026-09-01T00:00:00Z", until="2026-09-04T00:00:00Z"
        )
        call = feedback_calls["calls"][0]
        assert call["since"] == datetime(2026, 9, 1)
        assert call["until"] == datetime(2026, 9, 4)

    @pytest.mark.asyncio
    async def test_invalid_iso_raises_400(self, feedback_calls):
        with pytest.raises(HTTPException) as exc:
            await _call_feedback(_make_request(), until="어제")
        assert exc.value.status_code == 400
        assert feedback_calls["calls"] == []

    @pytest.mark.asyncio
    async def test_has_more_and_next_cursor(self, feedback_calls):
        feedback_calls["rows"] = [_row(id=f"m{i}") for i in range(4)]

        resp = await _call_feedback(_make_request(), limit=3)

        assert len(resp.items) == 3
        assert resp.has_more is True
        raw = resp.next_cursor + "=" * (-len(resp.next_cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(raw))
        assert payload["id"] == "m2"
        assert payload["ctx"] == _build_feedback_ctx("rated", None, None)

    @pytest.mark.asyncio
    async def test_order_forwarded_and_bound_into_cursor(self, feedback_calls):
        """과거순 선택이 repository 로 넘어가고 cursor ctx 에도 묶인다."""
        feedback_calls["rows"] = [_row(id=f"m{i}") for i in range(4)]

        resp = await _call_feedback(_make_request(), limit=3, order="asc")

        assert feedback_calls["calls"][0]["order"] == "asc"
        raw = resp.next_cursor + "=" * (-len(resp.next_cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(raw))
        assert payload["ctx"]["order"] == "asc"

    @pytest.mark.asyncio
    async def test_cursor_context_mismatch_raises_400(self, feedback_calls):
        """필터를 바꾼 채 cursor를 다시 보내면 400 — 페이지네이션 일관성 보호."""
        blob = json.dumps(
            {
                "ts": "2026-09-03T15:33:00Z",
                "id": "m1",
                "ctx": _build_feedback_ctx("negative", None, None),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        stale = base64.urlsafe_b64encode(blob).rstrip(b"=").decode()

        with pytest.raises(HTTPException) as exc:
            await _call_feedback(_make_request(), rating="rated", cursor=stale)
        assert exc.value.status_code == 400
        assert "cursor_context_mismatch" in exc.value.detail

    @pytest.mark.asyncio
    async def test_empty_result(self, feedback_calls):
        feedback_calls["rows"] = []
        resp = await _call_feedback(_make_request())
        assert resp.items == []
        assert resp.has_more is False

    @pytest.mark.asyncio
    async def test_long_answer_is_truncated(self, feedback_calls):
        """전문이 아니라 발췌 — 목록 화면이고 전문은 대화방으로 점프해 본다."""
        feedback_calls["rows"] = [_row(answer="가" * 5000)]
        resp = await _call_feedback(_make_request())
        assert len(resp.items[0].answer_snippet) < 300
        assert resp.items[0].answer_snippet.endswith("…")

    @pytest.mark.asyncio
    async def test_response_dump_omits_llm_internal_fields(self, feedback_calls):
        resp = await _call_feedback(_make_request())
        item_keys = set(resp.items[0].model_dump().keys())
        forbidden = {
            "llm_provider", "model_name",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "llm_invoke_count", "tool_calls", "langsmith_run_id",
            "session_id", "metadata",
        }
        leaked = forbidden & item_keys
        assert not leaked, f"응답에 LLM 내부 필드 누출: {leaked}"


class TestRouteOrdering:
    def test_feedback_declared_before_path_param(self):
        paths = [getattr(r, "path", "") for r in conv_router.router.routes]
        assert paths.index("/me/conversation/feedback") < paths.index(
            "/me/conversation/{conversation_id}"
        )
