"""``GET /me/conversation/search`` 핸들러 단위 테스트.

- 정상 검색 / 401 분기 / has_more·offset 전달
- ``_snippet`` 발췌 경계 (짧은 답변, 매칭 위치가 뒤쪽, 미매칭)
- 응답 노출 정책: LLM 내부 필드 키 절대 부재
- 라우트 선언 순서: /search 가 /{conversation_id} 보다 앞

repository 호출은 monkeypatch로 in-memory fake 함수로 대체한다.
"""
from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any, Optional

import pytest
from fastapi import HTTPException

from agent_arch.api.v1 import conversation as conv_router
from agent_arch.api.v1.conversation import _snippet, search_my_conversations

_VALID_UUID = "11111111-1111-1111-1111-111111111111"


def _make_request(sub: Optional[str] = "u-1") -> Any:
    """``request.state.user["sub"]``만 흉내내는 Request 대체. sub=None이면 401 분기."""
    state = SimpleNamespace(user={"sub": sub} if sub else None)
    return SimpleNamespace(state=state)


# 핸들러 직접 호출 시 FastAPI Query(...) 기본값 객체가 그대로 인자로 새는 문제 회피용.
# 파라미터가 늘면 여기도 반드시 함께 늘려야 한다 — 빠뜨리면 Query 객체가 _parse_iso 로
# 흘러 들어가 AttributeError 가 난다.
async def _call_search(
    req, *, q="배포", limit=20, offset=0, rating=None, since=None, until=None
):
    return await search_my_conversations(
        req, q=q, limit=limit, offset=offset, rating=rating, since=since, until=until
    )


def _row(**overrides) -> dict:
    base = {
        "id": "m1",
        "conversation_id": _VALID_UUID,
        "conversation_title": "배포 논의",
        "query": "배포 일정 언제야?",
        "answer": "다음 주 화요일입니다.",
        "created_at": datetime(2026, 5, 14, 3, 0, 0),
        "score": 0.72,
    }
    base.update(overrides)
    return base


@pytest.fixture
def search_calls(monkeypatch):
    """repository 검색 함수를 캡처용 fake로 교체. ``state["rows"]``로 반환값 제어."""
    state: dict = {"calls": [], "rows": [_row()]}

    async def _search(
        user_sub, query, *, limit, offset=0, rating_filter=None, since=None, until=None
    ):
        state["calls"].append(
            {
                "user_sub": user_sub,
                "query": query,
                "limit": limit,
                "offset": offset,
                "rating_filter": rating_filter,
                "since": since,
                "until": until,
            }
        )
        return list(state["rows"])

    monkeypatch.setattr(conv_router, "search_messages_for_user", _search)
    return state


class TestSearchHandler:
    """핸들러 정상/예외 분기."""

    @pytest.mark.asyncio
    async def test_success_maps_row_to_item(self, search_calls):
        """row가 응답 항목으로 매핑되고 repository에 user_sub/검색어가 전달된다."""
        resp = await _call_search(_make_request(), q="배포")

        assert search_calls["calls"] == [
            {
                "user_sub": "u-1",
                "query": "배포",
                "limit": 20,
                "offset": 0,
                "rating_filter": None,
                "since": None,
                "until": None,
            }
        ]
        assert len(resp.items) == 1
        item = resp.items[0]
        assert item.message_id == "m1"
        assert item.conversation_id == _VALID_UUID
        assert item.conversation_title == "배포 논의"
        assert item.query == "배포 일정 언제야?"
        assert item.answer_snippet == "다음 주 화요일입니다."
        assert item.created_at == "2026-05-14T03:00:00Z"
        assert item.score == pytest.approx(0.72)
        assert resp.has_more is False
        assert resp.total_hint is None

    @pytest.mark.asyncio
    async def test_missing_user_sub_raises_401(self, search_calls):
        """인증 컨텍스트 없으면 401 — repository 호출 없음."""
        with pytest.raises(HTTPException) as exc:
            await _call_search(_make_request(sub=None))

        assert exc.value.status_code == 401
        assert search_calls["calls"] == []

    @pytest.mark.asyncio
    async def test_has_more_when_extra_row_returned(self, search_calls):
        """repository가 limit+1건을 주면 초과분을 잘라내고 has_more=True."""
        search_calls["rows"] = [_row(id=f"m{i}") for i in range(4)]

        resp = await _call_search(_make_request(), limit=3)

        assert len(resp.items) == 3
        assert resp.has_more is True
        assert [i.message_id for i in resp.items] == ["m0", "m1", "m2"]

    @pytest.mark.asyncio
    async def test_offset_is_forwarded(self, search_calls):
        """offset이 repository로 그대로 넘어간다 (관련도 정렬이라 cursor 대신 offset)."""
        await _call_search(_make_request(), limit=10, offset=30)
        assert search_calls["calls"][0]["offset"] == 30

    @pytest.mark.asyncio
    async def test_empty_result(self, search_calls):
        """0건이면 빈 items + has_more=False."""
        search_calls["rows"] = []
        resp = await _call_search(_make_request())
        assert resp.items == []
        assert resp.has_more is False

    @pytest.mark.asyncio
    async def test_null_title_and_score_tolerated(self, search_calls):
        """제목 없는 대화방(null)과 score 누락도 응답이 깨지지 않는다."""
        search_calls["rows"] = [_row(conversation_title=None, score=None)]
        resp = await _call_search(_make_request())
        assert resp.items[0].conversation_title is None
        assert resp.items[0].score == 0.0

    @pytest.mark.asyncio
    async def test_response_dump_omits_llm_internal_fields(self, search_calls):
        """JSON 직렬화 후에도 LLM 내부 필드는 키 자체가 부재 (노출 정책)."""
        resp = await _call_search(_make_request())
        item_keys = set(resp.items[0].model_dump().keys())
        forbidden = {
            "llm_provider", "model_name",
            "prompt_tokens", "completion_tokens", "total_tokens",
            "llm_invoke_count", "tool_calls", "langsmith_run_id",
            "session_id", "metadata",
        }
        leaked = forbidden & item_keys
        assert not leaked, f"응답에 LLM 내부 필드 누출: {leaked}"


class TestSearchFilters:
    """기간·평가 필터 (민원: "날짜 조회가 되어야 한다")."""

    @pytest.mark.asyncio
    async def test_since_until_parsed_to_datetime(self, search_calls):
        """ISO8601 문자열이 naive UTC datetime으로 파싱돼 repository에 전달된다."""
        await _call_search(
            _make_request(),
            since="2026-09-01T00:00:00Z",
            until="2026-09-04T23:59:59+09:00",
        )
        call = search_calls["calls"][0]
        assert call["since"] == datetime(2026, 9, 1, 0, 0, 0)
        assert call["until"] == datetime(2026, 9, 4, 14, 59, 59)  # KST → UTC

    @pytest.mark.asyncio
    async def test_invalid_iso_raises_400(self, search_calls):
        with pytest.raises(HTTPException) as exc:
            await _call_search(_make_request(), since="어제")
        assert exc.value.status_code == 400
        assert "since" in exc.value.detail
        assert search_calls["calls"] == []

    @pytest.mark.asyncio
    async def test_rating_forwarded(self, search_calls):
        """'별로예요만' 좁히기 — rating이 repository로 그대로 넘어간다."""
        await _call_search(_make_request(), rating="negative")
        assert search_calls["calls"][0]["rating_filter"] == "negative"

    @pytest.mark.asyncio
    async def test_rating_any_normalized_to_none(self, search_calls):
        """``any``는 필터 미적용 — repository는 None을 받는다."""
        await _call_search(_make_request(), rating="any")
        assert search_calls["calls"][0]["rating_filter"] is None

    @pytest.mark.asyncio
    async def test_unknown_rating_raises_400(self, search_calls):
        with pytest.raises(HTTPException) as exc:
            await _call_search(_make_request(), rating="zzz")
        assert exc.value.status_code == 400
        assert search_calls["calls"] == []


class TestSnippet:
    """``_snippet`` 발췌 규칙."""

    def test_short_text_returned_whole(self):
        """width 이하 답변은 그대로, 말줄임표 없이."""
        assert _snippet("짧은 답변", "답변", width=200) == "짧은 답변"

    def test_empty_text(self):
        assert _snippet(None, "배포") == ""
        assert _snippet("   ", "배포") == ""

    def test_match_near_end_is_included(self):
        """매칭 부위가 뒤쪽이어도 발췌 안에 들어온다 (앞부분 자르기로는 안 보임)."""
        body = "가" * 5000 + "배포 일정 확정" + "나" * 5000
        out = _snippet(body, "배포", width=200)

        assert "배포" in out
        assert out.startswith("…") and out.endswith("…")
        assert len(out) <= 202  # width + 양쪽 말줄임표

    def test_no_match_falls_back_to_head(self):
        """리터럴 매칭 실패(형태소 차이 등) 시 앞부분으로 폴백."""
        body = "다" * 500
        out = _snippet(body, "배포", width=100)

        assert out == "다" * 100 + "…"

    def test_match_at_head_has_no_leading_ellipsis(self):
        """맨 앞이 매칭이면 앞쪽 말줄임표를 붙이지 않는다."""
        body = "배포 일정입니다. " + "라" * 500
        out = _snippet(body, "배포", width=100)

        assert not out.startswith("…")
        assert out.endswith("…")

    def test_case_insensitive_match(self):
        body = "x" * 300 + "Deploy 완료" + "y" * 300
        out = _snippet(body, "deploy", width=100)
        assert "Deploy" in out


class TestRouteOrdering:
    """/search 가 /{conversation_id} 에 삼켜지지 않는지 — 회귀 방지."""

    def test_search_declared_before_path_param(self):
        paths = [getattr(r, "path", "") for r in conv_router.router.routes]
        assert paths.index("/me/conversation/search") < paths.index(
            "/me/conversation/{conversation_id}"
        )

    def test_feedback_declared_before_path_param(self):
        """``/feedback``도 conversation_id로 삼켜지면 안 된다."""
        paths = [getattr(r, "path", "") for r in conv_router.router.routes]
        assert paths.index("/me/conversation/feedback") < paths.index(
            "/me/conversation/{conversation_id}"
        )
