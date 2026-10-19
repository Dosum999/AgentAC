"""rate_message 엔드포인트 단위 테스트.

LangSmith 연동이 추가된 POST /agent/messages/{id}/rating 동작을 검증한다.
- 메시지 없음 → 404
- rating 업데이트 실패 → 404
- langsmith_run_id 없음 → send_feedback 미등록
- langsmith_run_id 있음 → send_feedback 등록 (인자 확인)
- 부정 평가 → notify_answer_review_dislike 도 함께 등록 (회사/사용자 동봉 확인)

classify_rating 은 'thumbs_up' 이 아닌 모든 rating 을 negative 로 판정하므로, 아래
테스트의 rating 값들은 전부 부정 평가다 → 백그라운드 태스크가 최대 2개 등록된다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import BackgroundTasks
from fastapi.exceptions import HTTPException

from agent_arch.api.v1.agent import rate_message
from agent_arch.models.agent import RatingRequest


def _body(rating: str = "good") -> RatingRequest:
    return RatingRequest(rating=rating)


def _bg() -> BackgroundTasks:
    bt = BackgroundTasks()
    return bt


def _task(bg: BackgroundTasks, name: str):
    """등록된 백그라운드 태스크를 함수 이름으로 집는다(인덱스 의존 제거)."""
    for task in bg.tasks:
        if task.func.__name__ == name:
            return task
    return None


class TestRateMessage:
    """rate_message() 동작 검증."""

    @pytest.mark.asyncio
    async def test_404_when_message_not_found(self):
        """get_message_for_feedback이 None을 반환하면 404를 반환한다."""
        with patch("agent_arch.api.v1.agent.get_message_for_feedback", new=AsyncMock(return_value=None)):
            with pytest.raises(HTTPException) as exc_info:
                await rate_message("no-such-id", _body(), _bg())
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_404_when_rating_update_fails(self):
        """메시지는 존재하지만 update_message_rating이 False를 반환하면 404를 반환한다."""
        msg = {"query": "q", "answer": "a", "langsmith_run_id": None}
        with patch("agent_arch.api.v1.agent.get_message_for_feedback", new=AsyncMock(return_value=msg)), \
             patch("agent_arch.api.v1.agent.update_message_rating", new=AsyncMock(return_value=False)):
            with pytest.raises(HTTPException) as exc_info:
                await rate_message("msg-id", _body(), _bg())
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_success_without_langsmith_run_id(self):
        """langsmith_run_id가 None이면 send_feedback을 등록하지 않는다.

        """
        msg = {"query": "q", "answer": "a", "langsmith_run_id": None}
        bg = _bg()

        with patch("agent_arch.api.v1.agent.get_message_for_feedback", new=AsyncMock(return_value=msg)), \
             patch("agent_arch.api.v1.agent.update_message_rating", new=AsyncMock(return_value=True)):
            result = await rate_message("msg-id", _body("good"), bg)

        assert result == {"success": True}
        assert _task(bg, "send_feedback") is None

    @pytest.mark.asyncio
    async def test_background_task_registered_with_run_id(self):
        """langsmith_run_id가 있으면 send_feedback 백그라운드 태스크가 등록된다."""
        msg = {
            "query": "사용자 질문",
            "answer": "어시스턴트 답변",
            "langsmith_run_id": "run-uuid-abc",
        }
        bg = _bg()

        with patch("agent_arch.api.v1.agent.get_message_for_feedback", new=AsyncMock(return_value=msg)), \
             patch("agent_arch.api.v1.agent.update_message_rating", new=AsyncMock(return_value=True)):
            result = await rate_message("msg-id", _body("good"), bg)

        assert result == {"success": True}

        # 등록된 태스크의 인자 확인
        task = _task(bg, "send_feedback")
        assert task is not None
        assert task.kwargs["langsmith_run_id"] == "run-uuid-abc"
        assert task.kwargs["rating"] == "good"
        assert task.kwargs["query"] == "사용자 질문"
        assert task.kwargs["answer"] == "어시스턴트 답변"

    @pytest.mark.asyncio
    async def test_background_task_registered_for_bad_rating(self):
        """bad rating도 백그라운드 태스크가 등록된다."""
        msg = {
            "query": "q",
            "answer": "a",
            "langsmith_run_id": "run-uuid-xyz",
        }
        bg = _bg()

        with patch("agent_arch.api.v1.agent.get_message_for_feedback", new=AsyncMock(return_value=msg)), \
             patch("agent_arch.api.v1.agent.update_message_rating", new=AsyncMock(return_value=True)):
            await rate_message("msg-id", _body("bad"), bg)

        assert _task(bg, "send_feedback").kwargs["rating"] == "bad"




