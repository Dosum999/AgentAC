"""첨부 본문이 입력 앞에 붙은 턴 — 모델 원문은 그대로, 사용자가 직접 쓴 질문은 따로 싣고 소비처가 그것을 쓴다.

전에는 첨부 평문이 질문 앞에 붙은 채 저장돼, 라우터·압축·장기 메모리 추출이 표·문서 본문을 사용자 발화로 봤다.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_arch.agents.context import set_turn_question
from agent_arch.agents.react.history import recent_history, recent_user_turns
from agent_arch.llm.base import Message
from agent_arch.memory import conversation as conv
from agent_arch.services.memory_v2.prompts import format_turn

ATTACH = "[첨부: 거래처.csv]\n거래처,금액\nA,100\nB,200\n\n"
QUESTION = "거래처별 합계 알려줘"
INPUT = ATTACH + QUESTION


@pytest.fixture
def mocks(monkeypatch):
    pg = AsyncMock(return_value=("mid-1", "conv-1"))
    redis = AsyncMock()
    capture = MagicMock()
    monkeypatch.setattr(conv, "pg_save_turn", pg)
    monkeypatch.setattr(conv, "append_turn", redis)
    monkeypatch.setattr("agent_arch.services.memory_v2.on_turn_saved", capture)
    yield pg, redis, capture
    set_turn_question(None)


@pytest.mark.asyncio
async def test_question_saved_alongside_raw_input(mocks):
    pg, redis, capture = mocks
    set_turn_question(QUESTION)
    await conv.save_logged_in_turn("conv-1", "u1", INPUT, "합계는 300 입니다")

    kw = pg.await_args.kwargs
    assert kw["user_input"] == INPUT                       # 모델 원문은 그대로
    assert kw["metadata"]["question"] == QUESTION
    assert redis.await_args.kwargs["question"] == QUESTION
    assert capture.call_args.args[2] == QUESTION            # 장기 메모리는 질문만 본다


@pytest.mark.asyncio
async def test_no_question_when_input_is_the_question(mocks):
    pg, redis, _ = mocks
    set_turn_question(QUESTION)
    await conv.save_logged_in_turn("conv-1", "u1", QUESTION, "답")
    assert "question" not in (pg.await_args.kwargs["metadata"] or {})
    assert redis.await_args.kwargs["question"] is None


def test_router_history_sees_question_not_attachment():
    msgs = [Message(role="user", content=INPUT, question=QUESTION),
            Message(role="assistant", content="합계는 300 입니다")]
    assert "거래처,금액" not in recent_history(msgs)
    assert recent_user_turns(msgs) == [QUESTION]


def test_memory_extractor_input_uses_question():
    row = {"conversation_id": "c", "query": INPUT, "answer": "300", "metadata": {"question": QUESTION}}
    text = format_turn("t1", row)
    assert f"사용자: {QUESTION}" in text and "A,100" not in text
