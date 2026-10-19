"""스트림 반복 붕괴 차단 — 꼬리가 짧은 주기로 반복되면 업스트림을 닫고 done(repetition)."""
import pytest

from agent_arch.llm.base import StreamEvent
from agent_arch.llm.runaway import FINISH_REASON, MIN_CHARS, RunawayDetector, stop_on_runaway


def test_dash_runaway_trips():
    d = RunawayDetector()
    d.feed("| 항목 | 값 |\n| :")
    tripped = False
    for _ in range(200):
        tripped = d.feed("-" * 16) or tripped
    assert tripped and d.period == 1


def test_multi_char_period_trips():
    d = RunawayDetector()
    assert any(d.feed("ㅋㅋ좋아요! ") for _ in range(600))
    assert 1 < d.period <= 40


def test_long_normal_text_does_not_trip():
    d = RunawayDetector()
    text = "".join(f"{i}번째 문단입니다. 내용은 매번 조금씩 다르고 숫자 {i * 7} 도 섞입니다.\n" for i in range(400))
    assert not any(d.feed(text[i:i + 50]) for i in range(0, len(text), 50))


def test_short_repetition_below_min_is_ignored():
    d = RunawayDetector()
    assert not d.feed("-" * (MIN_CHARS - 1))


async def _stream(chunks, closed):
    try:
        for c in chunks:
            yield StreamEvent(type="text", content=c)
        yield StreamEvent(type="done", finish_reason="length")
    finally:
        closed.append(True)


@pytest.mark.asyncio
async def test_wrapper_stops_and_closes_upstream():
    closed = []
    chunks = ["표입니다\n| :"] + ["-" * 16] * 10_000
    out = [ev async for ev in stop_on_runaway(_stream(chunks, closed))]
    assert out[-1].type == "done" and out[-1].finish_reason == FINISH_REASON
    assert sum(len(ev.content or "") for ev in out if ev.type == "text") < 5000
    assert closed == [True]


@pytest.mark.asyncio
async def test_wrapper_passes_normal_stream_through():
    closed = []
    out = [ev async for ev in stop_on_runaway(_stream(["안녕하세요", " 반갑습니다"], closed))]
    assert [e.type for e in out] == ["text", "text", "done"]
    assert out[-1].finish_reason == "length"
