"""스트림 반복 붕괴(runaway) 차단.

모델이 같은 조각을 max_tokens 까지 되풀이하는 일이 있다(실측: 표 구분선 `| :---` 다음에 `-` 만 13만 자).
사후 길이 상한(react loop 의 30,000자 trim)만으로는 그때까지 토큰 비용을 다 내고, 화면에도 다 흘린 뒤다.
스트림을 소비하는 자리에서 꼬리가 짧은 주기로 반복되는지 보고, 그렇다면 그 자리에서 업스트림을 닫고
``finish_reason="repetition"`` 으로 끝낸다. 판정은 결정적이다(LLM 호출 없음).
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from agent_arch.llm.base import StreamEvent

logger = logging.getLogger(__name__)

# 이만큼 흘린 뒤부터 본다 — 짧은 답의 정상 반복(구분선 한 줄 등)은 건드리지 않는다.
MIN_CHARS = 2000
# 꼬리 이만큼이 한 주기로 채워져 있으면 붕괴로 본다.
WINDOW = 1500
# 반복 단위 최대 길이. 표 구분선·이모지 반복·짧은 구절 반복을 잡는다.
MAX_PERIOD = 40
# 매 청크마다 보지 않고 이만큼 새로 쌓일 때마다 본다.
CHECK_EVERY = 256

FINISH_REASON = "repetition"


class RunawayDetector:
    """흘러간 텍스트의 꼬리가 짧은 주기로 반복되는지 판정한다."""

    def __init__(self) -> None:
        self._tail = ""
        self._total = 0
        self._since_check = 0
        self.tripped = False
        self.period = 0

    def feed(self, chunk: str) -> bool:
        if not chunk or self.tripped:
            return self.tripped
        self._total += len(chunk)
        self._since_check += len(chunk)
        self._tail = (self._tail + chunk)[-(WINDOW * 2):]
        if self._total < MIN_CHARS or self._since_check < CHECK_EVERY:
            return False
        self._since_check = 0
        window = self._tail[-WINDOW:]
        if len(window) < WINDOW:
            return False
        for p in range(1, MAX_PERIOD + 1):
            if window[p:] == window[:-p]:
                self.tripped, self.period = True, p
                break
        return self.tripped


async def stop_on_runaway(events: AsyncIterator[StreamEvent]) -> AsyncIterator[StreamEvent]:
    """LLM 스트림을 감싸 반복 붕괴가 보이면 업스트림을 닫고 ``done(repetition)`` 으로 끝낸다.

    붕괴를 알아챈 청크까지의 텍스트·도구 호출 이벤트는 그대로 흘린다. 정상 ``done`` 은 그대로 전달한다.
    """
    detector = RunawayDetector()
    async for event in events:
        if event.type == "text" and event.content and detector.feed(event.content):
            # 붕괴를 알아챈 청크까지는 흘린다(그 청크 앞부분이 정상 본문일 수 있다). 그 뒤로는 받지 않는다.
            yield event
            logger.warning("[runaway] 스트림 반복 붕괴 감지 — 주기 %d자, 업스트림 종료", detector.period)
            aclose = getattr(events, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 — 닫기 실패가 답변을 막으면 안 된다
                    logger.debug("[runaway] 업스트림 닫기 실패(무시)", exc_info=True)
            yield StreamEvent(type="done", finish_reason=FINISH_REASON)
            return
        yield event
