"""memory_v2 테스트 공통 — 상위층 예약(schedule_upper_layers)을 기본으로 무력화하고 호출만 기록한다.

pipeline/explicit 이 함수 안에서 layers.schedule_upper_layers 를 import 하므로 모듈 속성을 바꾸면
전부 잡힌다. 진짜 예약 동작은 test_layers 가 ORIGINAL_SCHEDULE 로 직접 검증한다.
"""
from __future__ import annotations

import pytest

from agent_arch.services.memory_v2 import layers

ORIGINAL_SCHEDULE = layers.schedule_upper_layers


@pytest.fixture(autouse=True)
def upper_layers_spy(monkeypatch):
    calls: list[dict] = []

    def fake_schedule(user_sub, *, delay=layers.DELAY_AFTER_L1_SECONDS, **kwargs):
        calls.append({"user_sub": user_sub, "delay": delay, **kwargs})
        return None

    monkeypatch.setattr(layers, "schedule_upper_layers", fake_schedule)
    return calls


@pytest.fixture(autouse=True)
def stable_cache_spy(monkeypatch):
    """안정 블록 캐시 무효화 호출을 기록만 한다(Redis 없음). 캐시 동작 자체는 test_recall 이 fake redis 로 검증."""
    from agent_arch.services.memory_v2 import recall

    calls: list[str] = []

    async def fake_invalidate(user_sub):
        calls.append(user_sub)

    monkeypatch.setattr(recall, "invalidate_stable_block", fake_invalidate)
    return calls
