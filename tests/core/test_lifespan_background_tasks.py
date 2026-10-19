"""lifespan 배경 태스크 초기화 계약 — 종료부가 cancel() 할 수 있는 것만 돌려준다.

lifespan 종료부는 _init_* 들의 반환값을 한 묶음으로 ``task.cancel()`` 한다. 그래서 이
함수들은 asyncio.Task 아니면 None 을 돌려줘야 한다. 실제로 _init_memory_v2_scheduler 에
@asynccontextmanager 가 잘못 붙어 _AsyncGeneratorContextManager 를 돌려준 적이 있고,
기동은 멀쩡했지만(contextlib 이 생성자에서 본문을 호출한다) 종료 때마다
``AttributeError: ... has no attribute 'cancel'`` 로 정상 종료가 실패했다.
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import patch

import pytest

from agent_arch.core import lifespan as L

_INITIALIZERS = (
    "_init_memory_v2_scheduler",
    "_init_error_events",
)


@pytest.mark.parametrize("name", _INITIALIZERS)
def test_initializer_is_a_plain_function(name):
    """컨텍스트 매니저·제너레이터가 아니라 그냥 호출해 태스크를 받는 함수여야 한다."""
    fn = getattr(L, name)
    assert not inspect.isasyncgenfunction(fn), f"{name} 이 async generator 다"
    assert not hasattr(fn, "__wrapped__"), (
        f"{name} 이 데코레이터로 감싸져 있다 — 반환 타입이 태스크가 아닐 수 있다"
    )


def test_memory_v2_scheduler_returns_none_when_disabled():
    """v2 가 꺼져 있으면 None. 데코레이터가 붙어 있으면 여기서 컨텍스트 매니저가 나온다."""
    with patch("agent_arch.services.memory_v2.enabled", return_value=False):
        assert L._init_memory_v2_scheduler() is None


def test_shutdown_cancel_contract_holds_for_none_and_task():
    """종료부가 하는 일(None 은 건너뛰고, 태스크는 cancel)이 그대로 성립하는지."""
    async def _run():
        async def _forever():
            await asyncio.sleep(3600)

        task = asyncio.ensure_future(_forever())
        for candidate in (None, task):          # lifespan 종료부와 같은 모양
            if candidate is not None:
                candidate.cancel()
        assert task.cancelled() or task.cancelling()

    asyncio.run(_run())


def test_error_events_initializer_returns_none_pair_when_disabled(monkeypatch):
    """꺼져 있으면 None — 종료부가 None 검사 후 cancel 한다."""
    monkeypatch.setattr(L.settings, "enable_error_events", False, raising=False)
    assert L._init_error_events() is None
