"""agent_arch.agents.graphs.checkpointer 단위 테스트 — checkpointer 팩토리.

단일 주입 지점이 백엔드 선택을 올바르게 분기하는지 검증한다:
    (a) memory / 기본 설정 → MemorySaver 인스턴스
    (b) 알 수 없는 백엔드 → ValueError
    (c) postgres → dep 미설치/미배선이면 RuntimeError (오연결을 만들지 않음)
    (d) HITL 그래프가 팩토리로 여전히 컴파일된다 (회귀 방지)
"""
import importlib.util

import pytest

from agent_arch.agents.graphs.checkpointer import get_checkpointer


def test_memory_backend_returns_memory_saver():
    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(get_checkpointer("memory"), MemorySaver)


def test_default_setting_is_memory(monkeypatch):
    from agent_arch.agents.graphs import checkpointer as mod
    from langgraph.checkpoint.memory import MemorySaver

    monkeypatch.setattr(mod.settings, "langgraph_checkpointer", "memory", raising=False)
    assert isinstance(get_checkpointer(), MemorySaver)


def test_name_override_is_case_insensitive():
    from langgraph.checkpoint.memory import MemorySaver

    assert isinstance(get_checkpointer("  MEMORY "), MemorySaver)


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError):
        get_checkpointer("bogus")


def test_postgres_backend_raises_until_wired():
    # langgraph-checkpoint-postgres 가 설치돼 있으면 '미배선' RuntimeError,
    # 없으면 'dep 미설치' RuntimeError — 어느 쪽이든 오연결 대신 RuntimeError.
    with pytest.raises(RuntimeError):
        get_checkpointer("postgres")


def test_hitl_graph_still_compiles_via_factory():
    from agent_arch.agents.hitl.graph import get_action_graph

    graph = get_action_graph()
    assert graph is not None
    # 동일 프로세스 싱글턴 — 재호출 시 같은 인스턴스.
    assert get_action_graph() is graph
