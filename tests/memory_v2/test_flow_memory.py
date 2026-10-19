"""기억이 흐름에 닿는지 — 워커 스코프에 턴 블록 이월, 종합기에 안정 블록, ContextVar 왕복."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_arch.agents import context as C
from agent_arch.agents.graphs import supervisor as S
from agent_arch.core.config import settings
from agent_arch.llm.base import Message


@pytest.fixture
def recall_ctx():
    r = SimpleNamespace(
        system_block="[장기 메모리] 안내\n● 전역 지시:\n  - 답변에 표를 쓰지 않는다",
        turn_block="[관련 기억] 참고용\n● 별칭:\n  - \"watchdog\" → mattermost_channel 'watchdog' (id: ch1)",
        aliases=[{"payload": {"canonical_kind": "mattermost_channel", "canonical_id": "ch1"}}],
    )
    C.set_memory_recall(r)
    yield r
    C.set_memory_recall(None)


def test_context_roundtrip():
    assert C.get_memory_recall() is None
    C.set_memory_recall("x")
    assert C.get_memory_recall() == "x"
    C.set_memory_recall(None)


def test_worker_scope_carries_turn_memory(recall_ctx, monkeypatch):
    monkeypatch.setattr(settings, "supervisor_preserve_original", False, raising=False)
    msgs = [Message(role="system", content="sys"), Message(role="user", content="watchdog에 어제 뭐 올라왔어\n\n[현재 시각] 2026-09-07")]
    out = S._scope_messages(msgs, "watchdog 채널 어제 메시지", "talk")
    last_user = [m for m in out if m.role == "user"][-1].content
    assert last_user.startswith("watchdog 채널 어제 메시지")
    assert "(id: ch1)" in last_user                      # 별칭의 실제 대상 id 가 워커에게 도달


def test_worker_scope_without_recall_unchanged(monkeypatch):
    monkeypatch.setattr(settings, "supervisor_preserve_original", False, raising=False)
    C.set_memory_recall(None)
    msgs = [Message(role="system", content="sys"), Message(role="user", content="q")]
    out = S._scope_messages(msgs, "sub", "talk")
    assert [m for m in out if m.role == "user"][-1].content == "sub"


def test_synth_includes_stable_block(recall_ctx):
    res = SimpleNamespace(content="메일 2건", tool_calls=[], tool_results=[])
    msgs = S._synth_messages("메일이랑 일정", [("mail", res)])
    assert "답변에 표를 쓰지 않는다" in msgs[0].content
    assert "메일 2건" in msgs[1].content
    C.set_memory_recall(None)
    assert "답변에 표를" not in S._synth_messages("q", [("mail", res)])[0].content


def test_worker_scope_carries_datetime(monkeypatch):
    monkeypatch.setattr(settings, "supervisor_preserve_original", False, raising=False)
    C.set_memory_recall(None)
    msgs = [Message(role="system", content="sys"), Message(role="user", content="watchdog에 어제 뭐 올라왔어\n\n[현재 날짜/시간]\n- 오늘: 2026-09-07 (월)")]
    out = S._scope_messages(msgs, "watchdog 채널 어제 메시지", "talk")
    last_user = [m for m in out if m.role == "user"][-1].content
    assert "[현재 날짜/시간]" in last_user and "오늘:" in last_user      # 절대 날짜표가 워커에게 도달한다
    plain = S._scope_messages([Message(role="system", content="sys"), Message(role="user", content="q")], "sub", "talk")
    assert [m for m in plain if m.role == "user"][-1].content == "sub"      # 원문에 없던 블록은 만들어 넣지 않는다


def test_alias_hint_without_id_points_to_find_channels():
    from agent_arch.services.memory_v2.loadout import alias_tool_hint
    assert "find_channels" in alias_tool_hint("mattermost_channel", None, "ACME Heros 팀의 Scrum 채널")
    assert "get_channel_messages(channel_id=ch1)" in alias_tool_hint("mattermost_channel", "ch1", "x")
    assert alias_tool_hint("business_concept", None, "CAPA") == ""
    assert alias_tool_hint(None, "id", "x") == ""
