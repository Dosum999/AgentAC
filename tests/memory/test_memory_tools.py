"""장기 메모리 — 에이전트 도구 인자 견고성 (memory_remember).

LLM(특히 gemini)이 선언된 'content' 대신 'text'/'value' 로 호출해도 명시 저장이
조용히 실패하지 않아야 한다(라이브에서 gemini가 text= 로 호출 → content 비어 저장
실패 관측). 컨텍스트·저장 서비스는 mock 한다.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agent_arch.agents.tools.skills.memory import skills


@pytest.fixture
def logged_in(monkeypatch):
    """로그인 + 메모리(v2 active) 컨텍스트, save_explicit_atom 은 mock."""
    monkeypatch.setattr(skills.settings, "enable_user_memory", True, raising=False)
    monkeypatch.setattr(skills.settings, "memory_v2_mode", "active", raising=False)
    monkeypatch.setattr(skills, "get_user_context", lambda: {"sub": "u-1", "company": None})
    save = AsyncMock(return_value={"id": "a-1", "content": "x", "atom_type": "instruction", "action": "insert"})
    monkeypatch.setattr("agent_arch.services.memory_v2.explicit.save_explicit_atom", save)
    return save


class TestMemoryRememberArgTolerance:

    @pytest.mark.asyncio
    async def test_accepts_content(self, logged_in):
        res = await skills.MemoryRememberTool()._real_execute(content="답변에 표를 써")
        assert res.output.get("saved") is True
        assert logged_in.await_args.args[1] == "답변에 표를 써"

    @pytest.mark.asyncio
    async def test_accepts_text_alias(self, logged_in):
        # gemini 가 text= 로 호출해도 저장돼야 한다(라이브 관측 버그).
        res = await skills.MemoryRememberTool()._real_execute(text="답변에 표를 써")
        assert res.output.get("saved") is True
        assert logged_in.await_args.args[1] == "답변에 표를 써"

    @pytest.mark.asyncio
    async def test_accepts_value_alias(self, logged_in):
        res = await skills.MemoryRememberTool()._real_execute(value="3줄로 요약해")
        assert res.output.get("saved") is True

    @pytest.mark.asyncio
    async def test_blank_still_errors(self, logged_in):
        res = await skills.MemoryRememberTool()._real_execute(content="   ")
        assert res.success is False
        logged_in.assert_not_awaited()
