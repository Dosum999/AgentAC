"""L3 페르소나 — 냉시동 억제, (없음) 처리, fresh 삭제, 저장 경로(fake repo/LLM)."""
from __future__ import annotations

import pytest

from agent_arch.services.memory_v2 import persona as P


def test_clean_persona():
    assert P.clean_persona("```markdown\n## 에이전트\n- x\n```") == "## 에이전트\n- x"
    assert "→" in P.clean_persona("- 클레임 $\\rightarrow$ CAPA") and "$" not in P.clean_persona("- 클레임 $\\rightarrow$ CAPA")
    assert P.clean_persona("(없음)") is None
    assert P.clean_persona("   ") is None
    assert len(P.clean_persona("가" * 5000)) == P.MAX_PERSONA_CHARS


def test_format_persona_input_sections():
    text = P.format_persona_input("기존", [{"title": "메일 루틴", "heat": 2, "content": "본문\n줄"}],
                                  [{"atom_type": "instruction", "priority": -1, "content": "표 금지"}])
    assert "[기존 페르소나]\n기존" in text
    assert "- 메일 루틴 (heat 2.0): 본문 줄" in text
    assert "[atom 1개]\n- [instruction|p-1] 표 금지" in text
    assert "[기존 페르소나]\n(없음)" in P.format_persona_input(None, [], [])


class _Repo:
    def __init__(self, atoms, existing=None, scenes=None):
        self.atoms, self.existing, self.scenes = atoms, existing, scenes or []
        self.calls: list[tuple] = []

    async def list_active_atoms(self, user_sub, *, atom_type=None, limit=50):
        return self.atoms

    async def get_active_persona(self, user_sub, company_scope=None):
        return self.existing

    async def list_scenes(self, user_sub, *, limit=8, with_content=False):
        return self.scenes

    async def delete_persona(self, user_sub, company_scope=None):
        self.calls.append(("delete",))
        return True

    async def upsert_persona(self, user_sub, content, *, source_atom_count, company_scope=None):
        self.calls.append(("upsert", content, source_atom_count))
        return "pid"

    async def mark_l3_done(self, user_sub):
        self.calls.append(("l3_done",))

    async def insert_generation_log(self, **kw):
        self.calls.append(("log", kw["status"]))
        return "log"


@pytest.fixture
def wire(monkeypatch):
    def _wire(repo: _Repo, raw):
        for name in ("list_active_atoms", "get_active_persona", "list_scenes", "delete_persona", "upsert_persona", "mark_l3_done", "insert_generation_log"):
            monkeypatch.setattr(P.repo, name, getattr(repo, name))

        async def fake_invoke(messages):
            if isinstance(raw, Exception):
                raise raw
            _wire.last_input = messages[1].content
            return raw, "m", 2

        monkeypatch.setattr(P, "invoke_text", fake_invoke)
    return _wire


def _atoms(n):
    return [{"id": f"a{i}", "atom_type": "persona", "priority": 50, "content": f"c{i}"} for i in range(n)]


@pytest.mark.asyncio
async def test_too_few_atoms_skips_and_fresh_deletes_existing(wire):
    repo = _Repo(_atoms(3), existing={"content": "old"})
    wire(repo, "## x")
    assert await P.rebuild_persona("u1") is None and repo.calls == []
    assert await P.rebuild_persona("u1", fresh=True) is None and repo.calls == [("delete",)]


@pytest.mark.asyncio
async def test_rebuild_saves_and_marks_done(wire):
    repo = _Repo(_atoms(6), existing={"content": "old persona"}, scenes=[{"title": "메일", "heat": 1, "content": "c"}])
    wire(repo, "```\n## 사용자\n- 담당: MES\n```")
    out = await P.rebuild_persona("u1")
    assert out == "## 사용자\n- 담당: MES"
    assert ("upsert", out, 6) in repo.calls and ("l3_done",) in repo.calls and ("log", "succeeded") in repo.calls
    assert "[기존 페르소나]\nold persona" in wire.last_input       # 증분 모드는 기존 본을 준다


@pytest.mark.asyncio
async def test_fresh_hides_existing_from_prompt(wire):
    repo = _Repo(_atoms(6), existing={"content": "old persona"})
    wire(repo, "## 새로")
    await P.rebuild_persona("u1", fresh=True)
    assert "[기존 페르소나]\n(없음)" in wire.last_input


@pytest.mark.asyncio
async def test_empty_marker_deletes_when_fresh(wire):
    repo = _Repo(_atoms(6), existing={"content": "old"})
    wire(repo, "(없음)")
    assert await P.rebuild_persona("u1", fresh=True) is None
    assert ("delete",) in repo.calls and ("l3_done",) in repo.calls and ("log", "empty") in repo.calls
    assert not any(c[0] == "upsert" for c in repo.calls)


@pytest.mark.asyncio
async def test_llm_failure_logs_failed(wire):
    repo = _Repo(_atoms(6))
    wire(repo, RuntimeError("down"))
    assert await P.rebuild_persona("u1") is None
    assert repo.calls == [("log", "failed")]
