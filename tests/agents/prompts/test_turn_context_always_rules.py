"""항상 지킬 사용자 규칙은 턴 컨텍스트 맨 끝의 짧은 블록으로 — 안정 블록 속에 묻혀 절반쯤 놓치던 실측(로컬 2026-09-29)."""
from agent_arch.agents.prompts.system import build_turn_context


def test_always_rules_block_is_last_and_only_when_present():
    assert "[항상 지킬 사용자 규칙" not in build_turn_context(None)
    out = build_turn_context(None, glossary_block="[용어] a", always_rules=["답변 끝에 서명을 붙인다", " "])
    assert out.rstrip().endswith("- 답변 끝에 서명을 붙인다") and out.index("[용어]") < out.index("[항상 지킬 사용자 규칙")
