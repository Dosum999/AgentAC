"""후속 지시(그거/방금 것) 처리 규칙이 프롬프트에 존재하는지 잠금 검증한다.

멀티턴 실패(직전에 보여준 대상을 가리키는 후속 지시에 봇이 식별자를 되묻거나
"기능 없음"으로 회피)를 막기 위해, 일반(PRODUCTION)·물류(LOGISTICS) 프롬프트 양쪽에
"방금 보여준 번호를 되묻지 말고 바로 실행 / 없으면 find_*_tools 로 찾아라" 규칙을 넣었다.
이 테스트는 그 규칙이 실수로 삭제되지 않도록 문구 존재를 고정한다(행동 정확도는
tests/evals/tool_selection 하네스가 별도 측정).
"""
from __future__ import annotations

from agent_arch.agents.prompts.system import build_system_prompt


def _prompt(mode: str) -> str:
    return build_system_prompt({"sub": "u1", "name": "테스트"}, mode=mode)


def test_production_prompt_has_followup_rule():
    """일반 대화(메일·메신저 등) 프롬프트에 후속 지시 규칙이 있다."""
    p = _prompt("agent")
    assert "다시 되묻지 마" in p  # 방금 보여준 번호/ID 를 되묻지 말 것
    assert "find_*_tools" in p    # 도구 없으면 되묻기 전에 탐색


