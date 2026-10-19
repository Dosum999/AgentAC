"""장기 사용자 메모리 v2 — 4층 비동기 배치 파이프라인.

    L0 원문(agent.messages) → L1 atom(memory_atoms) → L2 장면(memory_scenes) → L3 페르소나(memory_persona)

핫 패스(매 턴)에서 하는 일은 둘뿐이다: (1) 회상 결과 주입, (2) 저장 후 체크포인트 이벤트 하나.
LLM 을 부르는 추출·판정·요약은 전부 스케줄러가 배치로 돈다(사용자 대기 없음).

전환 플래그: settings.memory_v2_mode = off | shadow | active (ENABLE_USER_MEMORY 가 마스터).
설계 문서: 세션 아티팩트 "에이전트 메모리 v2".
"""
from agent_arch.services.memory_v2.capture import enabled, is_active, on_turn_saved  # noqa: F401
from agent_arch.services.memory_v2.digest import build_tool_digest  # noqa: F401

__all__ = ["enabled", "is_active", "on_turn_saved", "build_tool_digest"]
