"""승인 액션 완료 **후속 제안** — 완료 문구(policy.ACTION_DONE_MESSAGES)의 확장.

/resume 가 승인 실행을 마치면, 실행된 도구 이름으로 등록된 빌더를 찾아 완료 문구에
접미를 덧붙인다(응답과 히스토리 영속 **양쪽** — 화면에만 보이는 문구는 다음 턴 LLM 에게
없는 문구다). "어떤 액션 뒤에 무엇을 제안하나"라는 도메인 지식은 확장 슬롯
(``extensions.followup_builders``)이 소유하고, 전송 계층은 ``followup_note`` generic 호출
한 줄만 갖는다.

방식 = **문구 유도**: 접미에 다음 액션에 필요한 값(예: 상신한 영수증의 품명·금액)을 박아
히스토리에 영속하고, 사용자가 다음 액션을 말하면 다음 턴의 워커가 그 문장에서 값을
**옮기기만** 한다(산수·매핑은 전부 빌더가 끝낸다). 초안 카드 자동 연쇄(다음 승인 카드를
체크포인트에 바로 적재)도 설계·구현·검증까지 갔으나 구조 변경 폭(HITL 그래프 메타·커밋
화이트리스트·resume 재배치)이 과하다고 판단해 채택하지 않았다.

빌더 계약: ``async (pending, effective_args, user_info, conversation_id) -> str``.
빈 문자열 = 제안 없음. 제안은 부가 기능 — 예외는 followup_note 가 전량 흡수한다
(판정 실패가 승인 완료 응답을 깨면 안 된다).
"""
from __future__ import annotations

import logging

from agent_arch.extensions import extensions

logger = logging.getLogger(__name__)


async def followup_note(
    pending: dict, effective_args: dict, user_info: dict, conversation_id: str
) -> str:
    """승인 완료 문구에 덧붙일 후속 제안 접미. 해당 없음/실패 = 빈 문자열."""
    builder = extensions.followup_builders.get(str(pending.get("tool_name") or ""))
    if builder is None:
        return ""
    try:
        return await builder(pending, effective_args, user_info, conversation_id)
    except Exception:
        logger.exception("[hitl] 후속 제안 조립 실패 — 안내 생략 (tool=%s)", pending.get("tool_name"))
        return ""
