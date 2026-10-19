"""HITL 승인 정책 — 어떤 도구가 사람 승인을 강제로 요구하는지 코드로 선언한다.

승인 필요 여부를 시스템 프롬프트/도구 설명이 아니라 이 레지스트리에서 결정한다. 승인 시
``confirm=True`` 로 실제 실행되는 비가역 액션의 이름 집합이다. 코어는 일반 협업 도메인
(일정·메일·메신저·인사)의 쓰기 도구만 안다. 업무 도메인의 쓰기 도구는 확장 슬롯
(``extensions.approval_required`` / ``extensions.action_done_messages``)으로 더한다.

워크스페이스 커밋 엔드포인트 같은 **승인 우회 폴백을 두지 않는 도구**가 있다면 /resume
(체크포인트 검증) 전용으로 둔다 — 실행부가 사용자 권한을 모르는 폴백을 열면 게이트 우회가 된다.
"""
from __future__ import annotations

import json

from agent_arch.extensions import extensions

# 비가역(전송/등록/삭제) 액션 — 사람 승인 없이는 실행 불가.
APPROVAL_REQUIRED_TOOLS: frozenset[str] = frozenset(
    {
        "calendar_create_event",
        "calendar_update_event",
        "calendar_delete_event",
        "mail_send_message",
        "mail_save_contact",
        "mattermost_post_message",
        "mattermost_skill_send_dm",
        "hr_request_leave",
        "hr_cancel_leave",
        "hr_request_attendance_correction",
    }
)


def requires_approval(tool_name: str) -> bool:
    """도구가 사람 승인(HITL 일시정지)을 요구하면 True를 반환한다."""
    return tool_name in APPROVAL_REQUIRED_TOOLS or tool_name in extensions.approval_required


# 승인 실행(재개) 완료 후 대화에 표시할 확인 문구. /resume 가 LLM 턴 없이 실행하므로
# 도구별로 정확한 완료 메시지를 코드가 제공한다(LLM 이 완료를 지어내지 않게).
ACTION_DONE_MESSAGES: dict[str, str] = {
    "calendar_create_event": "일정을 등록했어요.",
    "calendar_update_event": "일정을 수정했어요.",
    "calendar_delete_event": "일정을 삭제했어요.",
    "mail_send_message": "메일을 전송했어요.",
    "mail_save_contact": "연락처를 공유 주소록에 저장했어요.",
    "mattermost_post_message": "메시지를 전송했어요.",
    "mattermost_skill_send_dm": "DM을 전송했어요.",
    "hr_request_leave": "휴가를 신청했어요.",
    "hr_cancel_leave": "휴가 신청을 취소했어요.",
    "hr_request_attendance_correction": "근태 정정을 신청했어요.",
}


REJECTED_DRAFT_NOTE_HEADER = "[직전 거절된 초안]"


def build_rejected_draft_note(rejected: dict | None) -> str:
    """직전 거절된 초안을 LLM 이 읽을 컨텍스트 블록으로 렌더한다. 없으면 빈 문자열.

    **인자 원본(``arguments``)을 렌더하고 표시용 ``fields`` 는 쓰지 않는다.** fields 는 승인
    카드에 사람이 읽도록 포맷된 값이라(예: 종료가 리터럴 ``"(시작 +1시간)"``, 시각이 ISO 가
    아닌 ``"2026-08-13 09:00"``) 그대로 되먹이면 모델이 그 문자열을 도구에 재전송해
    날짜 형식 오류로 다시 끊긴다.

    블록은 (1) orchestrator 가 ``augmented_input`` 에 붙여 **라우터**가 정정 발화를 조회가
    아닌 실행으로 분류하게 하고, (2) supervisor ``_scope_messages`` 가 워커 메시지에 이월해
    실제 도구를 부르는 모델이 값을 보게 한다. 둘 중 하나만 하면 동작하지 않는다.

    Args:
        rejected: ``{"tool_name": str, "arguments": dict}``. None/형식 불일치면 빈 문자열.
    """
    if not isinstance(rejected, dict):
        return ""
    tool_name = rejected.get("tool_name")
    arguments = rejected.get("arguments")
    if not tool_name or not isinstance(arguments, dict):
        return ""
    # confirm 은 초안/실행을 가르는 시스템 인자라 재호출 원본에서 제외한다(항상 False 로 강제됨).
    args = {k: v for k, v in arguments.items() if k != "confirm"}
    try:
        rendered = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = str(args)
    return (
        f"\n\n{REJECTED_DRAFT_NOTE_HEADER}\n"
        "사용자가 방금 이 초안을 거절했습니다 — 아직 아무것도 등록/발송되지 않았습니다.\n"
        f"도구: {tool_name}\n"
        f"직전 호출 인자(그대로 재사용할 원본): {rendered}\n"
        "사용자가 **직전 초안을 일부만 고쳐 말하면**(예: \"10시로 바꿔줘\", \"장소는 회의실 A로\") "
        "위 인자에 그 수정만 반영해 같은 도구를 confirm=false 로 다시 호출하세요. "
        "언급되지 않은 인자는 위 값을 그대로 넘깁니다.\n"
        "반대로 이번 발화가 **그 자체로 완결된 새 요청**이면(예: \"다음주 수요일 오전 9시에 "
        "○○ 미팅 잡아줘\"처럼 날짜·시각·제목이 다 들어 있으면) 위 값을 **쓰지 말고** 발화 내용만으로 "
        "새로 만드세요 — 옛 초안의 날짜·요일·참석자를 물려받으면 안 됩니다.\n"
        "시작 시각만 바꾸라고 하면 종료 시각도 원래와 같은 길이를 유지하도록 함께 옮기세요.\n"
        "그 일정/문서는 존재하지 않으므로 조회하거나 수정·삭제(update·delete) 도구를 쓰지 마세요.\n"
        "사용자가 중단 의사를 밝히면(괜찮아·됐어·그만) 도구를 호출하지 말고 짧게 확인만 하세요."
    )


def action_done_message(tool_name: str, args: dict | None = None) -> str:
    """승인 실행 완료 후 사용자에게 보여줄 확인 문구를 반환한다.

    Args:
        tool_name: 실행된 도구 이름.
        args: 실행 인자. 한 도구가 인자에 따라 결과가 갈리면(예: 제출 vs 임시저장) 확장의
            문구 함수가 이 값을 보고 분기한다.
    """
    custom = extensions.action_done_messages.get(tool_name)
    if custom is not None:
        return custom(args or {}) if callable(custom) else custom
    return ACTION_DONE_MESSAGES.get(tool_name, "요청을 처리했어요.")
