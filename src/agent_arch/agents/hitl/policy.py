"""HITL 승인 정책 — 어떤 도구가 사람 승인을 강제로 요구하는지 코드로 선언한다.

승인 필요 여부를 시스템 프롬프트/도구 설명이 아니라 이 레지스트리에서 결정한다.
권위 소스는 ``app/api/v1/workspace.py`` 의 도메인 커밋 엔드포인트 매핑(승인 시
``confirm=True`` 로 실제 실행되는 비가역 액션)이며, 그 도구 이름 집합과 1:1로 정합한다.
도구를 추가/삭제할 때는 workspace.py 매핑과 이 집합을 함께 갱신한다.
"""
from __future__ import annotations

import json

# 비가역(전송/등록/삭제) 액션 — 사람 승인 없이는 실행 불가.
# workspace.py 의 커밋 엔드포인트가 confirm=True 로 호출하는 도구와 동일.
APPROVAL_REQUIRED_TOOLS: frozenset[str] = frozenset(
    {
        "calendar_create_event",   # POST /workspace/calendar/event
        "calendar_update_event",   # POST /workspace/calendar/event/update
        "calendar_delete_event",   # POST /workspace/calendar/event/delete
        "mail_send_message",       # POST /workspace/mail/send
        "mail_save_contact",       # POST /workspace/mail/contact
        "mattermost_post_message", # POST /workspace/mattermost/post
        "mattermost_skill_send_dm",# POST /workspace/mattermost/dm
        "hr_request_leave",        # POST /workspace/hr/leave
        "hr_cancel_leave",         # POST /workspace/hr/leave/cancel
        "hr_request_attendance_correction",  # POST /workspace/hr/attendance/correction
        "eapproval_create_document",  # POST /workspace/eapproval/submit (임시보관: /save-draft)
        "eapproval_approve_document", # POST /workspace/eapproval/approve
        "eapproval_reject_document",  # POST /workspace/eapproval/reject
        # 매입매출전표 스니펫 재생(accounting-replay-mcp, :8772) — workspace 커밋
        # 폴백이 없다: /resume(체크포인트 검증) 전용. 실행부가 사용자 권한을 모르므로
        # 폴백을 열면 게이트 우회가 된다 (accounting_create_expense_slips 선례, 이슈).
        # dmp_sid 세션 방식(2026-08-28) — 사용자 세션으로 실 게이트 뒤 프론트에 저장한다.
        # 회사는 세션이 강제(사용자 소속 회사)라 도구가 대상을 못 바꾸지만, 실 저장 =
        # 비가역이므로 승인 필수. preview(진입만, 저장 없음)는 이 목록에 넣지 않는다.
        "accounting_replay_session_save_voucher",
        # 수정·삭제도 같은 이유로 승인 필수다. 특히 삭제는 전표와 분개가 함께 지워지고
        # **복구 수단이 없다**(대상 시스템은 soft-delete 가 없어 삭제전표복구 메뉴 자체가 없다).
        "accounting_replay_update_voucher",
        "accounting_replay_delete_voucher",
        # SCMS 채널 전용 — workspace.py 커밋 엔드포인트가 없다(그 매핑은 dmp 워크스페이스
        # 전용). 승인은 /api/v1/scms/agent/{cid}/resume 로만 들어오고, 승인의 의미는
        # "이 값으로 발주요청 화면을 채워라"다(발주 제출은 사람이 그 화면에서 한다).
        "scms_purchase_order_draft",
        # 메신저 SCMS 발주요청 — 승인되면 요청자 세션으로 SCMS 에 초안을 만들고 스레드 도면을
        # 올린다(SCMS 에 실제로 남는다). 발주 제출은 아니지만 남의 이름으로 기록이 생기므로 승인 필수.
        "scms_order_request_draft",
        # (scms_quote_submit 은 여기 없다 — SCMS 쓰기는 scms_card 의 공급사 전용 카드 클릭 처리기에서만 돈다.
        #  도구 자체는 카드를 보내기만 하고 SCMS 를 바꾸지 않는다.)
        # 메신저 SCMS 계약서 파트(scms-mcp 쓰기) — 낙찰은 되돌릴 수 없고, 변경 통보·회신·최종 승인은 계약서를 다시 낸다.
        # 계약서 카드 버튼은 카드 자체가 확인이라 이 승인을 거치지 않는다(skills/scms_contract/flow). 채팅으로 요청하면 승인 카드.
        "scms_award",
        "scms_order_change",
        "scms_change_respond",
        # 예약 만들기 — 승인 뒤 /resume 이 confirm=True 로 저장한다(workspace 커밋 없음). 저장되면 정해진 때마다
        # 그 사람 이름으로 실행되므로, 일정과 첫 실행 시각을 카드로 확인받는다.
        "schedule_create",
        # 물류 수주서 등록(logistics-mcp) — 고객 발주서(PDF 등)를 읽어 수주서 1건을 만든다.
        # workspace 커밋 폴백이 없다: /resume 전용(accounting_replay 와 같은 이유 — 폴백을 열면
        # 게이트 우회). 회사는 게이트웨이 세션이 강제한다. 실 등록 = 비가역이라 승인 필수.
        "logistics_create_sales_order",
        # 물류 수주서 수정(logistics-mcp) — 등록된 수주의 줄 수량·납기·단가 변경과 품목 추가. 등록과 같은
        # 이유로 /resume 전용 · 승인 필수(실 수정 = 비가역, 바뀌기 전 값은 ERP 이력에만 남는다).
        "logistics_update_sales_order",
        # 물류 발주서 등록(logistics-mcp) — 수주 부족분을 공급처별 발주서로. 실제 구매 문서가 생기고 공급처에
        # 나갈 수 있는 문서라 승인 필수 · /resume 전용(등록 도구와 같은 이유).
        "logistics_create_purchase_orders",
        # 물류 발주서 메일(logistics-mcp) — 발주서 PDF 를 공급처에 실제로 메일 발송. 사외로 나가 되돌릴 수 없다.
        "logistics_send_purchase_order_mail",
        # 물류 생산지시 변경(logistics-mcp) — 지시수량·완료예정 변경 + 투입자재 재전개. 생산관리팀 카드 버튼은
        # 카드 자체가 확인이라 이 승인을 거치지 않고 실행한다(webhook._prdir_click). 채팅으로 요청하면 승인 카드.
        "logistics_update_production_order",
        # 물류 이관(logistics-mcp) — 드라이브 입고·출하 엑셀을 입고처리·출고처리 문서 수백 건으로 만들고 재고를
        # 움직인다. 승인 카드는 ERP 검증을 실제처럼 돌린 미리보기(dry_run)다. /resume 전용(등록 도구와 같은 이유).
        "logistics_migrate_drive_excel",
    }
)


def requires_approval(tool_name: str) -> bool:
    """도구가 사람 승인(HITL 일시정지)을 요구하면 True를 반환한다."""
    return tool_name in APPROVAL_REQUIRED_TOOLS


# 승인 실행(재개) 완료 후 대화에 표시할 확인 문구. /resume 가 LLM 턴 없이 실행하므로
# (옛 흐름의 에이전트 응답을 대체) 도구별로 정확한 완료 메시지를 코드가 제공한다.
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
    "eapproval_create_document": "전자결재 문서를 상신했어요.",
    "eapproval_approve_document": "결재를 승인했어요.",
    "eapproval_reject_document": "결재를 반려했어요.",
    "accounting_replay_session_save_voucher": "매입매출전표를 입력했어요.",
    "accounting_replay_update_voucher": "매입매출전표를 수정했어요.",
    "accounting_replay_delete_voucher": "매입매출전표를 삭제했어요.",
    "scms_purchase_order_draft": "발주 요청서 작성 화면으로 이동할게요. 값은 미리 채워 둘게요.",
    "scms_order_request_draft": "SCMS 발주요청서를 만들었어요.",
    "scms_award": "낙찰했어요. 계약서가 발행됐어요.",
    "scms_order_change": "SCMS 발주 변경을 반영했어요.",
    "scms_change_respond": "변경 통보에 회신했어요.",
    "schedule_create": "예약을 만들었어요.",
    "logistics_create_sales_order": "수주서를 등록했어요.",
    "logistics_update_sales_order": "수주서를 수정했어요.",
    "logistics_create_purchase_orders": "발주서를 등록했어요.",
    "logistics_send_purchase_order_mail": "공급처에 발주서 메일을 보냈어요.",
    "logistics_update_production_order": "생산지시를 바꿨어요.",
    "logistics_migrate_drive_excel": "엑셀 자료를 물류로 이관했어요.",
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
        args: 실행 인자(있으면 분기에 사용). 전자결재처럼 한 도구가 submit=true(상신)/
            false(임시보관)로 갈리는 경우, 인자를 보고 문구를 다르게 낸다.
    """
    if (
        tool_name == "eapproval_create_document"
        and args is not None
        and args.get("submit") is False
    ):
        return "전자결재 문서를 임시저장했어요."
    return ACTION_DONE_MESSAGES.get(tool_name, "요청을 처리했어요.")
