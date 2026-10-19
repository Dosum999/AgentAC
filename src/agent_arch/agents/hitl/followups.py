"""승인 액션 완료 **후속 제안** — 완료 문구(policy.ACTION_DONE_MESSAGES)의 확장.

/resume 가 승인 실행을 마치면, 실행된 도구 이름으로 여기 등록된 빌더를 찾아 완료 문구에
접미를 덧붙인다(응답과 히스토리 영속 **양쪽** — 화면에만 보이는 문구는 다음 턴 LLM 에게
없는 문구다). "어떤 액션 뒤에 무엇을 제안하나"라는 도메인 지식은 이 모듈이 소유하고,
agent.py(전송 계층)는 ``followup_note`` generic 호출 한 줄만 갖는다.

방식 = **문구 유도**(2026-09-02 담당자 확정): 접미에 영수증 값을 박아 히스토리에 영속하고,
사용자가 '전표 입력해줘'라고 말하면 다음 턴이 account 워커로 라우팅돼 LLM 이 그 문장에서
값을 **옮기기만** 한다(산수·매핑은 전부 코드가 여기서 끝냄). 초안 카드 자동 연쇄(nextPending
체크포인트 적재)도 설계·구현·검증까지 갔으나 구조 변경 폭(HITL 그래프 메타·커밋 화이트리스트
·resume 재배치)이 과하다고 판단해 채택하지 않았다 — 되살릴 일이 생기면
reports/expense_voucher_followup_action_plan.md 와 보존 패치(스크래치 followups-card-chain.py
·expense-voucher-full-with-card-chain.patch)를 참조.

빌더 계약: ``async (pending, effective_args, user_info, conversation_id) -> str``.
빈 문자열 = 제안 없음. 제안은 부가 기능 — 예외는 followup_note 가 전량 흡수한다
(판정 실패가 승인 완료 응답을 깨면 안 된다).
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

_Builder = Callable[[dict, dict, dict, str], Awaitable[str]]


# ── 경비청구 상신 → 매입매출전표 입력 유도 ──────────────────────────────────

VOUCHER_TOOL = "accounting_replay_session_save_voucher"

def _voucher_followup_suffix(hints: Optional[dict]) -> str:
    """완료 문구에 붙일 전표입력 유도 접미 — **영수증 값을 문장에 박는다.**

    voucher_hints 는 OCR 턴의 tool_ctx(직전 1턴만 보존)에 있어 다음 턴에는 구조적으로
    사라진다(2026-09-02 로컬 실측: LLM 이 "품명·금액을 알려달라"고 되물음). 이 접미는
    ① /resume 응답으로 사용자에게 보이고 ② _persist_action_outcome 으로 히스토리에
    평문 영속되므로, 다음 '전표 입력해줘' 턴의 워커가 이 문장에서 값을 옮긴다 —
    값의 산수·출처는 전부 코드가 정했고 LLM 은 옮기기만 한다.

    ⚠️ "해줘"·"응" 같은 **내용 없는 동의는 직전 도메인(eapproval)으로 라우팅**돼 accounting
    도구에 닿지 못한다 — 라우터는 발화와 prior_domain 힌트만 보고 대화 히스토리를 보지
    않는다(router._build_router_system_prompt). 그래서 발화에 도메인 단서('전표 입력')를
    남기게 유도한다.
    """
    base = "필요하시면 '전표 입력해줘'라고 말씀해 주세요."
    if not isinstance(hints, dict) or not hints:
        return f" 이어서 회계 매입매출전표 입력도 할 수 있어요 — {base}"
    parts: list[str] = []
    if hints.get("spent_at"):
        parts.append(str(hints["spent_at"]))
    if hints.get("merchant"):
        parts.append(str(hints["merchant"]))
    if hints.get("supply_amount"):
        parts.append(f"공급가액 {int(hints['supply_amount']):,}원")
        if hints.get("vat_amount"):
            parts.append(f"부가세 {int(hints['vat_amount']):,}원")
    elif hints.get("total_amount"):
        parts.append(f"합계 {int(hints['total_amount']):,}원")
    pm = hints.get("payment_method")  # create 흐름에서 사용자가 확정한 값(있으면 우선)
    if not pm and hints.get("payment_method_guess"):
        pm = {"card": "카드", "cash": "현금영수증"}.get(hints["payment_method_guess"])
    if pm:
        # "결제수단" 라벨 필수 — 값만 넣으면 LLM 이 카드사명(거래처)으로 오인해 trade_name 에
        # 넣는다(2026-09-02 실측: trade_name='법인카드'). wrapper 쪽 가드(_PAYMENT_WORDS)와 짝.
        parts.append(f"결제수단 {pm}")
    if hints.get("category_name"):
        # 분류는 **그대로** 전달한다 — 분류→계정과목→코드 매핑은 회계 도메인
        # (accounting-replay _resolve_account_code 3단 해석) 소유다. agent 가 매핑 표를
        # 들고 있으면 회계 판단이 두 레포로 갈라진다(2026-09-02 담당자 소유권 지적으로 이동).
        # LLM 은 이 단어를 account_code 인자로 복사만 한다(도구 설명이 지시).
        parts.append(f"분류 {hints['category_name']}")
    if not parts:
        return f" 이어서 회계 매입매출전표 입력도 할 수 있어요 — {base}"
    return (
        f" 이어서 회계 매입매출전표 입력도 할 수 있어요 — 영수증 기준: {' · '.join(parts)}. {base}"
    )


async def _voucher_followup_allowed(user_info: dict, pending: dict, approved_values: dict) -> bool:
    """경비청구 상신 승인 직후 '전표입력 후속' 문구를 붙여도 되는지 판정한다.

    ⚠️ 권한 게이트 필수(#230·#209 "권한 없음 vs 기능 없음"): 권한 없는 사용자에게
    안내하면 안내대로 말해도 도구가 안 붙어 "기능이 없다"로 끝난다. 도구 미등록
    (accounting-replay MCP 미연결 — env 게이트)이면 조용히 생략.
    """
    from agent_arch.agents.permissions.gate import is_allowed
    from agent_arch.agents.tools.registry import ToolRegistry
    from agent_arch.services.user_profile import get_profile

    args = {**(pending.get("arguments") or {}), **(approved_values or {})}
    if args.get("template_code") != "expense-report" or args.get("submit") is False:
        return False
    if ToolRegistry.get(VOUCHER_TOOL) is None:
        return False
    try:
        profile = await get_profile(user_info.get("sub"))
    except Exception:
        profile = None
    return is_allowed(VOUCHER_TOOL, profile)


async def _expense_voucher_note(
    pending: dict, effective_args: dict, user_info: dict, conversation_id: str
) -> str:
    """경비청구 상신 완료 → 전표입력 유도 접미(영수증 값 포함)."""
    if not await _voucher_followup_allowed(user_info, pending, effective_args):
        return ""
    from agent_arch.memory.conversation import clear_voucher_hints, load_voucher_hints

    hints = await load_voucher_hints(conversation_id)
    # 결제수단은 OCR 추정(guess)이 아니라 create 흐름에서 사용자가 확정한 값(payment_type
    # 인자)으로 승격한다 — 인자에 없으면 guess 표기로 남는다.
    ptype = str(effective_args.get("payment_type") or "").strip()
    if isinstance(hints, dict) and ptype:
        hints = {**hints, "payment_method": ptype}
    note = _voucher_followup_suffix(hints)
    if isinstance(hints, dict) and hints:
        # consume-once — 값이 접미로 히스토리에 영속된 순간 원본 힌트는 소비된 것이다.
        # 지우지 않으면(TTL 24h) 같은 대화의 다음 경비청구(영수증 없는 건) 완료 문구에
        # 이전 영수증 값이 접미로 붙는다.
        await clear_voucher_hints(conversation_id)
    return note


# 도구 이름 → 후속 접미 빌더. 새 후속 제안은 여기 한 줄 등록이 전부다.
_FOLLOWUP_BUILDERS: dict[str, _Builder] = {
    "eapproval_create_document": _expense_voucher_note,
}


async def followup_note(
    pending: dict, effective_args: dict, user_info: dict, conversation_id: str
) -> str:
    """승인 완료 문구에 덧붙일 후속 제안 접미. 해당 없음/실패 = 빈 문자열."""
    builder = _FOLLOWUP_BUILDERS.get(str(pending.get("tool_name") or ""))
    if builder is None:
        return ""
    try:
        return await builder(pending, effective_args, user_info, conversation_id)
    except Exception:
        logger.exception("[hitl] 후속 제안 조립 실패 — 안내 생략 (tool=%s)", pending.get("tool_name"))
        return ""
