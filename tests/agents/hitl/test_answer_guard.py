"""answer_guard.guard_phantom_action 단위 테스트.

pending_drafts(실제 승인 카드)가 없는데 LLM 이 완료/버튼을 환각한 문장을 제거하는지,
그리고 정당한 경우(카드 존재·조회 요약·읽기 답변)는 건드리지 않는지 검증한다.
"""
import pytest

from agent_arch.agents.hitl.answer_guard import guard_phantom_action, looks_non_korean, _FALLBACK


# ── 언어 감지 (한국어 강제 가드) ────────────────────────────────────

def test_english_answer_flagged():
    assert looks_non_korean("Hello! This is Agent. How can I help you today?") is True


def test_english_summary_flagged():
    text = ("Here's a summary of recent issues based on the Mattermost discussions: "
            "ERP logistics scheduled for July, MES support ongoing.")
    assert looks_non_korean(text) is True


def test_korean_answer_not_flagged():
    assert looks_non_korean("안녕하세요! 에이전트입니다. 무엇을 도와드릴까요?") is False


def test_korean_with_english_terms_not_flagged():
    text = "MES 채널의 최근 이슈는 ERP 물류 일정과 Mattermost 관리자 UI 작업입니다."
    assert looks_non_korean(text) is False


def test_short_english_not_flagged():
    assert looks_non_korean("OK") is False           # 너무 짧음
    assert looks_non_korean("Agent") is False         # 고유명사 수준


def test_empty_not_flagged():
    assert looks_non_korean("") is False


# ── 환각 제거(카드 없음) ────────────────────────────────────────────

def test_phantom_button_removed_when_no_drafts():
    """카드 없는데 승인 버튼 안내 → 제거."""
    out = guard_phantom_action("아래 [승인] 버튼을 눌러 주세요.", has_pending_drafts=False)
    assert "승인" not in out or out == _FALLBACK
    assert out == _FALLBACK  # 문장이 전부 환각이라 폴백


def test_phantom_done_terse_removed_when_no_drafts():
    """카드 없는데 짧은 완료 확인 → 제거."""
    out = guard_phantom_action("일정을 삭제했어요.", has_pending_drafts=False)
    assert out == _FALLBACK


def test_phantom_button_and_done_combo_removed():
    """버튼+완료 동반 환각 → 둘 다 제거, 앞의 정상 문장은 유지."""
    text = "요청을 확인했어요. 일정을 삭제했습니다. 아래 [승인] 버튼을 눌러 주세요."
    out = guard_phantom_action(text, has_pending_drafts=False)
    assert "요청을 확인했어요." in out
    assert "삭제" not in out
    assert "승인" not in out


def test_all_phantom_falls_back():
    out = guard_phantom_action("네, 삭제 완료했습니다. [승인]/[거절] 버튼을 눌러 주세요.",
                               has_pending_drafts=False)
    assert out == _FALLBACK


@pytest.mark.parametrize("answer", [
    "요청하신 'BASIC-MCP-UI-20261017103001-초안' 임시저장 및 미리보기 작업을 취소하고, 생성된 초안을 폐기하였습니다. 해당 문서의 저장, 상신, 승인 및 알림 발송은 이루어지지 않았습니다.",
    '요청하신 "BASIC-MCP-UI-20261017104056-초안"의 임시저장 및 상신 작업을 취소하였습니다. 해당 초안은 저장되지 않았으며, 관련하여 진행 중인 문서나 알림 발송은 없습니다.',
    "해당 합성 기안서 초안과 미리보기 작업을 요청하신 내용대로 모두 취소했습니다. 더 이상 진행 중인 초안은 없습니다.",
    "말씀하신 합성 기안서 초안 및 미리보기 작업의 취소를 완료하였습니다. 관련하여 대기 중인 작업도 없습니다.",
    "지정하신 합성 기안서 초안 및 미리보기 작업은 정상적으로 모두 폐기되었습니다.",
])
def test_unexecuted_draft_cancel_claim_corrected_with_dependent_assurance(answer: str) -> None:
    """긴 제목·폐기 동사와 뒤따르는 무대기 단정도 같은 산문 줄에서 교정한다."""
    assert guard_phantom_action(answer, has_pending_drafts=False) == _FALLBACK


@pytest.mark.parametrize("answer", [
    "초안을 취소하려면 현재 승인 대기 여부를 먼저 확인해야 합니다.",
    "초안을 취소하지 않았습니다.",
    "요청하신 초안의 취소 여부는 아직 확인하지 않았습니다.",
    "업무 이력에는 담당자가 지난주 합성 기안서 초안과 미리보기 작업을 취소하였습니다. 현재 상태는 이력을 확인해 주세요.",
    '"요청하신 초안을 취소하였습니다"라는 문구는 실제 처리 후에만 사용하세요.',
    "미리보기 화면에 '취소하였습니다'라고 표시되는 예시 문장을 설명합니다.",
])
def test_draft_cancel_guidance_negation_and_report_preserved(answer: str) -> None:
    """안내·부정·인용과 과거 이력 보고를 새 완료 주장으로 오인하지 않는다."""
    assert guard_phantom_action(answer, has_pending_drafts=False) == answer


def test_cancel_claim_correction_keeps_other_markdown_lines() -> None:
    """취소 완료 산문을 교정해도 독립된 표·코드 원문은 보존한다."""
    claim = "요청하신 합성 기안서 초안과 미리보기 작업을 모두 취소하였습니다. 남은 작업은 없습니다."
    table = "| 상태 | 설명 |\n|---|---|\n| 취소 | 기록 표시 |"
    code = "```\n요청하신 초안을 취소하였습니다.\n```"
    output = guard_phantom_action(f"{claim}\n\n{table}\n\n{code}", False)
    assert output == f"{_FALLBACK}\n\n{table}\n\n{code}"


def test_confirmed_direct_action_and_pending_card_contracts_unchanged() -> None:
    """직접 실행 근거 허용과 실제 카드가 있는 기존 호출 계약은 유지한다."""
    answer = "요청하신 합성 초안의 미리보기 작업을 취소하였습니다."
    assert guard_phantom_action(answer, False, allow_done_claims=True) == answer
    assert guard_phantom_action("초안을 준비했습니다. 아래 [승인] 버튼을 눌러 주세요.", True).startswith("초안을 준비했습니다")


@pytest.mark.parametrize("user_input", [
    "합성 기안서 초안의 미리보기 작업을 취소합니다.",
    "이 초안을 폐기해 주세요.",
    "미리보기를 취소해줘.",
    "기안서 초안은 그만둘게요.",
])
def test_draft_cancel_request_is_buffered(user_input: str) -> None:
    """실제 초안 취소 요청만 화면 방출 전 검사를 기다린다."""
    from agent_arch.agents.hitl.answer_guard import needs_draft_cancel_guard

    assert needs_draft_cancel_guard(user_input)


@pytest.mark.parametrize("user_input", [
    "초안 취소 방법을 설명해 주세요.",
    "초안을 취소하지 마세요.",
    "'초안 미리보기를 취소해 주세요'라는 문구를 번역해 주세요.",
    "방문 예약을 취소해 주세요.",
    "일반적인 화면 사용법을 알려 주세요.",
])
def test_other_requests_do_not_enable_cancel_buffer(user_input: str) -> None:
    """방법·부정·인용·다른 도메인 요청 때문에 일반 스트림을 보류하지 않는다."""
    from agent_arch.agents.hitl.answer_guard import needs_draft_cancel_guard

    assert not needs_draft_cancel_guard(user_input)


# ── 정당한 경우(건드리지 않음) ──────────────────────────────────────

def test_kept_when_drafts_present():
    """실제 카드가 있으면 버튼 안내는 정당 → 원문 유지."""
    text = "일정을 삭제하려면 아래 [승인] 버튼을 눌러 주세요."
    out = guard_phantom_action(text, has_pending_drafts=True)
    assert out == text


def test_long_summary_report_not_stripped():
    """긴 요약/보고 문장의 '삭제했습니다'는 액션 주장이 아니므로 오탐 없이 유지."""
    text = ("어제 스크럼 채널 요약입니다. 김대리가 배포 일정을 삭제했다고 팀에 공지했습니다. "
            "이후 QA 일정이 조정되었습니다.")
    out = guard_phantom_action(text, has_pending_drafts=False)
    assert out == text  # 버튼 없음 + 문장이 길어 완료-환각으로 안 봄


def test_ui_howto_button_guidance_not_stripped():
    """실제 화면 UI 사용법 안내(웹메일 스팸 신고 고정 사실)는 카드 환각이 아니다.

    회귀 (실측 2026-08-10): 스팸 신고 절차의 "버튼을 클릭합니다" 문장이 일반 버튼
    패턴에 걸려 저장본에서 잘렸다 — 스트리밍 화면과 히스토리가 어긋난다.
    """
    text = (
        "웹메일에서 스팸 신고를 하시려면 받은 편지함에서 신고할 메일을 클릭한 뒤, "
        "메일 미리보기 화면 상단의 **스팸 편지함** 버튼을 누르시면 됩니다. "
        "잘못 신고한 경우에는 스팸 편지함에서 해당 메일을 클릭한 뒤 상단의 "
        "**정상** 버튼을 눌러 해제할 수 있습니다."
    )
    assert guard_phantom_action(text, has_pending_drafts=False) == text


def test_generic_button_with_approval_context_still_stripped():
    """'버튼을 눌러 승인' 처럼 승인 문맥이 함께 있는 일반 버튼 문구는 여전히 제거."""
    out = guard_phantom_action("문서 하단의 버튼을 눌러 승인해 주세요.", has_pending_drafts=False)
    assert out == _FALLBACK


def test_plain_read_answer_unchanged():
    text = "개발팀 채널의 최근 메시지는 배포 준비에 관한 내용입니다."
    assert guard_phantom_action(text, has_pending_drafts=False) == text


def test_empty_content():
    assert guard_phantom_action("", has_pending_drafts=False) == ""


# ── 마크다운 구조 보존 (실측 2026-08-26 BOM 표 행 삭제 회귀) ──────────

_BOM_ANSWER = (
    "BOM 등록 제품 현황입니다.\n"
    "\n"
    "| 품번 | 품명 | 규격 | 단위 | BOM |\n"
    "|---|---|---|---|---|\n"
    "| G101 | 전동식 파워스티어링 컬럼 어셈블리 | - | EA | **등록 완료** |\n"
    "| G102 | 하이브리드 차량용 배터리 냉각 모듈 | - | EA | **등록 완료** |\n"
    "| G103 | 조향 너클 | - | EA | **등록 완료** |\n"
)


def test_table_rows_kept():
    """표 셀의 '등록 완료'는 데이터 상태 라벨 — 액션 주장이 아니다.

    회귀 (실측 2026-08-26): 짧은 G103 행만 _TERSE_MAX 이하라 저장본에서 삭제됐다.
    """
    assert guard_phantom_action(_BOM_ANSWER, has_pending_drafts=False) == _BOM_ANSWER


def test_table_kept_while_prose_phantom_corrected():
    """산문의 진짜 환각만 교정되고 표는 무손상."""
    text = ("네, 일정을 삭제했어요.\n"
            "\n"
            "| 품번 | BOM |\n"
            "|---|---|\n"
            "| G103 | 등록 완료 |")
    out = guard_phantom_action(text, has_pending_drafts=False)
    assert "| G103 | 등록 완료 |" in out
    assert "삭제했어요" not in out
    assert _FALLBACK in out
    assert out.count("\n") == text.count("\n")   # 줄 구조 보존


def test_newlines_preserved_after_removal():
    """제거가 일어나도 문단 개행이 유지된다(예전엔 답변 전체가 한 줄로 뭉개졌다)."""
    text = "조회 결과입니다.\n\n일정을 삭제했어요.\n\n추가로 확인할 내용은 없습니다."
    out = guard_phantom_action(text, has_pending_drafts=False)
    assert out.split("\n") == ["조회 결과입니다.", "", _FALLBACK, "", "추가로 확인할 내용은 없습니다."]


def test_code_fence_kept():
    """코드펜스 안의 로그 원문은 판정하지 않는다."""
    text = ("로그를 확인해 봤습니다.\n"
            "\n"
            "```\n"
            "2026-08-26 10:11 일정을 삭제했습니다.\n"
            "```\n"
            "\n"
            "실제 삭제 여부는 확인이 필요합니다.")
    assert guard_phantom_action(text, has_pending_drafts=False) == text


def test_heading_kept():
    """헤딩은 문서 구조 라벨."""
    text = "## 등록 완료\n\n조회 결과입니다."
    assert guard_phantom_action(text, has_pending_drafts=False) == text


def test_list_item_status_label_kept():
    """목록 항목은 조회 결과 나열이 대부분 — 완료-주장 판정 제외."""
    text = "BOM 현황입니다.\n\n- G103 조향 너클: 등록 완료\n- G104 링크 어셈블리: 등록 완료"
    assert guard_phantom_action(text, has_pending_drafts=False) == text


def test_list_item_phantom_button_removed():
    """목록이어도 없는 승인 버튼 안내는 제거하고 불릿 마커는 유지."""
    text = "요청 내용을 정리했어요.\n\n- 아래 [승인] 버튼을 눌러 주세요."
    out = guard_phantom_action(text, has_pending_drafts=False)
    assert "승인" not in out
    assert "요청 내용을 정리했어요." in out
    assert f"- {_FALLBACK}" in out


def test_correction_phrase_inserted_once():
    """환각이 여러 건이어도 교정 문구는 첫 자리에 한 번만."""
    text = "조회 결과를 정리했어요.\n일정을 삭제했어요.\n메일도 발송했습니다."
    out = guard_phantom_action(text, has_pending_drafts=False)
    assert out.count(_FALLBACK) == 1
    assert "삭제" not in out and "발송" not in out
    assert out.startswith("조회 결과를 정리했어요.")


# ── 승인 없이 실행되는 액션 도구(memory_forget 등)의 완료 주장은 환각이 아니다 ──
from agent_arch.agents.hitl.answer_guard import guard_phantom_action, has_direct_action


def test_has_direct_action_detects_memory_tools():
    assert has_direct_action([{"name": "memory_search"}, {"name": "memory_forget", "arguments": {}}])
    assert has_direct_action([{"function": {"name": "memory_remember"}}])
    assert not has_direct_action([{"name": "mail_list_messages"}])
    assert not has_direct_action(None) and not has_direct_action(["garbage"])



def test_done_claim_kept_when_direct_action_ran():
    text = "'답변에 표를 사용하지 않는다' 지시를 삭제했어요. 이제 표를 쓸 수 있어요."
    # 실측(2026-09-07): memory_forget 이 실제로 돌았는데 카드가 없다는 이유로 교정 문구로 바뀌었다.
    assert "아직 실제로 처리하지는 않았어요" in guard_phantom_action(text, False)
    assert guard_phantom_action(text, False, allow_done_claims=True) == text


def test_button_hallucination_still_removed_with_direct_action():
    text = "기억을 삭제했어요. 아래 [승인] 버튼을 눌러 주세요."
    out = guard_phantom_action(text, False, allow_done_claims=True)
    assert "삭제했어요" in out and "[승인]" not in out


class TestFileCreationIsNotAHallucination:
    """파일을 만들어 놓고 "안 만들었다" 고 말하는 역환각 — 2026-09-11 실측 회귀.

    개발기에서 워드 파일이 **실제로 만들어지고 다운로드 링크까지 답변에 실렸는데**,
    바로 그 위 문장이 "아직 실제로 처리하지는 않았어요" 로 바뀌었다. `_DONE_RE` 의
    `생성했습니다` 가 걸렸고 승인 카드(pending_drafts)는 없었기 때문이다.

    파일 생성은 비가역도 아니고 승인 대상도 아니다 — 밖으로 나가는 건
    `mattermost_post_message` 의 승인 카드가 따로 막는다. memory_forget 때와 같은
    성질의 버그이고(2026-09-07), 같은 장치(DIRECT_ACTION_TOOLS)로 막는다.
    """

    _ANSWER = (
        "요청하신 Agent AI 크레딧 정책 분석을 워드 파일로 생성했습니다.\n"
        "아래 링크에서 다운로드하실 수 있습니다.\n\n"
        "[Agent_AI_크레딧_정책_분석.docx](https://example/dl)"
    )

    def test_create_file_is_a_direct_action(self):
        from agent_arch.agents.hitl.answer_guard import DIRECT_ACTION_TOOLS, has_direct_action

        assert "create_file" in DIRECT_ACTION_TOOLS
        assert has_direct_action([{"name": "create_file"}])

    def test_claim_survives_when_the_file_was_really_made(self):
        from agent_arch.agents.hitl.answer_guard import guard_phantom_action, has_direct_action

        out = guard_phantom_action(
            self._ANSWER, False,
            allow_done_claims=has_direct_action([{"name": "create_file"}]))
        assert "워드 파일로 생성했습니다" in out
        assert "아직 실제로 처리하지는 않았어요" not in out
        assert "](https://example/dl)" in out          # 링크는 어느 경우든 보존

    def test_claim_is_still_removed_without_the_tool_call(self):
        """도구를 안 불렀으면 여전히 환각이다 — 가드를 통째로 끄면 안 된다."""
        from agent_arch.agents.hitl.answer_guard import guard_phantom_action

        out = guard_phantom_action(self._ANSWER, False, allow_done_claims=False)
        assert "워드 파일로 생성했습니다" not in out
        assert "아직 실제로 처리하지는 않았어요" in out

    def test_button_rule_still_applies_to_direct_actions(self):
        """승인 카드가 없는데 버튼을 안내하면 파일을 만들었어도 환각이다."""
        from agent_arch.agents.hitl.answer_guard import guard_phantom_action, has_direct_action

        answer = "파일을 생성했습니다.\n아래 [승인] 버튼을 눌러 주세요."
        out = guard_phantom_action(
            answer, False,
            allow_done_claims=has_direct_action([{"name": "create_file"}]))
        assert "[승인]" not in out
