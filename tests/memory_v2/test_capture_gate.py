"""캡처 게이트 — 실패·중단·빈 답변·가드 교정 턴은 메모리 추출에서 제외(레퍼런스: 실패 턴 캡처 스킵)."""
from __future__ import annotations

from agent_arch.memory.conversation import memory_capture_skip_reason


def test_skip_reasons():
    assert memory_capture_skip_reason(500, "서버 오류") == "status_500"
    assert memory_capture_skip_reason(499, "중단된 답변 일부") == "status_499"          # 사용자 중단
    assert memory_capture_skip_reason(200, "") == "empty_answer"
    assert memory_capture_skip_reason(200, "   ") == "empty_answer"
    assert memory_capture_skip_reason(200, "기억을 삭제했어요. 아직 실제로 처리하지는 않았어요. 원하시면 다시 요청해 주세요.") == "guard_fallback"


def test_normal_turn_not_skipped():
    assert memory_capture_skip_reason(200, "이번 주 메일은 2건입니다.") is None
    assert memory_capture_skip_reason(None, "답변") is None


def test_tool_error_turns_skipped_but_partial_success_kept():
    failed = [{"tool_call_id": "c1", "name": "find_mail_tools", "content": '{"found": 3}'},
              {"tool_call_id": "c2", "name": "mail_list_messages", "content": "[ERROR] since 는 YYYY-MM-DD 형식이어야 합니다"}]
    assert memory_capture_skip_reason(200, "날짜 형식이 잘못되어 조회할 수 없습니다.", failed) == "tool_error"
    partial = failed + [{"tool_call_id": "c3", "name": "mail_list_messages", "content": '{"result": [{"subject": "x"}]}'}]
    assert memory_capture_skip_reason(200, "메일 2건입니다.", partial) is None
    only_discovery = [{"tool_call_id": "c1", "name": "find_mail_tools", "content": "[ERROR] boom"}]
    assert memory_capture_skip_reason(200, "답변", only_discovery) is None          # 탐색 도구만 실패한 건 판정 대상 아님
    assert memory_capture_skip_reason(200, "답변", []) is None
