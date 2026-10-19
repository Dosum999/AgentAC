"""POS 턴 저장 시 정화(pos_history) 단위 테스트.

페이로드 형식은 2026-08-28 HAR 실측 기준 — POS 대시보드는 \r\n 개행으로 섹션을
구분하고, 1턴(플래닝)은 [POS Agent Catalog], 2턴(판단)은 [POS Query Plan]으로 시작한다.
"""
import logging

from agent_arch.utils.pos_history import (
    PosSanitizedTurn,
    extract_user_question,
    is_pos_input,
    sanitize_pos_turn,
    strip_pos_tag_blocks,
)

# ─── HAR 실측 형식 축약 fixture ───────────────────────────────────────────────

QUESTION = "개발실에 몇 명 있어?"

# 1턴(플래닝): 카탈로그 + 화면 컨텍스트 + 질문 + 역할 + 출력 계약, \r\n 개행
TURN1_INPUT = (
    "[POS Agent Catalog]\r\n"
    '{"catalog_version":"a8763c782572d421","locations":[{"location_code":"1F-DEV"}]}\r\n'
    "[화면 컨텍스트]\r\n"
    '{"id":"acme-office","label":"통합 관제 운영"}\r\n'
    "[사용자 질문]\r\n"
    f"{QUESTION}\r\n"
    "\r\n"
    "[역할]\r\n"
    "- 질문이 뜻하는 구역, 장치 종류를 스스로 해석하세요.\r\n"
    "\r\n"
    "[출력 계약]\r\n"
    "다른 문장 없이 아래 태그 하나만 반환하세요.\r\n"
    '[POS_QUERY_PLAN]{"intent":"..."}[/POS_QUERY_PLAN]'
)
TURN1_ANSWER = (
    '[POS_QUERY_PLAN]{"intent":"status","location_codes":["1F-DEV"],'
    '"include":["occupancy"],"needs_clarification":false}[/POS_QUERY_PLAN]'
)

# 2턴(판단): 플랜 + 컨텍스트 + 액션 목록(값이 [ 로 시작하는 멀티라인 JSON) + 질문 + 규칙
TURN2_INPUT = (
    "[POS Query Plan]\r\n"
    '{"intent":"status","location_codes":["1F-DEV"]}\r\n'
    "[POS Current Context]\r\n"
    '{"data":{"occupancy":[{"location_code":"1F-DEV","person_count":10}]}}\r\n'
    "[POS Executable Actions]\r\n"
    '[{"action_type":"light.off"},{"action_type":"light.on"}]\r\n'
    "[사용자 질문]\r\n"
    f"{QUESTION}\r\n"
    "\r\n"
    "[판단 규칙]\r\n"
    "- 위 Context만 사실로 사용하고 추측하지 마세요.\r\n"
    "\r\n"
    "[출력 계약]\r\n"
    '[POS_ACTION_PLAN]{"actions":[]}[/POS_ACTION_PLAN]'
)
TURN2_ANSWER = (
    "현재 개발실에는 10명이 있습니다.\n"
    '[POS_ACTION_PLAN]{"actions":[]}[/POS_ACTION_PLAN]'
)


class TestIsPosInput:
    def test_catalog_prefix(self):
        assert is_pos_input(TURN1_INPUT) is True

    def test_query_plan_prefix(self):
        assert is_pos_input(TURN2_INPUT) is True

    def test_plain_text(self):
        assert is_pos_input("개발실에 몇 명 있어?") is False

    def test_hitl_synthetic_turn(self):
        # agent.py HITL 합성 턴 — 같은 저장 깔때기를 타지만 정화 대상이 아니어야 한다
        assert is_pos_input("(초안 거절됨)") is False

    def test_empty(self):
        assert is_pos_input("") is False


class TestExtractUserQuestion:
    def test_turn1_crlf(self):
        assert extract_user_question(TURN1_INPUT) == QUESTION

    def test_turn2_crlf(self):
        assert extract_user_question(TURN2_INPUT) == QUESTION

    def test_lf_only(self):
        # 클라이언트 구현이 \n 만 쓰는 경우에도 동일 동작해야 한다
        assert extract_user_question(TURN1_INPUT.replace("\r\n", "\n")) == QUESTION

    def test_question_at_end_of_prompt(self):
        # 질문이 마지막 섹션이어도(뒤 섹션 없음) \Z 경계로 추출된다
        payload = "[POS Agent Catalog]\r\n{}\r\n[사용자 질문]\r\n" + QUESTION
        assert extract_user_question(payload) == QUESTION

    def test_inline_brackets_preserved(self):
        # 질문 본문 안의 인라인 대괄호는 섹션 경계가 아니다
        payload = TURN1_INPUT.replace(QUESTION, "[긴급] 회의실 [A동] 불 꺼줘")
        assert extract_user_question(payload) == "[긴급] 회의실 [A동] 불 꺼줘"

    def test_multiline_question(self):
        payload = TURN1_INPUT.replace(QUESTION, "개발실에\r\n몇 명 있어?")
        assert extract_user_question(payload) == "개발실에\r\n몇 명 있어?"

    def test_missing_section(self):
        assert extract_user_question("[POS Agent Catalog]\r\n{}") is None

    def test_empty_question_body(self):
        payload = "[POS Agent Catalog]\r\n{}\r\n[사용자 질문]\r\n\r\n[역할]\r\n- ..."
        assert extract_user_question(payload) is None


class TestStripPosTagBlocks:
    def test_tag_only_answer(self):
        assert strip_pos_tag_blocks(TURN1_ANSWER) == ""

    def test_text_with_trailing_tag(self):
        assert strip_pos_tag_blocks(TURN2_ANSWER) == "현재 개발실에는 10명이 있습니다."

    def test_multiple_distinct_tags(self):
        answer = (
            "안내드립니다.\n"
            "[POS_QUERY_PLAN]{}[/POS_QUERY_PLAN]\n"
            "[POS_ACTION_PLAN]{}[/POS_ACTION_PLAN]"
        )
        assert strip_pos_tag_blocks(answer) == "안내드립니다."

    def test_repeated_same_tag(self):
        answer = "[POS_ACTION_PLAN]{1}[/POS_ACTION_PLAN]중간 텍스트[POS_ACTION_PLAN]{2}[/POS_ACTION_PLAN]"
        assert strip_pos_tag_blocks(answer) == "중간 텍스트"

    def test_multiline_json_payload(self):
        answer = '답변.\n[POS_ACTION_PLAN]{\n  "actions": [\n    {"a": 1}\n  ]\n}[/POS_ACTION_PLAN]'
        assert strip_pos_tag_blocks(answer) == "답변."

    def test_unclosed_tag_tail_removed(self):
        # 스트림 중단(status 499)으로 닫는 태그 없이 잘린 케이스 — 꼬리 전체 제거
        answer = '10명이 있습니다.\n[POS_ACTION_PLAN]{"actions":[{"acti'
        assert strip_pos_tag_blocks(answer) == "10명이 있습니다."

    def test_unclosed_tag_only_is_internal(self):
        # 플래닝 턴이 태그 중간에서 끊겨도 internal 판정으로 이어져야 한다
        assert strip_pos_tag_blocks('[POS_QUERY_PLAN]{"inte') == ""

    def test_orphan_closing_tag_kept_with_warning(self, caplog):
        # 계약 밖 형식(고아 닫는 태그) — 추측 삭제 대신 잔존 + 경고
        answer = "앞 텍스트 [/POS_ACTION_PLAN] 뒤 텍스트"
        with caplog.at_level(logging.WARNING, logger="agent_arch.utils.pos_history"):
            result = strip_pos_tag_blocks(answer)
        assert "[/POS_ACTION_PLAN]" in result
        assert any("고아 태그" in r.message for r in caplog.records)

    def test_plain_brackets_untouched(self):
        answer = "확인했습니다. [참고] 자세한 내용은 문서를 보세요."
        assert strip_pos_tag_blocks(answer) == answer


class TestSanitizePosTurn:
    def test_turn1_planning(self):
        result = sanitize_pos_turn(TURN1_INPUT, TURN1_ANSWER)
        assert result == PosSanitizedTurn(query=QUESTION, answer="", internal=True)

    def test_turn2_decision(self):
        result = sanitize_pos_turn(TURN2_INPUT, TURN2_ANSWER)
        assert result == PosSanitizedTurn(
            query=QUESTION, answer="현재 개발실에는 10명이 있습니다.", internal=False
        )

    def test_non_pos_returns_none(self):
        assert sanitize_pos_turn("일반 질문입니다", "일반 답변입니다") is None

    def test_extraction_failure_returns_none_with_warning(self, caplog):
        # 질문 섹션이 없으면 반쪽 정화 대신 원자적 no-op(None) + 경고
        with caplog.at_level(logging.WARNING, logger="agent_arch.utils.pos_history"):
            result = sanitize_pos_turn("[POS Agent Catalog]\r\n{...}", TURN1_ANSWER)
        assert result is None
        assert any("추출 실패" in r.message for r in caplog.records)

    def test_empty_answer_input(self):
        # 응답이 아예 빈 턴(에러 저장 등)도 internal 로 수렴
        result = sanitize_pos_turn(TURN1_INPUT, "")
        assert result is not None
        assert result.internal is True
