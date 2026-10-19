"""압축 요약 점검 — 재압축이 이전 요약의 고유값·항목을 버리면 다시 시키고, 두 번 실패하면 압축 취소."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent_arch.llm.base import Message
from agent_arch.memory import compact_summary as cs


def _body(facts: str = "- 없음", values: str = "- 없음") -> str:
    return (f"## 목적과 진행 상황\n- 진행 중\n## 확정된 사실과 결정\n{facts}\n"
            f"## 사용한 도구와 결과\n- 없음\n## 고유값(이름·날짜·코드·수치)\n{values}\n## 남은 일\n- 없음")


def _llm(*contents):
    return SimpleNamespace(invoke=AsyncMock(side_effect=[SimpleNamespace(content=c) for c in contents]))


def test_missing_section_is_a_problem():
    assert "섹션" in cs.check_summary("## 목적과 진행 상황\n- x", "")


def test_lost_previous_value_is_a_problem():
    prev = _body(values="- 견적번호 PO-2026-0412, 합계 12,116,000")
    new = _body(values="- 견적번호 PO-2026-0412")
    assert "고유값" in cs.check_summary(new, prev)


def test_kept_values_pass_even_with_comma_changes():
    prev = _body(values="- 합계 12,116,000")
    assert cs.check_summary(_body(values="- 합계 12116000 원"), prev) is None


@pytest.mark.asyncio
async def test_retry_then_accept_and_questions_appended():
    prev = _body(values="- 코드 AB-1234") + f"\n\n{cs.QUESTIONS_HEADER}\n- 첫 질문"
    llm = _llm(_body(values="- 없음"), _body(values="- 코드 AB-1234"))
    out = await cs.summarize(llm, [Message(role="user", content="두 번째 질문"),
                                   Message(role="assistant", content="답")], prev)
    assert out is not None and "AB-1234" in out
    assert llm.invoke.await_count == 2
    body, qs = cs.split_questions(out)
    assert qs == ["첫 질문", "두 번째 질문"]
    assert cs.QUESTIONS_HEADER not in body


@pytest.mark.asyncio
async def test_two_failures_cancel_compaction():
    llm = _llm("엉터리", "또 엉터리")
    assert await cs.summarize(llm, [Message(role="user", content="q")], None) is None


def test_question_list_is_capped():
    msgs = [Message(role="user", content="질문 " + str(i) + " " + "가" * 300) for i in range(40)]
    qs = cs._question_lines(msgs, [])
    assert len(qs) == cs.MAX_QUESTIONS and all(len(q) <= cs.MAX_QUESTION_CHARS for q in qs)
    assert qs[-1].startswith("질문 39")
