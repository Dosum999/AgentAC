"""[현재 날짜/시간] 블록 — 상대 날짜를 '계산' 아닌 '조회'로 만드는지 검증.

회귀(실측 2026-08-10): 압축 목록("- 다음 주: 월 08-17 · 화 08-18 · 수 08-19 · …")이라
사용자 발화("다음주 수요일")와 글자가 일치하는 항목이 없었고, thinking 이 꺼진 프로바이더가
조회 대신 암산으로 새어 요일을 1-base 로 세는 off-by-one 을 냈다(→ 08-20(목)).
문구를 그대로 키로 박아 순수 문자열 조회가 되도록 바꿨다.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from agent_arch.agents.prompts import system as sys_mod

_KST = timezone(timedelta(hours=9))
_KO = ("월", "화", "수", "목", "금", "토", "일")


def _ctx_on(d: date) -> str:
    """해당 날짜(KST 09:00)를 '오늘'로 두고 블록을 렌더한다."""
    fixed = datetime(d.year, d.month, d.day, 9, 0, tzinfo=_KST)

    class _FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed

    with patch.object(sys_mod, "datetime", _FixedDatetime):
        return sys_mod._build_datetime_context()


# 2026-08-10(월) 주간을 기준으로 요일별 전 케이스를 돈다 — '오늘이 무슨 요일이든' 성립해야 한다.
@pytest.mark.parametrize("offset", range(7))
def test_every_weekday_phrase_is_present_and_correct(offset: int):
    today = date(2026, 8, 10) + timedelta(days=offset)
    ctx = _ctx_on(today)
    this_mon = today - timedelta(days=today.weekday())

    for label, mon in (("이번주", this_mon), ("다음주", this_mon + timedelta(days=7))):
        for i in range(7):
            d = mon + timedelta(days=i)
            line = f"- {label} {_KO[i]}요일: {d.strftime('%Y-%m-%d')} ({_KO[i]})"
            assert line in ctx, f"'{label} {_KO[i]}요일' 항목 누락/불일치 (today={today})"


def test_reported_offbyone_case_is_pinned():
    """실측 사고 재현 케이스: 2026-08-10(월) 기준 '다음주 수요일'은 08-19 이고 08-20 이 아니다."""
    ctx = _ctx_on(date(2026, 8, 10))
    assert "- 다음주 수요일: 2026-08-19 (수)" in ctx
    assert "- 다음주 금요일: 2026-08-21 (금)" in ctx   # 운영기에서 08-14 로 샜던 케이스
    assert "- 이번주 수요일: 2026-08-12 (수)" in ctx


def test_lookup_not_arithmetic_instruction():
    """모델에게 '찾아서 그대로 쓰라'고 지시하고, 실행 경로에도 동일 적용임을 명시한다."""
    ctx = _ctx_on(date(2026, 8, 10))
    assert "계산하지 마세요" in ctx
    assert "실행" in ctx.split("상대 날짜")[-1]      # 조회/실행 모두 적용 문구


def test_year_included_so_model_cannot_guess():
    """연도 없는 'MM-DD' 는 연말 경계에서 모델이 연도를 추측하게 만든다 — 항상 YYYY 포함."""
    ctx = _ctx_on(date(2026, 12, 28))               # 다음주가 2027 년으로 넘어가는 주
    assert "- 이번주 목요일: 2026-12-31 (목)" in ctx
    assert "- 다음주 금요일: 2027-01-08 (금)" in ctx
