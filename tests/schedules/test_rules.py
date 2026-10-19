"""예약 일정 규칙 — 검증·다음 실행 시각·설명. 시각은 naive UTC, 기본 시간대 Asia/Seoul(UTC+9)."""
from __future__ import annotations

from datetime import datetime

import pytest

from agent_arch.services.schedules import rules as R

KST = "Asia/Seoul"


def utc(*a) -> datetime:
    return datetime(*a)


def test_weekdays_skips_weekend():
    spec = R.normalize({"freq": "weekdays", "time": "09:40"})
    # 2026-10-02(금) 10:00 KST = 01:00 UTC → 다음은 10-05(월) 09:40 KST = 00:40 UTC
    assert R.next_run_after(spec, KST, utc(2026, 10, 2, 1, 0)) == utc(2026, 10, 5, 0, 40)
    # 금요일 09:00 KST(00:00 UTC) 면 그날 09:40
    assert R.next_run_after(spec, KST, utc(2026, 10, 2, 0, 0)) == utc(2026, 10, 2, 0, 40)


def test_daily_strictly_after():
    spec = R.normalize({"freq": "daily", "time": "9:05"})
    assert spec["time"] == "09:05"
    at = utc(2026, 10, 2, 0, 5)                       # 정확히 그 시각이면 다음 날
    assert R.next_run_after(spec, KST, at) == utc(2026, 10, 3, 0, 5)


def test_weekly_days_and_kst_date_boundary():
    spec = R.normalize({"freq": "weekly", "time": "08:00", "weekdays": [2, 0, 2]})
    assert spec["weekdays"] == [0, 2]                 # 정렬·중복 제거
    # 2026-10-04(일) 23:30 UTC = 10-05(월) 08:30 KST → 월요일 08:00 은 지났다 → 수요일 08:00 KST
    assert R.next_run_after(spec, KST, utc(2026, 10, 4, 23, 30)) == utc(2026, 10, 6, 23, 0)


def test_monthly_31_falls_back_to_month_end():
    spec = R.normalize({"freq": "monthly", "time": "09:00", "monthday": 31})
    # 2027-02 는 28일까지 → 2027-02-28 09:00 KST
    assert R.next_run_after(spec, KST, utc(2027, 2, 1)) == utc(2027, 2, 28, 0, 0)
    assert "말일" in R.describe(spec)


def test_once_past_returns_none():
    spec = R.normalize({"freq": "once", "at": "2026-10-05T09:40"})
    assert R.next_run_after(spec, KST, utc(2026, 10, 1)) == utc(2026, 10, 5, 0, 40)
    assert R.next_run_after(spec, KST, utc(2026, 10, 6)) is None


@pytest.mark.parametrize("spec, needle", [
    ({"freq": "hourly", "time": "09:00"}, "freq"),
    ({"freq": "daily", "time": "25:00"}, "HH:MM"),
    ({"freq": "daily"}, "HH:MM"),
    ({"freq": "weekly", "time": "09:00"}, "weekdays"),
    ({"freq": "weekly", "time": "09:00", "weekdays": [7]}, "0(월)~6(일)"),
    ({"freq": "monthly", "time": "09:00", "monthday": 0}, "1~31"),
    ({"freq": "once", "at": "내일"}, "YYYY-MM-DD"),
    ("매일 9시", "객체"),
])
def test_normalize_rejects_with_actionable_message(spec, needle):
    with pytest.raises(R.ScheduleError) as e:
        R.normalize(spec)
    assert needle in str(e.value)


def test_describe_and_local_text():
    assert R.describe({"freq": "weekdays", "time": "09:40"}) == "평일(월~금) 09:40"
    assert R.describe({"freq": "weekly", "time": "08:00", "weekdays": [0, 2]}) == "매주 월·수요일 08:00"
    assert R.local_text(utc(2026, 10, 5, 0, 40), KST) == "2026-10-05(월) 09:40"
    assert R.local_text(None, KST) == ""


def test_unknown_timezone():
    with pytest.raises(R.ScheduleError):
        R.next_run_after({"freq": "daily", "time": "09:00"}, "Mars/Base", utc(2026, 1, 1))
