"""예약 일정 규칙 — 구조화된 일정 값 검증, 다음 실행 시각 계산, 사람이 읽는 설명.

LLM 에 cron 문자열을 쓰게 하지 않는다. 요일 번호(0=일요일? 월요일?)·필드 순서 실수가 그대로 엉뚱한 시각의
실행이 되고, 사람도 카드에서 알아보지 못한다. 도구는 아래 다섯 가지 모양만 받고 서버가 시각을 계산한다
(ChatGPT 작업은 RRULE, Claude 루틴·LangGraph 는 cron+IANA 시간대 — 어느 쪽이든 사용자 시간대로 받는다).

    {"freq": "daily",    "time": "09:40"}
    {"freq": "weekdays", "time": "09:40"}                       # 월~금
    {"freq": "weekly",   "time": "09:40", "weekdays": [0, 2]}   # 0=월 … 6=일 (datetime.weekday)
    {"freq": "monthly",  "time": "09:40", "monthday": 31}       # 그 달에 없는 날이면 말일
    {"freq": "once",     "at": "2026-10-05T09:40"}              # 시간대 기준 현지 시각

시각은 모두 naive UTC 로 주고받는다(이 스키마의 다른 표와 같다). 공휴일은 고려하지 않는다.
"""
from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_TZ = "Asia/Seoul"
FREQS = ("daily", "weekdays", "weekly", "monthly", "once")
_WEEKDAY_KO = "월화수목금토일"
_TIME_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")
# 다음 실행을 찾을 때 훑는 최대 일수. 매월 31일(2월에도 말일로 잡힌다) 조차 한 달 안에 걸리므로 넉넉하다.
_SCAN_DAYS = 400


def utcnow() -> datetime:
    """지금(naive UTC) — 이 스키마의 시각 표기."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ScheduleError(ValueError):
    """일정 값이 틀렸다. 메시지는 모델이 고쳐 다시 부를 수 있게 무엇이 틀렸는지 말한다."""


def zone(name: Optional[str]) -> ZoneInfo:
    try:
        return ZoneInfo(name or DEFAULT_TZ)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ScheduleError(f"알 수 없는 시간대입니다: {name!r} (예: Asia/Seoul)") from e


def _parse_time(raw: Any) -> time:
    m = _TIME_RE.match(str(raw or "").strip())
    if not m:
        raise ScheduleError(f"time 은 24시간 'HH:MM' 이어야 합니다(예: '09:40'). 받은 값: {raw!r}")
    return time(int(m.group(1)), int(m.group(2)))


def normalize(spec: Any) -> dict:
    """도구 인자로 받은 일정을 검증하고 저장할 모양으로 정리한다. 틀리면 ScheduleError."""
    if not isinstance(spec, dict):
        raise ScheduleError("schedule 은 객체여야 합니다(예: {\"freq\": \"weekdays\", \"time\": \"09:40\"}).")
    freq = str(spec.get("freq") or "").strip().lower()
    if freq not in FREQS:
        raise ScheduleError(f"freq 는 {', '.join(FREQS)} 중 하나입니다. 받은 값: {spec.get('freq')!r}")
    if freq == "once":
        raw = str(spec.get("at") or "").strip()
        try:
            at = datetime.fromisoformat(raw)
        except ValueError as e:
            raise ScheduleError(f"once 는 at='YYYY-MM-DDTHH:MM'(현지 시각)이 필요합니다. 받은 값: {raw!r}") from e
        return {"freq": "once", "at": at.replace(tzinfo=None, second=0, microsecond=0).isoformat(timespec="minutes")}
    out: dict = {"freq": freq, "time": _parse_time(spec.get("time")).strftime("%H:%M")}
    if freq == "weekly":
        days = spec.get("weekdays")
        if not isinstance(days, list) or not days:
            raise ScheduleError("weekly 는 weekdays(0=월 … 6=일) 목록이 필요합니다(예: [0, 2]).")
        try:
            uniq = sorted({int(d) for d in days})
        except (TypeError, ValueError) as e:
            raise ScheduleError(f"weekdays 는 0~6 정수 목록입니다. 받은 값: {days!r}") from e
        if any(d < 0 or d > 6 for d in uniq):
            raise ScheduleError(f"weekdays 는 0(월)~6(일)입니다. 받은 값: {days!r}")
        out["weekdays"] = uniq
    if freq == "monthly":
        try:
            md = int(spec.get("monthday"))
        except (TypeError, ValueError) as e:
            raise ScheduleError("monthly 는 monthday(1~31)가 필요합니다.") from e
        if not 1 <= md <= 31:
            raise ScheduleError(f"monthday 는 1~31 입니다. 받은 값: {md}")
        out["monthday"] = md
    return out


def _matches(spec: dict, d: date) -> bool:
    freq = spec["freq"]
    if freq == "daily":
        return True
    if freq == "weekdays":
        return d.weekday() < 5
    if freq == "weekly":
        return d.weekday() in spec["weekdays"]
    if freq == "monthly":
        last = calendar.monthrange(d.year, d.month)[1]
        return d.day == min(spec["monthday"], last)
    return False


def _to_utc(local: datetime, tz: ZoneInfo) -> datetime:
    return local.replace(tzinfo=tz).astimezone(timezone.utc).replace(tzinfo=None)


def next_run_after(spec: dict, tz_name: Optional[str], after_utc: datetime) -> Optional[datetime]:
    """``after_utc``(naive UTC) **이후** 첫 실행 시각(naive UTC). 더 없으면(지난 한 번짜리) None."""
    tz = zone(tz_name)
    if spec["freq"] == "once":
        at = _to_utc(datetime.fromisoformat(spec["at"]), tz)
        return at if at > after_utc else None
    at_time = _parse_time(spec["time"])
    start = after_utc.replace(tzinfo=timezone.utc).astimezone(tz).date()
    for i in range(_SCAN_DAYS):
        d = start + timedelta(days=i)
        if not _matches(spec, d):
            continue
        cand = _to_utc(datetime.combine(d, at_time), tz)
        if cand > after_utc:
            return cand
    return None


def describe(spec: dict) -> str:
    """사람이 읽는 일정 — 승인 카드·목록·알림에 그대로 쓴다."""
    freq = spec.get("freq")
    if freq == "once":
        return f"{spec['at'].replace('T', ' ')} 한 번"
    t = spec.get("time", "")
    if freq == "daily":
        return f"매일 {t}"
    if freq == "weekdays":
        return f"평일(월~금) {t}"
    if freq == "weekly":
        return f"매주 {'·'.join(_WEEKDAY_KO[d] for d in spec.get('weekdays') or [])}요일 {t}"
    if freq == "monthly":
        return f"매월 {spec.get('monthday')}일 {t}" + (" (없는 달은 말일)" if spec.get("monthday", 0) > 28 else "")
    return str(spec)


def local_text(at_utc: Optional[datetime], tz_name: Optional[str]) -> str:
    """naive UTC → 'YYYY-MM-DD(요일) HH:MM' 현지 표기. None 이면 빈 문자열."""
    if at_utc is None:
        return ""
    local = at_utc.replace(tzinfo=timezone.utc).astimezone(zone(tz_name))
    return f"{local:%Y-%m-%d}({_WEEKDAY_KO[local.weekday()]}) {local:%H:%M}"
