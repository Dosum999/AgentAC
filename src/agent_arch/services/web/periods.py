"""웹 결과 본문에 적힌 **가장 최근 연·월** — 모델이 최신 출처를 고르고, 검증기가 "최신은 8월" 같은 단정을 대조하게 한다.

실측(2026-09-23): EPDM 단가 턴에서 모델이 IMARC 원 페이지(본문 앞머리에 "September 2026" 표)를 열고도 함께 받은 openPR 의
8월 보도자료에 끌려 "9월치는 아직 발표되지 않았다"고 답했고 검증기도 통과시켰다. 결과마다 시점을 표시해 두면 어느 출처가
최신인지 모델이 바로 본다. 오늘(KST)보다 뒤의 연·월은 전망치("2035년까지")라 세지 않는다.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional

_MONTHS = {m: i for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"], 1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_EN_RE = re.compile(r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?,?\s+(?:\d{1,2},?\s+)?(20\d{2})\b", re.I)
_KO_RE = re.compile(r"(20\d{2})\s*년\s*(\d{1,2})\s*월")
_ISO_RE = re.compile(r"\b(20\d{2})[-./](\d{1,2})(?:[-./]\d{1,2})?\b")


def _today() -> tuple[int, int]:
    now = datetime.now(timezone(timedelta(hours=9)))
    return now.year, now.month


def latest_period(text: Optional[str], *, today: Optional[tuple[int, int]] = None) -> Optional[str]:
    """본문의 연·월 표기 중 오늘 이하에서 가장 최근 것을 'YYYY-MM' 으로. 없으면 None."""
    if not text:
        return None
    limit = today or _today()
    best: Optional[tuple[int, int]] = None
    cands: list[tuple[int, int]] = []
    for m in _EN_RE.finditer(text):
        cands.append((int(m.group(2)), _MONTHS[m.group(1).lower()]))
    for m in _KO_RE.finditer(text):
        cands.append((int(m.group(1)), int(m.group(2))))
    for m in _ISO_RE.finditer(text):
        cands.append((int(m.group(1)), int(m.group(2))))
    for y, mo in cands:
        if 1 <= mo <= 12 and (y, mo) <= limit and (best is None or (y, mo) > best):
            best = (y, mo)
    return f"{best[0]:04d}-{best[1]:02d}" if best else None


# ─── 시간대 환산 ────────────────────────────────────────────────────────────────
# 실측(2026-09-28 09:08 KST): Bloomberg 원 페이지의 "20:02 EDT 09/27" 은 한국 시각 오늘 09:02 인데, 모델이 미국 날짜 9/27 을
# 그대로 읽고 "오늘 값 없음"이라 답했다(XE "9/27 16:31 UTC" = 오늘 01:31 KST 도 같은 오판). 날짜와 함께 적힌 시각만 KST 로
# 환산해 바로 뒤에 덧붙인다 — 판단(오늘 값인가)은 모델이, 환산은 코드가.
_TZ_OFFSETS = {"UTC": 0, "GMT": 0, "Z": 0, "EDT": -4, "EST": -5, "CDT": -5, "CST": -6, "PDT": -7, "PST": -8,
               "BST": 1, "CET": 1, "CEST": 2, "JST": 9, "SGT": 8, "HKT": 8,
               # 동남아·오세아니아(Pluang "07:00 WIB" 실측). IST 는 인도·아일랜드·이스라엘이 겹쳐 넣지 않는다.
               "WIB": 7, "WITA": 8, "WIT": 9, "ICT": 7, "AEST": 10, "AEDT": 11, "MSK": 3}
_TZ_ALT = "|".join(sorted((k for k in _TZ_OFFSETS if k != "Z"), key=len, reverse=True))
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
# 날짜 표기(앞·뒤 어느 쪽이든): 2026-09-27 / 09/27/26·09/27/2026(미국식 월/일) / Sep 27, 2026 / 27 Sep 2026
_DATE_RE = re.compile(
    r"(?P<iso>(20\d{2})-(\d{1,2})-(\d{1,2}))|(?P<us>(\d{1,2})/(\d{1,2})/(\d{2,4}))|"
    r"(?P<mdy>" + _MON + r"\s+(\d{1,2}),?\s+(20\d{2}))|(?P<dmy>(\d{1,2})\s+" + _MON + r"\s+(20\d{2}))", re.I)
_TIME_RE = re.compile(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?\s*(am|pm)?\s*(" + _TZ_ALT + r")\b", re.I)
_KST = timezone(timedelta(hours=9))


def _parse_date(m) -> Optional[tuple[int, int, int]]:
    g = m.groupdict()
    try:
        if g["iso"]:
            return int(m.group(2)), int(m.group(3)), int(m.group(4))
        if g["us"]:
            y = int(m.group(8)); y = y + 2000 if y < 100 else y
            return y, int(m.group(6)), int(m.group(7))
        if g["mdy"]:
            return int(m.group(12)), _MONTHS[m.group(10).lower()[:3]], int(m.group(11))
        if g["dmy"]:
            return int(m.group(16)), _MONTHS[m.group(15).lower()[:3]], int(m.group(14))
    except (KeyError, ValueError, TypeError):
        return None
    return None


# ISO 8601 시각(구조화 데이터·메타 태그): 2026-09-25T20:59:45Z / 2026-09-28T08:30:00+09:00
_ISO_TS_RE = re.compile(r"(20\d{2})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(Z|[+-]\d{2}:?\d{2})")


def _annotate_iso(text: str) -> str:
    def rep(m):
        tz = m.group(6)
        off = 0 if tz == "Z" else (1 if tz[0] == "+" else -1) * (int(tz[1:3]) * 60 + int(tz[-2:]))
        if off == 9 * 60:
            return m.group(0)                                  # 이미 KST
        try:
            src = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)),
                           tzinfo=timezone(timedelta(minutes=off)))
        except ValueError:
            return m.group(0)
        return f"{m.group(0)} [= {src.astimezone(_KST):%Y-%m-%d %H:%M} KST]"
    return _ISO_TS_RE.sub(rep, text)


def annotate_timezones(text: Optional[str], *, window: int = 40, limit: int = 12) -> Optional[str]:
    """'시각 + 시간대'와 가까이 적힌 날짜를 찾아 '[= YYYY-MM-DD HH:MM KST]' 를 덧붙인다. 날짜가 없으면 그대로 둔다.

    ISO 8601 시각(…T20:59:45Z)도 환산한다 — TradingView 구조화 데이터의 priceValidUntil 이 금요일 종가 시각인데, 모델이 그 값을
    "오늘 09:26 기준"으로 답했다(2026-09-28 재현).
    """
    if not text:
        return text
    text = _annotate_iso(text)
    out, last, n = [], 0, 0
    for tm in _TIME_RE.finditer(text):
        if n >= limit:
            break
        lo, hi = max(0, tm.start() - window), min(len(text), tm.end() + window)
        dm = min(_DATE_RE.finditer(text, lo, hi), key=lambda d: abs(d.start() - tm.start()), default=None)
        date = _parse_date(dm) if dm else None
        if not date:
            continue
        hh, mm, ampm, tz = int(tm.group(1)), int(tm.group(2)), (tm.group(3) or "").lower(), tm.group(4).upper()
        if ampm == "pm" and hh < 12:
            hh += 12
        if ampm == "am" and hh == 12:
            hh = 0
        try:
            src = datetime(date[0], date[1], date[2], hh, mm, tzinfo=timezone(timedelta(hours=_TZ_OFFSETS[tz])))
        except (ValueError, KeyError):
            continue
        end = max(tm.end(), dm.end())
        out.append(text[last:end]); out.append(f" [= {src.astimezone(_KST):%Y-%m-%d %H:%M} KST]")
        last, n = end, n + 1
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


# ─── 요약용 발췌 ────────────────────────────────────────────────────────────────
# 실측(2026-09-28): 충분성 판정기·검증기에 결과를 행마다 "본문 앞 280~500자"로 넘겼더니, Business Analytiq 9월 지역별 표와
# smbs 공식 매매기준율(EUR=1,538.04)이 본문 뒤쪽에 있어 잘렸다. 판정기는 "9월 수치 없음"을 2라운드 내내 냈고, 검증기는 정답을
# "도구 결과에 없는 주장"으로 두 번 되돌렸다. 앞머리는 남기되 나머지 자리는 **수치가 있고 가장 최근 시점·질문 용어가 적힌 줄**에 준다.
_SEG_SPLIT_RE = re.compile(r"\n+|(?<=[.!?。])\s+|(?<=&)(?=[A-Z]{3}=)")
_NUM_RE = re.compile(r"\d[\d,]*\.\d+|\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?\s*%")
_TERM_RE = re.compile(r"[0-9A-Za-z가-힣]{2,}")
_HEAD_CHARS = 120


def salient_excerpt(text: Optional[str], limit: int, *, terms: Optional[str] = None,
                    today: Optional[tuple[int, int]] = None) -> str:
    """본문에서 앞머리 + 수치·최근 시점·질문 용어가 있는 조각을 원래 순서로 골라 ``limit`` 자 안으로. 짧으면 그대로."""
    s = " ".join((text or "").split()) if "\n" not in (text or "") else (text or "").strip()
    if len(s) <= limit:
        return " ".join(s.split())
    newest = latest_period(s, today=today)
    words = {w.casefold() for w in _TERM_RE.findall(terms or "")}
    head = " ".join(s[:_HEAD_CHARS].split())
    scored = []
    pos = 0
    for i, seg in enumerate(p for p in _SEG_SPLIT_RE.split(s[_HEAD_CHARS:]) if p and p.strip()):
        seg = " ".join(seg.split())[:240]
        score = 2 * min(3, len(_NUM_RE.findall(seg)))
        if not score:
            continue                                          # 수치 없는 문장은 앞머리 외엔 싣지 않는다
        if newest and latest_period(seg, today=today) == newest:
            score += 3
        score += sum(1 for w in words if w in seg.casefold())
        scored.append((score, i, seg))
    budget = limit - len(head) - 3
    picked = []
    for score, i, seg in sorted(scored, key=lambda x: (-x[0], x[1])):
        if len(seg) + 3 > budget:
            continue
        picked.append((i, seg)); budget -= len(seg) + 3
    return " … ".join([head] + [seg for _, seg in sorted(picked)])[:limit]
