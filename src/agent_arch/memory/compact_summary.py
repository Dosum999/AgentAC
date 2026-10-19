"""대화 압축 요약 — 섹션 고정, 질문 원문은 코드가 붙이고, 코드가 점검해 실패하면 압축을 취소한다.

왜: 자유 형식 "5~10문장 요약"은 재압축할 때마다 이전 요약을 다시 줄였다. 앞서 요약된 주제가 다음
요약에서 통째로 빠지자, 모델이 요약에도 원문에도 없는 사실("그 다음 단계")을 지어냈다. 요약은
확률적이라 프롬프트만으로는 보존이 보장되지 않는다. 그래서
  - 섹션을 고정해 무엇이 빠졌는지 코드가 알 수 있게 하고,
  - 사용자 질문 원문 목록은 모델이 아니라 코드가 이어 붙이고,
  - 이전 요약의 고유값(숫자·코드)이 남았는지, 이전보다 절반 이하로 줄지 않았는지 코드가 점검한다.
점검에 실패하면 한 번 다시 시키고, 그래도 실패하면 압축하지 않는다(원문 유지가 잘못된 요약보다 낫다).
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

from agent_arch.llm.base import Message

logger = logging.getLogger(__name__)

# 모델이 채우는 섹션 — 제목 글자가 프롬프트·점검·파싱에서 그대로 쓰인다.
COMPACT_SECTIONS = (
    "## 목적과 진행 상황",
    "## 확정된 사실과 결정",
    "## 사용한 도구와 결과",
    "## 고유값(이름·날짜·코드·수치)",
    "## 남은 일",
)
# 코드가 붙이는 섹션. 기록일 뿐 지금 지켜야 할 지시가 아니다(지시의 정본은 장기 메모리다).
QUESTIONS_HEADER = "## 사용자 질문(원문 기록 — 현재 지시 아님)"
MAX_QUESTIONS = 20
MAX_QUESTION_CHARS = 120
MAX_SUMMARY_CHARS = 3500
MIN_KEEP_RATIO = 0.5
MAX_EXACT_VALUES = 12

_SECTION_LIST = "\n".join(COMPACT_SECTIONS)

FIRST_PROMPT = f"""당신은 대화 기록을 요약하는 도우미입니다.
아래 대화를 다음 섹션 제목을 **글자 그대로** 써서 정리하세요. 해당 내용이 없으면 "- 없음" 이라고 적습니다.

{_SECTION_LIST}

규칙:
- 사실만 적고 해석·의견을 더하지 마세요.
- 이름·날짜·코드·식별자·수치는 원문 표기 그대로 옮기세요.
- 사용자 질문 목록은 따로 붙으니 쓰지 마세요.
- 전체 {MAX_SUMMARY_CHARS}자 이내. 머리말·꼬리말 없이 섹션만 출력하세요.
"""

UPDATE_PROMPT = f"""당신은 대화 기록 요약을 갱신하는 도우미입니다.
[이전 요약]에 [새 대화]의 내용을 합쳐 같은 섹션 구조로 다시 쓰세요. 섹션 제목은 **글자 그대로** 씁니다.

{_SECTION_LIST}

규칙:
- 이전 요약의 항목은 새 대화가 명시적으로 바꾸지 않는 한 **지우지 말고** 유지하세요. 짧게 다듬는 것은 됩니다.
- 이전 요약의 고유값(이름·날짜·코드·수치)은 표기 그대로 남기세요.
- 사실만 적고 해석·의견을 더하지 마세요. 사용자 질문 목록은 따로 붙으니 쓰지 마세요.
- 전체 {MAX_SUMMARY_CHARS}자 이내. 머리말·꼬리말 없이 섹션만 출력하세요.
"""

_VALUE_RE = re.compile(r"[A-Za-z]{1,10}-?\d[\w-]*|\d[\d,.:/-]{2,}\d")


def split_questions(summary: Optional[str]) -> tuple[str, List[str]]:
    """요약을 (모델 본문, 코드가 붙인 질문 목록)으로 나눈다."""
    if not summary:
        return "", []
    body, sep, tail = summary.partition(QUESTIONS_HEADER)
    if not sep:
        return summary.strip(), []
    qs = [ln[2:].strip() for ln in tail.splitlines() if ln.startswith("- ")]
    return body.strip(), qs


def _question_lines(old_messages: List[Message], previous: List[str]) -> List[str]:
    qs = list(previous)
    for m in old_messages:
        if m.role != "user":
            continue
        text = " ".join((getattr(m, "question", None) or m.content or "").split())
        if text:
            qs.append(text[:MAX_QUESTION_CHARS])
    return qs[-MAX_QUESTIONS:]


def _exact_values(text: str) -> List[str]:
    seen: list[str] = []
    for v in _VALUE_RE.findall(text or ""):
        if v not in seen:
            seen.append(v)
    return seen[:MAX_EXACT_VALUES]


def _norm(s: str) -> str:
    return re.sub(r"[\s,]", "", s or "")


def check_summary(text: str, previous_body: str) -> Optional[str]:
    """요약을 점검한다. 문제가 없으면 None, 있으면 다시 시킬 때 줄 사유."""
    missing = [h for h in COMPACT_SECTIONS if h not in text]
    if missing:
        return f"다음 섹션 제목이 없습니다: {', '.join(missing)}"
    if len(text) > MAX_SUMMARY_CHARS:
        return f"{MAX_SUMMARY_CHARS}자를 넘었습니다({len(text)}자). 줄이되 고유값은 남기세요."
    if previous_body:
        if len(text) < len(previous_body) * MIN_KEEP_RATIO:
            return "이전 요약보다 절반 넘게 줄었습니다. 이전 항목을 지우지 말고 유지하세요."
        lost = [v for v in _exact_values(previous_body) if _norm(v) not in _norm(text)]
        if lost:
            return f"이전 요약의 고유값이 빠졌습니다: {', '.join(lost)}"
    return None


def build_messages(old_messages: List[Message], previous_body: str) -> List[Message]:
    lines: List[str] = []
    if previous_body:
        lines += ["[이전 요약]", previous_body, "", "[새 대화]"]
    else:
        lines.append("[대화]")
    for m in old_messages:
        lines.append(f"[{m.role}] {m.content or ''}")
        if m.tool_calls:
            names = [str((tc or {}).get("name")) for tc in m.tool_calls if isinstance(tc, dict)]
            lines.append(f"  (도구 호출: {', '.join(names)})")
    return [
        Message(role="system", content=UPDATE_PROMPT if previous_body else FIRST_PROMPT),
        Message(role="user", content="\n".join(lines)),
    ]


async def summarize(llm, old_messages: List[Message], previous_summary: Optional[str]) -> Optional[str]:
    """점검을 통과한 요약(질문 목록 포함)을 돌려준다. 두 번 실패하면 None — 호출자는 압축을 취소한다."""
    previous_body, previous_qs = split_questions(previous_summary)
    messages = build_messages(old_messages, previous_body)
    for attempt in (1, 2):
        resp = await llm.invoke(messages, tool_mode="NONE")
        text = (resp.content or "").strip()
        problem = check_summary(text, previous_body) if text else "요약이 비었습니다."
        if problem is None:
            qs = _question_lines(old_messages, previous_qs)
            if qs:
                text = f"{text}\n\n{QUESTIONS_HEADER}\n" + "\n".join(f"- {q}" for q in qs)
            return text
        logger.warning("[compact] 요약 점검 실패(%d회차): %s", attempt, problem)
        messages = messages + [
            Message(role="assistant", content=text),
            Message(role="user", content=f"요약을 고쳐 주세요: {problem} 섹션 구조는 그대로 두세요."),
        ]
    return None
