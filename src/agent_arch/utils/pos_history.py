"""POS 대시보드 턴의 저장 시 정화 — 대화 이력 오염 방지 임시 가드.

probotics POS 대시보드는 전용 엔드포인트 없이 범용 `/agent/stream`에 카탈로그·판단 규칙·
출력 계약 전문을 태그 프롬프트로 실어 보낸다(실측 2026-08-28 HAR). 그대로 저장하면:

- query 에 10KB 카탈로그 원문 + 대화방 제목이 ``[POS Agent Catalog] {...`` 로 파생
- answer 에 ``[POS_QUERY_PLAN]``/``[POS_ACTION_PLAN]`` 기계 태그 노출
- 질의 1건당 2턴(플래닝+판단)이 저장돼 같은 질문이 이력에 중복

이 모듈은 저장 직전(save_logged_in_turn)에 POS 형식 턴만 골라 ``[사용자 질문]`` 블록과
태그 없는 자연어 답변만 남긴다. POS 전용 API(`/pos/chat`) 전환이 완료되면 제거 예정.

순수 함수만 담는다(DB/Redis/로그 외 I/O 없음) — 훅과 독립적으로 단위 테스트 가능해야 한다.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# POS 프롬프트 식별 접두 — 1턴(플래닝)은 [POS Agent Catalog], 2턴(판단)은 [POS Query Plan]
# 으로 시작한다(HAR 실측). startswith 가드라 스프레드시트 첨부 등으로 앞에 다른 텍스트가
# 붙으면 감지가 조용히 빠지는데(fail-open, 원문 저장), POS 대시보드는 파일을 보내지 않으므로
# 실질 비발생이고, 일반 대화를 POS 로 오인하는 것(fail-close)보다 안전한 방향이다.
_POS_INPUT_PREFIXES = ("[POS Agent Catalog]", "[POS Query Plan]")

# pos_raw(정화 전 원문) 보관 상한. metadata JSONB 가 10KB+ 로 TOAST 되면
# metadata 를 읽는 집계(list_daily_usage 의 ->>'company' 등)마다 detoast 비용이 붙어
# 원문을 통째로 두되 상한만 건다 — 실측 페이로드가 12KB 내외라 정상 턴은 절단되지 않는다.
_POS_RAW_MAX = 30_000

# [사용자 질문] 섹션 추출. 경계는 "행 시작의 [섹션명] 단독 라인"만 인정한다 —
# 질문 본문 안의 인라인 대괄호([긴급] 등)는 행 전체가 헤더 형태가 아니므로 경계가 아니고,
# 다른 섹션 값의 멀티라인 JSON 은 [사용자 질문] 뒤에 오지 않는 한 무관하다(HAR 형식).
_QUESTION_RE = re.compile(
    r"^\[사용자 질문\][ \t]*\r?\n"
    r"(?P<q>.*?)"
    r"(?=\s*^\[[^\[\]\r\n]+\][ \t]*\r?$|\s*\Z)",
    re.MULTILINE | re.DOTALL,
)

# 기계 태그 짝 블록 — 역참조(\1)로 여는/닫는 태그 이름 일치를 강제, 비탐욕이라
# 같은 태그가 여러 번 나와도 블록 단위로 제거된다. POS_ 접두 한정이라 답변 본문의
# 일반 대괄호([참고] 등)는 절대 건드리지 않는다.
_TAG_PAIR_RE = re.compile(r"\[(POS_[A-Z0-9_]+)\].*?\[/\1\]", re.DOTALL)

# 짝 제거 후 남은 여는 태그부터 끝까지 — 스트림 중단(status 499, partial_text 저장) 시
# 닫는 태그 없이 잘린 꼬리가 실제로 발생하는 경로라(orchestrator 중단 저장), 미종결
# 태그는 정의상 기계 페이로드의 시작이므로 끝까지 제거하는 쪽이 이력 오염 방지 목적에 맞다.
_TAG_TAIL_RE = re.compile(r"\[POS_[A-Z0-9_]+\].*\Z", re.DOTALL)

# 꼬리 제거 후에도 남는 고아 태그 감지용(중간 위치의 닫는 태그 단독 등) — 제거하지 않고
# 경고만 남긴다. 여기까지 오면 형식이 계약을 벗어난 것이라 추측 삭제가 더 위험하다.
_TAG_ORPHAN_RE = re.compile(r"\[/?POS_[A-Z0-9_]+\]")


@dataclass(frozen=True)
class PosSanitizedTurn:
    """정화 결과. query 는 비어 있지 않음이 보장된다(빈 추출은 sanitize 가 None 반환)."""

    query: str  # [사용자 질문] 블록에서 추출한 사용자 발화 (strip 완료)
    answer: str  # 기계 태그 블록을 제거한 답변 ("" 이면 태그뿐이었던 플래닝 턴)
    internal: bool  # True = 플래닝 턴 — 이력 조회에서 숨기고 Redis 재주입도 스킵


def is_pos_input(user_input: str) -> bool:
    """입력이 POS 대시보드 태그 프롬프트 형식인지 판별한다."""
    return bool(user_input) and user_input.startswith(_POS_INPUT_PREFIXES)


def extract_user_question(user_input: str) -> Optional[str]:
    """POS 프롬프트에서 ``[사용자 질문]`` 블록의 본문을 추출한다.

    Returns:
        추출·strip 된 질문 문자열. 섹션이 없거나 본문이 비어 있으면 None —
        호출자는 정화를 포기하고 원문을 그대로 저장해야 한다(반쪽 정화 금지).
    """
    m = _QUESTION_RE.search(user_input)
    if m is None:
        return None
    question = m.group("q").strip()
    return question or None


def strip_pos_tag_blocks(answer: str) -> str:
    """답변에서 ``[POS_X]...[/POS_X]`` 기계 태그 블록을 제거한다.

    짝 블록 제거 → 미종결 꼬리 제거 → 공백 정리 순. 그래도 남는 고아 태그는
    경고 로그만 남기고 유지한다(계약 밖 형식에서의 추측 삭제 방지).
    """
    cleaned = _TAG_PAIR_RE.sub("", answer)
    cleaned = _TAG_TAIL_RE.sub("", cleaned)
    if _TAG_ORPHAN_RE.search(cleaned):
        logger.warning("POS 태그 정화 후 고아 태그 잔존 — 형식이 계약을 벗어남: %.200s", cleaned)
    return cleaned.strip()


def sanitize_pos_turn(
    user_input: str, assistant_output: str
) -> Optional[PosSanitizedTurn]:
    """POS 형식 턴이면 저장용 정화본을 만든다.

    Args:
        user_input: 저장 직전의 사용자 입력(= 오케스트레이터 ctx.input, PII 마스킹 후).
        assistant_output: 저장 직전의 어시스턴트 응답 원문.

    Returns:
        PosSanitizedTurn — POS 형식이고 질문 추출에 성공한 경우.
        None — POS 형식이 아니거나 질문 추출 실패(경고 로그). 호출자는 무변경 저장.
        query/answer 를 함께 반환해 원자적으로 치환하게 한다 — query 만 정화되고
        answer 는 원문인 반쪽 상태를 만들지 않기 위함.
    """
    if not is_pos_input(user_input):
        return None

    question = extract_user_question(user_input)
    if question is None:
        logger.warning(
            "POS 프롬프트에서 [사용자 질문] 추출 실패 — 원문 그대로 저장: %.200s",
            user_input,
        )
        return None

    answer = strip_pos_tag_blocks(assistant_output or "")
    return PosSanitizedTurn(query=question, answer=answer, internal=(answer == ""))
