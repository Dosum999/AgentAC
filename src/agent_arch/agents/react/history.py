"""최근 대화 발췌 — 판단이 필요한 자리(라우터·검증기)에 "앞 턴이 무슨 얘기였나"를 짧게 준다.

두 번의 실측이 같은 결함을 보였다(2026-09-15~16). 워커는 대화 이력을 전부 받아 "Kim" 이 앞 턴의 연구자라는 걸 알지만,
그 앞에서 도메인을 정하는 라우터와 답을 판정하는 검증기는 이번 턴 문장만 받았다. 정보가 있는 자리(워커)에는 권한이 없고
결정하는 자리(라우터·검증기)에는 정보가 없어서, "가격 관련해서 말이야"가 되묻기로 뒤집히고 "Kim 관련된거 찾아봐"가
사내 사람 검색으로 새어 워커 4개가 돌았다.

기본은 넉넉하게(직전 3턴, 발화당 600자) — 사용자 결정: "기본으로 길게 잡고 필요하면 더 길게". 도구 호출·도구 결과·검증
보강 지시·보강 전 초안은 대화가 아니므로 뺀다. 어시스턴트 답은 앞부분만 잘라 넣어 끝의 출처 각주는 자연히 빠진다.
"""
from __future__ import annotations

from typing import Any

FEEDBACK_NAME = "__verify_feedback__"
DEFAULT_TURNS = 3
DEFAULT_CHARS = 600
_TOOL_CTX_PREFIX = "[도구 결과]"


def recent_history(messages: Any, *, max_turns: int = DEFAULT_TURNS, max_chars: int = DEFAULT_CHARS) -> str:
    """이번 턴 이전의 사용자·어시스턴트 발화 최근 몇 턴. 마지막 user(이번 턴 요청)는 제외한다 — 별도 항목으로 들어간다."""
    convo: list[tuple[str, str]] = []
    for m in messages or []:
        role = getattr(m, "role", None)
        content = getattr(m, "content", "") or ""
        if role == "user" and getattr(m, "question", None):
            content = m.question  # 첨부가 붙은 턴은 사용자가 직접 쓴 질문만 본다
        if role not in ("user", "assistant"):
            continue
        if getattr(m, "name", None) == FEEDBACK_NAME:
            if convo and convo[-1][0] == "assistant":
                convo.pop()                                   # 보강 전 초안은 대화가 아니다
            continue
        if getattr(m, "tool_calls", None) or (role == "user" and content.startswith(_TOOL_CTX_PREFIX)):
            continue
        if isinstance(content, str) and content.strip():
            convo.append((role, content))
    if convo and convo[-1][0] == "user":
        convo.pop()                                           # 이번 턴 요청
    tail = convo[-(2 * max_turns):] if max_turns > 0 else []
    return "\n".join(f"{'사용자' if r == 'user' else '어시스턴트'}: {c.strip()[:max_chars]}" for r, c in tail)


def recent_user_turns(messages: Any, *, max_turns: int = DEFAULT_TURNS, max_chars: int = DEFAULT_CHARS) -> list[str]:
    """이번 턴 이전의 **사용자** 발화 최근 몇 턴(오래된 것부터). 도구가 모델 출력의 근거를 확인할 때 쓴다.

    recent_history 와 같은 필터(도구 결과·검증 피드백·이번 턴 요청 제외). 정정 턴은 참석자·결재자를 히스토리에서
    다시 넘기므로 이번 발화만 보면 근거가 없다(2026-09-28 참석자 근거 검증).
    """
    users: list[str] = []
    last_user_pending = False
    for m in messages or []:
        role = getattr(m, "role", None)
        content = getattr(m, "content", "") or ""
        if role == "user" and getattr(m, "question", None):
            content = m.question  # 첨부가 붙은 턴은 사용자가 직접 쓴 질문만 본다
        if role not in ("user", "assistant") or getattr(m, "name", None) == FEEDBACK_NAME:
            continue
        if getattr(m, "tool_calls", None) or (role == "user" and content.startswith(_TOOL_CTX_PREFIX)):
            continue
        if role == "user" and isinstance(content, str) and content.strip():
            users.append(content.strip()[:max_chars])
            last_user_pending = True
        elif role == "assistant":
            last_user_pending = False
    if last_user_pending and users:
        users.pop()                                           # 마지막 user 는 이번 턴 요청
    return users[-max_turns:] if max_turns > 0 else []
