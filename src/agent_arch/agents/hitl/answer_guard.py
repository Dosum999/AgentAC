"""HITL 답변 가드 — LLM 이 실제로 일어나지 않은 승인 액션을 텍스트로 환각하는 것을 차단.

react_graph 루프에서 승인 필요 도구(policy.APPROVAL_REQUIRED_TOOLS)는 ``confirm=False``
(초안)로만 실행되고, 실제 비가역 실행은 사용자 승인 후 ``/resume`` 에서 일어난다. 그때의
완료 문구도 ``policy.action_done_message`` 가 코드로 낸다(LLM 생성 아님). 또한 승인/거절
버튼(카드)은 ``pending_drafts`` 가 있을 때만 물리적으로 존재한다.

따라서 ``pending_drafts`` 가 비어 있는데 최종 답변이
  (a) "삭제/전송/… 했어요·되었습니다" 처럼 비가역 액션 **완료**를 주장하거나
  (b) "[승인]/[거절] 버튼을 눌러 주세요" 처럼 **없는 버튼**을 안내하면
둘 다 환각이다(도구를 실제로 호출하지 않은 것). 해당 문장을 교정 문구로 대체한다.

프롬프트(system.py)로도 막고 있으나 프롬프트는 확률적이라 새므로, 이 코드 가드가
결정적으로 차단한다. 프로바이더(gemini/gemma/openai) 무관하게 동작한다.

**판정 대상은 산문 줄이다** (실측 2026-08-26): 예전에는 문장 분리기가 ``\n`` 까지 문장 경계로
써서 마크다운의 모든 줄이 문장이 됐고, BOM 등록 현황 표의
``| G103 | 조향 너클 | - | EA | **등록 완료** |`` 행이 완료-주장으로 오인돼 저장본에서만
사라졌다(같은 표의 다른 행은 품명이 길어 ``_TERSE_MAX`` 를 넘긴 덕에 생존 — 글자 수 우연이
생사를 갈랐다). 표 셀의 "등록 완료"는 **데이터 상태 라벨**이지 액션 주장이 아니다. 그래서
지금은 표 행·코드펜스·헤딩을 판정에서 빼고, 목록 항목은 (조회 결과 나열이 대부분이라)
완료-주장 판정에서만 뺀다. 버튼 규칙은 승인 문맥을 요구해 오탐이 낮으므로 목록에도 적용한다.
재조립도 줄 구조를 보존한다 — 예전에는 제거가 한 번이라도 일어나면 답변 전체의 개행이
사라져 표·목록이 한 줄로 뭉개졌다.
"""
from __future__ import annotations

import json
import logging
import re
from urllib.parse import urlsplit

from agent_arch.core.config import settings
from agent_arch.utils.language import looks_off_target

logger = logging.getLogger(__name__)

# ── 답변 언어 감지 ───────────────────────────────────────────────────
# 판정 로직은 ``app.utils.language`` 가 단일 출처다 — 목표 언어(ko/en/zh/ja)를 받는
# ``looks_off_target`` 이 본체이고, 아래는 한국어 고정 시절의 호출부를 위한 별칭이다.


def looks_non_korean(text: str) -> bool:
    """답변이 사실상 한국어가 아닌지 판별한다 — ``looks_off_target(text, "ko")`` 별칭.

    한국어가 목표 언어인 경우의 특수화. 새 호출부는 턴의 목표 언어를 넘길 수 있는
    ``app.utils.language.looks_off_target`` 을 직접 쓴다.

    Args:
        text: 최종 답변 텍스트.

    Returns:
        한국어가 아닌 답변으로 판단되면 True.
    """
    return looks_off_target(text, "ko")


# 승인/거절 버튼 안내 — 카드(pending_drafts)가 있을 때만 정당하다.
# 일반 "버튼을 눌러/클릭" 문구는 그 자체로는 카드 환각이 아니다 — 웹메일 스팸 신고
# 안내(프롬프트 고정 사실)처럼 실제 화면 UI 의 사용법을 설명하는 정당한 문장이 있다
# (실측 2026-08-10: 스팸 신고 절차 문장이 저장본에서 잘려 스트리밍 화면과 히스토리가
# 어긋났다). 그래서 일반 버튼 문구는 같은 문장에 승인 문맥(승인·거절·결재·카드)이
# 함께 있을 때만 환각으로 본다.
_BUTTON_RE = re.compile(
    r"\[\s*승인\s*\]|\[\s*거절\s*\]|승인\s*버튼|거절\s*버튼|아래[^\n]{0,6}버튼"
)
_GENERIC_BUTTON_RE = re.compile(r"버튼을?\s*(?:눌러|누르|클릭|선택)")
_APPROVAL_CONTEXT_RE = re.compile(r"승인|거절|결재|카드")

# 비가역 액션 '완료' 주장(동사+정중 종결). policy.ACTION_DONE_MESSAGES 와 동일 범주.
_DONE_RE = re.compile(
    r"(?:삭제|취소|전송|발송|등록|생성|수정|신청|상신|접수|반려|승인|저장)\s*"
    r"(?:했어요|했습니다|하였습니다|되었어요|되었습니다|됐어요|됐습니다"
    r"|완료(?:했어요|했습니다|되었어요|되었습니다|됐어요|됐습니다)?)"
)

# 긴 합성 제목은 _TERSE_MAX 를 넘고, '폐기하였습니다'는 일반 완료 동사에 없었다.
# 초안 취소를 처리했다는 직접 답변만 좁게 잡는다. 인용된 문구·과거 이력은 실제 실행
# 주장이 아니므로 판정용 텍스트에서 제외한다(표시 원문은 그대로 보존한다).
_QUOTED_CANCEL_RE = re.compile(
    r'''["'“‘][^"'“”‘’\n]*(?:취소|폐기)[^"'“”‘’\n]*["'”’]\s*(?:라고|라는|라며|란)'''
)
_DRAFT_CANCEL_PREFIX_RE = re.compile(
    r"^(?:(?:네|알겠습니다)[,.!，\s]+)?(?:요청하신|말씀하신|지정하신|해당|이\s|기존|초안|미리보기)"
)
_DRAFT_CANCEL_DONE_RE = re.compile(
    r"(?:취소|폐기)\s*(?:(?:를\s*)?완료\s*)?"
    r"(?:했어요|했습니다|하였습니다|되었어요|되었습니다|됐어요|됐습니다)(?=[.!?。]|\s*$)"
)
_DRAFT_CANCEL_REQUEST_RE = re.compile(
    r"(?:취소|폐기)\s*(?:해\s*(?:줘|주|줄|$)|하(?:고\s*싶|겠|려|라|세|십|자)|합니|할(?:게|래))"
    r"|그만(?:둘|두|해)"
)


def needs_draft_cancel_guard(user_input: str) -> bool:
    """초안·미리보기의 취소 실행 요청만 스트림 보류 대상으로 고른다.

    Args:
        user_input: 현재 턴의 사용자 입력.

    Returns:
        취소·폐기·그만두기 요청이면 True. 단순 방법·인용 안내는 False.
    """
    text = _QUOTED_CANCEL_RE.sub("", user_input or "")
    return bool(re.search(r"초안|미리보기", text) and _DRAFT_CANCEL_REQUEST_RE.search(text))


def _is_draft_cancel_claim(text: str) -> bool:
    """초안 취소를 방금 처리했다는 직접 답변인지 길이와 무관하게 확인한다."""
    text = _QUOTED_CANCEL_RE.sub("", text)
    return bool(_DRAFT_CANCEL_PREFIX_RE.match(text)
                and re.search(r"초안|미리보기", text)
                and _DRAFT_CANCEL_DONE_RE.search(text))

# 문장 분리 — 줄 분리는 상위 루프가 하므로 여기서 ``\n`` 은 경계로 쓰지 않는다.
# (예전 패턴이 ``\n`` 을 포함해 표 행·목록이 전부 문장이 됐다 — 모듈 docstring 참조.)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?。])\s+")

# ── 마크다운 구조 줄 (판정 제외 — 원문 그대로 보존) ──────────────────
_FENCE_RE = re.compile(r"^\s*(?:```|~~~)")            # 코드펜스 여닫이
_HEADING_RE = re.compile(r"^\s*#{1,6}\s")             # 헤딩
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s)")   # 불릿·번호 목록

# 완료 환각은 대개 짧은 확인 문장("일정을 삭제했어요.")이다. 긴 요약/보고 문장
# ("김대리가 회의 일정을 삭제했다고 공지했습니다")의 오탐을 막기 위해, 완료-주장은
# 버튼 안내와 함께 나오거나 짧은 문장일 때만 환각으로 간주한다.
_TERSE_MAX = 40

_FALLBACK = "아직 실제로 처리하지는 않았어요. 원하시면 다시 요청해 주세요."

# 승인(HITL) 없이 곧바로 실행되는 액션 도구. 이 도구가 이번 턴에 **실제로 호출**됐으면
# "지웠어요/저장했어요" 완료 주장은 환각이 아니다 — 승인 카드가 없다는 이유로 지우면
# 진짜 한 일을 "안 했다"고 말하는 역환각이 된다(실측 2026-09-07: memory_forget 이 실행됐는데
# 답변이 "아직 실제로 처리하지는 않았어요" 로 바뀜). 버튼 안내 규칙은 그대로 적용한다.
#
# create_file: 같은 역환각이 파일 생성에서 재발했다(실측 2026-09-11 개발기). 파일을 실제로
# 만들어 다운로드 링크까지 답변에 실어 놓고 바로 다음 문장이 "아직 실제로 처리하지는
# 않았어요" 였다 — `_DONE_RE` 의 `생성했습니다` 가 걸렸는데 승인 카드는 없었기 때문이다.
# 파일 생성은 비가역도 아니고 승인 대상도 아니다(밖으로 나가는 건 mattermost_post_message
# 의 승인 카드가 따로 막는다). 만든 것을 만들었다고 말하는 건 환각이 아니다.
#
# schedule_delete: 예약 삭제도 승인 없이 바로 실행된다(다음 회차부터 안 돈다 — 되살리려면 다시 만들면 된다).
# 실측(2026-10-02 로컬 E2E): 실제로 지웠는데 답변 앞에 "아직 실제로 처리하지는 않았어요" 가 붙었다.
DIRECT_ACTION_TOOLS = frozenset({"memory_forget", "memory_remember", "create_file", "schedule_delete"})


def has_direct_action(tool_calls) -> bool:
    """이번 턴 도구 호출 목록에 승인 없이 실행되는 액션 도구가 있는가."""
    for tc in tool_calls or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name") or (tc.get("function") or {}).get("name")
        if name in DIRECT_ACTION_TOOLS:
            return True
    return False


def _is_table_row(line: str) -> bool:
    """마크다운 표 행인지(구분선 ``|---|`` 포함). 셀 값은 데이터라 판정 대상이 아니다."""
    stripped = line.strip()
    return stripped.startswith("|") or stripped.count("|") >= 2


def _is_phantom(sentence: str, *, judge_done: bool) -> bool:
    """문장이 환각(없는 버튼 안내 / 일어나지 않은 완료 주장)인지 판정한다."""
    is_button = bool(_BUTTON_RE.search(sentence)) or (
        bool(_GENERIC_BUTTON_RE.search(sentence))
        and bool(_APPROVAL_CONTEXT_RE.search(sentence))
    )
    # 완료-주장은 버튼과 동반되거나 짧은 확인 문장일 때만(요약문 오탐 방지)
    is_done = (
        judge_done
        and bool(_DONE_RE.search(_QUOTED_CANCEL_RE.sub("", sentence)))
        and (is_button or len(sentence) <= _TERSE_MAX)
    )
    return is_button or is_done


def _guard_line(line: str, *, judge_done: bool, removed: list[str],
                corrected: bool) -> tuple[str | None, bool]:
    """한 줄을 문장 단위로 훑어 환각 문장을 걷어낸다.

    Args:
        line: 원본 줄(들여쓰기·목록 마커 포함).
        judge_done: 완료-주장 규칙 적용 여부(산문 줄만 True).
        removed: 제거된 문장을 누적할 리스트(로그용, in-place).
        corrected: 교정 문구(_FALLBACK)를 이미 넣었는지.

    Returns:
        (재조립한 줄 또는 None, 교정 문구 삽입 여부). None 은 줄 전체가 환각이라
        지운다는 뜻이다(교정 문구는 첫 제거 지점에 이미 들어갔다).
    """
    indent = line[: len(line) - len(line.lstrip())]
    rest = line[len(indent):]
    marker_match = _LIST_RE.match(rest)
    marker = rest[: marker_match.end()] if marker_match else ""
    body = rest[marker_match.end():] if marker_match else rest

    if judge_done and _is_draft_cancel_claim(body):
        # 같은 줄의 '진행 중인 작업도 없다'는 후속 단정 역시 실제 취소 근거가 없다.
        # 완료 문장만 지우고 그 단정을 남기면 서로 모순되므로 산문 줄을 함께 교정한다.
        removed.append(body)
        return (None if corrected else indent + marker + _FALLBACK), True

    kept: list[str] = []
    for sentence in _SENT_SPLIT_RE.split(body):
        stripped = sentence.strip()
        if not stripped:
            continue
        if _is_phantom(stripped, judge_done=judge_done):
            removed.append(stripped)
            if not corrected:
                # 삭제만 하면 정보가 조용히 사라진다 — 첫 제거 자리에 교정 문구를 남긴다.
                kept.append(_FALLBACK)
                corrected = True
        else:
            kept.append(stripped)

    if not kept:
        return None, corrected
    return indent + marker + " ".join(kept), corrected


def guard_phantom_action(
    content: str, has_pending_drafts: bool, *, allow_done_claims: bool = False,
) -> str:
    """승인 카드가 없는데 LLM 이 완료/버튼을 환각한 문장을 교정 문구로 대체한다.

    판정은 산문 줄(+ 목록 항목의 버튼 문구)에만 적용하고, 표 행·코드펜스·헤딩은 원문
    그대로 둔다. 줄 구조(개행)도 보존한다 — 근거는 모듈 docstring.

    Args:
        content: LLM 최종 답변 텍스트.
        has_pending_drafts: 이번 턴에 실제로 생성된 승인 초안(카드)이 있는지.
            True 면 카드/대기 액션이 실재하므로 원문을 그대로 둔다.
        allow_done_claims: 승인 없이 실행되는 액션 도구(DIRECT_ACTION_TOOLS)가 실제로
            돌았을 때 True — 완료 주장은 정당하므로 그 규칙만 끄고 버튼 환각 규칙은 유지한다.

    Returns:
        환각 문장을 교정한 텍스트. 전부 환각이었으면 안전 안내 문구.
    """
    if has_pending_drafts or not content:
        return content

    out_lines: list[str] = []
    removed: list[str] = []
    corrected = False
    in_fence = False

    for line in content.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            out_lines.append(line)
            continue
        if in_fence or not line.strip() or _is_table_row(line) or _HEADING_RE.match(line):
            out_lines.append(line)   # 코드·표·헤딩·빈 줄 = 구조 → 판정 제외
            continue
        # 목록 항목은 조회 결과 나열이 대부분이라 완료-주장 판정에서 뺀다(버튼은 적용).
        new_line, corrected = _guard_line(
            line, judge_done=(not allow_done_claims) and not _LIST_RE.match(line),
            removed=removed, corrected=corrected,
        )
        if new_line is not None:
            out_lines.append(new_line)

    if not removed:
        return content

    logger.warning(
        "[answer_guard] pending_drafts 없이 환각 액션 문구 %d개 제거(산문 줄 한정): %r",
        len(removed), removed[:3],
    )
    cleaned = "\n".join(out_lines).strip()
    return cleaned or _FALLBACK


# ── 사람 이름 환각 가드 ──────────────────────────────────────────────
# 메신저 조회 답변에서 **없는 사람 이름이 지어지는** 사고가 실제로 있었다. 도구가 작성자를
# 영문 username 으로만 주던 시절, 모델이 한국어 답변을 만들려고 발음으로 옮겨 적었다:
#   jdoe → "김민수"(실제 김민서), jhpark2 → "박지훈"(실제 박지운), lho → "이호"(실제 홍길동).
# 5명 중 3명이 틀렸다. 도구 쪽은 실명을 내도록 고쳤지만(mattermost-mcp `names.py`),
# 프로필에 이름이 아예 없는 계정은 여전히 `@handle` 로 나가고 프롬프트만으로는 확률적이라,
# 이 코드 가드가 결정적으로 잡는다.
#
# 판정 원리: **답변에 나온 사람 이름이 이 턴의 입력(대화 이력·도구 결과) 어디에도 없으면
# 지어낸 것이다.** 근거 범위를 넓게 잡아(시스템 프롬프트·사용자 질문·이전 턴 포함) 오탐을
# 줄이고, 그럼에도 아무 데도 없는 이름만 걸러낸다.

# ── 위험 신호: 도구 결과에 섞인 "영문 ID 사람" ──────────────────────────────
#
# 예전에는 ``@handle`` 만 신호로 봤는데, **사고를 낸 형태가 정확히 그게 아니었다.**
# 도구가 맨 ``{"author": "jdoe"}`` 를 주면 가드가 무장조차 하지 않았다. 그래서 신호를
# 셋으로 넓힌다. 셋 다 "모델이 음차할 재료가 이 턴에 있다"는 같은 사실을 가리킨다.

# ① 핸들 ``@jdoe``. 앞의 부정 lookbehind 가 이메일(``user@example.com`` 의 ``@example.com``)을
#    걸러낸다 — 예전엔 이메일만 있어도 불필요하게 무장했다.
_HANDLE_RE = re.compile(r"(?<![A-Za-z0-9._%+-])@([A-Za-z][A-Za-z0-9._-]{1,30})")

# ② 이름 자리 키의 맨 라틴 값. 키를 화이트리스트로 좁히는 이유는 영문 채널명·URL·도구명
#    때문에 모든 턴이 무장되는 것을 막기 위해서다.
_PERSON_KEYS = (
    "author", "author_handle", "author_username", "username", "user_name", "handle",
    "sender", "from", "nickname", "display_name", "dm_with",
    "participants", "members", "first_name", "last_name", "display", "resolved_display_name",
)

# 사람 **나열**만 신호로 보는 키. ``channel_name`` 은 값이 사람일 수도, 그냥 영문 채널명
# (``watchdog``)일 수도 있다. 후자로도 무장하면 거의 모든 메신저 턴이 답변 전체를
# 버퍼링해 체감만 나빠진다(실측으로 확인). 그룹 DM 누출의 형태는 **콤마 나열**
# (``"gnoopy, hyunho_lim, miim"``)이므로 그 모양일 때만 신호로 친다.
_LIST_KEYS = ("channel_name",)

_KV_RE = re.compile(
    r'"(' + "|".join(_PERSON_KEYS + _LIST_KEYS) + r')"\s*:\s*"([^"\n]{1,200})"',
    re.IGNORECASE,
)
_LIST_KEY_SET = frozenset(k.lower() for k in _LIST_KEYS)

# ③ 한글 뒤 괄호 라틴 — ``김민수(jdoe)``. 모델이 **이름↔ID 결속을 스스로 주장**한 형태다.
#    JSON 키가 없는 supervisor 합성 경로(워커 산문만 근거)에서도 이게 걸린다.
_PAREN_ID_RE = re.compile(r"(?<=[가-힣])\s*\(([A-Za-z][A-Za-z0-9._-]{1,30})\)")

# 맨 라틴 토큰으로 인정할 값의 형태. 26자 Mattermost ID·이메일·공백 포함 값은 제외한다.
_BARE_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{1,30}$")


def _is_bare_handle(token: str) -> bool:
    """맨 영문 ID 로 볼 값인가 — 모델이 음차할 수 있는 형태인가."""
    if not _BARE_TOKEN_RE.match(token):
        return False
    if len(token) == 26 and token.isalnum():
        return False        # Mattermost 내부 ID — 사람 이름으로 안 읽힌다
    return True


def extract_id_tokens(evidence: str) -> set[str]:
    """근거에 등장하는 **영문 ID 사람 토큰**을 모은다.

    위험 판정(``has_unnamed_author``)과 탐지기의 후보집합을 **같은 함수**가 낸다.
    두 곳이 따로 판단하면 "무장은 했는데 못 잡는" 또는 그 반대가 생긴다.
    """
    if not evidence:
        return set()
    out: set[str] = set()
    out.update(m.group(1) for m in _HANDLE_RE.finditer(evidence))
    out.update(m.group(1) for m in _PAREN_ID_RE.finditer(evidence))
    for m in _KV_RE.finditer(evidence):
        key, value = m.group(1).lower(), m.group(2)
        if "," not in value:
            # 나열 전용 키(channel_name)는 단일 값이면 사람이 아니라 채널명으로 본다.
            if key in _LIST_KEY_SET:
                continue
            pieces = [value]
        else:
            # 그룹 DM 라벨처럼 콤마로 나열된 값은 쪼갠다 ("gnoopy, hyunho_lim, miim").
            pieces = value.split(",")
        for piece in pieces:
            piece = piece.strip().lstrip("@")
            if _is_bare_handle(piece):
                out.add(piece)
    return out

# 직급 — "이영희 이사님" 처럼 이름과 님 사이에 끼어 이름 탐지를 가리는 것들.
_TITLE_ALT = (
    "이사|부장|과장|차장|대리|팀장|주임|사원|실장|본부장|상무|전무|사장|대표"
    "|선임|책임|수석|매니저|반장|소장|국장|센터장|파트장"
)

# "이름 (선택)괄호ID (선택)직급 님" — 실제 오답 형태가 "김민수(jdoe) 님" 이었다.
#
# 직급을 **캡처**하는 이유: 직급이 끼면 그 앞 어절을 이름으로 집는데, 그게 이름이 아닐 때가
# 많다. 실측(운영 답변 40건): "부스 연출 **관련** 대표님", "…에 **대한** 이사님",
# "차이가 **있어** 이사님께" 처럼 평범한 단어가 잡혔다. 직급이 있는 경우에만 성씨 관문을
# 추가로 건다 — 순수 "이름 님" 은 검증된 경로라 그대로 둔다.
_HONORIFIC_NAME_RE = re.compile(
    r"(?<![가-힣])([가-힣]{2,4})\s*(?:\([^)\n]{1,40}\))?\s*(" + _TITLE_ALT + r")?\s*님"
)

# 이름이 아니라 역할·호칭인 말들 — 위 정규식이 이름 자리로 잡아내므로 제외한다.
_NAME_STOPWORDS = frozenset({
    "고객", "손님", "회원", "사용자", "관리자", "담당자", "여러분", "선생", "기사",
    "어머", "아버", "부모", "학생", "환자", "작성자", "발신자", "수신자",
    "이사", "부장", "과장", "차장", "대리", "팀장", "주임", "사원", "실장", "본부장",
    "상무", "전무", "사장", "대표", "선임", "책임", "수석", "매니저", "반장", "소장",
    "국장", "센터장", "파트장", "직원", "동료", "신입", "선배", "후배", "형님", "누님",
    # 목록·표의 첫 칸에 실제로 자주 오는 비인명(단서 C 전용으로 넓힌 것)
    "없음", "미정", "전체", "기타", "확인", "완료", "진행", "대기", "공지", "회의",
    "오늘", "어제", "내일", "이번", "지난", "다음", "채널", "부서", "요청", "보고",
    "일정", "내용", "결과", "상태", "제목", "구분", "비고", "참고", "첨부", "안건",
    # 운영 답변 실측(2026-09-03, 40건)에서 실제로 이름 자리에 잡혔던 비인명
    "관련", "대한", "있어", "채널", "채널명", "대화", "문제점", "님은", "요약", "링크",
})

# ── 단서 B/C 전용: 한국 성씨 선두 필터 ───────────────────────────────────────
#
# 한글 2~4음절은 어디에나 있어서 "이름 자리에 있으면 이름"으로는 오탐이 폭발한다.
# 다행히 **음차 환각은 언제나 "성+이름" 형태로 나온다**(모델이 로마자를 한국 이름으로
# 되짚기 때문). 그래서 첫 음절이 성씨인지를 추가 관문으로 둔다 — 재현율은 거의 안 깎이고
# 오탐은 크게 준다. 검증된 단서 A(``… 님``)에는 적용하지 않는다: 이미 정밀한 경로에
# 새 필터를 끼우면 회귀 위험만 생긴다.
_SURNAMES = frozenset(
    "김이박최정강조윤장임한오서신권황안송유홍전고문양손배백허남심노하곽성차주우구민진지"
    "엄채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모나탁국여어은편용"
)
_TWO_SYLLABLE_SURNAMES = frozenset({
    "남궁", "황보", "제갈", "선우", "독고", "사공", "서문", "동방",
})


def _has_surname_head(name: str) -> bool:
    """이름이 한국 성으로 시작하는가 — 단서 B/C 의 추가 관문."""
    return name[:2] in _TWO_SYLLABLE_SURNAMES or name[:1] in _SURNAMES


# 조사로 끝나는 어절은 이름이 아니다. 성씨 필터만으로는 "**채널**에서" 처럼 우연히 성씨로
# 시작하는 보통 어절이 통과한다(실측에서 실제로 남았다). 한국 사람 이름은 이 꼬리로
# 끝나지 않으므로 안전한 관문이다.
_PARTICLE_TAILS = ("에서", "에게", "으로", "부터", "까지", "라고", "하고", "이서", "면서")


def _ends_with_particle(name: str) -> bool:
    return len(name) >= 3 and name.endswith(_PARTICLE_TAILS)


# 단서 B — ``김민수(jdoe)`` : 모델이 **이름↔ID 결속을 스스로 주장**한 형태.
# 괄호 토큰이 이 턴의 ID 후보집합에 있으면 이름 후보로 본다. 오탐이 사실상 0이다.
_NAME_WITH_ID_RE = re.compile(r"(?<![가-힣])([가-힣]{2,4})\s*\(\s*@?([A-Za-z][A-Za-z0-9._-]{1,30})\s*\)")

# 단서 C — ``님`` 없이 이름 자리에만 오는 형태들. 단독으로는 오탐이 크므로
# 스톱워드 + 성씨 필터를 **둘 다** 통과해야 채택한다.
_NAME_POSITION_RES = (
    # 목록 항목 선두: "- 김민수: …", "* 김민수 —"
    re.compile(r"^[ \t]*[-*•]\s*(?:\*\*)?([가-힣]{2,4})(?:\*\*)?\s*(?=[:：\-–—(])", re.MULTILINE),
    # 표 셀: "| 김민수 |"
    # 뒤의 '|' 는 소비하지 않는다(앞을 봄) — 소비하면 한글 칸이 이어질 때 다음 칸을 못 봤다("| 서버 | 김민수 |").
    re.compile(r"\|\s*(?:\*\*)?([가-힣]{2,4})(?:\*\*)?\s*(?=\|)"),
    # 직급 직결(님 없음): "김민수 과장이 …"
    #
    # 직급 뒤에는 조사만 허용한다. 그냥 ``(?![가-힣])`` 로 막으면 "과장**이**" 가 안 걸리고,
    # 반대로 아무 한글이나 허용하면 "대표**적**인 사례" 의 "대표" 를 직급으로 오인해 앞
    # 어절을 이름으로 집는다.
    re.compile(
        r"(?<![가-힣])([가-힣]{2,4})\s*(?:" + _TITLE_ALT + r")"
        r"(?:이|가|은|는|을|를|의|에|도|과|와|께|만|께서)?(?![가-힣])"
    ),
)


# 단서 C 의 목록·표 자리는 '사람 자리' 일 때만 이름 후보로 본다. 성씨로 시작하는 보통 명사가 목록 머리·표 칸에 오면
# ("- 선택: …", "| 차별점 |") 이름으로 잡혀 답 전체에서 '이름미확인' 으로 바뀌었다(실측 2026-10-01: 차별점·선택·
# 원하시면). 사람 자리 = 그 줄에 사람이 한 일·호칭이 있거나(목록), 그 칸의 표 머리글이 사람을 가리킨다(표).
# 사람이 **한 일**의 동사 꼴만 본다 — "담당자"·"확인 후" 같은 명사·절차 표현은 안내문에 흔해 사람 자리로 오인했다
# (실측: "**선택:** 필수값(프로젝트명, 담당자…)" 의 '선택'). 줄 끝의 "… 공유" 처럼 명사로 끝나는 보고형도 받는다.
_PERSON_LINE_CUES = re.compile(
    r"(?:공유|작성|확인|요청|답변|회신|전달|보고|제출|문의|질문|언급|발언|안내)"
    r"(?:했|함|하셨|하였|드림|드렸|드립니다|해\s?주|중)"
    r"|(?:공유|요청|보고|회신|전달)\s*[.)]?\s*$"
    r"|말씀|올렸|올림|보냈|보냄|님(?:이|께서|은|의|과|께)|씨(?![가-힣])"
)
_PERSON_HEADER_CUES = re.compile(r"작성|담당|이름|성명|발신|보낸|요청자|수신|참석|사람|멤버|구성원|인원|대상자|책임")


def _list_line_is_person_slot(answer: str, pos: int) -> bool:
    start = answer.rfind("\n", 0, pos) + 1
    end = answer.find("\n", pos)
    line = answer[start:end if end != -1 else len(answer)]
    return bool(_PERSON_LINE_CUES.search(line) or re.search(_TITLE_ALT, line))


def _table_cell_is_person_slot(answer: str, pos: int) -> bool:
    """표 칸 하나가 사람 칸인가 — 같은 표의 머리글에서 그 칸 이름을 본다. 머리글을 못 찾으면 사람 칸으로 보지 않는다."""
    lines = answer.split("\n")
    offset, idx = 0, 0
    for i, ln in enumerate(lines):
        if offset + len(ln) >= pos:
            idx = i
            break
        offset += len(ln) + 1
    col = answer[offset:pos].count("|") - 1          # 이 칸이 몇 번째 칸인지(0부터)
    # 위로 올라가며 구분선(|---|) 바로 위 줄을 머리글로 본다.
    for j in range(idx, 0, -1):
        if re.fullmatch(r"\s*\|?\s*:?-{2,}.*", lines[j]) and "|" in lines[j]:
            cells = [c.strip() for c in lines[j - 1].strip().strip("|").split("|")]
            return 0 <= col < len(cells) and bool(_PERSON_HEADER_CUES.search(cells[col]))
    return False


def _normalize(text: str) -> str:
    """근거 대조용 정규화 — 공백을 없애 띄어쓰기 차이로 못 찾는 일을 막는다."""
    return re.sub(r"\s+", "", text or "")


# 시스템 프롬프트 안의 ``[요청자 정보]`` 블록.
#
# **줄 시작 + 뒤따르는 ``- `` 항목**을 함께 요구하는 것이 핵심이다. 프롬프트 본문에도
# "``[요청자 정보]`` 블록의 값만 그대로 사용하세요" 같은 **지시문**이 있어서, 단순히
# 대괄호 문자열만 찾으면 그 문장을 잡는다(실측: 요청자 이름이 근거에서 빠져 본인 이름이
# 오탐됐다). 실제 블록은 ``_build_user_context`` 가 만드는 ``- 이름: …`` 목록뿐이다.
_REQUESTER_BLOCK_RE = re.compile(r"^\[요청자 정보\]\n(?:-[^\n]*(?:\n|$))+", re.MULTILINE)


def requester_context(system_prompt: str) -> str:
    """시스템 프롬프트에서 ``[요청자 정보]`` 블록만 잘라낸다.

    **왜 이 함수가 필요한가.** 시스템 프롬프트 전문을 근거로 삼으면, 프롬프트에 예시로
    적힌 이름이 **화이트리스트**가 된다. 실제로 이름 규칙 조항의 반례에 실명이 박혀 있어
    사고를 낸 바로 그 이름이 영구히 탐지 불가였다(2026-09-03 확인). 그렇다고 프롬프트를
    통째로 빼면 **요청자 본인 이름**이 근거를 잃고 환각으로 오탐된다 — 요청자 실명은
    이 블록에만 있기 때문이다. 그래서 이 블록만 되살린다.

    Args:
        system_prompt: 시스템 프롬프트 전문.

    Returns:
        ``[요청자 정보]`` 블록 문자열. 없으면 빈 문자열.
    """
    m = _REQUESTER_BLOCK_RE.search(system_prompt or "")
    return m.group(0) if m else ""


def has_unnamed_author(evidence: str) -> bool:
    """도구 결과에 **영문 ID 사람**이 섞여 있는지 — 이름 환각 위험 신호.

    이름이 있는 사람만 등장하는 답변에서는 모델이 음차할 재료가 없다. 위험이 있을 때만
    스트리밍을 붙들어(가드가 검증할 때까지) 평소 응답 체감을 해치지 않는다.

    이름·시그니처를 그대로 두는 것은 의도적이다 — ``react_graph`` 와 ``supervisor`` 의
    **스트림 보류 조건과 검증 조건이 같은 함수**여야 한다. 붙들지 않은 턴에서 뒤늦게
    본문을 바꾸면 화면 ≠ 히스토리가 되고, SSE 에는 본문 교체 이벤트가 없다.
    """
    return bool(extract_id_tokens(evidence))


def find_unsourced_person_names(answer: str, evidence: str) -> list[str]:
    """답변의 사람 이름 중 **근거에 전혀 없는** 것을 찾는다(중복 제거, 등장순).

    단서 셋을 쓴다. 앞의 둘은 오탐이 사실상 0이고, 셋째만 관문을 더 건다.

    A. ``이름 [(id)] [직급] 님`` — 검증된 기존 경로. 손대지 않는다.
    B. ``이름(id)`` — 모델이 이름↔ID 결속을 **스스로 주장**한 형태. 괄호 토큰이 이 턴의
       ID 후보집합에 있을 때만.
    C. ``님`` 없는 이름 자리(목록 선두·표 셀·직급 직결). 한글 2~4음절은 어디에나 있어
       단독으로는 오탐이 크므로 **스톱워드 + 성씨 선두** 두 관문을 모두 통과해야 한다.

    Args:
        answer: 최종 답변 텍스트.
        evidence: 이 턴의 입력(대화 이력·도구 결과. 시스템 프롬프트 본문은 제외된다 —
            프롬프트에 적힌 이름이 화이트리스트가 되는 것을 막기 위해서다).

    Returns:
        지어낸 것으로 판단되는 이름 목록. 비어 있으면 이상 없음.
    """
    if not answer or not evidence:
        return []
    haystack = _normalize(evidence)
    id_tokens = extract_id_tokens(evidence)
    found: list[str] = []

    def _take(name: str, *, need_surname: bool) -> None:
        if name in _NAME_STOPWORDS or name in found:
            return
        if _ends_with_particle(name):
            return
        if need_surname and not _has_surname_head(name):
            return
        if _normalize(name) in haystack:
            return
        found.append(name)

    # A — 기존 경로. 직급이 낀 형태만 성씨 관문을 건다(위 정규식 주석 참고).
    for m in _HONORIFIC_NAME_RE.finditer(answer):
        _take(m.group(1), need_surname=bool(m.group(2)))
    # B — 이름(ID) 결속
    for m in _NAME_WITH_ID_RE.finditer(answer):
        if m.group(2) in id_tokens:
            _take(m.group(1), need_surname=False)
    # C — 이름 자리(관문 2개 + 사람 자리 관문). 목록 머리·표 칸은 사람 자리일 때만 본다.
    list_rx, table_rx, title_rx = _NAME_POSITION_RES
    for m in list_rx.finditer(answer):
        if _list_line_is_person_slot(answer, m.start(1)):
            _take(m.group(1), need_surname=True)
    for m in table_rx.finditer(answer):
        if _table_cell_is_person_slot(answer, m.start(1)):
            _take(m.group(1), need_surname=True)
    for m in title_rx.finditer(answer):
        _take(m.group(1), need_surname=True)
    return found


def redact_unsourced_names(answer: str, names: list[str]) -> str:
    """재생성으로도 안 고쳐진 이름을 ``이름미확인`` 으로 바꾼다.

    틀린 이름을 그대로 두는 것보다 낫다 — 사용자가 보고한 피해가 바로 **다른 사람에게
    발언을 잘못 붙이는 것**이었다. 괄호 안 핸들(``(jdoe)``)은 사실이므로 남겨 둔다.

    **문맥이 아니라 이름 문자열 전량을 치환한다.** 예전에는 ``… 님`` 문맥에서만 바꿨는데,
    탐지기를 ``님`` 밖으로 넓히면 "탐지는 됐는데 마스킹이 no-op" 이 되어 재생성 실패 후
    원문이 그대로 나간다 — 로그만 남고 사용자는 틀린 이름을 보는, 지금보다 나쁜 상태다.
    이미 "근거 어디에도 없다"고 판정된 이름이므로 답변 안 **모든 출현이 날조**다.
    """
    if not names:
        return answer
    out = answer
    for name in names:
        out = re.sub(
            r"(?<![가-힣])" + re.escape(name) + r"(?![가-힣])", "이름미확인", out
        )
    logger.warning("[name_guard] 근거 없는 이름 %d개 마스킹: %r", len(names), names[:3])
    return out


# ── 메신저 딥링크 날조 차단 ────────────────────────────────────────────────
#
# 2026-09-16 실사용: 답변의 「메신저 링크」가 `https://chat.example.com/dadc/pl/` 로 끝났다.
# **post_id 가 통째로 비어 있다.** 원래 값은
# `https://chat.example.com/team/pl/hiozn4yzoibg9nwu1icjkahuzy` 였고, 색인의
# `doc_url` 은 멀쩡했다 — 모델이 그 값을 쓰지 않고 **현재 대화 중인 팀 슬러그**(`dadc`)에
# `/pl/` 만 붙여 지어냈다. 같은 답변의 외부 문서 링크(`links` 에서 온 것)는 정상이었다.
#
# 프롬프트에 "`doc_url`·`permalink` 를 써라"가 이미 있는데도 샜다. 링크는 눌러 보기 전엔
# 진위를 알 수 없어 사용자가 **검증 못 하는 거짓말**이라, 이름 환각과 같은 등급으로 막는다.
#
# 판정은 길이다 — Mattermost post_id 는 **정확히 26자** 영숫자다(운영 색인 실측: 분포가
# {26} 단일값). 26자가 아니면 그 링크는 실재하지 않는다.
#
# 2026-09-18 보강 — **같은 날조가 경로를 바꿔 다시 나왔다.** 브라우저 패널 답변의 첨부
# 링크가 이렇게 나갔다(둘 다 id 는 26자로 **멀쩡하다** — 잘린 게 아니라 경로가 가짜다):
#
#     https://chat.example.com/team/files/pwr4bt3bopfqpbnin888qgmt3o
#
# Mattermost 웹앱에는 **파일 전용 웹 경로가 없다**(webapp 라우트 표: pl·channels·messages·
# threads·drafts·recaps·scheduled_posts·_playbooks 뿐). 그래서 MCP 도 파일 히트의
# `doc_url` 에 "파일이 올라간 **글**" 의 permalink 를 싣는다(knowledge/models.py). 모델이
# 그 값을 쓰지 않고 `file_id` 로 주소를 조립한 것이다.
#
# 그래서 판정을 `/pl/` 전용에서 **경로 allowlist** 로 넓힌다 — 실재하는 라우트가 아니면
# 전부 날조다. 그리고 지우고 끝내지 않고, 그 턴 도구 결과에서 얻은 진짜 permalink 로
# **바꿔 단다**(`build_file_link_map`). 지우기만 하면 이사님이 원한 "자료로 가는 길" 이
# 여전히 없다.
_MM_POST_ID_LEN = 26
#: Mattermost 웹앱이 실제로 가진 `/{team}/<route>/…` 경로. 벤더 소스의 라우트 표 그대로다.
#: **`files` 가 없다는 것이 이 목록의 요점이다.**
_MM_ROUTE_ALLOW = frozenset({
    "pl", "channels", "messages", "threads", "drafts", "recaps",
    "scheduled_posts", "integrations", "emoji", "_playbooks",
})
#: 팀 슬러그로 시작하지 않는 경로 — 판정 대상이 아니다(로그인·관리자·플러그인·API 등).
_MM_NON_TEAM_PREFIX = frozenset({
    "api", "oauth", "admin_console", "signup", "login", "logout",
    "plugins", "static", "help", "error", "landing", "_redirect",
})
#: URL 로 인정할 글자만 — 한글 조사(`…입니다`)나 괄호를 삼키지 않게 좁게 잡는다.
_MM_URL_RE = re.compile(r"https?://[A-Za-z0-9\-._~:/?#@!$&*+,;=%]+")
_MM_PERMALINK_RE = re.compile(
    r"https?://[^\s<>\)\]]*?/pl/(?P<pid>[A-Za-z0-9]*)"
)
#: 지어낸 링크를 지운 자리에 남길 말. 링크만 빼고 라벨은 살린다.
_LINK_DROPPED = "(링크 없음)"


def _mm_host() -> str:
    """메신저 호스트(netloc). 설정이 없으면 빈 문자열."""
    try:
        return (urlsplit(getattr(settings, "mattermost_base_url", "") or "").netloc or "").lower()
    except Exception:  # noqa: BLE001 — 가드는 설정 문제로 죽지 않는다
        return ""


def _is_fabricated_mm_url(url: str) -> bool:
    """메신저 호스트의 주소가 **실재할 수 없는 경로**인지 본다."""
    seg = [s for s in (urlsplit(url).path or "/").split("/") if s]
    if len(seg) < 2 or seg[0] in _MM_NON_TEAM_PREFIX:
        return False
    route = seg[1]
    if route not in _MM_ROUTE_ALLOW:
        return True
    if route == "pl":
        return len(seg[2] if len(seg) > 2 else "") != _MM_POST_ID_LEN
    return False


def find_fabricated_permalinks(answer: str) -> list[str]:
    """답변에서 **실재할 수 없는** 메신저 딥링크를 찾는다.

    메신저 호스트(`MATTERMOST_BASE_URL`)의 주소만 본다 — 외부 문서 링크(Confluence 등)는
    판정 대상이 아니다. 호스트 설정이 없으면 구 동작(`/pl/` 전용)으로 폴백한다.

    Args:
        answer: 모델이 쓴 답변 본문.

    Returns:
        실재하지 않는 딥링크 URL 목록(중복 제거, 등장 순서).
    """
    host = _mm_host()
    out: list[str] = []
    seen: set = set()
    for m in _MM_URL_RE.finditer(answer or ""):
        # 문장 끝 구두점은 주소가 아니다 — `…/pl/abc.` → `…/pl/abc`
        url = m.group(0).rstrip(".,;:!?")
        if host:
            if (urlsplit(url).netloc or "").lower() != host:
                continue
            bad = _is_fabricated_mm_url(url)
        else:
            pm = _MM_PERMALINK_RE.fullmatch(url)
            bad = bool(pm) and len(pm.group("pid")) != _MM_POST_ID_LEN
        if bad and url not in seen:
            seen.add(url)
            out.append(url)
    return out


def _walk_file_links(node, out: dict) -> None:
    """도구 결과 JSON 을 훑어 ``file_id`` → permalink 짝을 모은다(같은 dict 안에서만)."""
    if isinstance(node, dict):
        fid = node.get("file_id")
        # 검색 히트는 `doc_url`, `mattermost_read_file` 은 `url` 에 permalink 를 싣는다.
        url = node.get("doc_url") or node.get("url")
        if (isinstance(fid, str) and len(fid) == _MM_POST_ID_LEN
                and isinstance(url, str) and "/pl/" in url):
            out.setdefault(fid, url)
        for v in node.values():
            _walk_file_links(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_file_links(v, out)


def build_file_link_map(tool_results) -> dict[str, str]:
    """그 턴 도구 결과에서 ``file_id`` → **글 permalink** 매핑을 만든다.

    근거 텍스트를 정규식으로 긁지 않고 JSON 을 파싱한다 — 히트 dict 안에서 `doc_url` 과
    `file_id` 사이에 긴 `text`(파일 본문 발췌)가 끼어 있어 창(window) 매칭은 어긋난다.

    Args:
        tool_results: ``state["collected_tool_results"]`` 모양의 리스트
            (각 원소 ``{"tool_call_id", "name", "content"}``, content 는 JSON 문자열).

    Returns:
        ``{file_id: permalink}``. 뽑을 게 없으면 빈 dict.
    """
    out: dict[str, str] = {}
    for r in tool_results or []:
        content = r.get("content") if isinstance(r, dict) else None
        if not isinstance(content, str) or "file_id" not in content:
            continue
        try:
            _walk_file_links(json.loads(content), out)
        except (ValueError, TypeError):
            continue  # [ERROR] … 처럼 JSON 이 아닌 결과는 건너뛴다
    return out


def redact_fabricated_permalinks(
    answer: str,
    *,
    file_links: dict | None = None,
) -> tuple[str, list[str]]:
    """지어낸 메신저 딥링크를 **진짜 링크로 바꾸거나**, 못 바꾸면 걷어낸다.

    가짜 주소의 마지막 조각이 26자면 `file_id` 후보다. ``file_links`` 에 그 값이 있으면
    그 파일이 올라간 **글의 permalink** 로 바꾼다::

        [팝빌 8월이용료.pdf](…/pl/<post_id>)

    매핑이 없으면 기존 동작 — 마크다운 링크(``[이동](url)``)는 **라벨만 남기고** 풀고
    (표의 열이 통째로 비면 답이 깨져 보인다), 맨 URL 은 자리표시자로 바꾼다.

    Args:
        answer: 모델이 쓴 답변 본문.
        file_links: ``build_file_link_map`` 이 만든 ``{file_id: permalink}``.

    Returns:
        ``(정리된 본문, 걷어낸/바꾼 URL 목록)``. 손댈 것이 없으면 원문 그대로.
    """
    bad = find_fabricated_permalinks(answer)
    if not bad:
        return answer, []
    out = answer
    fixed = 0
    for url in bad:
        esc = re.escape(url)
        fid = (urlsplit(url).path or "").rstrip("/").rsplit("/", 1)[-1]
        real = (file_links or {}).get(fid)
        if real:
            # `![라벨](가짜)` 도 받는다 — permalink 는 이미지가 아니라 글 주소라 `!` 를 뗀다.
            out = re.sub(r"!?\[([^\]]*)\]\(" + esc + r"\)",
                         lambda m, _r=real: f"[{m.group(1)}]({_r})", out)
            out = out.replace(url, real)
            fixed += 1
        else:
            # [라벨](깨진주소) → 라벨  (표 셀이 비지 않게 라벨은 살린다)
            out = re.sub(r"!?\[([^\]]*)\]\(" + esc + r"\)", r"\1", out)
            # 남은 맨 URL
            out = out.replace(url, _LINK_DROPPED)
    logger.warning("[link_guard] 지어낸 메신저 링크 %d개 (치환 %d / 제거 %d): %r",
                   len(bad), fixed, len(bad) - fixed, bad[:3])
    return out, bad
