"""대화 히스토리 관리 모듈.

Redis에 대화방(또는 게스트) 단위 컨텍스트를 저장하고 불러온다.

Redis 키 prefix는 호출자가 결정한다:
- 로그인 사용자: f"conversation:{conversation_id}"   (orchestrator·save_logged_in_turn)
- 게스트: f"guest:{guest_id}"                         (app/api/v1/guest.py:_guest_history_key)

PostgreSQL 영구 저장은 save_logged_in_turn()을 통해 처리한다.
게스트 PG 저장은 app.db.repository.save_guest_turn()을 직접 호출한다.

토큰 기반 자동 압축:
    append_turn 끝에서 LIST 토큰 합계가 COMPACT_TRIGGER_TOKENS 를 넘으면
    비동기로 _compact_history_guarded 를 호출한다. 오래된 메시지를 LLM 으로
    요약하여 단일 system 메시지(name="__compact_summary__")로 치환하고
    최근 COMPACT_KEEP_RECENT_MESSAGES 개는 verbatim 유지한다. 압축이 완료되면
    `{key}:compact_announce` 키에 통지 페이로드를 저장한다. 클라이언트
    노출은 비활성화되어 있다 (답변 본문 앞에 통지 JSON 이 토큰처럼 박히는
    문제로 UX 사고 발생 가능). 키는 관측성용으로 작성·보존되며,
    COMPACT_ANNOUNCE_TTL 후 자동 만료되고 같은 conversation 에서 압축이
    재발동하면 덮어쓰기된다.
"""
import asyncio
import hashlib
import json
from contextvars import ContextVar
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from agent_arch.agents.context import (
    get_attachment_context,
    get_unattended_run,
    get_user_context,
    web_search_choice,
)
from agent_arch.core.config import settings
from agent_arch.llm.base import Message
from agent_arch.memory.redis_store import redis_store
from agent_arch.db.repository import save_turn as pg_save_turn
from agent_arch.services.preferences import get_for_user as get_user_preferences
from agent_arch.utils.pos_history import _POS_RAW_MAX, sanitize_pos_turn

logger = logging.getLogger(__name__)


# 전역 폴백 — 사용자가 개인설정을 등록하지 않은 경우(또는 게스트)에만 사용한다.
MAX_HISTORY = 20  # 최대 저장 메시지 수 (= 10턴)
SESSION_TTL = 86400  # 24시간 (초)

# tool_ctx 를 도구 없는 턴 몇 번까지 살려 둘지. 되묻기 1~2턴 동안은 대상(직전 조회 결과)이 필요하지만,
# 그 뒤에도 남기면 주제가 바뀐 대화에 오래된 결과가 끼어든다.
TOOL_CTX_MAX_IDLE_TURNS = 2


def _turn_anchor(user_input: str) -> str:
    """tool_ctx 를 만든 턴의 user 메시지를 이력에서 다시 찾기 위한 지문."""
    return hashlib.sha1((user_input or "").encode("utf-8")).hexdigest()[:16]


# 직전 1턴 도구 컨텍스트 캐싱 — tool_ctx 키에 저장하는 tool_result content 의 최대 길이.
#
# ⚠️ 이 값은 **답변 정확도와 무관하다.** 그 턴의 도구 결과는 `react/executor.py` 가 원문
# 그대로 LLM 에 넘긴다. 여기서 줄이는 건 **다음 턴으로 이월되는 사본**뿐이다.
#
# 8,000 → 20,000 (2026-08-25). 8,000이 문제였던 건 값이 작아서가 아니라 자르는 방식이
# JSON 을 깨뜨려서였고 그건 `trim_tool_result_content` 로 고쳤다. 그럼에도 값을 올린 이유는
# 실측 때문이다 — 지식검색이 top_k=10 이면 11,494~16,582자라 8,000에서는 본문이 158자까지
# 눌렸다(식별자는 살아남아 "찾기→읽기"는 됐지만 후속턴 인용·요약이 얇아진다).
# 20,000이면 top_k 상한(20)까지도 사실상 무손실이다.
MAX_TOOL_RESULT_CHARS = 20000

# 한 턴에 이월하는 **모든** 도구 결과의 합계 상한. 개별 상한만 있으면 최악이 묶이지 않는다:
# `mattermost_read_file` 하나가 30,000자까지 나오므로 "검색 + 파일 2개 읽기" 턴이면
# 이월분만 9만자(≈40k 토큰)가 되고, `load_history` 는 이걸 **하위 예산 없이 그대로**
# 프롬프트에 주입한다(방어선이 여기 말고는 없다). 전 도메인 공통 경로라 전자결재·메일의
# 큰 결과도 같이 실린다.
MAX_TOOL_CTX_TOTAL_CHARS = 40000

TRUNCATE_NOTICE = "\n\n(...결과 일부 생략, 자세히 보려면 다시 조회해주세요)"

# 필드 단위 절삭에서 본문 문자열을 줄여 볼 길이 후보(내림차순). 앞에서부터 시도해
# 상한에 처음 들어가는 길이를 쓴다 — 필요한 만큼만 줄인다.
#
# ⚠️ 첫 단계를 12,000보다 크게 잡지 말 것. 결과 하나만 보면 16,000이 더 좋아 보이지만
# (30,000자 read_file 이 12,094 대신 16,100자로 남는다), 총량 상한과 함께 보면 손해다:
# "검색 + read_file 2개" 턴이 20,000 공통상한에서 45,497자가 돼 예산을 넘고, 그러면
# 공통상한이 10,000으로 내려가 **전부** 더 깎인다(실측 37,485 → 약 26,300자).
# 결과 하나를 욕심내다 턴 전체를 잃는다.
_SHRINK_STEPS = (12000, 8000, 5000, 3000, 2000, 1200, 800, 500, 300, 150)

# 총량이 넘칠 때 **모든 결과에 공통으로** 적용해 볼 상한(내림차순). 공통 상한은 그보다
# 짧은 결과를 건드리지 않으므로, 자연히 **큰 결과부터** 줄어든다.
_TOTAL_CAP_FLOOR = 300
_SHRINK_MARK = "…(이하 생략)"
_OMITTED_KEY = "_생략된_항목수"


def _shrink_strings(node, limit: int):
    """구조는 그대로 두고 **긴 문자열만** 잘라 낸 복사본을 만든다(결정론적).

    dict 의 키·리스트의 항목 수는 건드리지 않는다 — 절삭으로 잃어도 되는 것은 본문
    뒷부분뿐이고, ``file_id``·파일명·채널 같은 짧은 식별자는 반드시 살아남아야 한다.
    """
    if isinstance(node, str):
        return node if len(node) <= limit else node[:limit] + _SHRINK_MARK
    if isinstance(node, dict):
        return {k: _shrink_strings(v, limit) for k, v in node.items()}
    if isinstance(node, list):
        return [_shrink_strings(v, limit) for v in node]
    return node


def _drop_tail_items(data, cap: int):
    """문자열을 최소로 줄여도 상한을 넘으면 **가장 긴 리스트의 뒤쪽 항목**을 덜어낸다.

    항목 수까지 줄여야 하는 경우에도 몇 개를 뺐는지 남긴다 — 조용히 사라지면 LLM 이
    "그게 전부"라고 오해한다.
    """
    if not isinstance(data, dict):
        return None
    key = max(
        (k for k, v in data.items() if isinstance(v, list) and v),
        key=lambda k: len(json.dumps(data[k], ensure_ascii=False)),
        default=None,
    )
    if key is None:
        return None
    items = list(data[key])
    while items:
        items.pop()
        probe = dict(data)
        probe[key] = items
        probe[_OMITTED_KEY] = len(data[key]) - len(items)
        text = json.dumps(probe, ensure_ascii=False)
        if len(text) <= cap:
            return text
    return None


def trim_tool_result_content(name: str, content: str, cap: int = MAX_TOOL_RESULT_CHARS) -> str:
    """도구 결과를 상한 안으로 줄인다 — **JSON 구조를 유지하면서.**

    ⚠️ 예전에는 ``content[:cap]`` 으로 문자 단위 절단을 했다. 도구 결과는
    ``json.dumps`` 산출물이라 그렇게 자르면 **JSON 이 깨지고 뒤쪽 항목이 통째로
    사라진다.** 실측(2026-08-24): 지식검색 결과가 10,331~16,909자로 8,000자 상한에
    걸려, 다음 턴 컨텍스트에 앞쪽 1~3건만 남고 뒤쪽 히트의 ``file_id`` 가 소실됐다.
    그 결과 "나열된 것 중 N번째 문서 찾아줘"·"그 문서 요약해줘"가 **재료가 없어서**
    실패했다(첨부 요약 전면 실패의 직접 원인).

    이 함수는 LLM 을 쓰지 않는다 — 라우터 재작성(확률적)이 정보를 떨어뜨려 원문 보존으로
    돌아섰던 사고(b51ac93)와는 다른 계급이다. 무엇이 줄어드는지 정확히 정해져 있다:
    **긴 문자열의 뒷부분**뿐이고, 짧은 식별자와 항목 수는 보존된다.

    Args:
        name: 도구 이름(로그용).
        content: 도구 결과 문자열(보통 JSON).
        cap: 상한 문자 수.

    Returns:
        상한 이하로 줄인 문자열. JSON 이 아니면 종전처럼 문자 절단한다.
    """
    if len(content) <= cap:
        return content
    try:
        data = json.loads(content)
    except (ValueError, TypeError):
        # JSON 이 아닌 도구(평문 반환)는 종전 동작 유지.
        logger.info("tool_result 문자 절단(JSON 아님): tool=%s original_len=%d",
                    name, len(content))
        return content[:cap] + TRUNCATE_NOTICE

    for limit in _SHRINK_STEPS:
        text = json.dumps(_shrink_strings(data, limit), ensure_ascii=False)
        if len(text) <= cap:
            logger.info(
                "tool_result 본문 축약: tool=%s original_len=%d → %d (문자열 %d자 상한)",
                name, len(content), len(text), limit,
            )
            return text

    minimal = _shrink_strings(data, _SHRINK_STEPS[-1])
    dropped = _drop_tail_items(minimal, cap)
    if dropped is not None:
        logger.info("tool_result 항목 일부 제외: tool=%s original_len=%d → %d",
                    name, len(content), len(dropped))
        return dropped

    logger.warning("tool_result 축약 실패 — 문자 절단으로 폴백: tool=%s original_len=%d",
                   name, len(content))
    return content[:cap] + TRUNCATE_NOTICE

def trim_tool_results(
    results: List[dict],
    *,
    per_cap: int = MAX_TOOL_RESULT_CHARS,
    total_cap: int = MAX_TOOL_CTX_TOTAL_CHARS,
) -> List[dict]:
    """한 턴의 도구 결과들을 **개별 상한 + 총량 상한** 안으로 줄인다.

    개별 상한만 두면 최악이 묶이지 않는다. ``mattermost_read_file`` 하나가 30,000자까지
    나오므로 "검색 + 파일 2개 읽기" 턴은 이월분만 9만자(≈40k 토큰)가 되는데,
    ``load_history`` 는 이걸 **하위 예산 없이 그대로** 프롬프트에 주입한다.

    줄이는 방식은 **모든 결과에 공통 상한을 걸고 그 상한을 절반씩 낮추는** 것이다.
    공통 상한은 그보다 짧은 결과를 건드리지 않으므로(``trim_tool_result_content`` 가
    상한 이하를 그대로 돌려준다), 자연히 **큰 결과부터** 줄어든다. 작은 결과를
    괜히 깎지 않으면서 결정론적이다.

    평상시에는 아무것도 하지 않는다 — 총합이 예산 안이면 개별 축약 결과 그대로다.

    Args:
        results: ``{tool_call_id, name, content}`` 딕셔너리 목록.
        per_cap: 결과 하나의 최대 길이.
        total_cap: 이 턴 전체 합계의 최대 길이.

    Returns:
        같은 순서·같은 키의 새 목록. ``content`` 만 줄어든다.
    """
    def _apply(cap: int) -> List[dict]:
        return [
            {
                "tool_call_id": r.get("tool_call_id"),
                "name": r.get("name"),
                "content": trim_tool_result_content(r.get("name") or "?", r.get("content", ""), cap),
            }
            for r in results
        ]

    def _size(rows: List[dict]) -> int:
        return sum(len(r["content"]) for r in rows)

    rows = _apply(per_cap)
    if _size(rows) <= total_cap:
        return rows

    cap = per_cap
    while cap > _TOTAL_CAP_FLOOR:
        cap = max(_TOTAL_CAP_FLOOR, cap // 2)
        rows = _apply(cap)
        if _size(rows) <= total_cap:
            logger.info(
                "tool_ctx 총량 축약: 도구 %d개 · 공통상한 %d자 → 합계 %d자(예산 %d)",
                len(results), cap, _size(rows), total_cap,
            )
            return rows

    # 여기까지 왔다면 도구가 비정상적으로 많다는 뜻이다(300자 × N > 예산이려면 N>130).
    # 예산은 약속이므로 마지막엔 문자 절단으로라도 지킨다.
    share = max(1, total_cap // max(1, len(rows)))
    logger.warning(
        "tool_ctx 총량 초과 — 도구 %d개, 결과당 %d자로 강제 절단(합계 %d자 > 예산 %d)",
        len(rows), share, _size(rows), total_cap,
    )
    # 안내 문구도 길이에 포함된다 — 빼고 자르지 않으면 결과당 share 를 넘겨 예산이 또 깨진다.
    # 몫이 안내 문구보다도 짧은 극단(도구 수천 개)에서는 문구를 붙이지 않는다.
    notice = TRUNCATE_NOTICE if share > len(TRUNCATE_NOTICE) else ""
    keep = max(1, share - len(notice))
    for r in rows:
        if len(r["content"]) > share:
            r["content"] = r["content"][:keep] + notice
    return rows


# 직전 방출 워크스페이스(app-embed) 아티팩트 캐시 키 접미사.
# 도구 재호출 없는 후속턴(예: 동일 질문을 대화 문맥으로 답한 턴)에서도 iframe 을
# 재방출하기 위한 재료. tool_ctx 와 달리 절삭·삭제되지 않아 견고하다.
LAST_WS_ARTIFACT_SUFFIX = "last_ws_artifact"

# 서버측 pre-stash 문서 draft body 캐시. OCR 같은 pre-fill 결과를 source(불변 원본)와
# current(사용자 정정 누적본)로 분리한다. create 는 current 만 소비·갱신하고 source 를
# 다시 병합하지 않는다. 그래야 거절 후 재정정 턴마다 OCR 원본이 최신 수정값을 덮지 않는다.
# 기존 pending_draft 키를 current 로 유지해 배포 전 생성된 stash 와 하위호환한다.
PENDING_DRAFT_SUFFIX = "pending_draft"
PENDING_DRAFT_SOURCE_SUFFIX = "pending_draft_source"

# 사용자가 [거절]한 HITL 초안의 **도구 호출 인자 원본**. 거절은 큐 인덱스만 넘기고(hitl/graph.py)
# 초안 값을 어디에도 남기지 않아, 다음 정정 턴("9시 말고 10시로")에 되쓸 재료가 없었다 —
# 모델이 존재하지도 않는 일정을 조회하다 "찾을 수 없다"로 대화를 끊었다. 도구명 분기 없이
# 전 도메인(캘린더·메일·전자결재…) 공통이다. TTL 이 SESSION_TTL(24h)이 아닌 이유: 정정은
# 거절 직후에 온다. 길게 잡으면 한참 뒤 무관한 턴이 이 stash 를 보고 엉뚱한 카드를 다시 만든다.
REJECTED_DRAFT_SUFFIX = "rejected_draft"
# 5분 — 정정("10시로 바꿔줘")은 거절 직후 수 초~1분 안에 온다. 길게 잡으면 한참 뒤의 완결된
# 새 요청까지 정정으로 오인돼 옛 값이 섞인다(실측 2026-08-10, 30분일 때). 새 카드가 뜨는
# 시점에도 폐기하므로(orchestrator._checkpoint_pending_action) 이 TTL 은 2차 방어선이다.
REJECTED_DRAFT_TTL = 300

# **되물을 값이 남은 초안이 진행 중**이라는 신호. 값이 아니라 포인터만 담는다
# ({tool_name, template_code}) — 값 자체는 pending_draft 에 있다.
#
# 왜 필요한가: 되묻기에 짧게 답하는 턴("1번", "둘 다")은 라우터가 도메인을 못 고르거나
# 조회로 분류한다. 그러면 READ 서브런이 requires_approval 도구를 전부 걷어내
# 결재 상신 같은 승인 도구가 바인딩되지 않고, 모델은 상신한 척 문구만 쓰거나
# "일시적 오류"로 끝난다(실측 2026-08-17: 4턴 중 3턴 실패). 이 포인터가 있으면
# supervisor 가 그 턴을 실행(WRITE)으로 태워 초안 카드를 결정적으로 만든다.
#
# pending_draft 를 직접 뒤지지 않는 이유: 키가 (대화, 양식코드) 조합이고 redis_store 에
# scan/keys 가 없어 "이 대화에 수집 중 stash 가 있나"를 O(1) 로 물을 수단이 없다.
COLLECTING_DRAFT_SUFFIX = "collecting_draft"
# 30분 — 초안이 만들어지는 순간 삭제되므로(그 뒤엔 sticky 가 도메인·모드를 이어받는다)
# 이 TTL 은 2차 방어선이다. 길게 잡으면 한참 뒤의 정상 조회("지난달 경비 뭐 올렸지?")가
# WRITE 로 밀려 엉뚱한 초안 카드가 뜬다 — 거절 stash TTL 을 30분→5분으로 줄인 것과 같은 함정.
COLLECTING_DRAFT_TTL = 1800

# 메신저에 올라간 **승인 카드 포스트의 위치**({channel_id, post_id}).
#
# 왜 필요한가: 같은 초안을 채팅 패널과 메신저 양쪽에서 볼 수 있는데, 패널에서 먼저 승인하면
# 메신저 카드는 아무도 갱신해 주지 않아 버튼이 그대로 남는다. 뒤늦게 그 버튼을 누르면
# 체크포인트가 이미 소진돼(load_pending → None) "이미 처리된 요청입니다"만 뜬다.
# 이 stash 가 있으면 패널 승인 시점에 메신저 카드도 '승인 완료'로 바꿔 그 상황 자체를 없앤다.
#
# 버튼 경로는 이 값이 필요 없다(Mattermost 가 클릭된 포스트를 직접 갱신해 준다) — 콜백
# 페이로드의 post_id 는 **서명 대상이 아니라** 신뢰 경로에 들이지 않는다.
APPROVAL_POST_SUFFIX = "mm_approval_post"


def _approval_post_key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}:{APPROVAL_POST_SUFFIX}"


def _pending_draft_key(conversation_id: str, template_code: str) -> str:
    return f"conversation:{conversation_id}:{PENDING_DRAFT_SUFFIX}:{template_code}"



# 이 턴(들)에 에이전트가 **만든** 파일. 사용자가 올린 첨부(attachment_context)와 반대
# 방향이다. 도구는 tool args 로 바이트를 주고받지 못하므로(executor 는 LLM 이 만든
# 텍스트/JSON 만 전달), 파일을 만든 도구와 그걸 보내는 도구 사이를 대화 스코프 stash 로
# 잇는다 — 영수증 OCR 이 document_body 를 넘기는 것과 같은 패턴이다.
#
# **base64 를 LLM 에 노출하지 않는 것이 요점이다.** 모델이 파일 내용을 지어내 전송 도구
# 인자에 넣으면 열리지 않는 파일이 올라가고, 사용자는 승인 카드를 누른 뒤에야 알게 된다.
GENERATED_FILE_TTL = 3600


def _generated_files_key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}:generated_files"


async def save_generated_file(
    conversation_id: Optional[str],
    *,
    filename: str,
    content_b64: str,
    content_type: str,
    ttl: int = GENERATED_FILE_TTL,
) -> None:
    """방금 만든 파일을 대화 스코프에 쌓아 둔다(최근 것이 뒤).

    같은 이름으로 다시 만들면 **교체한다** — "표 다시 만들어줘" 뒤에 옛 버전이 함께
    붙어 나가면 사용자가 어느 게 최신인지 알 수 없다.

    Args:
        conversation_id: 대화 ID. 없으면 무시(파일은 도구 응답의 링크로만 전달된다).
        filename: 파일명.
        content_b64: base64 인코딩된 내용.
        content_type: MIME 타입.
        ttl: 만료(초). 기본 1시간 — 만든 턴과 보내는 턴 사이만 살면 된다.
    """
    if not conversation_id or not filename or not content_b64:
        return
    key = _generated_files_key(conversation_id)
    try:
        current = await redis_store.get(key)
        items = [f for f in (current or [])
                 if isinstance(f, dict) and f.get("filename") != filename]
        items.append({"filename": filename, "content_b64": content_b64,
                      "content_type": content_type, "created_at": time.time()})
        await redis_store.set(key, items[-5:], expire=ttl)
    except Exception:
        # 캐시는 부가기능 — 실패해도 파일 생성 응답(다운로드 링크)에는 영향이 없다.
        logger.warning("generated_file 저장 실패 conversation_id=%s", conversation_id, exc_info=True)


async def load_generated_files(
    conversation_id: Optional[str],
    filenames: Optional[list] = None,
    *,
    max_age_seconds: Optional[float] = None,
) -> list:
    """이 대화에서 만든 파일을 꺼낸다. ``filenames`` 를 주면 그 이름만 (순서 유지).

    Args:
        conversation_id: 대화 ID.
        filenames: 가져올 파일명. None 이면 전부.
        max_age_seconds: 주면 **그보다 오래된 파일은 뺀다.** 이름 없이 자동 첨부할 때
            쓴다 — stash 는 1시간 살아 있어서 지난 턴 파일이 남아 있고, 그걸 붙이면
            사용자가 요청한 것과 다른 파일이 조용히 나간다(2026-09-15 실측). 생성 시각이
            없는 옛 항목(이 필드 도입 전 저장분)도 **제외**한다 — 오배송보다 미첨부가 낫다.

    Returns:
        ``[{filename, content_b64, content_type, created_at}, ...]``. 없으면 빈 리스트.
    """
    if not conversation_id:
        return []
    try:
        items = await redis_store.get(_generated_files_key(conversation_id)) or []
    except Exception:
        logger.warning("generated_file 조회 실패 conversation_id=%s", conversation_id, exc_info=True)
        return []
    items = [f for f in items if isinstance(f, dict) and f.get("content_b64")]
    if max_age_seconds is not None:
        cutoff = time.time() - max_age_seconds
        fresh = [f for f in items if float(f.get("created_at") or 0) >= cutoff]
        if len(fresh) != len(items):
            logger.info("generated_file 자동첨부 후보에서 오래된 %d건 제외",
                        len(items) - len(fresh))
        items = fresh
    if filenames is None:
        return items
    wanted = [str(n) for n in filenames if n]
    by_name = {f.get("filename"): f for f in items}
    return [by_name[n] for n in wanted if n in by_name]


def _turn_attachments_key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}:attachments"


async def save_turn_attachments(
    conversation_id: Optional[str], attachments: Optional[list]
) -> None:
    """이 턴에 영속화된 첨부 메타를 대화 스코프로 보관한다(첨부를 이어받는 후속 턴용).

    첨부 참조를 특정 판독 도구(예: 영수증 OCR)만 만들게 두면, 모델이 사진을 직접 읽고 그 도구를
    건너뛸 때(프로바이더에 따라 그렇게 한다 — 실측 2026-09-29 gemini) 참조가 아예 생기지 않아
    나중 턴의 등록·상신에 첨부가 비어 나간다. 첨부 사실 자체는 어느 도구를 부르든 변하지
    않으므로 대화 스코프에 따로 남겨, 나중 턴의 쓰기 도구가 이어받게 한다.

    전용 키, 대화당 마지막 1건(last-write-wins), 능동 삭제 없이 TTL 만료.
    """
    if not conversation_id or not attachments:
        return
    try:
        await redis_store.set(
            _turn_attachments_key(conversation_id), list(attachments), expire=SESSION_TTL
        )
    except Exception:
        logger.warning("turn_attachments 저장 실패 conversation_id=%s", conversation_id, exc_info=True)


async def load_turn_attachments(conversation_id: Optional[str]) -> list:
    """save_turn_attachments 로 보관한 첨부 메타. 없으면 빈 목록."""
    if not conversation_id:
        return []
    try:
        value = await redis_store.get(_turn_attachments_key(conversation_id))
    except Exception:
        return []
    return value if isinstance(value, list) else []





def _pending_draft_source_key(conversation_id: str, template_code: str) -> str:
    return f"conversation:{conversation_id}:{PENDING_DRAFT_SOURCE_SUFFIX}:{template_code}"


def _collecting_draft_key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}:{COLLECTING_DRAFT_SUFFIX}"


# 저장 형식 버전. 예전에는 body 만 저장했고, 지금은 **수집 상태(slots)를 함께** 저장한다
# — 경비청구의 결제수단·카드 4자리·대상자 응답 여부는 body 에서 되살릴 수 없기 때문이다.
# 옛 레코드(버전 표시 없이 body 만 있는 것)도 읽어야 한다: 배포 중이던 대화가 끊기지 않게.
_DRAFT_RECORD_VERSION = 2


def _wrap_draft(
    document_body: dict, slots: Optional[dict], title: Optional[str] = None
) -> dict:
    """저장 형식으로 만든다 — {v, body, slots, title}.

    title 은 정정 턴이 이어받을 문서 제목이다. 없던 필드라 옛 레코드에는 비어 있고,
    읽는 쪽이 None 을 감당한다(버전은 올리지 않는다 — 올리면 옛 레코드의 slots 까지
    레거시 경로로 떨어져 되묻기 상태를 잃는다).
    """
    record = {
        "v": _DRAFT_RECORD_VERSION,
        "body": document_body,
        "slots": slots if isinstance(slots, dict) else {},
    }
    if isinstance(title, str) and title.strip():
        record["title"] = title.strip()
    return record


def _draft_title(raw: Any) -> Optional[str]:
    """저장본에서 제목을 꺼낸다 — 없던 필드라 옛 레코드는 None."""
    if isinstance(raw, dict):
        title = raw.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    return None


def _unwrap_draft(raw: Any) -> tuple:
    """저장본에서 (body, slots) 를 꺼낸다 — body 만 저장된 옛 레코드도 받는다.

    Returns:
        (body|None, slots dict). body 가 dict 가 아니면 (None, {}).
    """
    if not isinstance(raw, dict):
        return None, {}
    if raw.get("v") == _DRAFT_RECORD_VERSION:
        body = raw.get("body")
        slots = raw.get("slots")
        return (body if isinstance(body, dict) else None,
                slots if isinstance(slots, dict) else {})
    return raw, {}   # 옛 형식 — 레코드 자체가 body 였다.


# 토큰 기반 자동 압축 — max_history(사용자 저장 상한)와 독립된 LLM
# 컨텍스트 최적화 트리거. 사용자 preferences 와 무관하게 시스템 상수로 관리.
COMPACT_TRIGGER_TOKENS = 6000          # 합계 토큰 추정 초과 시 비동기 압축
COMPACT_KEEP_RECENT_MESSAGES = 8       # 최근 4턴(=8 메시지) verbatim 보존
COMPACT_MIN_OLD_MESSAGES = 4           # 압축 대상이 이보다 적으면 스킵
COMPACT_LOCK_TTL_SECONDS = 60
HYDRATE_LOCK_TTL_SECONDS = 10          # 이력 폴백이 압축 락을 빌려 쓰는 시간(해제하지 않는다)
_HISTORY_CLEARED_TTL_SECONDS = 7 * 24 * 3600   # session_ttl_seconds 최대값만큼 남긴다
COMPACT_SUMMARY_MARKER = "__compact_summary__"  # Message.name 마커
COMPACT_ANNOUNCE_TTL = 3600            # 통지 키 TTL (1시간)
_TIKTOKEN_HEURISTIC_RATIO = 0.6        # tiktoken 실패 시 한국어 혼합 텍스트 추정 계수


async def save_last_workspace_artifact(
    conversation_id: Optional[str], artifact: dict, ttl: int = SESSION_TTL
) -> None:
    """직전에 방출한 워크스페이스(app-embed) 아티팩트를 대화별로 캐시한다.

    ``tool_ctx`` 와 달리 (1) 절삭되지 않고 (2) 도구 호출 없는 턴에도 폐기되지 않아,
    "동일 질문 재호출 → 도구 미호출 → iframe 유실" 을 막는 재방출 소스로 쓴다.

    Args:
        conversation_id: 대화 ID (Redis 키 prefix 구성). None/빈값이면 무시.
        artifact: build_action_artifact 가 만든 app-embed 아티팩트(resourceUrl 보유).
        ttl: 캐시 만료(초). 기본은 history 와 동일한 SESSION_TTL.
    """
    if not conversation_id or not isinstance(artifact, dict):
        return
    try:
        await redis_store.set(
            f"conversation:{conversation_id}:{LAST_WS_ARTIFACT_SUFFIX}",
            artifact,
            expire=ttl,
        )
    except Exception:
        # 캐시는 부가기능 — 실패해도 응답 스트림에 영향 주지 않는다.
        logger.warning(
            "last_ws_artifact 저장 실패 conversation_id=%s", conversation_id, exc_info=True
        )


# 직전 턴이 실행한 라우팅 도메인들. 후속 질의("근거가 뭐야", "그거 자세히")는 문장에 도메인 단서가 없어
# 라우터가 추측하다 틀리면 워커 울타리에 갇힌다 — supervisor 가 follow_up 판정 시 이 목록을 그대로 상속한다.
# tool_ctx 의 도구명에서 도메인을 역산하던 것(prior_domain)은 멀티 도메인 턴에서 하나만 남아 정보를 잃었다.
LAST_ROUTE_SUFFIX = "last_route"


# 이번 턴 라우트를 답 저장까지 미뤄 두는 자리. supervisor 는 여기에 적기만 하고, 실제 저장은
# save_logged_in_turn 이 턴을 저장해 대화 id 가 확정된 뒤에 한다. 그래야 (1) 실패해 저장되지 않은
# 턴의 라우트가 남지 않고 (2) 새 대화 첫 턴(대화 id 가 답 저장 때 생긴다)도 라우트가 남는다.
# 값은 dict 를 제자리 수정한다 — 자식 Task 에서 적어도 부모로 보인다(ContextVar 는 복사된다).
_turn_route: ContextVar[Optional[dict]] = ContextVar("turn_route", default=None)


def begin_turn_route() -> None:
    """턴 시작 — 라우트 보류 자리를 새로 연다(orchestrator 가 턴마다 부른다)."""
    _turn_route.set({})


def record_turn_route(domains: List[str]) -> bool:
    """이번 턴 라우트를 보류해 둔다. 보류 자리가 없으면(턴 밖 호출) False — 호출자가 직접 저장한다."""
    holder = _turn_route.get()
    if holder is None:
        return False
    holder["domains"] = [d for d in (domains or []) if isinstance(d, str)]
    return True


async def _flush_turn_route(conversation_id: Optional[str]) -> None:
    holder = _turn_route.get()
    if not holder or "domains" not in holder or not conversation_id:
        return
    domains = holder.pop("domains")
    await save_last_route(conversation_id, domains)


async def save_last_route(conversation_id: Optional[str], domains: List[str], ttl: int = SESSION_TTL) -> None:
    """이번 턴이 실행한 도메인 목록을 대화별로 남긴다. 무스코프 폴백이면 빈 목록을 남겨 '직전은 도메인 없음'을 기록한다."""
    if not conversation_id:
        return
    try:
        await redis_store.set(
            f"conversation:{conversation_id}:{LAST_ROUTE_SUFFIX}",
            {"domains": [d for d in (domains or []) if isinstance(d, str)], "saved_at": int(time.time())},
            expire=ttl,
        )
    except Exception:
        logger.warning("last_route 저장 실패 conversation_id=%s", conversation_id, exc_info=True)


async def load_last_route(conversation_id: Optional[str]) -> List[str]:
    """직전 턴의 실행 도메인 목록. 없으면 빈 리스트."""
    if not conversation_id:
        return []
    try:
        data = await redis_store.get(f"conversation:{conversation_id}:{LAST_ROUTE_SUFFIX}")
    except Exception:
        return []
    doms = (data or {}).get("domains") if isinstance(data, dict) else None
    return [d for d in doms if isinstance(d, str)] if isinstance(doms, list) else []


async def load_last_workspace_artifact(conversation_id: Optional[str]) -> Optional[dict]:
    """대화별 캐시된 직전 워크스페이스 아티팩트를 반환한다 (없으면 None)."""
    if not conversation_id:
        return None
    try:
        artifact = await redis_store.get(
            f"conversation:{conversation_id}:{LAST_WS_ARTIFACT_SUFFIX}"
        )
        return artifact if isinstance(artifact, dict) else None
    except Exception:
        return None


async def save_pending_draft(
    conversation_id: Optional[str],
    template_code: Optional[str],
    document_body: dict,
    ttl: int = SESSION_TTL,
    slots: Optional[dict] = None,
    resume_tool: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """pre-fill 문서 body 로 source 원본과 current 편집본을 함께 초기화한다.

    ``tool_ctx`` 와 달리 (1) 절삭되지 않고 (2) 도구 호출 없는 턴에도 폐기되지 않아, 만든 턴과
    create 호출 턴이 달라도 원본 body 를 잃지 않는다. 이후 사용자 정정은
    ``save_pending_draft_revision`` 으로 current 에만 누적한다. 새 OCR/pre-fill 이 호출되면
    새 문서로 간주해 source/current 둘 다 다시 초기화한다.

    Args:
        conversation_id: 대화 ID. None/빈값이면 무시.
        template_code: 양식 코드 (키 분리 — 다른 양식 create 에 섞이지 않게). 빈값이면 무시.
        document_body: 저장할 문서 body dict.
        ttl: 캐시 만료(초). 기본은 history 와 동일한 SESSION_TTL.
        slots: 수집 상태(경비의 결제수단·대상자 응답 여부 등). 안 쓰는 양식은 생략.
        resume_tool: 이 초안을 이어받을 승인 도구 이름(예: 문서 상신 도구).
            주면 "되물을 값이 남은 초안 진행 중" 포인터도 함께 기록해, 짧은 답변 턴이
            조회로 오분류돼 그 도구가 바인딩되지 않는 사고를 막는다.
    """
    if not conversation_id or not template_code or not isinstance(document_body, dict):
        return
    record = _wrap_draft(document_body, slots, title)
    if resume_tool:
        await save_collecting_draft(conversation_id, resume_tool, template_code)
    try:
        await redis_store.set(
            _pending_draft_source_key(conversation_id, template_code),
            record,
            expire=ttl,
        )
        await redis_store.set(
            _pending_draft_key(conversation_id, template_code), record, expire=ttl
        )
    except Exception:
        # 캐시는 부가기능 — 실패해도 생산자 응답에 영향 주지 않는다.
        logger.warning(
            "pending_draft 저장 실패 conversation_id=%s template=%s",
            conversation_id, template_code, exc_info=True,
        )


async def save_pending_draft_revision(
    conversation_id: Optional[str],
    template_code: Optional[str],
    document_body: dict,
    ttl: int = SESSION_TTL,
    slots: Optional[dict] = None,
    title: Optional[str] = None,
) -> None:
    """사용자 정정이 반영된 최신 current 편집본만 저장한다(source 는 보존).

    slots·title 을 안 주면 기존 값을 이어받는다 — 지워지면 같은 질문을 다시 하게 된다.
    """
    if not conversation_id or not template_code or not isinstance(document_body, dict):
        return
    key = _pending_draft_key(conversation_id, template_code)
    if slots is None or title is None:
        try:
            raw = await redis_store.get(key)
        except Exception:
            raw = None
        if slots is None:
            _, slots = _unwrap_draft(raw)
        if title is None:
            title = _draft_title(raw)
    try:
        await redis_store.set(key, _wrap_draft(document_body, slots, title), expire=ttl)
    except Exception:
        logger.warning(
            "pending_draft revision 저장 실패 conversation_id=%s template=%s",
            conversation_id, template_code, exc_info=True,
        )


async def load_pending_draft(
    conversation_id: Optional[str], template_code: Optional[str]
) -> Optional[dict]:
    """양식별 캐시된 문서 draft **body** 를 반환한다 (없으면 None).

    저장 형식이 바뀌어도 body 만 돌려준다. 수집 상태는 load_pending_draft_slots 로.
    """
    if not conversation_id or not template_code:
        return None
    try:
        body, _ = _unwrap_draft(
            await redis_store.get(_pending_draft_key(conversation_id, template_code))
        )
        return body
    except Exception:
        return None


async def load_pending_draft_title(
    conversation_id: Optional[str], template_code: Optional[str]
) -> Optional[str]:
    """정정 턴이 이어받을 문서 제목을 반환한다. 저장 전 초안·옛 레코드는 None.

    제목은 OCR 시점엔 없다(그때는 문서가 아직 없다). create 가 처음 제목을 정할 때
    함께 적재되고, 그 뒤 정정 턴은 모델이 제목을 다시 짓지 않아도 이어받는다.
    """
    if not conversation_id or not template_code:
        return None
    try:
        return _draft_title(
            await redis_store.get(_pending_draft_key(conversation_id, template_code))
        )
    except Exception:
        return None


async def load_pending_draft_slots(
    conversation_id: Optional[str], template_code: Optional[str]
) -> dict:
    """current 편집본의 수집 상태(slots)를 반환한다. 없으면 빈 dict(옛 레코드 포함)."""
    if not conversation_id or not template_code:
        return {}
    try:
        _, slots = _unwrap_draft(
            await redis_store.get(_pending_draft_key(conversation_id, template_code))
        )
        return slots
    except Exception:
        return {}


async def load_pending_draft_source(
    conversation_id: Optional[str], template_code: Optional[str]
) -> Optional[dict]:
    """비교·복구용 pre-fill 원본 body 를 반환한다. create 경로에서는 사용하지 않는다."""
    if not conversation_id or not template_code:
        return None
    try:
        body, _ = _unwrap_draft(
            await redis_store.get(_pending_draft_source_key(conversation_id, template_code))
        )
        return body
    except Exception:
        return None


def _pending_approval_line_key(conversation_id: str, template_code: str) -> str:
    return f"conversation:{conversation_id}:pending_approval_line:{template_code}"


async def save_pending_approval_line(
    conversation_id: Optional[str], template_code: Optional[str], approval_line: Optional[dict]
) -> None:
    """드라이런이 디렉터리로 확정한 결재선을 양식 단위로 남긴다 — 승인 confirm 이 우선 사용한다.

    HITL pending 에는 **모델 원본 인자**가 저장되고 confirm 에는 발화·임시 상태가 없어, 드라이런이
    회사 지정을 복원해 교정한 결재선이 confirm 에서 다시 기안자 회사 사람으로 되돌아갔다
    (2026-09-22 리뷰). 실패는 무시 — 저장이 안 되면 confirm 이 재해석으로 폴백한다.
    """
    if not conversation_id or not template_code or not isinstance(approval_line, dict):
        return
    try:
        await redis_store.set(
            _pending_approval_line_key(conversation_id, template_code), approval_line, expire=SESSION_TTL
        )
    except Exception:
        logger.debug("pending_approval_line 저장 실패 conversation_id=%s", conversation_id, exc_info=True)


async def load_pending_approval_line(
    conversation_id: Optional[str], template_code: Optional[str]
) -> Optional[dict]:
    """드라이런이 남긴 확정 결재선. 없으면 None."""
    if not conversation_id or not template_code:
        return None
    try:
        value = await redis_store.get(_pending_approval_line_key(conversation_id, template_code))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


async def clear_pending_draft(
    conversation_id: Optional[str], template_code: Optional[str]
) -> None:
    """양식별 source/current 캐시를 삭제한다 (생성 완료 후 재사용 방지). 실패는 무시."""
    if not conversation_id or not template_code:
        return
    await clear_collecting_draft(conversation_id)
    try:
        await redis_store.delete(_pending_draft_key(conversation_id, template_code))
        await redis_store.delete(_pending_draft_source_key(conversation_id, template_code))
        await redis_store.delete(_pending_approval_line_key(conversation_id, template_code))
    except Exception:
        logger.debug(
            "pending_draft 삭제 실패 conversation_id=%s template=%s",
            conversation_id, template_code, exc_info=True,
        )


async def save_collecting_draft(
    conversation_id: Optional[str], tool_name: Optional[str], template_code: Optional[str] = None
) -> None:
    """"되물을 값이 남은 초안 진행 중" 포인터를 기록한다 (값은 담지 않는다).

    tool_name 은 그 초안을 이어받을 승인 도구다 — supervisor 가 여기서 도메인을 유도하므로
    (거절 stash 와 동일한 모양) 도구명 분기 없이 전 도메인 공통으로 동작한다.
    """
    if not conversation_id or not tool_name:
        return
    try:
        await redis_store.set(
            _collecting_draft_key(conversation_id),
            {"tool_name": tool_name, "template_code": template_code},
            expire=COLLECTING_DRAFT_TTL,
        )
    except Exception:
        # 부가기능 — 실패하면 기존(운에 맡기는) 동작으로 자연 폴백한다.
        logger.warning(
            "collecting_draft 저장 실패 conversation_id=%s", conversation_id, exc_info=True
        )


async def load_collecting_draft(conversation_id: Optional[str]) -> Optional[dict]:
    """진행 중인 수집 초안 포인터를 반환한다 (없으면 None)."""
    if not conversation_id:
        return None
    try:
        value = await redis_store.get(_collecting_draft_key(conversation_id))
        return value if isinstance(value, dict) and value.get("tool_name") else None
    except Exception:
        return None


async def clear_collecting_draft(conversation_id: Optional[str]) -> None:
    """수집 포인터를 폐기한다.

    **초안이 만들어진 시점에 반드시 호출한다.** 그 뒤로는 sticky(직전 턴 도구)가 도메인·모드를
    이어받으므로 승격이 불필요하고, 남겨 두면 승인 후의 정상 조회까지 WRITE 로 밀려 엉뚱한
    초안 카드가 뜬다.
    """
    if not conversation_id:
        return
    try:
        await redis_store.delete(_collecting_draft_key(conversation_id))
    except Exception:
        logger.debug(
            "collecting_draft 삭제 실패 conversation_id=%s", conversation_id, exc_info=True
        )


def _rejected_draft_key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}:{REJECTED_DRAFT_SUFFIX}"


async def save_rejected_draft(
    conversation_id: Optional[str],
    tool_name: Optional[str],
    arguments: Optional[dict],
    fields: Optional[list] = None,
    ttl: int = REJECTED_DRAFT_TTL,
) -> None:
    """방금 거절된 초안의 도구명·인자 원본을 대화 단위로 stash 한다.

    다음 턴이 이 값을 읽어 "직전 거절된 초안" 컨텍스트로 주입하고, 사용자의 정정만 반영해
    같은 도구를 confirm=False 로 재호출한다(→ 새 승인 카드).

    ``tool_ctx`` 를 쓰지 않는 이유: (1) 결과가 상한(MAX_TOOL_RESULT_CHARS)에서 절삭되고 (2) 중간에 도구를 쓴 턴이
    끼면 덮이며 (3) 주입 앵커가 ``_persist_action_outcome`` 이 심는 가짜 user 턴
    ``"(승인 실행됨)"`` 이라 모델이 "사용자가 실행을 요청했다"로 오독한다.

    Args:
        conversation_id: 대화 ID. None/빈값이면 무시.
        tool_name: 거절된 액션의 도구명(예: ``calendar_create_event``). 빈값이면 무시.
        arguments: 그 도구 호출 인자 원본. **표시용 preview.fields 가 아니라 이 값이 권위 소스**
            — fields 는 사람이 읽도록 포맷돼 있어(예: 종료가 리터럴 ``"(시작 +1시간)"``)
            그대로 되먹이면 ISO 파싱 오류가 난다.
        fields: 사람이 읽는 요약 필드(로그·메모용, 선택).
        ttl: 캐시 만료(초). 기본 30분.
    """
    if not conversation_id or not tool_name or not isinstance(arguments, dict):
        return
    try:
        await redis_store.set(
            _rejected_draft_key(conversation_id),
            {"tool_name": tool_name, "arguments": arguments, "fields": fields or []},
            expire=ttl,
        )
        # 새 거절이니 지목 한 번을 다시 허용한다(앞 거절의 소비 표시를 지운다).
        await redis_store.delete(_rejected_pin_key(conversation_id))
    except Exception:
        # stash 는 부가기능 — 실패해도 /resume 응답에 영향 주지 않는다.
        logger.warning(
            "rejected_draft 저장 실패 conversation_id=%s tool=%s",
            conversation_id, tool_name, exc_info=True,
        )


async def load_rejected_draft(conversation_id: Optional[str]) -> Optional[dict]:
    """직전 거절된 초안 stash 를 반환한다. 없으면 None.

    Returns:
        ``{"tool_name": str, "arguments": dict, "fields": list}`` 또는 None.
    """
    if not conversation_id:
        return None
    try:
        data = await redis_store.get(_rejected_draft_key(conversation_id))
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("tool_name"):
        return None
    return data


def _rejected_pin_key(conversation_id: str) -> str:
    return f"conversation:{conversation_id}:rejected_pin_used"


async def rejected_pin_used(conversation_id: Optional[str]) -> bool:
    """거절 stash 의 **도구 지목**을 이미 한 번 썼는지.

    지목(`react_graph._pin_tool_for_rejected_draft`)은 거절 **직후 정정 턴** 하나를 위한
    장치인데, stash 가 TTL(5분) 동안 살아 있어 그 사이 모든 턴이 같은 도구로 묶였다.
    한 번 쓰고 표시해 두면 다음 턴부터는 모델이 도구를 고를 수 있다(ANY 강제는 유지).

    stash 본체가 아니라 **별도 키**에 둔다 — 본체를 다시 쓰면 TTL 이 5분으로 되살아나
    창을 오히려 늘린다(store 에 남은 TTL 을 읽는 수단이 없다).
    """
    if not conversation_id:
        return False
    try:
        return bool(await redis_store.exists(_rejected_pin_key(conversation_id)))
    except Exception:
        return False


async def mark_rejected_pin_used(conversation_id: Optional[str]) -> None:
    """지목을 소비했다고 표시한다. 실패는 삼킨다 — 종전처럼 계속 지목될 뿐이라 안전 쪽이다."""
    if not conversation_id:
        return
    try:
        await redis_store.set(
            _rejected_pin_key(conversation_id), True, expire=REJECTED_DRAFT_TTL
        )
    except Exception:
        logger.warning(
            "rejected_draft 지목 소비 표시 실패 conversation_id=%s",
            conversation_id, exc_info=True,
        )


async def clear_rejected_draft(conversation_id: Optional[str]) -> None:
    """거절 stash 를 삭제한다(승인으로 확정됐거나 더는 정정 대상이 아닐 때). 실패는 무시."""
    if not conversation_id:
        return
    try:
        await redis_store.delete(_rejected_draft_key(conversation_id))
        await redis_store.delete(_rejected_pin_key(conversation_id))
    except Exception:
        logger.debug(
            "rejected_draft 삭제 실패 conversation_id=%s", conversation_id, exc_info=True
        )


async def save_approval_post(
    conversation_id: Optional[str], channel_id: str, post_id: str,
    *, pending_id: str = "", title: str = "", fields: Optional[list] = None,
) -> None:
    """메신저에 올린 승인 카드의 위치를 기억한다. 실패는 무시(카드 게시를 막지 않는다).

    TTL 은 history 와 같은 ``SESSION_TTL`` — 초안이 살아 있을 수 있는 기간과 맞춘다.
    값이 없어져도 기능이 예전처럼 동작할 뿐이라(패널 승인 시 메신저 카드가 안 바뀜) 안전하다.

    Args:
        conversation_id: 대화방 UUID
        channel_id: 카드가 올라간 채널
        post_id: 카드 포스트 ID
        pending_id: 카드가 승인할 대기 항목 id — 결과를 반영할 카드가 맞는지 대조한다.
        title, fields: 카드 본문 재료 — 새 카드가 이 카드를 대체할 때 다시 그린다(그때는 대기 항목이
            이미 새 것으로 바뀌어 이 카드의 draft_view 가 없다).
    """
    if not conversation_id or not post_id:
        return
    record: Dict[str, Any] = {"channel_id": channel_id, "post_id": post_id}
    if pending_id:
        record["pending_id"] = pending_id
    if title:
        record["title"] = title
    if fields:
        record["fields"] = fields
    try:
        await redis_store.set(
            _approval_post_key(conversation_id),
            record,
            expire=SESSION_TTL,
        )
    except Exception:
        logger.debug(
            "approval_post 저장 실패 conversation_id=%s", conversation_id, exc_info=True
        )


async def load_approval_post(conversation_id: Optional[str]) -> Optional[dict]:
    """메신저 승인 카드 위치를 반환한다. 없으면 None(= 메신저에서 시작한 대화가 아님).

    Returns:
        ``{"channel_id", "post_id", "pending_id"?, "title"?, "fields"?}`` 또는 None.
    """
    if not conversation_id:
        return None
    try:
        data = await redis_store.get(_approval_post_key(conversation_id))
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("post_id"):
        return None
    return data


async def clear_approval_post(conversation_id: Optional[str]) -> None:
    """승인 카드 위치 stash 를 삭제한다(이미 결과가 반영됐을 때). 실패는 무시."""
    if not conversation_id:
        return
    try:
        await redis_store.delete(_approval_post_key(conversation_id))
    except Exception:
        logger.debug(
            "approval_post 삭제 실패 conversation_id=%s", conversation_id, exc_info=True
        )


async def load_prev_tool_names(conversation_id: Optional[str]) -> List[str]:
    """직전 턴 도구 호출 이름 목록을 Redis tool_ctx 에서 직접 읽는다(후속턴 도메인 상속용).

    conversation_id 로 직접 읽어 load_history 의 tool_ctx LLM 주입(dormant 가능)과 무관하게
    견고하다. 없으면 빈 리스트. supervisor 의 prior_domain 도출에 쓰인다.
    """
    if not conversation_id:
        return []
    try:
        tool_ctx = await redis_store.get(f"conversation:{conversation_id}:tool_ctx")
    except Exception:
        logger.debug("tool_ctx 읽기 실패 (conversation_id=%s)", conversation_id, exc_info=True)
        return []
    if not tool_ctx:
        return []
    calls = tool_ctx.get("tool_calls") or []
    return [c.get("name") for c in calls if isinstance(c, dict) and c.get("name")]


def _utc_now_naive() -> datetime:
    """맥락 비움 워터마크 시각 — ``agent.messages.created_at`` 과 같은 naive UTC.

    ``datetime.now()`` 는 컨테이너 로컬 시각이라 NKS(TZ=Asia/Seoul)에서 9시간 앞서고, 이 값으로
    이력을 거르면(``fetch_recent_turns(since=...)``) 비운 뒤 9시간 동안의 대화가 되살아나지 않는다.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _history_cleared_key(key: str) -> str:
    """맥락을 비운 시각(워터마크). ``/compact`` 가 Redis 만 지우고 PG 는 남기므로, 폴백이 그
    약속("이전 대화를 기억하지 않는다")을 되살려 깨지 않게 이 시각 이후 턴만 복원한다."""
    return f"{key}:history_cleared_at"


def _compact_lock_key(key: str) -> str:
    """이 history 키의 압축 락. 압축과 폴백이 같은 문을 쓰게 한다."""
    return f"{key}:compact_lock"


async def _hydrate_history(key: str, conversation_id: str, user_sub: str) -> List[dict]:
    """Redis 가 비었을 때 Postgres 에서 이력 페이로드를 되살린다. 없으면 빈 목록.

    이력은 Redis 에만 24시간(``SESSION_TTL``) 산다. 그 뒤 같은 대화방에서 이어 물으면 이력이
    통째로 없어 "어제 그거" 가 안 됐다 — 턴 자체는 agent.messages 에 영구히 있는데 아무도 읽지
    않았을 뿐이다. 사용자 프로필(services/user_profile.get_profile)의 캐시 미스 → DB → 되적재
    패턴을 그대로 따른다.

    **정화하지 않은 원문을 돌려준다.** 호출자(load_history)가 Redis 경로와 같은 자리에서
    ``_strip_guard_artifact`` 를 태우므로, 여기서 미리 정화하면 Redis 에 적재되는 내용이 PG 와
    달라진다(프롬프트 프리픽스 캐시 전제도 깨진다).

    **한 턴은 통째로 넣거나 통째로 뺀다.** 질문만 있고 답이 없는 턴을 반쪽만 넣으면 user 메시지가
    연달아 두 개 생겨 역할 교대가 깨진다.

    **압축·다른 폴백과 겹치지 않게 한다.** ``_compact_history`` 는 delete → rpush 라 리스트가
    잠깐 비고 백그라운드로 돈다. 그 틈에 되적재하면 압축 결과 위에 옛 이력이 얹힌다. 같은 대화에
    동시 요청이 둘이면(패널 연타, 패널+메신저) 둘 다 비었다고 보고 둘 다 적재해 이력이 두 배가
    된다. 그래서 **압축 락을 set_nx 로 같이 잡는다** — 한 명령으로 둘 다 막힌다. 락은 풀지 않고
    TTL 로 만료시킨다(풀면 그 뒤 압축이 새로 잡은 락을 지울 수 있다). 못 잡으면 이번 턴은 맥락만
    쓰고 적재는 다음 턴으로 미룬다.

    Args:
        key: history Redis 키("conversation:UUID").
        conversation_id: 대화방 UUID.
        user_sub: 소유자 sub — 남의 대화는 조회가 한 건도 돌려주지 않는다.

    Returns:
        ``[{"role","content"}, ...]`` 오래된 순, 정화 전. 실패·없음이면 [].
    """
    if not getattr(settings, "enable_history_hydration", True):
        return []
    try:
        prefs = await get_user_preferences(user_sub)
        max_history = prefs.max_history if prefs else MAX_HISTORY
        ttl = prefs.session_ttl_seconds if prefs else SESSION_TTL

        since = None
        cleared = await redis_store.get(_history_cleared_key(key))
        if cleared:
            try:
                since = datetime.fromisoformat(str(cleared))
            except (TypeError, ValueError):
                since = None

        from agent_arch.db.repository.messages import fetch_recent_turns

        turns = await fetch_recent_turns(conversation_id, user_sub, max_history // 2, since=since)
        if not turns:
            return []

        raw: List[dict] = []
        for t in turns:
            q = str(t.get("query") or "").strip()
            a = str(t.get("answer") or "").strip()
            if not (q and a):          # 반쪽 턴은 통째로 뺀다(역할 교대 보존)
                continue
            raw += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        if not raw:
            return []

        # 압축 요약은 PG 에 남지 않는다 — 길어서 압축됐던 대화를 원문으로 되살리면 압축 전보다
        # 커질 수 있다. 압축이 트리거되는 그 문턱으로 오래된 쌍부터 덜어낸다.
        while len(raw) > 2 and _estimate_tokens([Message(**m) for m in raw]) > COMPACT_TRIGGER_TOKENS:
            raw = raw[2:]

        wrote = False
        if await redis_store.set_nx(_compact_lock_key(key), "1", expire=HYDRATE_LOCK_TTL_SECONDS):
            await redis_store.rpush(key, *raw)
            await redis_store.ltrim(key, -max_history, -1)
            await redis_store.expire(key, ttl)
            wrote = True
        logger.info(
            "이력 폴백: %d턴 복원 (conversation=%s, 메시지=%d, 되적재=%s)",
            len(turns), conversation_id, len(raw), wrote,
        )
        return raw
    except Exception:
        logger.warning("이력 폴백 실패(무시) conversation=%s", conversation_id, exc_info=True)
        return []


async def load_history(
    key: str, *, conversation_id: Optional[str] = None, user_sub: Optional[str] = None
) -> List[Message]:
    """Redis에서 대화 히스토리를 불러온다.

    직전 1턴 도구 컨텍스트가 `{key}:tool_ctx` 에 저장되어 있으면 마지막
    user 메시지 *직후* 에 다음 메시지들을 삽입하여 LLM 이 직전 도구 호출과 결과를
    볼 수 있게 한다.
        - Message(role="assistant", content="", tool_calls=[...])
        - Message(role="tool", content=..., tool_call_id=..., name=...) (N개)
    결과적으로 LLM 이 보는 순서는:
        ..., 직전 user, [tool_call assistant], [tool result], 직전 assistant final, ...

    Args:
        key: 완전한 Redis 키. 호출자가 prefix를 결정한다.
             로그인: f"conversation:{conversation_id}"
             게스트: f"guest:{guest_id}"

    Returns:
        오래된 순으로 정렬된 메시지 리스트. tool_ctx 가 있으면 직전 1턴이 풀
        컨텍스트로 복원됨. 키가 없거나 연결 실패 시 빈 리스트.
    """
    raw = await redis_store.lrange(key)
    messages = [_strip_guard_artifact(Message(**m)) for m in raw]
    if not messages:
        # Redis 는 24시간뿐이다. 대화는 Postgres 에 영구히 있으므로 되살린다(_hydrate_history).
        # 인자를 안 받은 호출부(게스트 등)는 예전 그대로 빈 목록이다.
        if not (conversation_id and user_sub):
            return messages
        raw = await _hydrate_history(key, conversation_id, user_sub)
        if not raw:
            return messages
        # 되살린 이력에는 tool_ctx 를 붙이지 않는다. 그 값이 살아 있다면 리스트만 사라진
        # 비정상 상태이고, 남의 턴 도구 결과가 최신 user 메시지 뒤에 잘못 붙는다.
        return [_strip_guard_artifact(Message(**m)) for m in raw]

    tool_ctx = await redis_store.get(f"{key}:tool_ctx")
    if not tool_ctx or not tool_ctx.get("tool_calls"):
        return messages

    # 마지막 user 메시지 위치 찾기 (뒤에서부터)
    last_user_idx: Optional[int] = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].role == "user":
            last_user_idx = i
            break
    if last_user_idx is None:
        logger.warning(
            "tool_ctx 가 있지만 history 에 user 메시지가 없음 (key=%s) — 주입 스킵",
            key,
        )
        return messages

    # 도구 결과는 그것을 만든 턴의 user 메시지 바로 뒤에 둔다. 마지막 user 뒤에 붙이면, 도구 없는
    # 턴(되묻기 등)을 지난 뒤 오래된 결과가 직전 질문을 위해 조회한 것처럼 보였다.
    anchor = tool_ctx.get("anchor")
    if anchor:
        anchor_idx = next(
            (i for i in range(len(messages) - 1, -1, -1)
             if messages[i].role == "user" and _turn_anchor(messages[i].content or "") == anchor),
            None,
        )
        if anchor_idx is None:
            # 만든 턴이 압축·트림으로 이력에서 빠졌다 — 붙일 자리가 없으면 주입하지 않는다.
            logger.debug("tool_ctx 의 원 턴이 이력에 없음 (key=%s) — 주입 스킵", key)
            return messages
        last_user_idx = anchor_idx

    tool_calls = tool_ctx["tool_calls"]
    tool_results = tool_ctx.get("tool_results", [])

    inject: List[Message] = [
        Message(role="assistant", content="", tool_calls=tool_calls),
    ]
    for tr in tool_results:
        inject.append(Message(
            role="tool",
            content=tr.get("content", ""),
            name=tr.get("name"),
            tool_call_id=tr.get("tool_call_id"),
        ))

    logger.debug(
        "tool_ctx 주입: key=%s tool_calls=%d tool_results=%d",
        key, len(tool_calls), len(tool_results),
    )
    return messages[: last_user_idx + 1] + inject + messages[last_user_idx + 1 :]


async def clear_history(key: str) -> None:
    """Redis 히스토리 키 1개를 삭제하고 맥락 비움 워터마크를 남긴다.

    워터마크가 없으면 다음 턴에 이력 폴백(_hydrate_history)이 PG 에서 전부 되살려, 비웠다는
    사실 자체가 무효가 된다.

    Args:
        key: 완전한 Redis 키.
    """
    await redis_store.delete(key)
    await redis_store.set(
        _history_cleared_key(key), _utc_now_naive().isoformat(),
        expire=_HISTORY_CLEARED_TTL_SECONDS,
    )


async def append_turn(
    key: str,
    user_input: str,
    assistant_output: str,
    user_sub: Optional[str] = None,
    tool_calls: Optional[List[dict]] = None,
    tool_results: Optional[List[dict]] = None,
) -> None:
    """대화 LIST에 RPUSH한다. 트림·TTL은 사용자 preferences 기반(없으면 전역 폴백).

    직전 1턴 도구 컨텍스트 캐싱:
        tool_calls 가 있으면 이번 턴의 tool_calls + tool_results 를 별도 키
        `{key}:tool_ctx` 에 STRING(JSON) 으로 저장한다. TTL 은 history LIST 와
        동일한 값(prefs.session_ttl_seconds 또는 전역 폴백)을 사용하여 사용자
        설정이 자동 반영된다.

        tool_calls 가 없으면(도구 호출 없는 응답 — 주로 되물음·확인 질문) 이전
        tool_ctx 를 바로 지우지 않는다. 되묻는 턴이 직전에 조회한 대상(초점 엔티티)을
        지워버려 후속 "방금 것" 이 초점을 잃던 멀티턴 그라운딩 실패를 막기 위함이다.
        다만 도구 없는 턴이 TOOL_CTX_MAX_IDLE_TURNS 를 넘으면 버리고, 주입은 그 결과를
        만든 턴(anchor) 바로 뒤에 한다.

    Args:
        key: 완전한 Redis 키.
        user_input: 사용자 입력 텍스트.
        assistant_output: 어시스턴트 응답 텍스트.
        user_sub: 로그인 사용자 sub. preferences 조회용. 게스트는 None.
        tool_calls: 이번 턴에서 LLM 이 호출한 tool_calls 리스트 (ToolCall.model_dump()).
        tool_results: 각 tool_calls 에 대응하는 결과 리스트.
            형식: [{"tool_call_id": str, "name": str, "content": str}]
    """
    if user_sub:
        prefs = await get_user_preferences(user_sub)
        max_history = prefs.max_history if prefs else MAX_HISTORY
        ttl = prefs.session_ttl_seconds if prefs else SESSION_TTL
    else:
        max_history = MAX_HISTORY
        ttl = SESSION_TTL

    await redis_store.rpush(
        key,
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant_output},
    )
    await redis_store.ltrim(key, -max_history, -1)
    await redis_store.expire(key, ttl)

    # ── 직전 1턴 도구 컨텍스트 ──────────────────────────────────────
    # TTL 은 history LIST 와 동일한 ttl 변수 사용 → 사용자 preferences 자동 반영.
    tool_ctx_key = f"{key}:tool_ctx"
    if tool_calls:
        # 개별 상한 + **총량 상한**. 개별만 두면 최악이 안 묶인다 — read_file 하나가
        # 30,000자까지 나오므로 "검색 + 파일 2개" 턴이면 9만자가 그대로 이월된다.
        trimmed_results = trim_tool_results(tool_results or [])
        payload = {
            "tool_calls": tool_calls,
            "tool_results": trimmed_results,
            "saved_at": int(time.time()),
            "anchor": _turn_anchor(user_input),
            "idle_turns": 0,
        }
        await redis_store.set(tool_ctx_key, payload, expire=ttl)
        logger.debug(
            "tool_ctx 저장: key=%s tool_calls=%d ttl=%ds",
            tool_ctx_key, len(tool_calls), ttl,
        )
    else:
        # 도구 호출 없는 응답(주로 되물음·확인 질문)에도 직전 도구 턴 컨텍스트를 유지한다.
        # 예전엔 여기서 tool_ctx 를 삭제했는데, 에이전트가 "어느 대상?" 하고 되묻는
        # 턴(도구 미호출)이 바로 그 대상(직전 조회한 엔티티)을 지워버려, 사용자가
        # "방금 것" 이라 답해도 초점을 잃는 멀티턴 그라운딩 실패가 있었다. 삭제 대신
        # TTL 을 history LIST 와 맞춰 갱신해 대화가 이어지는 동안 조기 만료를 막는다.
        # 새 도구 턴이 오면 위 분기에서 덮어쓰인다. 도구 없는 턴이 TOOL_CTX_MAX_IDLE_TURNS 를 넘으면
        # 버린다 — 주제가 바뀐 뒤에도 남아 직전 도메인·sticky 판단을 오래된 턴 기준으로 끌고 갔다.
        prev = await redis_store.get(tool_ctx_key)
        if isinstance(prev, dict) and prev.get("tool_calls"):
            idle = int(prev.get("idle_turns") or 0) + 1
            if idle > TOOL_CTX_MAX_IDLE_TURNS:
                await redis_store.delete(tool_ctx_key)
            else:
                await redis_store.set(tool_ctx_key, {**prev, "idle_turns": idle}, expire=ttl)

    # ── 토큰 기반 자동 압축 트리거 ──────────────────────────────────
    # 비동기 fire-and-forget. 사용자 응답 레이턴시에 영향 없음.
    # 락으로 동일 키 중복 트리거 차단되므로 매 턴 호출해도 안전.
    asyncio.create_task(_compact_history_guarded(key, ttl))


# 답변이 이 문구로 끝나면 가드가 환각 액션을 교정한 턴이다 — 사용자가 한 일이 아니라 모델이 못 한 일.
_CAPTURE_SKIP_MARKERS: tuple[str, ...] = ("아직 실제로 처리하지는 않았어요",)


# 가드 교정 문구를 **이력에서만** 걷어내는 정규식. 사용자가 본 답변·DB 저장본은 건드리지
# 않는다 — 화면과 히스토리가 어긋나면 그것대로 사고다(2026-08-10 실측).
#
# 왜 필요한가: 이 문장은 `answer_guard` 가 코드로 써 넣은 **교정 산출물**이지 모델의 말이
# 아니다. 그런데 그대로 이력에 쌓이면 다음 턴의 프롬프트에 예시로 들어가고, 모델이 그걸
# 따라 쓴다. 실측(2026-09-11 개발기): 가드 오발동을 고쳐 배포한 **뒤에도** 같은 문장이
# 계속 나왔다 — `[answer_guard]` 경고 로그가 없는데도 나왔으니 가드가 아니라 모델이
# 앞선 턴을 모방한 것이다(같은 스레드 00:40·00:45 → 01:11·01:12).
_GUARD_ARTIFACT_RE = re.compile(
    r"아직 실제로 처리하지는 않았어요\.?\s*(?:원하시면 다시 요청해 주세요\.?)?\s*"
)

# 같은 이유로 걷어내는 두 번째 산출물: HITL 초안 안내 문구.
# ⚠️ 원문은 `app/agents/graphs/supervisor._DRAFT_NOTE` 다 — 한쪽을 고치면 여기도 고칠 것
#    (import 하면 memory ↔ graphs 순환이 생겨 문자열을 복제한다).
#
# 이 문장도 **코드가 붙이는 말**이다(초안 큐가 비어 있지 않을 때만). 그런데 이력에 쌓이면
# 다음 턴의 모델이 따라 쓴다 — 실측(2026-09-15): 전송 도구를 **한 번도 부르지 않은** 턴에서
# "발송/등록 초안을 준비했어요" 라고 답했다. 카드도 초안도 없는데 사용자에게는 승인을
# 기다리라고 말한 셈이라, 하지 않은 일을 했다고 보고하는 가장 나쁜 형태의 오답이다.
_DRAFT_NOTE_RE = re.compile(
    r"발송/등록 초안을 준비했어요\s*(?:—|-|–)?\s*(?:아래 카드에서 승인해 주세요\.?)?\s*"
)

_ARTIFACT_PHRASES: tuple = (
    ("아직 실제로 처리하지는 않았어요", _GUARD_ARTIFACT_RE),
    ("발송/등록 초안을 준비했어요", _DRAFT_NOTE_RE),
)


def _strip_guard_artifact(msg: Message) -> Message:
    """이력으로 되돌려 주는 assistant 메시지에서 **코드가 써 넣은 문구**를 제거한다.

    가드 교정 문구와 HITL 초안 안내가 대상이다. 둘 다 모델의 말이 아니라 코드 산출물이라,
    이력에 남으면 다음 턴이 그걸 예시로 삼아 따라 쓴다.

    문구만 지우고 나머지(다운로드 링크 등)는 보존한다. 지운 결과가 비면 원문을 그대로
    둔다 — 빈 assistant 턴은 대화 구조를 깨뜨린다.
    """
    if msg.role != "assistant" or not msg.content:
        return msg
    cleaned = msg.content
    for marker, pattern in _ARTIFACT_PHRASES:
        if marker in cleaned:
            cleaned = pattern.sub("", cleaned)
    cleaned = cleaned.strip()
    if cleaned == msg.content or not cleaned:
        return msg
    return msg.model_copy(update={"content": cleaned})


_TOOL_ERROR_PREFIX = "[ERROR]"          # app/agents/react/executor.serialize_result 의 실패 직렬화 형식
_DISCOVERY_TOOL_PREFIX = "find_"        # 도메인 도구 탐색은 실행이 아니라 준비 단계 — 성공/실패 판정에서 뺀다


_PLACE_NAME_MAX = 120


def turn_place(user_info: Optional[dict]) -> Optional[dict]:
    """이 턴이 **어디서** 일어났는지 — 앱 전용 창 이름 또는 메신저 ``origin`` 을 저장용 모양으로 접는다.

    브라우저 패널은 둘 다 없으므로 None 이 되고, 그 턴의 metadata 는 이 기능이 없을 때와
    바이트까지 같다. 즉 이 기록은 **분리된 자리**(앱 전용 창·메신저 채널) 전용이다.

    왜 필요한가: 채널·스레드는 지금까지 시스템 프롬프트의 ``[현재 위치]`` 블록에만 있었고
    어디에도 저장되지 않았다. 그래서 그 턴에서 뽑힌 기억은 자기가 어느 채널·어느 스레드에서
    비롯됐는지 알 수 없었다(2026-09-18 진단: 메신저 턴 metadata 키는 회사·단계수·도구요약뿐).
    장소를 아는 회상은 이 기록 위에서만 만들 수 있다.

    ``conversation_id`` 로 대신할 수 없다 — 스레드↔대화 매핑이 7일 TTL 이라 같은 스레드도
    그 뒤에는 다른 대화가 되고, 대화 id 자체는 어느 채널인지 말해 주지 않는다.

    Args:
        user_info: 요청 스코프 사용자 정보. 앱 전용 창은 ``memory_surface``, 메신저 경로는
            ``origin`` 을 싣는다.

    Returns:
        ``{"provider", "channel_id", "is_dm"[, "channel_name"][, "thread_root"]}`` 또는 None.
    """
    # 순환 import 회피 (memory.conversation ← memory_v2.capture ← …)
    from agent_arch.services.memory_v2 import place as place_mod

    info = user_info or {}
    # 앱 전용 창(전용 앱 채팅·음성)은 채널이 아니라 창 이름이 자리다 — 판정은 요청 경로가
    # 소유하고(source_routing.resolve_memory_surface → build_user_info) 여기선 접기만 한다.
    if surface := place_mod.surface_place(info.get("memory_surface")):
        return surface
    origin = info.get("origin")
    if not isinstance(origin, dict):
        return None
    channel_id = str(origin.get("channel_id") or "").strip()
    if not channel_id:
        return None
    place: dict = {"provider": "mattermost", "channel_id": channel_id,
                   "is_dm": bool(origin.get("is_direct"))}
    name = str(origin.get("channel_name") or "").strip()[:_PLACE_NAME_MAX]
    if name:
        place["channel_name"] = name
    root = str(origin.get("root_id") or "").strip()
    if root:
        place["thread_root"] = root
    return place


def tool_outcome(tool_results: Optional[list]) -> dict:
    """이 턴 도구 결과 요약 — 실행 도구(탐색 도구 제외)를 몇 번 불렀고 몇 번 성공·실패했나.

    ``{"calls": 실행+탐색 호출 수, "actions": 실행 도구 결과 수, "ok": 성공, "errors": 실패}``. 탐색(find_*)은
    준비 단계라 성공으로 치지 않는다 — 탐색만 하고 끝난 턴은 아무것도 조회하지 못한 턴이다. 예약 실행기가
    "실행은 됐지만 작업은 못 했다"를 가르는 근거로 쓴다(답변 문구가 아니라 도구 결과로 판정).
    """
    rows = [r for r in (tool_results or []) if isinstance(r, dict)]
    actions = [r for r in rows if not str(r.get("name") or "").startswith(_DISCOVERY_TOOL_PREFIX)]
    errors = sum(1 for r in actions if str(r.get("content") or "").lstrip().startswith(_TOOL_ERROR_PREFIX))
    return {"calls": len(rows), "actions": len(actions), "ok": len(actions) - errors, "errors": errors}


def memory_capture_skip_reason(
    status_code: Optional[int],
    assistant_output: Optional[str],
    tool_results: Optional[list] = None,
) -> Optional[str]:
    """이 턴을 장기 메모리 추출에서 빼야 하는 이유(없으면 None).

    - status_code >= 400: 업스트림 실패, 사용자 중단(499). 실패 경위는 기억이 아니다.
    - 빈 답변: 근거로 삼을 응답이 없다.
    - 가드 교정 문구: 모델이 하지 않은 일을 했다고 말해 코드가 바꾼 턴.
    - 도구 전부 실패: 실행 도구(탐색 도구 제외) 결과가 모두 ``[ERROR]`` 면 그 턴은 아무 일도 못 한 턴이다
      (실측: since 인자 오류로 실패한 메일 조회 턴들이 HTTP 200 으로 저장돼 status 만으로는 안 걸렸다).
    """
    if status_code is not None and int(status_code) >= 400:
        return f"status_{int(status_code)}"
    text = (assistant_output or "").strip()
    if not text:
        return "empty_answer"
    if any(m in text for m in _CAPTURE_SKIP_MARKERS):
        return "guard_fallback"
    action_results = [
        r for r in (tool_results or [])
        if isinstance(r, dict) and not str(r.get("name") or "").startswith(_DISCOVERY_TOOL_PREFIX)
    ]
    if action_results and all(str(r.get("content") or "").lstrip().startswith(_TOOL_ERROR_PREFIX) for r in action_results):
        return "tool_error"
    return None


async def save_logged_in_turn(
    conversation_id: Optional[str],
    user_sub: str,
    user_input: str,
    assistant_output: str,
    tool_calls: Optional[List[Any]] = None,
    tool_results: Optional[List[dict]] = None,
    response_time: Optional[float] = None,
    status_code: int = 200,
    llm_invoke_count: Optional[int] = None,
    save_to_redis: bool = True,
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    user_email: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    langsmith_run_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    """로그인 사용자 대화 한 턴을 PostgreSQL과 Redis에 저장한다.

    PG 먼저 → conversation_id 확정(lazy-create 내부 동작) → Redis 적재 순서.
    PG 실패 시 conversation_id를 알 수 없으므로 Redis 적재는 생략한다.
    Redis 실패는 로그만 — PG 저장에 영향 없음(다음 turn에서 RPUSH로 누적되니 손실 없음).

    Args:
        conversation_id: 대상 대화방 UUID. None이면 save_turn 내부에서 lazy-create.
        user_sub: 소유자 sub. lazy-create 시 소유자로 들어가고 preferences 조회에도 사용.
        user_input: 사용자 입력 텍스트.
        assistant_output: 어시스턴트 응답 텍스트.
        tool_calls: ReAct 루프에서 실행된 도구 호출 목록 (선택).
        tool_results: 각 도구 호출의 실행 결과 (선택). Redis 의 tool_ctx 키에 저장되어
            다음 턴 load_history 에서 직전 1턴 컨텍스트로 주입된다.
            형식: [{"tool_call_id": str, "name": str, "content": str}]. PG 저장에는
            관여하지 않으며 Redis 전용.
        response_time: 최종 답변까지 걸린 시간 (초, 선택).
        status_code: 응답 상태 코드 (기본 200).
        llm_invoke_count: 총 LLM 호출 횟수 (선택).
        save_to_redis: Redis 적재 여부. stateless 프로바이더는 False.
        llm_provider: LLM 프로바이더 식별자 (선택).
        model_name: 실제 사용된 모델명 (선택).
        user_email: Keycloak 이메일 주소 (선택).
        prompt_tokens: 입력 토큰 수 합계 (선택).
        completion_tokens: 출력 토큰 수 합계 (선택).
        total_tokens: 전체 토큰 수 합계 (선택).
        langsmith_run_id: LangSmith @traceable run ID (선택). rating 연결에 사용.
        finish_reason: 마지막 LLM 응답의 finish_reason (선택). PG 저장 전용(Redis 미관여).
        metadata: 턴 메타데이터 dict (선택). PG의 metadata(JSONB) 컬럼에 저장(Redis 미관여).
            그래프 스텝 계측(react_steps) 등 admin 전용 신호.

    Returns:
        (message_id, conversation_id) 튜플. PG 실패 시 message_id는 None.
        conversation_id는 lazy-create 후 확정된 값 (실패하면 None).

    Note (POS 턴 정화 — 임시 가드, 전용 API 전환 시 제거):
        POS 대시보드 태그 프롬프트 턴은 저장 전에 정화된다 — query 는 [사용자 질문]만,
        answer 는 기계 태그 제거본, 원문은 ``metadata.pos_raw``. 응답이 태그뿐인
        플래닝 턴은 ``metadata.pos_internal=true`` 로 표시되어 이력 조회에서 숨겨지고
        Redis 재주입·메모리 추출도 스킵된다(POS 는 매 턴 컨텍스트를 자체 동봉하므로
        손실 없음). 플래닝 턴의 message_id 도 스트림 done 이벤트로는 반환되므로
        rating 이 달릴 수는 있으나, rated_count 집계에서 제외되어 무해하다.
    """
    message_id: Optional[str] = None
    conv_id: Optional[str] = conversation_id
    # file-service 로 영속화된 이번 턴 첨부 메타 — 요청 스코프 ContextVar 에서 읽는다
    # (orchestrator._preprocess_request 가 set). 저장 호출 지점이 10곳이라 파라미터
    # 대신 이미지 컨텍스트와 같은 패턴으로 한 지점에서 흘려보낸다.
    attachments = get_attachment_context() or None
    # 요청 시점의 선택 회사를 metadata 에 남긴다 — agent.messages 에 company 컬럼이 없어
    # 지금까지 회사는 LangSmith run metadata 에만 있었고, 트레이싱이 꺼진 구간의 대화는
    # 회사를 알 수 없었다(관측 '회사 리뷰' 회사 필터에서 통째로 빠짐, 2026-07-30). 저장 지점이
    # 여러 곳이라 파라미터가 아니라 첨부 컨텍스트와 같은 요청 스코프 ContextVar 에서 읽는다.
    company = (get_user_context() or {}).get("company")
    if company:
        metadata = {**(metadata or {}), "company": company}
    # 이 턴의 장소(채널·스레드·DM 여부). 회사와 같은 이유로 파라미터가 아니라 요청 스코프
    # ContextVar 에서 읽는다. 메신저 경로만 값이 생기고, 패널 턴에는 키가 붙지 않는다.
    place = turn_place(get_user_context())
    if place:
        metadata = {**(metadata or {}), "place": place}
    # 사용량 관측의 서비스 축. 회사와 같은 요청 스코프 ContextVar 에서 읽지만, metadata 가
    # 아니라 실컬럼에 넣는다 — 일별 사용량 집계의 GROUP BY 축이라 JSONB 추출로는 인덱스를
    # 못 탄다(docker/postgres/migration_v34.sql 참조).
    service = (get_user_context() or {}).get("service")
    # 답변이 만들어진 조건(웹 검색 사용 여부). 서비스·회사·첨부와 같은 요청 스코프에서
    # 읽는다 — 저장 지점이 10곳이라 파라미터로 뚫으면 한 경로씩 조용히 빠진다.
    # **미기록(None)과 '안 썼다'(False)가 다른 사실**이다. 패널 토글을 안 거친 경로에서 False 로
    # 접으면 그 구분을 잃으므로 None 을 그대로 남긴다.
    web_search = web_search_choice()
    # POS 대시보드 태그 프롬프트 턴 정화 — 여기(단일 깔때기)서 치환해야 pg_save_turn 의
    # lazy-create 제목 파생·Redis 재주입·메모리 추출이 전부 정화본을 받는다.
    # (저장 호출 지점 10곳 무수정 — attachments/company 와 같은 이유.)
    pos = sanitize_pos_turn(user_input, assistant_output)
    if pos is not None:
        metadata = {
            **(metadata or {}),
            "pos_raw": {
                "input": user_input[:_POS_RAW_MAX],
                "answer": assistant_output[:_POS_RAW_MAX],
            },
        }
        if pos.internal:
            metadata["pos_internal"] = True  # JSONB true → ->>'pos_internal' = 'true'
        user_input, assistant_output = pos.query, pos.answer
    pos_internal = pos.internal if pos is not None else False
    # 메모리 캡처 게이트 — 실패(4xx/5xx)·사용자 중단(499)·빈 답변·가드 교정 문구로 끝난 턴은 기억의
    # 근거가 아니다(실측: since 인자 오류로 실패한 메일 턴들이 그대로 추출됐다). 저장은 하되
    # metadata.capture_skip 을 남겨 v1·v2 추출 모두에서 제외한다.
    capture_skip = memory_capture_skip_reason(status_code, assistant_output, tool_results)
    # 예약 실행(무인) 턴은 그 사람이 한 말이 아니다 — 매일 같은 요청이 "사용자는 매일 환율을 물었다" 같은
    # 기억으로 쌓이지 않게 추출에서 뺀다. 어느 예약·회차인지는 남긴다(대화 화면·리뷰가 구분한다).
    scheduled = get_unattended_run()
    if scheduled:
        metadata = {**(metadata or {}), "schedule_run": {k: scheduled.get(k) for k in ("schedule_id", "run_id")}}
        capture_skip = capture_skip or "scheduled_run"
        # 질문 자리에는 내부 실행 지시문이 아니라 사람이 읽는 한 줄을 남긴다(리뷰·감사 화면에 그대로 보인다).
        if scheduled.get("display_input"):
            user_input = str(scheduled["display_input"])
    # 승인 카드만 띄운 턴은 아직 아무 일도 일어나지 않았다 — 도구 인자(예약 일정·메일 수신자)가 "사용자는 매일
    # 09:40 환율 알림을 예약했다" 같은 사실로 추출돼, 승인하지 않은 예약을 다음 대화에서 있다고 답했다
    # (2026-10-16 운영). 승인되면 agent._persist_action_outcome 이 "(승인 실행됨)" 턴을 따로 저장해 그 턴이 근거가 된다.
    if finish_reason == "action_pending":
        capture_skip = capture_skip or "pending_approval"
    if capture_skip:
        metadata = {**(metadata or {}), "capture_skip": capture_skip}
    # 메모리 v2 L0 다이제스트 — 도구 **결과**의 id·이름 필드만 추려 metadata.tool_digest 에 남긴다.
    # tool_calls 컬럼은 호출 인자뿐이고 결과는 Redis tool_ctx 에 1턴만 살아서, 배치 추출기가
    # 별칭의 canonical_id 를 얻을 길이 없었다. 결정론적(LLM 없음)·2KB 이내.
    if tool_results and not pos_internal and not capture_skip:
        try:
            from agent_arch.services.memory_v2 import build_tool_digest, enabled as _memory_v2_enabled
            if _memory_v2_enabled():
                # 남의 대화 본문은 **메신저 턴에만** 남긴다. 패널에서 오던 기억은 잘 돌고 있었으므로
                # 그 경로의 도구 요약·추출 입력·프롬프트를 이 기능 이전과 글자까지 같게 둔다.
                messenger = isinstance((get_user_context() or {}).get("origin"), dict)
                digest = build_tool_digest(tool_calls, tool_results, conversation_excerpt=messenger)
                if digest:
                    metadata = {**(metadata or {}), "tool_digest": digest}
        except Exception:
            logger.debug("메모리 v2 tool_digest 생성 실패(무시)", exc_info=True)
    try:
        message_id, conv_id = await pg_save_turn(
            conversation_id=conversation_id,
            user_input=user_input,
            assistant_output=assistant_output,
            user_id=user_sub,
            user_email=user_email,
            tool_calls=tool_calls,
            response_time=response_time,
            status_code=status_code,
            llm_invoke_count=llm_invoke_count,
            llm_provider=llm_provider,
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            langsmith_run_id=langsmith_run_id,
            finish_reason=finish_reason,
            metadata=metadata,
            attachments=attachments,
            service=service,
            web_search=web_search,
        )
    except Exception:
        logger.error(
            "PostgreSQL 대화 저장 실패 (user_sub=%s, conversation_id=%s)",
            user_sub, conversation_id, exc_info=True,
        )

    # 플래닝 턴(pos_internal)은 Redis 재주입 제외 — 이력 조회와 같은 기준으로 숨긴다.
    if save_to_redis and conv_id and not pos_internal:
        try:
            await append_turn(
                f"conversation:{conv_id}",
                user_input,
                assistant_output,
                user_sub=user_sub,
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
        except Exception:
            logger.error(
                "Redis 히스토리 적재 실패 (conv_id=%s)", conv_id, exc_info=True,
            )

    # 보류해 둔 이번 턴 라우트를 확정된 대화 id 로 저장한다(begin_turn_route 참조).
    if conv_id and not pos_internal:
        await _flush_turn_route(conv_id)

    # 장기 사용자 메모리 트리거 (fire-and-forget, ENABLE_USER_MEMORY 게이트).
    # PG 저장이 성공(message_id 확정)했을 때만 — provenance 링크가 유효해야 하므로.
    # 순환 import 회피 위해 함수 내부 import (LLMFactory/prompts 와 동일 패턴).
    #
    # 턴 저장 시점에는 체크포인트 이벤트만 남기고 추출은 배치로 미룬다(memory_v2).
    # capture_skip(실패·중단·가드 교정 턴)은 건너뛴다.
    if message_id and conv_id and not pos_internal and not capture_skip:
        try:
            from agent_arch.services.memory_v2 import on_turn_saved as _memory_v2_capture
            _memory_v2_capture(user_sub, message_id, user_input)
        except Exception:
            logger.error("메모리 v2 캡처 트리거 실패 (conv_id=%s)", conv_id, exc_info=True)

    return message_id, conv_id


async def clear_user_conversations(user_sub: str) -> int:
    """본인 모든 대화방의 Redis 히스토리 키를 일괄 삭제한다.

    /compact 슬래시 커맨드와 DELETE /me/preferences/history 엔드포인트가 사용한다.
    DB에서 사용자의 모든 conversation_id(archived 포함)를 조회한 뒤 각각의 Redis
    키를 delete한다. PG의 대화방·메시지 행은 읽기만 하고 변경하지 않는다.
    archived까지 포함하는 이유는 사용자가 "내 컨텍스트 즉시 비우기"
    를 명시한 흐름이라 archived 방의 상태도 같이 초기화되는 게 자연스럽기 때문.

    Args:
        user_sub: 본인 sub.

    Returns:
        PG에서 조회된 본인 대화방 수 (= 삭제 시도 대상 수). 대부분의 키는 TTL로
        이미 만료된 상태일 수 있어 "실제 삭제된 Redis 키 수"가 아니다.
        사용자 노출용 수치로 쓰지 말 것.
    """
    from agent_arch.db.repository import list_conversations

    convs = await list_conversations(
        user_sub=user_sub, limit=1000, include_archived=True,
    )
    for c in convs:
        await redis_store.delete(f"conversation:{c['id']}")
        # 직전 1턴 도구 컨텍스트 키도 함께 삭제 (잔존 시 stale 주입 위험).
        await redis_store.delete(f"conversation:{c['id']}:tool_ctx")
        # 압축 통지 / 락 키도 정리 (사용자가 명시적으로 컨텍스트를 비우는 흐름이라
        # 압축 상태도 함께 초기화되는 게 자연스럽다).
        await redis_store.delete(f"conversation:{c['id']}:compact_announce")
        await redis_store.delete(f"conversation:{c['id']}:compact_lock")
        # 맥락 비움 워터마크 — 이게 없으면 다음 턴에 이력 폴백이 PG 에서 전부 되살려
        # "이제 이전 대화 내용을 기억하지 않습니다" 라는 약속을 깬다(utils/commands 의 /compact).
        # 불리언이 아니라 시각이라야 비운 뒤 이어간 대화는 정상적으로 되살아난다.
        await redis_store.set(
            _history_cleared_key(f"conversation:{c['id']}"),
            _utc_now_naive().isoformat(), expire=_HISTORY_CLEARED_TTL_SECONDS,
        )
    return len(convs)


# ─────────────────────────────────────────────────────────────────────────────
# 토큰 기반 자동 압축
# ─────────────────────────────────────────────────────────────────────────────
#
# 흐름:
#   append_turn → asyncio.create_task(_compact_history_guarded(key, ttl))
#                   → Redis SET NX 락 시도
#                   → _compact_history: 토큰 합 계산 → LLM 요약 → LIST 재구성
#                   → `{key}:compact_announce` 키 set
#
#   announce 키는 외부 노출되지 않고 관측성 용도로만 보존된다
#   (COMPACT_ANNOUNCE_TTL 만료 시 자동 정리).


async def consume_compact_announce(key: str) -> Optional[dict]:
    """압축 통지 키를 1회 소비한다.

    announce 키는 외부 클라이언트에 노출되지 않고 관측성 목적으로만 보존된다
    (응답 본문에 통지 JSON 이 인라인되는 UX 사고 방지). 단위 테스트가
    함수 행동을 계속 보장한다.

    `{key}:compact_announce` 에 저장된 통지 페이로드를 GETDEL 로 atomic
    하게 가져온다. Redis 6.2+ 에선 완전 atomic, 그 미만은 GET+DEL 폴백.

    Args:
        key: history Redis 키 (예: "conversation:UUID" 또는 "guest:UUID").

    Returns:
        통지가 있으면 {"ts": int, "n_summarized": int}, 없으면 None.
    """
    return await redis_store.getdel(f"{key}:compact_announce")


# tiktoken encoder 는 모듈 전역 캐시. 첫 호출 시 lazy load.
_tiktoken_encoder = None
_tiktoken_unavailable = False


def _get_tiktoken_encoder():
    """tiktoken cl100k_base encoder 를 lazy load + 캐시.

    실패 시 _tiktoken_unavailable 플래그를 세팅하여 이후 호출은 휴리스틱으로 폴백.
    이 함수는 직접 호출되기보다 _estimate_tokens 내부에서 사용된다.

    Returns:
        tiktoken Encoding 객체. import/로드 실패 시 None.
    """
    global _tiktoken_encoder, _tiktoken_unavailable
    if _tiktoken_unavailable:
        return None
    if _tiktoken_encoder is not None:
        return _tiktoken_encoder
    try:
        import tiktoken
        _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
        return _tiktoken_encoder
    except Exception as e:
        logger.warning("tiktoken 로드 실패 → 휴리스틱 폴백 사용: %s", e)
        _tiktoken_unavailable = True
        return None


def _estimate_tokens(messages: List[Message]) -> int:
    """메시지 리스트의 토큰 합계를 추정한다.

    1순위: tiktoken cl100k_base 인코더.
    2순위(폴백): `int(len(text) * 0.6)` — 한국어 혼합 텍스트 rough heuristic.

    각 메시지의 content / name / tool_calls(JSON dump) 를 합산한다.

    Args:
        messages: 추정 대상 메시지 리스트.

    Returns:
        추정 토큰 수.
    """
    encoder = _get_tiktoken_encoder()
    total = 0
    for m in messages:
        parts: List[str] = []
        if m.content:
            parts.append(m.content)
        if m.name:
            parts.append(m.name)
        if m.tool_calls:
            try:
                parts.append(json.dumps(m.tool_calls, ensure_ascii=False))
            except Exception:
                pass
        text = "\n".join(parts)
        if not text:
            continue
        if encoder is not None:
            try:
                total += len(encoder.encode(text))
                continue
            except Exception:
                pass
        # 폴백: 휴리스틱
        total += int(len(text) * _TIKTOKEN_HEURISTIC_RATIO)
    return total


async def _compact_history_guarded(key: str, ttl: int) -> None:
    """압축 락을 획득한 뒤 _compact_history 를 실행한다.

    동일 키에 대한 중복 트리거를 차단한다. SET NX 로 락을 잡고, try/finally 로
    반드시 해제. 락 획득 실패 시 즉시 return (다른 작업 진행 중).

    Args:
        key: history Redis 키 (예: "conversation:UUID" 또는 "guest:UUID").
        ttl: history LIST 의 TTL — 압축 후 LIST 재생성 시에도 동일 적용.
    """
    lock_key = f"{key}:compact_lock"
    acquired = await redis_store.set_nx(lock_key, "1", expire=COMPACT_LOCK_TTL_SECONDS)
    if not acquired:
        return
    try:
        await _compact_history(key, ttl)
    except Exception:
        logger.exception("자동 압축 실패: key=%s", key)
    finally:
        await redis_store.delete(lock_key)


async def _compact_history(key: str, ttl: int) -> None:
    """오래된 메시지를 LLM 으로 요약하여 LIST 를 재구성한다.

    동작:
        1. LIST 스냅샷 + 토큰 추정. trigger 미만이면 즉시 return (idempotent).
        2. 메시지 수가 keep_recent + min_old 미만이면 스킵.
        3. 가장 오래된 메시지가 기존 summary(name==MARKER) 면 previous_summary 추출.
        4. LLM 요약(memory/compact_summary.summarize: 섹션 고정·질문 원문은 코드가 붙임·코드 점검).
           호출 실패나 점검 2회 실패면 LIST 무변경.
        5. 요약한 앞쪽 구간만 원자적으로 [summary] 로 바꾼다(redis_store.replace_prefix).
           압축하는 사이 붙은 턴은 뒤쪽이라 그대로 남는다. 앞쪽이 그사이 트림으로 바뀌었으면
           아무것도 바꾸지 않고 다음 턴에 다시 시도한다.
        7. `{key}:compact_announce` 키에 통지 페이로드 set.

    Args:
        key: history Redis 키.
        ttl: history LIST 의 TTL.
    """
    raw_strs = await redis_store.lrange_raw(key)
    if not raw_strs:
        return
    messages = [Message(**json.loads(r)) for r in raw_strs]

    if _estimate_tokens(messages) < COMPACT_TRIGGER_TOKENS:
        return  # idempotent — 이미 작아진 경우

    if len(messages) < COMPACT_KEEP_RECENT_MESSAGES + COMPACT_MIN_OLD_MESSAGES:
        return  # 압축 대상이 충분치 않음

    old_messages = messages[:-COMPACT_KEEP_RECENT_MESSAGES]
    keep_messages = messages[-COMPACT_KEEP_RECENT_MESSAGES:]
    # 교체할 앞쪽 구간(이전 요약 포함)의 원문 — 원자 교체 때 그대로 대조한다.
    prefix_raw = raw_strs[: len(raw_strs) - COMPACT_KEEP_RECENT_MESSAGES]

    # 기존 summary 흡수
    previous_summary: Optional[str] = None
    if old_messages and old_messages[0].name == COMPACT_SUMMARY_MARKER:
        previous_summary = old_messages[0].content
        old_messages = old_messages[1:]
        if not old_messages:
            # 기존 summary 만 있고 추가 압축 대상 없음 → 스킵
            return

    # LLM 호출 — 순환 import 회피 위해 함수 내부 import.
    from agent_arch.llm.factory import LLMFactory
    from agent_arch.memory.compact_summary import summarize
    try:
        llm = LLMFactory.create_with_fallback()
        summary_text = await summarize(llm, old_messages, previous_summary)
    except Exception:
        logger.exception("압축 LLM 호출 실패: key=%s", key)
        return

    if not summary_text:
        # 빈 응답이거나 점검(섹션·고유값·축소율)을 두 번 통과하지 못했다 — 원문 유지.
        logger.warning("압축 요약이 점검을 통과하지 못함 — 이번 압축 취소: key=%s", key)
        return

    new_summary = Message(
        role="system",
        name=COMPACT_SUMMARY_MARKER,
        content=summary_text,
    )

    # 요약한 앞쪽만 원자적으로 바꾼다. 예전에는 길이로 "그사이 붙은 턴"을 찾고 delete → rpush 했는데,
    # append_turn 이 rpush 뒤 ltrim 하면 길이가 그대로라 새 턴을 못 보고 덮어써 잃었다.
    swapped = await redis_store.replace_prefix(
        key, prefix_raw, [new_summary.model_dump(exclude_none=True)], ttl,
    )
    if not swapped:
        logger.info("압축 중 이력 앞쪽이 바뀜(트림 등) — 이번 압축 취소, 다음 턴에 재시도: key=%s", key)
        return

    # 통지 키 set — 관측성 목적으로 보존 (외부 노출되지 않으며 COMPACT_ANNOUNCE_TTL 만료 시 정리).
    await redis_store.set(
        f"{key}:compact_announce",
        {
            "ts": int(time.time()),
            "n_summarized": len(old_messages),
        },
        expire=COMPACT_ANNOUNCE_TTL,
    )
    logger.info(
        "자동 압축 완료: key=%s n_summarized=%d kept=%d summary_chars=%d",
        key, len(old_messages), len(keep_messages), len(summary_text),
    )
