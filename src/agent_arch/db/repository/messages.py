"""로그인 사용자 메시지 CRUD.

agent.messages 테이블을 다룬다. 메시지는 (query, answer) 단일 튜플로 저장한다.
메시지는 conversation_id 단위로 묶이며, agent.sessions는 더 이상 사용하지 않는다.
"""
import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Optional

import asyncpg

from agent_arch.db.connection import db_pool
from agent_arch.db.repository.conversations import (
    create_conversation,
    derive_conversation_title,
    touch_conversation,
)

logger = logging.getLogger(__name__)


async def save_turn(
    conversation_id: Optional[str],
    user_input: str,
    assistant_output: str,
    user_id: Optional[str],
    user_email: Optional[str] = None,
    tool_calls: Optional[list[Any]] = None,
    response_time: Optional[float] = None,
    status_code: int = 200,
    llm_invoke_count: Optional[int] = None,
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    prompt_tokens: Optional[int] = None,
    completion_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    langsmith_run_id: Optional[str] = None,
    finish_reason: Optional[str] = None,
    metadata: Optional[dict] = None,
    attachments: Optional[list] = None,
    web_search: Optional[bool] = None,
    service: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """로그인 사용자 대화 한 턴을 저장한다.

    conversation_id가 None이면 lazy-create — 진입부에서 새 대화방을 만든 후
    그 id로 INSERT한다. 저장 직후 touch_conversation으로 last_message_at을 갱신.

    user_sub / user_email 컬럼에 호출자 정보를 함께 적재한다. user_email은 "당시 스냅샷"
    의미로 저장되어 사용자가 이메일을 바꿔도 과거 행은 그대로 유지된다 (audit 정합성).
    현재 이메일이 필요하면 agent.users(user_sub) JOIN으로 조회.

    Args:
        conversation_id: 대상 대화방 UUID. None이면 lazy-create.
        user_input: 사용자 입력 텍스트 (query).
        assistant_output: 어시스턴트 응답 텍스트 (answer).
        user_id: Keycloak user sub. lazy-create 시 소유자로 사용되며 user_sub 컬럼에도 저장.
        user_email: Keycloak 이메일 주소. user_email 컬럼에 스냅샷으로 저장 (선택).
        tool_calls: ReAct 루프에서 실행된 도구 호출 목록 (선택).
        response_time: 최종 답변까지 걸린 시간 (초, 선택).
        status_code: 응답 상태 코드 (기본값: 200).
        llm_invoke_count: 총 LLM 호출 횟수 (선택).
        llm_provider: LLM 프로바이더 식별자 (선택).
        model_name: 실제 사용된 모델명 (선택).
        prompt_tokens: 입력 토큰 수 합계 (선택).
        completion_tokens: 출력 토큰 수 합계 (선택).
        total_tokens: 전체 토큰 수 합계 (선택).
        langsmith_run_id: LangSmith @traceable run ID (선택). rating 연결에 사용.
        finish_reason: 마지막 LLM 응답의 finish_reason (선택). 운영/학습 substrate 분석용.
        metadata: 턴 메타데이터 dict (선택). agent.messages.metadata(JSONB)에 저장. 그래프
            스텝 계측(react_steps: node_trace/timings/iterations/discovered 등) 등 admin 전용 신호.
        service: 이 턴이 발생한 서비스 면 (선택). browser_panel / messenger / app / guest. 관측 대시보드의 서비스별 사용량 축이며, None 이면 '미분류'로
            남는다 — 판정하지 못한 면에 기본값을 지어내면 남의 사용량이 브라우저 패널
            청구에 조용히 섞인다.
        web_search: 이 턴에 웹 검색이 실제로 열렸는가 (선택). None 은 기록 없음이며 False 와
            다르다 — False 로 접으면 이 축이 생기기 전 적재분 전체가 '웹을 쓰지 않은 답변'이
            된다(migration_v39 주석 참조).
        attachments: 이 턴의 첨부파일 메타 목록 (선택).
            ``[{"file_id", "filename", "content_type"}]`` — file-service 로 영속화된
            파일의 참조만 저장한다(presigned URL 은 만료되므로 절대 저장 금지).
            히스토리 조회 시 file_id 로 다운로드 URL 을 재발급해 이미지를 복원한다.

    Returns:
        (message_id, conversation_id) 튜플. DB 비활성화 시 (None, conversation_id 인자 그대로).
        호출자가 conversation_id를 받아 응답에 동봉할 수 있도록 함께 반환한다.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None, conversation_id

    # lazy-create: conversation_id가 없으면 새 방을 만든다. 첫 질의(user_input)에서 제목을
    # 파생해 목록에 의미 있는 제목이 뜨게 한다. 소유자(user_sub)가 없으면 비정상 경로.
    if conversation_id is None:
        if not user_id:
            logger.error("save_turn: conversation_id와 user_id가 모두 비어 lazy-create 불가")
            return None, None
        conversation_id = await create_conversation(
            user_sub=user_id, title=derive_conversation_title(user_input)
        )
        if conversation_id is None:
            logger.error("save_turn: lazy-create 실패 — DB 풀 None 또는 INSERT 실패")
            return None, None

    tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    attachments_json = json.dumps(attachments, ensure_ascii=False) if attachments else None
    answer_length = len(assistant_output) if assistant_output else 0

    row = await pool.fetchrow(
        """
        INSERT INTO agent.messages
            (conversation_id, user_sub, user_email, query, answer, tool_calls,
             response_time, status_code, answer_length, llm_invoke_count,
             llm_provider, model_name,
             prompt_tokens, completion_tokens, total_tokens,
             langsmith_run_id, finish_reason, metadata, attachments, service,
             web_search)
        VALUES ($1::uuid, $2, $3, $4, $5, $6::jsonb, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18::jsonb, $19::jsonb, $20, $21)
        RETURNING id
        """,
        conversation_id,
        user_id,
        user_email,
        user_input,
        assistant_output,
        tool_calls_json,
        response_time,
        status_code,
        answer_length,
        llm_invoke_count,
        llm_provider,
        model_name,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        langsmith_run_id,
        finish_reason,
        metadata_json,
        attachments_json,
        service,
        web_search,
    )
    message_id = str(row["id"]) if row else None

    # 대화방 last_message_at 갱신 (목록 정렬 캐시)
    await touch_conversation(conversation_id)

    return message_id, conversation_id


async def update_message_rating(message_id: str, rating: str) -> bool:
    """로그인 사용자 메시지의 rating을 텍스트로 저장한다.

    Args:
        message_id: 대상 메시지 UUID 문자열.
        rating: 사용자 평가 텍스트 (자유 형식).

    Returns:
        업데이트 성공 여부. 해당 message_id가 없으면 False.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return False

    result = await pool.execute(
        "UPDATE agent.messages SET rating=$1 WHERE id=$2::uuid",
        rating,
        message_id,
    )
    return result.endswith("1")


async def set_message_action_view(message_id: str, action_view: dict) -> bool:
    """메시지에 표시용 액션 뷰(HITL 승인 카드 등)를 저장한다.

    스트림에서 draft 카드(actionKind=draft + form)를 방출한 턴 직후 호출해, 과거 대화
    조회 시 카드 테이블을 읽기전용으로 복원할 수 있게 한다. 표시 데이터만 담는다.

    Args:
        message_id: 대상 메시지 UUID 문자열.
        action_view: 프론트 아티팩트(제목/폼/표시 필드). JSONB로 저장.

    Returns:
        업데이트 성공 여부. message_id 부재 시 False.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return False

    result = await pool.execute(
        "UPDATE agent.messages SET action_view=$1::jsonb WHERE id=$2::uuid",
        json.dumps(action_view, ensure_ascii=False),
        message_id,
    )
    return result.endswith("1")


def _append_rating_clause(
    rating_filter: Optional[str],
    clauses: list[str],
    params: list[Any],
    column: str = "m.rating",
) -> None:
    """rating 필터를 WHERE 절 목록에 덧붙인다 (in-place).

    메시지를 읽는 세 쿼리(대화방 안 목록 / 전체 검색 / 피드백 모아보기)가 같은 판정을
    쓰도록 한 곳에 모았다. 분류 카탈로그는 ``app.utils.rating``이 단일 출처이며 여기서
    set 리터럴을 직접 쓰지 않는다.

    Args:
        rating_filter: ``"positive"`` | ``"negative"`` | ``"rated"`` | ``"unrated"``
            | ``None``. None(또는 그 밖의 값)이면 아무것도 붙이지 않는다 —
            라우터가 ``"any"``를 None으로 정규화한다.
        clauses: WHERE 절 조각 목록. 조건이 있으면 여기에 append 된다.
        params: 바인딩 파라미터 목록. positives 배열이 필요할 때만 append 된다.
        column: 판정할 컬럼 표현식.
    """
    from agent_arch.utils.rating import _POSITIVE_PRESETS, rated_sql, unrated_sql

    if rating_filter == "positive":
        params.append(list(_POSITIVE_PRESETS))
        clauses.append(f"{column} = ANY(${len(params)}::text[])")
    elif rating_filter == "negative":
        params.append(list(_POSITIVE_PRESETS))
        # "평가됨 AND 긍정 프리셋이 아님" — 자유 텍스트 사유가 전부 여기 걸린다.
        clauses.append(
            f"({rated_sql(column)} AND {column} <> ALL(${len(params)}::text[]))"
        )
    elif rating_filter == "rated":
        clauses.append(rated_sql(column))
    elif rating_filter == "unrated":
        clauses.append(unrated_sql(column))


async def list_messages_in_conversation(
    conversation_id: str,
    user_sub: str,
    limit: int,
    cursor: Optional[tuple[datetime, str]] = None,
    rating_filter: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    order: str = "desc",
) -> list[dict]:
    """본인 소유 대화방의 메시지를 페이지네이션 조회한다.

    소유자 검증: ``JOIN agent.conversations c ON c.id = m.conversation_id``의
    ``WHERE c.user_sub = $user_sub``로 SQL 레벨에서 강제. 타인 conversation_id
    또는 존재하지 않는 id는 빈 리스트로 응답 → 호출자(라우터)가 별도 존재 체크 후
    404로 변환한다.

    Args:
        conversation_id: 대상 대화방 UUID 문자열.
        user_sub: 소유자 sub (Keycloak).
        limit: 페이지 크기. 호출자가 ``has_more`` 판단을 위해 ``limit+1``을 SELECT
            하도록 본 함수 내부에서 ``LIMIT limit+1``로 쿼리한다.
        cursor: ``(created_at, message_id)`` 튜플. 호출자가 디코딩한 cursor 위치.
        rating_filter: ``"positive"`` | ``"negative"`` | ``"rated"`` | ``"unrated"``
            | ``None``. None이면 필터 미적용. ``_POSITIVE_PRESETS`` 카탈로그 기반
            분류 — 카탈로그 확장 시 자동 정합.
        since: 시작 시각 (포함). None이면 미제한.
        until: 종료 시각 (포함). None이면 미제한.
        order: ``"desc"`` (최신순 기본) | ``"asc"`` (대화 흐름 순). cursor 비교
            연산자가 정렬 방향에 맞춰 반전된다.

    Returns:
        dict 리스트 (최대 ``limit+1`` 개). 호출자가 trim해 ``has_more``를 판단한다.
        SELECT 컬럼: id, conversation_id, query, answer, rating, response_time,
        status_code, answer_length, created_at, action_view(표시용 카드 뷰 — 제목/폼/
        필드만, LLM 내부 신호 아님), attachments(이 턴의 첨부 메타 — 히스토리에서
        이미지 복원용, 표시 데이터). **LLM 내부 필드(llm_provider, model_name,
        *_tokens, tool_calls, langsmith_run_id 등) 미포함** — 노출 정책에 따라 절대
        추가하지 말 것.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    clauses = [
        "m.conversation_id = $1::uuid",
        "c.user_sub = $2",
        "c.archived_at IS NULL",
        # POS 플래닝 턴(응답이 기계 태그뿐이라 정화 후 answer="") 숨김 — 기존 행은
        # 키 부재 → NULL → 통과(하위호환). 계약은 memory/conversation.save_logged_in_turn 참조.
        "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'",
    ]
    params: list[Any] = [conversation_id, user_sub]

    if cursor is not None:
        params.append(cursor[0])
        params.append(cursor[1])
        op = "<" if order == "desc" else ">"
        clauses.append(
            f"(m.created_at, m.id) {op} "
            f"(${len(params) - 1}::timestamp, ${len(params)}::uuid)"
        )

    # None / "any"는 필터 미적용 — 라우터가 "any"를 None으로 정규화한다.
    _append_rating_clause(rating_filter, clauses, params)

    if since is not None:
        params.append(since)
        clauses.append(f"m.created_at >= ${len(params)}")
    if until is not None:
        params.append(until)
        clauses.append(f"m.created_at <= ${len(params)}")

    where = " AND ".join(clauses)
    order_sql = (
        "m.created_at DESC, m.id DESC" if order == "desc" else "m.created_at ASC, m.id ASC"
    )
    params.append(limit + 1)

    sql = f"""
        SELECT m.id, m.conversation_id, m.query, m.answer, m.rating,
               m.response_time, m.status_code, m.answer_length,
               m.created_at, m.action_view, m.attachments
          FROM agent.messages m
          JOIN agent.conversations c ON c.id = m.conversation_id
         WHERE {where}
         ORDER BY {order_sql}
         LIMIT ${len(params)}
    """
    rows = await pool.fetch(sql, *params)
    return [dict(r) for r in rows]


async def fetch_recent_turns(
    conversation_id: str, user_sub: str, limit: int, since: Optional[datetime] = None
) -> list[dict]:
    """본인 소유 대화방의 최근 턴을 **오래된 순**으로 돌려준다(LLM 이력 복원용).

    왜 필요한가: 대화 이력은 Redis 에만 24시간(``SESSION_TTL``) 산다. 그 뒤 같은 대화방에서
    이어 물으면 ``load_history`` 가 빈 목록을 돌려줘 "어제 그거" 가 안 됐다. 턴 자체는
    여기(agent.messages)에 영구히 있으므로, 캐시가 비었을 때 되살리는 데 쓴다
    (``memory/conversation.load_history`` → Redis 재적재).

    **기준은 "Redis 에 무엇이 들어갔는가" 다.** 되살리는 것이 캐시이지 화면 목록이 아니므로,
    적재 규칙을 그대로 따라간다.
    - ``metadata.pos_internal`` 숨김 턴 제외 — Redis 적재에서도 빠진다.
    - ``status_code = 499`` 제외. 사용자가 중단한 턴은 부분답변을 PG 에만 남기고 Redis 에는
      일부러 넣지 않는다(orchestrator 의 ``save_to_redis=False``, "재시도/다음 턴이 중단된
      부분답변에 오염되지 않게"). 이걸 빼먹으면 되살릴 때 정확히 그 오염이 되살아난다.
    - ``c.archived_at`` 은 **보지 않는다**. 보관 처리는 Redis 키를 지우지 않아서 지금도 보관된
      대화의 이력은 살아 있다. 여기서 막으면 캐시보다 인색해진다(list_messages_in_conversation
      과 다른 유일한 지점).
    - ``since`` 가 오면 그 시각 이후 턴만. ``/compact`` 로 맥락을 지운 뒤의 턴만 되살리는 데 쓴다.

    LLM 내부 컬럼(tool_calls·토큰·provider)은 선택하지 않는다. 이력 복원에 필요 없고,
    이 모듈이 그 컬럼들을 사용자 응답 경로로 흘리지 않는다는 규약을 지킨다.

    Args:
        conversation_id: 대화방 UUID.
        user_sub: 소유자 sub(다른 사람 대화는 한 건도 돌려주지 않는다).
        limit: 가져올 최근 턴 수.
        since: 이 시각 이후 턴만(맥락 비움 워터마크). None 이면 전체.

    Returns:
        ``[{"query", "answer", "created_at"}, ...]`` 오래된 순. 풀 미초기화면 [].
    """
    pool = db_pool.get_pool()
    if pool is None or not conversation_id or not user_sub:
        return []
    params: list[Any] = [conversation_id, user_sub, int(limit)]
    since_clause = ""
    if since is not None:
        params.append(since)
        since_clause = f"AND m.created_at > ${len(params)}"
    rows = await pool.fetch(
        f"""
        SELECT m.query, m.answer, m.created_at
          FROM (
                SELECT m.query, m.answer, m.created_at, m.id
                  FROM agent.messages m
                  JOIN agent.conversations c ON c.id = m.conversation_id
                 WHERE m.conversation_id = $1::uuid
                   AND c.user_sub = $2
                   AND (m.metadata->>'pos_internal') IS DISTINCT FROM 'true'
                   AND (m.status_code IS NULL OR m.status_code <> 499)
                   {since_clause}
                 ORDER BY m.created_at DESC, m.id DESC
                 LIMIT $3
               ) m
         ORDER BY m.created_at ASC, m.id ASC
        """,
        *params,
    )
    return [dict(r) for r in rows]


def _like_pattern(keyword: str) -> str:
    """사용자 입력을 ``ILIKE '%...%'`` 패턴으로 만든다 (LIKE 메타문자 이스케이프).

    사용자가 "50%" 나 "_test" 를 그대로 치면 ``%``/``_`` 가 와일드카드로 해석돼 엉뚱한
    결과가 걸린다. 백슬래시를 **가장 먼저** 치환해야 뒤이어 넣는 이스케이프까지 다시
    이스케이프되는 사고가 안 난다. SQL 쪽에는 ``ESCAPE '\\'`` 를 명시해 짝을 맞춘다.
    """
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


async def search_messages_for_user(
    user_sub: str,
    query: str,
    *,
    limit: int,
    offset: int = 0,
    rating_filter: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> list[dict]:
    """본인 소유 대화 전체를 가로질러 메시지를 키워드 검색한다 (대화 검색창용).

    매칭은 2단이다. pg_trgm ``word_similarity``(``<%``)가 오타를 관용하고, ``ILIKE``가
    **어절 중간**을 잡는다. 지원 인덱스는 migration_v26의 ``idx_messages_query_trgm`` /
    ``idx_messages_answer_trgm`` 이며 둘 다 ``gin_trgm_ops`` 라 ILIKE 도 같이 탄다.

    ILIKE를 더한 이유: ``word_similarity`` 는 **어절 시작 경계**부터만 매칭한다.
    ``word_similarity('보라', '김보라님에게')`` = 0.33 으로 기본 임계 0.6 미만이라,
    성을 뺀 이름으로는 대화를 못 찾았다(사내에서 흔한 호칭 방식이라 실사용에 걸렸다).
    대안으로 임계값을 0.3까지 낮춰 봤으나 실측상 "보라" 히트가 102건 중 33건으로 튀어
    폐기했다. ILIKE 는 깨진 케이스만 정확히 고치고 업무 단어(확인/첨부/요청/공유)의
    히트 수는 그대로 둔다.

    알려진 부작용: "니다" "에서" 같은 **문법 조각**을 검색하면 답변 본문에 널려 있어
    결과가 쏟아진다. 실사용에서 그런 검색어를 치는 사람은 없다고 보고 감수한 선택이다.

    ``memory_records.search_user_conversations``(recall_conversation 도구용)와 SQL 형태가
    닮았지만 계약이 다르다 — 이쪽은 UI 경로라 아카이브된 대화방을 제외하고 대화방 제목을
    함께 내려보낸다. 한 함수로 합치면 양쪽 계약이 얽히므로 의도적으로 분리해 둔다.

    소유자 검증: ``JOIN agent.conversations c``의 ``c.user_sub``로 강제한다.
    ``m.user_sub``는 v19에서 뒤늦게 추가된 컬럼이라 이전 행이 NULL일 수 있어 기준으로
    쓰지 않는다 (``list_messages_in_conversation``과 동일한 판단).

    Args:
        user_sub: 소유자 sub (Keycloak).
        query: 검색어(자연어). 공백뿐이면 빈 리스트.
        limit: 페이지 크기. ``has_more`` 판단을 위해 내부에서 ``LIMIT limit+1``로 조회한다.
        offset: 건너뛸 건수. 관련도 정렬이라 ``(created_at, id)`` keyset cursor를 쓸 수
            없어 offset 방식을 택했다 (``list_review_turns``와 동일한 패턴).
        rating_filter: ``"positive"`` | ``"negative"`` | ``"rated"`` | ``"unrated"``
            | ``None``. 판정은 ``_append_rating_clause`` 단일 출처.
        since: 생성 시각 하한 (**포함**). None이면 미제한.
        until: 생성 시각 상한 (**포함**). None이면 미제한. ``list_review_turns``는
            ``<`` 이지만 그건 내부 배치 계약이고, 사용자 화면 API 는
            ``list_messages_in_conversation``과 같은 포함 기준으로 맞춘다.

    Returns:
        dict 리스트 (최대 ``limit+1`` 개). 호출자가 trim해 ``has_more``를 판단한다.
        SELECT 컬럼: id, conversation_id, conversation_title, query, answer, created_at,
        score. **LLM 내부 필드(llm_provider, model_name, *_tokens, tool_calls,
        langsmith_run_id, metadata 등) 미포함** — 노출 정책에 따라 절대 추가하지 말 것.
    """
    pool = db_pool.get_pool()
    if pool is None or not user_sub or not (query or "").strip():
        return []

    # $1~$3 은 **고정**이다 — $2(검색어)/$3(LIKE 패턴)은 아래 SELECT 절 score 계산식에서도
    # 참조하므로 번호가 밀리면 점수가 엉뚱한 값으로 계산된다. 추가 필터는 뒤에만 덧붙인다.
    params: list[Any] = [user_sub, query, _like_pattern(query)]
    clauses = [
        "c.user_sub = $1",
        "c.archived_at IS NULL",
        # 예약 실행 결과 방 제외 — 목록에서 숨긴 방이 검색으로 다시 드러나지 않게(conversations.HIDDEN_KIND_SQL).
        "COALESCE(c.metadata->>'kind', '') <> 'schedule'",
        # POS 플래닝 턴 제외 — 이력 조회 API(list/get)·목록 집계·회상 검색과 같은
        # 기준. 정화 후에도 query 에는 사용자 질문이 그대로 남아 검색에는 걸리는데,
        # 목록/단건에서는 숨겨져 있어 결과를 눌러도 404 가 된다. 판단 턴과 query 가
        # 동일해 같은 질문이 두 번 뜨는 문제도 같이 막는다.
        # 기존 행은 키 부재 → NULL → 통과(하위호환).
        "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'",
        # 매칭 술어. 컬럼을 COALESCE 등 표현식으로 감싸는 순간 GIN trgm 인덱스를 못 탄다.
        r"($2 <% m.query OR $2 <% m.answer "
        r"OR m.query ILIKE $3 ESCAPE '\' OR m.answer ILIKE $3 ESCAPE '\')",
    ]

    _append_rating_clause(rating_filter, clauses, params)

    if since is not None:
        params.append(since)
        clauses.append(f"m.created_at >= ${len(params)}")
    if until is not None:
        params.append(until)
        clauses.append(f"m.created_at <= ${len(params)}")

    params.append(limit + 1)
    limit_pos = len(params)
    params.append(offset)
    offset_pos = len(params)
    where = " AND ".join(clauses)

    # COALESCE(m.answer, '') 로 감싸지 말 것 — 인덱스는 answer 컬럼 자체에 걸려 있어
    # 표현식이 달라지는 순간 GIN trgm 인덱스를 못 탄다(EXPLAIN 상 Seq Scan 고정).
    # answer 는 migration_v3 에서 NOT NULL 이므로 COALESCE 자체가 불필요하다.
    #
    # score 의 CASE 하한: 어절 중간 매칭은 word_similarity 가 실측상 전부 0.33 으로
    # 뭉쳐, "질문에 그 이름이 그대로 있음"(강한 신호)과 "답변 어딘가에 스쳐 지나감"
    # (약한 신호)이 구분되지 않는다. 0.65 는 trgm 정타(0.67) 바로 아래에 놓아 순서를
    # 만든 값이고, 정렬 품질을 보며 조정해도 되는 튜닝 값이다.
    rows = await pool.fetch(
        rf"""
        SELECT m.id, m.conversation_id, c.title AS conversation_title,
               m.query, m.answer, m.created_at,
               GREATEST(
                   word_similarity($2, m.query),
                   word_similarity($2, m.answer),
                   CASE WHEN m.query  ILIKE $3 ESCAPE '\' THEN 0.65 ELSE 0.0 END,
                   CASE WHEN m.answer ILIKE $3 ESCAPE '\' THEN 0.35 ELSE 0.0 END
               ) AS score
          FROM agent.messages m
          JOIN agent.conversations c ON c.id = m.conversation_id
         WHERE {where}
         ORDER BY score DESC, m.created_at DESC, m.id DESC
         LIMIT ${limit_pos} OFFSET ${offset_pos}
        """,
        *params,
    )
    return [dict(r) for r in rows]


async def list_rated_messages_for_user(
    user_sub: str,
    *,
    limit: int,
    cursor: Optional[tuple[datetime, str]] = None,
    rating_filter: str = "rated",
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    order: str = "desc",
) -> list[dict]:
    """대화방을 가로질러 **본인이 평가를 남긴 턴**만 시간순으로 조회한다.

    "내가 별로예요 준 게 어디 있었지"를 대화방을 하나씩 열어보지 않고 한 목록에서 훑기
    위한 쿼리다. ``list_messages_in_conversation``과 필터 규칙(rating/시각/숨김 턴)은
    같지만 스코프가 대화방 하나가 아니라 사용자 전체이고, 어느 방의 턴인지 알아야 하므로
    대화방 제목을 함께 내려보낸다.

    ``search_messages_for_user``와 달리 관련도 정렬이 아니라 시간 정렬이라
    ``(created_at, id)`` keyset cursor가 성립한다 — offset 이 아니라 cursor 를 쓴다.

    소유자 검증: ``JOIN agent.conversations c``의 ``c.user_sub``로 강제한다.
    ``m.user_sub``는 v19에서 뒤늦게 추가된 컬럼이라 이전 행이 NULL일 수 있어 기준으로
    쓰지 않는다 (``search_messages_for_user``·``list_messages_in_conversation``과 동일 판단).

    Args:
        user_sub: 소유자 sub (Keycloak).
        limit: 페이지 크기. 본 함수가 ``LIMIT limit+1``로 조회 — 호출자가 ``has_more`` 판단.
        cursor: ``(created_at, message_id)`` 튜플. 비교 연산자는 ``order``에 맞춰 반전된다.
        rating_filter: ``"positive"`` | ``"negative"`` | ``"rated"``. 그 밖의 값은
            ``"rated"``로 폴백한다 — 라우터가 이미 화이트리스트로 막지만, 이 화면에서
            "미평가 전량"이 새어 나가면 전 대화 덤프가 되므로 여기서도 닫는다.
        since: 생성 시각 하한 (포함). None이면 미제한.
        until: 생성 시각 상한 (포함). None이면 미제한.
        order: ``"desc"``(최신순 기본) | ``"asc"``(과거순).

    Returns:
        dict 리스트 (최대 ``limit+1`` 개). SELECT 컬럼: id, conversation_id,
        conversation_title, query, answer, rating, created_at.
        **LLM 내부 필드(llm_provider, model_name, *_tokens, tool_calls,
        langsmith_run_id, metadata 등) 미포함** — 노출 정책에 따라 절대 추가하지 말 것.
    """
    pool = db_pool.get_pool()
    if pool is None or not user_sub:
        return []

    if rating_filter not in ("positive", "negative", "rated"):
        rating_filter = "rated"

    clauses = [
        "c.user_sub = $1",
        "c.archived_at IS NULL",
        # POS 플래닝 턴 숨김 — 이력 조회 API 전반과 같은 기준.
        "(m.metadata->>'pos_internal') IS DISTINCT FROM 'true'",
    ]
    params: list[Any] = [user_sub]

    _append_rating_clause(rating_filter, clauses, params)

    if cursor is not None:
        params.append(cursor[0])
        params.append(cursor[1])
        op = "<" if order == "desc" else ">"
        clauses.append(
            f"(m.created_at, m.id) {op} "
            f"(${len(params) - 1}::timestamp, ${len(params)}::uuid)"
        )
    if since is not None:
        params.append(since)
        clauses.append(f"m.created_at >= ${len(params)}")
    if until is not None:
        params.append(until)
        clauses.append(f"m.created_at <= ${len(params)}")

    where = " AND ".join(clauses)
    params.append(limit + 1)
    order_sql = (
        "m.created_at DESC, m.id DESC" if order == "desc" else "m.created_at ASC, m.id ASC"
    )

    sql = f"""
        SELECT m.id, m.conversation_id, c.title AS conversation_title,
               m.query, m.answer, m.rating, m.created_at
          FROM agent.messages m
          JOIN agent.conversations c ON c.id = m.conversation_id
         WHERE {where}
         ORDER BY {order_sql}
         LIMIT ${len(params)}
    """
    rows = await pool.fetch(sql, *params)
    return [dict(r) for r in rows]


async def get_message_in_conversation(
    conversation_id: str,
    message_id: str,
    user_sub: str,
) -> Optional[dict]:
    """대화방 안 단건 메시지 조회 — 소유자·소속 검증 한 번에 강제.

    Args:
        conversation_id: 메시지가 속한 대화방 UUID 문자열.
        message_id: 메시지 UUID 문자열.
        user_sub: 소유자 sub. 다른 사용자 소유의 방·메시지면 None 반환.

    Returns:
        메시지 dict. 존재하지 않거나 타인 소유면 None — 호출자가 404로 변환.
        POS 플래닝 턴(pos_internal)도 None — 목록에서 숨긴 행이 단건 GET 으로만
        200이 되는 비일관을 막는다. SELECT 컬럼은 ``list_messages_in_conversation``과 동일.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None

    row = await pool.fetchrow(
        """
        SELECT m.id, m.conversation_id, m.query, m.answer, m.rating,
               m.response_time, m.status_code, m.answer_length,
               m.created_at, m.action_view, m.attachments
          FROM agent.messages m
          JOIN agent.conversations c ON c.id = m.conversation_id
         WHERE m.id = $1::uuid
           AND m.conversation_id = $2::uuid
           AND c.user_sub = $3
           AND (m.metadata->>'pos_internal') IS DISTINCT FROM 'true'
        """,
        message_id,
        conversation_id,
        user_sub,
    )
    return dict(row) if row else None


async def get_message_for_feedback(message_id: str) -> Optional[dict]:
    """rating 제출 시 LangSmith 연동에 필요한 메시지 정보를 조회한다.

    Args:
        message_id: 대상 메시지 UUID 문자열.

    이메일은 ``list_review_turns`` 와 같은 이유로 ``agent.users`` 조인으로 읽는다 —
    ``messages.user_email`` 은 저장 시점 스냅샷이라 비어 있는 턴이 많다. 회사는 컬럼이 없어
    ``metadata->>'company'`` 에서 읽고, 없으면 common_app 이 이메일→Employee 소속으로 보강한다.

    Returns:
        query, answer, langsmith_run_id, action_view, user_email, company를 담은 dict.
        action_view는 이 턴에 뜬 HITL 승인 카드(제목/폼/표시 필드) — 리뷰 화면 복원용.
        user_email/company는 '별로예요' 접수 시 common_app 알림에 회사·사용자를 실으려고
        함께 읽는다(10분 주기 배치 적재를 기다리지 않게 한다). service(사용량 서비스 축)도
        같은 이유다 — 접수 시점에 안 실으면 그 건이 배치가 올 때까지 리뷰 화면에서
        '미분류'로 보인다.
        해당 메시지가 없거나 DB 비활성화 시 None.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None

    row = await pool.fetchrow(
        "SELECT m.query, m.answer, m.langsmith_run_id, m.action_view, m.service, "
        "       m.model_name, m.response_time, m.web_search, "
        "       COALESCE(m.user_email, u.email) AS user_email, "
        "       m.metadata->>'company' AS company "
        "  FROM agent.messages m "
        "  LEFT JOIN agent.users u ON u.user_sub = m.user_sub "
        " WHERE m.id=$1::uuid",
        message_id,
    )
    if row is None:
        return None
    # action_view(JSONB)는 asyncpg 코덱상 문자열로 올 수 있어 dict 로 정규화한다.
    raw_view = row["action_view"]
    action_view: Optional[dict] = None
    if isinstance(raw_view, dict):
        action_view = raw_view
    elif isinstance(raw_view, str) and raw_view.strip():
        try:
            parsed = json.loads(raw_view)
            action_view = parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError):
            action_view = None
    return {
        "query": row["query"],
        "answer": row["answer"],
        "langsmith_run_id": row["langsmith_run_id"],
        "action_view": action_view,
        "user_email": row["user_email"],
        "company": row["company"],
        "service": row["service"],
        # 접수 시점에 함께 싣는다 — 기록 푸시(10분 주기)를 기다리면 '별로예요'를 누른 직후
        # 운영자가 보는 화면에서 이 축들이 비어 있다(service 와 같은 이유).
        "model_name": row["model_name"],
        "response_time": row["response_time"],
        "web_search": row["web_search"],
    }


async def get_last_message_provider(conversation_id: str) -> Optional[str]:
    """대화에서 가장 최근(llm_provider 비어있지 않은) 메시지의 LLM 프로바이더를 반환한다.

    HITL 재진입(continuation)이 메인 턴과 **동일한 프로바이더**를 이어쓰도록 복구하는 용도.
    /resume 에는 모델 선택 정보가 없어, 직전 턴이 실제 사용한 provider 를 DB 에서 되살린다.
    없거나 DB 비활성화 시 None(호출자가 기본값으로 폴백).
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None
    row = await pool.fetchrow(
        """
        SELECT llm_provider
          FROM agent.messages
         WHERE conversation_id = $1::uuid AND llm_provider IS NOT NULL
         ORDER BY created_at DESC
         LIMIT 1
        """,
        conversation_id,
    )
    return row["llm_provider"] if row else None


async def get_last_answer_model(conversation_id: str) -> Optional[tuple[Optional[str], str]]:
    """대화에서 가장 최근으로 모델이 기록된 답변의 (provider, model_name).

    승인 카드의 버튼 턴은 LLM 을 부르지 않아 자기 모델이 없다. 그러나 그 실행을 이끈 것은
    **그 카드를 띄운 직전 답변**이고, 관측(PA '에이전트 리뷰')이 모델 축으로 묻는 단위도 그것이다.
    대화 키로 찾으므로 같은 사람의 다른 대화를 끌어오지 않는다 — 실측(2026-09-21)에서 버튼을
    3.8일 뒤에 누른 턴이 있었고, 시각 근접으로 찾으면 그런 건이 엉뚱한 대화를 가리킨다.

    없거나 DB 비활성화 시 None(호출자가 비워 둔다 — 지어내지 않는다).
    """
    pool = db_pool.get_pool()
    if pool is None:
        return None
    row = await pool.fetchrow(
        """
        SELECT llm_provider, model_name
          FROM agent.messages
         WHERE conversation_id = $1::uuid AND model_name IS NOT NULL
         ORDER BY created_at DESC
         LIMIT 1
        """,
        conversation_id,
    )
    return (row["llm_provider"], row["model_name"]) if row else None


def _as_naive_utc(value: Optional[datetime]) -> Optional[datetime]:
    """aware datetime → naive UTC. asyncpg 바인딩 방어.

    ``agent.messages.created_at`` 은 **timestamp without time zone**(naive UTC 저장)이다.
    거기에 tz-aware datetime 을 바인딩하면 asyncpg 가 인코딩 단계에서
    ``can't subtract offset-naive and offset-aware datetimes`` 로 죽는다. 호출부(워터마크,
    ``datetime.now(timezone.utc)``)는 aware 를 쓰는 게 자연스러우므로, 경계인 여기서 한 번만
    맞춘다 — 호출부마다 변환하면 한 곳을 빼먹는 순간 같은 사고가 반복된다.
    """
    if value is None:
        return None
    return value.astimezone(timezone.utc).replace(tzinfo=None) if value.tzinfo else value


def _as_date(value) -> Optional[date]:
    """'YYYY-MM-DD' 문자열/datetime/date → date. asyncpg 바인딩 방어.

    SQL 에서 ``$1::date`` 로 캐스팅하면 asyncpg 가 파라미터 타입을 date 로 추론해 **문자열을
    받지 않는다**(``'str' object has no attribute 'toordinal'``). 문자열을 그대로 넘기던 실수를
    막기 위해 경계에서 date 로 바꿔 준다.
    """
    if value is None or isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return datetime.strptime(str(value).strip(), "%Y-%m-%d").date()


async def list_review_turns(
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: int = 500,
    offset: int = 0,
    require_email: bool = True,
) -> list[dict]:
    """회사 리뷰 적재 푸시용 턴 목록을 created_at 오름차순으로 페이지 조회한다.

    common_app '회사 리뷰' 패널은 LangSmith 를 소스로 썼다. 그런데 무료 플랜은 트레이스가
    **14일 보존 + 월 5,000건**이라, 오래된 구간과 한도 소진 구간이 화면에서 통째로 빠진다.
    대화 원본인 이 DB 를 소스로 바꾸기 위해 여기서 읽어 푸시한다
    (app.services.common_app.review_history).

    **이메일은 반드시 agent.users 조인으로 읽는다.** ``messages.user_email`` 은 저장 시점
    스냅샷이라 비어 있는 턴이 많다(2026-07-31 기준 460턴, 그중 457턴이 실사용자 — 대부분
    ``query='(승인 실행됨)'`` 카드 승인 턴). 그 턴들은 ``langsmith_run_id`` 도 없어 이 푸시가
    아니면 리뷰 화면에 들어올 방법이 아예 없다. 스냅샷만 보면 조용히 누락된다.

    회사는 컬럼이 없어 metadata->>'company' 에서 읽는다(2026-07-30 이후 저장분에만 있다).
    그 이전 턴은 회사가 비어 나가고, common_app 이 이메일→Employee 소속으로 보강한다.

    Args:
        since: 이 시각 **이상**(>=)만. None이면 처음부터.
        until: 이 시각 **미만**(<)만. None이면 끝까지.
        limit: 한 페이지 건수. offset: 시작 위치.
        require_email: True(기본)면 신원이 확인되는 턴만. 게스트/비로그인 턴을 운영 리뷰
            화면에 섞지 않기 위한 기본값이다. 전수 점검이 필요할 때만 False.

    Returns:
        list[dict]: {message_id, created_at, user_email, company, query, answer, rating,
                     langsmith_run_id, service, model_name, response_time,
                     web_search, finish_reason, status_code, capture_skip, llm_provider}.
                     DB 비활성화 시 빈 리스트.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    conds = []
    args: list[Any] = []
    if since is not None:
        args.append(_as_naive_utc(since))
        conds.append(f"m.created_at >= ${len(args)}")
    if until is not None:
        args.append(_as_naive_utc(until))
        conds.append(f"m.created_at < ${len(args)}")
    if require_email:
        # 게스트(u.is_guest)와 신원 미상(양쪽 이메일 모두 없음) 턴은 제외한다.
        conds.append("COALESCE(m.user_email, u.email) IS NOT NULL")
        conds.append("u.is_guest IS NOT TRUE")
    where = f"WHERE {' AND '.join(conds)}" if conds else ""

    args.append(limit)
    limit_pos = len(args)
    args.append(offset)
    offset_pos = len(args)

    rows = await pool.fetch(
        f"""
        SELECT m.id, m.created_at, m.query, m.answer, m.rating, m.langsmith_run_id,
               COALESCE(m.user_email, u.email) AS user_email,
               m.metadata->>'company' AS company,
               m.service AS service,
               -- 답변이 만들어진 조건. 모델·소요 시간은 v7 부터 이 표에 있었지만 푸시에
               -- 실리지 않아 PA 리뷰 화면에서 볼 수 없었다.
               m.model_name AS model_name,
               m.response_time AS response_time,
               m.web_search AS web_search,
               -- 턴 실패 분류의 원천(계약 1장). common_app 이 이 넷으로 failure_category 를
               -- 파생한다 — status_code 가 없으면 '미기록'이지 정상이 아니다.
               m.finish_reason AS finish_reason,
               m.status_code AS status_code,
               m.metadata->>'capture_skip' AS capture_skip,
               m.llm_provider AS llm_provider
          FROM agent.messages m
          LEFT JOIN agent.users u ON u.user_sub = m.user_sub
          {where}
         ORDER BY m.created_at ASC, m.id ASC
         LIMIT ${limit_pos} OFFSET ${offset_pos}
        """,
        *args,
    )
    return [
        {
            "message_id": str(r["id"]),
            "created_at": r["created_at"],
            "user_email": r["user_email"],
            "company": r["company"],
            "query": r["query"],
            "answer": r["answer"],
            "rating": r["rating"],
            "langsmith_run_id": r["langsmith_run_id"],
            # 사용량 집계(list_daily_usage)와 같은 컬럼이다. 리뷰 화면이 메신저 에이전트 대화와
            # 브라우저 패널 대화를 구분하는 축 — 여기서 빼면 두 면이 한 목록에 섞인다.
            "service": r["service"],
            # 리뷰 화면이 "왜 이런 답이 나왔나"를 되짚는 세 축. 없는 턴은 None 그대로 나간다 —
            # common_app 이 실려 온 축만 얹으므로 빈 값이 기존 적재를 지우지 않는다.
            "model_name": r["model_name"],
            "response_time": r["response_time"],
            "web_search": r["web_search"],
            "finish_reason": r["finish_reason"],
            "status_code": r["status_code"],
            "capture_skip": r["capture_skip"],
            "llm_provider": r["llm_provider"],
        }
        for r in rows
    ]


async def list_daily_usage(
    since_day: Optional[str] = None,
    until_day: Optional[str] = None,
) -> list[dict]:
    """LLM 사용량을 (KST 일자, 사용자, 회사, 모델, 서비스) 단위로 집계한다. 관측 대시보드 적재용.

    common_app 의 'LLM 사용 대시보드'는 LangSmith root run 을 긁어 회사별로 집계했다. 무료
    플랜이 월 5,000 트레이스에서 하드 캡이라 한도를 넘긴 순간 화면이 통째로 비어, 소스를 대화
    원본인 이 DB 로 옮긴다.

    ``messages.user_email`` 은 스냅샷이라 비어 있는 턴이 있어 ``agent.users`` 조인으로
    보강하고, 그래도 신원이 없으면 NULL 로 보내 common_app 이 '미매핑'으로 합산한다 —
    총량에서 조용히 빠지지 않게 하기 위함이다.

    **회사는 요청 시점의 선택 회사(metadata->>'company')를 함께 보낸다.** 이메일만 보내면
    common_app 이 활성 Employee 소속으로 역추적하는데, 소속이 2곳 이상인 사용자는 어느 회사
    컨텍스트에서 물었는지 알 수 없어 판정을 포기하고 '미매핑'으로 빠진다(임의 선택은 조용히
    틀린 태깅이라 금지). 그 정보는 요청 시점에만 있고 이 컬럼이 유일한 기록이다 — 리뷰 적재
    (list_review_history)가 이미 같은 값을 쓴다. **2026-07-30 이후 저장분에만 있고** 그 전
    턴은 NULL 로 나가 common_app 이 이메일→Employee 로 폴백한다.

    **서비스(면)도 축으로 보낸다.** 사용자에게 청구할 면이 에이전트 브라우저 패널 하나가 아니게
    되면서(메신저 에이전트·AI 도면 등) "어디서 얼마 썼나"를 물을 수 있어야 한다. 축이 없는 과거
    턴은 NULL 로 나가 common_app 이 '미분류' 버킷으로 합산한다 — 총량에서 빠지지 않는다.
    푸시는 이 축으로 **서비스별로 나눠** 보낸다(한 서비스의 실패가 다른 서비스 행을 지우지
    않게 하기 위함 — common_app 의 교체 키가 (일자 × 서비스)다).

    회사·서비스가 축으로 들어가 행 수가 (사용자 × 회사 × 모델 × 서비스)로 늘어난다 — 한
    사용자가 하루에 두 회사로 물으면 2행이다. 그게 이 변경의 목적이고, 상한(PUSH_DAY_ROW_MAX)은
    이제 (일자 × 서비스) 단위로 적용된다.

    일자 경계는 **KST**다(대시보드 표기가 KST). created_at 은 naive UTC 라 9시간을 더해 자른다.

    Args:
        since_day: 'YYYY-MM-DD'(KST, 포함). None이면 처음부터.
        until_day: 'YYYY-MM-DD'(KST, **제외**). None이면 끝까지.

    Returns:
        list[dict]: {usage_date, user_email, company, model_name, service, request_count,
                     input_tokens, output_tokens, total_tokens, missing_token_rows,
                     last_activity_at}.
                     ``missing_token_rows`` 는 **LLM 을 호출했는데** 토큰이 NULL 인 턴 수 —
                     크레딧이 과소 집계되는 구간을 드러내기 위해 함께 올린다.
                     ``failed_turns`` 는 그중 **호출 자체가 실패한** 턴 수다
                     (``finish_reason`` = error / fallback). 연결 실패·429·503·401·400 이라
                     추론이 시작되지 않아 **토큰 소비가 없다** — 청구 누락이 아니라 답변
                     실패이므로 화면이 둘을 구분해야 한다(2026-09-03 실측: 남은 미기록
                     93건 중 31건이 이 실패였고, 원인은 vLLM 연결 실패·Gemini 503·OpenAI
                     크레딧 소진이었다).
                     DB 비활성화 시 빈 리스트.

    **LLM 호출이 없는 턴은 집계에서 제외한다.** 승인 카드 실행·거절 기록
    (``query='(승인 실행됨)'`` / ``'(초안 거절됨)'``)은 모델을 부르지 않고 결과만 남기는
    턴이라 토큰이 없는 것이 정상이다. 그런데 이 원장은 **LLM 사용량** 원장이므로 그 턴이
    섞이면 두 숫자가 동시에 틀어진다 — 요청 수가 부풀고, ``missing_token_rows`` 가
    "토큰 누락"으로 잡혀 화면이 "크레딧이 적게 잡혔다"고 과장한다. 2026-09-03 실측에서
    미기록 367건 중 274건(75%)이 이 액션 기록이었다.

    판별은 ``llm_invoke_count IS NOT NULL OR total_tokens IS NOT NULL`` 이다 — 액션 기록은
    둘 다 NULL 이고, LLM 턴은 최소 하나가 채워진다(그 시절 컬럼이 없던 옛 턴도 토큰으로
    걸러진다).
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    # LLM 호출이 없는 턴(승인/거절 액션 기록)은 이 원장의 대상이 아니다.
    conds = ["(m.llm_invoke_count IS NOT NULL OR m.total_tokens IS NOT NULL)"]
    args: list[Any] = []
    if since_day:
        args.append(_as_date(since_day))
        conds.append(f"(m.created_at + interval '9 hours')::date >= ${len(args)}")
    if until_day:
        args.append(_as_date(until_day))
        conds.append(f"(m.created_at + interval '9 hours')::date < ${len(args)}")
    where = f"WHERE {' AND '.join(conds)}"

    rows = await pool.fetch(
        f"""
        SELECT to_char(m.created_at + interval '9 hours', 'YYYY-MM-DD') AS usage_date,
               COALESCE(m.user_email, u.email) AS user_email,
               m.metadata->>'company' AS company,
               m.model_name,
               m.service AS service,
               count(*) AS request_count,
               COALESCE(sum(m.prompt_tokens), 0) AS input_tokens,
               COALESCE(sum(m.completion_tokens), 0) AS output_tokens,
               COALESCE(sum(m.total_tokens), 0) AS total_tokens,
               count(*) FILTER (WHERE m.total_tokens IS NULL) AS missing_token_rows,
               count(*) FILTER (
                   WHERE m.total_tokens IS NULL
                     AND m.finish_reason IN ('error', 'fallback')
               ) AS failed_turns,
               max(m.created_at) AS last_activity_at
          FROM agent.messages m
          LEFT JOIN agent.users u ON u.user_sub = m.user_sub
          {where}
         GROUP BY 1, 2, 3, 4, 5
         ORDER BY 1, 2, 3, 4, 5
        """,
        *args,
    )
    return [
        {
            "usage_date": r["usage_date"],
            "user_email": r["user_email"],
            "company": r["company"],
            "model_name": r["model_name"],
            "service": r["service"],
            "request_count": r["request_count"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
            "total_tokens": r["total_tokens"],
            "missing_token_rows": r["missing_token_rows"],
            "failed_turns": r["failed_turns"],
            "last_activity_at": r["last_activity_at"],
        }
        for r in rows
    ]


async def list_daily_meters(
    since_day: Optional[str] = None,
    until_day: Optional[str] = None,
) -> list[dict]:
    """벤더 호출 계측(agent.usage_meters)을 일별 사용량과 **같은 축**으로 집계한다.

    축은 list_daily_usage 와 같다 — (KST 일자, 사용자, 회사, 모델, 서비스). 그래야 푸시 단계에서
    두 결과를 한 행으로 합칠 수 있다. meter 별로 열을 나눠 돌려준다:

    - ``llm_*``: 벤더 경계에서 센 토큰(턴 밖 호출 포함). USAGE_METER_LEDGER_FROM 이후 날짜는
      이 값이 턴 토큰을 **대체**한다(usage_daily._merge_meters). ``llm_cached_input_tokens``·
      ``llm_cache_write_tokens`` 는 입력 토큰 중 캐시 적중·쓰기 몫이다(migration_v45).
    - ``brave_search_calls`` / ``openai_search_calls``: 성공한 검색 API 호출 수.
    - ``stt_seconds``: 전사한 오디오 길이(초).

    Args:
        since_day: 'YYYY-MM-DD'(KST, 포함). None이면 처음부터.
        until_day: 'YYYY-MM-DD'(KST, **제외**). None이면 끝까지.

    Returns:
        list[dict]. 표가 없거나(migration_v42 미적용) DB 비활성화면 빈 리스트.

    Raises:
        Exception: 표 부재 외의 조회 실패 — 0건과 구분해야 하므로 삼키지 않는다.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    conds: list[str] = []
    args: list[Any] = []
    if since_day:
        args.append(_as_date(since_day))
        conds.append(f"(created_at + interval '9 hours')::date >= ${len(args)}")
    if until_day:
        args.append(_as_date(until_day))
        conds.append(f"(created_at + interval '9 hours')::date < ${len(args)}")
    where = f"WHERE {' AND '.join(conds)}" if conds else ""

    try:
        rows = await pool.fetch(
            f"""
            SELECT to_char(created_at + interval '9 hours', 'YYYY-MM-DD') AS usage_date,
                   user_email,
                   company,
                   model_name,
                   service,
                   count(*) FILTER (WHERE meter = 'llm_tokens') AS llm_calls,
                   count(*) FILTER (WHERE meter = 'llm_tokens' AND usage_missing) AS llm_missing,
                   COALESCE(sum(input_tokens) FILTER (WHERE meter = 'llm_tokens'), 0) AS llm_input_tokens,
                   COALESCE(sum(output_tokens) FILTER (WHERE meter = 'llm_tokens'), 0) AS llm_output_tokens,
                   COALESCE(sum(cached_input_tokens) FILTER (WHERE meter = 'llm_tokens'), 0) AS llm_cached_input_tokens,
                   COALESCE(sum(cache_write_tokens) FILTER (WHERE meter = 'llm_tokens'), 0) AS llm_cache_write_tokens,
                   COALESCE(sum(quantity) FILTER (
                       WHERE meter = 'web_search_call' AND provider = 'brave'), 0) AS brave_search_calls,
                   COALESCE(sum(quantity) FILTER (
                       WHERE meter = 'web_search_call' AND provider = 'openai'), 0) AS openai_search_calls,
                   COALESCE(sum(quantity) FILTER (WHERE meter = 'stt_seconds'), 0) AS stt_seconds,
                   max(created_at) AS last_activity_at
              FROM agent.usage_meters
              {where}
             GROUP BY 1, 2, 3, 4, 5
             ORDER BY 1, 2, 3, 4, 5
            """,
            *args,
        )
    except asyncpg.exceptions.UndefinedTableError:
        # 표가 아직 없는 인스턴스(migration_v42 미적용)에서도 턴 사용량 푸시는 돌아야 한다.
        logger.warning("usage_meters 표 없음 — 계측 없이 턴 사용량만 푸시", exc_info=True)
        return []
    # 그 밖의 실패(열 없음 — migration_v45 미적용 —, 연결 끊김, 타임아웃)는 '계측 0건'이 아니다.
    # [] 로 삼키면 대체 시작일 이후 날짜의 토큰이 0 으로 푸시되고 common_app 이 그 날을 지운 뒤
    # 0 으로 다시 넣는다. 올려서 호출자(usage_daily.sync_usage_daily)가 그 주기 푸시를 건너뛴다.
    return [
        {
            "usage_date": r["usage_date"],
            "user_email": r["user_email"],
            "company": r["company"],
            "model_name": r["model_name"],
            "service": r["service"],
            "llm_calls": int(r["llm_calls"] or 0),
            "llm_missing": int(r["llm_missing"] or 0),
            "llm_input_tokens": int(r["llm_input_tokens"] or 0),
            "llm_output_tokens": int(r["llm_output_tokens"] or 0),
            "llm_cached_input_tokens": int(r["llm_cached_input_tokens"] or 0),
            "llm_cache_write_tokens": int(r["llm_cache_write_tokens"] or 0),
            "brave_search_calls": int(r["brave_search_calls"] or 0),
            "openai_search_calls": int(r["openai_search_calls"] or 0),
            "stt_seconds": float(r["stt_seconds"] or 0),
            "last_activity_at": r["last_activity_at"],
        }
        for r in rows
    ]


async def list_company_user_usage(
    company: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[dict]:
    """회사 1곳의 LLM 사용량을 (사용자, provider) 단위로 집계한다. DMP '직원별 LLM 사용량' 탭용.

    이 탭은 LangSmith root run 을 긁어 집계했는데, 무료 플랜이 월 5,000 트레이스에서 하드
    캡이라 한도를 넘긴 뒤로 늘 0 이 나왔다. 일별 원장(list_daily_usage)과 같은 소스·같은
    필터로 옮긴다 — 두 화면의 숫자가 같은 원본에서 나와야 서로 맞춰 볼 수 있다.

    회사는 요청 시점의 선택 회사(``metadata->>'company'``)다. **2026-07-30 이전 턴에는 이 값이
    없어** 어느 회사 사용량인지 알 수 없으므로 집계에 들어오지 않는다.

    Args:
        company: 회사명(한글, Company.name).
        start: created_at 하한(naive UTC, 포함). None이면 처음부터.
        end: created_at 상한(naive UTC, 포함). None이면 끝까지.

    Returns:
        list[dict]: {user_sub, user_email, provider, turns, prompt_tokens,
                     completion_tokens, total_tokens}. DB 비활성화 시 빈 리스트.
    """
    pool = db_pool.get_pool()
    if pool is None:
        return []

    # LLM 호출이 없는 턴(승인/거절 액션 기록)은 사용량이 아니다 — list_daily_usage 와 같은 판별.
    args: list[Any] = [company]
    conds = [
        "m.metadata->>'company' = $1",
        "(m.llm_invoke_count IS NOT NULL OR m.total_tokens IS NOT NULL)",
    ]
    if start is not None:
        args.append(_as_naive_utc(start))
        conds.append(f"m.created_at >= ${len(args)}")
    if end is not None:
        args.append(_as_naive_utc(end))
        conds.append(f"m.created_at <= ${len(args)}")

    rows = await pool.fetch(
        f"""
        SELECT m.user_sub,
               COALESCE(m.user_email, u.email) AS user_email,
               m.llm_provider AS provider,
               count(*) AS turns,
               COALESCE(sum(m.prompt_tokens), 0) AS prompt_tokens,
               COALESCE(sum(m.completion_tokens), 0) AS completion_tokens,
               COALESCE(sum(m.total_tokens), 0) AS total_tokens
          FROM agent.messages m
          LEFT JOIN agent.users u ON u.user_sub = m.user_sub
         WHERE {' AND '.join(conds)}
         GROUP BY 1, 2, 3
        """,
        *args,
    )
    return [
        {
            "user_sub": r["user_sub"],
            "user_email": r["user_email"],
            "provider": r["provider"],
            "turns": r["turns"],
            "prompt_tokens": r["prompt_tokens"],
            "completion_tokens": r["completion_tokens"],
            "total_tokens": r["total_tokens"],
        }
        for r in rows
    ]
