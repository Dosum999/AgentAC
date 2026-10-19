from contextvars import ContextVar
from typing import Optional

_user_context: ContextVar[dict] = ContextVar("user_context", default={})
_profile_context: ContextVar[Optional[dict]] = ContextVar(
    "profile_context", default=None
)
# 요청에 첨부된 이미지(ImageInput 목록). 도구/스킬은 tool args 로 이미지 바이트를 받지
# 못하므로(executor 는 LLM 이 만든 텍스트/JSON 만 전달), OCR 계열 스킬이 "이번 요청에
# 첨부된 이미지"를 읽을 수 있게 요청 스코프 ContextVar 로 흘려보낸다. _cancel_event /
# tool_search._discovered 와 동일한 패턴 — orchestrator 가 set 하면 execute_tools 의
# asyncio.gather 자식 Task 들이 컨텍스트를 복사 상속해 값을 본다(스킬은 읽기만 한다).
_image_context: ContextVar[list] = ContextVar("image_context", default=[])
# 이번 요청에서 file-service 로 영속화한 첨부파일 메타([{file_id, filename, content_type}]).
# 턴 저장(save_logged_in_turn)이 읽어 agent.messages.attachments 에 기록한다 — 히스토리
# 조회 시 presigned URL 을 재발급해 과거 이미지를 복원하기 위함. _image_context 와 동일하게
# orchestrator._preprocess_request 가 set 하고 같은 Task 체인의 저장 지점이 읽는다.
_attachment_context: ContextVar[list] = ContextVar("attachment_context", default=[])
# 현재 요청의 대화 ID. 도구/스킬은 tool args 로 conversation_id 를 받지 못하므로(executor 는
# LLM 이 만든 인자만 전달), 대화 스코프 Redis stash(예: 영수증 OCR document_body)를 읽고
# 쓰려는 스킬이 이 값을 참조한다. orchestrator._preprocess_request 와 HITL 재개(resume)
# 경로가 set 하며, execute_tools 자식 Task 가 컨텍스트를 복사 상속한다(이미지 컨텍스트와 동일).
_conversation_id: ContextVar[Optional[str]] = ContextVar("conversation_id", default=None)

# 메신저 스레드의 **지금 대화**. 에이전트가 답글을 다는 그 스레드에서 사람들끼리 나눈 말은 대화
# 이력에 없다 — 이력은 에이전트와 그 사람이 주고받은 턴만 쌓이기 때문이다. 도구로 읽게 두면
# 라우팅이 도메인을 바꾸는 순간 볼 방법이 사라져서(2026-09-22 실측), 봇이 직접 읽어 여기
# 실어 보내고 턴 컨텍스트가 매 턴 주입한다. 메신저 경로만 값을 넣는다.
_thread_context: ContextVar[list] = ContextVar("thread_context", default=[])

# SCMS 메신저의 ``[거래 구조]`` 블록(발주사·공급사·협력사 차수). 봇이 스레드·SCMS 주문에서 갱신해 만들고
# 턴 컨텍스트가 스레드 블록 바로 뒤에 싣는다(app/services/scms/deal). SCMS 채널 밖에서는 None.
_deal_context: ContextVar[Optional[str]] = ContextVar("deal_context", default=None)

# 웹 검색 opt-in(요청 범위). 패널 토글(web_search=true) 과 회사 설정(agent_web_search_enabled)이 둘 다 켜졌을 때만 True.
# 기본 None/False = 웹 도메인 닫힘 — 라우터 카드·관문·sticky·도구 실행이 전부 이 값을 본다(fail-closed).
_web_search: ContextVar[Optional[bool]] = ContextVar("web_search_enabled", default=None)
# 이번 턴에 열어도 되는 URL(web_fetch) — 사용자가 준 URL + 이번 턴 웹 검색 결과에 나온 URL. 턴 시작에 새 집합을 만들어 두면
# LangGraph 노드·gather 자식 태스크가 같은 객체를 공유한다(나중에 만든 집합은 형제 태스크로 전파되지 않는다).
_web_seen_urls: ContextVar[Optional[dict]] = ContextVar("web_seen_urls", default=None)   # {url 키: 제목}
# 메신저 SCMS 발주요청 — 이 요청에서 scms_order 도메인을 라우팅 후보로 낼지. 메신저 봇이
# 요청자 회사가 허용 목록(SCMS_MESSENGER_COMPANIES)에 있을 때만 켠다. 꺼져 있으면 라우터 카드도
# 도구도 없어 이 기능 이전과 글자까지 같다(패널·다른 회사).
_scms_order: ContextVar[bool] = ContextVar("scms_order_enabled", default=False)
# 이번 턴에 쓰면 안 되는 도구 — {이름: 모델에 돌려줄 사유}. 메신저 SCMS 공개 범위(원청이 있을 수 있는 채널에서 플랫폼
# 쪽 사람의 SCMS 내부 조회)가 넣는다. 바인딩에서 빼고(orchestrator._prepare_tools) 실행기에서도 막는다(react/executor).
_withheld_tools: ContextVar[Optional[dict]] = ContextVar("withheld_tools", default=None)
# 예약 실행(무인) — 지금 이 턴을 보고 있는 사람이 없다. 예약 실행기가 회차를 돌리기 전에 켠다.
# 켜져 있으면 권한 게이트가 승인 필요 도구(메일 발송·상신 등)를 전부 막는다: 저장해 둔 요청은 실행 중
# 액션에 대한 동의가 아니다(Claude 루틴과 같은 원칙). 저장 깔때기는 이 턴을 메모리 추출에서 뺀다.
_unattended_run: ContextVar[Optional[dict]] = ContextVar("unattended_run", default=None)
# 답변 방식(요청 범위). 엔드포인트(_apply_answer_mode)가 회사 설정과 대조해 확정한 값을 넣는다.
# 저장 지점이 10곳이라 파라미터가 아니라 여기로 흘려보낸다 — 첨부·회사·서비스 축과 같은 이유다.
_answer_mode: ContextVar[Optional[str]] = ContextVar("answer_mode", default=None)
# 직전 턴에 사용자가 [거절]한 HITL 초안({tool_name, arguments, fields}). orchestrator 가
# 요청 전처리에서 Redis stash 를 읽어 심고, supervisor 의 라우팅 보정(_decide)과 워커 메시지
# 스코핑(_scope_messages)이 읽는다. 함수 인자로 뚫지 않는 이유는 _image_context 와 같다 —
# run_supervisor / run_supervisor_events 두 진입점과 그 아래 호출 체인 전체에 kwarg 를
# 추가해야 하는데, 값은 읽기 전용 요청 스코프 정보라 ContextVar 가 자연스럽다.
_rejected_draft: ContextVar[Optional[dict]] = ContextVar("rejected_draft", default=None)
# 되물을 값이 남은 초안이 진행 중이라는 포인터({tool_name, template_code}).
_collecting_draft: ContextVar[Optional[dict]] = ContextVar("collecting_draft", default=None)


def set_user_context(user_info: dict) -> None:
    """현재 비동기 컨텍스트에 사용자 정보를 저장한다.

    Args:
        user_info (dict): 사용자 정보 딕셔너리 (sub, email, company 등)
    """
    _user_context.set(user_info or {})


def get_user_context() -> dict:
    """현재 비동기 컨텍스트의 사용자 정보를 반환한다.

    Returns:
        dict: 저장된 사용자 정보 딕셔너리. 없으면 빈 딕셔너리
    """
    return _user_context.get()


def set_image_context(images: Optional[list]) -> None:
    """현재 요청 컨텍스트에 첨부 이미지 목록(ImageInput)을 저장한다.

    OCR 계열 스킬이 ``get_image_context()`` 로 읽어 소비한다(읽기 전용). 값은
    ``app.models.agent.ImageInput`` 리스트로, 각 항목은 ``data``(base64/dataURL)·
    ``mime_type`` 를 가진다. 이미지가 없으면 빈 리스트를 심는다.

    Args:
        images: 요청에 첨부된 ImageInput 목록. ``None`` 이면 빈 리스트로 저장.
    """
    _image_context.set(images or [])


def get_image_context() -> list:
    """현재 요청 컨텍스트의 첨부 이미지 목록을 반환한다.

    Returns:
        list: ImageInput 목록. 첨부가 없으면 빈 리스트.
    """
    return _image_context.get()


def set_attachment_context(attachments: Optional[list]) -> None:
    """현재 요청 컨텍스트에 영속화된 첨부파일 메타 목록을 심는다.

    엔드포인트가 file-service 업로드 후 만든
    ``[{"file_id", "filename", "content_type"}]`` 목록. 턴 저장 시
    ``save_logged_in_turn`` 이 읽어 ``agent.messages.attachments`` 에 기록한다.

    Args:
        attachments: 첨부 메타 목록. ``None`` 이면 빈 리스트로 저장.
    """
    _attachment_context.set(attachments or [])


def get_attachment_context() -> list:
    """현재 요청 컨텍스트의 영속화된 첨부파일 메타 목록을 반환한다.

    Returns:
        list: ``[{"file_id", "filename", "content_type"}]`` 목록. 없으면 빈 리스트.
    """
    return _attachment_context.get()


def set_conversation_id(conversation_id: Optional[str]) -> None:
    """현재 요청 컨텍스트에 대화 ID 를 저장한다.

    대화 스코프 Redis stash(영수증 OCR document_body 등)를 읽고 쓰는 스킬이
    ``get_conversation_id()`` 로 읽는다. None 이면 stash 를 건너뛴다.

    Args:
        conversation_id: 현재 대화 ID. 게스트/미확정이면 None.
    """
    _conversation_id.set(conversation_id)


def get_conversation_id() -> Optional[str]:
    """현재 요청 컨텍스트의 대화 ID 를 반환한다 (없으면 None)."""
    return _conversation_id.get()


def set_rejected_draft_context(rejected: Optional[dict]) -> None:
    """직전 턴에 거절된 HITL 초안을 현재 요청 컨텍스트에 심는다.

    Args:
        rejected: ``{"tool_name": str, "arguments": dict, "fields": list}`` 또는 None(없음).
    """
    _rejected_draft.set(rejected if isinstance(rejected, dict) else None)


def get_rejected_draft_context() -> Optional[dict]:
    """직전 턴에 거절된 HITL 초안을 반환한다 (없으면 None)."""
    return _rejected_draft.get()


def set_collecting_draft_context(collecting: Optional[dict]) -> None:
    """되물을 값이 남은 초안 포인터를 현재 요청 컨텍스트에 심는다.

    supervisor 가 이 값으로 짧은 답변 턴을 조회가 아닌 **실행**으로 태워, 승인 도구가
    바인딩되지 않아 초안이 못 만들어지던 사고를 막는다. 값이 아니라 포인터만 담는다.

    Args:
        collecting: ``{"tool_name": str, "template_code": str|None}`` 또는 None(없음).
    """
    _collecting_draft.set(collecting if isinstance(collecting, dict) else None)


def get_collecting_draft_context() -> Optional[dict]:
    """진행 중인 수집 초안 포인터를 반환한다 (없으면 None)."""
    return _collecting_draft.get()


def set_profile_context(profile: Optional[dict]) -> None:
    """Frappe 프로필 dict를 현재 요청 컨텍스트에 저장한다.

    ``user_info`` 와 분리해 보관해, 권한 raw (``assigned_roles`` /
    ``permissions`` / ``service_access``) 가 시스템 프롬프트 빌더로
    흘러들지 않도록 격리한다.

    Args:
        profile: ``app.services.user_profile.get_profile`` 응답 dict.
            ``None`` 이면 미인증 또는 fetch 실패.
    """
    _profile_context.set(profile)


def get_profile_context() -> Optional[dict]:
    """현재 요청 컨텍스트의 Frappe 프로필 dict를 반환한다.

    Returns:
        프로필 dict 또는 ``None`` (미설정).
    """
    return _profile_context.get()


# ── 장기 메모리 회상 결과(구조체) ─────────────────────────────────────────────
# _build_messages 가 회상한 RecallResult 를 턴 전체(도구 계획·라우터·워커·종합기)가 읽을 수 있게
# 요청 스코프로 흘려보낸다. 텍스트 블록은 워커 스코프 재작성에서 사라질 수 있지만 ContextVar 는
# asyncio.gather 자식(워커·도구 실행)에 복사돼 살아남는다(supervisor/tool_search 와 같은 근거).
_memory_recall: ContextVar[Optional[object]] = ContextVar("memory_recall", default=None)


def set_memory_recall(recall: Optional[object]) -> None:
    _memory_recall.set(recall)


def get_memory_recall() -> Optional[object]:
    return _memory_recall.get()


def set_web_search_enabled(flag: Optional[bool]) -> None:
    """이 요청에서 웹 검색을 쓸 수 있는가(패널 토글 ∧ 회사 설정). 엔드포인트가 계산해 넣고 오케스트레이터가 턴 시작에 다시 넣는다."""
    _web_search.set(bool(flag) if flag is not None else None)
    _web_seen_urls.set({})


def _url_key(url: str) -> str:
    """URL 비교 키 — 호스트 소문자·www 제거, 경로 끝 '/' 제거, 쿼리·조각 무시(모델이 추적 파라미터를 떼거나 붙인다)."""
    from urllib.parse import urlsplit
    try:
        p = urlsplit(str(url).strip())
    except Exception:
        return ""
    host = (p.hostname or "").lower().removeprefix("www.")
    return f"{host}{p.path.rstrip('/')}" if host else ""


def remember_web_urls(urls) -> None:
    """web_fetch 로 열어도 되는 URL 을 이번 턴 기록에 더한다(검색 결과·사용자 입력). 항목은 URL 또는 (URL, 제목).

    제목은 원 페이지가 막혔을 때 검색 인덱스에서 같은 페이지를 다시 찾는 검색어로 쓴다(web_fetch 폴백).
    """
    seen = _web_seen_urls.get()
    if seen is None:
        seen = {}
        _web_seen_urls.set(seen)
    for item in urls or []:
        url, title = (item if isinstance(item, tuple) else (item, ""))
        k = _url_key(url)
        if k and (k not in seen or (title and not seen[k])):
            seen[k] = str(title or "")


def web_url_seen(url: str) -> bool:
    k = _url_key(url)
    return bool(k) and k in (_web_seen_urls.get() or {})


def web_url_title(url: str) -> str:
    """이번 턴 검색 결과에 적힌 그 URL 의 제목(없으면 빈 문자열)."""
    return (_web_seen_urls.get() or {}).get(_url_key(url), "")


def web_url_key(url: str) -> str:
    return _url_key(url)


def web_search_enabled() -> bool:
    return _web_search.get() is True


def web_search_choice() -> Optional[bool]:
    """이 요청의 웹 검색 여부를 **미기록(None)까지 구분해** 돌려준다.

    ``web_search_enabled`` 는 게이트용이라 None 을 False 로 접는다(fail-closed). 저장은
    반대다 — 접으면 이 축이 생기기 전 적재분과 '웹을 쓰지 않은 턴'이 같은 값이 되어, 리뷰
    화면에서 그 구분을 세는 순간 조용히 틀린다.
    """

    return _web_search.get()


def set_answer_mode(mode: Optional[str]) -> None:
    """이 요청의 답변 방식('quick'|'reasoning'). 엔드포인트가 확정값을 넣는다."""

    _answer_mode.set((mode or "").strip() or None)


def answer_mode() -> Optional[str]:
    """이 요청의 답변 방식. 고르지 않았거나 패널 밖 경로(메신저 등)면 None = 기록 없음."""

    return _answer_mode.get()
# ── 직전 어시스턴트 답변 ──────────────────────────────────────────────────────
# "방금 분석한 내용을 워드파일로 만들어줘" 류의 요청에서, 파일 본문은 **이미 직전 답변에
# 있다.** 그걸 모델이 도구 인자로 통째로 다시 받아쓰게 하면 실패한다 — 2026-09-14 운영
# 실측에서 create_file 이 13회 연속 빈 인자(`filename 이 필요합니다`)로 실패하고 반복
# 상한을 소진했다. vLLM(max_tokens 1024)과 Gemini(8192) **양쪽 모두** 같은 증상이라
# 프로바이더 문제가 아니라 "긴 본문을 인자로 재생성"이라는 작업 자체가 원인이다.
#
# 그래서 본문을 요청 스코프로 흘려보내고, 도구는 인자가 비면 이 값을 쓴다. 모델이 만들
# 인자는 파일명·형식뿐이 된다(토큰·지연도 함께 준다).
_prior_answer: ContextVar[Optional[str]] = ContextVar("prior_answer", default=None)


def set_prior_answer(text: Optional[str]) -> None:
    """이번 턴 직전의 어시스턴트 답변을 요청 스코프에 심는다.

    Args:
        text: 직전 답변 전문. 없으면 ``None``.
    """
    _prior_answer.set((text or "").strip() or None)


def get_prior_answer() -> Optional[str]:
    """직전 어시스턴트 답변. 없으면 ``None``."""
    return _prior_answer.get()


def set_thread_context(messages: Optional[list]) -> None:
    """이번 턴에 함께 볼 스레드 대화를 설정한다(메신저 전용).

    Args:
        messages: ``[{"author", "text", "at"}, ...]`` 오래된 것부터. None 이면 비운다.
    """
    _thread_context.set(list(messages or []))


def get_thread_context() -> list:
    """이번 턴의 스레드 대화. 메신저 스레드 밖에서는 빈 목록."""
    try:
        return _thread_context.get()
    except LookupError:
        return []


def set_deal_context(block: Optional[str]) -> None:
    """이번 턴의 ``[거래 구조]`` 블록을 설정한다(SCMS 메신저 전용). None/빈 값이면 비운다."""
    _deal_context.set((block or "").strip() or None)


def get_deal_context() -> Optional[str]:
    """이번 턴의 ``[거래 구조]`` 블록. SCMS 메신저 밖에서는 None."""
    return _deal_context.get()


# 이번 턴의 사용자 발화 원문 — 도구가 모델 출력의 **근거**를 확인할 때 쓴다(예: 모델이 넘긴 회사 토큰이 발화에
# 더 긴 회사명의 접두로만 있으면 축소로 보고 버린다). 발화에서 값을 되찾는 파싱은 하지 않는다(2026-09-28).
_turn_input: ContextVar[Optional[str]] = ContextVar("turn_input", default=None)


def set_turn_input(text: Optional[str]) -> None:
    """이번 턴의 사용자 발화 원문을 설정한다(이미지·메모 주입 전의 raw)."""
    _turn_input.set(text)


def get_turn_input() -> str:
    """이번 턴의 사용자 발화 원문. 없으면 빈 문자열."""
    return _turn_input.get() or ""


# 이번 턴 식별자 — 도구가 "이 값을 사용자에게 물었다"를 기록할 때, 그 뒤에 사용자 턴이 실제로 있었는지 가르는 기준.
# 같은 턴 안에서 모델이 묻지 않고 같은 호출을 되풀이하면 '물었다'가 성립하지 않는다(2026-09-28 운영: 대상자 미수집
# 응답 뒤 같은 인자로 재호출 → '물었음'으로 통과해 대상자 없는 초안). 설정 안 된 경로(승인 재개 등)는 None.
_turn_id: ContextVar[Optional[str]] = ContextVar("turn_id", default=None)


def set_turn_id(turn_id: Optional[str]) -> None:
    """이번 턴 식별자를 설정한다(오케스트레이터가 턴 시작에 새로 만든다)."""
    _turn_id.set(turn_id)


def get_turn_id() -> Optional[str]:
    """이번 턴 식별자. 설정되지 않았으면 None."""
    return _turn_id.get()


# 최근 사용자 턴 원문(오래된 것부터, 이번 턴 제외) — 정정 턴은 참석자·결재자를 히스토리에서 다시 넘기므로 이번
# 발화만으론 근거가 없다. 부모(_preprocess_request)가 그래프 시작 전에 설정하고 자식 태스크는 읽기만 한다
# (ContextVar 는 자식의 set 이 부모에 전파되지 않는다 — reset_company_requests 참조). 승인 재개(resume) 경로는
# 설정하지 않으므로 그 턴엔 근거 검증이 자연히 비활성이다.
_recent_user_turns: ContextVar[Optional[list]] = ContextVar("recent_user_turns", default=None)


def set_recent_user_turns(texts: Optional[list]) -> None:
    """최근 사용자 턴 원문 목록을 설정한다(이번 턴 제외). None/빈 목록이면 근거는 이번 발화뿐."""
    _recent_user_turns.set([str(t) for t in (texts or []) if str(t or "").strip()])


def get_recent_user_turns() -> list:
    """최근 사용자 턴 원문 목록. 없으면 빈 목록."""
    return list(_recent_user_turns.get() or [])


def grounding_text() -> str:
    """모델 출력의 근거로 삼는 사용자 텍스트 = 이번 발화 ∪ 최근 사용자 턴. 없으면 빈 문자열(검증 생략)."""
    return "\n".join(t for t in [get_turn_input(), *get_recent_user_turns()] if t).strip()


# 이번 턴에 모델이 search_colleagues 로 "이름 + 회사" 를 조회한 기록 — {이름: {"company", "company_id", "known"}}.
# create 의 사람 확정이 이걸 읽어, 모델이 approval_line 에서 회사를 떨어뜨렸어도 같은 회사로 조회하고,
# 목록에 없는 회사를 말했으면 자사 동명이인으로 확정하지 않고 멈춘다.
_company_requests: ContextVar[Optional[dict]] = ContextVar("company_requests", default=None)


def reset_company_requests() -> None:
    """턴 시작(부모 태스크)에서 빈 기록 dict 를 만든다.

    ⚠️ 도구는 ``asyncio.gather`` 자식 태스크에서 실행되고 ContextVar 는 자식의 ``set()`` 을 부모에
    전파하지 않는다. 그래서 부모가 dict 를 먼저 만들어 두고 자식은 **제자리 수정**만 한다 — 같은
    객체를 참조하므로 다음 노드·다음 iteration 에서 보인다(2026-09-22 실측).
    """
    _company_requests.set({})


def note_company_request(name: str, company: str, company_id: Optional[str]) -> None:
    """이름에 대해 모델이 지정한 회사(원문)와 목록 매칭 결과(company_id, 없으면 None)를 기록한다."""
    key = str(name or "").strip()
    if not key:
        return
    current = _company_requests.get()
    if current is None:
        current = {}
        _company_requests.set(current)      # 부모가 안 만들었으면 이 태스크 안에서만 보인다(폴백)
    current[key] = {"company": str(company or "").strip(), "company_id": company_id, "known": bool(company_id)}


def get_company_requests() -> dict:
    """이번 턴의 이름 → 회사 지정 기록. 없으면 빈 dict."""
    return dict(_company_requests.get() or {})


def set_scms_order_enabled(flag: Optional[bool]) -> None:
    """이 요청에서 메신저 SCMS 발주요청을 쓸 수 있는가. 오케스트레이터가 턴 시작에 넣는다."""
    _scms_order.set(bool(flag))


def scms_order_enabled() -> bool:
    return _scms_order.get() is True


def set_withheld_tools(tools: Optional[dict]) -> None:
    """이번 턴에 막을 도구 ``{이름: 사유}``. None·빈 dict 면 해제."""
    _withheld_tools.set(dict(tools) if tools else None)


def withheld_tools() -> dict:
    return dict(_withheld_tools.get() or {})


def set_unattended_run(run: Optional[dict]) -> None:
    """예약 실행 표시 — ``{"schedule_id", "run_id"}``. None 이면 해제(사람이 보는 턴)."""
    _unattended_run.set(dict(run) if run else None)


def get_unattended_run() -> Optional[dict]:
    return _unattended_run.get()


def is_unattended() -> bool:
    return _unattended_run.get() is not None
