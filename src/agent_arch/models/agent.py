from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator

# MIME 타입별 최대 파일 크기 (raw bytes 기준)
#
# 전부 20MB 로 맞춘다.
#
# 이미지·PDF 는 파서를 거치지 않고 **멀티모달 LLM 에 바이트째로** 들어간다
# (app/llm/gemini.py 의 inline_data). 같은 경로로 같은 모델에 가는데 이미지 20MB /
# PDF 50MB 로 상한만 2.5배 달랐다 — 근거가 없는 비대칭이다.
#
# 게다가 inline 은 base64 라 요청 본문이 1.33배로 불어난다(50MB → 약 67MB). 프로바이더의
# 인라인 요청 한계를 넘기면 **업로드는 통과하고 모델 호출에서 깨진다** — 사용자에게는
# "올라갔는데 답을 못 함"으로 보이는 가장 나쁜 실패 모양이다. 업로드 단계에서 끊으면
# 이유가 분명한 400 이 나간다.
#
# ⚠️ 도면 레인(app/api/v1/drawing.py)은 별도로 50MB 를 쓴다("Gemini inline_data 한도 고려"
#    라고 적혀 있다). 두 값이 다른 채로 남아 있으니, 실제 한계를 확인하면 양쪽을 맞출 것.
_MAX_SIZE_BY_MIME: dict[str, int] = {
    "image/jpeg": 20 * 1024 * 1024,
    "image/png":  20 * 1024 * 1024,
    "image/webp": 20 * 1024 * 1024,
    "image/heic": 20 * 1024 * 1024,
    "image/heif": 20 * 1024 * 1024,
    "application/pdf": 20 * 1024 * 1024,
}
_MAX_FILES_PER_REQUEST = 5
_SUPPORTED_MIME_TYPES = set(_MAX_SIZE_BY_MIME.keys())

# 스프레드시트 MIME 타입별 최대 파일 크기
_MAX_SIZE_BY_SPREADSHEET_MIME: dict[str, int] = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": 10 * 1024 * 1024,  # xlsx
    "application/vnd.ms-excel": 10 * 1024 * 1024,                                            # xls
    "text/csv": 5 * 1024 * 1024,                                                              # csv
    "application/csv": 5 * 1024 * 1024,                                                       # csv (일부 OS)
}
_SPREADSHEET_MIME_TYPES = set(_MAX_SIZE_BY_SPREADSHEET_MIME.keys())
_MAX_SPREADSHEET_FILES_PER_REQUEST = 3

# 문서 MIME 타입별 최대 파일 크기 (워드·PPT·한글·HTML·텍스트·마크다운)
# pptx·hwp 가 20MB 인 것은 그 형식이 실제로 크기 때문이다 — 사내 발표자료는 이미지가
# 슬라이드마다 박혀 10MB 를 예사로 넘고, 한글 결과보고서도 그렇다. 상한을 docx 와 같이
# 두면 정작 읽어야 할 문서가 대부분 거부된다(실파일 확인).
# 이미지·PDF 와 달리 모델이 바이트째로 못 읽어 서버가 평문으로 풀어 질문에 붙인다
# (app/utils/document.py). 그래서 상한은 "모델이 읽을 분량"이 아니라 파싱 부하 기준이다
# — 실제로 모델에 실리는 양은 document._MAX_PARSE_CHARS 가 따로 제한한다.
_MAX_SIZE_BY_DOCUMENT_MIME: dict[str, int] = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": 10 * 1024 * 1024,  # docx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": 20 * 1024 * 1024,  # pptx
    "application/x-hwp": 20 * 1024 * 1024,                                                        # hwp
    "application/hwp+zip": 20 * 1024 * 1024,                                                      # hwpx
    "text/html": 5 * 1024 * 1024,                                                                 # html/htm
    "text/plain": 20 * 1024 * 1024,                                                               # txt
    "text/markdown": 20 * 1024 * 1024,                                                            # md
}
# txt·md 가 2MB 였던 것은 근거가 없다. 평문 2MB 면 한국어 기준 약 100만 자인데, 어차피
# `document._MAX_PARSE_CHARS`(30,000자)에서 잘려 모델에 실리는 양은 같다. 즉 상한을 짜게
# 둬도 지키는 게 없고, 로그 파일·덤프 같은 실제 업무 파일만 막혔다. 파싱 부하 기준으로
# 다른 문서 형식과 같은 20MB 로 맞춘다.
_DOCUMENT_MIME_TYPES = set(_MAX_SIZE_BY_DOCUMENT_MIME.keys())
_MAX_DOCUMENT_FILES_PER_REQUEST = 3


class ImageInput(BaseModel):
    """파일 입력 모델 (이미지 및 PDF).

    클라이언트가 전송하는 파일 데이터를 표현합니다.
    data는 순수 base64 문자열 또는 data URL 형식 모두 허용됩니다.

    Attributes:
        data: base64 인코딩된 파일 데이터 (data URL 형식 포함 허용)
            예: "data:image/jpeg;base64,/9j/4AAQ..." 또는 "/9j/4AAQ..."
        mime_type: 파일 MIME 타입 (기본값: "image/jpeg")
        file_name: 원본 파일명 (선택). **모델에게는 전달되지 않는다** — 프로바이더 파트는
            mime+data 뿐이라, 모델에게 물으면 그럴듯한 이름을 지어낸다(실측). 코드가 판독
            결과에 출처를 붙이는 용도로만 쓴다(도면 판독). 첨부 메타(`_save_attachments`)는
            best-effort 라 이미지와 개수·순서가 어긋날 수 있어 인덱스로 짝지을 수 없다.

    Raises:
        ValueError: 지원하지 않는 MIME 타입이거나 파일 크기가 제한을 초과하는 경우
    """

    data: str = Field(..., description="base64 인코딩된 파일 데이터 (data URL 형식 포함 허용)")
    mime_type: str = Field("image/jpeg", description="파일 MIME 타입")
    file_name: Optional[str] = Field(None, description="원본 파일명 (모델에 전달되지 않음)")

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        """MIME 타입이 지원되는 형식인지 검증한다.

        Args:
            v: 검증할 MIME 타입 문자열

        Returns:
            검증된 MIME 타입 문자열

        Raises:
            ValueError: 지원하지 않는 MIME 타입인 경우
        """
        if v not in _SUPPORTED_MIME_TYPES:
            raise ValueError(f"지원하지 않는 파일 타입입니다: {v}. 지원 타입: {_SUPPORTED_MIME_TYPES}")
        return v

    @model_validator(mode="after")
    def validate_data_size(self) -> "ImageInput":
        """파일 데이터 크기가 MIME 타입별 제한을 초과하지 않는지 검증한다.

        data URL 형식인 경우 base64 부분만 추출하여 크기를 계산한다.

        Returns:
            검증된 ImageInput 인스턴스

        Raises:
            ValueError: 파일 크기가 MIME 타입별 최대 크기를 초과하는 경우
        """
        raw = self.data.split(",", 1)[-1] if self.data.startswith("data:") else self.data
        max_bytes = _MAX_SIZE_BY_MIME.get(self.mime_type, 20 * 1024 * 1024)
        max_b64_len = max_bytes * 4 // 3
        if len(raw) > max_b64_len:
            size_mb = len(raw) * 3 // 4 / (1024 * 1024)
            max_mb = max_bytes // (1024 * 1024)
            raise ValueError(f"파일 크기가 너무 큽니다: {size_mb:.1f}MB (최대 {max_mb}MB)")
        return self


class ImageContext(BaseModel):
    """이미지 분석 결과 모델.

    Attributes:
        screen_description: 스크린샷에서 감지된 화면/기능 설명
        user_intent: 사용자가 하려는 것으로 추론된 작업
        suggested_query: 가이드 KB 검색에 적합한 한국어 쿼리
    """

    screen_description: str = Field(..., description="어떤 화면 또는 기능인지 설명")
    user_intent: str = Field(..., description="사용자가 하려는 것으로 추론된 작업")
    suggested_query: str = Field(..., description="가이드 KB 검색에 적합한 한국어 쿼리")


class AgentRequest(BaseModel):
    """에이전트 요청 모델.

    에이전트 실행에 필요한 모든 파라미터를 정의합니다.

    Attributes:
        input: 사용자 입력 텍스트
        images: 첨부 파일 목록 (최대 5개, 이미지·PDF 최대 20MB)
        session_id: 대화 컨텍스트 유지를 위한 세션 ID
        llm_provider: LLM 프로바이더 (gemini, openai, vllm)
        stream: 스트리밍 응답 활성화 여부 (기본값: False)
        tools: 사용할 특정 도구 목록
        answer_language: 답변 언어 (ko/en/zh/ja). 미전송 시 입력에서 감지
        metadata: 추가 메타데이터

    Raises:
        ValueError: 첨부 파일 개수가 5개를 초과하는 경우
    """

    input: str = Field(..., description="사용자 입력 텍스트")
    images: Optional[List[ImageInput]] = Field(None, description="첨부 파일 목록 (최대 5개, 이미지·PDF 최대 20MB)")
    session_id: Optional[str] = Field(None, description="대화 컨텍스트 유지를 위한 세션 ID")
    llm_provider: Optional[str] = Field(None, description="LLM 프로바이더 (gemini, openai, vllm)")
    stream: bool = Field(False, description="스트리밍 응답 활성화 여부")
    tools: Optional[List[str]] = Field(None, description="사용할 특정 도구 목록")
    answer_language: Optional[str] = Field(
        None,
        description=(
            "답변 언어 (ko/en/zh/ja). 미전송이면 사용자 입력에서 감지하고, "
            "판정이 안 되면 직전 발화 언어 → 서버 기본값(DEFAULT_ANSWER_LANGUAGE) 순으로 정한다. "
            "STT 인식 언어와는 별개다."
        ),
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")

    @field_validator("images")
    @classmethod
    def validate_images_count(cls, v: Optional[List[ImageInput]]) -> Optional[List[ImageInput]]:
        """첨부 파일 개수가 제한을 초과하지 않는지 검증한다.

        Args:
            v: 검증할 ImageInput 리스트

        Returns:
            검증된 ImageInput 리스트

        Raises:
            ValueError: 파일 개수가 최대 허용 개수를 초과하는 경우
        """
        if v is not None and len(v) > _MAX_FILES_PER_REQUEST:
            raise ValueError(f"파일은 요청당 최대 {_MAX_FILES_PER_REQUEST}개까지 첨부할 수 있습니다.")
        return v


class AgentResponse(BaseModel):
    """에이전트 응답 모델.

    에이전트 실행 결과를 표준화된 형식으로 반환합니다.

    Attributes:
        success: 작업 성공 여부
        output: 에이전트 생성 응답 텍스트
        job_id: 비동기 작업 ID
        llm_provider: 사용된 LLM 프로바이더
        tool_calls: 호출된 도구 정보
        error: 오류 메시지
        message_id: 저장된 메시지 UUID (rating 제출에 사용)
        metadata: 응답 메타데이터
    """

    success: bool = Field(description="작업 성공 여부")
    output: Optional[str] = Field(None, description="에이전트 생성 응답 텍스트")
    job_id: Optional[str] = Field(None, description="비동기 작업 ID")
    llm_provider: Optional[str] = Field(None, description="사용된 LLM 프로바이더")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="호출된 도구 정보")
    error: Optional[str] = Field(None, description="오류 메시지")
    message_id: Optional[str] = Field(None, description="저장된 메시지 UUID (rating 제출에 사용)")
    conversation_id: Optional[str] = Field(
        None,
        description=(
            "소속 대화방 UUID. 후속 요청에 동봉하면 같은 방에 이어 붙는다. "
            "프론트는 React state·Zustand 등 메모리에만 보관 (localStorage 금지)."
        ),
    )
    metadata: Optional[Dict[str, Any]] = Field(None, description="응답 메타데이터")


class GuestAgentRequest(BaseModel):
    """게스트(비로그인) 에이전트 요청 모델.

    Attributes:
        message: 사용자 입력 텍스트
        guest_id: 대화 연속성 유지를 위한 게스트 ID (없으면 서버 생성)
        conversation_id: 대화 ID (향후 확장용)
        llm_provider: LLM 프로바이더 (gemini, openai, vllm)
    """

    message: str = Field(..., description="사용자 입력 텍스트")
    guest_id: Optional[str] = Field(None, description="대화 연속성 유지를 위한 게스트 ID (없으면 서버 생성)")
    conversation_id: Optional[str] = Field(None, description="대화 ID (향후 확장용)")
    llm_provider: Optional[str] = Field(None, description="LLM 프로바이더 (gemini, openai, vllm)")


class GuestAgentResponse(BaseModel):
    """게스트 에이전트 응답 모델.

    Attributes:
        success: 작업 성공 여부
        output: LLM 응답 텍스트
        guest_id: 생성 또는 전달받은 게스트 ID
        conversation_id: 대화 ID
        error: 오류 메시지
        message_id: 저장된 메시지 UUID (rating 제출에 사용)
    """

    success: bool = Field(description="작업 성공 여부")
    output: Optional[str] = Field(None, description="LLM 응답 텍스트")
    guest_id: str = Field(description="생성 또는 전달받은 게스트 ID")
    conversation_id: Optional[str] = Field(None, description="대화 ID")
    error: Optional[str] = Field(None, description="오류 메시지")
    message_id: Optional[str] = Field(None, description="저장된 메시지 UUID (rating 제출에 사용)")


class RatingRequest(BaseModel):
    """Rating 제출 요청 모델.

    Attributes:
        rating: 사용자 평가 텍스트 (자유 형식)
    """

    rating: str = Field(..., description="사용자 평가 텍스트 (자유 형식)")


class InquirySummaryRequest(BaseModel):
    """문의하기 요약 요청 모델.

    사용자가 답변 하단 '문의하기'를 누르고 요약을 선택했을 때 온다. 대화 원문을 DB에서
    다시 읽지 않고 화면이 들고 있는 것을 그대로 받는다 — 중단됐거나 아직 저장되지 않은
    턴에서도 문의는 쓸 수 있어야 하기 때문이다(그런 턴엔 message_id가 없다).

    Attributes:
        question: 사용자 질문 원문
        answer: 그에 대한 Agent 답변 원문
        conversation_id: 추적용 대화방 ID (요약에는 쓰지 않는다)
        message_id: 추적용 답변 ID (없을 수 있음)
        llm_provider: 그 답변을 만든 provider. 회사 허용목록으로 다시 게이트된다.
    """

    question: Optional[str] = Field(None, description="사용자 질문 원문")
    answer: Optional[str] = Field(None, description="Agent 답변 원문")
    conversation_id: Optional[str] = Field(None, description="추적용 대화방 ID")
    message_id: Optional[str] = Field(None, description="추적용 답변 ID")
    llm_provider: Optional[str] = Field(None, description="답변을 만든 LLM provider")


class InquirySummaryResponse(BaseModel):
    """문의하기 요약 응답 모델.

    실패해도 4xx/5xx를 내지 않고 빈 문자열을 200으로 돌려준다 — 요약은 문의를 돕는
    보조 수단이지 전제 조건이 아니다. 화면은 빈 값을 받으면 요약 없이 문의를 연다.

    Attributes:
        title: 한 줄 제목(문의 목록에서 훑을 용도)
        summary: 무엇이 문제인지 2~3문장
    """

    title: str = Field("", description="한 줄 제목")
    summary: str = Field("", description="문제 요약 2~3문장")


class TranscribeResponse(BaseModel):
    """STT 전용 응답 모델.

    Attributes:
        success: 작업 성공 여부
        text: STT로 추출된 텍스트
        language: 인식에 사용한 언어 코드. 요청이 언어를 지정하지 않으면 ``"auto"``
            — 모델이 스스로 감지했다는 뜻이다(실사용이 한국어·영어·중국어라 기본값이
            자동 감지다. 근거는 settings.stt_language 주석). 감지된 실제 언어는
            서버가 알려주지 않으므로 여기에 담지 못한다.
    """

    success: bool = Field(description="작업 성공 여부")
    text: str = Field(description="STT로 추출된 텍스트")
    language: str = Field(description='인식에 사용한 언어 코드. 미지정 요청은 "auto"')


class AudioAgentResponse(AgentResponse):
    """오디오 STT + 에이전트 응답 모델.

    Attributes:
        transcription: STT로 추출된 원본 텍스트
    """

    transcription: str = Field(description="STT로 추출된 원본 텍스트")


class AgentStreamChunk(BaseModel):
    """스트리밍 청크 모델.

    Server-Sent Events(SSE)를 통한 실시간 응답 청크를 정의합니다.

    Attributes:
        type: 청크 타입 ('token', 'tool_call', 'error', 'done')
        content: 청크 내용
        data: 추가 데이터
    """

    type: str = Field(description="청크 타입 ('token', 'tool_call', 'chart', 'mermaid', 'error', 'done')")
    content: Optional[str] = Field(None, description="청크 내용")
    data: Optional[Dict[str, Any]] = Field(None, description="추가 데이터")


# orchestrator.stream() 이 답변 토큰 사이에 끼워 보내는 진행 이벤트 타입.
#
# 스트림 소비자(웹 SSE · 음성 SSE · Mattermost 드래프트)는 청크가 JSON 이면 이 목록과
# 대조해 "진행 이벤트"와 "답변 글자"를 가른다. **여기 없는 타입은 답변 본문으로 강등되어
# 사용자 화면에 JSON 이 그대로 찍힌다.** 그래서 소비자마다 사본을 두지 않고 이 한 곳에서만
# 정의한다 — 예전엔 세 벌이었고, 그중 음성 사본에 workspace.* 가 빠져 있었다. 새 진행
# 이벤트를 만들면 여기에만 더하면 되고, 소비자는 고칠 필요가 없다.
#
# "workspace.*": demo 모드 헤더 검출 시, 그리고 agent 모드에서 action 도구(view 동봉)가
# 실행됐을 때 open→update→complete 시퀀스로 발사된다(dmp-frontend 워크스페이스 패널 재활용).
# workspace.error 는 헤더 파싱 실패·LLM 예외 시 발사된다. Mattermost 봇은 workspace.update
# 에 실린 draft artifact 로 승인 버튼을 그린다.
#
# 여기 **없는** 것: run_started · stopped · transcription 은 orchestrator 가 아니라 API
# 핸들러가 직접 만들어 내보내므로 이 대조를 거치지 않는다. 목록에 넣을 필요가 없다.
PROGRESS_EVENT_TYPES: frozenset = frozenset({
    "tool_start", "tool_end", "searching", "thinking", "answering",
    "tools_found", "analyzing_image", "image_analyzed",
    "workspace.open", "workspace.step", "workspace.update",
    "workspace.complete", "workspace.error",
})


# ---------------------------------------------------------------------------
# 대화방(Conversation) 응답 모델
# ---------------------------------------------------------------------------


class ConversationListItem(BaseModel):
    """대화방 목록 응답 항목 (GET /me/conversation 의 items[i])."""

    conversation_id: str = Field(description="대화방 UUID")
    title: Optional[str] = Field(
        None,
        description=(
            "대화방 제목. 현 시점엔 항상 null (백엔드 자동 생성 안 함, 결정 #5). "
            "프론트는 null이면 created_at을 '대화 시작 시각' 라벨로 표시한다. "
            "자세한 컨벤션은 conversation_session_refactor_phase0.md §3.7 참조."
        ),
    )
    created_at: str = Field(description="생성 시각 ISO8601 UTC")
    last_message_at: Optional[str] = Field(
        None, description="마지막 turn 저장 시각. 빈 방이면 null"
    )
    message_count: int = Field(0, description="대화방 안 turn 개수 (Phase 1에서 집계로 채움)")
    rated_count: int = Field(
        0,
        description=(
            "평가된 turn 개수. ``positive_count + negative_count`` 와 항상 같다 "
            "(빈 문자열 rating 은 미평가로 본다 — classify_rating 과 같은 기준)."
        ),
    )
    positive_count: int = Field(0, description="'좋아요' turn 개수")
    negative_count: int = Field(
        0, description="'별로예요'(부정) turn 개수. 목록에서 피드백 배지로 쓴다."
    )


class ConversationListResponse(BaseModel):
    """대화방 목록 응답 (GET /me/conversation)."""

    items: List[ConversationListItem]
    next_cursor: Optional[str] = Field(None, description="다음 페이지 cursor (opaque)")
    has_more: bool


class ConversationDetail(BaseModel):
    """대화방 단건 메타 — GET /me/conversation/{id} 응답."""

    conversation_id: str = Field(description="대화방 UUID")
    title: Optional[str] = Field(None, description="대화방 제목. null이면 created_at 폴백 표시")
    created_at: str = Field(description="생성 시각 ISO8601 UTC")
    last_message_at: Optional[str] = Field(None, description="마지막 turn 시각")


# ---------------------------------------------------------------------------
# 대화 메시지(history viewer) 응답 모델
# ---------------------------------------------------------------------------
#
# 노출 정책: 다음 필드는 본 응답에 **절대 포함하지 않는다** — admin 라우터에서 별도 다룰 것:
#   - llm_provider, model_name
#   - prompt_tokens / completion_tokens / total_tokens
#   - llm_invoke_count
#   - tool_calls
#   - langsmith_run_id
#   - finish_reason
#   - session_id
#   - metadata
#
# "한 줄만 추가하면 되는데"로 끼워 넣지 말 것. 누출 차단은 본 모델 + repository SELECT
# 컬럼 + 응답 노출 테스트 3중 체크로 보호된다.


class ConversationMessage(BaseModel):
    """대화방 안 turn 1개.

    GET /me/conversation/{id}/messages 목록 항목과 단건 상세 응답이 동일 필드 집합을 공유한다.
    rating_polarity / rating_kind는 라우터 핸들러가 ``classify_rating()``으로 채워 내려보낸다.
    """

    message_id: str = Field(description="메시지 UUID")
    conversation_id: str = Field(description="소속 대화방 UUID")
    query: str = Field(description="사용자 입력")
    answer: str = Field(description="어시스턴트 응답")
    rating: Optional[str] = Field(
        None,
        description=(
            "사용자 평가 raw TEXT. preset 코드값(예: 'thumbs_up') 또는 "
            "자유 입력 텍스트가 그대로 들어옴. 미평가 시 null."
        ),
    )
    rating_polarity: Optional[str] = Field(
        None,
        description="rating을 컨벤션으로 분류한 폴라리티. 'positive' | 'negative' | null(미평가)",
    )
    rating_kind: Optional[str] = Field(
        None,
        description="rating 원천 종류. 'preset'(코드 카탈로그 매칭) | 'freeform'(자유 입력) | null",
    )
    response_time_sec: Optional[float] = Field(None, description="최종 답변까지 걸린 시간 (초)")
    status_code: Optional[int] = Field(None, description="실행 상태 코드 (200/500 등)")
    answer_length: Optional[int] = Field(None, description="답변 문자 수")
    created_at: str = Field(description="생성 시각 ISO8601 UTC")
    action_view: Optional[dict] = Field(
        None,
        description=(
            "이 턴에서 뜬 HITL 승인 카드 등 표시용 액션 뷰(제목/폼/표시 필드). 과거 대화 "
            "조회 시 카드 테이블을 읽기전용으로 복원하는 용도. 없으면 null. LLM 내부 신호 아님."
        ),
    )
    attachments: Optional[List[dict]] = Field(
        None,
        description=(
            "이 턴의 첨부파일 목록 [{file_id, filename, content_type, download_url}]. "
            "download_url 은 presigned URL — 조회 시점마다 새로 발급되며 유효시간이 "
            "있으므로 클라이언트는 저장하지 말고 즉시 렌더링에 사용할 것. "
            "file-service 미설정/발급 실패 시 download_url 없이 메타만 내려간다. 없으면 null."
        ),
    )


class ConversationMessageListResponse(BaseModel):
    """대화방 메시지 목록 응답 (GET /me/conversation/{id}/messages)."""

    items: List[ConversationMessage]
    next_cursor: Optional[str] = Field(None, description="다음 페이지 cursor (opaque)")
    has_more: bool


# ---------------------------------------------------------------------------
# 대화 검색(history search) 응답 모델 — GET /me/conversation/search
# ---------------------------------------------------------------------------
#
# 위 ConversationMessage와 동일한 노출 정책이 그대로 적용된다 — LLM 내부 필드는
# 이 응답에도 절대 포함하지 않는다. 차단은 본 모델 + repository SELECT 컬럼 +
# 응답 노출 테스트 3중 체크로 보호된다.


class ConversationSearchItem(BaseModel):
    """대화 검색 결과 항목 (GET /me/conversation/search 의 items[i]).

    대화방을 가로질러 매칭된 turn 1개. 프론트는 conversation_id로 해당 대화방으로
    점프하고, message_id로 그 turn 위치까지 스크롤할 수 있다.
    """

    message_id: str = Field(description="메시지 UUID")
    conversation_id: str = Field(description="이 turn이 속한 대화방 UUID (결과 클릭 시 이동 대상)")
    conversation_title: Optional[str] = Field(
        None,
        description="대화방 제목. null이면 프론트가 created_at을 라벨로 폴백 (목록 응답과 동일 컨벤션)",
    )
    query: str = Field(description="사용자 입력 (전문)")
    answer_snippet: str = Field(
        description=(
            "답변 발췌 — 검색어가 매칭된 부위를 중심으로 잘라낸 평문. 전문이 아니며, "
            "잘린 쪽에는 '…'가 붙는다. 하이라이팅은 클라이언트가 검색어로 처리한다. "
            "전문이 필요하면 메시지 단건 상세 endpoint를 쓸 것."
        )
    )
    created_at: str = Field(description="생성 시각 ISO8601 UTC")
    score: float = Field(
        description=(
            "관련도 (0~1). 정렬 기준이며 표시 필수는 아님. pg_trgm word_similarity 와 "
            "부분문자열 매칭 여부(질문 포함 0.65 / 답변 포함 0.35) 중 큰 값"
        )
    )


class ConversationSearchResponse(BaseModel):
    """대화 검색 응답 (GET /me/conversation/search).

    관련도 정렬이라 목록 endpoint의 opaque cursor를 쓸 수 없다 — limit/offset 방식.
    """

    items: List[ConversationSearchItem]
    total_hint: Optional[int] = Field(
        None,
        description="전체 매칭 건수 힌트. 현재는 항상 null (COUNT 미산출) — 필요해지면 채운다",
    )
    has_more: bool = Field(description="다음 offset에 결과가 더 있는지")


# ---------------------------------------------------------------------------
# 피드백 모아보기 응답 모델 — GET /me/conversation/feedback
# ---------------------------------------------------------------------------
#
# 위 모델들과 동일한 노출 정책이 그대로 적용된다 — LLM 내부 필드는 이 응답에도 절대
# 포함하지 않는다. 차단은 본 모델 + repository SELECT 컬럼 + 응답 노출 테스트 3중 체크.


class FeedbackItem(BaseModel):
    """피드백 모아보기 항목 (GET /me/conversation/feedback 의 items[i]).

    대화방을 가로질러 "내가 평가를 남긴 turn" 1개. 프론트는 conversation_id로 대화방으로
    점프하고 message_id로 그 turn 위치까지 스크롤한다.
    """

    message_id: str = Field(description="메시지 UUID")
    conversation_id: str = Field(description="이 turn이 속한 대화방 UUID (결과 클릭 시 이동 대상)")
    conversation_title: Optional[str] = Field(
        None, description="대화방 제목. null이면 프론트가 created_at을 라벨로 폴백"
    )
    query: str = Field(description="사용자 입력 (전문)")
    answer_snippet: str = Field(
        description=(
            "답변 발췌 — 앞부분 평문. 전문이 아니며 잘린 쪽에 '…'가 붙는다. "
            "전문이 필요하면 메시지 단건 상세 endpoint를 쓸 것."
        )
    )
    rating: Optional[str] = Field(
        None, description="사용자 평가 raw TEXT ('별로예요'는 자유 입력 사유가 그대로 들어온다)"
    )
    rating_polarity: Optional[str] = Field(
        None, description="'positive' | 'negative' | null(미평가)"
    )
    rating_kind: Optional[str] = Field(None, description="'preset' | 'freeform' | null")
    created_at: str = Field(description="생성 시각 ISO8601 UTC")


class FeedbackListResponse(BaseModel):
    """피드백 모아보기 응답 (GET /me/conversation/feedback).

    시간 정렬이라 목록 endpoint와 같은 opaque cursor 방식을 쓴다(검색과 달리 offset 아님).
    """

    items: List[FeedbackItem]
    next_cursor: Optional[str] = Field(None, description="다음 페이지 cursor (opaque)")
    has_more: bool



# ---------------------------------------------------------------------------
# LLM 사용량(usage) 응답 모델 — GET /me/usage
# ---------------------------------------------------------------------------
#
# 출처: LangSmith root run의 표준 토큰 필드 + metadata.user_id (DB 아님).
# cost는 LangSmith Model Pricing 설정 후 provider 단위로 확장 — 현재 미노출.


class ProviderUsage(BaseModel):
    """provider 단위 LLM 토큰 사용량 (UsageResponse.by_provider[i])."""

    provider: str = Field(
        description="LLM provider 식별자 (gemini/vllm/openai/ollama/openclaw/unknown)"
    )
    turns: int = Field(0, description="해당 provider로 실행된 root run(턴) 수")
    prompt_tokens: int = Field(0, description="입력 토큰 합계")
    completion_tokens: int = Field(0, description="출력 토큰 합계")
    total_tokens: int = Field(0, description="전체 토큰 합계")


class UsageResponse(BaseModel):
    """본인 LLM 사용량 집계 — GET /me/usage 응답 (LangSmith 출처)."""

    total_tokens: int = Field(0, description="전체 토큰 합계")
    prompt_tokens: int = Field(0, description="입력 토큰 합계")
    completion_tokens: int = Field(0, description="출력 토큰 합계")
    by_provider: List[ProviderUsage] = Field(
        default_factory=list, description="provider별 토큰 분해 (total_tokens 내림차순)"
    )


class CompanyUsage(BaseModel):
    """회사 1곳의 LLM 사용량 집계 (관측 화면 회사별 카드용).

    company는 LangSmith run metadata의 **한글 회사명**(예: '(주)회사')이 그대로 온다.
    Frappe Company.name 으로의 재매핑은 호출측(common_app)이 담당한다.
    """

    company: Optional[str] = Field(
        None, description="회사 식별자(한글 회사명). 회사 메타데이터가 없는 run은 null"
    )
    request_count: int = Field(0, description="기간 내 LLM 호출(root run) 수")
    input_tokens: int = Field(0, description="입력(prompt) 토큰 합")
    output_tokens: int = Field(0, description="출력(completion) 토큰 합")
    total_tokens: int = Field(0, description="총 토큰 합")
    cost_usd: Optional[float] = Field(
        None,
        description="비용(USD). 단가 등록된 run이 하나도 없으면 null. 일부만 등록 시 부분합.",
    )
    last_activity_at: Optional[str] = Field(
        None, description="해당 회사의 마지막 LLM 호출 시각 (KST ISO8601). 없으면 null"
    )


class CompanyUserProviderUsage(BaseModel):
    """회사 직원 1명의 provider별 토큰 사용량 (CompanyUserUsage.by_provider[i])."""

    provider: str = Field(
        description="LLM provider 식별자 (운영 3종 gemini/openai/vllm, 그 외 unknown)"
    )
    total_tokens: int = Field(0, description="해당 provider 토큰 합계")
    turns: int = Field(0, description="해당 provider로 실행된 root run(턴) 수")


class CompanyUserUsage(BaseModel):
    """회사 직원 1명의 LLM 사용량 집계 (CompanyUsageResponse.by_user[i])."""

    user: str = Field(
        description="직원 식별자(이메일/계정). 프론트 row key. 이메일 없으면 user_id(sub) 폴백"
    )
    name: Optional[str] = Field(
        None, description="표시용 이름(프로필 full_name). 없으면 프론트가 user로 폴백"
    )
    department: Optional[str] = Field(
        None, description="부서명(프로필 departmentName). 없으면 프론트가 '-' 표시"
    )
    total_tokens: int = Field(0, description="직원 전체 토큰 합계")
    prompt_tokens: int = Field(0, description="입력 토큰 합계")
    completion_tokens: int = Field(0, description="출력 토큰 합계")
    turns: int = Field(0, description="직원의 호출(root run) 횟수")
    by_provider: List[CompanyUserProviderUsage] = Field(
        default_factory=list, description="provider별 분해 (total_tokens 내림차순)"
    )


class CompanyUsageResponse(BaseModel):
    """회사 전체 · 직원별 LLM 사용량 집계 — GET /company/usage 응답 (agent.messages 출처).

    사용량이 없거나 DB 비활성화여도 4xx/5xx가 아니라 0 집계(빈 by_user)를 200으로
    내려준다 — DMP "직원 별 LLM 사용량" 탭이 거짓 에러 대신 빈 상태로 보이게 하기 위함.
    """

    total_tokens: int = Field(0, description="회사 전체 토큰 합계")
    prompt_tokens: int = Field(0, description="회사 전체 입력 토큰 합계")
    completion_tokens: int = Field(0, description="회사 전체 출력 토큰 합계")
    by_user: List[CompanyUserUsage] = Field(
        default_factory=list, description="직원별 사용량 (total_tokens 내림차순)"
    )
