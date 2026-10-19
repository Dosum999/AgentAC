from typing import Dict, Any, Optional
from pydantic import Field, validator, ConfigDict
from pydantic_settings import BaseSettings
import yaml
from pathlib import Path


class Settings(BaseSettings):
    """통합 설정 관리 클래스.

    .env 환경 변수와 YAML 설정 파일을 통합하여 애플리케이션 설정을 관리합니다.
    Pydantic BaseSettings를 상속하여 타입 검증과 환경 변수 로드를 자동화합니다.

    Attributes:
        app_name: 애플리케이션 이름
        app_version: 애플리케이션 버전
        debug: 디버그 모드 여부
        environment: 실행 환경 (development, production 등)
        api_v1_prefix: API v1 경로 접두사
        api_docs_enabled: API 문서(/docs, /redoc, /openapi.json, /swagger/) 공개 여부
        host: 서버 호스트 주소
        port: 서버 포트
        secret_key: JWT 서명 키
        redis_host: Redis 호스트
        redis_port: Redis 포트
        redis_db: Redis 데이터베이스 번호
        redis_password: Redis 비밀번호 (선택사항)
        redis_url: Redis 연결 URL
        qdrant_host: Qdrant 벡터 DB 호스트
        qdrant_port: Qdrant 벡터 DB 포트
        qdrant_url: Qdrant 연결 URL
        qdrant_collection_name: Qdrant 컬렉션 이름
        default_llm_provider: 기본 LLM 프로바이더
        gemini_api_key: Google Gemini API 키
        vllm_base_url: vLLM 서버 URL
        demo_mode: 데모 모드 여부 (True면 도구가 모의 데이터 반환)
        enable_pii_masking: PII 마스킹 활성화 여부
        max_iterations: 에이전트 최대 반복 횟수
        agent_timeout: 에이전트 타임아웃 (초)
        postgres_host: PostgreSQL 호스트
        postgres_port: PostgreSQL 포트
        postgres_db: PostgreSQL 데이터베이스 이름
        postgres_user: PostgreSQL 사용자
        postgres_password: PostgreSQL 비밀번호
        database_url: 데이터베이스 연결 URL
        mongodb_url: MongoDB 연결 URL
        mongodb_db: MongoDB 데이터베이스 이름
        keycloak_url: Keycloak 서버 URL
        keycloak_realm: Keycloak 영역 이름
        keycloak_client_id: Keycloak 클라이언트 ID
        keycloak_client_secret: Keycloak 클라이언트 비밀
        cors_origins: CORS 허용 출처
        log_level: 로그 레벨
        log_format: 로그 포맷
        llm_providers: LLM 프로바이더 설정 (YAML에서 로드)
        tools_config: 도구 설정 (YAML에서 로드)
        logging_config: 로깅 설정 (YAML에서 로드)
    """

    # Application 설정
    app_name: str = Field(default="Agent Agent Backend", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    debug: bool = Field(default=False, alias="DEBUG")
    environment: str = Field(default="development", alias="ENVIRONMENT")

    # API 설정
    api_v1_prefix: str = Field(default="/api/v1", alias="API_V1_PREFIX")
    # 운영은 false. 켜 두면 인증 없는 경로까지 포함한 API 목록 전체가 외부에 공개.
    # ENVIRONMENT 와 별개 설정 = 환경 이름과 문서 공개 여부를 각각 정할 수 있게
    api_docs_enabled: bool = Field(default=True, alias="API_DOCS_ENABLED")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # Security 설정
    secret_key: str = Field(alias="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, alias="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Redis 설정
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    redis_url: str = Field(alias="REDIS_URL")

    # Qdrant 벡터 DB 설정
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_url: str = Field(alias="QDRANT_URL")
    qdrant_collection_name: str = Field(default="agent_memory", alias="QDRANT_COLLECTION_NAME")

    # LLM 프로바이더 설정
    default_llm_provider: str = Field(default="openai", alias="DEFAULT_LLM_PROVIDER")
    guest_llm_provider: str = Field(default="openai", alias="GUEST_LLM_PROVIDER")
    # 이미지 분석(멀티모달) 전용 프로바이더. 주력 LLM 이 비전 미지원(vllm)일 때
    # 이미지 컨텍스트 추출에 사용한다. gemini·openai 가 비전 지원 프로바이더다.
    vision_llm_provider: str = Field(default="openai", alias="VISION_LLM_PROVIDER")
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    vllm_base_url: str = Field(default="http://localhost:8001", alias="VLLM_BASE_URL")
    # vLLM 서버를 `--api-key` 로 띄웠을 때 보내는 키(Authorization: Bearer). 비우면 더미값을 보낸다 —
    # 인증 없는 서버는 값을 보지 않는다. 비밀값이라 yaml 이 아니라 env(Secret)로만 받는다.
    vllm_api_key: Optional[str] = Field(default=None, alias="VLLM_API_KEY")

    # DMP 포털(dmp-frontend) 오리진. 답변에 사용자용 화면 링크를 넣을 때의 기준값이다
    # (예: 메신저 앱 다운로드 센터 = {포털}/main/download — SPA basename 이 /main).
    # 운영 apex 는 example.com, dev/sb 는 example.com 이므로 환경별로 덮어쓴다.
    dmp_portal_url: str = Field(default="https://example.com", alias="DMP_PORTAL_URL")

    # common_app(Frappe) 연동 — 회사별 LLM provider 허용목록 조회
    common_app_url: str = Field(default="https://dmp-common.example.com", alias="COMMON_APP_URL")
    frappe_api_key: Optional[str] = Field(default=None, alias="FRAPPE_API_KEY")
    frappe_api_secret: Optional[str] = Field(default=None, alias="FRAPPE_API_SECRET")
    # 회사별 허용목록 Redis 캐시 TTL(초). 핫패스 지연 최소화 + PA 변경 반영 지연의 균형.
    company_providers_cache_ttl: int = Field(default=120, alias="COMPANY_PROVIDERS_CACHE_TTL")
    # 용어 사전(플랫폼 매니저 > 시스템 > 용어 사전) 조회용 내부 키. common_app 의
    # INTERNAL_API_KEY 목록 중 하나여야 한다 — REST v1 은 사용자 Bearer 를 요구하고
    # 서비스에는 그게 없다. 비어 있으면 용어 사전 없이 동작한다(답변은 정상).
    common_app_internal_api_key: Optional[str] = Field(
        default=None, alias="COMMON_APP_INTERNAL_API_KEY"
    )
    # 용어 사전 Redis 캐시 TTL(초). PA 가 용어를 고치면 이만큼 뒤에 답변에 반영된다.
    # 플랫폼 매니저 화면이 저장 후 이 시간만큼 `반영 중` 을 표시한다(GLOSSARY_SYNC_SECONDS)
    # — 한쪽만 바꾸면 화면이 "반영됨"이라는데 실제로는 아직인 구간이 생긴다.
    glossary_cache_ttl: int = Field(default=60, alias="GLOSSARY_CACHE_TTL")
    # 회사별 provider 게이트 활성화 스위치. False면 기존 동작(요청 provider or 기본값) 유지.
    # common_app 정의 등록 + FRAPPE_API_KEY 설정 + 회사 허용목록 구성이 끝난 환경에서만 켤 것
    # (켜면 허용목록 미설정 회사는 vLLM으로 강등된다 — 안전 폴백이자 의도된 동작).
    enable_company_provider_gate: bool = Field(
        default=False, alias="ENABLE_COMPANY_PROVIDER_GATE"
    )
    # SCMS 채널(/scms/agent/*) provider 고정 — 패널이 gemini 를 실어 보내는데 그 키가 죽으면
    # (2026-09-09 실제: 선불 크레딧 소진 → 429) 채널이 통째로 폴백 문구만 낸다. MES 채널의
    # LOGISTICS_BOT_LLM_PROVIDER 와 같은 계약이되, 미설정이면 패널 값 그대로(기존 동작).
    scms_channel_llm_provider: Optional[str] = Field(default=None, alias="SCMS_CHANNEL_LLM_PROVIDER")
    # 관측 적재 푸시(대화 원본 → common_app). LangSmith 무료 플랜은 트레이스가 14일 보존 +
    # 월 5,000건 하드 캡이라, 한도를 넘긴 순간 관측 '회사 리뷰'와 'LLM 사용 대시보드'가 통째로
    # 빈다(2026-07-31 실제 발생). 그래서 두 화면의 기본 소스를 이 DB 푸시로 옮겼다 — 끄면 그
    # 구간이 다시 빈다. 동작 조건은 FRAPPE_API_KEY/SECRET 설정 — 없으면 푸시를 건너뛴다.
    enable_review_history_push: bool = Field(
        default=True, alias="ENABLE_REVIEW_HISTORY_PUSH"
    )
    review_history_push_interval_sec: int = Field(
        default=600, alias="REVIEW_HISTORY_PUSH_INTERVAL_SEC"
    )
    enable_usage_daily_push: bool = Field(
        default=True, alias="ENABLE_USAGE_DAILY_PUSH"
    )
    usage_daily_push_interval_sec: int = Field(
        default=600, alias="USAGE_DAILY_PUSH_INTERVAL_SEC"
    )
    # 벤더 호출 계측(agent.usage_meters, migration_v42). 끄면 기록만 멈추고 요청은 그대로 돈다.
    enable_usage_meters: bool = Field(default=True, alias="ENABLE_USAGE_METERS")
    # 이 KST 일자(포함)부터 일별 사용량 푸시의 **토큰**을 턴 원장(agent.messages)이 아니라 계측
    # 원장(agent.usage_meters)에서 센다. 비우면 기존대로 턴 토큰만 센다. 배포 당일은 계측이
    # 반나절뿐이라, **배포 다음 날**로 설정한다. 검색·STT 계측은 이 값과 무관하게 항상 실린다
    # (턴 원장에 없던 값이라 이중 계상이 없다).
    usage_meter_ledger_from: Optional[str] = Field(default=None, alias="USAGE_METER_LEDGER_FROM")
    # 푸시 행에 비토큰 계측 열(brave_search_calls·openai_search_calls·stt_seconds)을 싣는다.
    # **common_app 이 그 열을 받도록 배포된 뒤에만 켠다** — 수신 스키마가 extra="forbid" 라
    # 먼저 켜면 그 날 푸시가 통째로 거절되고 대시보드가 빈다.
    usage_meter_push_fields: bool = Field(default=False, alias="USAGE_METER_PUSH_FIELDS")
    # 푸시 행에 캐시 적중·쓰기 토큰(cached_input_tokens·cache_write_tokens, migration_v45)을
    # 싣는다. USAGE_METER_PUSH_FIELDS 와 따로 둔다 — 그 플래그는 이미 켜져 있어 재사용하면
    # 수신 쪽(common_app)보다 먼저 배포될 때 푸시가 통째로 거절된다. **common_app 운영 migrate
    # 완료(원장에 열 존재) 확인 뒤에만 켠다.**
    usage_cache_push_field: bool = Field(default=False, alias="USAGE_CACHE_PUSH_FIELD")
    # 시스템 오류 이벤트(agent.error_events, migration_v44) 수집 + common_app 푸시. 관측 「에이전트
    # 모니터링」 화면의 원인 축이다. 끄면 수집·푸시가 모두 멈추고(요청은 그대로), 화면 배지는
    # 수신이 끊겨 '끊김'으로 바뀐다. 푸시 인증은 COMMON_APP_INTERNAL_API_KEY 우선, 없으면
    # FRAPPE_API_KEY/SECRET(legacy whitelist) — 둘 다 없으면 수집만 하고 푸시를 건너뛴다.
    enable_error_events: bool = Field(default=True, alias="ENABLE_ERROR_EVENTS")
    # 예약 작업 — "평일 9시 40분에 환율 알려줘" 같은 반복 요청을 정해진 시각에 그 사람 자격으로 돌리고
    # 결과를 헤더 알림으로 보낸다(app/services/schedules). **기본 OFF** — 끄면 실행기가 안 뜨고
    # 라우터 카드·도구도 안 나간다. 켜기 전에 migration_v46 를 적용해야 한다.
    enable_scheduled_jobs: bool = Field(default=False, alias="ENABLE_SCHEDULED_JOBS")
    # 사용자당 활성 예약 수 상한(Gemini·Copilot 이 10). 주기는 하루 단위 이상만 받는다(rules.FREQS).
    schedule_max_active_per_user: int = Field(default=10, alias="SCHEDULE_MAX_ACTIVE_PER_USER")
    # 이만큼(분) 넘게 늦게 집힌 회차는 돌리지 않고 skipped 로 남긴다 — 환율처럼 그 시각의 값이 의미인
    # 작업을 서버 재기동 뒤에 몰아서 보내지 않는다(K8s startingDeadlineSeconds 와 같은 개념).
    schedule_start_grace_minutes: int = Field(default=30, alias="SCHEDULE_START_GRACE_MINUTES")
    # 연속 실패가 이 횟수에 닿으면 예약을 멈추고 알린다.
    schedule_max_consecutive_failures: int = Field(default=5, alias="SCHEDULE_MAX_CONSECUTIVE_FAILURES")
    # 헤더 알림 종류(common_app Employee Notification.type). common_app·@org/ui 에 이 값이 추가돼야
    # 받는다 — 그 전에는 생성 API 가 422 로 거절하고 실행 결과는 대화에만 남는다.
    schedule_notification_type: str = Field(default="agent", alias="SCHEDULE_NOTIFICATION_TYPE")
    # 알림을 눌렀을 때 열 주소. {conversation_id} 자리표시자를 채운다. 비우면 링크 없이 알린다.
    schedule_result_url_template: str = Field(default="", alias="SCHEDULE_RESULT_URL_TEMPLATE")
    error_events_push_interval_sec: int = Field(
        default=60, alias="ERROR_EVENTS_PUSH_INTERVAL_SEC"
    )
    # 푸시 payload 의 instance(≤64). 비우면 socket.gethostname() — 컨테이너면 컨테이너 id 라
    # 재배포마다 바뀌므로 운영은 이름을 박아 두는 편이 화면 신선도 표가 읽기 쉽다.
    agent_instance: Optional[str] = Field(default=None, alias="AGENT_INSTANCE")

    # 에이전트 설정
    demo_mode: bool = Field(default=True, alias="DEMO_MODE")
    # 통합 도구 인덱싱 — 정적 스킬·MCP 도구 설명을 동적 API와 같은 Qdrant
    # 컬렉션에 인덱싱해 retrieved 티어(질의 검색 노출)를 활성화한다.
    # False면 exposure=retrieved 도구도 core로 승격된다 (도구 소실 방지).
    unified_tool_indexing: bool = Field(default=False, alias="UNIFIED_TOOL_INDEXING")
    agent_mode: str = Field(default="default", alias="AGENT_MODE")
    # demo 모드 분류 선행 호출(F2 call 1) 최대 토큰 — 상한일 뿐 실제 출력은 헤더 한 줄.
    # gemini 등 thinking 모델이 헤더 앞에 추론을 흘리는 경우 256 이면 헤더 도달 전
    # 잘려 분류가 실패한다. 추론 분량을 흡수할 여유를 둔다 (헤더만 내면 즉시 stop).
    demo_classify_max_tokens: int = Field(default=1024, alias="DEMO_CLASSIFY_MAX_TOKENS")
    # gemini thinking_budget override — 도구 없는 추론 전용 호출에만 켠다(도구 체인 무관).
    # 0=off, -1=dynamic(모델 자율), 양수=토큰 상한. gemini 외 프로바이더는 이 kwarg를 무시.
    # 캘린더 분류(날짜/슬롯 추출) 정확도용. demo_classify_max_tokens=1024 가 추론 분량 흡수.
    demo_classify_thinking_budget: int = Field(default=-1, alias="DEMO_CLASSIFY_THINKING_BUDGET")
    # 답변 언어 가드 — 최종 답변이 **턴의 목표 언어**가 아니면 도구 없이 그 언어로 1회
    # 재생성한다. 프롬프트의 답변 언어 지시를 gemini 가 다른 언어 입력에서 미러링하며
    # 무시하는 경우의 결정적 보강. 목표 언어는 요청 필드(language) 또는 입력 감지로 정한다
    # (app/utils/language.py). 한국어 전용이던 시절의 이름·env 키를 유지하는 것은 배포된
    # .env 를 조용히 깨지 않기 위함이다. off 로 즉시 비활성 가능.
    korean_answer_guard: bool = Field(default=True, alias="KOREAN_ANSWER_GUARD")
    # 언어를 못 정했을 때의 기본 답변 언어(ko/en/zh/ja). 요청이 language 를 안 보내고
    # 입력에서도 판정이 안 될 때(짧은 확인 발화 등)만 쓰인다.
    default_answer_language: str = Field(default="ko", alias="DEFAULT_ANSWER_LANGUAGE")
    # 사람 이름 환각 가드 — 답변에 이 턴 입력 어디에도 없는 "○○○ 님" 이 있으면 1회 재생성,
    # 그래도 남으면 마스킹. 실측 사고: 도구가 영문 ID 만 주자 모델이 음차해 jdoe→"김민수"
    # (실제 김민서), lho→"이호"(실제 홍길동) 로 답했다. off 로 즉시 비활성 가능.
    name_answer_guard: bool = Field(default=True, alias="NAME_ANSWER_GUARD")
    enable_pii_masking: bool = Field(default=False, alias="ENABLE_PII_MASKING")
    # 외부 LLM 가명화 래퍼(app/llm/pii) — 외부 provider 호출 직전에 이름·이메일·전화 등을
    # 자리표시 토큰으로 바꾸고 응답에서 복원한다(위의 별표 마스킹과 달리 가역). off 는 래퍼를
    # 씌우지 않음, shadow 는 계산·기록만 하고 원문 전송, enforce 는 실제 치환. 회사별 opt-in
    # (기본 비활성화)이라 모드가 켜져도 켠 회사만 적용된다. 가드가 켜진 턴은 별표 마스킹을
    # 건너뛴다(먼저 돌면 복원할 원문이 없다).
    pii_guard_mode: str = Field(default="off", alias="PII_GUARD_MODE")
    # 래퍼를 씌울 provider(쉼표). vllm 은 사내 전제로 기본 제외 — 임대 GPU 를 외부로 보면 추가.
    pii_guard_providers: str = Field(default="gemini,openai,anthropic", alias="PII_GUARD_PROVIDERS")
    # 토큰 HMAC 비밀. 비우면 SECRET_KEY 를 쓴다. 대화 ID 만으로 키를 만들면 역산 가능하므로 서버 비밀 필수.
    pii_guard_secret: str = Field(default="", alias="PII_GUARD_SECRET")
    # 차단(복원 없음)할 종류. 계좌·카드는 송장·주문번호 오탐을 shadow 로 본 뒤 추가한다.
    pii_guard_block_kinds: str = Field(default="SSN,APIKEY,PASSWORD", alias="PII_GUARD_BLOCK_KINDS")
    # 비어 있지 않으면 이 회사 목록이 회사 스위치를 대신한다(운영 강제·로컬 테스트). 한글 회사명, 쉼표.
    pii_guard_force_companies: str = Field(default="", alias="PII_GUARD_FORCE_COMPANIES")
    # 개발 전용: shadow 샘플에 원문 값을 함께 남긴다. 운영(ENVIRONMENT=production)에선 무시된다.
    pii_guard_shadow_reveal: bool = Field(default=False, alias="PII_GUARD_SHADOW_REVEAL")
    # 임직원 디렉터리 어휘의 요청자별 Redis 캐시 TTL(초). 입·퇴사는 만료 후 반영.
    pii_guard_lexicon_ttl: int = Field(default=3600, alias="PII_GUARD_LEXICON_TTL")
    # 회사 사번 형식(정규식). 디렉터리엔 사번 필드가 없어 이것과 도구 결과 수확으로만 잡는다. 예: E-\d{4}
    pii_guard_empno_regex: str = Field(default="", alias="PII_GUARD_EMPNO_REGEX")
    max_iterations: int = Field(default=3, alias="MAX_ITERATIONS")
    enable_parallel_eval: bool = Field(default=False, alias="ENABLE_PARALLEL_EVAL")
    qdrant_top_k: int = Field(default=12, alias="QDRANT_TOP_K")
    agent_timeout: int = Field(default=300, alias="AGENT_TIMEOUT")
    # HITL(Human-In-The-Loop) — 비가역 액션(메일/메신저/일정/휴가)을 LLM 이 아니라
    # 코드가 강제로 멈추고(confirm=False 초안 강제 + interrupt 체크포인트) 사람 승인
    # 후 재개한다. False면 기존 경로(프롬프트 유도 + workspace POST) 그대로.
    use_langgraph_hitl: bool = Field(default=False, alias="USE_LANGGRAPH_HITL")

    # 장기 사용자 메모리(NapMem Phase 1) — 턴 종료 시 비동기로 memory record를 추출·
    # reconcile·인덱싱한다(agent.memory_records + Qdrant). False(기본)면 추출/저장을
    # 전혀 수행하지 않는다(스케줄 진입점에서 즉시 no-op). 켜기 전 migration_v24 적용 필요.
    enable_user_memory: bool = Field(default=False, alias="ENABLE_USER_MEMORY")

    # 장기 사용자 메모리 v2(4층 배치 파이프라인, app/services/memory_v2) 전환 모드.
    #   off    : v2 코드 경로 전부 비활성(기본). v1 그대로.
    #   shadow : v2 가 캡처·추출·저장을 병행하되 회상·도구는 v1. 비교 검증 단계.
    #   active : 회상·도구가 v2 를 보고 v1 추출은 멈춘다.
    # ENABLE_USER_MEMORY 가 마스터 스위치다(False 면 이 값과 무관하게 전부 no-op).
    # 켜기 전 migration_v35 적용 필요.
    memory_v2_mode: str = Field(default="off", alias="MEMORY_V2_MODE")

    # 기억의 "자리"(장소) 스코프 — 회상 순위 배수와 상시 블록 자격을 자리로 가른다.
    #   자리 = MES 앱 창(agent:mes — /agent/mes·/audio) · 메신저 채널(mattermost:<channel_id>).
    #          브라우저 패널과 DM 은 자리가 없다(중립 — 어디서나 그대로 적용).
    #   True : 그 자리에서 못 박은 규칙은 그 자리에서만 매 턴 주입되고, 회상 순위도 같은
    #          자리를 올린다(다른 자리는 살짝 내린다). MES 앱 창 분리와 메신저 채널 분리가
    #          함께 켜진다.
    #   False(기본): 회상은 이 기능 이전과 동일하다.
    # 쓰기(턴·기억에 자리 스탬프)는 이 값과 무관하게 항상 돈다 — 데이터가 쌓여 있어야 켠
    # 다음에 볼 것이 생긴다. ENABLE_USER_MEMORY·MEMORY_V2_MODE=active 가 선행 조건이다.
    # 이력 폴백(Redis 가 비면 Postgres 에서 되살림). 공용 핫패스라 배포 없이 끌 수 있게 둔다.
    enable_history_hydration: bool = Field(default=True, alias="ENABLE_HISTORY_HYDRATION")
    memory_v2_place_scope: bool = Field(default=False, alias="MEMORY_V2_PLACE_SCOPE")

    # 메모리 v2 턴별 L1 회상(관련 기억을 마지막 user 메시지에 주입). False 면 안정 블록(페르소나·장면·
    # 전역 지시)과 별칭 어휘 매칭만 주입하고 세부 기억은 에이전트가 memory_search 로 찾는다 —
    # 핫 패스의 임베딩·Qdrant 호출을 없애는 대신 회상을 도구에 맡기는 A/B 용.
    memory_v2_turn_recall: bool = Field(default=True, alias="MEMORY_V2_TURN_RECALL")

    # 기억 → 흐름 힌트: 이 턴에 매칭된 별칭이 도메인을 못 박으면 그 도메인 도구를 탐색 없이 미리
    # 바인딩하고(모놀리식·워커 공통), 직전 도메인이 없을 때 라우터에 그 도메인을 힌트로 준다.
    # 강제가 아니라 로드아웃이다. v2 active 일 때만 의미 있다.
    memory_tool_hint: bool = Field(default=True, alias="MEMORY_TOOL_HINT")

    # LangGraph 상태 영속(checkpointer) 백엔드 — 단일 주입 지점.
    #   memory  : MemorySaver (기본). in-process·단일 워커 한정, 재배포 시 미승인 초안 소실.
    #   postgres: AsyncPostgresSaver (멀티워커·재배포 생존). langgraph-checkpoint-postgres 필요.
    # HITL 및 향후 모든 그래프가 app.agents.graphs.checkpointer.get_checkpointer() 로 참조한다.
    langgraph_checkpointer: str = Field(default="memory", alias="LANGGRAPH_CHECKPOINTER")

    # ReAct 루프를 LangGraph StateGraph(app/agents/graphs/react_graph.py)로 실행한다.
    # False 면 기존 react_loop_events 경로. 두 경로는 동일 LoopResult·동일 스트리밍
    # 계약(패리티 테스트로 검증). 전환은 orchestrator._run_react/_stream_react/
    # _run_react_continuation 세 지점에서만 일어난다.
    #
    # **기본값이 True 인 이유**: 답변 가드(이름 환각·한국어·환각 액션)는 그래프 경로에만
    # 있다. loop.py 에는 셋 다 없다. 운영 `.env` 는 진작 true 로 켜 놨는데 코드 기본값만
    # False 로 남아 있어서, `.env` 줄 하나가 빠진 환경은 **가드가 통째로 없는 채로** 돈다.
    # 기본값을 배포 현실에 맞춘다. 테스트는 autouse 픽스처가 OFF 를 강제하므로 영향 없다.
    use_langgraph_react: bool = Field(default=True, alias="USE_LANGGRAPH_REACT")

    # 멀티 오케스트레이션(supervisor) — 라우터가 도메인 coarse 선택 → 도메인 스코프 워커
    # (react 그래프) 병렬 실행 → synthesizer 합성. False(기본)면 supervisor 미사용.
    # 우선순위: supervisor ON → supervisor / (OFF) use_langgraph_react → 그래프 / else 루프.
    use_langgraph_supervisor: bool = Field(default=False, alias="USE_LANGGRAPH_SUPERVISOR")
    # 라우터 전용 LLM 프로바이더(과부하 메인과 분리 권장, 예: vllm). None이면 메인과 동일
    # 프로바이더의 fresh(언바운드) 인스턴스 사용.
    router_llm_provider: Optional[str] = Field(default=None, alias="ROUTER_LLM_PROVIDER")
    # 라우터(도메인/서브쿼리 분류) gemini thinking_budget. 도구 없는 호출이라 체인 무관.
    # 0=off, -1=dynamic, 양수=상한. parse_route 가 JSON 을 본문 어디서든 정규식으로 찾아
    # thinking 프리앰블이 있어도 파싱 안전. gemini 외 프로바이더는 무시.
    router_thinking_budget: int = Field(default=-1, alias="ROUTER_THINKING_BUDGET")
    # 라우팅 신뢰도 임계값 — 미만이면 무스코프 단일 react 로 fallback.
    router_confidence_threshold: float = Field(default=0.6, alias="ROUTER_CONFIDENCE_THRESHOLD")
    # 후속 질의에서 라우터 판단이 직전 도메인 상속을 이기는 확신도(supervisor._decide). 9/15 오분류 0.70 · 9/29 web 0.95~1.00 실측 사이.
    router_followup_override_confidence: float = Field(default=0.85, alias="ROUTER_FOLLOWUP_OVERRIDE_CONFIDENCE")
    # 답변 검증(react/verify): off | shadow(판정만 기록) | on(보강 루프). 그림자 실측 뒤 켠다.
    answer_verify_mode: str = Field(default="off", alias="ANSWER_VERIFY_MODE")
    answer_verify_max_rounds: int = Field(default=1, alias="ANSWER_VERIFY_MAX_ROUNDS")
    # 워커 입력 원문 보존(A안) — _scope_messages 가 마지막 user content 를 라우터 서브쿼리로
    # 교체하는 대신, 원문을 유지하고 "[담당 지시]"(서브쿼리=담당 범위 라벨)만 덧붙인다.
    # 라우터 재작성 유실 계급(이미지 컨텍스트 4507889 · 주차 표현 4b31ed7 · 거절 초안
    # 0a38065)을 구조적으로 차단한다. 도구 검색(resolve_tools_fn)은 계속 서브쿼리를 쓴다
    # (임베딩 검색엔 짧은 질의가 유리 — 유실이 검색 단계엔 무해하다는 비대칭 활용).
    # False(기본)면 기존 동작(교체 + time/img/reject 노트 재부착). 상세:
    # reports/supervisor_subquery_replacement_analysis.md. 검증 후 기본 ON 전환 예정.
    supervisor_preserve_original: bool = Field(
        default=False, alias="SUPERVISOR_PRESERVE_ORIGINAL"
    )

    # Guide KB 검색 파이프라인
    # — Abstain gate: "KB에 답 없는 쿼리"를 retrieval 단계에서 걸러 hallucination 차단
    guide_kb_enable_abstain: bool = Field(
        default=True, alias="GUIDE_KB_ENABLE_ABSTAIN"
    )
    # top_score 이 값 이하 → 즉시 abstain (LLM 호출 없음)
    guide_kb_abstain_hard_threshold: float = Field(
        default=0.65, alias="GUIDE_KB_ABSTAIN_HARD_THRESHOLD"
    )
    # top_score 이 값 이상 → 즉시 pass (LLM 호출 없음)
    guide_kb_abstain_soft_threshold: float = Field(
        default=0.73, alias="GUIDE_KB_ABSTAIN_SOFT_THRESHOLD"
    )
    # 경계 구간 [hard, soft) 에서 호출하는 LLM judge provider
    # 비우면 DEFAULT_LLM_PROVIDER 를 따른다. 특정 프로바이더로 고정하고 싶을 때만 채운다.
    guide_kb_abstain_judge_provider: str = Field(
        default="", alias="GUIDE_KB_ABSTAIN_JUDGE_PROVIDER"
    )
    # 서비스 도메인 설명 (LLM judge 프롬프트에 주입됨)
    guide_kb_domain_desc: str = Field(
        default=(
            "Agent 사용자 가이드 지식베이스. HR 포털, DMP, 메신저, 메일, 캘린더, 드라이브 등의 "
            "사내 시스템 사용법·절차·설정 방법을 다룬다."
        ),
        alias="GUIDE_KB_DOMAIN_DESC",
    )
    # — Group-by 검색: 카테고리(서비스)별 다양성 보장. "문서 업로드" 같은 모호한
    #   질문이 한 카테고리로 쏠리는 문제를 해결한다.
    # 각 카테고리 그룹당 가져올 청크 수
    guide_kb_group_size: int = Field(default=2, alias="GUIDE_KB_GROUP_SIZE")
    # 검색이 반환할 최대 그룹(=카테고리) 수
    guide_kb_group_limit: int = Field(default=6, alias="GUIDE_KB_GROUP_LIMIT")
    # 단일 카테고리 분기 판정에 필요한 top1-top2 점수 차이
    guide_kb_dominance_threshold: float = Field(
        default=0.05, alias="GUIDE_KB_DOMINANCE_THRESHOLD"
    )
    # 2위 이하 그룹이 multi 분기에 포함되기 위한 최소 top_score
    guide_kb_runner_up_floor: float = Field(
        default=0.60, alias="GUIDE_KB_RUNNER_UP_FLOOR"
    )
    # — 원문 폴백: 가이드 도구가 모델이 넘긴(라우터가 재작성한) 검색어로 검색한 top 이 soft
    #   임계값(0.73) 미만이면 **사용자 발화 원문**(get_turn_input)으로 한 번 더 검색해 합친다.
    #   재작성에서 "접근 권한이 없습니다" 같은 핵심 표현이 빠져 정답 페이지가 8슬롯 밖으로
    #   밀리는 드리프트 방어(2026-10-13 로컬 실측: 재작성 20건 중 놓친 1건 top 0.709, 폴백 대상
    #   4/20). 비용 = 폴백 때만 임베딩 1회 추가(직렬이라 그 요청만 ~0.4s), LLM 슬롯 수는 그대로.
    #   원문이 검색어와 같거나 비어 있으면 추가 검색 없음.
    #   기본 OFF — 운영 기본값은 abstain(judge) 실평가 뒤에 결정(담당자 2026-10-13). dev 는
    #   GUIDE_KB_ORIGINAL_FALLBACK=true 로 켜서 잰다.
    guide_kb_original_fallback: bool = Field(
        default=False, alias="GUIDE_KB_ORIGINAL_FALLBACK"
    )
    # 이 길이(문자)를 넘는 발화는 병합하지 않는다 — 긴 복합 요청은 임베딩이 뭉개져 잡음만 보탠다.
    guide_kb_original_fallback_max_chars: int = Field(
        default=500, alias="GUIDE_KB_ORIGINAL_FALLBACK_MAX_CHARS"
    )

    # LangSmith 트레이싱 설정
    enable_langsmith: bool = Field(default=False, alias="ENABLE_LANGSMITH")
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="agent-backend", alias="LANGSMITH_PROJECT")
    langsmith_dataset_name: str = Field(default="agent-feedback", alias="LANGSMITH_DATASET_NAME")

    # 외부 서비스 Base URL (API Gateway 경유)
    hr_service_base_url: str = Field(default="http://hr-service", alias="HR_SERVICE_BASE_URL")
    logistics_service_base_url: str = Field(default="http://logistics-service", alias="LOGISTICS_SERVICE_BASE_URL")
    logistics_tenant_id: str = Field(default="%ED%98%84%ED%98%B8%EA%B8%88%EC%86%8D", alias="LOGISTICS_TENANT_ID")
    # Drive(Nextcloud) API Gateway base — `/api/drive` prefix 는 client 가 자동 부착.
    # 빈 문자열로 명시 override 하면 demo fetcher 가 mock 폴백.
    drive_service_base_url: str = Field(default="https://api.example.com", alias="DRIVE_SERVICE_BASE_URL")
    # SCMS API Gateway base — `/api/scms` prefix 는 client 가 자동 부착.
    # 빈 문자열로 명시 override 하면 demo fetcher 가 mock 폴백.
    scms_service_base_url: str = Field(default="https://api.example.com", alias="SCMS_SERVICE_BASE_URL")
    # SCMS 백엔드 직결 오리진 — 메신저 발주요청 초안(/api/v1/order-drafts) 전용.
    # 게이트웨이 /api/scms 프록시(위 scms_service_base_url)는 Keycloak scms 클라이언트에 token
    # exchange 권한이 없어 400 이 난다(scms-mcp 실측). SCMS 자신처럼 요청자 dmp_sid 를
    # X-DMP-Session-ID 로 실어 백엔드에 직결한다.
    scms_api_base_url: str = Field(default="https://procurement.example.com", alias="SCMS_API_BASE_URL")
    # 메신저에서 SCMS 발주요청을 쓸 회사(CSV, 한글 회사명). **비우면 꺼짐** — 발주 도구를 SCMS 를
    # 안 쓰는 회사 채널에 들려 줄 이유가 없다. 켜려면 회사를 적는다(예: CustomerA 의 회사명).
    scms_messenger_companies: str = Field(default="", alias="SCMS_MESSENGER_COMPANIES")
    # SCMS 메신저 거래 구조 전체 스위치(봇 훅: 채널 최근 글·글쓴이 소속·[거래 구조] 블록). **기본 꺼짐** — dev 는 언제든
    # 운영에 나갈 수 있어야 해서(AGENTS.md §3) 검증 전 기능은 꺼 두고, 개발기에서만 SCMS_DEAL_CONTEXT_ENABLED=true 로 켠다.
    # 끄면 SCMS 메신저는 이 기능이 없던 때와 같다(추출·외부 LLM 호출·첫 멘션 대기 없음).
    scms_deal_context_enabled: bool = Field(default=False, alias="SCMS_DEAL_CONTEXT_ENABLED")
    # SCMS 메신저 거래 구조 — 대화에서 당사자를 읽는 추출기(LLM 1회/멘션). 끄면 SCMS 주문·글쓴이 소속만 쓴다.
    # 프로바이더를 비우면 그 턴 채널 프로바이더(와 그 기본 모델)를 따른다(app/services/scms/deal/extract.py).
    # 기본 = openai gpt-6.1-sol(사용자 결정 2026-10-15, luna 에서 변경). stress_v1 98건 실측: sol 92/98·luna 92/98 로
    # 정확도 동률, 지연 p50 sol 6.3초 vs luna 13.7초, 출력 토큰 sol 366 vs luna 1194. 정답셋 sol 15/15 4회.
    scms_deal_extract_enabled: bool = Field(default=True, alias="SCMS_DEAL_EXTRACT_ENABLED")
    scms_deal_extract_provider: str = Field(default="openai", alias="SCMS_DEAL_EXTRACT_PROVIDER")
    # 추출기 모델(비우면 그 프로바이더의 기본 모델).
    scms_deal_extract_model: str = Field(default="gpt-6.1-sol", alias="SCMS_DEAL_EXTRACT_MODEL")
    # 추출기 시간 제한(초). 넘기면 이번 턴은 추론 없이 간다. 추론형 모델은 길다(sol 17~19초, luna 20~30초/회 실측).
    scms_deal_extract_timeout: float = Field(default=45.0, alias="SCMS_DEAL_EXTRACT_TIMEOUT")
    # 출력 상한. 추론형 모델은 추론 토큰이 같은 상한을 쓴다 — 2000 이면 luna 가 추론에 다 써 본문이 빈다(실측).
    scms_deal_extract_max_tokens: int = Field(default=8000, alias="SCMS_DEAL_EXTRACT_MAX_TOKENS")
    # 추론 강도(OpenAI reasoning_effort: low/medium/high). 비우면 싣지 않는다(모델 기본). luna 실측: 기본 15/15,
    # medium 14/15, low 13~14/15(대신 절반 시간).
    scms_deal_extract_reasoning: str = Field(default="", alias="SCMS_DEAL_EXTRACT_REASONING")
    # 누락 점검(gleaning) — 거래 글에 이름이 나왔는데 관계에 안 든 회사가 있을 때만 1회 더 묻는다.
    scms_deal_extract_glean: bool = Field(default=True, alias="SCMS_DEAL_EXTRACT_GLEAN")
    # 지연 개선 — 추출(LLM·SCMS 조회)을 답변과 따로 돌린다. 턴 블록은 저장된 이해로 바로 만들고, 추출 결과는
    # 다음 턴부터 반영된다. 예산(초) 안에 추출이 끝나면 이번 턴에도 반영한다. 끄면(false) 종전처럼 추출을 기다린다.
    # 예산 기본 0: gpt-6.1-sol 추출은 실채널에서 18~32초(p50 7초)라 2초 예산에 들어온 적이 없다 — 기다림만 붙었다.
    # 빠른 모델(gemini 2~4초)로 바꾸면 2~4 로 올려 이번 턴에도 반영되게 할 수 있다.
    scms_deal_async: bool = Field(default=True, alias="SCMS_DEAL_ASYNC")
    scms_deal_sync_budget: float = Field(default=0.0, alias="SCMS_DEAL_SYNC_BUDGET")
    # 채널을 처음 읽는 턴은 추출을 끝까지 기다린다 — 저장된 이해가 없어 빈 블록으로 답하게 되기 때문이다.
    # 그 채널의 두 번째 멘션부터는 비동기로 돈다. 끄면(false) 첫 턴도 빈 이해로 바로 답한다.
    scms_deal_cold_sync: bool = Field(default=True, alias="SCMS_DEAL_COLD_SYNC")
    # 기다리는 턴(처음 읽는 채널·ASYNC=false)의 상한(초). 실측 18~33초 + 여유. 넘으면 답은 먼저 나가고 갱신은 뒤에서 마친다.
    scms_deal_cold_budget: float = Field(default=60.0, alias="SCMS_DEAL_COLD_BUDGET")
    # 동기화 2단계 — 메신저 플러그인(feed 플러그인)이 SCMS 팀 새 글을 알리면(멘션 없어도) 채널별로 잠잠해질 때까지
    # (QUIET 초) 기다렸다가(첫 알림 뒤 최대 MAX_WAIT 초) 책갈피 뒤 글을 읽어 갱신한다. 기본 꺼짐 — 켜려면 키도 넣는다
    # (플러그인 설정 FeedKey 와 같은 값). 키가 비면 알림은 401.
    # 거래 문서 비전 판독(D3) — 글자 층이 없는 그림·스캔 PDF 를 비전 모델로 읽는다. 파일이 외부(Gemini)로 나간다.
    # 프로바이더·모델은 config/llm_providers.yaml 의 scms_deal_vision 레인(용도별 모델은 yaml 한 곳). 거래 구조
    # (SCMS_DEAL_CONTEXT_ENABLED)가 꺼져 있으면 돌지 않는다.
    scms_deal_vision_enabled: bool = Field(default=True, alias="SCMS_DEAL_VISION_ENABLED")
    scms_deal_feed_enabled: bool = Field(default=False, alias="SCMS_DEAL_FEED_ENABLED")
    scms_deal_feed_key: str = Field(default="", alias="SCMS_DEAL_FEED_KEY")
    scms_deal_feed_quiet: float = Field(default=60.0, alias="SCMS_DEAL_FEED_QUIET")
    scms_deal_feed_max_wait: float = Field(default=300.0, alias="SCMS_DEAL_FEED_MAX_WAIT")
    # 거래 구조의 글쓴이 소속 단서. 플랫폼 회사(문의방의 한쪽)와 그 직원 메일 도메인, 플랫폼이 발급한 협력사 계정 도메인.
    # 협력사 계정(app.example.com)이 '[X] 고객사 문의' 방에서 말하면 X 소속으로 본다(추정). 2026-10-15 실채널 계정 실측.
    scms_deal_platform_company: str = Field(default="(주)회사", alias="SCMS_DEAL_PLATFORM_COMPANY")
    scms_deal_platform_domains: str = Field(default="example.com", alias="SCMS_DEAL_PLATFORM_DOMAINS")
    scms_deal_partner_domains: str = Field(default="app.example.com", alias="SCMS_DEAL_PARTNER_DOMAINS")
    # 겸직 직원의 메일 도메인 → 그 회사(CSV "도메인=회사"). 계정 소속이 플랫폼이어도 그 회사의 문의방에서 말하면 그 회사로
    # 본다 — 고객사 A 메일의 회사 직원이 동아 방에 동아 견적을 올리던 실측(정답셋 C05).
    scms_deal_domain_companies: str = Field(default="고객사 A.kr=(주)고객사 A놀로지",
                                            alias="SCMS_DEAL_DOMAIN_COMPANIES")
    # 메신저 발주요청의 도면 판독 모델(이미지를 읽는 프로바이더). 판독은 스레드 도면을 발주 시점에 직접 읽는다.
    scms_order_drawing_provider: str = Field(default="gemini", alias="SCMS_ORDER_DRAWING_PROVIDER")
    # SCMS 웹(프론트) base — 발주요청 초안 승인 시 돌려줄 화면 URL 조립용.
    # https 여야 한다(build_action_artifact 가 https 만 resourceUrl 로 통과). SCMS 프론트는
    # 이 URL 의 경로만 써서 자기 origin 으로 이동하므로 로컬 개발에서도 이 값이면 된다.
    scms_web_base_url: str = Field(default="https://procurement.example.com", alias="SCMS_WEB_BASE_URL")
    # 메신저 견적 합의(scms_quote) 의 SCMS 견적 쓰기를 **로그만 남기고 건너뛴다**(개발기 리허설·운영 보호).
    # 켜면 등록 카드는 그대로 뜨지만 SCMS 에는 아무것도 생기지 않는다.
    scms_bid_write_dry_run: bool = Field(default=False, alias="SCMS_BID_WRITE_DRY_RUN")
    # **개발기 테스트 전용** — 메신저 견적 합의에서 말한 사람의 쪽(공급사/발주사)을 메일로 고정한다(CSV "메일=쪽",
    # 쪽은 supplier | buyer). 테스트 계정이 둘 다 회사라 공급사 역할을 할 사람이 없을 때 쓴다. 판정만 바꾼다 —
    # SCMS 서버는 여전히 실제 소속을 검사하므로 견적 등록은 SCMS_BID_WRITE_DRY_RUN 과 함께 쓴다. **운영에는 넣지 않는다.**
    scms_test_party_overrides: str = Field(default="", alias="SCMS_TEST_PARTY_OVERRIDES")
    # 전자결재(dmp-basic) API Gateway base — `/api/basic` prefix 는 client 가 자동 부착.
    # 빈 문자열로 명시 override 하면 연차 신청 fetcher 가 mock 폴백.
    eapproval_service_base_url: str = Field(default="https://api.example.com", alias="EAPPROVAL_SERVICE_BASE_URL")
    # 사내 디렉터리(Frappe common_app) API Gateway base — `/api/dmp/method/...` 경로는
    # client 가 자동 부착. 세션 sid 패스스루로 호출자 본인의 계열사 범위 직원만 조회.
    directory_service_base_url: str = Field(default="https://api.example.com", alias="DIRECTORY_SERVICE_BASE_URL")
    session_cookie_name: str = Field(default="dmp_sid", alias="SESSION_COOKIE_NAME")

    # file-service(file-service) — 채팅 첨부파일 Object Storage 영속화.
    # base_url + api_key 둘 다 설정돼야 활성화. 미설정이면 첨부 저장을 조용히 건너뛴다
    # (LLM 분석 등 기존 채팅 흐름은 그대로). URL 은 게이트웨이 미경유 직접 호출.
    file_service_base_url: str = Field(default="", alias="FILE_SERVICE_BASE_URL")
    file_service_api_key: str = Field(default="", alias="FILE_SERVICE_API_KEY")
    # 저장 경로 접두 — file-service 는 경로를 강제하지 않으므로 호출측(여기)이 정한다.
    # 최종 키: {prefix}/{subdir}/{file_id}{ext}
    file_service_prefix: str = Field(default="agent-backend/dev", alias="FILE_SERVICE_PREFIX")

    # MatterMost 서비스
    mattermost_base_url: str = Field(alias="MATTERMOST_BASE_URL")
    mattermost_token: Optional[str] = Field(default=None, alias="MATTERMOST_TOKEN")
    # 봇 계정 ID — 무한루프 방지용. 미설정 시 봇 자신 필터링 skip (개발환경 편의)
    mattermost_bot_user_id: Optional[str] = Field(default=None, alias="MATTERMOST_BOT_USER_ID")
    # 봇 계정명(@username) — 사용자 안내 문구 전용(프롬프트 "메신저에서 에이전트 부르는 법").
    # 런타임 멘션 판정은 monitor 가 기동 시 get_me() 로 받은 값을 쓰므로 이 설정과 무관하다.
    # 환경마다 봇 계정이 다르므로(운영 dat-v ↔ dev 계정) 하드코딩하지 않는다.
    mattermost_bot_username: str = Field(default="dat-v", alias="MATTERMOST_BOT_USERNAME")
    # Outgoing Webhook 시크릿 — 미설정 시 토큰 검증 skip
    mattermost_webhook_token: Optional[str] = Field(default=None, alias="MATTERMOST_WEBHOOK_TOKEN")
    # SCMS 전용 Outgoing Webhook(/webhooks/mattermost/scms) 시크릿 — 이 경로로 들어온 멘션은
    # SCMS 웹 채널(/scms/agent/*)과 같은 전용 봇(scms 모드·조달 도구만)으로 돈다. 봇 계정은
    # 일반 메신저와 같다. 미설정 시 토큰 검증 skip (MATTERMOST_WEBHOOK_TOKEN 과 같은 계약)
    mattermost_scms_webhook_token: Optional[str] = Field(
        default=None, alias="MATTERMOST_SCMS_WEBHOOK_TOKEN"
    )
    # SCMS 전용 팀(쉼표 구분 — 팀 URL 이름·표시이름·팀 ID, 예: "scms"). 이 팀 채널의 멘션은 어느
    # 경로(WS·일반 웹훅·SCMS 웹훅)로 들어와도 SCMS 전용 봇으로 돈다. WS 를 켜 두면 같은 글을 WS 가
    # 먼저 집어 가므로(post_id dedup 에서 SCMS 웹훅이 진다) SCMS 웹훅만으로는 부족하다. DM 은 팀이
    # 없어 해당하지 않는다. 채널 대신 팀으로 받는 이유: 팀에 채널이 새로 생겨도 env 를 안 고친다.
    mattermost_scms_teams: str = Field(default="", alias="MATTERMOST_SCMS_TEAMS")
    # 감시할 채널 ID 목록 (쉼표 구분 문자열, 예: "abc123,def456")
    mattermost_watch_channel_ids: str = Field(default="", alias="MATTERMOST_WATCH_CHANNEL_IDS")
    # 감시 모드 인사이트 결과를 포스팅할 채널 ID
    mattermost_insights_channel_id: Optional[str] = Field(default=None, alias="MATTERMOST_INSIGHTS_CHANNEL_ID")
    # 채널 감시 쿨다운 (초, 기본 300초 = 5분)
    mattermost_watch_cooldown_seconds: int = Field(default=300, alias="MATTERMOST_WATCH_COOLDOWN_SECONDS")
    # WebSocket 상시 연결 모니터 — 기본 OFF. 켜면 Outgoing Webhook 없이도 멘션·DM·
    # 스레드 후속을 받는다. 레플리카가 여럿이면 Redis 리더 락으로 한 프로세스만 소비.
    mattermost_ws_enabled: bool = Field(default=False, alias="MATTERMOST_WS_ENABLED")
    # WS 응답 허용 채널 ID 목록 (쉼표 구분). 빈 값이면 봇이 참여한 모든 채널.
    # DM은 이 목록과 무관하게 항상 응답한다.
    mattermost_ws_channel_ids: str = Field(default="", alias="MATTERMOST_WS_CHANNEL_IDS")
    # 승인 버튼 콜백 URL — Mattermost 서버에서 접근 가능한 주소여야 한다
    # (예: https://api.example.com/api/agent/api/v1/webhooks/mattermost/interactions).
    # 비워두면 승인 버튼을 붙이지 않고 초안 안내 문구만 남긴다.
    mattermost_interaction_callback_url: str = Field(
        default="", alias="MATTERMOST_INTERACTION_CALLBACK_URL"
    )

    # 게이트웨이 내부 세션 발급 (POST {url}/auth/internal/session).
    # 브라우저를 거치지 않는 채널(메신저 봇)이 요청자 명의로 도구를 쓰려면 게이트웨이
    # 세션(dmp_sid)이 필요한데, 봇 경로에는 사용자 쿠키가 실리지 않는다. 이 설정이 있으면
    # 신원 복원 후 게이트웨이에서 그 사람 세션을 받아 X-Session-Id 로 흘려보낸다.
    # **둘 중 하나라도 비면 발급을 시도하지 않는다** — 미설정 환경의 동작은 지금과 같다
    # (세션 패스스루 도구만 계속 실패).
    gateway_internal_url: str = Field(default="", alias="GATEWAY_INTERNAL_URL")
    gateway_internal_api_key: Optional[str] = Field(
        default=None, alias="GATEWAY_INTERNAL_API_KEY"
    )
    # 내부 발급 세션 재사용 — 사용자당 하나를 agent Redis 에 두고 쓰기 전에 /auth/me 로 살아 있는지 확인한다.
    # 끄면 예전처럼 매번 새로 발급한다(메신저 턴·예약 회차마다 14일짜리 세션이 하나씩 남는다).
    gateway_session_reuse: bool = Field(default=True, alias="GATEWAY_SESSION_REUSE")

    # MCP 도구 서버들 — URL+TOKEN 둘 다 설정돼야 등록된다.
    # 토큰은 각 서버의 MCP_SERVICE_TOKEN과 동일 값 (서버 목록: mcp/provider.py)
    mailcow_mcp_url: str = Field(default="", alias="MAILCOW_MCP_URL")
    mailcow_mcp_token: Optional[str] = Field(default=None, alias="MAILCOW_MCP_TOKEN")
    # Nextcloud(drive·calendar) MCP — 게이트웨이 세션 패스스루 방식 (nextcloud-mcp/)
    nextcloud_mcp_url: str = Field(default="", alias="NEXTCLOUD_MCP_URL")
    nextcloud_mcp_token: Optional[str] = Field(default=None, alias="NEXTCLOUD_MCP_TOKEN")
    # HR(Horilla) MCP — 내부키+이메일 직통 (hr-mcp/). 구 hr_service_* 동적 도구 대체
    hr_mcp_url: str = Field(default="", alias="HR_MCP_URL")
    hr_mcp_token: Optional[str] = Field(default=None, alias="HR_MCP_TOKEN")
    # Mattermost MCP — 봇 PAT 자동발급·저장을 MCP로 이관 (mattermost-mcp/).
    # 구 mattermost_skill_* 정적 스킬 + mattermost_service_* 동적 도구 대체
    mattermost_mcp_url: str = Field(default="", alias="MATTERMOST_MCP_URL")
    mattermost_mcp_token: Optional[str] = Field(default=None, alias="MATTERMOST_MCP_TOKEN")
    # 전자결재(eapproval) MCP — 게이트웨이 세션 패스스루 방식 (eapproval-mcp/).
    eapproval_mcp_url: str = Field(default="", alias="EAPPROVAL_MCP_URL")
    eapproval_mcp_token: Optional[str] = Field(default=None, alias="EAPPROVAL_MCP_TOKEN")
    # 웹 검색 — 브레이브 검색 MCP 서버(deploy/brave-mcp). 서버가 토큰을 받지 않으므로 URL 만으로 등록된다.
    # 회사 정책(허용 목록·일일 한도·식별자 차단·감사 원장)은 agent 쪽(services/web/guard)에 남는다.
    brave_mcp_url: str = Field(default="", alias="BRAVE_MCP_URL")
    brave_mcp_token: Optional[str] = Field(default=None, alias="BRAVE_MCP_TOKEN")
    web_search_allowed_companies: str = Field(default="", alias="WEB_SEARCH_ALLOWED_COMPANIES")  # CSV, 빈값=전체
    # 회사별/일 호출 한도, 0=무제한(기본). 2026-09-23 사용자 결정: 턴당 3~8회라 200 이면 회사 전체 하루 30~60 질문 — 운영에 못 쓴다.
    # 비용 관리는 Brave 선불 잔액·결제 알림으로. 폭주 사고 대비 안전장치가 필요하면 env 로 켠다.
    web_search_daily_quota: int = Field(default=0, alias="WEB_SEARCH_DAILY_QUOTA")
    # 답변 방식 '깊이있게 추론'(ChatGPT 위탁) 허용 회사. CSV. 값이 있으면 common_app 회사 설정 대신 이 목록이 결정한다
    # — common_app 에 정의가 배포되기 전이나 운영에서 급히 막을 때 쓰는 스위치. 비우면 회사 설정을 따른다.
    answer_mode_allowed_companies: str = Field(default="", alias="ANSWER_MODE_ALLOWED_COMPANIES")
    web_search_default_country: str = Field(default="KR", alias="WEB_SEARCH_DEFAULT_COUNTRY")   # 모델이 비우면 이 나라
    web_search_default_language: str = Field(default="ko", alias="WEB_SEARCH_DEFAULT_LANGUAGE")  # 모델이 비우면 이 언어
    web_source_goggle_path: str = Field(default="", alias="WEB_SOURCE_GOGGLE_PATH")  # 출처 정책(Brave Goggles) 파일, 빈 값=config/web_source_goggle.txt
    web_sufficiency_rounds: int = Field(default=2, alias="WEB_SUFFICIENCY_ROUNDS")  # 근거 충분성 판정·재검색 라운드(react/sufficiency), 0=끔
    web_evidence_max_chars: int = Field(default=60_000, alias="WEB_EVIDENCE_MAX_CHARS")  # 작은 창 모델(vLLM 65k) 한 턴 웹 결과 총량(react/evidence_budget), 0=무제한
    web_evidence_max_chars_large: int = Field(default=400_000, alias="WEB_EVIDENCE_MAX_CHARS_LARGE")  # 큰 창 모델(gemini 등) 웹 결과 총량
    web_fetch_deadline_seconds: float = Field(default=12, alias="WEB_FETCH_DEADLINE_SECONDS")  # 페이지 하나 열기 전체 마감(초), 넘기면 인덱스 사본으로
    web_max_rounds_per_turn: int = Field(default=3, alias="WEB_MAX_ROUNDS_PER_TURN")  # 턴당 웹 조회 단계(스텝) 상한 — 단계 안에선 여러 호출 병렬, 0=무제한
    web_max_calls_per_turn: int = Field(default=8, alias="WEB_MAX_CALLS_PER_TURN")   # 턴당 웹 호출 상한(react/budget), 0=무제한
    # 물류 MCP(erp 물류+생산현장) — 게이트웨이 세션 패스스루 방식
    # (logistics-mcp/, 구 mes-mcp — 2026-08-24 도메인 스왑). 레거시 물류 서비스용
    # LOGISTICS_SERVICE_BASE_URL(mespoc 동적 도구)과는 별개 키다.
    logistics_mcp_url: str = Field(default="", alias="LOGISTICS_MCP_URL")
    logistics_mcp_token: Optional[str] = Field(default=None, alias="LOGISTICS_MCP_TOKEN")
    # 회계 스니펫 재생(accounting-replay-mcp, :8772) — 매입매출전표 카드 경비 입력 PoC.
    # 굳힌 스니펫을 playwright-mcp 로 재생하는 래퍼. 호스트 프로세스라 host.docker.internal.
    accounting_replay_mcp_url: str = Field(default="", alias="ACCOUNTING_REPLAY_MCP_URL")
    accounting_replay_mcp_token: Optional[str] = Field(default=None, alias="ACCOUNTING_REPLAY_MCP_TOKEN")
    # SCMS(공급망 관리) MCP — 게이트웨이 세션 패스스루 방식 (scms-mcp/).
    # SCMS 조회 도구(현황·발주 상세)를 노출한다. 위 SCMS_SERVICE_BASE_URL 과는 별개 키다 —
    # 그쪽은 데모 시나리오 fetcher 가 agent 안에서 직접 부르는 ScmsClient 용이고,
    # 이쪽은 MCP 서버를 거치는 에이전트 도구용이다.
    # ⚠ 호스트 포트 8772 를 accounting-replay-mcp 와 나눠 쓴다. agent 는 도커 네트워크
    #   별칭으로 부르므로 컨테이너끼리는 안 겹치지만, 같은 장비에서 둘 다 호스트에
    #   발행하면 뒤에 뜨는 쪽이 바인드에 실패한다(scms-mcp Makefile HOST_PORT= 로 끈다).
    scms_mcp_url: str = Field(default="", alias="SCMS_MCP_URL")
    scms_mcp_token: Optional[str] = Field(default=None, alias="SCMS_MCP_TOKEN")

    # Confluence 사용자 가이드 KB 설정
    confluence_base_url: str = Field(default="", alias="CONFLUENCE_BASE_URL")
    # 인덱싱할 Space 키 목록 (쉼표 구분, 예: "GUIDE,HR") — CONFLUENCE_ANCESTOR_PAGE_IDS 미설정 시 사용
    confluence_space_keys: str = Field(default="", alias="CONFLUENCE_SPACE_KEYS")
    # QA 도구 표면 내보내기(GET /api/v1/qa/tool-surface) 허용 계정 — 쉼표 구분 이메일. 비우면 엔드포인트 꺼짐(404).
    qa_export_allowed_emails: str = Field(default="", alias="QA_EXPORT_ALLOWED_EMAILS")
    # 인덱싱 범위를 특정 페이지 하위로 제한 (쉼표 구분 페이지 ID, 예: "123456,789012")
    # 설정 시 CONFLUENCE_SPACE_KEYS 대신 해당 페이지의 모든 자손 페이지를 수집 (CQL ancestor 사용)
    confluence_ancestor_page_ids: str = Field(default="", alias="CONFLUENCE_ANCESTOR_PAGE_IDS")
    # 해당 레이블이 붙은 페이지만 인덱싱 (선택, 예: "agent-guide")
    confluence_label_filter: str = Field(default="", alias="CONFLUENCE_LABEL_FILTER")
    # Atlassian 계정 이메일 (Basic Auth용, Cloud 필수)
    confluence_email: str = Field(default="", alias="CONFLUENCE_EMAIL")
    # Atlassian API 토큰 (https://id.atlassian.com/manage-profile/security/api-tokens)
    confluence_api_token: str = Field(default="", alias="CONFLUENCE_API_TOKEN")

    # 최적화 정식화 LLM (파인튜닝된 Qwen3-Coder-30B; OpenAI 호환 vLLM)
    # 자연어 최적화 문제 → CPMpy(OR-Tools CP-SAT) 정식화 코드 생성 (app/services/optimization).
    # base_url 미설정 시 서비스 비활성(skill 이 안내 메시지 반환). 외부 프록시면 INSECURE=true(-k).
    optimization_llm_base_url: str = Field(default="", alias="OPTIMIZATION_LLM_BASE_URL")
    optimization_llm_model: str = Field(default="oropt", alias="OPTIMIZATION_LLM_MODEL")
    optimization_llm_api_key: str = Field(default="EMPTY", alias="OPTIMIZATION_LLM_API_KEY")
    optimization_llm_insecure: bool = Field(default=True, alias="OPTIMIZATION_LLM_INSECURE")
    optimization_exec_timeout: int = Field(default=45, alias="OPTIMIZATION_EXEC_TIMEOUT")
    # 생성 코드는 cpmpy/ortools 가 깔린 *별도 솔버 venv* 에서 실행한다 (메인 env 의 protobuf 등과
    # 격리 + ortools↔gemini protobuf 충돌 회피). 미설정 시 현재 인터프리터 사용(cpmpy 필요).
    optimization_exec_python: str = Field(default="", alias="OPTIMIZATION_EXEC_PYTHON")

    # 소싱(공급사 추천) 데이터 소스. 현재는 스크랩 CSV(gs_full.csv/tn_full.csv) 디렉터리.
    # 차후 SCMS/ERP 어댑터로 교체 — SupplierSource 인터페이스만 맞추면 됨. 빈값=레포 루트.
    sourcing_data_dir: str = Field(default="", alias="SOURCING_DATA_DIR")

    # Odoo CRM (전시회 이벤트 참가자 → crm.lead 통합)
    # JSON-RPC 는 root 의 /jsonrpc 에 노출됨. /odoo prefix 는 웹 UI 전용이므로 base_url 에 포함 X.
    odoo_base_url: str = Field(default="https://crm.example.com", alias="ODOO_BASE_URL")
    odoo_db: str = Field(default="", alias="ODOO_DB")
    odoo_username: str = Field(default="", alias="ODOO_USERNAME")
    odoo_api_key: Optional[str] = Field(default=None, alias="ODOO_API_KEY")
    # STK 2026 전시회 UTM 메타데이터 ID
    odoo_campaign_id: Optional[int] = Field(default=None, alias="ODOO_CAMPAIGN_ID")
    odoo_medium_id: Optional[int] = Field(default=None, alias="ODOO_MEDIUM_ID")
    odoo_source_id: Optional[int] = Field(default=None, alias="ODOO_SOURCE_ID")
    odoo_team_id: Optional[int] = Field(default=None, alias="ODOO_TEAM_ID")
    # STK 2026 고정 태그 — crm.lead.tag_ids(many2many→crm.tag) 에 무조건 부착.
    # 라이브 Odoo 조회로 확인한 id: 1=STK 2026, 33=에이전트챌린지. 인스턴스 변경 시 env override.
    odoo_lead_tag_ids: str = Field(default="1,33", alias="ODOO_LEAD_TAG_IDS")

    # 임베딩 모델. 모델명으로 프로바이더가 결정된다(app/llm/embeddings.py):
    #   "models/..."/"gemini" → Gemini(GEMINI_API_KEY), 그 외 → OpenAI(OPENAI_API_KEY).
    #   OpenAI 로 쓰려면 EMBEDDING_MODEL=text-embedding-3-large (둘 다 3072차원).
    embedding_model: str = Field(default="models/gemini-embedding-001", alias="EMBEDDING_MODEL")

    # STT 설정
    stt_model: str = Field(default="gpt-4o-transcribe", alias="STT_MODEL")
    # **비어 있으면 언어를 전달하지 않고 모델의 자동 감지에 맡긴다(권장).**
    # 실사용 발화가 한국어·영어·중국어 세 가지라 특정 언어를 못 박을 수 없고, 못 박으면
    # 다른 언어 입력에서 전사 대신 **번역**이 나온다(2026-09-03 실측: language=ko 로
    # 영어 mp3 를 넣으면 "아프터 이른 밤이 되면..." 처럼 한국어로 번역해서 낸다).
    # 자동 감지는 영어 3개 형식 모두 원문 그대로 전사했고 지연도 동일했다.
    stt_language: str = Field(default="", alias="STT_LANGUAGE")
    # 자체 호스팅 STT(OpenAI 호환 서버, 예: vLLM qwen3-asr). **비어 있으면 OpenAI 를 그대로
    # 쓴다** — 미설정 환경의 동작은 바뀌지 않는다. 값이 있으면 그 서버로 붙고, 아래 두 개를
    # extra_body 로 실어 보낸다(모델이 요구하는 안전장치).
    stt_base_url: str = Field(default="", alias="STT_BASE_URL")
    stt_api_key: Optional[str] = Field(default=None, alias="STT_API_KEY")
    # 오디오 1초당 허용 출력 토큰 — 잡음 구간에서 모델이 폭주하는 것을 막는 천장 계수.
    stt_tokens_per_second: int = Field(default=8, alias="STT_TOKENS_PER_SECOND")
    # 위 계수로 계산한 천장의 절대 상한(긴 파일에서 무한정 커지지 않게).
    stt_max_completion_tokens: int = Field(default=8000, alias="STT_MAX_COMPLETION_TOKENS")
    # 반복 루프 억제.
    stt_repetition_penalty: float = Field(default=1.1, alias="STT_REPETITION_PENALTY")
    # 자체 호스팅이 **호출 시점에** 실패하면 OpenAI 로 승계한다. 자체 호스팅을 켜는 순간
    # 그 서버가 음성 채널의 단일 실패점이 되므로 기본값은 켜둔다.
    #
    # 다만 승계가 일어나면 오디오 원본이 외부로 나간다 — 프라이버시를 이유로 자체 호스팅을
    # 택한 환경이라면 꺼야 한다(그 경우 서버가 죽으면 음성 채널도 같이 멈춘다).
    # STT_BASE_URL 이 비어 있으면(=OpenAI 가 이미 주경로) 이 설정은 무의미하다.
    stt_fallback_to_openai: bool = Field(default=True, alias="STT_FALLBACK_TO_OPENAI")
    # 승계 대상 모델. 주경로 모델명(예: qwen3-asr)은 OpenAI 가 모르므로 별도로 둔다.
    stt_fallback_model: str = Field(default="gpt-4o-transcribe", alias="STT_FALLBACK_MODEL")

    # ── 음성 유무 판정(VAD) ────────────────────────────────────────────────
    # 전사 **전에** 오디오에서 사람 말을 찾는다. 자동 감지 구성에서는 비음성 입력의
    # 환각을 텍스트로 막을 수 없기 때문이다(근거는 app/services/stt/vad.py 상단).
    stt_vad_enabled: bool = Field(default=True, alias="STT_VAD_ENABLED")
    # 프레임(20ms) 레벨의 동적 범위 p95-p10 이 이 값 미만이면 "일정한 소리"로 본다.
    # 클라이언트가 보내는 실제 포맷(opus)에서 측정해 정했다 — 상세는 vad.py 상단.
    # 현장 녹음이 모이면 이 값을 다시 잡아야 한다.
    stt_vad_min_dynamic_range_db: float = Field(
        default=3.0, alias="STT_VAD_MIN_DYNAMIC_RANGE_DB"
    )
    # 가장 큰 프레임이 이보다 조용하면 들리는 소리가 없다고 본다.
    stt_vad_min_peak_dbfs: float = Field(default=-60.0, alias="STT_VAD_MIN_PEAK_DBFS")

    # 음성 요청 로그에 **전사 텍스트**를 남긴다. 업무 내용이 로그에 쌓이므로 기본은 끔.
    # 현장 검증 기간에만 켠다 — 식별자(LOT/시리얼/품번)가 어떻게 전사되는지 보려면 필요하다.
    # 끈 상태에서도 텍스트 통계(길이·숫자 포함·문자 종류)는 항상 남는다.
    stt_log_transcript: bool = Field(default=False, alias="STT_LOG_TRANSCRIPT")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    # Claude(Anthropic). 회사별 BYO 키가 오면 그 값이 우선하고, 여기 값은 미지정 시 기본이다.
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")

    # PostgreSQL 데이터베이스 설정
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="agent_agent", alias="POSTGRES_DB")
    postgres_user: str = Field(default="postgres", alias="POSTGRES_USER")
    postgres_password: str = Field(default="postgres", alias="POSTGRES_PASSWORD")
    database_url: str = Field(alias="DATABASE_URL")

    # MongoDB 설정
    mongodb_url: str = Field(default="mongodb://localhost:27017", alias="MONGODB_URL")
    mongodb_db: str = Field(default="agent_agent", alias="MONGODB_DB")

    # Keycloak 인증 설정
    keycloak_url: str = Field(default="https://sso.example.com", alias="KEYCLOAK_URL")
    keycloak_realm: str = Field(default="example", alias="KEYCLOAK_REALM")
    keycloak_client_id: str = Field(default="agent-backend", alias="KEYCLOAK_CLIENT_ID")
    keycloak_client_secret: Optional[str] = Field(default=None, alias="KEYCLOAK_CLIENT_SECRET")

    # CORS 설정
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")

    # 응답 정화(sanitize) — 내부 도구명 마스킹 화이트리스트를 어느 소스에서 모을지.
    # dynamic=llm_tools.json(동적 API 도구) / registry=ToolRegistry(정적 스킬+MCP) /
    # config=tools_config.yaml 키. 콤마 구분, 기본은 셋 다.
    # 기본 ON 인 이유: "새 기능은 기본 OFF" 관례의 취지는 기존 동작 보전인데, 여기서는
    # dynamic 단독이던 기존 동작 자체가 버그였다(MCP·정적 스킬 도구명이 전혀 안 걸림).
    # 이 스위치는 신기능 게이트가 아니라 오탐 발생 시의 후퇴 레버다 — 오탐은 200 응답에
    # 로그도 안 남고 "말은 되는데 단어가 사라진 답변"으로 조용히 나가 발견이 늦으므로,
    # 코드 revert·재빌드 없이 .env + 재기동으로 소스를 하나씩 빼며 범인을 좁힐 수 있어야 한다.
    sanitize_tool_name_sources: str = Field(
        default="dynamic,registry,config", alias="SANITIZE_TOOL_NAME_SOURCES"
    )

    # 로깅 설정
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: str = Field(default="json", alias="LOG_FORMAT")

    # YAML 설정 파일에서 로드되는 설정
    llm_providers: Dict[str, Any] = {}
    tools_config: Dict[str, Any] = {}
    logging_config: Dict[str, Any] = {}

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="allow",
    )

    def __init__(self, **kwargs):
        """Settings 인스턴스를 초기화합니다.

        Args:
            **kwargs: 추가 설정 파라미터
        """
        super().__init__(**kwargs)
        self._load_yaml_configs()

    def _load_yaml_configs(self):
        """YAML 설정 파일들을 로드합니다.

        config 디렉토리의 YAML 파일들을 읽어서 해당 속성에 저장합니다.
        """
        config_dir = Path(__file__).resolve().parents[3] / "config"  # 레포 루트 config/

        # LLM Providers 설정 로드
        llm_config_path = config_dir / "llm_providers.yaml"
        if llm_config_path.exists():
            with open(llm_config_path, "r", encoding="utf-8") as f:
                self.llm_providers = yaml.safe_load(f)

        # Tools 설정 로드
        tools_config_path = config_dir / "tools_config.yaml"
        if tools_config_path.exists():
            with open(tools_config_path, "r", encoding="utf-8") as f:
                self.tools_config = yaml.safe_load(f)

        # Logging 설정 로드
        logging_config_path = config_dir / "logging.yaml"
        if logging_config_path.exists():
            with open(logging_config_path, "r", encoding="utf-8") as f:
                self.logging_config = yaml.safe_load(f)

    def get_llm_config(self, provider: str) -> Dict[str, Any]:
        """특정 LLM 프로바이더 설정을 반환합니다.

        Args:
            provider (str): LLM 프로바이더 이름 (gemini, vllm 등)

        Returns:
            Dict[str, Any]: 프로바이더 설정 **복사본**. 없으면 빈 딕셔너리.

        복사본을 주는 이유: 호출자(LLMFactory.create)가 `config.update(kwargs)` 로
        temperature·model·api_key 를 덮는다. 원본을 그대로 주면 그 수정이 프로세스 전역
        yaml 설정에 남아 **다음 요청까지 오염된다**(실측: 잘못된 모델명을 한 번 보낸 뒤,
        모델을 지정하지 않은 다음 요청까지 같은 모델명으로 404. 회사별 API 키를 쓰기
        시작하면 이건 타사 키 유출 경로가 된다).

        Examples:
            >>> config = settings.get_llm_config("gemini")
            >>> print(config.get("model"))
            'gemini-pro'
        """
        return dict(self.llm_providers.get("providers", {}).get(provider, {}))

    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """특정 도구 설정을 반환합니다.

        Args:
            tool_name (str): 도구 이름

        Returns:
            Dict[str, Any]: 도구 설정 딕셔너리. 없으면 빈 딕셔너리

        Examples:
            >>> config = settings.get_tool_config("web_search")
            >>> print(config.get("enabled"))
            True
        """
        return self.tools_config.get("tools", {}).get(tool_name, {})

    @property
    def odoo_lead_tag_id_list(self) -> list[int]:
        """ODOO_LEAD_TAG_IDS(CSV)를 정수 ID 리스트로 파싱한다.

        Returns:
            쉼표로 구분된 crm.tag id 리스트. 미설정·빈 문자열이면 빈 리스트.
        """
        if not self.odoo_lead_tag_ids:
            return []
        return [int(x.strip()) for x in self.odoo_lead_tag_ids.split(",") if x.strip()]

    @property
    def qa_export_allowed_email_set(self) -> set[str]:
        """QA_EXPORT_ALLOWED_EMAILS 를 소문자 이메일 집합으로. 미설정이면 빈 집합(=기능 꺼짐)."""
        return {e.strip().lower() for e in (self.qa_export_allowed_emails or "").split(",") if e.strip()}

    @property
    def confluence_space_key_list(self) -> list[str]:
        """CONFLUENCE_SPACE_KEYS를 파싱하여 Space 키 리스트를 반환한다.

        Returns:
            쉼표로 구분된 Space 키 리스트. 미설정 시 빈 리스트
        """
        if not self.confluence_space_keys:
            return []
        return [k.strip() for k in self.confluence_space_keys.split(",") if k.strip()]

    @property
    def confluence_ancestor_page_id_list(self) -> list[str]:
        """CONFLUENCE_ANCESTOR_PAGE_IDS를 파싱하여 페이지 ID 리스트를 반환한다.

        Returns:
            쉼표로 구분된 페이지 ID 리스트. 미설정 시 빈 리스트
        """
        if not self.confluence_ancestor_page_ids:
            return []
        return [k.strip() for k in self.confluence_ancestor_page_ids.split(",") if k.strip()]

    @property
    def mattermost_watch_channel_list(self) -> list:
        """감시 채널 ID 목록을 파싱해 반환한다.

        Returns:
            쉼표로 구분된 MATTERMOST_WATCH_CHANNEL_IDS를 파싱한 채널 ID 리스트.
            미설정이거나 빈 문자열이면 빈 리스트를 반환한다.
        """
        if not self.mattermost_watch_channel_ids:
            return []
        return [c.strip() for c in self.mattermost_watch_channel_ids.split(",") if c.strip()]

    @property
    def mattermost_ws_channel_list(self) -> list:
        """WS 응답 허용 채널 ID 목록을 파싱해 반환한다.

        Returns:
            쉼표로 구분된 MATTERMOST_WS_CHANNEL_IDS를 파싱한 채널 ID 리스트.
            미설정이거나 빈 문자열이면 빈 리스트(=채널 제한 없음)를 반환한다.
        """
        if not self.mattermost_ws_channel_ids:
            return []
        return [c.strip() for c in self.mattermost_ws_channel_ids.split(",") if c.strip()]

    @property
    def mattermost_scms_team_keys(self) -> set:
        """SCMS 전용 팀 키(MATTERMOST_SCMS_TEAMS, 소문자). 미설정이면 빈 집합."""
        return {t.strip().lower() for t in (self.mattermost_scms_teams or "").split(",") if t.strip()}

    @property
    def is_production(self) -> bool:
        """프로덕션 환경인지 확인합니다.

        Returns:
            bool: 프로덕션 환경이면 True, 아니면 False

        Examples:
            >>> if settings.is_production:
            ...     print("프로덕션 모드 실행 중")
        """
        return self.environment.lower() == "production"


# 전역 설정 싱글톤 인스턴스
settings = Settings()
