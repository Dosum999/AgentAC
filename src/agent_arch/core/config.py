from typing import Dict, Any, Optional
from pydantic import Field, ConfigDict
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
        enable_pii_masking: PII 마스킹 활성화 여부
        max_iterations: 에이전트 최대 반복 횟수
        agent_timeout: 에이전트 타임아웃 (초)
        postgres_host: PostgreSQL 호스트
        postgres_port: PostgreSQL 포트
        postgres_db: PostgreSQL 데이터베이스 이름
        postgres_user: PostgreSQL 사용자
        postgres_password: PostgreSQL 비밀번호
        database_url: 데이터베이스 연결 URL
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
    # 용어 사전 Redis 캐시 TTL(초). PA 가 용어를 고치면 이만큼 뒤에 답변에 반영된다.
    # 플랫폼 매니저 화면이 저장 후 이 시간만큼 `반영 중` 을 표시한다(GLOSSARY_SYNC_SECONDS)
    # — 한쪽만 바꾸면 화면이 "반영됨"이라는데 실제로는 아직인 구간이 생긴다.
    glossary_cache_ttl: int = Field(default=60, alias="GLOSSARY_CACHE_TTL")
    # 벤더 호출 계측(agent.usage_meters, migration_v42). 끄면 기록만 멈추고 요청은 그대로 돈다.
    enable_usage_meters: bool = Field(default=True, alias="ENABLE_USAGE_METERS")
    # 시스템 오류 이벤트(agent.error_events, migration_v44) 수집 + common_app 푸시. 관측 「에이전트
    # 모니터링」 화면의 원인 축이다. 끄면 수집·푸시가 모두 멈추고(요청은 그대로), 화면 배지는
    # 수신이 끊겨 '끊김'으로 바뀐다. 푸시 인증은 COMMON_APP_INTERNAL_API_KEY 우선, 없으면
    # FRAPPE_API_KEY/SECRET(legacy whitelist) — 둘 다 없으면 수집만 하고 푸시를 건너뛴다.
    enable_error_events: bool = Field(default=True, alias="ENABLE_ERROR_EVENTS")

    # 통합 도구 인덱싱 — 정적 스킬·MCP 도구 설명을 동적 API와 같은 Qdrant
    # 컬렉션에 인덱싱해 retrieved 티어(질의 검색 노출)를 활성화한다.
    # False면 exposure=retrieved 도구도 core로 승격된다 (도구 소실 방지).
    unified_tool_indexing: bool = Field(default=False, alias="UNIFIED_TOOL_INDEXING")
    agent_mode: str = Field(default="default", alias="AGENT_MODE")
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
    #   자리 = 전용 앱 창(agent:app — /agent/app·/audio) · 메신저 채널(mattermost:<channel_id>).
    #          브라우저 패널과 DM 은 자리가 없다(중립 — 어디서나 그대로 적용).
    #   True : 그 자리에서 못 박은 규칙은 그 자리에서만 매 턴 주입되고, 회상 순위도 같은
    #          자리를 올린다(다른 자리는 살짝 내린다). 전용 앱 창 분리와 메신저 채널 분리가
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


    # LangSmith 트레이싱 설정
    enable_langsmith: bool = Field(default=False, alias="ENABLE_LANGSMITH")
    langsmith_api_key: Optional[str] = Field(default=None, alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="agent-backend", alias="LANGSMITH_PROJECT")
    langsmith_dataset_name: str = Field(default="agent-feedback", alias="LANGSMITH_DATASET_NAME")

    # 외부 서비스 Base URL (API Gateway 경유)
    hr_service_base_url: str = Field(default="http://hr-service", alias="HR_SERVICE_BASE_URL")
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
    mattermost_base_url: str = Field(default="", alias="MATTERMOST_BASE_URL")
    # 봇 계정명(@username) — 사용자 안내 문구 전용(프롬프트 "메신저에서 에이전트 부르는 법").
    # 런타임 멘션 판정은 monitor 가 기동 시 get_me() 로 받은 값을 쓰므로 이 설정과 무관하다.
    # 환경마다 봇 계정이 다르므로(운영 ↔ 개발 계정) 하드코딩하지 않는다.
    mattermost_bot_username: str = Field(default="agent", alias="MATTERMOST_BOT_USERNAME")


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
    # 웹 검색 — 브레이브 검색 MCP 서버(deploy/brave-mcp). 서버가 토큰을 받지 않으므로 URL 만으로 등록된다.
    # 회사 정책(허용 목록·일일 한도·식별자 차단·감사 원장)은 agent 쪽(services/web/guard)에 남는다.
    brave_mcp_url: str = Field(default="", alias="BRAVE_MCP_URL")
    brave_mcp_token: Optional[str] = Field(default=None, alias="BRAVE_MCP_TOKEN")
    web_search_allowed_companies: str = Field(default="", alias="WEB_SEARCH_ALLOWED_COMPANIES")  # CSV, 빈값=전체
    # 회사별/일 호출 한도, 0=무제한(기본). 2026-09-23 사용자 결정: 턴당 3~8회라 200 이면 회사 전체 하루 30~60 질문 — 운영에 못 쓴다.
    # 비용 관리는 Brave 선불 잔액·결제 알림으로. 폭주 사고 대비 안전장치가 필요하면 env 로 켠다.
    web_search_daily_quota: int = Field(default=0, alias="WEB_SEARCH_DAILY_QUOTA")
    web_search_default_country: str = Field(default="KR", alias="WEB_SEARCH_DEFAULT_COUNTRY")   # 모델이 비우면 이 나라
    web_search_default_language: str = Field(default="ko", alias="WEB_SEARCH_DEFAULT_LANGUAGE")  # 모델이 비우면 이 언어
    web_source_goggle_path: str = Field(default="", alias="WEB_SOURCE_GOGGLE_PATH")  # 출처 정책(Brave Goggles) 파일, 빈 값=config/web_source_goggle.txt
    web_sufficiency_rounds: int = Field(default=2, alias="WEB_SUFFICIENCY_ROUNDS")  # 근거 충분성 판정·재검색 라운드(react/sufficiency), 0=끔
    web_evidence_max_chars: int = Field(default=60_000, alias="WEB_EVIDENCE_MAX_CHARS")  # 작은 창 모델(vLLM 65k) 한 턴 웹 결과 총량(react/evidence_budget), 0=무제한
    web_evidence_max_chars_large: int = Field(default=400_000, alias="WEB_EVIDENCE_MAX_CHARS_LARGE")  # 큰 창 모델(gemini 등) 웹 결과 총량
    web_fetch_deadline_seconds: float = Field(default=12, alias="WEB_FETCH_DEADLINE_SECONDS")  # 페이지 하나 열기 전체 마감(초), 넘기면 인덱스 사본으로
    web_max_rounds_per_turn: int = Field(default=3, alias="WEB_MAX_ROUNDS_PER_TURN")  # 턴당 웹 조회 단계(스텝) 상한 — 단계 안에선 여러 호출 병렬, 0=무제한
    web_max_calls_per_turn: int = Field(default=8, alias="WEB_MAX_CALLS_PER_TURN")   # 턴당 웹 호출 상한(react/budget), 0=무제한


    # 임베딩 모델. 모델명으로 프로바이더가 결정된다(app/llm/embeddings.py):
    #   "models/..."/"gemini" → Gemini(GEMINI_API_KEY), 그 외 → OpenAI(OPENAI_API_KEY).
    #   OpenAI 로 쓰려면 EMBEDDING_MODEL=text-embedding-3-large (둘 다 3072차원).
    embedding_model: str = Field(default="models/gemini-embedding-001", alias="EMBEDDING_MODEL")


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
