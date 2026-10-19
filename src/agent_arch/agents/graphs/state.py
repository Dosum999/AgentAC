import operator
from typing import TypedDict, List, Optional, Any, Dict, Set, Annotated, Tuple
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """에이전트 그래프 상태.

    LangGraph에서 사용하는 상태를 정의한다.
    MessagesState를 상속하여 messages 필드를 자동으로 포함한다.

    Attributes:
        input: 사용자 입력
        llm_provider: LLM 공급자
        tool_results: 도구 실행 결과 리스트
        iteration: 현재 반복 횟수
        max_iterations: 최대 반복 횟수
        output: 최종 출력
        error: 오류 메시지
        metadata: 메타데이터
        should_continue: 종료 플래그
    """

    # 입력
    input: str

    # LLM 관련
    llm_provider: Optional[str] = None

    # 도구 실행 결과
    tool_results: List[Dict[str, Any]] = []

    # 반복 제어
    iteration: int = 0
    max_iterations: int = 10

    # 최종 출력
    output: Optional[str] = None

    # 에러 처리
    error: Optional[str] = None

    # 메타데이터
    metadata: Dict[str, Any] = {}

    # 종료 플래그
    should_continue: bool = True


class ReactState(TypedDict, total=False):
    """ReAct StateGraph 상태 (app/agents/graphs/react_graph.py).

    ``react_loop_events`` 의 지역 변수 집합을 그래프 상태로 옮긴 것. 체크포인터를 붙이지
    않으므로(단일 턴, interrupt 없음) BaseLLM·도구 인스턴스·커스텀 ``Message`` 등
    비직렬화 객체를 그대로 담는다. 커스텀 ``Message`` 를 쓰므로 ``MessagesState`` /
    ``add_messages`` 리듀서는 쓰지 않는다(모든 채널 기본 LastValue = 마지막 값 우선).

    노드는 자신이 바꾼 키만 부분 반환하고, in-place 로 변형하는 컨테이너
    (``messages``/``all_tools``/``collected_*``/``token_usage``/``known_tool_names``)는
    변형 후 반드시 함께 반환해 채널을 갱신한다.
    """

    # ── 의존성(런타임 주입, ainvoke 초기 state 로 전달) ──
    llm: Any            # BaseLLM (이미 bind_tools 된 상태로 주입 — react_loop_events 와 동일)
    search_fn: Any      # 동적 재검색 콜백(현재 루프 본문 미사용, 시그니처 대칭용)
    input: str          # 사용자 입력(augmented) — 답글/리액션 의도 판정에 사용
    max_iterations: int

    # ── in-place 확장되는 컨테이너 ──
    messages: List[Any]
    all_tools: List[Any]

    # ── 루프 누적 상태 ──
    token_usage: Dict[str, int]
    collected_tool_calls: List[dict]
    collected_tool_results: List[dict]
    known_tool_names: Set[str]
    used_tool_names: List[str]
    # 같은 도구가 **같은 이유로** 연속 실패한 횟수 — 상한을 넘으면 루프를 끊는다.
    # 2026-09-14 운영: create_file 이 13회 연속 `filename 이 필요합니다` 로 실패하며
    # 5분을 돌다 반복 상한으로 끝났다. 사용자는 그동안 아무것도 못 받는다.
    fail_signature: Optional[str]
    fail_streak: int
    # 이번 턴에 웹 호출을 실제로 실행한 스텝 수(react/budget 라운드 상한). 단계 안의 병렬 호출은 1로 센다.
    web_rounds: int
    # 첫 모델 호출 전에 코드가 먼저 실행할 도구 호출([{name, arguments}]) — 웹 조사 계획(router.web_queries).
    prefetch_calls: List[dict]
    # 선실행 단계(검색 선실행 → 상위 페이지 자동 열기)를 도는 중인가 — 이 단계의 호출은 웹 단계 예산에서 뺀다.
    prefetch_phase: bool
    auto_fetch_done: bool
    auto_fetch_enabled: bool   # 라우터 web_open — 특정 페이지의 최신 값이 답일 때만 자동 열기
    suff_rounds: int           # 근거 충분성 판정을 돈 횟수(react/sufficiency)
    suff_meta: List[dict]      # 라운드별 판정(step_meta 관측용)
    llm_invoke_count: int
    used_tools: bool
    forced_any_mode: bool
    forced_any_done: bool
    # 진행 중 초안(수집 중·거절됨) 턴에 첫 호출을 도구로 강제할지. **채널 스코핑
    # 모드에서만** 호출자가 켠다(react_graph `_force_tool_reason` 참조).
    pending_draft_force: bool
    # guide-first 가 서브쿼리에서 미매치일 때 턴 원문(get_turn_input)으로 한 번 더 판정해도 되는가.
    # supervisor 가 단일 도메인·단일 서브런 턴에만 True 로 넘긴다(복합 발화의 다른 워커가 원문의
    # '방법' 때문에 강제되는 것 방지). 직접 호출 경로는 input 이 곧 원문이라 무관(2026-10-13).
    turn_input_guide_fallback: bool
    # guide-first: 방법·설정 질문이면 첫 call_model 의 도구 호출을 이 도구로 고정한다
    # (react/howto.py). 소비 후 None. guide_first 는 집계용(적용됐는지).
    force_tool_name: Optional[str]
    guide_first: bool
    # 턴 원문으로만 guide-first 가 판정된 경우의 검색어. None 이면 기존처럼 서브쿼리를 쓴다(2026-10-13).
    guide_query_text: Optional[str]
    iteration: int
    # HITL 승인 대기 초안 누적 — 승인 도구를 만나도 즉시 멈추지 않고 여기 모아두고 루프를
    # 계속(조회/요약 완료 유도), 최종 답변(터미널) 시점에 이 큐로 승인 대기한다.
    pending_drafts: List[Any]

    # ── per-iteration 스크래치(call_model 이 세팅, 라우터/execute 가 읽음) ──
    iter_tool_calls: List[Any]
    finish_reason: str
    text_buf: List[str]
    cancelled: bool
    # 목표 언어가 아닌 답변을 감지해 토큰 방출을 보류했는지 — 최종 답변 노드가
    # 재생성본을 한 번에 방출해야 하는지 판단한다(_AnswerLanguageGate 참조).
    answer_withheld: bool
    # 답변 검증(react/verify): 보강 루프 횟수, 라운드별 판정 기록(step_meta 로 저장), 이번 판정의 분기 결정.
    verify_rounds: int
    verify_meta: List[dict]
    verify_decision: str
    # 보강 라운드를 다 쓴 뒤 검증기가 rewrite(근거와 어긋남)를 내면 도구 없이 1회 고쳐 쓴다 — 그 1회를 썼는지.
    rewrite_done: bool
    # 이번 턴 이전 대화(앞 답·앞 도구 결과)의 본문 — 검증기가 앞 턴에서 확인한 값을 '근거 없음'으로 잡지 않게(react/verify).
    history_evidence: str
    # 이 턴의 답변 목표 언어(ko/en/zh/ja). orchestrator 가 요청 필드 또는 입력 감지로
    # 정해 주입한다. 없으면 한국어 — 가드가 언어를 못 정한 채 도는 일은 없게 한다.
    target_language: str
    # 이름 가드 관측 신호 {armed, detected, regenerated, masked} — step_meta 로 나가
    # agent.messages.metadata 와 LangSmith run 에 함께 저장된다.
    name_guard: dict

    # ── 스텝 계측(집계용, operator.add 리듀서로 노드 경계 넘어 누적) ──
    # node_trace: 방문 노드 순서 + 소요시간 [(node_name, elapsed_sec), ...] — 경로/타이밍/재시도 도출.
    # discovered_tool_names: 각 iteration 에서 발견·바인딩된 도구 이름 누적.
    node_trace: Annotated[List[Tuple[str, float]], operator.add]
    discovered_tool_names: Annotated[List[str], operator.add]

    # ── 종결 ──
    result: Optional[Any]  # LoopResult (그래프 최종 산출물)
