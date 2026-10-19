"""pytest 공통 픽스처 모음."""
import os
import pytest


def pytest_configure(config):
    """pytest 설정 단계에서 마커와 경고 필터를 등록한다."""
    # retrieval 품질 평가는 실제 Qdrant/Gemini를 호출하므로 opt-in으로 분리한다.
    # 실행: `pytest tests/evals -m eval -s`
    config.addinivalue_line(
        "markers",
        "eval: retrieval 품질 평가 (opt-in, 실제 Qdrant/Gemini 호출)",
    )
    # google-genai SDK가 aiohttp.ClientSession을 상속하는 방식이 aiohttp 3.13+과 충돌.
    # SDK 내부 버그로 우리가 수정할 수 없으므로 필터로 억제한다.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:Inheritance class AiohttpClientSession from ClientSession is discouraged:DeprecationWarning",
    )

@pytest.fixture(autouse=True)
def _langgraph_supervisor_off_by_default(monkeypatch):
    """테스트를 런타임 .env 의 USE_LANGGRAPH_SUPERVISOR 와 독립시킨다(기본 OFF).

    운영 .env 에 supervisor flag 가 켜져 있어도 테스트가 그걸 읽어 supervisor 경로로
    새면(라우터 llm 생성 등) 깨진다. 기본 OFF 로 고정하고, supervisor 라우팅을 검증하는
    테스트는 patch.object(settings, "use_langgraph_supervisor", True) 로 명시 override 한다.
    """
    monkeypatch.setattr("agent_arch.core.config.settings.use_langgraph_supervisor", False, raising=False)


@pytest.fixture(autouse=True)
def _answer_verify_off_by_default(monkeypatch):
    """테스트를 런타임 .env 의 ANSWER_VERIFY_MODE 와 독립시킨다(기본 off).

    dev .env 는 shadow 로 판정을 기록하는데, 테스트가 그걸 읽으면 react 그래프가 verify 노드로 새어
    검증기 LLM 을 실제로 부르고 invoke 수가 어긋난다(패리티 실패). 검증 노드를 다루는 테스트는
    monkeypatch.setattr(settings, "answer_verify_mode", ...) 로 명시 override 한다.
    """
    monkeypatch.setattr("agent_arch.core.config.settings.answer_verify_mode", "off", raising=False)
    # 웹 근거 충분성 루프(react/sufficiency)도 판정 LLM 을 실제로 부른다 — 기본 끔, 그 루프를 다루는 테스트만 켠다.
    monkeypatch.setattr("agent_arch.core.config.settings.web_sufficiency_rounds", 0, raising=False)
    # 그래프 HITL 은 레거시 loop 와 동작이 갈라짐(승인 pause 지연) → 실 settings 쓰는 테스트는
    # 기본 loop 경로로. 그래프 자체는 run_react_graph 직접 호출 테스트가 커버, flag 테스트는 override.
    monkeypatch.setattr("agent_arch.core.config.settings.use_langgraph_react", False, raising=False)
    # SCMS 메신저 거래 구조 추출기도 LLM 을 실제로 부른다(로컬 .env 에 키가 있으면 진짜 호출) — 기본 끔.
    # 거래 구조 테스트(tests/services/scms/test_deal_context.py)는 켜고 call_llm 을 대역으로 바꾼다.
    monkeypatch.setattr("agent_arch.core.config.settings.scms_deal_extract_enabled", False, raising=False)
    # 거래 문서 비전 판독도 테스트에선 끈다 — 켜 두면 그림 첨부가 실제 Gemini 를 부른다.
    monkeypatch.setattr("agent_arch.core.config.settings.scms_deal_vision_enabled", False, raising=False)
    # 메모리 v2 도 런타임 .env(MEMORY_V2_MODE=active 등)와 독립시킨다(기본 off). v2 경로를 검증하는
    # 테스트(tests/memory_v2)는 monkeypatch 로 명시 override 한다. 안 그러면 v1 도구·저장 훅 테스트가
    # v2 백엔드(DB 필요)로 새어 실패한다.
    monkeypatch.setattr("agent_arch.core.config.settings.memory_v2_mode", "off", raising=False)
    # 가명화 래퍼(app/llm/pii)도 런타임 .env 와 독립(기본 off). 래퍼를 검증하는 테스트만 shadow/enforce 로 override.
    monkeypatch.setattr("agent_arch.core.config.settings.pii_guard_mode", "off", raising=False)
    monkeypatch.setattr("agent_arch.core.config.settings.pii_guard_force_companies", "", raising=False)
    yield


@pytest.fixture(autouse=True)
def _web_search_flag_reset():
    """웹 검색 opt-in 컨텍스트(ContextVar)를 테스트마다 닫힘으로 되돌린다 — 동기 테스트에서 set 한 값이 다음 테스트로 새지 않게."""
    from agent_arch.agents.context import set_web_search_enabled
    set_web_search_enabled(None)
    yield
    set_web_search_enabled(None)


@pytest.fixture(autouse=True)
def clean_langsmith_env():
    """각 테스트 전후 LangSmith 환경변수를 원상 복구한다."""
    keys = ["LANGSMITH_TRACING", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"]
    saved = {k: os.environ.get(k) for k in keys}

    yield

    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture
def channel_modes(monkeypatch):
    """전용 채널 모드 두 개를 확장 슬롯으로 등록한다(코어는 업무 도메인을 모른다).

    원본에서 코어에 박혀 있던 'scms'·'logistics' 채널 모드를 테스트가 계속 쓸 수 있게
    extensions.mode_prompts 로 꽂는다. 등록된 모드 = 채널 스코프 모드.
    """
    from agent_arch.extensions import extensions

    monkeypatch.setattr(extensions, "mode_prompts", {
        "scms": lambda surface: "구매 전용 채널 프롬프트",
        "logistics": lambda surface: "물류 전용 채널 프롬프트",
    })
