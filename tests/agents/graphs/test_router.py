"""도메인 라우터 단위 테스트 (Phase 2a-i).

파싱 견고성 + route() 의 LLM 상호작용(성공/파싱실패/예외)을 mock llm 으로 검증한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.agents.graphs.router import (
    DOMAINS, RoutePlan, parse_route, route, domain_from_tool_names, _build_router_system_prompt,
)
from agent_arch.agents.tools.tool_search import _DOMAIN_TOOLS
from agent_arch.llm.base import LLMResponse


def test_domains_match_tool_search():
    """라우팅 도메인 = tool_search 도메인 키(단일 출처)."""
    assert set(DOMAINS) == set(_DOMAIN_TOOLS.keys())
    assert "hr" in DOMAINS and "talk" in DOMAINS


def test_parse_single_domain():
    p = parse_route('{"domains": ["hr"], "subqueries": {"hr": "연차 조회"}, "confidence": 0.9}')
    assert p is not None
    assert p.domains == ["hr"]
    assert p.subquery_for("hr", "orig") == "연차 조회"
    assert p.confidence == 0.9


def test_parse_multi_domain():
    p = parse_route('{"domains": ["hr", "talk"], "subqueries": {"hr":"연차","talk":"공유"}, "confidence": 0.8}')
    assert p is not None
    assert p.domains == ["hr", "talk"]


def test_parse_filters_unknown_domain():
    """환각 도메인은 걸러지고, 유효 도메인만 남는다."""
    p = parse_route('{"domains": ["hr", "quantum"], "confidence": 0.7}')
    assert p is not None and p.domains == ["hr"]


def test_parse_empty_domains_is_none():
    assert parse_route('{"domains": [], "confidence": 0.1}') is None


def test_parse_all_unknown_is_none():
    assert parse_route('{"domains": ["nope", "quantum"]}') is None


def test_parse_invalid_json_is_none():
    assert parse_route("이건 JSON 아님") is None
    assert parse_route("") is None
    assert parse_route('{"domains": not-json}') is None


def test_parse_json_embedded_in_text():
    """추론 텍스트 앞뒤에 섞여 있어도 JSON 블록을 추출한다."""
    p = parse_route('생각: HR 같아요.\n{"domains": ["hr"], "confidence": 0.6}\n끝')
    assert p is not None and p.domains == ["hr"]


def test_subquery_for_default_when_missing():
    p = parse_route('{"domains": ["hr"], "confidence": 0.9}')
    assert p.subquery_for("hr", "원질의") == "원질의"


def _llm_returning(content):
    llm = MagicMock()
    llm.invoke = AsyncMock(return_value=LLMResponse(content=content, finish_reason="stop"))
    return llm


@pytest.mark.asyncio
async def test_route_success():
    llm = _llm_returning('{"domains": ["calendar"], "subqueries": {"calendar":"내일 회의"}, "confidence": 0.95}')
    plan = await route(llm, "내일 오후 2시 회의 잡아줘")
    assert plan is not None and plan.domains == ["calendar"]


@pytest.mark.asyncio
async def test_route_parse_fail_returns_none():
    llm = _llm_returning("죄송하지만 JSON 못 만들겠어요")
    assert await route(llm, "뭔가") is None


@pytest.mark.asyncio
async def test_route_llm_exception_returns_none():
    llm = MagicMock()
    llm.invoke = AsyncMock(side_effect=RuntimeError("503"))
    assert await route(llm, "뭔가") is None


# ── 후속턴 라우팅 (2d) ────────────────────────────────────────────────────────

def test_domain_from_tool_names_by_prefix():
    assert domain_from_tool_names(["hr_leave_balance", "hr_my_payslips"]) == "hr"
    assert domain_from_tool_names(["calendar_create_event"]) == "calendar"
    assert domain_from_tool_names(["mattermost_post_message"]) == "talk"


def test_domain_from_find_tool():
    assert domain_from_tool_names([_DOMAIN_TOOLS["hr"][0]]) == "hr"       # find_hr_tools → hr


def test_domain_from_tool_names_dominant():
    assert domain_from_tool_names(["hr_a", "hr_b", "calendar_x"]) == "hr"


def test_domain_from_tool_names_empty_or_unknown():
    assert domain_from_tool_names([]) is None
    assert domain_from_tool_names(["totally_unknown_thing"]) is None


def test_prior_domain_injected_in_prompt():
    p = _build_router_system_prompt(prior_domain="hr")
    assert "hr" in p and "참조/후속" in p


def test_prompt_omits_preservation_rule_when_preserve_original(monkeypatch):
    """A안(원문 보존) ON: 보존 규칙 제외 — 워커가 원문을 직접 보므로 서브쿼리는
    담당 라벨이면 충분하다. 라우터 출력도 짧아져 JSON 잘림(punt) 여유가 는다."""
    import agent_arch.agents.graphs.router as _r
    monkeypatch.setattr(_r.settings, "supervisor_preserve_original", True)
    p = _build_router_system_prompt()
    assert "원문 그대로 남겨라" not in p
    assert "domains=[]" in p                      # punt 출력 규약은 보존 규칙과 무관 — 유지


def test_prompt_keeps_preservation_rule_when_legacy(monkeypatch):
    """레거시(교체) OFF: 보존 규칙 유지 — 서브쿼리가 워커의 유일한 입력이라
    무손실 재작성이 필요하다(4b31ed7). 기본 ON 전환 전까지 레거시 보호를 잃으면 안 된다."""
    import agent_arch.agents.graphs.router as _r
    monkeypatch.setattr(_r.settings, "supervisor_preserve_original", False)
    p = _build_router_system_prompt()
    assert "원문 그대로 남겨라" in p
    assert "다음주" in p


# ── 글로벌(도메인 무관) 유틸리티 분류 ─────────────────────────────────────────
from agent_arch.agents.graphs.router import is_domainless_utility, select_global_tools


class _Tool:
    def __init__(self, name):
        self.name = name


class _SkillTool(_Tool):
    pass


# 스킬 도구는 모듈 경로에 .skills. 가 있어 글로벌에서 제외됨을 재현.
_SkillTool.__module__ = "agent_arch.agents.tools.skills.optimization.skills"


def test_global_utility_true_for_domainless():
    for name in ("calculator", "search_colleagues", "list_departments"):
        assert is_domainless_utility(_Tool(name)) is True, name


def test_global_utility_false_for_finder_domain_skill():
    assert is_domainless_utility(_Tool("find_mail_tools")) is False          # 도메인 finder
    assert is_domainless_utility(_Tool("mail_send_message")) is False        # 도메인 귀속(prefix)
    assert is_domainless_utility(_Tool("hr_request_leave")) is False         # 도메인 귀속(prefix)
    assert is_domainless_utility(_Tool("web_search")) is False               # 웹 도메인(prefix web_)
    assert is_domainless_utility(_Tool("brave_llm_context")) is False          # 웹 도메인(prefix brave_ = 브레이브 MCP)
    assert is_domainless_utility(_SkillTool("optimization_solve")) is False  # 스킬(.skills. 모듈)


def test_select_global_tools_filters():
    tools = [_Tool("calculator"), _Tool("find_hr_tools"), _Tool("hr_leave"),
             _SkillTool("optimization_solve"), _Tool("search_colleagues")]
    assert sorted(t.name for t in select_global_tools(tools)) == ["calculator", "search_colleagues"]
    assert "참조/후속" not in _build_router_system_prompt()  # prior 없으면 규칙 없음



def test_parse_follow_up_flag_and_domainless_follow_up_survives():
    p = parse_route('{"domains": ["web"], "confidence": 0.9, "follow_up": true}')
    assert p is not None and p.follow_up is True and p.domains == ["web"]
    p = parse_route('{"domains": [], "confidence": 0.3, "follow_up": "true"}')
    assert p is not None and p.follow_up is True and p.domains == []           # 후속인데 도메인을 모르면 상속용 빈 계획
    assert parse_route('{"domains": [], "confidence": 0.3, "follow_up": false}') is None
    assert parse_route('{"domains": ["hr"], "confidence": 0.9}').follow_up is False


def test_router_prompt_mentions_follow_up_and_prior_domains():
    text = _build_router_system_prompt(prior_domain="web", prior_domains=["web", "talk"])
    assert '"follow_up": true|false' in text and "follow_up:" in text
    assert "web, talk 도메인에서 나왔다" in text and "직전 대화 도메인: 'web'" in text
    assert "도메인에서 나왔다" not in _build_router_system_prompt()


# ── 라우터도 최근 대화를 본다(react/history) ─────────────────────────────────────

@pytest.mark.asyncio
async def test_route_puts_recent_history_in_user_message_and_prompt_has_context_rule():
    llm = MagicMock()
    llm.invoke = AsyncMock(return_value=LLMResponse(content='{"domains": ["web"], "subqueries": {"web": "PSE 연구자 Kim"}, "confidence": 0.9, "follow_up": true}', finish_reason="stop"))
    hist = "사용자: production systems engineering 파라미터 추정 연구 있어?\n어시스턴트: 선형 응답 기반 추정 등 4건이 있습니다."
    plan = await route(llm, "Kim 관련된거 찾아봐", prior_domains=["web"], history=hist)
    sent = llm.invoke.call_args[0][0]
    assert sent[1].role == "user" and sent[1].content.startswith("[최근 대화 — 이번 요청 이전]\n사용자: production systems")
    assert sent[1].content.endswith("[이번 요청]\nKim 관련된거 찾아봐")
    assert "직전 대화의 주제·대상을 이어받는 후속" in sent[0].content and "표면 단서" in sent[0].content     # 규칙은 system 에(캐시 유지)
    assert plan.follow_up is True and plan.subqueries == {"web": "PSE 연구자 Kim"}
    # 이력이 없으면 종전처럼 문장만
    await route(llm, "오늘 날씨", history="")
    assert llm.invoke.call_args[0][0][1].content == "오늘 날씨"


def test_web_card_only_when_request_opted_in():
    from agent_arch.agents.context import set_web_search_enabled
    from agent_arch.agents.graphs.router import domain_available
    assert "- web:" not in _build_router_system_prompt() and not domain_available("web") and domain_available("talk")
    set_web_search_enabled(True)
    assert "- web:" in _build_router_system_prompt() and domain_available("web")
# ── 소스 우선순위 규칙 (2026-09-15) ──────────────────────────────────────────

def test_router_prompt_sends_conversational_questions_to_talk_alone():
    """대화·논의·공유 성격은 메신저 단독 — 사내에서 오간 이야기는 메신저에 남는다.

    실측: "공유한 문서 찾아줘" 가 drive 로도 라우팅돼 엉뚱한 hwp 가 답변 본문을 차지했다.
    """
    p = " ".join(_build_router_system_prompt().split())
    assert "talk 단독" in p
    assert "누가 공유한" in p and "뭐였지" in p


def test_router_prompt_requires_explicit_file_signal_for_drive():
    """'문서'·'자료' 만으로 drive 를 넣으면 대화형 질문이 파일 검색으로 샌다."""
    p = " ".join(_build_router_system_prompt().split())
    assert "파일 신호가 분명할 때만" in p
    for signal in ("드라이브", "확장자", "양식/템플릿"):
        assert signal in p
    assert "'문서'·'자료'라는 말만으로는 drive 를 넣지 마라" in p


def test_parse_route_web_queries_normalized():
    """웹 조사 계획 — 검색어 최대 3개, 공백 정리·중복 제거·문자열만(실사용 2026-09-22: 질문 문장 그대로 1회 검색)."""
    from agent_arch.agents.graphs.router import parse_route
    raw = ('{"domains": ["web"], "subqueries": {"web": "유가"}, "confidence": 0.9, '
           '"web_queries": ["브렌트유  가격 2026년 7월", "Brent crude price 2026", "brent crude PRICE 2026", 3, "", "x", "y"]}')
    plan = parse_route(raw)
    assert plan.web_queries == ["브렌트유 가격 2026년 7월", "Brent crude price 2026", "x"]
    assert parse_route('{"domains": ["web"], "confidence": 0.9}').web_queries == []


def test_web_prefetch_builds_context_calls_with_english_locale():
    from agent_arch.agents.graphs import supervisor as S
    from agent_arch.agents.graphs.router import RoutePlan
    plan = RoutePlan(domains=["web"], web_queries=["브렌트유 가격 2026년 9월", "Brent crude price September 2026"])
    calls = S._web_prefetch(plan, "web")
    assert [c["name"] for c in calls] == ["brave_llm_context"] * 2
    assert calls[0]["arguments"] == {"query": "브렌트유 가격 2026년 9월", "maximum_number_of_tokens": 4096}
    assert calls[1]["arguments"]["search_lang"] == "en" and calls[1]["arguments"]["country"] == "US"
    assert S._web_prefetch(plan, "mail") == [] and S._web_prefetch(RoutePlan(domains=["web"]), "web") == []


def test_web_recency_parsed_and_prefetches_fresh_news():
    from agent_arch.agents.graphs import supervisor as S
    from agent_arch.agents.graphs.router import parse_route
    plan = parse_route('{"domains": ["web"], "confidence": 0.9, "web_queries": ["유가 오늘"], "web_recency": "DAY"}')
    assert plan.web_recency == "day"
    calls = S._web_prefetch(plan, "web")
    assert calls[0] == {"name": "brave_news_search", "arguments": {"query": "유가 오늘", "freshness": "pd"}}
    assert calls[1]["name"] == "brave_llm_context"
    assert parse_route('{"domains": ["web"], "confidence": 0.9, "web_recency": "yesterday"}').web_recency == ""


def test_web_open_flag_parsed():
    """자동 페이지 열기는 특정 페이지의 최신 값 질문만 — 목록형에서 정리 블로그를 끌어와 점수가 떨어졌다(2026-09-23)."""
    from agent_arch.agents.graphs.router import parse_route
    assert parse_route('{"domains": ["web"], "confidence": 0.9, "web_open": true}').web_open is True
    assert parse_route('{"domains": ["web"], "confidence": 0.9, "web_open": "true"}').web_open is True
    assert parse_route('{"domains": ["web"], "confidence": 0.9}').web_open is False


def test_router_prompt_requires_currency_pair_in_fx_queries():
    """'유로 기준환율' 에 영어 검색어가 'Euro exchange rate today' → EUR/USD 페이지를 열었다(2026-09-28)."""
    from agent_arch.agents import context as C
    from agent_arch.agents.graphs.router import _build_router_system_prompt
    C.set_web_search_enabled(True)
    p = _build_router_system_prompt(None, None)
    assert "두 통화를 모두 적는다" in p and "EUR KRW" in p
