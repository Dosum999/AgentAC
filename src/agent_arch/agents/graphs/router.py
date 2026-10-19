"""도메인 라우터 — 멀티 오케스트레이션(supervisor)의 coarse 도메인 선택.

LLM 구조화출력으로 사용자 질의가 어느 도메인(들)에 속하는지 정한다. 도구는 bind 하지
않고, ``tool_search._DOMAIN_TOOLS``(intent + 트리거 예시)를 **카드로 재사용**해 system
프롬프트에 주입하고 JSON 을 받는다. 워커(react 그래프)는 이 도메인으로 스코프된 도구만
받는다 — 2단 retrieval 을 대체하지 않고 그 **위에 도메인 coarse 선택만 얹는다(A안)**.

파싱 실패 / 매칭 도메인 0 → ``None`` 반환 → 호출자(supervisor)가 무스코프 단일 react
로 fallback(무회귀). 신뢰도 임계값 적용은 호출자가 담당한다.

멀티 도메인(domains 길이 2+)도 표현 가능(Phase2b 병렬에서 소비). 2a 는 domains[0]만 쓴다.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from agent_arch.core.config import settings

from agent_arch.agents.context import web_search_enabled
from agent_arch.agents.tools.tool_search import _DOMAIN_TOOLS

logger = logging.getLogger(__name__)

# 라우팅 대상 도메인 = tool_search 의 도메인 키(find_<domain>_tools 가 이미 쓰는 분류).
DOMAINS = tuple(_DOMAIN_TOOLS.keys())
WEB_DOMAIN = "web"


def domain_available(domain: str) -> bool:
    """이 요청에서 라우팅·상속할 수 있는 도메인인가. 요청 단위 opt-in 도메인은 플래그가 켜져야 한다.

    웹은 코어가 아는 opt-in 도메인이다. 업무 도메인의 opt-in 판정은 확장(extensions.domain_available)이 한다.
    """
    if domain == WEB_DOMAIN:
        return web_search_enabled()
    from agent_arch.extensions import domain_available as _ext_available
    return _ext_available(domain)

# supervisor 라우팅에서 제외할 도메인 — 전용 채널(앱 전용 채팅·음성 등)에서만 다루는 도메인.
# 그 채널은 supervisor 를 우회하므로, 일반 채널의 라우터가 이 도메인으로 보내면 전용 셋이
# 일반 채팅에 새어 노출된다. 관문을 꺼도(tools_config enabled:false) 도메인 목록은 _DOMAIN_TOOLS 를
# 그대로 펼치므로, 여기서 빼지 않으면 라우터가 도구 0개인 워커로 보내 도구 없이 답을 짓는다.
_SUPERVISOR_EXCLUDED_DOMAINS: frozenset[str] = frozenset()
ROUTABLE_DOMAINS = frozenset(d for d in DOMAINS if d not in _SUPERVISOR_EXCLUDED_DOMAINS)

# find_<domain>_tools 이름 → 도메인 역매핑 (직전 턴 도구에서 도메인 도출용).
_FIND_TOOL_DOMAIN = {spec[0]: domain for domain, spec in _DOMAIN_TOOLS.items()}
# 도메인 → find_<domain>_tools 이름 (워커 스코프에 자기 도메인 발견도구 포함용).
FIND_TOOL_FOR_DOMAIN = {domain: spec[0] for domain, spec in _DOMAIN_TOOLS.items()}

# 도구 이름 프리픽스 → 도메인 (로더 domain 필드가 비었을 때 폴백).
_PREFIX_DOMAIN = {
    "hr_": "hr", "calendar_": "calendar", "mail_": "mail", "mailcow_": "mail",
    "mattermost_": "talk", "drive_": "drive", "nextcloud_": "drive",
    "memory_": "memory",
    # 웹 = 브레이브 MCP 도구(brave_*). "web_" 는 과거 대화 이력의 web_search/web_fetch 도메인 상속용으로 유지.
    "brave_": "web", "web_": "web",
}


def _prefix_domain(name: str) -> Optional[str]:
    for pfx, d in _PREFIX_DOMAIN.items():
        if name.startswith(pfx):
            return d
    return None


def domain_from_tool_names(names) -> Optional[str]:
    """도구 이름 목록에서 최다 도메인을 도출한다(직전 턴 도메인 상속용). 없으면 None.

    우선순위: find_<domain>_tools 역매핑 → 로더의 domain 필드 → 이름 프리픽스 폴백.
    """
    from agent_arch.services.api_registry.loader import tool_loader

    counts: dict = {}
    for n in names or []:
        if not n:
            continue
        d = _FIND_TOOL_DOMAIN.get(n)
        if not d:
            try:
                td = tool_loader.get_by_name(n)
                d = getattr(td, "domain", None) if td is not None else None
            except Exception:
                d = None
        if not d:
            d = _prefix_domain(n)
        if d in DOMAINS:
            counts[d] = counts.get(d, 0) + 1
    if not counts:
        return None
    return max(counts, key=lambda k: counts[k])


# 도메인에 귀속돼 있지만 **어느 도메인 워커든** 들어야 하는 읽기 도구.
# mattermost_read_file 은 메신저(talk) 도메인 도구라 is_domainless_utility 에서 빠진다. 그런데
# 첨부 파일 읽기는 메신저 일이 아니라 **요청의 재료를 여는 일**이다 — "이 주문서로 등록해줘"
# 는 업무 도메인 워커로 가고, 그 워커가 첨부 PDF 를 못 열면 완수가 물리적으로 불가능하다(2026-10-08
# 실측: 업무 도메인 WRITE 워커 → `[executor] tool NOT FOUND: mattermost_read_file`
# → "파일을 읽을 수 없으니 품목을 텍스트로 알려 달라"). 모델은 정확히 호출했고 스코프에만 없었다.
# 읽기 전용이고 권한은 MCP 서버가 요청자 PAT 로 Mattermost 에 강제하므로 얹어도 side-effect 가 없다.
CROSS_DOMAIN_READERS = frozenset({"mattermost_read_file"})


def is_domainless_utility(tool) -> bool:
    """도메인에 안 묶인 횡단 유틸리티(계산기·사내 디렉터리·웹검색 등)인지 판정한다.

    supervisor 워커는 자기 도메인 도구만 스코프로 받으므로, 어느 도메인에도 귀속되지
    않는 글로벌 유틸리티 — 모놀리식 경로(_prepare_tools)에선 CORE 로 항상 바인딩되던 것 —
    가 워커에서 통째로 빠진다("김철수한테 메일" 에서 수신자 해석용 search_colleagues 등).
    이 판정으로 그런 도구를 골라 워커 스코프에 항상 얹어 모놀리식 패리티를 회복한다.

    제외:
        - 도메인 finder(find_*): 워커가 자기 도메인 것을 _with_domain_finder 로 따로 받음.
        - 도메인 귀속 도구: domain_from_tool_names 가 도메인을 돌려주면 그 도메인 워커의 것.
        - 스킬(도메인 특화, 모듈이 ``.skills.``). 별도 라우팅 도메인이
          없어 fallback(모놀리식)으로 처리되므로 워커에 얹을 필요가 없다.
    """
    name = getattr(tool, "name", None)
    if not name:
        return False
    if name in FIND_TOOL_FOR_DOMAIN.values():
        return False
    if domain_from_tool_names([name]) is not None:
        return False
    module = type(tool).__module__ or ""
    if ".skills." in module:
        return False
    return True


def select_global_tools(tools) -> list:
    """도구 목록에서 도메인 무관 글로벌 유틸리티만 골라 반환한다(워커 스코프 주입용)."""
    return [t for t in (tools or []) if is_domainless_utility(t)]


@dataclass
class RoutePlan:
    """라우터 결정 — 도메인(들) + 도메인별 조회(read)/실행(write) 분리 + 신뢰도.

    subqueries[domain] = 그 도메인의 **조회/정보(read)** 질의(없을 수 있음).
    actions[domain]    = 그 도메인의 **실행(write: 발송/등록/삭제/신청)** 질의(없을 수 있음).
    한 도메인이 조회만/실행만/둘 다 일 수 있다.
    """

    domains: List[str] = field(default_factory=list)   # 선택 도메인(순서=우선순위)
    subqueries: dict = field(default_factory=dict)      # {domain: read 질의}
    actions: dict = field(default_factory=dict)         # {domain: write 질의}
    confidence: float = 0.0
    # 직전 답변을 가리키는 후속(근거·출처·자세히·왜·그거)인가. True 면 supervisor 가 도메인을 새로
    # 추측하지 않고 직전 턴이 실행한 도메인들을 상속한다(라우터 오분류로 워커 울타리에 갇히는 것 방지).
    follow_up: bool = False
    # 웹 조사 계획 — web 도메인일 때 라우터가 함께 내는 검색어(최대 3). 워커가 돌기 전에 코드가 병렬로 먼저 실행한다
    # (supervisor → react_graph prefetch). 실사용(실측 2026-09-22): 워커 모델이 질문 문장을 그대로 한 번만 검색해
    # "7월부터 유가 추이"에 브렌트·WTI 대신 한국 소비자물가 기사로 답했다. 지침으로는 여러 질의를 만들지 않았다.
    web_queries: List[str] = field(default_factory=list)
    # 최신성 요구 — "오늘·현재·이번 주·최근" 값을 묻는지(day|week|month, 없으면 ""). 있으면 뉴스 검색을 기간 필터로 1건 더
    # 선실행한다. 실측(2026-09-23): 본문 조각 검색(기간 필터 없음)만으로는 "오늘 기준 유가"에 9월 초·5월 자료가 왔다.
    web_recency: str = ""
    # 원 페이지 열기 필요 — 답이 자주 갱신되는 특정 페이지의 최신 값(가격지수·단가·시세·환율·모집 공고·일정·통계)에 달렸나.
    # 참일 때만 선실행 뒤 상위 페이지를 코드가 자동으로 연다. 목록·조사형(업체 추천·논문·동향)에서 자동으로 열면 "Top 10"
    # 정리 블로그를 끌어와 근거로 삼아 점수가 떨어졌다(2026-09-23: 로봇 제조사 7.1→4.7, 최근 논문 4.8→3.2).
    web_open: bool = False

    def subquery_for(self, domain: str, default: str) -> str:
        """해당 도메인의 조회(read) 서브쿼리. 없으면 default(원 질의)."""
        v = self.subqueries.get(domain)
        return v if isinstance(v, str) and v.strip() else default

    def read_query(self, domain: str) -> Optional[str]:
        """조회(read) 질의 — 없으면 None."""
        v = self.subqueries.get(domain)
        return v.strip() if isinstance(v, str) and v.strip() else None

    def action_query(self, domain: str) -> Optional[str]:
        """실행(write) 질의 — 없으면 None."""
        v = self.actions.get(domain)
        return v.strip() if isinstance(v, str) and v.strip() else None


def _build_router_system_prompt(prior_domain: Optional[str] = None, prior_domains: Optional[List[str]] = None) -> str:
    """_DOMAIN_TOOLS 를 카드로 펼친 라우터 system 프롬프트. prior_domain 있으면 상속 규칙 추가."""
    lines = [
        "너는 사용자 요청을 어느 '도메인'이 처리할지 정하는 라우터다.",
        "아래 도메인 중에서만 고르고, 반드시 JSON 하나만 출력한다(설명 금지).",
        "",
        "도메인 목록:",
    ]
    for domain, (_tool, intent, examples) in _DOMAIN_TOOLS.items():
        # 관문(find_*)이 tools_config 에서 꺼진 도메인은 카드도 내지 않는다 — 카드만 있고 도구가 없으면
        # 라우터가 그 도메인으로 보내고 워커는 빈 스코프로 폴백해 한 턴을 허비한다.
        # web 은 요청 단위 opt-in(패널 토글 ∧ 회사 설정)이라 전역 설정이 아니라 요청 플래그를 본다.
        from agent_arch.extensions import extensions as _ext
        if domain == WEB_DOMAIN or domain in _ext.domain_available:
            if not domain_available(domain):
                continue
        elif not settings.get_tool_config(_tool).get("enabled", True):
            continue
        if domain in _SUPERVISOR_EXCLUDED_DOMAINS:
            continue  # 전용 채널 도메인은 supervisor router 후보에서 제외
        ex = " / ".join(examples[:3])
        lines.append(f"- {domain}: {intent}")
        lines.append(f"    예: {ex}")
    if web_search_enabled():
        # 토글을 켠 것은 "사외 정보를 찾아달라"는 **사용자의 명시적 의도**다. 지금까지 그 사실은
        # web 카드를 목록에 낼지만 정하고 분류에는 전달되지 않아, 애매한 질의에서 라우터가
        # 사내 도메인으로 흘렀다(실측: "epdm·cr polymer 단가 조회" → 사내 도메인 conf 0.70,
        # 같은 질의를 다음 턴에 다시 물으면 web conf 1.00 — 직전 맥락이 붙어야 교정됐다).
        lines += [
            "",
            "이번 요청에서 사용자가 **웹 검색을 직접 켰다**.",
            "- 사외 공개 정보로 답하는 요청(시세·단가·가격 동향, 뉴스·업계 동향, 외부 업체·제품,"
            " 법규, 일반 상식)이면 web 을 우선한다.",
            "- 다만 사내 데이터를 묻는 것이면(메일·메신저·일정·드라이브·근태·결재·전표·재고)"
            " 그대로 그 사내 도메인을 고른다 — 토글이 켜졌다고 무조건 web 이 아니다.",
            "- web 을 고르면 출력 JSON 에 \"web_queries\": [\"...\", ...] 도 넣는다 — 답에 필요한 사실을 찾을 **검색어 2~3개**."
            " 서로 다른 관점으로 나눈다(예: 가격 수치 / 원인·뉴스). 사용자 문장을 그대로 옮기지 말고 검색엔진이 알아듣는"
            " 핵심어로 쓰고, 기간이 있으면 [현재 날짜] 기준 연·월을 넣는다(예: '7월부터 유가 추이' → '브렌트유 WTI 가격 2026년 7월 9월',"
            " 'Brent crude oil price July September 2026'). 해외 시세·원자재·국제 이슈는 영어 검색어를 하나 포함한다."
            " 환율은 두 통화를 모두 적는다(원화 기준이면 KRW — '유로 원 환율', 'EUR KRW exchange rate'). 'Euro exchange rate' 처럼"
            " 한쪽만 쓰면 유로/달러 결과가 나온다(2026-09-28 실측: '유로 기준환율'에 EUR→USD 페이지를 열었다)."
            " 오늘·현재·지금·최근 값이나 최신 소식을 묻는 요청이면 \"web_recency\": \"day\"|\"week\"|\"month\" 도 넣는다"
            "(오늘 시세·오늘 기준=day, 이번 주·요즘=week, 최근 한 달·이번 달=month). 기간과 무관한 질문이면 넣지 않는다."
            " 답이 자주 갱신되는 **특정 페이지의 최신 값**(원자재 단가·가격지수·시세·환율·금리·모집 공고와 신청 기간·행사 일정·통계치)에"
            " 달려 있으면 \"web_open\": true 도 넣는다. 업체·제품 추천, 논문·뉴스 모음, 동향·비교처럼 여러 출처를 훑는 요청은 false.",
        ]
    lines += [
        "",
        "출력(JSON only):",
        '{"domains": ["<domain>", ...], '
        '"subqueries": {"<domain>": "<그 도메인의 조회/정보 요청>"}, '
        '"actions": {"<domain>": "<그 도메인의 실행 요청(발송/등록/수정/삭제/신청 등)>"}, '
        '"confidence": 0.0~1.0, "follow_up": true|false}',
        "",
        "규칙:",
        "- 한 도메인이면 domains 에 하나만. 여러 도메인이면 모두 나열.",
        "- 각 도메인 요청을 **조회(read)와 실행(write)로 분리**하라:",
        "    · 조회·확인·목록·검색·요약 → subqueries",
        "    · 발송·전송·등록·생성·수정·삭제·신청·상신 등 되돌릴 수 없는 실행 → actions",
        "    · 한 도메인에 둘 다면 둘 다 채운다"
        " (예: '받은 메일 확인하고 보내줘' → subqueries.mail=확인, actions.mail=발송).",
        "    · **첨부 파일을 읽고 그 내용으로 수정·등록·반영해 달라는 요청**도 둘 다다 — 파일 읽기는 subqueries,"
        " 수정·등록은 actions (예: '변경된 일정표야, 시간이 바뀌었어 수정해줘' → subqueries.calendar=첨부 확인,"
        " actions.calendar=일정 수정). 끝에 '수정해줘·등록해줘·반영해줘' 가 있으면 확인만으로 끝내지 마라.",
    ]
    if not settings.supervisor_preserve_original:
        # 레거시(서브쿼리 교체) 전용 보존 규칙 — 워커가 서브쿼리만 보므로 재작성이
        # 무손실이어야 한다(4b31ed7). 원문 보존(A안)에서는 워커가 원문을 직접 보므로
        # 서브쿼리는 담당 라벨이면 충분하다 — 규칙을 빼서 라우터 출력을 줄인다
        # (짧을수록 thinking 잠식에 의한 JSON 잘림 punt 여유가 는다, 실측 2026-07-31).
        lines += [
            "- 질의의 **시점·기간(어제·최근·이번주·다음주 수요일·오전 9시 등)과"
            " 범위(회사·전사·팀·특정 채널명)** 표현은 subquery 와 action 질의 **양쪽 모두에**"
            " 원문 그대로 남겨라 (워커가 검색 기간·등록 일시를 정하는 유일한 근거다)."
            " 특히 '다음주'·'이번주' 같은 주차 표현을 빼먹으면 워커가 엉뚱한 주에 일정을 잡는다.",
        ]
    lines += [
        "",
        "소스 우선순위 (사내 정보를 찾는 질문):",
        "- **대화·논의·공유·이슈 성격이면 talk 단독으로 보내라.** '어제/최근 무슨 일',",
        "  '○○ 논의 어떻게 됐지', '뭐였지/뭐였더라', '누가 공유한 ~ 찾아줘', '○○ 관련 내용'",
        "  → domains=['talk']. 회사에서 오간 이야기는 메신저에 남고, 그 글에 문서·위키 링크가",
        "  함께 들어 있어서 메신저만으로 답이 닫히는 경우가 대부분이다.",
        "- **drive 는 파일 신호가 분명할 때만** 넣어라: '드라이브', 폴더 경로('전사 공용 문서'),",
        "  파일 확장자(엑셀·xlsx·ppt·한글·hwp·pdf·워드), 파일명처럼 보이는 토큰, '양식/템플릿'.",
        "  '문서'·'자료'라는 말만으로는 drive 를 넣지 마라 — 그 말은 메신저에 공유된 링크에도",
        "  똑같이 붙는다(실측: '공유한 문서 찾아줘' 가 drive 로 새어 엉뚱한 파일을 답했다).",
        "- **'드라이브'가 논의의 주제어면 파일 신호가 아니다.** '○○ 드라이브로 전환 관련해 이야기한 것",
        "  찾아줘'처럼 **누가 무슨 얘기를 했는지** 찾는 질문은 domains=['talk'] 단독이다. '드라이브에서',",
        "  '드라이브에 있는'처럼 **위치**로 쓰였을 때만 drive 를 넣어라.",
        "- 두 성격이 **정말 다 필요한 질문**(예: '드라이브에서 찾고 메신저 논의도 정리해줘')만",
        "  domains 에 둘 다 넣는다.",
        "- 어느 도메인에도 안 맞으면 domains=[] (빈 배열), confidence 낮게.",
        "- follow_up: 이번 요청이 **직전 대화의 주제·대상을 이어받는 후속**이면 true, 새 요청이면 false. 표면 단서"
        "(근거·출처·자세히·왜·그거·다시)가 없어도 [최근 대화]로 보아 앞 얘기의 연장이면 후속이다 — 예: 앞 턴이 어떤"
        " 연구 분야였고 이번이 '○○ 관련된거 찾아봐'면 그 분야의 ○○ 이다. 후속이면 직전 도메인을 먼저 고려하되,"
        " 이번 요청이 **다른 기능·출처를 요구하면**(예: '웹 검색으로 찾아줘', '메일에서 봐줘') 그 도메인을 domains 에 쓰고"
        " confidence 를 높게 둬라 — 이번 턴에 사용자가 말한 것이 앞 턴 도메인보다 우선한다. 이번 요청만으로 도메인을"
        " 정할 수 없으면 domains=[] 로 두면 직전 도메인이 이어진다. 서브쿼리에는 앞 대화의 대상을 채워 넣어라"
        "('Kim' → 'PSE 연구자 Kim').",
    ]
    if prior_domains:
        lines += [
            f"- 직전 답변은 {', '.join(prior_domains)} 도메인에서 나왔다. follow_up=true 인데 domains 를 비우거나 확신이"
            " 낮으면 그 도메인들이 이어진다. 이번 요청이 그 도메인으로 답할 수 없는 것을 요구하면 맞는 도메인을 써라.",
        ]
    if prior_domain:
        lines += [
            f"- 직전 대화 도메인: '{prior_domain}'. 이번 요청이 앞 대화를 가리키는 참조/후속"
            "(예: '그 중', '거기서', '위의 것', '제일 큰 건')이고 명확한 다른 도메인 단서가"
            f" 없으면 domains 에 '{prior_domain}' 를 유지하라.",
        ]
    return "\n".join(lines)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_route(content: str) -> Optional[RoutePlan]:
    """LLM 응답 텍스트에서 RoutePlan 을 파싱한다. 실패/무매칭 → None."""
    if not content:
        return None
    m = _JSON_RE.search(content)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    raw = data.get("domains")
    if not isinstance(raw, list):
        return None
    domains = [d for d in raw if d in ROUTABLE_DOMAINS]  # 라우팅 허용 도메인만(환각·전용채널 도메인 제거)
    fu = data.get("follow_up")
    follow_up = fu is True or (isinstance(fu, str) and fu.strip().lower() == "true")
    if not domains and not follow_up:
        return None
    subqueries = data.get("subqueries")
    if not isinstance(subqueries, dict):
        subqueries = {}
    actions = data.get("actions")
    if not isinstance(actions, dict):
        actions = {}
    try:
        confidence = float(data.get("confidence", 0.0))
    except (ValueError, TypeError):
        confidence = 0.0
    return RoutePlan(domains=domains, subqueries=subqueries, actions=actions, confidence=confidence, follow_up=follow_up,
                     web_queries=_parse_web_queries(data.get("web_queries")),
                     web_recency=_RECENCY.get(str(data.get("web_recency") or "").strip().lower(), ""),
                     web_open=data.get("web_open") is True or str(data.get("web_open")).strip().lower() == "true")


_RECENCY = {"day": "day", "week": "week", "month": "month"}


_MAX_WEB_QUERIES = 3
_MAX_WEB_QUERY_LEN = 120


def _parse_web_queries(raw) -> List[str]:
    """검색어 목록 정규화 — 문자열만, 공백 정리, 중복 제거(대소문자 무시), 최대 3개·각 120자."""
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set = set()
    for q in raw:
        if not isinstance(q, str):
            continue
        q = " ".join(q.split())[:_MAX_WEB_QUERY_LEN]
        if q and q.casefold() not in seen:
            seen.add(q.casefold())
            out.append(q)
    return out[:_MAX_WEB_QUERIES]


async def route(llm, query: str, *, prior_domain: Optional[str] = None, prior_domains: Optional[List[str]] = None,
                history: str = "", max_tokens: int = 2048) -> Optional[RoutePlan]:
    """질의를 도메인으로 라우팅한다. LLM 실패/파싱 실패/무매칭 → None(호출자 fallback).

    prior_domain: 직전 턴 도메인. 참조/후속 질의의 도메인 상속 힌트로 프롬프트에 주입.
    history: 최근 대화 발췌(react/history). 라우터가 이번 문장만 보면 "Kim 관련된거 찾아봐"를 사내 사람 검색으로
        읽는다(실측 2026-09-16: 앞 턴이 PSE 연구였는데 talk·mail·drive·hr 4도메인으로 팬아웃). user 메시지에 넣어
        system 프롬프트 캐시는 유지한다.
    max_tokens: Gemini 는 thinking 토큰이 max_output_tokens **안에서** 소모된다 —
        512 로는 auto thinking(ROUTER_THINKING_BUDGET=-1)이 예산을 잡아먹어 JSON 이
        중간에 잘리는 punt 가 빈발했다(실측 2026-07-31: 5분에 5회, 잘린 정정 턴이
        sticky fallback→READ 로 빠져 승인도구 NOT FOUND). thinking 을 유지하려고
        한도를 넉넉히 올린다(캡이라 실사용량·비용은 그대로).
    """
    from agent_arch.llm.base import Message

    user_text = f"[최근 대화 — 이번 요청 이전]\n{history}\n\n[이번 요청]\n{query}" if history else query
    if web_search_enabled():
        # 검색어에 연·월을 넣으려면 오늘을 알아야 한다(학습 시점 연도로 쓰지 않게). user 쪽이라 system 캐시는 유지된다.
        from datetime import datetime, timedelta, timezone
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
        user_text = f"[현재 날짜] {today}\n\n{user_text}"
    messages = [
        Message(role="system", content=_build_router_system_prompt(prior_domain, prior_domains)),
        Message(role="user", content=user_text),
    ]
    try:
        # 도구 없는 도메인 분류 호출 — thinking 을 켜도 도구 체인에 영향 없다.
        # parse_route 가 프리앰블을 허용하므로 thinking 출력이 앞에 붙어도 파싱 안전.
        resp = await llm.invoke(
            messages,
            max_tokens=max_tokens,
            thinking_budget=settings.router_thinking_budget,
        )
    except Exception:
        logger.warning("[router] LLM invoke 실패 → fallback", exc_info=True)
        return None
    plan = parse_route(resp.content or "")
    if plan is None:
        logger.info("[router] 파싱/매칭 실패 → fallback (raw[:120]=%r)", (resp.content or "")[:120])
    else:
        logger.info("[router] domains=%s subqueries=%s conf=%.2f web_queries=%s",
                    plan.domains, plan.subqueries, plan.confidence, plan.web_queries)
    return plan
