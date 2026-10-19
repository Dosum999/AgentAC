"""멀티 오케스트레이션 supervisor — 라우터 + 도메인 스코프 워커(react 그래프) + synthesizer.

흐름:
    route(질의) → RoutePlan(domains, subqueries, confidence)
      - 무매칭/저신뢰                → 무스코프 단일 react 로 fallback(무회귀)
      - 단일 도메인                  → 그 도메인 스코프 워커 1개
      - 멀티 도메인                  → 워커 병렬(asyncio.gather) → synthesizer 1콜 합성

워커 = Phase1 react 그래프(run_react_graph/stream_react_graph)에 **도메인 스코프 도구**를
주입한 것. 라우터는 2단 retrieval 을 대체하지 않고 도메인 coarse 선택만 얹는다(A안).

병렬 정확성 3요건:
    1) 워커마다 **fresh llm 인스턴스**(make_worker_llm_fn) — bind_tools 레이스 방지.
    2) 워커마다 **messages 복사**(list(messages)) — 병렬 append 오염 방지.
    3) **ContextVar 격리** — asyncio.gather 가 각 워커를 Task 로 감싸며 context 를 copy 해
       _discovered/user_context/profile_context 가 워커 간 격리(실측: test_supervisor).

의존성 주입(그래프와 decouple + 테스트 용이):
    resolve_tools_fn(subquery, domain) -> list[BaseTool]   (도메인 스코프 retrieval+게이트)
    make_worker_llm_fn() -> BaseLLM                        (워커용 fresh llm)
    fallback_fn() -> LoopResult                            (run 무스코프 단일 react)
    fallback_stream_fn() -> async iterator[ProgressEvent|LoopResult]  (stream fallback)

라우팅 결정은 LangSmith 현재 run 메타(route)로 남겨 관찰 가능하게 한다.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import replace
from typing import List, Optional, Tuple

from langsmith import get_current_run_tree

from agent_arch.core.config import settings
from agent_arch.agents.graphs.router import (
    CROSS_DOMAIN_READERS,
    SCHEDULE_DOMAIN,
    SCMS_ORDER_DOMAIN,
    ROUTABLE_DOMAINS,
    FIND_TOOL_FOR_DOMAIN,
    RoutePlan,
    domain_from_tool_names,
    route,
    domain_available,
)
from agent_arch.agents.tools.registry import ToolRegistry
from agent_arch.agents.hitl.policy import requires_approval
from agent_arch.agents.graphs.react_graph import run_react_graph, stream_react_graph
from agent_arch.utils.language import DEFAULT_LANGUAGE
from agent_arch.agents.react.history import recent_history
from agent_arch.agents.react.messenger_links import apply_file_links, file_refs
from agent_arch.agents.react.provenance import append_sources_footer, sources_footer
from agent_arch.agents.react.types import LoopResult, ProgressEvent, merge_usage
from agent_arch.agents.hitl.answer_guard import (
    build_file_link_map,
    find_unsourced_person_names,
    has_unnamed_author,
    redact_fabricated_permalinks,
    redact_unsourced_names,
    requester_context,
)
from agent_arch.utils.sanitize import sanitize_tool_names
from agent_arch.agents.react.link_policy import TurnLinkPolicy

logger = logging.getLogger(__name__)


def _attach_route_meta(plan: Optional[RoutePlan], *, fallback: bool, chosen: Optional[list] = None) -> None:
    """라우팅 결정을 LangSmith 현재 run 메타/태그로 남긴다(관찰용)."""
    try:
        run = get_current_run_tree()
        if run is None:
            return
        run.add_metadata({"route": {
            "domains": (plan.domains if plan else []),
            "confidence": (round(plan.confidence, 3) if plan else None),
            "fallback": fallback,
            "executed": chosen or [],
        }})
        tag = "route:fallback" if fallback else f"route:{'+'.join(chosen or [])}"
        run.add_tags([tag])
    except Exception:
        logger.debug("[supervisor] route 메타 부착 실패(무시)", exc_info=True)


def _pending_draft_stashes() -> List[dict]:
    """실행으로 태워야 할 초안 상태 **전부** — 거절된 초안 + 수집 중인 초안.

    둘 다 ``{"tool_name": ...}`` 모양이라 도메인 유도 코드가 같다. 거절은 값을 되쓰기 위한
    것이고, 수집 중은 되물을 값이 남아 초안을 아직 못 만든 상태다. 둘 중 하나라도 있으면
    그 턴은 조회가 아니라 실행이다.

    ⚠️ 하나만 고르면(예전 `rejected or collecting`) **도메인이 가려진다** — 캘린더 초안을
    거절한 상태에서 경비 수집이 진행 중이면 캘린더 stash 가 이겨 경비 승격이 걸리지 않았다.
    그래서 목록으로 돌려주고 호출자가 **요청 도메인과 일치하는 것**을 고른다.
    """
    from agent_arch.agents.context import (
        get_collecting_draft_context,
        get_rejected_draft_context,
    )

    return [s for s in (get_rejected_draft_context(), get_collecting_draft_context())
            if isinstance(s, dict) and s.get("tool_name")]


def _stash_domains() -> List[str]:
    """진행 중 초안들이 가리키는 도메인 목록(중복·빈값 제거, 순서 보존)."""
    out: List[str] = []
    for stash in _pending_draft_stashes():
        d = domain_from_tool_names([stash.get("tool_name")])
        if d and d not in out:
            out.append(d)
    return out


def _has_pending_draft(domain: Optional[str]) -> bool:
    """그 도메인에 진행 중(거절/수집) 초안이 있는가 — 승격 판정의 유일한 근거."""
    return bool(domain) and domain in _stash_domains()


# 실행을 분명히 시키는 말(명령형) — "수정해줘", "등록해 주세요", "반영해줄래", "바꿔줘" …
_IMPERATIVE_WRITE = re.compile(
    r"(수정|등록|반영|변경|정정|발주|생성|입력)\s*(을|를)?\s*(해|하여|해서)?\s*(줘|주세요|줄래|주라|주십시오|줄\s*수)"
    r"|(바꿔|고쳐|올려|넣어)\s*(줘|주세요|줄래|주라)"
    # 고객 요청을 전하는 말 — "납기도 1주 당겨 달라고 합니다", "5,000개로 늘려 달랍니다"(시연 대본 2026-10-12)
    r"|(늘려|당겨|줄여|미뤄|바꿔|올려|변경해|수정해)\s*(달라|달랍|주라고|달래)"
)


def _promote_imperative_to_action(plan: "RoutePlan", user_input: str) -> "RoutePlan":
    """한 도메인을 조회로만 분류했는데 문장이 실행을 분명히 시키면 실행도 함께 돌린다(조회는 그대로).

    2026-10-12 운영: 변경 발주서 PDF 를 붙이고 "…5000개로 바뀌고 납기일도 일주일 앞당겨졌어 수정해줘" 라고 했는데
    라우터가 '첨부 PDF 변경 내용 확인' 조회로만 분류했다 — 조회 워커에는 승인 도구(수주서 수정)가 없어 "ERP 에서 직접
    반영하라" 고 답했다. 첨부가 붙으면 라우터가 '읽기' 에 끌려 끝의 '수정해줘' 를 실행으로 넘기지 않는다.
    실행은 승인 카드까지만 가므로(confirm 게이트) 잘못 올려도 카드가 뜰 뿐이다. 조회 결과는 실행 워커에 인계된다.
    """
    if plan is None or len(plan.domains) != 1 or plan.actions or not _IMPERATIVE_WRITE.search(user_input or ""):
        return plan
    d = plan.domains[0]
    logger.info("[supervisor] 명령형 실행 요청 → %s 실행 질의 추가(라우터는 조회만)", d)
    return RoutePlan(domains=plan.domains, subqueries=dict(plan.subqueries), actions={d: user_input},
                     confidence=plan.confidence, follow_up=plan.follow_up, **_web_plan(plan))


# 드라이브 엑셀을 물류로 옮기라는 말 — "물류 마이그레이션 폴더 엑셀파일들 물류로 마이그레이션 해줘"(2026-10-13).
# ⚠️ '마이그레이션' 만으로 보지 말 것 — 폴더 이름이 '물류 마이그레이션' 이라 "물류 마이그레이션 폴더에 뭐 있어?"(조회)도
#    걸린다. 실행 말('해줘·해 주세요·진행·시켜·부탁')이 붙을 때만.
_MIGRATE_WORD = re.compile(
    r"(마이그레이션|마이그레이트|이관|migrat\w*)\s*(을|를)?\s*(좀\s*)?(해|하여|진행|시켜|부탁|하자|할래)", re.IGNORECASE)
_MIGRATE_FILE = re.compile(r"엑셀|xlsx|파일|폴더|드라이브", re.IGNORECASE)


def _route_migration_to_logistics(plan: "RoutePlan", user_input: str) -> "RoutePlan":
    """드라이브 엑셀 → 물류 이관 요청이면 logistics **실행** 하나로 보낸다(drive 조회는 뺀다).

    '폴더'·'엑셀파일' 이 파일 신호라 라우터가 drive 조회로 보내거나(조회 워커엔 이관 도구가 없다) drive 조회 +
    logistics 실행으로 나눈다. 이관 도구(logistics_migrate_drive_excel)는 **파일을 직접 받으므로** drive 워커가
    읽을 필요가 없고, 병렬로 돌면 파일 목록 답이 이관 카드와 섞인다. 실행은 승인 카드까지만 간다(confirm 게이트).
    """
    text = user_input or ""
    if not (_MIGRATE_WORD.search(text) and _MIGRATE_FILE.search(text)) or not domain_available("logistics"):
        return plan
    if plan is not None and plan.domains == ["logistics"] and plan.action_query("logistics"):
        return plan
    others = [d for d in (plan.domains if plan else []) if d not in ("drive", "logistics")]
    logger.info("[supervisor] 물류 이관 요청 → logistics 실행 (router=%s)", plan.domains if plan else None)
    return RoutePlan(
        domains=["logistics", *others],
        subqueries={d: q for d, q in (plan.subqueries.items() if plan else []) if d in others},
        actions={"logistics": text, **{d: q for d, q in (plan.actions.items() if plan else []) if d in others}},
        confidence=1.0, follow_up=bool(plan and plan.follow_up), **_web_plan(plan),
    )


def _bias_to_action_for_pending_draft(
    plan: RoutePlan, prior_domain: Optional[str], user_input: str
) -> RoutePlan:
    """초안이 **거절됐거나 수집 중**이면 그 도메인 질의를 read → action 으로 승격한 사본을 낸다.

    거절 후 정정 턴("9시 말고 10시로 변경해줘")은 '변경'이라는 말 때문에 라우터가 자주
    subqueries(조회)로 분류한다. 그런데 READ 서브런은 requires_approval 도구를 전부 제거하므로
    (_run_worker) 초안을 다시 만들 수단이 없고, 모델은 존재하지도 않는 일정을 조회하다
    "취소되어 찾을 수 없습니다"로 대화를 끝낸다. 그래서 WRITE 로 태운다.

    조건을 `prior_was_action`(직전이 승인도구 턴) 이 아니라 **거절 stash 존재**로 좁힌 이유:
    승인까지 끝난 뒤의 정상 후속 조회("방금 만든 일정 언제였지?")도 prior_was_action 이라
    WRITE 로 밀려 엉뚱한 초안 카드가 뜬다. 거절 stash 는 정확히 이 시나리오에만 존재한다.

    승격 시 같은 도메인의 subquery 는 **제거**한다. 남겨 두면 read/write 서브런이 함께 돌아
    "일정을 찾을 수 없습니다" + 새 초안 카드가 한 응답에 같이 나온다(_combine_worker 가 read
    요약 뒤에 초안 안내를 덧붙이는 구조라 서로 모순된 답이 된다).

    오분류하더라도 실행은 confirm=False 강제 게이트가 막으므로 최악이 초안 카드다(안전 불변).
    """
    # 도메인은 **stash 의 도구 이름**에서 유도한다. prior_domain 은 tool_ctx(TTL·덮어쓰기 있음)
    # 파생이라 없을 수 있는데, stash 는 도구명을 직접 들고 있어 항상 정확하다.
    # 라우터가 고른 도메인 중 진행 중 초안이 있는 것만 승격한다(다른 도메인은 손대지 않는다).
    domain = next((d for d in (plan.domains or []) if _has_pending_draft(d)), None)
    if not domain:
        return plan
    if plan.action_query(domain):
        return plan  # 이미 실행으로 분류됨 — 손대지 않는다.
    actions = {**plan.actions, domain: plan.read_query(domain) or user_input}
    subqueries = {k: v for k, v in plan.subqueries.items() if k != domain}
    logger.info(
        "[supervisor] 진행 중 초안(거절/수집) → domain=%s read를 action 으로 승격", domain
    )
    return replace(plan, actions=actions, subqueries=subqueries)


def _followup_override_confidence() -> float:
    """후속 질의에서 라우터 판단이 직전 도메인 상속을 이기는 확신도(ROUTER_FOLLOWUP_OVERRIDE_CONFIDENCE)."""
    try:
        return float(getattr(settings, "router_followup_override_confidence", 0.85))
    except (TypeError, ValueError):
        return 0.85


async def _decide(router_llm, user_input: str, confidence_threshold: float,
                  prior_domain: Optional[str] = None, *,
                  prior_was_action: bool = False,
                  prior_domains: Optional[List[str]] = None,
                  history: str = "") -> Tuple[Optional[RoutePlan], bool]:
    """route() 실행 + 임계값 적용 + sticky fallback. ok=False면 호출자가 무스코프 fallback.

    prior_domain 이 있으면 (1) route() 프롬프트에 상속 힌트로 주입하고, (2) 라우터가
    그래도 punt(None/빈/저신뢰)하면 그 도메인으로 스코프를 유지한다(sticky) — 참조/후속
    질의가 도메인 키워드 0개라 라우터가 놓치는 경우 무스코프로 떨어지지 않게.

    prior_was_action: 직전 턴에 승인(실행) 도구가 쓰였는지. sticky fallback 은 도메인만
    복원하고 read/action 구분을 잃어 정정 턴("결제수단 바꿔줘")이 기본값(전체 조회)으로
    READ 서브런에 떨어졌고, READ 는 requires_approval 필터로 승인도구를 제거해 create
    NOT FOUND → LLM 이 상신한 척 환각하는 사고가 반복됐다(실측 2026-07-27~31). 직전이
    action 이었으면 punt 된 후속 발화도 action 으로 보고 WRITE 서브런에 태운다 — 실행은
    confirm=False 강제 게이트가 막으므로 오분류여도 초안 카드가 뜰 뿐 안전 불변.

    단 위 sticky 보정은 **punt 했을 때만** 걸린다. "9시 말고 10시로 변경해줘" 같은 정정 턴은
    유창한 문장이라 라우터가 높은 신뢰도로 답해 아래 조기 return 을 타고, 그 결과 READ 로
    분류되면 같은 사고가 재현된다 — 그래서 진행 중 초안이 있는 턴에는 _bias_to_action_for_pending_draft 를
    한 번 더 통과시킨다.
    """
    plan = await route(router_llm, user_input, prior_domain=prior_domain, prior_domains=prior_domains, history=history)
    # 드라이브 엑셀 → 물류 이관은 직전 턴(드라이브 목록 조회)을 잇지 않는다 — 상속·저신뢰 punt 보다 먼저 본다.
    migrated = _route_migration_to_logistics(plan, user_input)
    if migrated is not plan:
        return migrated, True
    # 메신저 발주요청은 요청 단위로만 생기는 도메인이라 직전 턴에는 없었다 — 상속으로 덮으면 그 턴에 없던
    # 선택지를 다시 빼앗는다(실측 2026-09-23: 같은 스레드에서 앞선 "발주요청해줘" 가 전자결재로 샜고,
    # 기능을 켠 뒤엔 라우터가 scms_order 1.00 을 냈는데 follow-up 상속이 eapproval 로 되돌렸다).
    # 이 도메인의 도구는 승인 도구 하나라 조회로 태우면 도구가 빠진다 — 늘 실행 질의로 넘긴다.
    # 예약(schedule)도 같다: "평일 9시 40분에 환율 알려줘" 는 문장이 조회처럼 보여 READ 로 가기 쉬운데, READ 는
    # 승인 도구(schedule_create)를 빼므로 예약을 만들 수단이 없다. 목록·삭제는 승인 도구가 아니라 실행 질의에도 있다.
    forced = [d for d in _ALWAYS_ACTION_DOMAINS
              if plan is not None and d in plan.domains and domain_available(d)]
    if forced and plan.confidence >= confidence_threshold:
        acts = dict(plan.actions)
        for d in forced:
            acts.setdefault(d, plan.read_query(d) or user_input)
        subq = {d: q for d, q in plan.subqueries.items() if d not in forced}
        return RoutePlan(domains=plan.domains, subqueries=subq, actions=acts,
                         confidence=plan.confidence, follow_up=plan.follow_up, **_web_plan(plan)), True
    # 후속 질의의 직전 도메인은 **힌트이지 덮어쓰기가 아니다** — 라우터가 이번 요청의 도메인을 확신하면 그 판단을 쓰고,
    # 도메인을 못 정했거나 확신이 낮을 때만 직전 턴이 실행한 도메인들로 채운다(대화 상태 추적의 "이번 턴에 말하지
    # 않았을 때만 이어받기"와 같은 원칙; Dialogflow 컨텍스트·LangGraph supervisor 도 앞 맥락은 매칭 가능성만 높인다).
    # - 채우기가 필요한 이유(2026-09-15): "근거가 뭐야"를 라우터가 talk 로 오분류(0.70)해 워커가 가이드 검색으로 샜다.
    # - 덮어쓰면 안 되는 이유(2026-09-29 실측 사용자): 웹을 끈 채 물은 턴이 memory 로 처리된 뒤, 웹을 켜고
    #   "웹 검색으로 오늘 기준 환율"을 물었는데 라우터 web(0.95~1.00)을 상속이 세 번 연속 memory 로 덮어
    #   메모리 워커가 "웹 검색을 쓸 수 없다"고 답했다. 두 실측의 확신도가 후속 전용 기준(기본 0.85)에서 갈린다.
    if plan is not None and plan.follow_up and not (
            any(domain_available(d) for d in plan.domains) and plan.confidence >= _followup_override_confidence()):
        inherit = [d for d in (prior_domains or ([prior_domain] if prior_domain else []))
                   if d in ROUTABLE_DOMAINS and domain_available(d)]   # 웹이 꺼진 요청은 직전 web 턴을 잇지 않는다
        if inherit:
            subq, acts = dict(plan.subqueries), dict(plan.actions)
            for d in inherit:
                if plan.read_query(d) or plan.action_query(d):
                    continue
                if len(inherit) == 1 and (prior_was_action or _has_pending_draft(d)):
                    acts[d] = user_input
                else:
                    subq[d] = user_input
            logger.info("[supervisor] follow-up → 직전 도메인 상속 %s (router=%s conf=%.2f)",
                        inherit, plan.domains, plan.confidence)
            # 라우터의 웹 조사 계획(검색어·최신성·원 페이지 필요)도 이어받는다 — 버리면 선실행·충분성 루프가 통째로 빠진다
            # (실측 2026-09-28: 후속 "오늘 유로 기준환율"이 검색 없이 이전 답의 씨티은행 URL 만 열려다 막혔다).
            return RoutePlan(domains=inherit, subqueries=subq, actions=acts, confidence=1.0, follow_up=True,
                             **_web_plan(plan, user_input if "web" in inherit else "")), True
    if plan is not None and plan.domains and plan.confidence >= confidence_threshold:
        plan = _promote_imperative_to_action(plan, user_input)
        return _bias_to_action_for_pending_draft(plan, prior_domain, user_input), True
    # punt — 도메인은 sticky(직전 턴 도구)로 복원하고, 없으면 진행 중 초안 stash 로 복원한다
    # (tool_ctx 는 TTL·덮어쓰기가 있어 비어 있을 수 있다).
    domain = (prior_domain if prior_domain and prior_domain in ROUTABLE_DOMAINS and domain_available(prior_domain) else None)
    domain = domain or next(iter(_stash_domains()), None)
    if domain and domain in ROUTABLE_DOMAINS:
        # 진행 중 초안이 **같은 도메인**이면 짧은 답변 턴("1번")은 조회가 아니라 실행이다 —
        # READ 로 떨어지면 승인 도구가 필터로 빠져 초안을 만들 수단이 없다(실측 2026-08-17).
        # 도메인이 다르면 승격하지 않는다(경비 수집 중의 캘린더 후속턴 등 오염 방지).
        is_action = prior_was_action or _has_pending_draft(domain)
        mode = "action" if is_action else "read"
        logger.info(
            "[supervisor] sticky fallback → prior_domain=%s mode=%s", domain, mode
        )
        query_map = {domain: user_input}
        return RoutePlan(
            domains=[domain],
            actions=query_map if is_action else {},
            confidence=1.0,
            **(_web_plan(None, user_input) if domain == "web" else {}),
        ), True
    return plan, False


def _web_plan(plan: Optional[RoutePlan], fallback_query: str = "") -> dict:
    """RoutePlan 을 새로 만들 때 옮겨야 할 웹 조사 계획. 라우터 검색어가 없으면 사용자 문장을 검색어로(선실행·충분성 루프가 돌게)."""
    queries = list(getattr(plan, "web_queries", None) or [])
    if not queries and fallback_query.strip():
        queries = [" ".join(fallback_query.split())[:120]]
    return {"web_queries": queries, "web_recency": getattr(plan, "web_recency", "") or "",
            "web_open": bool(getattr(plan, "web_open", False))}


def _current_conversation_id() -> Optional[str]:
    try:
        from agent_arch.agents.context import get_conversation_id
        return get_conversation_id()
    except Exception:
        return None


async def _load_prior_domains(conversation_id: Optional[str]) -> List[str]:
    """직전 턴이 실행한 도메인들(follow_up 상속용). 없으면 빈 리스트."""
    if not conversation_id:
        return []
    try:
        from agent_arch.memory.conversation import load_last_route
        return await load_last_route(conversation_id)
    except Exception:
        return []


async def _save_route(conversation_id: Optional[str], domains: List[str]) -> None:
    if not conversation_id:
        return
    try:
        from agent_arch.memory.conversation import save_last_route
        await save_last_route(conversation_id, domains)
    except Exception:
        logger.debug("[supervisor] last_route 저장 실패(무시)", exc_info=True)


def _bound_tools_guidance(tools) -> str:
    """워커에 바인딩된 **retrieved 도구**의 prompt_guidance 를 모아 지침에 덧붙인다.

    system 프롬프트의 도구 지침 문단은 core 도구만 싣는다(system._collect_tool_guidance). 그래서 retrieved
    티어 도구(brave_* 웹 도구 등)의 사용 원칙이 어느 프롬프트에도 들어가지 않았다(실측 2026-09-15: 웹 답변에
    출처 미표기). 워커가 실제로 든 도구의 지침만 붙이므로 프롬프트가 도메인 밖 규칙으로 부풀지 않는다.
    """
    try:
        from agent_arch.agents.tools.exposure import Exposure, resolve_exposure
    except Exception:
        return ""
    parts: List[str] = []
    seen: set = set()
    for t in tools or []:
        g = (getattr(t, "prompt_guidance", "") or "").strip()
        if not g or g in seen:
            continue
        try:
            if resolve_exposure(t) is Exposure.CORE:
                continue  # core 도구 지침은 system 프롬프트가 이미 싣는다
        except Exception:
            pass
        seen.add(g)
        parts.append(g)
    return ("\n\n" + "\n\n".join(parts)) if parts else ""


# 도메인별 추가 지침 — 도메인 격리 원칙(위 공통 지침)은 그대로 두고, 그 도메인 특유의 절차만 덧붙인다.
# web: 검색 1회 → 답으로 끝나던 실측(브라켓 종류 혼재·무관 시장으로 채움·2017년 위키를 '최근 논문'으로)에 대한 절차.
# 도구는 브레이브 MCP(brave_llm_context = 검색+본문 조각, brave_web_search = 링크, brave_news_search = 뉴스).
# 라우터가 조회로 분류해도 늘 실행(WRITE) 질의로 넘기는 도메인 — 핵심 도구가 승인 도구라 READ 에서 빠진다.
_ALWAYS_ACTION_DOMAINS = (SCMS_ORDER_DOMAIN, SCHEDULE_DOMAIN)

_DOMAIN_ADDENDA = {
    # 예약: 지금 답을 내지 말고 예약을 만든다. 실행 때는 이 대화가 없으니 요청문을 완결되게 쓴다.
    "schedule": (
        "\n- '매일/평일/매주 ~에 … 해줘' 는 **지금 답하지 말고 schedule_create 로 예약한다.** 예약할 내용을 "
        "지금 조회하지 않는다.\n- prompt 는 실행 때 혼자 읽어도 되는 완결된 요청문으로 쓴다. 사용자가 정해 둔 규칙이 "
        "있으면 넣는다.\n- 반복 주기와 시각은 **사용자가 말한 값만** 쓴다. 말하지 않았으면 schedule_create 를 부르지 말고 "
        "언제 할지 묻는다 — 도구 설명의 예시 값으로 채우지 않는다. 승인이 필요한 일(메일 발송·상신 등)은 예약할 수 없다고 안내한다.\n"
        "- 목록은 schedule_list, 취소는 schedule_list 로 찾은 뒤 schedule_delete."
    ),
    # 메신저 발주요청: 도구를 안 부르고 "품명·수량·납기를 알려주세요" 로 처음부터 되묻던 실측(2026-09-23, 3턴
    # 연속 도구 호출 0회 — 스레드에 품명·수량이 있었다). 무엇이 있고 무엇을 물을지는 도구가 계산하므로
    # 도구를 먼저 부르는 절차만 정한다.
    "scms_order": (
        "\n- 발주요청은 **먼저 scms_order_request_draft 를 부른다.** 스레드 대화와 첨부에서 찾은 값만 넣고 "
        "모르는 값은 비운 채 부른다 — 값이 모자라도 부른다. 도구가 확인된 값과 물어볼 값을 "
        "돌려준다. 스레드 도면은 도구가 직접 읽으니 파일을 따로 읽지 않는다.\n"
        "- 사용자에게는 도구가 돌려준 required_next 항목만 묻는다. 도구를 부르지 않은 채 품명·수량·납기를 "
        "처음부터 묻지 마라. 사용자가 답하면 **답한 값만** 넣어 다시 부른다(앞서 준 값은 도구가 기억한다).\n"
        "- 사용자가 그만두겠다고 하면 cancel=true, 선택 항목을 넘어가겠다고 하면 skip_optional=true 로 부른다.\n"
        "- 도구가 `질문` 으로 확인한 값에 사용자가 이번 말로 답하면(예: '네', '그걸로요') 그 field 를 answered 에 "
        "넣어 부른다. 스레드에서 다른 사람과 나눈 말은 답이 아니다 — 넣지 않는다.\n"
        "- 거래유형·견적 마감 기한·대금 지급 조건은 묻지 않는다 — SCMS 화면에서 사람이 고른다.\n"
        "- 지난 턴에 같은 질문을 했더라도 그 답을 되풀이하지 말고, 이번 턴의 대화·답으로 다시 부른다."
    ),
    # 벤치(2026-09-23, ChatGPT 대비 30문항)로 다시 썼다. 옛 문구 "검색 1회로 끝내지 마라"가 난이도와 무관하게 모든 질문을
    # 상한(8회)까지 밀었다(하도급법 60일 11회·주가 9회, 턴당 32만 토큰 — ChatGPT 는 평균 5회·3.9만). 감점의 대부분은 검색 부족이
    # 아니라 근거 없는 덧붙임·시점 섞인 수치·리포트팜 출처였다 → 노력은 질문에 맞추고, 답은 결과에 있는 것만.
    # 답변 형식: 메인 시스템 프롬프트의 "인사이트 도출 의무·인사이트/제안 1줄 이하 금지"(사내 데이터용)가 웹 답에도 걸려 근거 없는
    # 전망·원인 추정을 채우게 했다("일관된 상승세", "9월치는 10월 초 발표" 등, 2026-09-23) → 웹 워커에서만 형식을 덮어쓴다.
    "web": (
        "\n- 노력은 질문에 맞춰라: 한 가지 사실(환율·주가·날씨·법 조항·정의·일정·결과)은 1~2회 조회로 확인하고 답한다. "
        "조사·동향·비교·목록 요청은 첫 단계에서 관점이 다른 질의 2~4개(필요하면 영어 포함)를 **한 번에 함께** 부르고, "
        "빠진 부분이 있을 때만 한 번 더 보강한다. 표현만 바꾼 같은 검색을 되풀이하지 마라. 웹 조회는 턴당 3단계까지다"
        "(단계마다 여러 호출을 함께 부를 수 있다 — 하나씩 차례로 부르지 마라). "
        "찾는 값이 아직 발표되지 않았으면 더 뒤지지 말고 가장 최근 확인값과 그 시점으로 답한다. "
        "본문 근거는 brave_llm_context, 링크 목록은 brave_web_search, 최신 뉴스는 brave_news_search.\n"
        "- **'찾을 수 없다·확인되지 않는다·아직 없다'고 쓰기 전에** 요청한 값이 있을 법한 결과 URL(가격지수·공고·통계·공식 발표 "
        "페이지) 1~2개를 web_fetch 로 열어 지금 페이지에서 확인하라 — 검색 조각은 인덱스 스냅샷이라 최신 갱신이 빠져 있다.\n"
        "- **시세(환율·주가·금리·원자재) '오늘·지금' 질문**: 결과 중 **가장 최근 시각의 값**을 첫 문장에 기준 시각과 함께 쓴다"
        "(예: '9월 28일 08:33 기준 1달러=약 1,357원'). 시장이 열렸는지·몇 시에 여는지를 네 기억으로 판단하지 말고 결과에 찍힌 "
        "시각으로 판단하라 — 거래 시간은 바뀐다(원·달러는 2026-07 부터 평일 24시간). 해외 사이트의 시각은 그 나라 시간대다 — "
        "'오늘'인지는 한국 시각으로 판단한다(결과 속 [= … KST] 가 환산값이고, web_fetch 의 fetched_at 은 방금 연 시각이다). "
        "역수·배수 표기(KRW-USD ×100 등)는 요청한 "
        "방향으로 환산한다. 출처별 수치를 늘어놓거나 '금융기관에서 확인하세요'로 끝내지 마라.\n"
        "- 검색어에는 사용자가 말한 대상·조건만 넣는다. 앞 결과에서 본 수치나 짐작한 수치를 검색어에 넣지 마라.\n"
        "- 답에 쓰는 수치·날짜·고유명사는 이번 결과에 있는 것만 쓰고, 그 문장 끝에 [매체명](URL) 로 출처를 단다. "
        "결과에 없는 부연(참여 기업·수상·전망 수치 등)을 덧붙이지 마라. 수치는 기준 시점(날짜, 종가/장중/예보 발표 시각)과 함께 쓰고, "
        "출처끼리 다르면 다르다고 밝혀라.\n"
        "- **추이·변동률은 같은 출처의 시계열로만** 말한다: 조사기관마다 기준(규격·조건·지역 정의)이 달라, A 기관의 8월 값과 B 기관의 "
        "9월 값으로 변동률을 계산하면 틀린 추세가 된다. 출처가 밝힌 증감률을 그대로 쓰고, 기관별 값은 따로 제시하라. "
        "시점은 본문에 적힌 기간(예: 'September 2026')을 따르고, 페이지 메타 날짜로 본문 기간을 뒤집지 마라.\n"
        "- 원하는 기간의 값이 결과에 없으면 '확인한 자료는 ○월(게시일 ○/○)까지'라고 말하라. '아직 공개되지 않았다·존재하지 않는다'고 "
        "단정하지 마라 — 검색 인덱스는 페이지 갱신을 늦게 반영해, 원 페이지에는 이미 최신 값이 있을 수 있다.\n"
        "- 공식 기관·기업 공식 자료·주요 언론을 우선 근거로 쓴다. 시장규모 리포트 판매 사이트·광고성 블로그·요청 기간 밖의 옛 글은 "
        "근거로 쓰지 마라. 요청 조건(기간·유형·대상)에 맞지 않는 결과는 버리고 그렇다고 말하라 — 무관한 내용으로 답을 채우지 마라.\n"
        "- **웹 답변 형식(위 '답변 구조'·'인사이트 도출 의무'·'인사이트·제안 섹션 금지 패턴'보다 우선한다)**: 첫 문장에 질문에 대한 "
        "직접 답 → 이번 결과로 확인된 핵심 근거(수치·날짜·기준 시점·출처)를 항목 몇 개나 표 하나로 → 확인하지 못한 부분이 있으면 "
        "한 줄. 결과로 뒷받침되지 않는 인사이트·전망·원인 추정·실무 제안 섹션은 쓰지 마라. 필요한 만큼만 짧게 쓴다.\n"
        "- 대상이 여러 뜻으로 갈리고(예: 브라켓 = 치과/산업/TV) 앞선 대화로도 정해지지 않으면 한 줄로 되물어라. "
        "후속 질의는 앞선 대화의 대상을 잇는다."
    ),
}


# 사내 데이터 답변용 형식 규칙 — 웹 답에는 해가 된다. "인사이트 도출 의무·인사이트/제안 1줄 이하 금지"가 근거 없는 전망·원인
# 추정을 채우게 했고, 웹 워커 지침으로 "이 형식보다 우선"이라 덮어써도 gemini 는 ①~④ 구조·인사이트·제안을 계속 붙였다
# (실측 2026-09-28 "오늘 환율": 묻지 않은 위안·유로·엔 표 + 인사이트 + 제안). 덮어쓰지 말고 웹 워커 프롬프트에서 뺀다.
_INTERNAL_ANSWER_SECTIONS_RE = re.compile(r"\n## (?:답변 구조|에이전틱 답변 원칙)[^\n]*\n.*?(?=\n## |\Z)", re.S)


def _strip_internal_answer_sections(system_prompt: str) -> str:
    return _INTERNAL_ANSWER_SECTIONS_RE.sub("\n", system_prompt or "")


def _worker_directive(domain: str) -> str:
    """워커 전용 도메인 격리 지침(system 프롬프트에 덧붙임).

    조회-먼저·초안-중복-방지는 이제 read/write 서브런 분해(코드)가 강제하므로, 여기선
    도메인 격리만 남긴다(다른 도메인 도구 호출 억제 — 스코프로도 막히지만 이중 방어).

    가이드 검색 예외(2026-08-31): "휴가 신청 어떻게 해?" 같은 방법·증상 질문도 라우터가
    도메인 워커로 보내는데(주제로는 정분류), 종전 지침의 전면 금지("이 도메인 도구만")가
    글로벌 유틸로 함께 바인딩된 search_user_guide_kb 까지 막아 워커가 가이드 근거 없이
    실행 인테이크·데이터 추측으로 새는 문제가 실측(guide 골드셋 agent 모드 howto
    0/12)으로 확인됐다. 격리 원칙은 유지하되 가이드 검색을 조건부 예외로 편입한다 —
    명확한 실행 요청("~해줘")은 종전대로 도메인 도구 직행이라 실행 UX 는 불변.
    """
    return (
        f"\n\n[워커 지침] 너는 '{domain}' 도메인 전담이다. 이 도메인 도구만 사용하고, "
        "다른 도메인 도구나 탐색 도구(find_*)는 호출하지 마라. 단, 함께 제공된 범용 "
        "유틸리티는 사용해도 된다:\n"
        "- 사람을 이름으로 지정한 실행(수신자·결재선 등)이면 사내 인물 검색"
        "(search_colleagues)으로 email 을 해석하라.\n"
        "- 사용자가 방법·절차·설정 위치를 묻거나('~어떻게 해?', '어디서 해?') 증상·문제를 "
        "신고하면('갑자기 ~됐어요', '~가 안 돼요') 도메인 도구로 실행·조회부터 하지 말고 "
        "가이드 검색(search_user_guide_kb)으로 문서를 확인해 방법·원인을 안내하고, 그 작업을 "
        "이 도메인 도구로 대신 수행할 수 있으면 마지막에 실행을 제안하라. 명확한 실행 요청"
        "('~해줘')은 이 예외에 해당하지 않는다 — 종전대로 도메인 도구로 바로 진행한다.\n"
        "- 사용자가 직전 답변의 근거·출처를 물으면 대화 기록에 있는 도구 결과와 답변 끝의 '출처' 표기로 "
        "답하라. 기록에 없으면 없다고 말하고, 어디서 나온 정보인지 지어내지 마라."
        + _DOMAIN_ADDENDA.get(domain, "")
    )


# 원문에는 있는데 라우터 서브쿼리에서 사라지면 워커가 엉뚱한 날짜를 잡는 시점 표현.
# 주차·상대일 표현만 잡는다 — 요일·시각은 라우터가 잘 보존하고, 넓게 잡으면 서브쿼리에
# 노이즈가 붙는다. 실측(2026-08-10): "다음주 금요일 …잡아줘" 의 action 질의에서 '다음주'가
# 빠져 워커가 이번주 금요일(08-14)에 등록했다(같은 발화의 조회 경로는 정상 → 라우터 유실 확정).
_TIME_MARKERS = (
    "지지난주", "지난주", "이번주", "다음주", "다다음주",
    "이번 주", "다음 주", "지난 주", "다다음 주",
    "그저께", "엊그제", "어제", "오늘", "내일", "모레", "글피",
)


def _lost_time_note(original: str | None, subquery: str) -> str:
    """원문에만 있고 서브쿼리에서 사라진 시점 표현을 되붙일 문구를 만든다. 없으면 빈 문자열.

    라우터 프롬프트에도 "시점 표현을 action 질의까지 보존하라"는 규칙이 있지만, 그건 LLM
    출력이라 확률적이다. 이 함수는 같은 보호를 **결정적으로** 건다 — 프로바이더·thinking
    설정이 무엇이든 '다음주'가 워커에 반드시 도달한다.

    '다음 주'(띄어쓰기)처럼 표기가 갈리는 경우를 위해 공백을 지운 형태로도 대조한다.
    """
    if not original:
        return ""
    sq_norm = subquery.replace(" ", "")
    lost = [
        m for m in _TIME_MARKERS
        if m in original and m.replace(" ", "") not in sq_norm
    ]
    if not lost:
        return ""
    # 긴 표현이 짧은 표현을 포함하는 경우(다음주 ⊃ 음주 아님, 지난주/난주 등) 중복 제거:
    # 이미 채택된 표현의 부분 문자열은 버린다.
    picked: list[str] = []
    for m in sorted(lost, key=len, reverse=True):
        if not any(m in p for p in picked):
            picked.append(m)
    return f"\n\n(원문의 시점 표현: {' · '.join(picked)} — 이 시점을 반드시 반영하세요.)"


# READ → WRITE 인계 블록 상한. 조회 답변 전문을 그대로 넣으면 WRITE 프롬프트가 부풀고,
# 핵심(무엇을 찾았는지)은 대개 앞부분에 있다.
_HANDOFF_MAX_CHARS = 1200


def _read_handoff(read_result) -> str:
    """READ 서브런이 찾아낸 것을 WRITE 서브런에 넘길 블록. 없으면 빈 문자열.

    **왜 필요한가.** 두 서브런은 같은 ``messages`` 를 받는다 — 즉 WRITE 는 READ 가 무엇을
    찾았는지 모른다. 실측(2026-09-15): READ 가 "에이전트_요금제_안내.xlsx" 를 찾아냈는데,
    WRITE 는 "찾은 파일을 회사 스크럼방에 전송하기" 라는 지시만 받고 **그 파일이 뭔지 몰라**
    같은 조회 도구를 다시 부르다 전송에 도달하지 못했다(post_message 호출 0회).

    내용은 READ 의 최종 답변을 쓴다 — 이미 정화를 거쳐 내부 식별자가 빠져 있고, 파일명 등
    사람이 읽는 사실은 남아 있다. WRITE 가 그 이름으로 stash 첨부를 지목할 수 있다.

    Args:
        read_result: READ 서브런의 LoopResult(없으면 None).

    Returns:
        마지막 user 메시지 뒤에 덧붙일 ``[조회 결과]`` 블록.
    """
    text = (getattr(read_result, "content", "") or "").strip()
    if not text:
        return ""
    if len(text) > _HANDOFF_MAX_CHARS:
        text = text[:_HANDOFF_MAX_CHARS].rstrip() + " …"
    return (
        "\n\n[조회 결과] 같은 턴의 조회 담당이 이미 아래를 확인했다. "
        "**다시 조회하지 말고 이 내용을 근거로 실행하라.**\n"
        + text
    )


def _focus_directive(subquery: str, mode: str) -> str:
    """원문 보존(A안) 경로의 담당 범위 지시문 — 서브쿼리를 입력 원고가 아닌 라벨로 쓴다.

    원문에는 조회·실행 요청이 섞여 있으므로 서브런 종류(mode)별로 자기 몫만 가리킨다.
    READ 워커가 원문의 실행 요청을 보고 과약속하는 것을 억제한다(실행 차원은 도구
    스코프·승인 게이트가 이미 결정적으로 차단).

    WRITE 지시문은 "실행 + 그 실행에 필요한 같은 도메인의 선행 조회"를 허용 범위로
    명시한다 — WRITE 서브런은 일부러 조회+승인 도구를 함께 바인딩하므로(598670e:
    UID·원본 메일·수신자 등 선행 조회 수단이 없으면 NOT FOUND 로 헤맴), "조회를
    다루지 마라"로만 쓰면 모델이 선행 조회까지 금지로 해석해 코드 정책과 충돌한다.
    특히 라우터가 actions 만 주는 턴(read 서브런 없음, 빈발)에서는 조회를 미룰 "별도
    담당"이 존재하지도 않는다. 금지 대상은 사용자가 **별도로 요청한 독립적인**
    조회·요약(read 서브런 몫)뿐이다.

    Args:
        subquery: 라우터가 만든 이 서브런의 질의(read_q/action_q) — 담당 범위 라벨.
        mode: 서브런 종류("read"/"write").

    Returns:
        마지막 user 원문 뒤에 덧붙일 지시문.
    """
    if mode == "write":
        return (
            f"\n\n[담당 지시] 위 요청 중 실행 부분인 '{subquery}' 와, 그 실행에 필요한 "
            "같은 도메인의 선행 조회(대상·UID·원본 메일·수신자 확인 등)만 처리하라. "
            "사용자가 별도로 요청한 독립적인 조회·요약 부분은 다른 담당자가 처리하므로 "
            "다루지 마라."
        )
    return (
        f"\n\n[담당 지시] 위 요청 중 조회 부분인 '{subquery}' 만 처리하라. "
        "발송·등록·수정·삭제 같은 실행 부분은 별도 담당이 처리하므로 다루지 마라."
    )


def _scope_messages(messages: list, subquery: str, domain: str, *, mode: str = "read", tools=None,
                    handoff: str = "") -> list:
    """워커용 메시지 — (1) system 프롬프트에 도메인 전담 지침 추가, (2) 마지막 user 를
    서브런 스코프에 맞게 조정. 원본 리스트/메시지 불변(model_copy). 히스토리 유지.

    범용(모놀리식) 프롬프트를 그대로 주면 워커가 전 도메인 find_ 를 부르거나(교차 도메인
    NOT FOUND) 승인 액션부터 잡아 조회를 건너뛴다 → 도메인 격리 + '읽기 먼저, 쓰기 마지막'
    순서를 프롬프트로 유도한다.

    (2)는 supervisor_preserve_original 플래그로 두 방식:
        - True(A안): 원문을 유지하고 ``_focus_directive``(서브쿼리=담당 라벨)만 덧붙인다.
          라우터 재작성이 무엇을 떨어뜨리든 워커가 원문을 직접 보므로, 교체 기인 유실
          계급(이미지 컨텍스트 4507889 · 주차 표현 4b31ed7 · 거절 초안 0a38065)이
          구조적으로 사라진다. 보상 노트(time/img/reject)도 불필요 — 원문에 이미 있다.
        - False(기본, 레거시): 서브쿼리로 교체 + 유실 보상 노트 재부착.
          검증(goldset/parallel_eval 전후 비교) 뒤 기본 ON 전환 예정.

    Args:
        messages: 원본 대화 메시지 리스트.
        subquery: 이 서브런의 라우터 질의(read_q 또는 action_q).
        domain: 워커 담당 도메인.
        mode: 서브런 종류("read"/"write") — A안 담당 지시문 문구 분기용(레거시 무영향).
        handoff: 같은 턴 READ 서브런이 찾아낸 것(``_read_handoff``). WRITE 만 받는다.
    """
    out = list(messages)
    # (1) system 프롬프트(첫 메시지)에 워커 지침 덧붙임
    if out and getattr(out[0], "role", None) == "system":
        try:
            base = out[0].content or ""
            if domain == "web":
                base = _strip_internal_answer_sections(base)
            out[0] = out[0].model_copy(
                update={"content": base + _worker_directive(domain) + _bound_tools_guidance(tools)}
            )
        except Exception:
            pass
    # (2-A안) 원문 유지 + 담당 지시 — 교체하지 않으므로 아래 레거시 경로의 보상 노트가
    #     전부 불필요하다(시점 표현·이미지 블록·거절 초안 블록이 원문에 그대로 있다).
    #     단, 서브쿼리가 원문 그대로인 경로(sticky fallback의 query_map=user_input,
    #     _domain_queries 빈 플랜 기본값, 거절 승격의 user_input 폴백)에서는 지시문을
    #     생략한다 — 담당 범위가 요청 전체라 라벨이 무의미하고, 덧붙이면 원문이
    #     본문+라벨로 두 번 실린다(원문 2회 주입).
    if settings.supervisor_preserve_original:
        for i in range(len(out) - 1, -1, -1):
            if getattr(out[i], "role", None) == "user":
                original = out[i].content or ""
                attach = bool(subquery.strip()) and subquery.strip() != original.strip()
                # ⚠️ 인계(handoff)는 담당지시 부착 여부와 **독립**이다. 서브쿼리가 원문과
                #    같아 라벨을 생략하는 턴에도 조회 결과는 넘겨야 한다 — 안 그러면 그
                #    턴만 WRITE 가 다시 조회한다.
                suffix = (_focus_directive(subquery, mode) if attach else "") + handoff
                if suffix:
                    try:
                        out[i] = out[i].model_copy(update={"content": original + suffix})
                    except Exception:
                        pass
                # INFO 유지: soak 기간에 부착/생략 분포를 로그로 관측한다(서브런당 1줄).
                logger.info(
                    "[scope] 원문보존(A안) domain=%s mode=%s 담당지시=%s",
                    domain, mode, "부착" if attach else "생략(서브쿼리=원문)",
                )
                break
        return out

    # (2-레거시) 마지막 user 메시지를 도메인 서브쿼리로 교체(워커가 자기 서브쿼리에만 집중).
    #     단, content 를 서브쿼리로 갈아끼우면 원본 augmented_input 의 "[이미지 컨텍스트]"
    #     텍스트 단서가 사라진다 — 이미지는 message.images 에 그대로 남지만(model_copy 가
    #     content 만 교체) 워커가 텍스트로는 "이미지 첨부됨"을 못 읽어 이미지 기반 도구
    #     (예: eapproval_extract_receipt)를 안 부르는 문제가 생긴다. 첨부가 있으면 신호를
    #     덧붙인다(도메인 무관 — 이미지 없으면 빈 문자열이라 무영향).
    #     같은 이유로 "[직전 거절된 초안]" 블록도 이월한다. orchestrator 가 augmented_input 에
    #     붙여 라우터가 정정 턴을 실행으로 분류하게 했는데, 여기서 content 를 갈아끼우면 정작
    #     도구를 부르는 워커는 초안 인자 원본을 못 본다 — 이월이 없으면 주입 전체가 무의미하다.
    from agent_arch.agents.context import get_image_context, get_rejected_draft_context
    from agent_arch.agents.hitl.policy import build_rejected_draft_note

    img_note = ""
    if get_image_context():
        img_note = (
            "\n\n(참고: 이번 요청에 이미지가 첨부돼 있습니다 — 이미지에서 정보를 읽어야 "
            "하면 해당 도메인 도구로 처리하세요.)"
        )
    reject_note = build_rejected_draft_note(get_rejected_draft_context())
    # 장기 메모리 턴 블록(별칭→실제 대상 id, 관련 기억)도 이월한다 — 도구를 부르는 건 워커인데
    # content 를 갈아끼우면 별칭의 채널 id 같은 근거가 워커에게서만 사라진다.
    memory_note = _memory_turn_note()
    # 현재 시각 블록(절대 날짜표)도 이월한다. 원문 user 메시지 끝에 붙어 있던 것이 서브쿼리로
    # 갈아끼우며 사라져, 워커가 date_from/date_to 를 학습 시점 날짜(2024-05-xx)로 지어냈다(실측
    # 2026-09-07). _lost_time_note 는 '어제' 같은 상대 표현만 되붙이고 기준 날짜는 주지 않는다.
    for i in range(len(out) - 1, -1, -1):
        if getattr(out[i], "role", None) == "user":
            try:
                original = out[i].content or ""
                time_note = _lost_time_note(original, subquery)
                # 원문에 시각 블록이 붙어 있었을 때만 되붙인다(오케스트레이터 경로). 없던 것을 만들어 넣지 않는다.
                time_ctx = _datetime_note() if _DATETIME_MARKER in original else ""
                out[i] = out[i].model_copy(
                    update={"content": subquery + time_note + img_note + reject_note + time_ctx + memory_note + handoff}
                )
            except Exception:
                pass
            break
    return out


_DRAFT_NOTE = "발송/등록 초안을 준비했어요 — 아래 카드에서 승인해 주세요."


def _draft_note(drafts: list) -> str:
    """초안 카드 위 한 줄. 카드(view)가 자기 안내(`draft_note`)를 주면 그것을, 아니면 공통 문구.

    공통 문구는 메일·일정·발주 등 모든 승인 도구에 붙어 "발송/등록" 이 맞지 않는 카드가 있다 — 물류 이관 카드에
    "발송/등록 초안을 준비했어요" 가 떴다(2026-10-13). 첫 초안 카드의 문구만 쓴다(나머지는 차례로 뜬다).
    """
    view = (drafts[0] or {}).get("draft_view") if drafts and isinstance(drafts[0], dict) else None
    note = str((view or {}).get("draft_note") or "").strip() if isinstance(view, dict) else ""
    return note or _DRAFT_NOTE

# WRITE 서브런에 예외적으로 얹는 글로벌 유틸.
# - search_colleagues: 사람 해석(이름→email) — 수신자·결재선 지정 실행에 필수.
# - search_user_guide_kb: 방법·증상 질문이 write 로 분류돼도 가이드 근거를 조달할 수
#   있게 한다(읽기 전용 검색이라 side-effect 없음). 이게 없으면 라우터가 howto 를
#   write 서브런으로 보내는 턴에서 워커 지침의 가이드 예외가 물리적으로 불가능해진다.
# - create_file: "보고서 만들어서 올려줘" 는 **보낼 대상을 만드는 것까지가 그 실행**이다.
#   이게 없으면 write 서브런이 전송 도구만 쥔 채 붙일 파일을 만들 수 없어 완수가
#   물리적으로 불가능하다(실측 2026-09-11 개발기: `[executor] tool NOT FOUND: create_file`
#   — 모델은 정확히 호출했고 워커 스코프에만 없었다). 파일 생성 자체는 대화 스코프 stash
#   에만 쓰므로 밖으로 나가지 않는다 — 실제 발송은 mattermost_post_message 의 승인 카드가
#   따로 막는다. "글로벌을 얹으면 초안 대신 엉뚱한 도구를 맴돈다"는 우려도 여기엔
#   해당하지 않는다: 파일 만들기는 요청에 명시될 때만 부르는 도구다.
# (전체 글로벌은 READ 전용 — 배경·실측 근거는 _run_worker docstring 예외 항목 참조)
# 첨부 읽기(router.CROSS_DOMAIN_READERS)도 WRITE 에 얹지만 여기 두지 않는다 — 메신저 MCP
# 도구라 레지스트리에 없는 환경이 있고, 이 목록은 '항상 존재하는 로컬 도구'를 전제한다.
_WRITE_GLOBAL_ALLOWLIST = frozenset({
    "search_colleagues", "search_user_guide_kb", "create_file",
})


# READ 서브런에 얹는 글로벌을 도메인별로 좁힌다. web: 사외 공개 웹만 보는 워커에 사내 디렉터리·가이드·메모리·화면 열기는
# 쓸 데가 없고 스키마 토큰과 오호출만 늘렸다(벤치 2026-09-23: 웹 질문 30개 중 가이드 검색 오호출 3회, 워커 도구 12개).
# 계산(환산·증감률)과 파일 만들기("표로 엑셀 만들어줘")만 남긴다.
_DOMAIN_GLOBAL_ALLOWLIST = {
    "web": frozenset({"calculator", "create_file"}),
}


def _domain_globals(global_tools: Optional[list], domain: str) -> list:
    allow = _DOMAIN_GLOBAL_ALLOWLIST.get(domain)
    tools = list(global_tools or [])
    # 첨부 읽기는 도메인별로 좁혀도 남긴다 — 첨부가 요청의 재료면 어느 워커든 열어야 한다.
    return tools if allow is None else [t for t in tools if t.name in allow or t.name in CROSS_DOMAIN_READERS]


def _with_domain_finder(tools: list, domain: str) -> list:
    """스코프에 자기 도메인 find_<domain>_tools 포함(NOT FOUND 방지 + 자체복구).

    web 은 도구가 셋(brave_*)뿐이라 셋이 이미 손에 있으면 관문을 얹지 않는다 — 얹으면 모델(gemini)이 바인딩된 도구를 두고
    관문부터 부르는 헛걸음을 했다(벤치 2026-09-23: 30문항 중 15회).
    """
    if domain == "web":
        from agent_arch.services.web.brave_output import BRAVE_TOOLS
        if BRAVE_TOOLS <= {t.name for t in tools}:
            return tools
    find_name = FIND_TOOL_FOR_DOMAIN.get(domain)
    if find_name and all(t.name != find_name for t in tools):
        finder = ToolRegistry.get(find_name)
        if finder is not None:
            return [*tools, finder]
    return tools


def _merge_tools(*groups) -> list:
    """도구 그룹들을 이름 기준 dedup(먼저 나온 인스턴스 우선)하여 하나로 합친다."""
    seen: set = set()
    out: list = []
    for g in groups:
        for t in (g or []):
            if t.name not in seen:
                seen.add(t.name)
                out.append(t)
    return out


def _domain_sticky(sticky_tools, domain: str) -> list:
    """직전 턴 sticky 도구 중 이 도메인에 속하는 것만 고른다(교차 도메인 오염 방지).

    sticky = 모놀리식 경로가 직전 턴 도구를 되살리던 것(참조 후속턴 "그거 취소" 대응).
    워커별로 자기 도메인 것만 얹어 도메인 격리를 유지한다. 도메인 무관(글로벌) sticky 는
    global_tools 로 이미 커버되므로 여기선 제외된다.
    """
    return [t for t in (sticky_tools or []) if domain_from_tool_names([t.name]) == domain]


def _sticky_has_action(sticky_tools, domain: Optional[str]) -> bool:
    """직전 턴 sticky 도구 중 해당 도메인의 승인(실행) 도구가 있는지 — sticky fallback 의
    read/action 모드 판정 원료(_decide prior_was_action). 도메인 없으면 False."""
    if not domain:
        return False
    return any(requires_approval(t.name) for t in _domain_sticky(sticky_tools, domain))


def _pending_draft_tools(domain: str) -> list:
    """진행 중 초안(거절/수집)이 이 도메인 것이면 이어받을 승인 도구를 레지스트리에서 꺼낸다.

    WRITE 로 태우는 것만으로는 부족하다 — retrieval 도 sticky 도 그 도구를 못 줄 수 있다.
    수집 중 첫 답변 턴이 정확히 그렇다: 직전 턴이 OCR·조회로 끝나 sticky 에 승인 도구가 없고,
    "1번" 같은 짧은 질의로는 의미검색도 못 찾는다(실측 2026-08-17 → NOT FOUND).
    stash 는 이어받을 도구 **이름**을 직접 들고 있어 그 경우에도 결정적으로 바인딩된다.
    """
    out = []
    for stash in _pending_draft_stashes():
        name = stash.get("tool_name")
        if domain_from_tool_names([name]) != domain:
            continue
        tool = ToolRegistry.get(name)
        if tool is not None:
            out.append(tool)
    return out


def _domain_queries(
    plan: RoutePlan, domain: str, user_input: str, *, sticky_tools=None,
) -> Tuple[Optional[str], Optional[str]]:
    """도메인의 (조회 질의, 실행 질의). 둘 다 없으면 전체를 조회로.

    **직전 턴 write 의도 이어받기**: 라우터가 이 도메인에 action 을 안 줬지만 직전 턴
    sticky 에 그 도메인의 승인 도구가 있으면 실행 질의로도 태운다.

    도구가 되물어서 사용자가 짧게 답하는 턴이 정확히 이 모양이다 — "담당자" 처럼 동사가
    없어 라우터가 read 로 분류하고, read 서브런은 승인 도구를 의도적으로 제외하므로
    (``_run_worker`` 참조) 직전 턴에 쓰던 write 도구가 사라진다. LLM 은 그걸 불러야 한다고
    알고 있어 호출하고 NOT FOUND 로 끝난다(실측 2026-08-31: 동명이인 되묻기 → "담당자"
    답변 → `tool NOT FOUND: mattermost_skill_send_dm`).

    ``_pending_draft_tools`` 는 초안이 **만들어진 뒤** 거절·수집 흐름을 덮는다. 여기는 초안이
    만들어지기 **전에** 도구가 되물은 경우라 stash 가 없어 그 장치가 걸리지 않는다.

    write 로 태워도 실행은 confirm=False 강제 게이트와 HITL 승인이 막으므로 안전 불변이다 —
    바인딩이 늘어날 뿐이다. 폴백으로 넣은 read 질의는 접는다(write 서브런이 도메인 조회
    도구까지 포함하므로 같은 질의로 두 번 돌 이유가 없다). 라우터가 **명시한** read 질의는
    유지한다.
    """
    read_q = plan.read_query(domain)
    action_q = plan.action_query(domain)
    if not read_q and not action_q:
        read_q = user_input
    if not action_q and _sticky_has_action(sticky_tools, domain):
        action_q = user_input
        if read_q == user_input:
            read_q = None
    return read_q, action_q


_PREFETCH_CONTEXT_TOKENS = 4096
_NEWS_FRESHNESS = {"day": "pd", "week": "pw", "month": "pm"}


def _web_prefetch(plan: RoutePlan, domain: str) -> list:
    """웹 조사 계획(router.web_queries) → 첫 모델 호출 전에 실행할 brave_llm_context 호출들.

    실사용(실측 2026-09-22~23, gemini): 워커가 질문 문장을 그대로 1~2회만 검색해 "7월부터 유가 추이"에 한국 CPI,
    EPDM·카본블랙 9월 단가는 "찾기 어렵다"로 끝났다(ChatGPT 는 같은 질문에 4~7회). 지침만으로는 여러 질의를 만들지
    않아 범위를 코드가 보장한다. 영문 검색어는 영어권 결과로 보낸다(기본 country=KR·ko 이면 영어 원자료가 밀린다).
    본문 조각은 호출당 4096 토큰 — 셋이 한꺼번에 들어오므로 기본(8192)의 절반.
    """
    if domain != "web":
        return []
    calls = []
    # 최신 값 요청이면 뉴스를 기간 필터로 먼저 — 본문 조각 검색은 기간 필터가 없어 몇 달 전 글이 섞인다(2026-09-23 실측).
    fresh = _NEWS_FRESHNESS.get(plan.web_recency)
    if fresh and plan.web_queries:
        calls.append({"name": "brave_news_search", "arguments": {"query": plan.web_queries[0], "freshness": fresh}})
    for q in plan.web_queries:
        args = {"query": q, "maximum_number_of_tokens": _PREFETCH_CONTEXT_TOKENS}
        if q.isascii():
            args.update(country="US", search_lang="en")
        calls.append({"name": "brave_llm_context", "arguments": args})
    return calls


def _make_worker(make_fn, domain: str):
    """워커 LLM 생성 — 도메인을 받는 팩토리면 넘긴다(도메인별 레인 설정). 인자 없는 팩토리(테스트·옛 호출부)는 그대로."""
    import inspect
    try:
        takes_domain = "domain" in inspect.signature(make_fn).parameters
    except (TypeError, ValueError):
        takes_domain = False
    return make_fn(domain=domain) if takes_domain else make_fn()


async def _run_subrun(domain: str, query: str, messages: list, tools: list, *,
                      make_worker_llm_fn, max_iterations: int, search_fn,
                      mode: str = "read",
                      handoff: str = "",
                      target_language: str = DEFAULT_LANGUAGE,
                      prefetch: Optional[list] = None, auto_fetch: bool = False,
                      turn_input_guide: bool = False) -> LoopResult:
    """도메인 서브런 하나 — fresh llm bind + 도메인 스코프 메시지로 react 그래프 실행.

    주의(의도적 보류): run_react_graph 의 ``input`` 인자는 원문 보존(A안)에서도 아직
    서브쿼리다. input 은 LLM 에 가지 않고 코드 쪽 의도 감지(_wants_thread_reply /
    _wants_react_latest)에만 쓰이는데, 서브쿼리에서 '답글'류 단어가 빠지면 자동 주입을
    놓칠 수 있다(교체 유실 계급의 축소판 — 증상은 "스레드 대신 채널 새 글" 수준).
    필요가 실측되면 user_input 을 여기까지 배관해 input 자리에 넘길 것.
    """
    wllm = _make_worker(make_worker_llm_fn, domain)    # fresh 인스턴스(bind 격리) + 도메인 레인 설정
    await wllm.bind_tools(tools)
    scoped = _scope_messages(messages, query, domain, mode=mode, tools=tools, handoff=handoff)
    return await run_react_graph(wllm, query, scoped, tools, search_fn, max_iterations,
                                 target_language=target_language, prefetch=prefetch, auto_fetch=auto_fetch,
                                 turn_input_guide=turn_input_guide)


def _turn_input_guide_allowed(single_domain: bool, read_q: Optional[str], action_q: Optional[str]) -> bool:
    """guide-first 의 턴 원문 폴백을 이 워커에 허용할지 — **단일 도메인이고 서브런이 정확히 하나**일 때만.

    복합 발화는 워커마다 같은 원문을 다시 보므로 "채널 찾는 방법 알려주고 메일 보내줘" 의 메일 워커가
    원문의 '방법' 때문에 강제된다. 같은 도메인이라도 read+write 두 서브런이면 write 쪽이 같은 문제라
    도메인 수가 아니라 실제 서브런 수로 판정한다(2026-10-13).
    """
    return bool(single_domain) and (bool(read_q) != bool(action_q))


async def _run_worker(
    domain: str, read_q: Optional[str], action_q: Optional[str], messages: list, *,
    resolve_tools_fn, make_worker_llm_fn, max_iterations: int, search_fn,
    global_tools: Optional[list] = None, sticky_tools: Optional[list] = None,
    target_language: str = DEFAULT_LANGUAGE, prefetch: Optional[list] = None, auto_fetch: bool = False,
    single_domain: bool = False,
) -> Tuple[str, Optional[LoopResult]]:
    """도메인 워커 — 조회(read)/실행(write) 서브런 분해(코드로 순서·격리·중복 강제).

    read 서브런: **비-승인 도구만** bind → 조회·요약(승인 도구가 없어 초안 물리적 불가).
    write 서브런: **도메인 전체 도구(조회+승인) + find_ 발견도구** bind → 실행 전 필요한
        조회를 스스로 수행하고 초안 카드를 만든다. 승인 도구만 주면 조회 수단이 없어
        NOT FOUND 로 헤맴. 실제 실행은 confirm=False 강제 게이트가 막으므로 안전 불변.
    라우터가 action 을 안 준 도메인은 write 서브런 자체를 안 돌린다(스푸리어스 발송 방지).

    글로벌 유틸리티(global_tools: calculator/디렉터리/웹검색 등)와 직전 턴 sticky 도구는
    모놀리식 경로가 항상 붙이던 것 — 워커 패리티를 회복한다(수신자 해석 등 횡단 작업).
    단 **글로벌은 READ 서브런에만** 얹는다: 막연한 실행 요청("일정 잡아줘")에 글로벌
    (웹검색·계산기·느린 디렉터리 호출)을 얹으면 LLM 이 초안 대신 불필요한 도구를 반복
    호출하며 지연·블록한다(실측 회귀). WRITE 도구 스코프는 위 규칙(도메인 전체 + find_)을
    따른다. 서브런 실행 여부(skip)는 도메인 관련 도구 유무로 결정하고, 글로벌은 read 실행
    확정 후 병합한다(도구 없는 도메인이 글로벌만으로 억지로 도는 것 방지).

    예외(_WRITE_GLOBAL_ALLOWLIST): 사람 검색(search_colleagues)만은 WRITE 에도 얹는다 —
    라우터가 actions 만 주는 턴(subqueries 빈값, 빈발)은 read 서브런이 아예 없어서,
    "사람을 이름으로 지정하는 실행"이 이름→email 해석 수단 없이 진행된다(실측:
    메일 수신자에 한글 이름 투입 → SMTP non-ASCII 502 / 전자결재 결재선 email 누락 →
    Mattermost 알림 멘션 빈칸). 위 지연 회귀의 원인(웹검색·계산기 반복 호출)이 아닌
    단건 조회라 선별 허용한다.
    """
    read_result: Optional[LoopResult] = None
    write_result: Optional[LoopResult] = None
    dom_sticky = _domain_sticky(sticky_tools, domain)

    if read_q:
        rtools = [t for t in (await resolve_tools_fn(read_q, domain)) if not requires_approval(t.name)]
        rtools = _merge_tools(rtools, [t for t in dom_sticky if not requires_approval(t.name)])
        rtools = _with_domain_finder(rtools, domain)
        if rtools:  # 도메인 관련 도구가 있을 때만 read 실행 — 글로벌은 그 위에 얹는다.
            rtools = _merge_tools(rtools, _domain_globals(global_tools, domain))
            logger.info("[supervisor] worker domain=%s READ q=%r tools=%d %s",
                        domain, read_q, len(rtools), [t.name for t in rtools][:8])
            read_result = await _run_subrun(domain, read_q, messages, rtools,
                                            make_worker_llm_fn=make_worker_llm_fn,
                                            max_iterations=max_iterations, search_fn=search_fn,
                                            mode="read", target_language=target_language,
                                            prefetch=prefetch, auto_fetch=auto_fetch,
                                            turn_input_guide=_turn_input_guide_allowed(single_domain, read_q, action_q))

    if action_q:
        # WRITE = 도메인 전체 도구(조회+승인) + 직전 턴 sticky + find_ (#105) — 실행 전
        # 도메인 내부 조회(수신자·uid) 가능. 글로벌은 넣지 않는다(막연한 실행에서 헤매는
        # 지연 회귀 방지 — 글로벌은 READ 에만). 실제 실행은 confirm=False 강제 게이트가
        # 막아 안전 불변. 단 사람 검색만은 예외로 얹는다 — 사유는 docstring 예외 항목 참조.
        # dom_sticky 병합: 정정 턴("결제수단 바꿔줘")은 retrieval 이 직전 승인도구(create)를
        # 못 찾을 수 있다 — 직전 턴 도구를 이어받아 NOT FOUND 를 결정적으로 차단한다.
        wtools = _merge_tools(list(await resolve_tools_fn(action_q, domain)), dom_sticky)
        # 진행 중 초안의 승인 도구를 이름으로 직접 얹는다 — retrieval·sticky 둘 다 놓치는
        # 수집 중 첫 답변 턴에서도 초안을 만들 수단이 사라지지 않게(NOT FOUND 결정적 차단).
        wtools = _merge_tools(wtools, _pending_draft_tools(domain))
        wtools = _with_domain_finder(wtools, domain)
        if wtools:
            wtools = _merge_tools(
                wtools,
                [t for t in (global_tools or [])
                 if t.name in _WRITE_GLOBAL_ALLOWLIST or t.name in CROSS_DOMAIN_READERS],
            )
            logger.info("[supervisor] worker domain=%s WRITE q=%r tools=%d %s",
                        domain, action_q, len(wtools), [t.name for t in wtools][:8])
            handoff = _read_handoff(read_result)
            if handoff:
                logger.info("[supervisor] READ→WRITE 인계 %d자", len(handoff))
            write_result = await _run_subrun(domain, action_q, messages, wtools,
                                             make_worker_llm_fn=make_worker_llm_fn,
                                             max_iterations=max_iterations, search_fn=search_fn,
                                             mode="write", handoff=handoff,
                                             target_language=target_language,
                                             turn_input_guide=_turn_input_guide_allowed(single_domain, read_q, action_q))

    return domain, _combine_worker(read_result, write_result)


#: 두 답변을 "같은 말"로 볼 유사도. 재서술은 문장 다듬기 정도만 달라 0.9 를 넘기고,
#: 되물음("어떤 휴가를 며칠 신청할까요?")은 0.3 도 넘지 않는다 — 그 사이를 넉넉히 가른다.
_RESTATEMENT_RATIO = 0.6


def _is_restatement(read_text: str, write_text: str) -> bool:
    """WRITE 텍스트가 READ 답변을 다시 말한 것인가.

    WRITE 서브런은 READ 결과를 인계(`_read_handoff`)받는다. 실행할 것이 없는 턴에서는 그
    인계 내용을 **그대로 다시 서술**하기 쉬운데, 그걸 이어붙이면 같은 답이 두 번 나간다.
    실행(도구 호출·초안)이 하나라도 있었으면 재서술이 아니므로 판정하지 않는다(호출부).

    공백·목록 기호 정도만 지우고 문자 단위로 비교한다 — 한국어라 형태소 분석 없이도
    재서술은 비율이 높게 나오고, 되물음·오류 안내는 낮게 나온다.
    """
    if not read_text or not write_text:
        return False
    import difflib
    import re as _re

    def _norm(t: str) -> str:
        return _re.sub(r"[\s*•\-#>|]+", "", t)

    a, b = _norm(read_text), _norm(write_text)
    if not a or not b:
        return False
    # 한쪽이 다른 쪽을 통째로 포함하면 길이 차이와 무관하게 재서술이다 — WRITE 가 READ 를
    # 그대로 옮기고 한두 줄 덧붙이는 형태가 흔한데, 비율만 보면 덧붙인 만큼 낮아져 놓친다.
    if a in b or b in a:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= _RESTATEMENT_RATIO


def _combine_worker(read_result: Optional[LoopResult], write_result: Optional[LoopResult]) -> Optional[LoopResult]:
    """read/write 서브런 결과를 하나의 LoopResult 로 합침. 둘 다 없으면 None(워커 제외).

    write 서브런 텍스트는 **초안(pending_queue)이 있을 때만** 버린다(카드와 중복 서술 방지).
    초안이 없으면(필수값 되물음·실패 안내 등) write 텍스트를 보존한다 — 버리면 사용자가
    빈 응답을 받고 프론트가 무한 로딩된다(실측: "휴가 신청하고 싶어" → 재질문 유실).
    content = read 요약 (+ 초안 있으면 고정 한 줄 / 무초안이면 write 텍스트).
    """
    if read_result is None and write_result is None:
        return None
    tool_calls: list = []
    tool_results: list = []
    tu = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    invoke = 0
    for r in (read_result, write_result):
        if r is None:
            continue
        tool_calls.extend(r.tool_calls or [])
        tool_results.extend(r.tool_results or [])
        merge_usage(tu, r.token_usage)
        invoke += r.llm_invoke_count
    drafts = list(write_result.pending_queue) if (write_result and getattr(write_result, "pending_queue", None)) else []
    content = (read_result.content.strip() if (read_result and read_result.content) else "")
    write_text = (write_result.content or "").strip() if write_result else ""
    if drafts:
        content = (content + ("\n\n" if content else "") + _draft_note(drafts)).strip()
    elif write_text and _is_restatement(content, write_text):
        # WRITE 가 READ 결과를 인계받아 **같은 말을 다시 한** 경우. 이어붙이면 사용자에게
        # 답이 통째로 두 번 나간다(2026-09-15 실측: "메신저에서 공유한 링크 찾아줘" →
        # 같은 문단이 2회, DB answer 917자에 같은 제목이 두 번). 둘 중 더 많이 담은 쪽만
        # 남긴다 — 어느 쪽이 더 자세한지는 턴마다 다르다.
        kept = "read" if len(content) >= len(write_text) else "write"
        logger.info("[supervisor] write 재서술 감지 → %s 만 유지(read=%d자 write=%d자)",
                    kept, len(content), len(write_text))
        content = content if kept == "read" else write_text
    elif write_text:
        # 무초안 write(되물음·실패 안내): 재서술이 아니면 정보가 있는 것이므로 보존한다.
        # 버리면 사용자가 빈 응답을 받고 프론트가 무한 로딩된다(실측: "휴가 신청하고 싶어").
        content = (content + ("\n\n" if content else "") + write_text).strip()
    # 서브런 스텝 계측 보존(관찰성 + 워커 격리 검증). 발견도구는 워커 레벨로 집계.
    discovered: list = []
    for r in (read_result, write_result):
        if r is not None and isinstance(getattr(r, "step_meta", None), dict):
            discovered += r.step_meta.get("discovered_tools") or []
    step_meta = {
        "discovered_tools": discovered,
        "read": getattr(read_result, "step_meta", None),
        "write": getattr(write_result, "step_meta", None),
    }
    return LoopResult(
        content=content,
        tool_calls=tool_calls,
        tool_results=tool_results,
        llm_invoke_count=invoke,
        token_usage=tu,
        finish_reason=("action_pending" if drafts else "stop"),
        pending_action=(drafts[0] if drafts else None),
        pending_queue=(drafts or None),
        step_meta=step_meta,
    )


_DATETIME_MARKER = "[현재 날짜/시간]"   # app/agents/prompts/system._build_datetime_context 의 머리말


def _datetime_note() -> str:
    """워커 user 메시지에 덧붙일 현재 시각·상대 날짜 기준표(코드 계산, 결정론). 실패 시 빈 문자열."""
    try:
        from agent_arch.agents.prompts.system import _build_datetime_context
        block = _build_datetime_context()
        return f"\n\n{block}" if block else ""
    except Exception:
        return ""


def _memory_turn_note() -> str:
    """이 턴 회상 블록(별칭·관련 기억)을 워커 user 메시지에 덧붙일 노트. 없으면 빈 문자열."""
    try:
        from agent_arch.agents.context import get_memory_recall
        recall = get_memory_recall()
        block = getattr(recall, "turn_block", "") if recall is not None else ""
        return f"\n\n{block}" if block else ""
    except Exception:
        return ""


def _memory_stable_block() -> str:
    """페르소나·전역 지시 등 안정 블록 — 종합기가 사용자의 규칙(표 금지 등)을 잃지 않게."""
    try:
        from agent_arch.agents.context import get_memory_recall
        recall = get_memory_recall()
        return getattr(recall, "system_block", "") if recall is not None else ""
    except Exception:
        return ""


def _synth_messages(user_input: str, ok_results: List[Tuple[str, LoopResult]],
                    language: str = DEFAULT_LANGUAGE) -> list:
    """합성기 메시지. 합성 결과가 최종 답변이므로 언어도 여기서 목표 언어로 못 박는다."""
    from agent_arch.agents.prompts.system import render_answer_language_section
    from agent_arch.llm.base import Message
    # 메신저 결과를 **먼저** 놓는다. 사내에서 오간 이야기가 질문의 근거인 경우가 많은데,
    # 드라이브 파일 검색 결과가 앞에 오면 합성기가 그걸 본문으로 삼아 정답이 각주로 밀린다
    # (실측 2026-09-15: 메신저가 정답 글을 찾았는데 답변 본문은 드라이브의 hwp 였다).
    ordered = sorted(ok_results, key=lambda kv: 0 if kv[0] == _PRIMARY_KNOWLEDGE_DOMAIN else 1)
    parts = [f"[{d}] 담당 결과:\n{(r.content or '').strip()}" for d, r in ordered]
    system = (
        "너는 여러 영역 담당자의 결과를 하나로 합치는 편집자다. 사용자 원 요청에 대해 "
        "아래 각 영역 결과를 자연스러운 답변 하나로 통합하라. 중복은 정리하고, "
        "각 영역의 핵심을 빠짐없이 담되 매끄럽게 이어라. 없는 내용을 지어내지 마라.\n"
        "**사내에서 오간 이야기(talk = 메신저) 결과가 있으면 그것을 답변의 본문으로 삼고, "
        "드라이브(drive) 파일은 '참고 자료'로 뒤에 붙여라.** 사용자가 파일·폴더를 직접 "
        "지목한 질문이면 그 반대로 해도 된다.\n"
        "**담당 결과가 번호 목록(시각·작성자·인용·링크)이면 그 목록을 문단으로 풀지 말고 그대로 "
        "살려라** — 항목·순서·링크를 바꾸지 말고, 다른 영역 결과는 목록 뒤에 덧붙인다. "
        "사용자는 건별 출처를 보고 직접 확인하려는 것이다."
        f"\n\n{render_answer_language_section(language)}"
    )
    # 멀티 도메인 턴에서 페르소나·전역 지시가 사라지던 공백을 메운다(사용자 규칙은 종합 단계가 최종 결정).
    stable = _memory_stable_block()
    if stable:
        system = f"{system}\n\n{stable}"
    user = f"사용자 요청: {user_input}\n\n" + "\n\n".join(parts)
    return [Message(role="system", content=system), Message(role="user", content=user)]


def _merge(ok_results: List[Tuple[str, LoopResult]], synth_content: str,
           synth_usage: Optional[dict], plan: RoutePlan, router_invokes: int = 1) -> LoopResult:
    """멀티 워커 결과 + synthesizer 를 하나의 LoopResult 로 병합(DB/history/집계용)."""
    tool_calls: list = []
    tool_results: list = []
    tu = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    invoke = router_invokes
    for _d, r in ok_results:
        tool_calls.extend(r.tool_calls or [])
        tool_results.extend(r.tool_results or [])
        merge_usage(tu, r.token_usage)
        invoke += r.llm_invoke_count
    merge_usage(tu, synth_usage)
    invoke += 1  # synthesizer
    step_meta = {
        "route": {"domains": plan.domains, "confidence": round(plan.confidence, 3)},
        "workers": [{"domain": d, "step_meta": getattr(r, "step_meta", None)} for d, r in ok_results],
        "synthesized": True,
    }
    all_results = [tr for _d, r in ok_results for tr in (r.tool_results or [])]
    return LoopResult(
        content=append_sources_footer(synth_content, all_results),
        tool_calls=tool_calls,
        tool_results=tool_results,
        llm_invoke_count=invoke,
        token_usage=tu,
        finish_reason="stop",
        step_meta=step_meta,
    )


def _collect_pending(ok_results: List[Tuple[str, LoopResult]]) -> list:
    """워커 결과들에서 승인대기(HITL) 액션을 모아 하나의 큐로 통합한다. 없으면 []."""
    queue: list = []
    for _d, r in ok_results:
        pq = getattr(r, "pending_queue", None)
        if pq:
            queue.extend(pq)
        elif getattr(r, "pending_action", None):
            queue.append(r.pending_action)
    return queue


def _merge_pending(ok_results: List[Tuple[str, LoopResult]], pending_queue: list, plan: RoutePlan) -> LoopResult:
    """승인대기 워커가 있으면 synth 대신 pending 을 실은 LoopResult 반환(HITL 체크포인트용).

    완료(비-pending) 워커 content 는 합쳐 담아 승인 전에 함께 보여준다. tool_calls/results 는
    전 워커 병합(history/DB). finish_reason="action_pending" + pending_queue 로 오케스트레이터가
    HITL 체크포인트를 만들어 /resume 이 실제 실행하게 한다.
    """
    tool_calls: list = []
    tool_results: list = []
    tu = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    invoke = 1  # router
    texts = []
    for _d, r in ok_results:
        tool_calls.extend(r.tool_calls or [])
        tool_results.extend(r.tool_results or [])
        merge_usage(tu, r.token_usage)
        invoke += r.llm_invoke_count
        # 완료 워커든 승인대기 워커든 content(읽기 요약 등)를 모두 보존 — 승인 카드와 함께
        # "받은 메일 요약" 같은 조회 결과가 답변에 남게(예전엔 pending 워커 content 를 버려 유실).
        c = (r.content or "").strip()
        if c:
            texts.append(c)
    return LoopResult(
        content="\n\n".join(texts),
        tool_calls=tool_calls,
        tool_results=tool_results,
        llm_invoke_count=invoke,
        token_usage=tu,
        finish_reason="action_pending",
        pending_action=pending_queue[0],
        pending_queue=pending_queue,
        step_meta={
            "route": {"domains": plan.domains, "confidence": round(plan.confidence, 3)},
            "workers": [{"domain": d, "step_meta": getattr(r, "step_meta", None)} for d, r in ok_results],
            "pending": len(pending_queue),
        },
    )


# ── 소스 우선순위: 메신저 먼저, 드라이브는 신호가 있을 때 ────────────────────
#: 드라이브·폴더라는 **낱말** 자체. 대화를 찾는 질문에선 주제어일 수 있어 따로 둔다.
_DRIVE_WORD_RE = re.compile(r"드라이브|폴더")

#: 그 낱말이 **위치**로 쓰였는가("드라이브에서", "폴더 안"). 이건 대화 질문에서도 파일 신호다.
_DRIVE_PLACE_RE = re.compile(r"(?:드라이브|폴더)\s*(?:에|안|속)")

#: **누가 무슨 얘기를 했는지** 찾는 질문. 이때 '드라이브'는 대개 논의의 주제다 —
#: "confluence 에서 DMP 드라이브로 전환 관련해 이야기한 것 찾아줘"(2026-10-01 실사용)가
#: 드라이브 검색과 병렬로 돌아 합성 단계에서 메신저 후보 목록이 한 문단으로 뭉개졌다.
_TALK_LOOKUP_RE = re.compile(
    r"(?:이야기|얘기|논의|대화|언급|말)\s*(?:를|을)?\s*"
    r"(?:한|했|하신|하셨|나눈|나눴|된|됐|오간|있었)"
)

#: 드라이브를 **지금 당장** 봐야 한다는 명시 신호. 폴더·확장자·"드라이브"처럼 사용자가 파일을
#: 직접 가리킨 경우만 잡는다. '문서'·'자료' 는 일부러 뺐다 — 메신저에 공유된 링크에도 똑같이
#: 붙는 말이라, 그걸로 드라이브를 부르면 대화형 질문이 파일 검색으로 샌다(실측 2026-09-15:
#: "공유한 문서 찾아줘" → drive_search_files 가 엉뚱한 hwp 를 물어와 답을 차지했다).
_DRIVE_SIGNAL_RE = re.compile(
    r"드라이브|폴더|nextcloud|전사\s*공용|공유\s*폴더|양식|템플릿"
    # 확장자: 점이 붙은 파일명(정산.xlsx)과 맨 토큰("xlsx 파일 어디 있어") 둘 다 잡는다.
    # 맨 토큰 목록은 문서 형식으로만 쓰이는 것들로 좁혔다 — txt·md·png 는 다른 문맥에서도
    # 흔해 잘못 걸리면 대화형 질문이 파일 검색으로 샌다.
    r"|\.(?:xlsx?|pptx?|docx?|hwpx?|pdf|csv|txt|md|png|jpe?g)\b"
    r"|\b(?:xlsx?|pptx?|docx?|hwpx?|pdf|csv)\b"
    r"|엑셀|피피티|파워포인트|한글\s*파일|워드|스프레드시트",
    re.IGNORECASE,
)

#: 미루기 판정을 적용할 조회 도메인 쌍. 다른 도메인(메일·일정·전자결재…)은 성격이 달라
#: 동시에 봐야 하는 것이 정상이므로 건드리지 않는다.
_PRIMARY_KNOWLEDGE_DOMAIN = "talk"
_DEFERRABLE_KNOWLEDGE_DOMAIN = "drive"


def _has_drive_signal(*texts: Optional[str], talk_lookup: bool = False) -> bool:
    """사용자가 파일·드라이브를 **직접 가리켰는가**.

    ``talk_lookup`` (대화를 찾는 질문)이면 '드라이브'·'폴더' 낱말만으로는 신호로 치지 않는다 —
    위치로 쓰였거나("드라이브에서") 확장자·양식 같은 다른 파일 신호가 있을 때만 참.
    """
    for t in texts:
        if not t:
            continue
        if not talk_lookup:
            if _DRIVE_SIGNAL_RE.search(t):
                return True
            continue
        if _DRIVE_PLACE_RE.search(t) or _DRIVE_SIGNAL_RE.search(_DRIVE_WORD_RE.sub(" ", t)):
            return True
    return False


def _split_deferred(plan: RoutePlan, user_input: str) -> Tuple[List[str], List[str]]:
    """(지금 실행할 도메인, 뒤로 미룰 도메인). 미룸은 talk+drive 조회쌍에서만 일어난다.

    왜 미루나: 둘을 **병렬로 돌리면 합성 단계에서 섞인다.** 메신저에 정답이 있어도 드라이브
    검색이 물어온 파일이 답의 본문을 차지하는 일이 실제로 있었다(2026-09-15). 메신저를 먼저
    보고, 거기서 답이 나오면 드라이브는 아예 돌리지 않는다 — 답이 깨끗해지고 호출도 준다.

    **좁게 적용한다.** 드라이브가 실행(actions)을 맡았거나 사용자가 파일·폴더·확장자를 직접
    가리켰으면 미루지 않는다. 그런 턴에서 드라이브를 미루면 사용자가 요청한 일을 안 한 것이
    되고, "빈약" 판정이 틀리면 결과가 영영 안 나온다.
    """
    domains = list(plan.domains or [])
    if (_PRIMARY_KNOWLEDGE_DOMAIN not in domains
            or _DEFERRABLE_KNOWLEDGE_DOMAIN not in domains):
        return domains, []
    if (plan.actions or {}).get(_DEFERRABLE_KNOWLEDGE_DOMAIN):
        return domains, []       # 드라이브가 실행을 맡았다 — 미루면 요청을 안 한 것이 된다
    drive_q = (plan.subqueries or {}).get(_DEFERRABLE_KNOWLEDGE_DOMAIN)
    if _has_drive_signal(user_input, drive_q,
                         talk_lookup=bool(_TALK_LOOKUP_RE.search(user_input or ""))):
        return domains, []       # 사용자가 파일·폴더를 직접 가리켰다 — 동시에 본다
    return ([d for d in domains if d != _DEFERRABLE_KNOWLEDGE_DOMAIN],
            [_DEFERRABLE_KNOWLEDGE_DOMAIN])


def _worker_found_something(result: Optional[LoopResult]) -> bool:
    """이 워커가 **근거를 실제로 가져왔는가**(미룬 도메인을 돌릴지 판단).

    본문 유무만 보면 안 된다 — 도구가 0건을 돌려줘도 모델은 "없습니다" 라는 문장을 만든다.
    도구 결과가 하나라도 있어야 근거가 있는 것이다.
    """
    if result is None:
        return False
    return bool(result.tool_results) and bool((result.content or "").strip())


async def run_supervisor(
    llm, user_input: str, messages: list, *,
    resolve_tools_fn, make_worker_llm_fn, fallback_fn,
    max_iterations: int = 10, confidence_threshold: float = 0.0,
    router_llm=None, search_fn=None, prior_domain: Optional[str] = None,
    global_tools: Optional[list] = None, sticky_tools: Optional[list] = None,
    target_language: str = DEFAULT_LANGUAGE,
) -> LoopResult:
    """멀티 오케스트레이션(비스트리밍). route → 단일/병렬 워커 → (멀티면)synth. 실패→fallback_fn.

    global_tools/sticky_tools: 모놀리식 경로가 항상 붙이던 도메인 무관 유틸리티/직전 턴 도구.
    각 워커 스코프에 얹어 횡단 기능(수신자 해석 등)과 참조 후속턴 패리티를 회복한다.

    target_language: 답변 목표 언어. 워커의 언어 가드와 합성기 지시에 함께 실린다 —
    워커만 맞추고 합성기를 두면 합성 단계에서 언어가 다시 뒤집힌다.
    """
    cid = _current_conversation_id()
    prior_domains = await _load_prior_domains(cid)
    plan, ok = await _decide(
        router_llm or llm, user_input, confidence_threshold, prior_domain,
        prior_was_action=_sticky_has_action(sticky_tools, prior_domain),
        prior_domains=prior_domains,
        history=recent_history(messages),   # 라우터도 앞 턴을 본다 — "Kim 관련된거"가 사내 사람 검색으로 새던 실측
    )
    if not ok:
        await _save_route(cid, [])
        _attach_route_meta(plan, fallback=True)
        logger.info("[supervisor] fallback (plan=%s)", getattr(plan, "domains", None))
        return await fallback_fn()

    domains, deferred = _split_deferred(plan, user_input)
    # guide-first 턴 원문 폴백 허용의 "단일 도메인"은 **라우터 계획 전체** 기준 — talk+drive 처럼 drive 를
    # 뒤로 미룬 턴은 지금 도는 도메인이 하나여도 복합 발화다(deferred 워커도 같은 원문을 다시 본다).
    single_domain_turn = len(domains) == 1 and not deferred

    def _spawn(ds: List[str]) -> list:
        return [
            _run_worker(d, *_domain_queries(plan, d, user_input, sticky_tools=sticky_tools),
                        messages,
                        resolve_tools_fn=resolve_tools_fn, make_worker_llm_fn=make_worker_llm_fn,
                        max_iterations=max_iterations, search_fn=search_fn,
                        global_tools=global_tools, sticky_tools=sticky_tools,
                        target_language=target_language, prefetch=_web_prefetch(plan, d),
                        auto_fetch=(d == "web" and plan.web_open),
                        single_domain=single_domain_turn)
            for d in ds
        ]

    results = list(await asyncio.gather(*_spawn(domains)))
    if deferred:
        primary = next((r for d, r in results if d == _PRIMARY_KNOWLEDGE_DOMAIN), None)
        if _worker_found_something(primary):
            logger.info("[supervisor] %s 에서 근거 확보 → %s 생략", _PRIMARY_KNOWLEDGE_DOMAIN, deferred)
        else:
            logger.info("[supervisor] %s 근거 없음 → %s 로 확장", _PRIMARY_KNOWLEDGE_DOMAIN, deferred)
            results += list(await asyncio.gather(*_spawn(deferred)))
    ok_results = [(d, r) for d, r in results if r is not None]

    if not ok_results:
        _attach_route_meta(plan, fallback=True)
        logger.info("[supervisor] 모든 워커 스코프 0 → fallback")
        return await fallback_fn()

    executed = [d for d, _ in ok_results]
    await _save_route(cid, executed)
    _attach_route_meta(plan, fallback=False, chosen=executed)

    if len(ok_results) == 1:
        return ok_results[0][1]  # 단일 워커 → synth 불필요, 그대로(pending 있으면 그대로 전파)

    pending = _collect_pending(ok_results)
    if pending:
        logger.info("[supervisor] 워커 %d건 승인대기 → 합성 스킵, HITL 큐 통합", len(pending))
        return _merge_pending(ok_results, pending, plan)

    await llm.bind_tools([])  # synth 는 도구 없이(메인 llm 이 스코프도구 바인딩돼 있을 수 있음)
    synth_resp = await llm.invoke(_synth_messages(user_input, ok_results, target_language))
    logger.info("[supervisor] synthesized %d workers: %s", len(ok_results), executed)
    # synth 는 워커 결과를 **새로 쓴** 문장이라 별도 정화가 필요하다. 워커 content 는 react
    # 루프/그래프가 이미 정화했지만(그래서 synth 입력에는 도구명이 없다), 합성 LLM 이 스스로
    # 도구명을 지어내 쓰는 경우까지는 막지 못한다.
    synth_text = _guard_synth_names(
        sanitize_tool_names(synth_resp.content or "",
                            link_permits=_link_policy(user_input, ok_results).permits),
        user_input, ok_results, _requester_block(messages))
    # 합성 답변은 react 그래프의 최종 노드를 지나지 않는다 — 워커 결과를 **새로 쓴** 문장이라
    # 용어 표기도 다시 샌다. 단일 워커 경로(위)는 그래프가 이미 교정했다.
    from agent_arch.services.glossary import enforce_glossary

    synth_text = await enforce_glossary(synth_text, target_language)
    return _merge(ok_results, synth_text, synth_resp.usage, plan)


def _synth_evidence(user_input: str, ok_results: list, requester: str = "") -> str:
    """합성 답변의 이름 대조 근거 — 워커들이 실제로 낸 문장 + 사용자 질문 + 요청자 정보.

    워커 content 는 이미 이름 가드를 통과했으므로, 합성 LLM 이 그 위에서 **새로** 지어낸
    이름만 여기서 걸린다(예: 워커가 준 ``@lho`` 를 "이호"로 옮겨 적는 경우).

    요청자 정보(``[요청자 정보]`` 블록)도 넣는다 — 합성 답이 질문한 사람을 "○○○ 이사님" 으로 부르면 그 이름은 이
    블록에만 있다. 빠져 있어 **질문자 본인 이름**을 지어낸 이름으로 잡아 "이름미확인 이사님" 으로 바꿨다(실측
    2026-10-02 메신저: talk+drive 합성 답의 호칭). 일반 경로(react_graph)는 같은 블록을 이미 근거에 넣는다.
    """
    parts = [user_input or "", requester or ""]
    for _, r in ok_results:
        c = getattr(r, "content", None)
        if isinstance(c, str) and c:
            parts.append(c)
    return "\n".join(parts)


def _worker_tool_results(ok_results: list) -> list:
    """워커들이 실제로 받은 도구 결과 원본 — 링크 치환·각주가 같은 출처를 쓴다.

    ``_synth_evidence`` 는 워커 **산문**만 모으지만, 파일 링크를 고치려면 결과 JSON 의
    ``file_id``·``doc_url`` 짝이 필요하다. 그 값은 ``LoopResult.tool_results`` 에 그대로
    붙어 있다(바로 아래 ``sources_footer`` 가 이미 같은 것을 소비한다).
    """
    return [tr for _d, r in ok_results for tr in (getattr(r, "tool_results", None) or [])]


def _link_policy(user_input: str, ok_results: list) -> TurnLinkPolicy:
    """합성 답변용 링크 정책 — 워커들이 받은 도구 결과 + 사용자 입력이 근거다.

    워커 본문은 각자 정화를 거쳤지만 **합성은 새로 쓰는 LLM** 이라, 워커가 제목만 말한
    가이드 문서에 주소를 붙여 링크로 만들 수 있다. 판정 규칙은 `react/link_policy.py`.
    (시스템 프롬프트의 고정 링크는 `link_policy.static_allowed()` 가 따로 연다.)
    """
    policy = TurnLinkPolicy()
    policy.observe_results(_worker_tool_results(ok_results))
    policy.allow_text(user_input)
    return policy


def _link_hold_needed(ok_results: list) -> bool:
    """합성 토큰을 붙들 턴인가 — 워커 결과에 메신저 첨부 히트가 있으면 붙든다.

    합성은 워커 산문을 **새로 쓰는** LLM 이라 없는 파일 주소를 다시 조립할 수 있다.
    붙들지 않으면 화면에는 죽은 링크가 찍히고 저장본만 고쳐진다(이름 가드와 같은 이유).
    """
    return any(
        isinstance(c := tr.get("content"), str) and '"file_id"' in c
        for tr in _worker_tool_results(ok_results) if isinstance(tr, dict)
    )


def _requester_block(messages: list) -> str:
    """대화 메시지의 시스템 프롬프트에서 [요청자 정보] 블록만(이름 대조 근거용, react_graph 와 같은 규칙)."""
    for m in messages or []:
        if getattr(m, "role", None) == "system" and isinstance(getattr(m, "content", None), str):
            block = requester_context(m.content)
            if block:
                return block
    return ""


def _guard_synth_names(text: str, user_input: str, ok_results: list, requester: str = "") -> str:
    """합성 답변에서 근거에 없는 사람 이름을 마스킹한다(합성은 재생성하지 않는다).

    워커 단계에서 이미 재생성 기회를 한 번 썼고, 합성은 워커 결과의 재서술이라 근거가
    좁고 확실하다. 여기서는 결정적으로 마스킹만 한다.
    """
    # 링크 가드는 이름 가드와 **독립**이다 — `name_answer_guard` 를 꺼도 지어낸 딥링크는
    # 막아야 한다(사용자가 눌러 보기 전엔 진위를 알 수 없는 거짓말이다).
    tool_results = _worker_tool_results(ok_results)
    text, _ = redact_fabricated_permalinks(
        text or "",
        file_links=build_file_link_map(tool_results),
    )
    # 첨부 링크의 목적지는 코드가 정한다 — 합성도 파일과 주소를 뒤바꿔 짝지을 수 있다.
    text, _ = apply_file_links(text, file_refs(tool_results),
                               download_base=settings.mattermost_base_url)
    if not settings.name_answer_guard or not text:
        return text
    evidence = _synth_evidence(user_input, ok_results, requester)
    # 스트리밍 보류 조건(has_unnamed_author)과 같은 게이트를 쓴다 — 붙들지 않은 턴에서
    # 본문을 바꾸면 화면 ≠ 히스토리가 되고, demo 모드의 가상 이름도 건드리게 된다.
    if not has_unnamed_author(evidence):
        return text
    return redact_unsourced_names(text, find_unsourced_person_names(text, evidence))


async def stream_supervisor(
    llm, user_input: str, messages: list, *,
    resolve_tools_fn, make_worker_llm_fn, fallback_stream_fn,
    max_iterations: int = 10, confidence_threshold: float = 0.0,
    router_llm=None, search_fn=None, prior_domain: Optional[str] = None,
    global_tools: Optional[list] = None, sticky_tools: Optional[list] = None,
    target_language: str = DEFAULT_LANGUAGE,
):
    """멀티 오케스트레이션(스트리밍). 단일=워커 직스트림, 멀티=진행이벤트+synth만 스트림.

    global_tools/sticky_tools: run_supervisor 와 동일 — 워커 스코프에 얹는 횡단 유틸리티/
    직전 턴 도구(모놀리식 패리티).
    """
    cid = _current_conversation_id()
    prior_domains = await _load_prior_domains(cid)
    plan, ok = await _decide(
        router_llm or llm, user_input, confidence_threshold, prior_domain,
        prior_was_action=_sticky_has_action(sticky_tools, prior_domain),
        prior_domains=prior_domains,
        history=recent_history(messages),   # 라우터도 앞 턴을 본다 — "Kim 관련된거"가 사내 사람 검색으로 새던 실측
    )
    if not ok:
        await _save_route(cid, [])
        _attach_route_meta(plan, fallback=True)
        async for ev in fallback_stream_fn():
            yield ev
        return

    domains = plan.domains

    # 단일 도메인 + 조회 전용(실행 없음) → read 워커를 그대로 스트리밍(토큰 UX 보존).
    if len(domains) == 1:
        d = domains[0]
        read_q, action_q = _domain_queries(plan, d, user_input, sticky_tools=sticky_tools)
        if read_q and not action_q:
            rtools = [t for t in (await resolve_tools_fn(read_q, d)) if not requires_approval(t.name)]
            rtools = _merge_tools(rtools, [t for t in _domain_sticky(sticky_tools, d) if not requires_approval(t.name)])
            rtools = _with_domain_finder(rtools, d)
            if not rtools:  # 도메인 관련 도구 0 → 무스코프 fallback(글로벌만으로 억지 실행 안 함)
                await _save_route(cid, [])
                _attach_route_meta(plan, fallback=True)
                async for ev in fallback_stream_fn():
                    yield ev
                return
            # 실행 확정 후 글로벌 유틸리티 병합 — 비스트림 _run_worker 와 **같은 함수**로(도메인별 글로벌 좁히기·
            # 워커 레인 설정·선실행). 예전엔 여기만 복제본이라 web 워커 개선이 패널(스트리밍)에 안 닿았다(2026-09-23).
            rtools = _merge_tools(rtools, _domain_globals(global_tools, d))
            _attach_route_meta(plan, fallback=False, chosen=[d])
            await _save_route(cid, [d])
            wllm = _make_worker(make_worker_llm_fn, d)
            await wllm.bind_tools(rtools)
            async for ev in stream_react_graph(wllm, read_q,
                                               _scope_messages(messages, read_q, d, mode="read", tools=rtools),
                                               rtools, search_fn, max_iterations,
                                               target_language=target_language, prefetch=_web_prefetch(plan, d),
                                               auto_fetch=(d == "web" and plan.web_open),
                                               turn_input_guide=True):   # plan.domains 가 1개(미룸 없음)·read 전용 = 서브런 1개
                yield ev
            return
        # 실행(write) 포함 단일 도메인은 아래 공통 경로(gather + 카드)로.

    # 실행 포함 or 멀티 도메인 → 워커 병렬(비스트림) + 진행이벤트, 최종은 카드/synth.
    yield ProgressEvent("thinking", content="요청을 영역별로 처리하고 있어요...")
    domains, deferred = _split_deferred(plan, user_input)
    # guide-first 턴 원문 폴백 허용의 "단일 도메인"은 **라우터 계획 전체** 기준 — talk+drive 처럼 drive 를
    # 뒤로 미룬 턴은 지금 도는 도메인이 하나여도 복합 발화다(deferred 워커도 같은 원문을 다시 본다).
    single_domain_turn = len(domains) == 1 and not deferred

    def _spawn(ds: List[str]) -> list:
        return [
            _run_worker(d, *_domain_queries(plan, d, user_input, sticky_tools=sticky_tools),
                        messages,
                        resolve_tools_fn=resolve_tools_fn, make_worker_llm_fn=make_worker_llm_fn,
                        max_iterations=max_iterations, search_fn=search_fn,
                        global_tools=global_tools, sticky_tools=sticky_tools,
                        target_language=target_language, prefetch=_web_prefetch(plan, d),
                        auto_fetch=(d == "web" and plan.web_open),
                        single_domain=single_domain_turn)
            for d in ds
        ]

    results = list(await asyncio.gather(*_spawn(domains)))
    if deferred:
        primary = next((r for d, r in results if d == _PRIMARY_KNOWLEDGE_DOMAIN), None)
        if _worker_found_something(primary):
            logger.info("[supervisor] (stream) %s 에서 근거 확보 → %s 생략",
                        _PRIMARY_KNOWLEDGE_DOMAIN, deferred)
        else:
            logger.info("[supervisor] (stream) %s 근거 없음 → %s 로 확장",
                        _PRIMARY_KNOWLEDGE_DOMAIN, deferred)
            yield ProgressEvent("thinking", content="메신저에서 못 찾아 드라이브도 확인할게요...")
            results += list(await asyncio.gather(*_spawn(deferred)))
    ok_results = [(d, r) for d, r in results if r is not None]

    if not ok_results:
        _attach_route_meta(plan, fallback=True)
        async for ev in fallback_stream_fn():
            yield ev
        return

    executed = [d for d, _ in ok_results]
    await _save_route(cid, executed)
    _attach_route_meta(plan, fallback=False, chosen=executed)

    if len(ok_results) == 1:
        r = ok_results[0][1]
        if r.content:
            yield ProgressEvent("token", content=r.content)
        yield r
        return

    # 승인대기 워커가 있으면 synth 스킵 — 완료분 content 흘리고 pending LoopResult 로 마무리.
    pending = _collect_pending(ok_results)
    if pending:
        logger.info("[supervisor] (stream) 워커 %d건 승인대기 → 합성 스킵, HITL 큐 통합", len(pending))
        lr = _merge_pending(ok_results, pending, plan)
        if lr.content:
            yield ProgressEvent("token", content=lr.content)
        yield lr
        return

    # synthesizer 스트림
    # 링크 정책은 여기서 보낸다 — 병렬 워커는 ainvoke(run_react_graph)라 자기 정책을 흘릴 수
    # 없다. 합성 토큰보다 먼저 나가야 화면 정화기가 제때 물린다.
    policy = _link_policy(user_input, ok_results)
    if policy.active:
        yield ProgressEvent("link_policy", content=policy.dumps())
    await llm.bind_tools([])
    buf: list = []
    synth_usage = None
    # 워커 결과에 @handle 작성자가 섞였으면 합성이 그걸 한글 이름으로 옮겨 적을 수 있다.
    # 그 턴만 토큰을 붙들었다가 검증 후 한 번에 낸다 — 틀린 이름이 화면에 찍힌 뒤 저장본만
    # 고치면 화면 ≠ 히스토리가 되고, SSE 에는 본문 교체 이벤트가 없어 되돌릴 수 없다.
    hold = (settings.name_answer_guard and has_unnamed_author(
        _synth_evidence(user_input, ok_results))) or _link_hold_needed(ok_results)
    async for ev in llm.stream(_synth_messages(user_input, ok_results, target_language), tool_mode="NONE"):
        if ev.type == "text":
            if ev.content:
                buf.append(ev.content)
                if not hold:
                    yield ProgressEvent("token", content=ev.content)
        elif ev.type == "done":
            synth_usage = ev.usage
    logger.info("[supervisor] synthesized(stream) %d workers: %s", len(ok_results), executed)
    # 화면에 나가는 토큰은 orchestrator 의 StreamSanitizer 가 정화하지만, 이 buf 는 DB/Redis
    # 히스토리 저장용이라 여기서 따로 정화해야 한다. 안 하면 저장본에만 도구명 원문이 남고,
    # 그 히스토리가 다음 턴 LLM 맥락으로 되돌아와 모델에게 "노출해도 된다"는 선례가 된다
    # (화면만 봐서는 안 보이는 누출 경로).
    synth_text = _guard_synth_names(
        sanitize_tool_names("".join(buf), link_permits=policy.permits), user_input, ok_results,
        _requester_block(messages))
    if hold and synth_text:
        # 붙들었던 본문을 검증 후 한 번에 방출 — 저장본과 정확히 같은 텍스트다.
        yield ProgressEvent("token", content=synth_text)
    footer = sources_footer(synth_text, _worker_tool_results(ok_results))
    if footer:
        yield ProgressEvent("token", content=footer)   # _merge 가 같은 각주를 content 에 붙인다(화면 = 히스토리)
    yield _merge(ok_results, synth_text, synth_usage, plan)
