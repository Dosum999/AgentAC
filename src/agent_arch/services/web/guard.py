"""웹 도구 문지기 — 회사 허용 목록, 사내 식별자 차단, 일일 한도, 감사 원장.

검색 실행은 브레이브 검색 MCP 서버(deploy/brave-mcp)가 하고, 여기는 **회사 정책**만 본다. MCP 서버는 회사·사용자를
모르므로 이 계층이 agent 에 남는 유일한 웹 전용 로직이다. 어떤 검색 도구든(웹·뉴스·본문 조각) 같은 문지기를 지난다.

- 회사 허용 목록(WEB_SEARCH_ALLOWED_COMPANIES, CSV, 비우면 전체): 회사별 on/off.
- 사내 식별자 차단: 검색어에 이메일 주소·요청자 이름·계정이 들어가면 밖으로 내보내지 않는다(유출 방지, 사내 질문 오호출의
  흔한 모양이기도 하다).
- 일일 한도(WEB_SEARCH_DAILY_QUOTA, 회사별, 0=무제한): Redis INCR. Redis 가 없으면 fail-open(한도 미적용).
- 감사 원장: Redis 리스트 `web:audit:{company}` — 누가 무엇을 검색했고 결과가 어땠는지(회사별 최근 500건, 30일).
- opt-in(2026-09-16): 웹 검색은 **기본 꺼짐**. 패널 토글(요청 web_search=true)과 회사 설정(common_app Company Setting
  `agent_web_search_enabled`, 매니저 테넌트 설정에서 켠다)이 둘 다 켜졌을 때만 그 요청에 웹 도메인이 열린다.
  `WEB_SEARCH_ALLOWED_COMPANIES` 가 비어 있지 않으면 운영 강제 스위치로 회사 설정을 대신한다.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_AUDIT_MAX = 500
_AUDIT_TTL_SECONDS = 30 * 24 * 3600


def company_allowed(company: Optional[str]) -> bool:
    raw = (getattr(settings, "web_search_allowed_companies", "") or "").strip()
    if not raw:
        return True
    allowed = {c.strip() for c in raw.split(",") if c.strip()}
    return bool(company) and company in allowed


def blocked_identifier(query: str, user: dict) -> Optional[str]:
    """검색어에 사내 식별자가 들어 있으면 그 종류를 돌려준다. 이메일 주소는 무조건, 요청자 본인 값은 그대로 매칭."""
    if _EMAIL_RE.search(query or ""):
        return "email"
    q = (query or "").lower()
    for key in ("email", "username", "name"):
        v = str((user or {}).get(key) or "").strip().lower()
        if len(v) >= 2 and v in q:
            return key
    local = str((user or {}).get("email") or "").split("@")[0].lower()
    if len(local) >= 4 and local in q:
        return "account"
    return None


def daily_quota() -> int:
    try:
        return int(getattr(settings, "web_search_daily_quota", 0) or 0)
    except Exception:
        return 0


async def incr_quota(company: Optional[str]) -> Optional[int]:
    """오늘 누적 호출 수(+1). Redis 없으면 None(한도 미적용, fail-open)."""
    try:
        from agent_arch.memory.redis_store import redis_store
        r = redis_store.redis
        if r is None:
            return None
        key = f"web:quota:{company or 'anon'}:{datetime.now(timezone.utc).strftime('%Y%m%d')}"
        n = await r.incr(key)
        if n == 1:
            await r.expire(key, 2 * 24 * 3600)
        return int(n)
    except Exception:
        logger.debug("web 한도 카운터 실패(무시)", exc_info=True)
        return None


async def _lexicon_blocks(query: str, company: Optional[str]) -> bool:
    """가명화 어휘(임직원·거래처 담당자 이름·이메일)가 검색어에 있으면 막는다 — 래퍼가 도구 인자를 원문으로
    복원한 뒤 Brave 로 나가는 구멍. 가드 모드가 off 이거나 회사가 켜지 않았으면 어휘를 만들지 않는다(기존 동작 그대로)."""
    try:
        from agent_arch.llm.pii.policy import company_enabled, global_mode
        if global_mode() == "off" or not await company_enabled(company):
            return False
        from agent_arch.llm.pii.turn import build_turn_lexicon
        lex = await build_turn_lexicon()
        return bool(lex.pattern is not None and lex.pattern.search(query or ""))
    except Exception:  # noqa: BLE001
        return False


def _audit_query(query: str) -> str:
    """감사 원장·로그용 검색어. 가드 모드가 켜져 있으면 식별자를 토큰으로 바꿔 남긴다(원문 값 비노출)."""
    try:
        from agent_arch.llm.pii.policy import global_mode, secret
        if global_mode() == "off":
            return query
        from agent_arch.llm.pii.engine import Substituter
        from agent_arch.llm.pii.tokens import TokenIssuer
        out, _ = Substituter(TokenIssuer(secret(), "web-audit")).substitute_text(query or "")
        return out
    except Exception:  # noqa: BLE001
        return query


async def audit(company: Optional[str], user: dict, tool: str, query: str, outcome: str) -> None:
    query = _audit_query(query)
    rec = {"ts": time.time(), "user": (user or {}).get("sub"), "email": (user or {}).get("email"),
           "tool": tool, "query": query, "outcome": outcome}
    logger.info("[web] company=%s user=%s tool=%s outcome=%s q=%r", company, (user or {}).get("email"), tool, outcome, query)
    try:
        from agent_arch.memory.redis_store import redis_store
        r = redis_store.redis
        if r is None:
            return
        key = f"web:audit:{company or 'anon'}"
        await r.lpush(key, json.dumps(rec, ensure_ascii=False))
        await r.ltrim(key, 0, _AUDIT_MAX - 1)
        await r.expire(key, _AUDIT_TTL_SECONDS)
    except Exception:
        logger.debug("web 감사 원장 기록 실패(무시)", exc_info=True)


async def precheck(query: str, user: dict, tool: str) -> Optional[dict]:
    """호출 전 문지기. 막으면 구조화 오류 dict(모델이 읽고 다음 행동을 고를 수 있게), 통과면 None.

    순서: 회사 허용 → 식별자 → 한도. 막힌 호출도 감사 원장에 남긴다.
    """
    company = (user or {}).get("company")
    if not company_allowed(company):
        await audit(company, user, tool, query, "company_disabled")
        return {"error": "web_search_disabled",
                "message": "이 회사에는 웹 검색이 열려 있지 않습니다. 사내 도구로 답하거나 웹 검색이 필요하다고 안내하세요."}
    kind = blocked_identifier(query, user) or ("person" if await _lexicon_blocks(query, company) else None)
    if kind:
        await audit(company, user, tool, query, f"blocked_identifier:{kind}")
        return {"error": "blocked_identifier", "kind": kind,
                "message": "검색어에 사내 식별자(이메일·이름·계정)가 들어 있어 외부로 보내지 않았습니다. 사내 정보는 사내 도구로 조회하세요."}
    quota = daily_quota()
    if quota > 0:
        n = await incr_quota(company)
        if n is not None and n > quota:
            await audit(company, user, tool, query, "quota_exceeded")
            return {"error": "quota_exceeded", "message": ("오늘 이 회사의 웹 검색 한도를 넘어 이번 검색은 실행되지 않았습니다. 이번 턴에 이미 받은 검색 결과가 있으면 "
                                "그것으로 답하고(더 찾지 못한 부분만 한도 때문이라고 밝힌다), 받은 결과가 없을 때만 한도 초과를 안내하세요.")}
    return None


# ─── opt-in: 회사 설정(common_app) + 패널 토글 ──────────────────────────────────

_FEATURES_METHOD = "common_app.setup.doctype.company_setting.company_setting.get_company_agent_features"
# v2: 가명화 래퍼 플래그(pii_guard)가 응답에 추가되면서 접두사를 올렸다 — 구 캐시엔 그 키가 없어 TTL 동안 가드가 못 켜진다.
_FEATURES_CACHE_PREFIX = "company_agent_features:v2:"


async def fetch_company_agent_features(company: Optional[str]) -> dict:
    """common_app 의 Agent 패널 기능 플래그 {"web_search": bool, ...}. 실패·미설정은 {} (fail-closed).

    LLM 허용 프로바이더 조회(services/llm_settings/provider_policy)와 같은 서버-투-서버 호출·Redis 캐시다.
    """
    if not company:
        return {}
    cache_key = f"{_FEATURES_CACHE_PREFIX}{company}"
    try:
        from agent_arch.memory.redis_store import redis_store
        cached = await redis_store.get(cache_key)
        if isinstance(cached, dict):
            # 빈 캐시는 "조회는 됐는데 플래그가 하나도 없다"는 뜻이고, TTL 동안 웹이 닫힌
            # 채로 굳는다. 토글을 껐다 켜도 안 열리는 구간이 생기므로 눈에 보이게 남긴다.
            if not cached:
                logger.info("[web] 회사 기능 플래그 캐시가 비어 있음 (company=%s) — TTL 동안 닫힘", company)
            return cached
    except Exception:
        cached = None
    if not (settings.frappe_api_key and settings.frappe_api_secret):
        logger.debug("FRAPPE_API_KEY/SECRET 미설정 — Agent 기능 플래그 조회 불가(웹 검색 닫힘)")
        return {}
    url = f"{settings.common_app_url.rstrip('/')}/api/method/{_FEATURES_METHOD}"
    headers = {"Authorization": f"token {settings.frappe_api_key}:{settings.frappe_api_secret}"}
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, params={"company": company}, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.warning("common_app Agent 기능 플래그 조회 실패 (company=%s) — 웹 검색 닫힘", company, exc_info=True)
        return {}
    payload = data.get("message") if isinstance(data, dict) else None
    features = {k: bool(v) for k, v in payload.items()} if isinstance(payload, dict) else {}
    if not features:
        logger.info("[web] common_app 이 기능 플래그를 비워 돌려줌 (company=%s payload=%r)", company, payload)
    try:
        from agent_arch.memory.redis_store import redis_store
        await redis_store.set(cache_key, features, expire=int(getattr(settings, "company_providers_cache_ttl", 120) or 120))
    except Exception:
        pass
    return features


async def company_web_search_enabled(company: Optional[str]) -> bool:
    """회사가 웹 검색을 켰는가. .env 허용 목록이 있으면 그것이 결정(운영 강제), 없으면 common_app 설정."""
    raw = (getattr(settings, "web_search_allowed_companies", "") or "").strip()
    if raw:
        return company_allowed(company)
    return bool((await fetch_company_agent_features(company)).get("web_search"))


async def resolve_web_search(requested: Optional[bool], company: Optional[str]) -> bool:
    """이 요청의 유효 플래그 = 패널 토글 ∧ 회사 설정. 토글이 없으면 회사 설정을 조회하지도 않는다.

    ``requested`` 는 **정확히 True** 일 때만 켠다 — 엔드포인트 함수를 직접 부르는 경로(테스트 등)에서는 기본값이
    FastAPI 의 Form 객체로 들어오는데, truthy 라는 이유로 웹을 열면 안 된다.
    """
    if requested is not True:
        return False
    return await company_web_search_enabled(company)
