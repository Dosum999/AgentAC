"""브레이브 검색 MCP 도구 어댑터 — 원격 호출 앞뒤에 **회사 문지기**와 **닫힌 결과 형식**을 붙인다.

검색 자체는 브레이브 검색 MCP 서버(deploy/brave-mcp, brave/brave-search-mcp-server)가 한다. 이 클래스가 하는 일은
세 가지뿐이다.
1. 호출 전: 회사 허용 목록 → 사내 식별자 차단 → 일일 한도 (services/web/guard.precheck). 막히면 구조화 오류를 돌려
   모델이 사내 도구로 돌아가거나 검색어를 고칠 수 있게 한다.
2. 호출: 정책 인자 병합(기본값·강제값·숨김)은 RemoteMCPTool 이 한다.
3. 호출 후: 도구마다 다른 MCP 응답을 한 모양(`kind: results|context`, `results[].{title,url,published,content}`)으로 접고
   본문은 신뢰 불가 봉투에 넣는다(services/web/brave_output). 결과 없음은 오류가 아니라 0건이다.
감사 원장에는 막힌 호출·오류·성공(건수)을 모두 남긴다.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from agent_arch.agents.context import get_user_context, remember_web_urls, web_search_enabled
from agent_arch.agents.tools.base import ToolResult
from agent_arch.agents.tools.mcp.remote_tool import RemoteMCPTool
from agent_arch.agents.tools.mcp.transport import user_context_headers
from agent_arch.services.web import brave_output, guard
from agent_arch.services.usage.meter import METER_WEB_SEARCH, record_meter
from agent_arch.services.monitoring.error_events import event_extra

logger = logging.getLogger(__name__)


class BraveWebTool(RemoteMCPTool):

    async def _real_execute(self, **kwargs) -> ToolResult:
        user: Dict[str, Any] = get_user_context() or {}
        company = user.get("company")
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return ToolResult(success=False, output=None, error="query 가 필요합니다.", metadata={"tool": self._name, "reason": "input"})

        if not web_search_enabled():
            # 관문이 실리지 않아 보통은 여기 못 오지만, sticky·직접 지정(tools=) 경로의 마지막 방어선.
            await guard.audit(company, user, self._name, query, "web_search_off")
            msg = "웹 검색이 꺼져 있습니다. 패널의 웹 검색을 켜면(회사 설정에서 허용된 경우) 사용할 수 있습니다. 지금은 사내 도구나 아는 범위에서 답하세요."
            return ToolResult(success=False, output={"error": "web_search_off", "message": msg}, error=msg,
                              metadata={"tool": self._name, "code": "web_search_off"})

        blocked = await guard.precheck(query, user, self._name)
        if blocked:
            return ToolResult(success=False, output=blocked, error=blocked["message"],
                              metadata={"tool": self._name, "code": blocked["error"]})

        try:
            result = await self._call_remote(self._merge_arguments(kwargs), user_context_headers(user))
        except Exception as e:  # noqa: BLE001 — 전송·서버 장애
            logger.warning("[web] MCP 호출 실패 name=%s: %s: %r", self._name, type(e).__name__, e,
                           extra=event_extra("mcp_tool", exc=e, mcp_server="brave"))
            await guard.audit(company, user, self._name, query, "transport_error")
            return ToolResult(success=False, output={"error": "provider_unavailable", "message": "웹 검색 서버에 연결할 수 없습니다."},
                              error="웹 검색 서버에 연결할 수 없습니다.", metadata={"tool": self._name, "code": "provider_unavailable"})

        texts = [c.text for c in (result.content or []) if getattr(c, "text", None)]
        payload = brave_output.normalize(self._name, query, structured=result.structured_content, texts=texts,
                                         is_error=bool(getattr(result, "is_error", False)))
        if "error" in payload:
            if payload["error"] == "provider_credit_exhausted":
                logger.error("[web] 브레이브 선불 크레딧 소진 — 웹 검색 전면 중단, 충전 필요 (tool=%s)", self._name)
            await guard.audit(company, user, self._name, query, payload["error"])
            return ToolResult(success=False, output=payload, error=payload.get("message"),
                              metadata={"tool": self._name, "code": payload["error"]})
        await guard.audit(company, user, self._name, query, f"ok:{payload.get('count', 0)}")
        # 계측 — Brave 는 성공한 요청 **건수**로 청구한다. 가드가 막았거나 전송에 실패한 호출은
        # 요청이 나가지 않았거나 과금되지 않으므로 여기(성공)에서만 센다.
        record_meter(METER_WEB_SEARCH, provider="brave", model_name="brave-search",
                     quantity=1, source=f"brave_mcp.{self._name}")
        remember_web_urls((r.get("url"), r.get("title") or "") for r in payload.get("results") or []
                          if isinstance(r, dict) and r.get("url"))   # web_fetch 로 열 수 있는 주소(+ 막혔을 때 인덱스 재조회용 제목)
        return ToolResult(success=True, output=payload,
                          metadata={"tool": self._name, "kind": payload.get("kind"), "count": payload.get("count", 0)})
