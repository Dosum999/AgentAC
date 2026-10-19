"""가이드 참고 링크 각주 — 도구 결과에 실제로 있는 URL 만, 코드가 결정적으로 붙인다.

문제(실측 2026-09-16~17, stg=vllm·로컬=gemini 모두): 프롬프트가 답변 말미 "🔗 참고 링크" 목록을 요구했는데
가이드 KB 청크 1388건 중 URL 이 든 청크는 20개 안팎이다. 모델은 섹션을 채우려고 라벨은 청크 제목으로, URL 자리는
`None`·제목을 영어로 번역한 가짜 경로(`example.com/user-guide/messenger-channel-settings-and-operations`)·
무의미 반복(`docs.example.com/652vpj8/652vpj8/…`)으로 메웠다. 형식이 원문에 없는 것을 요구하는 구조라 프롬프트
문구로는 무너지는 방식만 바뀐다.

해법(provenance.py 의 웹 출처 각주와 같은 패턴): 섹션 소유를 모델에서 코드로 옮긴다.
- 모델이 쓴 "참고 링크" 섹션(`- [라벨](URL)` 목록)은 sanitize.strip_reference_links_section 이 지운다.
- 턴을 끝내는 코드 지점(react 그래프 최종 결과·guide 루프)에서 이번 턴의 ``search_user_guide_kb`` 결과
  청크 본문에 있는 URL 을 뽑아, 답이 **참조한** 청크의 것만 `- URL` 로 나열한다(라벨 없음 — 라벨을 본문에서
  깎아내는 휴리스틱은 사례마다 어긋난다). 이 꼴은 정화기가 건드리지 않으므로 스트림·저장본 어디서도 살아남는다.
- URL 이 하나도 없으면 섹션이 존재하지 않는다. 지어낼 재료 자체가 없다.

도구 결과 모양은 GuideKBTool 이 닫아 둔다: ``[{"category", "category_top_score", "chunks": [{"title", "content",
"score", "permission_blocked", …}]}]`` (executor.truncate_result 가 20건 초과 리스트를 ``{"items": [...]}`` 로 감쌀
수 있다). ``page_url``(Confluence 원문 주소)은 직렬화되지 않으므로 여기서도 보지 않는다.
"""
from __future__ import annotations

import json
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit

from agent_arch.agents.react.provenance import _norm, is_referenced
from agent_arch.utils.sanitize import _BARE_URL_RE, _MD_LINK_RE, is_wiki_url

GUIDE_TOOL = "search_user_guide_kb"
FOOTER_HEADER = "🔗 참고 링크"
MAX_LINKS = 5

def _rows(data: Any) -> list:
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    return data if isinstance(data, list) else []


def _valid_url(raw: str) -> Optional[str]:
    """http(s)·실제 호스트(점 포함)·위키 아님. ``https://단말기IP`` 같은 플레이스홀더는 버린다."""
    url = (raw or "").strip().rstrip(".,;:!?)]}>\"'")
    if not url.startswith(("http://", "https://")):
        return None
    host = (urlsplit(url).hostname or "").lower()
    return url if "." in host and not is_wiki_url(url) else None


def _page_title(title: str) -> str:
    return (title or "").split(" - ")[0].strip()


def extract_guide_links(tool_results: Optional[Iterable[Any]]) -> list[dict]:
    """이번 턴 가이드 검색 결과 청크 **본문** 의 URL 목록(등장 순, URL 중복 제거).

    항목: ``{"url", "title", "site", "group_score"}``. 마크다운 링크·생 URL 둘 다 잡고, 권한으로 가려진 청크·
    이미지 링크·위키 주소·플레이스홀더 호스트는 제외한다. 라벨은 만들지 않는다 — 각주는 URL 만 나열한다.
    """
    by_url: dict[str, dict] = {}
    for tr in tool_results or []:
        name = tr.get("name") if isinstance(tr, dict) else getattr(tr, "name", None)
        content = tr.get("content") if isinstance(tr, dict) else getattr(tr, "content", None)
        if name != GUIDE_TOOL or not isinstance(content, str) or content.lstrip().startswith("[ERROR]"):
            continue
        try:
            data = json.loads(content)
        except Exception:
            continue
        for g in _rows(data):
            if not isinstance(g, dict):
                continue
            gscore = float(g.get("category_top_score") or 0)
            for c in _rows(g.get("chunks")):
                if not isinstance(c, dict) or c.get("permission_blocked") or "http" not in (c.get("content") or ""):
                    continue
                text = c["content"]
                raws = [m.group(3) for m in _MD_LINK_RE.finditer(text) if not m.group(1)]
                covered = [(m.start(), m.end()) for m in _MD_LINK_RE.finditer(text)]
                raws += [m.group(0) for m in _BARE_URL_RE.finditer(text) if not any(s <= m.start() < e for s, e in covered)]
                for raw in raws:
                    url = _valid_url(raw)
                    if not url:
                        continue
                    e = by_url.setdefault(url, {"url": url, "title": c.get("title") or "",
                                                "site": urlsplit(url).hostname or "", "group_score": gscore})
                    e["group_score"] = max(e["group_score"], gscore)
    return list(by_url.values())


def select_guide_links(content: Optional[str], links: list[dict]) -> list[dict]:
    """답이 **참조한** 페이지(제목·URL·호스트가 본문에 있음)의 링크. 없으면 최상위 점수 그룹의 링크로 폴백."""
    if not links:
        return []
    body = _norm(content or "")
    chosen = [l for l in links
              if is_referenced({"url": l["url"], "title": _page_title(l["title"]), "site": l["site"], "body": ""}, body)]
    if not chosen:
        top = max(l["group_score"] for l in links)
        chosen = [l for l in links if l["group_score"] == top]
    return chosen


def guide_links_footer(content: Optional[str], tool_results: Optional[Iterable[Any]]) -> str:
    """답에 아직 없는 가이드 URL 만 각주로. 붙일 게 없으면 빈 문자열."""
    links = extract_guide_links(tool_results)
    body = content or ""
    missing = [l for l in select_guide_links(body, links) if l["url"] not in body]
    if not missing:
        return ""
    return "\n".join(["", "", FOOTER_HEADER] + [f"- {l['url']}" for l in missing[:MAX_LINKS]])


def apply_guide_links_footer(result: Any) -> str:
    """LoopResult 의 content 에 각주를 붙이고(제자리, 끝 공백 정리) 붙인 각주를 돌려준다. 없으면 빈 문자열.

    반드시 sanitize_tool_names **뒤**에 호출한다 — 정화가 모델이 쓴 섹션을 지운 자리에 코드 각주가 붙는다.
    """
    footer = guide_links_footer(getattr(result, "content", ""), getattr(result, "tool_results", None))
    if footer:
        result.content = (result.content or "").rstrip() + footer
    return footer
