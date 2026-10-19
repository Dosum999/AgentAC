"""가이드 발 링크 차단 — 근거 없는 주소는 답변에 남기지 않는다.

문제(2026-09-21 신고): "에이전트 사용방법 안내해줘" 답변 말미에 ``더 자세한 내용은 아래 가이드를
참고해 주세요.`` + ``[Agent 사용 심층 가이드](…)`` 같은 **가이드 문서 링크**가 붙는데, 눌러 보면
엉뚱한 페이지로 가거나 접속이 안 된다. 주소는 **모델이 지어낸 것**이다 — Confluence 원문 주소
(``page_url``)는 LLM 에 주지 않고(``tools/guide_kb.py``), 청크 본문의 위키 링크도 3층에서 지운다
(색인 ``html_normalizer`` · 도구 직렬화 · 출력 ``sanitize.strip_wiki_links``). 즉 **진짜 가이드 주소는
애초에 모델에게 가지 않는데**, 프롬프트가 "본문에 있는 URL 은 링크로 넣어라"고 허가하니 모델이
제목만 아는 페이지의 주소를 지어내 빈칸을 메운다. ``react/guide_links.py`` 가 2026-09-16~17 에 답변
말미 '참고 링크 섹션'에서 잡은 것과 **같은 실패가 본문 인라인에서** 재발한 것이다.

정책: 가이드 검색을 쓴 턴에서는 **모델이 실제로 본 텍스트에 그대로 있던 주소만** 남긴다.
  - 이번 턴 도구 결과 본문 — 가이드 청크에 적힌 문의 폼 주소, ``mattermost_open_screen`` 의 채널 url,
    웹 검색 결과 url 등
  - 시스템 프롬프트가 코드로 심는 고정 링크(다운로드 센터·앱스토어)
  - 사용자가 직접 적어 보낸 주소
그 밖의 주소는 근거가 없으므로 지운다(``sanitize.strip_unallowed_links``). 가이드 페이지 주소는
어느 쪽에도 없으니 자동으로 사라지고, 실무 주소는 살아남는다.

**호스트 화이트리스트는 쓰지 않는다.** 실측된 환각 주소가 ``https://example.com/user-guide/…`` 라
진짜 주소와 **같은 호스트**다 — 호스트로 열면 그대로 뚫린다. 그래서 정확일치가 기본이고, 예외는
메신저 호스트 하나뿐이다: ``react/messenger_links._download_url()`` 이 ``{base}/api/v4/files/{id}`` 를
**코드로 조립**해 도구 결과 문자열에는 없기 때문이다(그 링크의 진위는 ``hitl/answer_guard`` 가 따로 소유).

대화 이력은 근거에 넣지 않는다 — 과거에 새어 나간 가짜 주소가 화이트리스트가 된다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

from agent_arch.core.config import settings
from agent_arch.utils.sanitize import _BARE_URL_RE, _MD_LINK_RE

#: 이 도구가 돈 턴에만 정책이 켜진다 — 가이드와 무관한 턴의 링크는 지금처럼 그대로 둔다.
GUIDE_TOOL = "search_user_guide_kb"

#: 시스템 프롬프트(`prompts/system.py`)가 "답변에 그대로 노출하라"고 적어 둔 고정 링크.
APPSTORE_URL = "https://apps.apple.com/kr/app/chat-app/id6782740913"

#: 스트림으로 실어 보내는 허용 목록 상한 — 이벤트 payload 가 부풀지 않게.
MAX_ALLOWED = 200

_TRAILING = ".,;:!?)]}>\"'"


def normalize_url(raw: Optional[str]) -> str:
    """비교용 정규화. http(s)·실호스트가 아니면 빈 문자열(= 이 가드의 대상 아님).

    모델이 같은 주소를 조금 다르게 옮겨 적었다고 지우면 실무 링크가 죽는다. 그래서 꼬리
    문장부호·fragment·경로 끝 ``/`` 를 떼고 scheme/host 를 소문자로 맞춘 뒤 비교한다.
    쿼리는 의미가 있으므로 유지한다.
    """
    url = (raw or "").strip().rstrip(_TRAILING)
    if not url.lower().startswith(("http://", "https://")):
        return ""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        return ""
    if "." not in host:
        return ""   # `https://단말기IP` 같은 플레이스홀더
    netloc = f"{host}:{port}" if port else host
    return urlunsplit((parts.scheme.lower(), netloc, parts.path.rstrip("/"), parts.query, ""))


def harvest_urls(text: Optional[str]) -> set:
    """텍스트에 그대로 등장하는 주소들(정규화). 마크다운 링크·생 URL 둘 다 잡는다."""
    out: set = set()
    if not text or "http" not in text:
        return out
    for m in _MD_LINK_RE.finditer(text):
        if (n := normalize_url(m.group(3))):
            out.add(n)
    for m in _BARE_URL_RE.finditer(text):
        if (n := normalize_url(m.group(0))):
            out.add(n)
    return out


def static_allowed() -> set:
    """코드가 프롬프트에 심는 고정 링크 — 도구 결과에는 없지만 노출이 **의도**된 주소.

    다운로드 센터 주소는 프롬프트와 **같은 함수**에서 받는다(`DMP_PORTAL_URL` 이 바뀌어도
    프롬프트와 가드가 어긋나지 않게). 호스트가 아니라 정확일치로만 연다.
    """
    out = {normalize_url(APPSTORE_URL)}
    try:
        # 지연 import — 순환 참조 회피 + 프롬프트 쪽이 바뀌어도 정화가 죽지 않게(고정 링크 하나가 빠질 뿐).
        from agent_arch.agents.prompts.system import _render_download_center_url

        out.add(normalize_url(_render_download_center_url()))
    except Exception:
        pass
    return {u for u in out if u}


def _messenger_host() -> str:
    base = (settings.mattermost_base_url or "").strip()
    if not base:
        return ""
    try:
        return (urlsplit(base).hostname or "").lower()
    except ValueError:
        return ""


@dataclass
class TurnLinkPolicy:
    """한 턴의 링크 허용 판정. ``permits`` 를 정화기에 콜백으로 넘긴다.

    ``active`` 가 False 면 **모든 주소를 통과**시킨다 — 가이드 검색을 쓰지 않은 턴의 동작은
    이번 변경 전과 완전히 같다(변경 범위를 좁히려는 의도된 설계).
    """

    active: bool = False
    allowed: set = field(default_factory=set)

    def observe(self, name: Optional[str], content: Any) -> None:
        """도구 결과 하나를 본다 — 가이드 도구면 정책을 켜고, 본문의 주소는 허용에 담는다."""
        if name == GUIDE_TOOL:
            self.active = True
        if isinstance(content, str):
            self.allowed |= harvest_urls(content)

    def observe_results(self, tool_results: Optional[Iterable[Any]]) -> None:
        for tr in tool_results or []:
            if isinstance(tr, dict):
                self.observe(tr.get("name"), tr.get("content"))
            else:
                self.observe(getattr(tr, "name", None), getattr(tr, "content", None))

    def allow_text(self, text: Optional[str]) -> None:
        """도구 결과가 아닌 근거(시스템 프롬프트·사용자 입력)의 주소를 허용에 담는다."""
        self.allowed |= harvest_urls(text)

    def permits(self, url: Optional[str]) -> bool:
        if not self.active:
            return True
        n = normalize_url(url)
        if not n:
            return True     # http(s) 아닌 것·플레이스홀더는 이 가드의 대상이 아니다
        if n in self.allowed or n in static_allowed():
            return True
        host = (urlsplit(n).hostname or "").lower()
        mh = _messenger_host()
        return bool(mh) and host == mh

    def dumps(self) -> str:
        """스트림 이벤트로 실어 보낼 스냅샷(JSON). 매번 전문이라 순서·중복에 강하다."""
        return json.dumps(
            {"active": self.active, "allowed": sorted(self.allowed)[:MAX_ALLOWED]},
            ensure_ascii=False,
        )

    def update_from(self, payload: Optional[str]) -> None:
        """스냅샷을 **제자리** 병합한다 — 이미 정화기에 바인딩된 ``permits`` 가 계속 유효하다."""
        try:
            data = json.loads(payload or "{}")
        except (ValueError, TypeError):
            return
        if not isinstance(data, dict):
            return
        self.active = bool(data.get("active")) or self.active
        for u in data.get("allowed") or []:
            if isinstance(u, str) and u:
                self.allowed.add(u)


def policy_for(tool_results: Optional[Iterable[Any]], *, texts: Iterable[Any] = ()) -> TurnLinkPolicy:
    """최종 본문 정화용 — 도구 결과와 부가 근거(시스템 프롬프트·사용자 입력)로 정책을 만든다."""
    policy = TurnLinkPolicy()
    policy.observe_results(tool_results)
    for t in texts:
        policy.allow_text(t if isinstance(t, str) else None)
    return policy
