"""기억 → 에이전트 흐름 힌트 (결정론, LLM·임베딩 없음).

회상된 별칭은 "사용자가 지금 이 표현으로 어떤 대상을 가리켰다"는 확정 신호다. 그 대상의 종류
(canonical_kind)가 도메인을 못 박으면, 그 도메인의 도구를 **탐색 홉 없이** 이 턴에 미리 바인딩하고
라우터에 직전 도메인처럼 힌트를 준다. 실측(60일, 891 도구 턴): 도메인 탐색을 거친 턴은 LLM 호출
평균 4.34회, 안 거친 턴은 2.96회. find_channels 는 103턴에서 불렸는데 그 채널 id 는 대부분 별칭
기억에 이미 있었다.

강제가 아니다 — 도구를 손에 들려 주고 라우터에 기울기를 주는 것뿐이며, 모델은 여전히 다른 도구를
부르거나 탐색 도구를 쓸 수 있다.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# 별칭 대상 종류 → 도메인. 도메인이 애매한 종류(person·company·web_page·ui_element)는 넣지 않는다.
KIND_DOMAIN: dict[str, str] = {
    "mattermost_channel": "talk", "mattermost_dm": "talk", "mattermost_user": "talk", "channel": "talk",
    "mail_folder": "mail", "mailbox": "mail", "mail_label": "mail",
    "calendar_event": "calendar", "calendar": "calendar",
    "drive_file": "drive", "drive_folder": "drive", "document": "drive", "folder": "drive",
    "eapproval_doc": "eapproval", "approval_document": "eapproval",
    "logistics_order": "logistics", "purchase_order": "logistics", "item": "logistics",
}
MAX_TOOLS_PER_DOMAIN = 12

# 별칭 종류 → "id 를 받는 직접 조회 도구와 인자". 회상 줄에 붙여 모델이 id 를 어느 도구의 어느 인자에
# 넣을지 잇게 한다(실측: 채널 id 가 기억에 있어도 모델이 키워드 검색 도구를 골라 팀 해석에 실패).
# 참조 구현이 도구 안내를 사용법(레시피)까지 적는 것과 같은 결.
KIND_TOOL_HINT: dict[str, str] = {
    "mattermost_channel": "mattermost_skill_get_channel_messages(channel_id={id}) 로 바로 조회 — 팀·채널 키워드 해석 불필요",
    "mattermost_dm": "mattermost_skill_get_direct_messages 로 바로 조회",
}
# id 를 아직 모르는 별칭 — id 를 얻는 길을 적는다(라이브 API 도구). 얻으면 배치 판정이 payload 에 합친다.
KIND_TOOL_HINT_NO_ID: dict[str, str] = {
    "mattermost_channel": "channel_id 미확보 — mattermost_skill_find_channels(query=\"{name}\") 로 id 를 얻은 뒤 get_channel_messages",
}


def alias_tool_hint(kind: Optional[str], canonical_id: Optional[str], canonical_name: Optional[str] = None) -> str:
    """별칭 줄에 덧붙일 도구 사용 힌트. 종류를 모르면 빈 문자열."""
    if not kind:
        return ""
    k = str(kind).strip().lower()
    if canonical_id:
        tpl = KIND_TOOL_HINT.get(k)
        return tpl.format(id=canonical_id) if tpl else ""
    tpl = KIND_TOOL_HINT_NO_ID.get(k)
    return tpl.format(name=(canonical_name or "").replace('"', "'")) if (tpl and canonical_name) else ""


def alias_domains(recall, known_domains: Iterable[str]) -> list[str]:
    """이 턴에 어휘 매칭된 별칭들이 못 박는 도메인(중복 제거, 등장 순). 모르는 종류·없는 도메인은 무시."""
    known = set(known_domains or ())
    out: list[str] = []
    for a in getattr(recall, "aliases", None) or []:
        kind = str(((a.get("payload") or {}).get("canonical_kind")) or "").strip().lower()
        d = KIND_DOMAIN.get(kind)
        if d and d in known and d not in out:
            out.append(d)
    return out


def hint_domain(recall, known_domains: Iterable[str]) -> Optional[str]:
    """라우터 힌트로 쓸 도메인 — 별칭이 정확히 한 도메인을 가리킬 때만(둘 이상이면 판단을 라우터에)."""
    ds = alias_domains(recall, known_domains)
    return ds[0] if len(ds) == 1 else None


def select_tools_for_domains(
    domains: Iterable[str],
    tools: Iterable,
    domain_of: Callable[[str], Optional[str]],
    *,
    exclude: Callable[[str], bool] = lambda name: False,
    cap: int = MAX_TOOLS_PER_DOMAIN,
) -> list:
    """등록된 도구 중 해당 도메인 것만 고른다(탐색 도구 find_* 와 exclude 조건 제외, 도메인당 cap)."""
    wanted = list(dict.fromkeys(domains))
    if not wanted:
        return []
    per_domain: dict[str, list] = {d: [] for d in wanted}
    for t in tools:
        name = getattr(t, "name", None)
        if not name or name.startswith("find_") or exclude(name):
            continue
        d = domain_of(name)
        if d in per_domain and len(per_domain[d]) < cap:
            per_domain[d].append(t)
    out: list = []
    for d in wanted:
        out.extend(sorted(per_domain[d], key=lambda t: t.name))
    return out
