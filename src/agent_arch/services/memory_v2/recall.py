"""회상 — 한 함수, 두 블록. 위층(페르소나·장면 내비)은 system, 턴별(전역 지시·별칭·관련 atom)은 마지막 user.

분할 근거: system 프롬프트는 대화 이력 전체의 프리픽스 캐시 열쇠다. 페르소나·내비는 배치마다
드물게 바뀌지만 관련 atom 은 질의마다 다르다. 가변 블록을 마지막 user 메시지에 붙이면 system +
이력이 바이트 동일해 캐시가 산다(2026-09-02 실측: 적중 70.7%→82.7%).

실패 정책: 전체 1.5초 타임아웃, 초과·예외 시 도구 안내만 담은 블록으로 돌아간다 — 메모리는
턴을 깨지 않는다. 개별 소스(PG/Qdrant) 실패는 그 소스만 비운다.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from agent_arch.core.config import settings
from agent_arch.db.repository import memory_v2 as repo
from agent_arch.memory.redis_store import redis_store
from agent_arch.services.memory_v2 import place as place_mod

logger = logging.getLogger(__name__)

RECALL_TIMEOUT_SECONDS = 1.5
# 콜드(안정 블록 캐시 미스) 예산. 이때만 소스가 5개고 임베딩 API 왕복이 들어가 실측 2~3초까지
# 간다 — 웜과 같은 1.5초를 주면 사용자당 30분에 한 번씩 첫 턴이 페르소나·전역지시 없이 답한다.
# 웜(2 소스)은 430~570ms 라 1.5초로 충분하다.
RECALL_TIMEOUT_COLD_SECONDS = 3.0
STABLE_CACHE_TTL_SECONDS = 1800   # 안정 블록 캐시(세션 초기화형). 쓰기 경로가 무효화하므로 TTL 은 안전망
MAX_STRICT = 8
MAX_ALIAS_POOL = 60
MAX_TURN_ATOMS = 5
MAX_TURN_CHARS = 1200
NAV_SCENES = 8
RRF_K = 60
# 관련성 하한(튜닝 대상). 한국어 짧은 임베딩은 기본 유사도가 높아 하한 없이 top-k 만 쓰면
# 무관 기억이 매 턴 딸려온다. 실측(gemini-embedding-001, 2026-09-07): 관련 질의 0.70~0.78,
# 무관 질의("매출 보여줘","회의실 예약","안녕") 0.55~0.675 → 0.70 이 경계. 주입은 정밀도 우선이고,
# 더 깊은 탐색은 에이전트가 memory_search(하한 0.55)로 한다.
VECTOR_MIN_SCORE = 0.70
KEYWORD_MIN_SCORE = 0.3
TOOL_VECTOR_MIN_SCORE = 0.55
TOOL_KEYWORD_MIN_SCORE = 0.2

TOOLS_GUIDE = (
    "[장기 메모리] 위에 주입된 기억으로 부족할 때만 도구로 더 찾는다 — memory_search(정제 기억)는 바로 부르고, "
    "recall_conversation(대화 원문)·memory_read_scene(장면 문서 전문)은 find_memory_tools 로 꺼내 쓴다. "
    "한 턴에 검색은 합쳐서 3회까지, "
    "3회 뒤에도 없으면 기억에 없는 것이니 아는 범위에서 답한다. "
    "사용자가 '앞으로 ~해/기억해둬'라고 하면 memory_remember 로 저장하고, 저장된 기억을 지우거나 고쳐 달라고 하면 "
    "memory_search 로 대상을 찾아 memory_forget 으로 지운다('저장 도구가 없다'고 답하지 말 것). "
    "값이 변하는 정보(일정·근태·재고)는 기억이 아니라 도구로 최신 조회한다. "
    "별칭에 (id: …)가 있으면 이름·키워드로 다시 검색하지 말고 그 id 를 받는 조회 도구를 먼저 쓴다."
)
# 장면 내비게이션 헤더 — 도구가 없는 채널에서는 도구 언급을 뺀 문구로 갈아 끼운다.
_SCENES_HEADER = "● 기억 장면(관련 있으면 memory_read_scene(name) 으로 전문 열람):"
_SCENES_HEADER_TOOLLESS = "● 기억 장면(과거 기록 요약):"


def strip_tool_guidance(system_block: str) -> str:
    """안정 블록에서 **메모리 도구를 가리키는 지시**만 걷어낸다.

    메모리 도구를 바인딩하지 않는 채널(가이드 모드, MES 앱·SCMS 전용 채널)에서 이
    안내를 그대로 주면 부를 수 없는 도구를 가리키는 죽은 지시가 된다. 특히
    ``TOOLS_GUIDE`` 는 "'저장 도구가 없다'고 답하지 말 것" 까지 못 박아, "이거 기억해둬"
    요청에서 모델을 막다른 길로 몬다.

    페르소나·전역 지시·장면 요약은 도구와 무관하게 유효하므로 그대로 남긴다.

    Args:
        system_block: ``format_system_block`` 이 만든 안정 블록.

    Returns:
        도구 지시를 뺀 블록. 입력이 비어 있으면 그대로 돌려준다.
    """
    if not system_block:
        return system_block
    out = system_block.replace(TOOLS_GUIDE, "", 1)
    out = out.replace(_SCENES_HEADER, _SCENES_HEADER_TOOLLESS)
    return out.strip()


TURN_HEADER = (
    "[관련 기억] 이번 질의로 회상된 기억이다. "
    "사용자가 이번 메시지에서 다르게 요청하면 그 요청이 우선한다."
)
# 턴 블록의 두 갈래. 사용자가 정한 규칙(instruction)은 과거 기록과 권위가 다르다 — 한데 섞어
# "참고용"으로 실으면 도구가 "사용자에게 받은 값만 써라"고 할 때 규칙이 기록으로 밀려 같은 값을
# 또 묻는다(실측 2026-09-28: 결재선·결제수단 기억이 주입됐는데 경비 상신에서 재질문).
_TURN_INSTRUCTIONS_HEADER = (
    "● 사용자가 정한 규칙(사용자가 직접 말한 것 — 해당하는 작업이면 이 값을 사용자가 준 값으로 쓴다):"
)
_TURN_ATOMS_HEADER = "● 관련 기억(과거 기록 — 현재 작업 상태가 아님, 참고용):"


@dataclass
class RecallResult:
    system_block: str = ""
    turn_block: str = ""
    strict: list[dict] = field(default_factory=list)
    aliases: list[dict] = field(default_factory=list)
    atoms: list[dict] = field(default_factory=list)
    persona: Optional[str] = None
    scenes: list[dict] = field(default_factory=list)
    timed_out: bool = False
    error: Optional[str] = None


# ─── 별칭 어휘 매칭 (v1 검증본 이식) ──────────────────────────────────────

# 별칭 뒤에 바로 붙을 수 있는 흔한 한글 조사 — 이 형태는 유효 매칭으로 인정한다.
_JOSA = (
    "으로부터", "으로서", "으로써", "에서는", "에게서",
    "에서", "에게", "한테", "께서", "으로", "부터", "까지", "보다", "처럼", "만큼",
    "이랑", "이라도", "라도", "이나", "마다", "조차", "밖에", "이라는", "라는", "이라",
    "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "도", "만", "로",
    "랑", "나", "야", "아", "여", "께", "이며", "이고",
)


def alias_hit(alias: str, q: str) -> bool:
    """별칭이 질의에 '독립된 표현'으로 등장하는가(경계 + 조사 규칙).

    단순 부분문자열은 "회사"가 "회사미디어" 안에서도 걸린다. 왼쪽은 단어문자가 아니어야 하고,
    오른쪽은 끝·비단어문자·한글 조사 중 하나여야 한다. isalnum 이 한글/영문/숫자를 포괄한다.
    """
    n = len(alias)
    start = 0
    while True:
        i = q.find(alias, start)
        if i < 0:
            return False
        before = q[i - 1] if i > 0 else ""
        after = q[i + n:]
        left_ok = (before == "") or (not before.isalnum())
        right_ok = (after == "") or (not after[0].isalnum()) or any(after.startswith(j) for j in _JOSA)
        if left_ok and right_ok:
            return True
        start = i + 1


def match_aliases(query: str, alias_atoms: Iterable[dict]) -> list[dict]:
    """질의에 별칭 표현이 실제 등장하는 alias atom 만 고른다(의미 검색 아님 — 짧은 한국어 임베딩은 무관 매칭이 잦다)."""
    q = (query or "").lower()
    if not q:
        return []
    out: list[dict] = []
    for a in alias_atoms:
        for alias in (a.get("payload") or {}).get("aliases") or []:
            if alias and alias_hit(str(alias).lower(), q):
                out.append(a)
                break
    return out


def format_alias(atom: dict) -> str:
    """별칭 줄. id 가 있고 종류가 알려진 조회 도구에 대응하면 "어느 도구의 어느 인자에 넣으라"까지 적는다."""
    from agent_arch.services.memory_v2.loadout import alias_tool_hint

    p = atom.get("payload") or {}
    aliases = ", ".join(f'"{a}"' for a in (p.get("aliases") or [])) or "?"
    name = p.get("canonical_name") or "?"
    kind = p.get("canonical_kind") or ""
    cid = p.get("canonical_id")
    line = f"{aliases} → {(kind + ' ') if kind else ''}'{name}'{f' (id: {cid})' if cid else ''}"
    hint = alias_tool_hint(kind, cid, name if name != "?" else None)
    return f"{line} → {hint}" if hint else line


# ─── 하이브리드 검색 (키워드 + 벡터, RRF) ────────────────────────────────

async def _vector_search(user_sub: str, query: str, k: int, atom_type: Optional[str]) -> list[dict]:
    try:
        from agent_arch.services.memory_v2.indexer import get_v2_indexer
        idx = await get_v2_indexer()
        return await idx.search(user_sub, query, top_k=k, atom_type=atom_type)
    except Exception as e:
        logger.debug("memory_v2 벡터 검색 실패(키워드만 사용): %s", e)
        return []


async def hybrid_search(
    user_sub: str,
    query: str,
    *,
    limit: int,
    atom_type: Optional[str] = None,
    exclude_types: tuple[str, ...] = (),
    keyword_min: float = KEYWORD_MIN_SCORE,
    vector_min: float = VECTOR_MIN_SCORE,
    place: Optional[str] = None,
) -> list[dict]:
    """pg_trgm 과 Qdrant 를 병렬로 돌려 RRF(k=60)로 융합한 상위 limit 개(하이드레이트된 atom dict + score)."""
    q = (query or "").strip()
    if not q:
        return []
    k = max(limit * 3, 5)
    kw_rows, vec_hits = await asyncio.gather(
        repo.search_atoms_keyword(user_sub, q, limit=k, atom_type=atom_type),
        _vector_search(user_sub, q, k, atom_type),
        return_exceptions=True,
    )
    if isinstance(kw_rows, BaseException):
        logger.debug("memory_v2 키워드 검색 실패: %s", kw_rows)
        kw_rows = []
    if isinstance(vec_hits, BaseException):
        vec_hits = []
    kw_rows = [r for r in kw_rows if float(r.get("score") or 0) >= keyword_min]
    vec_hits = [h for h in vec_hits if float(h.get("score") or 0) >= vector_min]

    fused: dict[str, float] = {}
    for rank, r in enumerate(kw_rows):
        fused[r["id"]] = fused.get(r["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, h in enumerate(vec_hits):
        fused[h["id"]] = fused.get(h["id"], 0.0) + 1.0 / (RRF_K + rank + 1)
    if not fused:
        return []

    by_id: dict[str, dict] = {r["id"]: r for r in kw_rows}
    missing = [i for i in fused if i not in by_id]
    if missing:
        for a in await repo.get_active_atoms_by_ids(user_sub, missing):
            by_id[a["id"]] = a
    # 장소 배수는 융합 점수에만 건다 — 후보를 떨어뜨리지 않는다. 다른 채널 기억도 강하게 맞으면
    # 여전히 올라오고, 장소가 없는 기억은 배수 1 이라 이 기능 이전과 순위가 같다.
    scored = [(aid, score * place_mod.rank_factor(by_id.get(aid), place)) for aid, score in fused.items()]
    out: list[dict] = []
    for aid, score in sorted(scored, key=lambda kv: kv[1], reverse=True):
        a = by_id.get(aid)
        if not a or a.get("atom_type") in exclude_types:
            continue
        out.append({**a, "score": round(score, 6)})
        if len(out) >= limit:
            break
    return out


async def search_atoms(user_sub: str, query: str, *, atom_type: Optional[str] = None, limit: int = 8) -> list[dict]:
    """memory_search 도구용 — 하한을 낮춰 넓게 찾는다(에이전트가 결과를 스스로 걸러 쓴다)."""
    return await hybrid_search(
        user_sub, query, limit=limit, atom_type=atom_type,
        keyword_min=TOOL_KEYWORD_MIN_SCORE, vector_min=TOOL_VECTOR_MIN_SCORE,
    )


# ─── 블록 포매터 ───────────────────────────────────────────────────────────

def format_system_block(persona: Optional[str], scenes: list[dict], strict: Optional[list[dict]] = None) -> str:
    """안정 블록: 도구 안내 + 전역 지시(priority -1) + 페르소나(저권위 프레이밍) + 장면 내비게이션(heat 순).

    전역 지시만 여기 둔다 — 질의와 무관하게 늘 적용되는 규칙이고 드물게 바뀌어 캐시를 거의 깨지 않는다.
    그 외 지시는 다른 atom 과 같이 관련도로 회상되고, 반복되면 페르소나(상호작용 규약)가 흡수한다.
    """
    lines = [TOOLS_GUIDE]
    if strict:
        lines.append("● 전역 지시(사용자가 못 박은 규칙 — 이번 메시지에서 다르게 요청하면 그 요청 우선):")
        lines.extend(f"  - {a.get('content')}" for a in strict)
    if persona and persona.strip():
        lines.append("● 사용자 페르소나(대화에서 추론된 요약 — 사실로 단정 말 것, 이번 요청과 어긋나면 이번 요청 우선):")
        lines.extend(f"  {ln}" for ln in persona.splitlines() if ln.strip())
    if scenes:
        lines.append(_SCENES_HEADER)
        for s in scenes:
            summary = (s.get("summary") or "").strip()
            lines.append(f"  - {s.get('name')}: {s.get('title')}" + (f" — {summary}" if summary else ""))
    return "\n".join(lines)


def _fmt_atom(a: dict) -> str:
    scene = a.get("scene_name") or "-"
    return f"  - [{a.get('atom_type')}|{scene}] {a.get('content')}"


def format_turn_block(strict: list[dict], aliases: list[dict], atoms: list[dict], *, max_chars: int = MAX_TURN_CHARS) -> str:
    """가변 블록: 별칭 → 관련 기억. 비면 빈 문자열(주입 생략). 전역 지시는 system 안정 블록에 있다."""
    if not (aliases or atoms):
        return ""
    lines = [TURN_HEADER]
    if aliases:
        lines.append("● 별칭(사용자 표현 → 실제 대상 — id 가 있으면 도구 인자에 바로 쓴다):")
        lines.extend(f"  - {format_alias(a)}" for a in aliases)
    # 규칙을 먼저 싣는다 — 예산이 모자라면 과거 기록이 먼저 잘린다.
    rules = [a for a in atoms if a.get("atom_type") == "instruction"]
    records = [a for a in atoms if a.get("atom_type") != "instruction"]
    used = sum(len(ln) for ln in lines)
    for header, group in ((_TURN_INSTRUCTIONS_HEADER, rules), (_TURN_ATOMS_HEADER, records)):
        if not group:
            continue
        section: list[str] = []
        for a in group:
            ln = _fmt_atom(a)
            if used + len(header) + len(ln) > max_chars:
                break
            section.append(ln)
            used += len(ln)
        if section:
            lines.append(header)
            used += len(header)
            lines.extend(section)
    return "\n".join(lines)


def recalled_user_rules(recall: Optional["RecallResult"]) -> list[str]:
    """이 턴에 회상된 **사용자가 정한 규칙** 문장 — 전역 지시 + 관련도로 회상된 지시(instruction).

    도구 결과 뒤 리마인더(``rule_reminder``)의 재료다. 과거 기록·페르소나는 규칙이 아니라 싣지 않는다.
    """
    if recall is None:
        return []
    out: list[str] = []
    for a in [*(getattr(recall, "strict", None) or []), *(getattr(recall, "atoms", None) or [])]:
        if a.get("atom_type") != "instruction":
            continue
        text = str(a.get("content") or "").strip()
        if text and text not in out:
            out.append(text)
    return out


RULE_REMINDER_MAX_CHARS = 600
_RULE_REMINDER_HEADER = "[사용자가 정한 규칙 — 이번 작업에 해당하면 적용]"
_RULE_REMINDER_FOOTER = (
    "위 규칙에 있는 값은 사용자가 이미 준 값이다. 도구가 그 값을 아직 받아야 할 값으로 나열해도 "
    "사용자에게 다시 묻지 말고 도구 인자에 담아 진행하라. 이번 메시지에서 다르게 말한 항목만 이번 "
    "메시지 값을 따르고, 나머지 항목은 규칙 값을 그대로 쓴다. 규칙에 없는 값만 묻는다."
)


def rule_reminder(recall: Optional["RecallResult"]) -> str:
    """도구 결과 뒤에 덧붙일 규칙 리마인더. 이 턴에 회상된 규칙이 없으면 빈 문자열(비용 0).

    도구와 무관한 공용 장치다 — ReAct 루프가 도구 결과를 대화에 붙이는 한 지점에서 쓴다.
    모델은 무엇을 되물을지 고르는 순간 가장 가까운 도구 데이터(받아야 할 값 목록)를 따른다.
    규칙이 system·user 끝의 기억 블록에만 있으면 그 목록에 밀려 같은 값을 또 묻는다(실측
    2026-09-28 경비 상신: 기억 블록만으로 16회 중 9회). 판단 직전 자리에 규칙을 다시 둔다.
    """
    rules = recalled_user_rules(recall)
    if not rules:
        return ""
    lines = [_RULE_REMINDER_HEADER]
    used = len(_RULE_REMINDER_HEADER)
    for r in rules:
        ln = f"- {r}"
        if used + len(ln) > RULE_REMINDER_MAX_CHARS:
            break
        lines.append(ln)
        used += len(ln)
    if len(lines) == 1:
        return ""
    lines.append(_RULE_REMINDER_FOOTER)
    return "\n".join(lines)


# ─── 안정 블록 캐시 (세션 초기화형) ─────────────────────────────────────────
#
# 페르소나·장면 내비·전역 지시는 배치가 돌 때만 바뀐다. 매 턴 PG 를 세 번 읽는 대신 사용자별로
# 한 번 만들어 Redis 에 두고, 쓰기 경로(L1 쓰기·L2/L3 재빌드·명시 저장·forget)가 지운다.
# 부산물로 system 프롬프트가 턴 사이에 바이트 단위로 같아져 vLLM 프리픽스 캐시에도 유리하다.

def _stable_key(user_sub: str) -> str:
    return f"memory2:stable:{user_sub}"


async def _get_stable_cache(
    user_sub: str, company_scope: Optional[str], here: Optional[str] = None
) -> Optional[dict]:
    try:
        cached = await redis_store.get(_stable_key(user_sub))
    except Exception:
        return None
    if not isinstance(cached, dict) or cached.get("company") != (company_scope or None):
        return None
    # 장소에 매인 전역 지시가 하나라도 있으면 그 블록은 장소마다 다르다. 다른 장소에서 쓰면
    # 채널 전용 규칙이 30분 동안 아무 자리에나 따라붙는다 — 그 경우만 미스로 친다.
    bound = cached.get("place_bound")
    if bound is not None and bound != here:
        return None
    if not isinstance(cached.get("block"), str):
        return None
    return cached


async def _set_stable_cache(
    user_sub: str, company_scope: Optional[str], block: str, strict_ids: list[str],
    place_bound: Optional[str] = None,
) -> None:
    try:
        await redis_store.set(
            _stable_key(user_sub),
            {"company": company_scope or None, "block": block, "strict_ids": strict_ids,
             "place_bound": place_bound},
            expire=STABLE_CACHE_TTL_SECONDS,
        )
    except Exception as e:
        logger.debug("memory_v2 안정 블록 캐시 저장 실패(무시): %s", e)


async def invalidate_stable_block(user_sub: str) -> None:
    """쓰기 경로가 호출한다 — 다음 턴이 안정 블록을 다시 만든다."""
    try:
        await redis_store.delete(_stable_key(user_sub))
    except Exception as e:
        logger.debug("memory_v2 안정 블록 캐시 무효화 실패(무시): %s", e)


# ─── 회상 진입점 ───────────────────────────────────────────────────────────

async def _safe(coro, default, *, name: str, failures: Optional[list] = None):
    """소스 하나를 감싼다. 실패하면 기본값으로 대체하되 **조용히 넘기지 않는다**.

    예전엔 debug 로 남겨 로그에서 한 줄도 안 보였다. 회상은 실패해도 답변이 나가기
    때문에(기억만 빠진 채로) 이 로그가 유일한 단서다 — WARNING 으로 올린다.

    Args:
        name: 소스 이름. 어느 층이 죽었는지 로그로 바로 알기 위한 것.
        failures: 주면 실패한 소스 이름을 담는다. 안정 블록 캐시 여부 판단에 쓴다.
    """
    try:
        return await coro
    except Exception as e:
        logger.warning("memory_v2 회상 소스 실패(비움) source=%s: %s: %s", name, type(e).__name__, e)
        if failures is not None:
            failures.append(name)
        return default


def _current_place() -> Optional[str]:
    """이번 턴의 장소. 플래그가 꺼져 있으면 항상 None — 순위·자격이 이 기능 이전과 같아진다."""
    if not bool(getattr(settings, "memory_v2_place_scope", False)):
        return None
    try:
        from agent_arch.agents.context import get_user_context

        return place_mod.current_key(get_user_context())
    except Exception:  # noqa: BLE001 — 장소를 몰라도 회상은 돌아야 한다
        logger.debug("memory_v2 장소 확인 실패(무시)", exc_info=True)
        return None


async def _recall(
    user_sub: str, query: str, company_scope: Optional[str], cached: Optional[dict] = None,
    here: Optional[str] = None,
) -> RecallResult:
    turn_recall = bool(getattr(settings, "memory_v2_turn_recall", True))
    if cached is not None:
        alias_pool, related = await asyncio.gather(
            _safe(repo.list_active_atoms(user_sub, atom_type="alias", limit=MAX_ALIAS_POOL), [], name="alias"),
            _safe(hybrid_search(user_sub, query, limit=MAX_TURN_ATOMS, exclude_types=("alias",), place=here), [], name="related") if turn_recall else _noop([]),
        )
        strict_ids = set(cached.get("strict_ids") or [])
        system_block = cached["block"]
        strict, persona_text, scenes = [], None, []
    else:
        # 안정 블록을 이루는 세 소스(지시·페르소나·장면)만 캐시 판단에 넣는다. 별칭·관련 아톰은
        # 캐시 대상이 아니라 매 턴 다시 조회하므로, 그쪽 실패로 캐시까지 막을 이유가 없다.
        stable_failures: list = []
        strict, alias_pool, persona, scenes, related = await asyncio.gather(
            _safe(repo.list_strict_instructions(user_sub, MAX_STRICT), [], name="strict", failures=stable_failures),
            _safe(repo.list_active_atoms(user_sub, atom_type="alias", limit=MAX_ALIAS_POOL), [], name="alias"),
            _safe(repo.get_active_persona(user_sub, company_scope), None, name="persona", failures=stable_failures),
            _safe(repo.list_scenes(user_sub, limit=NAV_SCENES), [], name="scenes", failures=stable_failures),
            _safe(hybrid_search(user_sub, query, limit=MAX_TURN_ATOMS, exclude_types=("alias",), place=here), [], name="related") if turn_recall else _noop([]),
        )
        # 장소에 매인 전역 지시는 그 장소에서만 상시 블록에 넣는다. 블록을 만들기 전에 거른다.
        place_bound = here if any(place_mod.atom_places(a) for a in strict) else None
        strict = [a for a in strict if place_mod.eligible(a, here)]
        strict_ids = {a["id"] for a in strict}
        persona_text = (persona or {}).get("content") if persona else None
        system_block = format_system_block(persona_text, scenes, strict)
        if stable_failures:
            # 실패해서 비어버린 블록을 캐시하면 TTL(30분) 동안 그 사용자는 페르소나도 전역 지시도
            # 없는 사람이 된다. 캐시 히트 경로가 strict 를 캐시 값으로만 채우기 때문이다.
            # 다음 턴에 한 번 더 조회하는 편이 30분을 빈 기억으로 사는 것보다 낫다.
            logger.warning(
                "memory_v2 안정 블록 캐시 보류 — 실패한 소스 %s (다음 턴에 재조회)",
                ", ".join(stable_failures),
            )
        else:
            await _set_stable_cache(user_sub, company_scope, system_block, sorted(strict_ids),
                                    place_bound=place_bound)
    aliases = match_aliases(query, alias_pool)
    atoms = [a for a in related if a["id"] not in strict_ids][:MAX_TURN_ATOMS]
    logger.info(
        "memory_v2 회상 user=%s place=%s 지시=%d 별칭=%d 관련=%d(같은장소 %d)",
        user_sub[:8], here or "-", len(strict), len(aliases), len(atoms),
        sum(1 for a in atoms if here and here in place_mod.atom_places(a)),
    )
    return RecallResult(
        system_block=system_block,
        turn_block=format_turn_block(strict, aliases, atoms),
        strict=strict, aliases=aliases, atoms=atoms, persona=persona_text, scenes=scenes,
    )


async def _noop(value):
    return value


async def recall_for_turn(
    user_sub: Optional[str],
    query: str,
    *,
    company_scope: Optional[str] = None,
    timeout: Optional[float] = None,
) -> RecallResult:
    """한 턴의 회상. 실패·초과 시에도 도구 안내(system_block)는 돌려준다 — 에이전트가 저장 능력을 잊지 않게.

    Args:
        timeout: 예산(초). 주지 않으면 안정 블록 캐시 유무로 정한다 — 웜은
            ``RECALL_TIMEOUT_SECONDS``, 콜드는 ``RECALL_TIMEOUT_COLD_SECONDS``.
    """
    if not user_sub:
        return RecallResult()
    # 캐시를 여기서 한 번만 조회해 (1) 예산을 정하고 (2) 그 값을 _recall 에 넘긴다.
    # 조회 자체가 실패하면 콜드로 친다 — 빠듯한 예산으로 5소스를 태우는 쪽이 더 나쁘다.
    here = _current_place()
    cached = await _get_stable_cache(user_sub, company_scope, here)
    if timeout is None:
        timeout = RECALL_TIMEOUT_SECONDS if cached is not None else RECALL_TIMEOUT_COLD_SECONDS
    # shield: 시간 초과 시 회상 태스크를 **취소하지 않고** 뒤에서 끝나게 둔다. wait_for 의 취소는
    # asyncpg 질의를 도중에 끊어 풀 연결을 망가뜨리고("unexpected connection_lost"), Qdrant 첫
    # 연결·임베딩 워밍업도 매번 처음부터 다시 하게 만든다(라이브 실측). 결과만 버리면 다음 턴은 빠르다.
    task = asyncio.ensure_future(_recall(user_sub, query or "", company_scope, cached, here))
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout)
    except asyncio.TimeoutError:
        logger.warning("memory_v2 회상 타임아웃(%.1fs, user=%s) — 빈 블록(태스크는 백그라운드 완주)", timeout, user_sub[:8])
        task.add_done_callback(_discard_result)
        return RecallResult(system_block=TOOLS_GUIDE, timed_out=True)
    except Exception as e:
        logger.exception("memory_v2 회상 실패(user=%s) — 빈 블록", user_sub[:8])
        return RecallResult(system_block=TOOLS_GUIDE, error=f"{type(e).__name__}: {e}"[:200])


def _discard_result(task: "asyncio.Future") -> None:
    """타임아웃 뒤 완주한 회상 태스크의 결과·예외를 소비한다('Future exception was never retrieved' 방지)."""
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc:
        logger.debug("memory_v2 늦은 회상 실패(무시): %s: %s", type(exc).__name__, exc)


async def warm_up() -> None:
    """기동 시 Qdrant 연결 + 임베딩 1회로 첫 회상의 콜드 스타트를 제거한다(실패는 무시)."""
    try:
        from agent_arch.services.memory_v2.indexer import get_v2_indexer
        idx = await get_v2_indexer()
        await idx.search("__warmup__", "워밍업", top_k=1)
        logger.info("memory_v2 회상 워밍업 완료")
    except Exception as e:
        logger.warning("memory_v2 회상 워밍업 실패(첫 회상이 느릴 수 있음): %s: %s", type(e).__name__, e)
