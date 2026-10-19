"""배치 중복·모순 판정 — 후보 회수(벡터 + 키워드) → 전역 후보 풀 → LLM 배치 판정 1회.

임계값이 없다. v1 은 코사인 0.70/0.92·trgm 0.75 네 개가 결과를 갈랐고 어휘가 다른 모순
(존댓말/반말)은 후보에서 탈락했다. v2 는 후보를 넓게 모아 판정을 LLM 에 맡기고, 배치 안의
신규 atom 끼리 겹치는 경우도 한 번에 잡는다(판정기가 신규 전체를 본다).

실패 정책: 후보 회수 실패·LLM 호출 실패·파싱 실패 전부 → **모두 store**(유실보다 중복).
"""
from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Optional

from agent_arch.db.repository.memory_v2 import get_active_atoms_by_ids, search_atoms_keyword
from agent_arch.llm.base import Message
from agent_arch.services.memory_v2.llm import extract_json, invoke_text
from agent_arch.services.memory_v2.models import (
    DEDUP_ACTIONS,
    AtomDraft,
    DedupDecision,
    JudgeResult,
    normalize_atom_type,
)
from agent_arch.services.memory_v2.prompts import DEDUP_JUDGE_PROMPT, format_dedup_input

logger = logging.getLogger(__name__)

CANDIDATES_PER_DRAFT = 5
MAX_POOL = 40


def _all_store(drafts: list[AtomDraft], **kw) -> JudgeResult:
    return JudgeResult(decisions=[DedupDecision(index=i, action="store") for i in range(len(drafts))], **kw)


async def _vector_hits(user_sub: str, text: str) -> list[str]:
    try:
        from agent_arch.services.memory_v2.indexer import get_v2_indexer
        idx = await get_v2_indexer()
        hits = await idx.search(user_sub, text, top_k=CANDIDATES_PER_DRAFT)
        return [h["id"] for h in hits]
    except Exception as e:
        logger.debug("memory_v2 dedup 벡터 후보 회수 실패(키워드만 사용): %s", e)
        return []


async def recall_candidates(user_sub: str, drafts: list[AtomDraft]) -> tuple[list[dict], list[list[str]]]:
    """draft 별 후보 atom id 와, 전역 중복 제거된 후보 풀(하이드레이트된 dict)을 돌려준다."""
    per_draft: list[list[str]] = []
    seen: dict[str, None] = {}
    for d in drafts:
        ids: list[str] = []
        for hid in await _vector_hits(user_sub, d.content):
            if hid not in ids:
                ids.append(hid)
        try:
            for row in await search_atoms_keyword(user_sub, d.content, limit=CANDIDATES_PER_DRAFT):
                if row["id"] not in ids:
                    ids.append(row["id"])
        except Exception as e:
            logger.debug("memory_v2 dedup 키워드 후보 회수 실패: %s", e)
        per_draft.append(ids)
        for i in ids:
            seen.setdefault(i, None)
    pool_ids = list(seen.keys())[:MAX_POOL]
    pool = await get_active_atoms_by_ids(user_sub, pool_ids) if pool_ids else []
    # 하이드레이트 순서를 후보 등장 순으로 맞춘다(태그 안정성).
    order = {pid: k for k, pid in enumerate(pool_ids)}
    pool.sort(key=lambda a: order.get(a["id"], 10**6))
    active_ids = {a["id"] for a in pool}
    per_draft = [[i for i in ids if i in active_ids] for ids in per_draft]
    return pool, per_draft


_NEAR_DUP_RATIO = 0.85


def _near_duplicate_in_batch(drafts: list[AtomDraft], idx: int) -> bool:
    """draft[idx] 와 거의 같은 문장(SequenceMatcher ≥0.85)이 배치의 **앞선** 신규 atom 에 있는가.

    앞선 것만 보는 이유: 서로가 서로를 skip 하면 둘 다 사라진다. 뒤의 것만 skip 되게 한다.
    """
    me = drafts[idx].content
    for j in range(idx):
        other = drafts[j].content
        if abs(len(other) - len(me)) > max(20, len(me) // 3):
            continue
        if SequenceMatcher(None, me, other).ratio() >= _NEAR_DUP_RATIO:
            return True
    return False


def parse_decisions(
    arr: list,
    drafts: list[AtomDraft],
    n_tags: dict[str, int],
    e_tags: dict[str, str],
) -> list[DedupDecision]:
    """LLM 배열 → draft 마다 정확히 하나의 결정. 누락·불량은 store, 대상 없는 update/merge 는 store,
    신규끼리만 가리키는 update/merge 는 skip(앞선 신규가 살아남는다)."""
    by_index: dict[int, DedupDecision] = {}
    for item in arr:
        if not isinstance(item, dict):
            continue
        idx = n_tags.get(str(item.get("new") or ""))
        if idx is None or idx in by_index:
            continue
        action = str(item.get("action") or "store").strip().lower()
        if action not in DEDUP_ACTIONS:
            action = "store"
        raw_targets = item.get("targets") or []
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        targets = [e_tags[t] for t in raw_targets if isinstance(t, str) and t in e_tags]
        points_to_new = any(isinstance(t, str) and t in n_tags and n_tags[t] != idx for t in raw_targets)
        if action in ("update", "merge") and not targets:
            action = "skip" if points_to_new else "store"
        if action == "skip" and not targets and not points_to_new:
            # 근거 없는 skip 은 유실 위험 — 기존 기억도, 같은 배치의 다른 신규도 가리키지 않으면 저장한다.
            # 단 배치 안에 거의 같은 문장의 다른 신규 atom 이 있으면(LLM 이 targets 를 빠뜨린 배치 내 중복,
            # 백필 실측) skip 을 그대로 인정한다.
            action = "skip" if _near_duplicate_in_batch(drafts, idx) else "store"
        atype = normalize_atom_type(item.get("atom_type")) if item.get("atom_type") else None
        content = item.get("content") if action in ("update", "merge") else None
        content = str(content).strip()[:1000] if content else None
        by_index[idx] = DedupDecision(
            index=idx, action=action, target_ids=targets, content=content or None,
            atom_type=atype, priority=item.get("priority"),
        )
    return [by_index.get(i) or DedupDecision(index=i, action="store") for i in range(len(drafts))]


async def judge_batch(user_sub: str, drafts: list[AtomDraft]) -> JudgeResult:
    """배치 판정. 후보가 하나도 없고 신규가 1개면 LLM 을 부르지 않는다(비교 대상 없음)."""
    if not drafts:
        return JudgeResult(decisions=[])
    try:
        pool, candidates = await recall_candidates(user_sub, drafts)
    except Exception as e:
        logger.warning("memory_v2 dedup 후보 회수 실패 — 전부 store: %s", e)
        return _all_store(drafts, error=f"recall: {e}"[:300])
    if not pool and len(drafts) == 1:
        return JudgeResult(decisions=[DedupDecision(index=0, action="store")], pool_size=0)

    text, n_tags, e_tags = format_dedup_input(drafts, pool, candidates)
    messages = [Message(role="system", content=DEDUP_JUDGE_PROMPT), Message(role="user", content=text)]
    try:
        raw, model, latency = await invoke_text(messages)
    except Exception as e:
        logger.warning("memory_v2 dedup LLM 호출 실패 — 전부 store: %s", e)
        return _all_store(drafts, pool_size=len(pool), error=f"{type(e).__name__}: {e}"[:300], llm_called=True)
    arr = extract_json(raw, "array")
    if arr is None:
        obj = extract_json(raw, "object")
        if isinstance(obj, dict):
            inner = obj.get("decisions") or obj.get("results")
            arr = inner if isinstance(inner, list) else [obj]
    if not isinstance(arr, list):
        logger.warning("memory_v2 dedup 응답 파싱 실패 — 전부 store")
        return _all_store(drafts, pool_size=len(pool), raw=raw, model=model, latency_ms=latency,
                          error="parse_failed", llm_called=True)
    decisions = parse_decisions(arr, drafts, n_tags, e_tags)
    return JudgeResult(decisions=decisions, pool_size=len(pool), raw=raw, model=model,
                       latency_ms=latency, llm_called=True)
