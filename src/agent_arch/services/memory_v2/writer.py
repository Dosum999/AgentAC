"""판정 결과 적용 — store / update / merge / skip 을 PG + Qdrant 에 쓴다.

update·merge 는 행을 덮어쓰지 않는다: 새 atom(version = max(대상)+1) 을 넣고 대상은
superseded 로 바꾸며 Qdrant 포인트를 즉시 지운다. source_message_ids·timestamps 는 합집합으로
이어져 한 사실의 타임라인이 병합 뒤에도 남는다.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from agent_arch.db.repository.memory_v2 import (
    get_active_atoms_by_ids,
    insert_atom,
    supersede_atoms,
    touch_atoms,
)
from agent_arch.services.memory_v2 import place as place_mod
from agent_arch.services.memory_v2.models import (
    PRIORITY_STRICT,
    AtomDraft,
    DedupDecision,
    WriteStats,
    clamp_priority,
)
from agent_arch.services.monitoring.error_events import event_extra

logger = logging.getLogger(__name__)

_MAX_LIST = 50
_MERGE_BONUS = 5
# 자동 merge 로 오를 수 있는 상한. 같은 루틴이 하루 수십 번 관측되면(백필 실측: 16회 merge → 80)
# 사건 하나가 명시 지시(80)와 같은 무게가 된다. 명시 저장·판정기 지정 priority 는 이 상한을 안 받는다.
_MERGE_CAP = 70


_STRICT_FALLBACK_PRIORITY = 80


def _without_strict(draft: AtomDraft, decision: DedupDecision, targets: list[dict], action: str) -> int:
    """-1 을 뺀 값들로 최종 priority 를 낸다. 전부 -1 이었으면 일반 지시 상한을 준다."""
    pool = [p for p in [draft.priority] + [int(t.get("priority", 50)) for t in targets]
            if p != PRIORITY_STRICT]
    if decision.priority is not None and clamp_priority(decision.priority) != PRIORITY_STRICT:
        return clamp_priority(decision.priority)
    if not pool:
        return _STRICT_FALLBACK_PRIORITY
    if action == "merge":
        best = max(pool)
        return best if best >= _MERGE_CAP else min(_MERGE_CAP, best + _MERGE_BONUS)
    return max(pool) if action == "update" else pool[0]


def _strict_allowed(draft: AtomDraft, targets: list[dict]) -> bool:
    """항상 주입되는 등급(priority -1)에 오를 수 있는가 — 남의 말이 섞인 기억은 못 오른다.

    저장·검색·회상은 그대로 된다. 막는 것은 상시 규칙이 되는 길 하나뿐이다. 점수 감점이 아니라
    선행 조건이라, 아무리 자주 회상돼도 mixed 는 -1 이 되지 않는다.
    """
    if draft.origin == place_mod.ORIGIN_MIXED:
        return False
    return all(place_mod.atom_origin(t) != place_mod.ORIGIN_MIXED for t in targets)


def resolve_priority(draft: AtomDraft, decision: DedupDecision, targets: list[dict], action: str) -> int:
    """최종 priority. 판정이 명시하면 그것, merge 면 최대값+5(정보가 확실해짐, 상한 70), 강제(-1)는 전파.

    단 남의 말이 섞인 기억(origin=mixed)은 -1 로 올라가지 못하고 일반 지시 상한으로 내려앉는다.
    """
    if not _strict_allowed(draft, targets):
        return _without_strict(draft, decision, targets, action)
    pool = [draft.priority] + [int(t.get("priority", 50)) for t in targets]
    if action in ("update", "merge") and PRIORITY_STRICT in pool:
        return PRIORITY_STRICT
    # 전역 여부는 쓰는 쪽(명시 저장의 scope·추출기)이 정한다 — 중복 판정기가 새 전역 지시의 등급을 낮추지 않는다.
    if draft.priority == PRIORITY_STRICT:
        return PRIORITY_STRICT
    if decision.priority is not None:
        return clamp_priority(decision.priority)
    if action == "merge":
        best = max(pool)
        if best == PRIORITY_STRICT:
            return best
        return best if best >= _MERGE_CAP else min(_MERGE_CAP, best + _MERGE_BONUS)
    return draft.priority


def resolve_atom_type(draft: AtomDraft, decision: DedupDecision, targets: list[dict], action: str) -> str:
    """최종 종류. 판정이 정하면 그것, 아니면 draft 의 것 — 단 지시(instruction)를 대체·병합한 결과가
    persona 로 내려앉지는 않는다(priority -1 전파와 같은 부류의 결정적 규칙).

    "앞으로 점심 상신은 기존 결재선으로 해" 같은 규칙을 모델이 record_type=preference 로 저장하면
    판정기가 update 로 먼저 뽑힌 지시를 persona 로 갈아끼웠다(실측 2026-09-28). persona 는 회상에서
    "사실로 단정 말 것" 참고 정보로 실려, 사용자가 못 박은 규칙이 참고 사항으로 강등된다.
    """
    atom_type = decision.atom_type or draft.atom_type
    if (action in ("update", "merge") and atom_type == "persona"
            and any(t.get("atom_type") == "instruction" for t in targets)):
        return "instruction"
    # 전역(-1)으로 못 박은 새 지시는 지시로 남는다 — 지운 규칙을 다시 저장하자 판정기가 "규칙을 삭제했다" 기록에 합쳐
    # persona 로 바꿨고, 전역 등급만 얹힌 persona 는 전역 목록(지시만 읽음)에서도 빠졌다(로컬 2026-09-29).
    if draft.atom_type == "instruction" and draft.priority == PRIORITY_STRICT and atom_type != "instruction":
        return "instruction"
    return atom_type


def _union(*lists: list, limit: int = _MAX_LIST) -> list:
    out: list = []
    for lst in lists:
        for x in lst or []:
            if x not in out:
                out.append(x)
    return out[-limit:]  # 최신 쪽을 남긴다


def _id_grounding_targets(draft: AtomDraft, targets: list[dict]) -> list[dict]:
    """skip 판정을 update 로 올려야 하는 대상 — 신규 alias 가 canonical_id 를 갖고, 대상 alias 는
    같은 종류(또는 종류 미상)인데 id 가 없는 경우. 그 외(비-alias, id 이미 있음, 종류 불일치)는 빈 목록."""
    if draft.atom_type != "alias":
        return []
    p = draft.payload or {}
    new_id = p.get("canonical_id")
    if not new_id:
        return []
    kind = str(p.get("canonical_kind") or "").strip().lower()
    out: list[dict] = []
    for t in targets:
        if t.get("atom_type") != "alias":
            continue
        tp = t.get("payload") if isinstance(t.get("payload"), dict) else {}
        if tp.get("canonical_id"):
            continue
        tkind = str(tp.get("canonical_kind") or "").strip().lower()
        if kind and tkind and kind != tkind:
            continue
        out.append(t)
    return out


def _strict_promotion_targets(draft: AtomDraft, targets: list[dict]) -> list[dict]:
    """skip 판정을 전역 승격(update)으로 올려야 하는 대상 — 신규가 전역 지시(-1)인데 같은 내용의 기존 지시는 아직 일반 등급.

    판정기는 문장만 보고 "같은 규칙"이라 skip 한다. 그러면 사용자가 "그건 항상 지켜"라고 다시 말해도 등급이 그대로라
    상관없는 질문에는 계속 회상되지 않는다. 남의 말이 섞인 기억은 오를 수 없다(``_strict_allowed``).
    """
    if draft.atom_type != "instruction" or draft.priority != PRIORITY_STRICT or not _strict_allowed(draft, targets):
        return []
    return [t for t in targets if t.get("atom_type") == "instruction" and int(t.get("priority", 50)) != PRIORITY_STRICT]


def stamp_place(payload: dict, draft: AtomDraft, targets: list[dict]) -> dict:
    """장소는 합집합, 출처 등급은 더 엄한 쪽(mixed)이 이긴다.

    합쳐진 기억은 여러 채널에서 관측됐을 수 있고, 그중 한 곳에만 있어도 같은 맥락으로 본다.
    등급은 반대다 — 한 번이라도 남의 말이 섞였으면 그 기억은 계속 mixed 다(천장이 유지된다).
    """
    places = list(draft.places or [])
    origin = draft.origin
    for t in targets:
        for key in place_mod.atom_places(t):
            if key not in places:
                places.append(key)
        if place_mod.atom_origin(t) == place_mod.ORIGIN_MIXED:
            origin = place_mod.ORIGIN_MIXED
    out = dict(payload or {})
    if places:
        out[place_mod.PAYLOAD_PLACES] = places[:place_mod.MAX_PLACES_PER_ATOM]
    else:
        out.pop(place_mod.PAYLOAD_PLACES, None)
    # 할 말이 있을 때만 적는다 — 장소도 없고 등급도 user 면 payload 는 이 기능 이전과 같다.
    # 패널에서 오던 기억은 잘 돌고 있었으므로 그 경로의 저장 모양을 바꾸지 않는다.
    if origin == place_mod.ORIGIN_MIXED:
        out[place_mod.PAYLOAD_ORIGIN] = origin
    else:
        out.pop(place_mod.PAYLOAD_ORIGIN, None)
    return out


def merge_payload(draft: AtomDraft, targets: list[dict], atom_type: str) -> dict:
    """alias 는 aliases 합집합 + canonical 채움(있는 값 우선). 그 외는 draft payload 우선 병합."""
    if atom_type != "alias":
        merged: dict[str, Any] = {}
        for t in targets:
            if isinstance(t.get("payload"), dict):
                merged.update(t["payload"])
        merged.update(draft.payload or {})
        return merged
    aliases: list[str] = []
    canon: dict[str, Any] = {}
    for src in [*(t.get("payload") or {} for t in targets), draft.payload or {}]:
        for a in src.get("aliases") or []:
            if a and a not in aliases:
                aliases.append(str(a))
        for k in ("canonical_name", "canonical_kind", "canonical_id"):
            if src.get(k) not in (None, ""):
                canon[k] = src[k]
    return {"aliases": aliases[:20], **canon}


async def _delete_points(ids: list[str]) -> None:
    try:
        from agent_arch.services.memory_v2.indexer import get_v2_indexer
        await (await get_v2_indexer()).delete_points(ids)
    except Exception as e:
        logger.debug("memory_v2 Qdrant 포인트 삭제 실패(다음 sync 가 보정하지 않음 — 주의): %s", e)


async def _index_atom(atom: dict) -> None:
    try:
        from agent_arch.services.memory_v2.indexer import get_v2_indexer
        await (await get_v2_indexer()).index_atoms([atom])
    except Exception as e:
        logger.debug("memory_v2 인덱싱 지연(embedding_indexed=false 로 남김): %s", e)


async def apply_decisions(
    user_sub: str,
    drafts: list[AtomDraft],
    decisions: list[DedupDecision],
    *,
    generation_id: Optional[str],
    scene_name: Optional[str],
    timestamps_for: Optional[dict[int, list[datetime]]] = None,
    company_scope: Optional[str] = None,
) -> WriteStats:
    """draft 와 결정을 짝지어 적용한다. 개별 실패는 로그만 남기고 다음으로 간다."""
    stats = WriteStats()
    by_index = {d.index: d for d in decisions}
    # 출처 메시지 시각(timestamps_for = agent.messages.created_at, naive UTC)과 같은 목록에 섞이므로
    # 기본값도 naive UTC 다. datetime.now() 는 컨테이너 로컬(NKS 는 KST)이라 9시간 어긋난다.
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    for i, draft in enumerate(drafts):
        decision = by_index.get(i) or DedupDecision(index=i, action="store")
        action = decision.action
        try:
            targets = await get_active_atoms_by_ids(user_sub, decision.target_ids) if decision.target_ids else []
            if action == "skip":
                promoted = _id_grounding_targets(draft, targets) or _strict_promotion_targets(draft, targets)
                if not promoted:
                    if decision.target_ids:
                        await touch_atoms(user_sub, decision.target_ids)
                    stats.skipped += 1
                    continue
                # 같은 별칭이라도 정식 id 를 새로 확보한 것은 새 정보다 — 판정기는 문장만 보고
                # skip 을 내리지만, id 가 있어야 회상이 "바로 조회" 레시피를 낼 수 있다. 문장은 기존
                # 것을 유지하고 payload 만 채운 update 로 올린다(결정적 규칙, 판정기 재호출 없음).
                targets = promoted
                action = "update"
                if draft.atom_type == "alias":
                    decision = DedupDecision(
                        index=decision.index, action="update", target_ids=[t["id"] for t in promoted],
                        content=next((t.get("content") for t in reversed(promoted) if t.get("content")), None),
                        atom_type="alias",
                        priority=max([draft.priority, *[int(t.get("priority", 50)) for t in promoted]]),
                    )
                    stats.id_grounded += 1
                else:   # 같은 규칙을 '항상'으로 다시 못 박음 — 문장은 기존 것, 등급만 전역으로
                    decision = DedupDecision(
                        index=decision.index, action="update", target_ids=[t["id"] for t in promoted],
                        content=next((t.get("content") for t in reversed(promoted) if t.get("content")), None),
                        atom_type="instruction", priority=PRIORITY_STRICT,
                    )

            if action in ("update", "merge") and not targets:
                action = "store"

            atom_type = resolve_atom_type(draft, decision, targets, action)
            content = (decision.content or draft.content).strip()
            priority = resolve_priority(draft, decision, targets, action)
            version = max([int(t.get("version", 1)) for t in targets], default=0) + 1
            src_ids = _union(*[t.get("source_message_ids") or [] for t in targets], draft.source_message_ids)
            ts = _union(*[t.get("timestamps") or [] for t in targets], (timestamps_for or {}).get(i) or [now])
            payload = merge_payload(draft, targets, atom_type) if targets else dict(draft.payload or {})
            payload = stamp_place(payload, draft, targets)
            scene = draft.scene_name or scene_name or next((t.get("scene_name") for t in targets if t.get("scene_name")), None)

            new_id = await insert_atom(
                user_sub, atom_type=atom_type, content=content, priority=priority, scene_name=scene,
                payload=payload, source_message_ids=src_ids, timestamps=ts, version=version,
                generation_id=generation_id, company_scope=company_scope or draft.company_scope,
            )
            if targets:
                superseded = await supersede_atoms(user_sub, [t["id"] for t in targets], new_id)
                if superseded:
                    await _delete_points(superseded)
            await _index_atom({
                "id": new_id, "user_sub": user_sub, "atom_type": atom_type, "content": content,
                "scene_name": scene, "company_scope": company_scope or draft.company_scope, "priority": priority,
            })
            stats.output_atom_ids.append(new_id)
            if action == "update":
                stats.updated += 1
            elif action == "merge":
                stats.merged += 1
            else:
                stats.stored += 1
        except Exception:
            # 로그에는 내용 앞부분을 남기지만 오류 이벤트(외부로 푸시된다)에는 싣지 않는다 — 기억
            # 본문은 사용자 발화에서 뽑은 것이다.
            logger.exception(
                "memory_v2 atom 쓰기 실패(user=%s, action=%s): %s", user_sub[:8], action, draft.content[:60],
                extra=event_extra("background", message=f"memory_v2 atom 쓰기 실패(action={action})"),
            )
    return stats
