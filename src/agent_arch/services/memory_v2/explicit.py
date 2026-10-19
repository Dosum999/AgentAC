"""명시 저장·잊기 — memory_remember / memory_forget 도구의 v2 백엔드.

배치를 기다리지 않고 즉시 L1 에 쓰되, 자동 추출과 **같은 판정기**를 거친다(중복이면 skip/merge,
모순이면 update). 명시 저장은 priority 80, "항상/무조건/절대" 가 박히면 -1(전역 강제).
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Optional

from agent_arch.agents.context import get_user_context
from agent_arch.db.repository import memory_v2 as repo
from agent_arch.services.memory_v2 import dedup, place as place_mod, writer
from agent_arch.services.memory_v2.extractor import looks_like_sor
from agent_arch.services.memory_v2.models import PRIORITY_STRICT, AtomDraft, normalize_atom_type

logger = logging.getLogger(__name__)

EXPLICIT_PRIORITY = 80
STRICT_MARKERS: tuple[str, ...] = ("항상", "무조건", "절대", "앞으로 계속", "매번")


def infer_atom_type(content: str, requested: Optional[str]) -> str:
    """도구가 준 종류를 v2 4종으로. 없으면 instruction(v1 의 directive 기본과 같은 뜻)."""
    return normalize_atom_type(requested) or "instruction"


async def save_explicit_atom(
    user_sub: str,
    content: str,
    *,
    atom_type: Optional[str] = None,
    company_scope: Optional[str] = None,
) -> Optional[dict]:
    """명시 저장. SoR 값이면 None(저장 거부). 아니면 {id, action, atom_type, content}.

    action 이 skip 이면 "이미 같은 기억이 있다"는 뜻이라 도구는 saved=True 로 알린다.
    """
    text = (content or "").strip()
    if not text:
        return None
    atype = infer_atom_type(text, atom_type)
    if looks_like_sor(text, atype):
        return None
    priority = PRIORITY_STRICT if atype == "instruction" and any(m in text for m in STRICT_MARKERS) else EXPLICIT_PRIORITY
    # 지금 있는 자리를 찍는다 — 분리된 창(전용 앱)·메신저 채널에서 못 박은 규칙이 다른 자리까지
    # 따라다니면 기억이 섞인다. 자동 추출은 근거 턴의 metadata 로 같은 일을 하는데
    # (extractor.stamp_places) 명시 저장만 빠져 있었다. 장소 없는 자리(패널·DM)는 None →
    # 종전처럼 중립으로 저장된다. 기록은 플래그와 무관하고 쓰임만 플래그가 가른다.
    here = place_mod.current_key(get_user_context())
    draft = AtomDraft(
        atom_type=atype, content=text, priority=priority, company_scope=company_scope,
        places=[here] if here else [],
    )

    judge = await dedup.judge_batch(user_sub, [draft])
    gen_id = str(uuid.uuid4())
    stats = await writer.apply_decisions(
        user_sub, [draft], judge.decisions, generation_id=gen_id, scene_name=None, company_scope=company_scope,
    )
    action = judge.decisions[0].action if judge.decisions else "store"
    if action in ("update", "merge") and stats.output_atom_ids:
        # 종류는 쓰기 단계에서 바뀔 수 있다(지시를 대체한 persona 는 지시로 남는다) — 도구 응답·로그가
        # 실제로 저장된 종류를 말하게 한다.
        try:
            written = await repo.get_active_atoms_by_ids(user_sub, stats.output_atom_ids[:1])
            if written:
                atype = written[0].get("atom_type") or atype
        except Exception:
            logger.debug("memory_v2 명시 저장 종류 재조회 실패(무시)", exc_info=True)
    try:
        await repo.insert_generation_log(
            log_id=gen_id, user_sub=user_sub, layer="l1", status="succeeded",
            model=f"explicit/{judge.model or '-'}", prompt_sha=None, input_message_ids=[],
            output_atom_ids=stats.output_atom_ids, decision_counts={**stats.counts(), "pool": judge.pool_size},
            latency_ms=judge.latency_ms, error=judge.error, raw_output=judge.raw or None,
        )
    except Exception:
        logger.debug("memory_v2 명시 저장 로그 실패(무시)", exc_info=True)
    logger.info("memory_v2 explicit user=%s action=%s type=%s priority=%s", user_sub[:8], action, atype, priority)
    if action in ("update", "store", "merge"):
        from agent_arch.services.memory_v2.recall import invalidate_stable_block
        await invalidate_stable_block(user_sub)
        # 명시 저장은 사용자가 지금 기다리며 확인하는 기억이다 — 상위층도 바로(모순이면 fresh) 갱신한다.
        from agent_arch.services.memory_v2.layers import schedule_upper_layers
        schedule_upper_layers(
            user_sub, delay=0, force_scenes=True,
            force_persona=(action == "update"), fresh_persona=(action == "update"), company_scope=company_scope,
        )
    return {
        "id": stats.output_atom_ids[0] if stats.output_atom_ids else None,
        "action": action, "atom_type": atype, "content": text, "priority": priority,
    }


_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")


def is_atom_id(value: str) -> bool:
    """atom id(UUID) 형식인가 — 아니면 DB 에 넣지 않는다(장면 이름을 넣으면 asyncpg DataError 로 도구가 터졌다)."""
    return bool(_UUID_RE.match((value or "").strip()))


async def forget_scene(user_sub: str, name: str) -> Optional[dict]:
    """장면 1개(이름)를 지운다. 없으면 None. 페르소나는 장면 본문을 읽으므로 fresh 로 다시 쓴다.

    실측(2026-09-28 운영): 기억을 지운 뒤 남은 장면을 "그것들 지워줘"로 지우려 하자 모델이 memory_forget 에 장면
    이름을 넣었고, 장면을 지울 길이 없어 도구 오류 → 재시도로 반복 한도까지 헤맸다.
    """
    from agent_arch.services.memory_v2.scenes import normalize_name
    candidates = [c for c in dict.fromkeys([(name or "").strip(), normalize_name(name) or ""]) if c]
    for cand in candidates:
        scene = await repo.get_scene(user_sub, cand)
        if not scene:
            continue
        await repo.delete_scene(user_sub, cand)
        logger.info("memory_v2 forget scene user=%s name=%s", user_sub[:8], cand)
        from agent_arch.services.memory_v2.recall import invalidate_stable_block
        await invalidate_stable_block(user_sub)
        from agent_arch.services.memory_v2.layers import schedule_upper_layers
        schedule_upper_layers(user_sub, delay=0, force_persona=True, fresh_persona=True)
        return {"name": cand, "title": scene.get("title"), "summary": scene.get("summary")}
    return None


async def forget_atom(user_sub: str, atom_id: str) -> Optional[dict]:
    """본인 atom 1건 soft-delete + Qdrant 포인트 제거 + 상위층 fresh 재작성 예약. 없으면 None(멱등)."""
    if not is_atom_id(atom_id):
        return None
    row = await repo.soft_delete_atom(user_sub, atom_id)
    if not row:
        return None
    try:
        from agent_arch.services.memory_v2.indexer import get_v2_indexer
        await (await get_v2_indexer()).delete_points([row["id"]])
    except Exception as e:
        logger.debug("memory_v2 forget Qdrant 삭제 실패(다음 검색은 PG status 로 걸러짐): %s", e)
    logger.info("memory_v2 forget user=%s atom=%s type=%s", user_sub[:8], atom_id[:8], row.get("atom_type"))
    from agent_arch.services.memory_v2.recall import invalidate_stable_block
    await invalidate_stable_block(user_sub)
    # 지운 기억이 페르소나·장면에 남지 않게 상위층을 fresh 로 즉시 재작성한다(v1 sticky 프로필 실측).
    from agent_arch.services.memory_v2.layers import schedule_upper_layers
    schedule_upper_layers(user_sub, delay=0, force_scenes=True, force_persona=True, fresh_persona=True)
    return row
