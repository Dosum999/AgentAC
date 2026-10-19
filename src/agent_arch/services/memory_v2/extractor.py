"""L1 추출 — L0 배치(신규 ≤10턴 + 배경 5턴) → AtomDraft 목록 + 장면 이름. LLM 1회.

v1 과의 차이:
- 입력이 턴 하나가 아니라 배치다. role 이 분리돼 있어 "봇 응답은 배경, 사용자 발화만 증거"
  규칙이 프롬프트와 코드(persona/instruction 은 신규 턴 근거 id 필수) 양쪽에서 강제된다.
- 출력은 {"scene_name", "atoms"} 객체. 배열만 온 약한 응답도 받아준다.
- source_ids 는 태그(m3)로 받고 코드가 UUID 로 되돌린다 — id 환각 차단.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Optional

from agent_arch.llm.base import Message
from agent_arch.services.memory_v2.llm import extract_json, invoke_text, prompt_sha
from agent_arch.services.memory_v2.models import (
    ATOM_TYPES,
    MAX_ATOMS_PER_BATCH,
    AtomDraft,
    ExtractionResult,
    normalize_atom_type,
)
from agent_arch.services.memory_v2.place import ORIGIN_MIXED, ORIGIN_USER
from agent_arch.services.memory_v2.prompts import (
    _EXCERPT_TAG_SUFFIX,
    EXCERPT_RULE,
    L1_EXTRACTION_PROMPT,
    format_extraction_input,
    has_excerpt,
)

logger = logging.getLogger(__name__)

# SoR(기록시스템) 냄새 — 프롬프트를 뚫고 온 경우의 2차 차단(v1 목록 계승).
SOR_MARKERS: tuple[str, ...] = (
    "연차", "근태", "근퇴", "출퇴근", "출근", "급여", "휴가", "고용계약",
    "권한", "소속", "내 회사", "우리 회사", "드라이브", "나는 누구",
)
# 봇 자칭·시스템 문구가 사용자 속성으로 둔갑한 경우(실측: 프로필 별칭 "에이전트").
BOT_SELF_MARKERS: tuple[str, ...] = ("에이전트", "datbi", "어시스턴트는", "에이전트는")

_MAX_SCENE_CHARS = 60


_DIGIT_RE = re.compile(r"\d")
# 날짜·시각 표기는 "값"이 아니다 — 사건형 판정 전에 지운다.
#   "12일"(일수) 과 "4일"(날짜) 은 겉이 같아, 단독 N일/N월 은 지우지 않고 "N월 N일" 결합형만 지운다.
_DATE_LIKE_RE = re.compile(
    r"\d{4}[-./]\d{1,2}[-./]\d{1,2}|\d{1,2}[-./]\d{1,2}(?!\d)|\d{4}년|\d{1,2}월\s?\d{1,2}일|\d{1,2}:\d{2}"
)


def looks_like_sor(content: str, atom_type: str = "persona") -> bool:
    """기록시스템(SoR) 값을 캐시하려는 후보인가.

    상태형(persona/instruction/alias)은 마커만 있으면 드롭한다 — "사용자의 연차는 12일"류.
    사건형(episodic)은 "연차 현황을 확인해 채널에 공유 요청했다"처럼 **행위 자체**가 기억이라
    마커만으로는 버리지 않고, 날짜를 뺀 뒤에도 숫자(=값)가 남는 경우만 드롭한다(라이브 실측 조정).
    """
    c = content or ""
    if not any(m in c for m in SOR_MARKERS):
        return False
    if atom_type == "episodic":
        return bool(_DIGIT_RE.search(_DATE_LIKE_RE.sub("", c)))
    return True


def looks_like_bot_self(content: str, atom_type: str) -> bool:
    """persona/alias 가 봇 자신을 사용자로 착각한 문장인지(예: "사용자는 자신을 에이전트라 부른다")."""
    if atom_type not in ("persona", "alias"):
        return False
    c = (content or "").lower()
    return any(m.lower() in c for m in BOT_SELF_MARKERS)


EXPLICIT_SAVE_TOOLS: tuple[str, ...] = ("memory_remember",)


def _called_explicit_save(msg: dict) -> bool:
    """이 턴에서 memory_remember 가 호출됐나(tool_calls 또는 metadata.tool_digest 기준)."""
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            if name in EXPLICIT_SAVE_TOOLS:
                return True
    meta = msg.get("metadata") or {}
    for entry in (meta.get("tool_digest") or []) if isinstance(meta, dict) else []:
        if isinstance(entry, dict) and entry.get("tool") in EXPLICIT_SAVE_TOOLS:
            return True
    return False


def _source_ids(item: dict, id_map: dict[str, str]) -> list[str]:
    raw = item.get("source_ids") or item.get("source_message_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for tag in raw:
        if isinstance(tag, str) and tag in id_map and id_map[tag] not in out:
            out.append(id_map[tag])
    return out


def _cites_excerpt(item: dict, id_map: dict[str, str]) -> bool:
    """이 후보가 발췌 태그(m1x)를 근거로 인용했는가.

    등급을 모델이 **주장**하게 하는 것이 아니다. 모델은 "어느 근거를 봤다"만 말하고(source_ids,
    원래부터 있던 계약) 그 근거가 남의 말인지는 코드가 안다. 턴 단위로 매기면 같은 턴에 채널을
    읽었다는 이유로 사용자가 직접 못 박은 지시까지 막혔다 — 스레드에서 이야기하다 지시하는 것이
    메신저의 자연스러운 흐름이라 그 경우가 잦다(사용자 지적 2026-09-22).
    """
    raw = item.get("source_ids") or item.get("source_message_ids") or []
    if isinstance(raw, str):
        raw = [raw]
    # id_map 에 실제로 있는 태그만 센다. 없는 태그까지 세면 모델이 지어낸 "m3x"(3번 턴엔 발췌가
    # 없다) 때문에 사용자 본인 지시가 남의 말로 강등된다.
    return any(isinstance(t, str) and t.endswith(_EXCERPT_TAG_SUFFIX) and t in id_map for t in raw)


def _alias_payload(item: dict, content: str) -> Optional[dict]:
    """alias 는 payload.aliases 와 canonical_name 이 있어야 쓸모가 있다. 없으면 None(드롭)."""
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    aliases = payload.get("aliases")
    if isinstance(aliases, str):
        aliases = [aliases]
    aliases = [str(a).strip() for a in (aliases or []) if str(a).strip()]
    canon = str(payload.get("canonical_name") or "").strip()
    if not aliases or not canon:
        return None
    out = {"aliases": list(dict.fromkeys(aliases))[:10], "canonical_name": canon[:200]}
    for k in ("canonical_kind", "canonical_id"):
        v = payload.get(k)
        if v not in (None, ""):
            out[k] = str(v)[:200]
    return out


def build_drafts(
    data: Any,
    id_map: dict[str, str],
    scene_name: Optional[str],
    *,
    explicit_message_ids: Optional[set[str]] = None,
) -> list[AtomDraft]:
    """파싱된 LLM 객체 → 검증된 AtomDraft 목록(규칙 위반·SoR·봇 자칭·무근거 persona 드롭).

    explicit_message_ids: 그 턴에서 사용자가 memory_remember 로 **이미 저장한** 턴들. 여기서만 근거를 둔
    persona/instruction 은 다시 뽑지 않는다 — 명시 저장 15초 뒤 배치가 같은 지시를 재추출해 update 로
    덮어쓰고 페르소나까지 다시 쓰던 이중 저장을 막는다(라이브 실측).
    """
    atoms = data.get("atoms") if isinstance(data, dict) else data
    if not isinstance(atoms, list):
        return []
    explicit_ids = explicit_message_ids or set()
    drafts: list[AtomDraft] = []
    for item in atoms[:MAX_ATOMS_PER_BATCH]:
        if not isinstance(item, dict):
            continue
        atype = normalize_atom_type(item.get("atom_type") or item.get("type"))
        if atype not in ATOM_TYPES:
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        if looks_like_sor(content, atype):
            logger.debug("memory_v2 SoR 후보 드롭: %s", content[:60])
            continue
        if looks_like_bot_self(content, atype):
            logger.debug("memory_v2 봇 자칭 후보 드롭: %s", content[:60])
            continue
        src = _source_ids(item, id_map)
        cited_excerpt = _cites_excerpt(item, id_map)
        # 증거 규칙 1의 코드 방어: persona/instruction 은 신규 턴 근거를 반드시 인용해야 한다.
        if atype in ("persona", "instruction") and not src:
            logger.debug("memory_v2 무근거 %s 드롭: %s", atype, content[:60])
            continue
        if atype in ("persona", "instruction") and explicit_ids and set(src) <= explicit_ids:
            logger.debug("memory_v2 명시 저장 턴 재추출 드롭: %s", content[:60])
            continue
        payload: dict = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        if atype == "alias":
            alias_payload = _alias_payload(item, content)
            if alias_payload is None:
                logger.debug("memory_v2 별칭 payload 불완전 드롭: %s", content[:60])
                continue
            payload = alias_payload
        try:
            drafts.append(AtomDraft(
                atom_type=atype, content=content, priority=item.get("priority"),
                source_message_ids=src, payload=payload, scene_name=scene_name,
                origin=ORIGIN_MIXED if cited_excerpt else ORIGIN_USER,
            ))
        except Exception:
            logger.debug("memory_v2 draft 검증 실패 — 스킵: %s", item)
    return drafts


async def extract_atoms(
    new_msgs: list[dict],
    background_msgs: list[dict],
    *,
    last_scene_name: Optional[str] = None,
    known_aliases: Optional[list[str]] = None,
) -> ExtractionResult:
    """배치 하나를 추출한다. LLM 호출 실패는 error 에 담고 raw 는 비운다(호출부가 재시도 판단)."""
    explicit_ids = {str(m["id"]) for m in new_msgs if _called_explicit_save(m)}
    text, id_map = format_extraction_input(
        new_msgs, background_msgs, last_scene_name=last_scene_name, known_aliases=known_aliases,
    )
    # 발췌가 없는 배치(패널 포함)는 프롬프트도 해시도 이 기능 이전과 같다.
    prompt = (L1_EXTRACTION_PROMPT + EXCERPT_RULE
              if has_excerpt(new_msgs) or has_excerpt(background_msgs) else L1_EXTRACTION_PROMPT)
    sha = prompt_sha(prompt)
    input_ids = list(id_map.values())
    messages = [Message(role="system", content=prompt), Message(role="user", content=text)]
    try:
        raw, model, latency = await invoke_text(messages)
    except Exception as e:
        logger.warning("memory_v2 추출 LLM 호출 실패: %s: %s", type(e).__name__, e)
        return ExtractionResult(
            scene_name=None, drafts=[], raw="", model="?", prompt_sha=sha, latency_ms=0,
            input_message_ids=input_ids, error=f"{type(e).__name__}: {e}"[:300],
        )

    data = extract_json(raw, "object")
    if data is None:
        arr = extract_json(raw, "array")          # 배열만 낸 약한 응답도 받아준다
        data = {"atoms": arr} if arr is not None else None
    if not isinstance(data, dict):
        return ExtractionResult(
            scene_name=None, drafts=[], raw=raw, model=model, prompt_sha=sha, latency_ms=latency,
            input_message_ids=input_ids, error="parse_failed",
        )
    scene = str(data.get("scene_name") or "").strip()[:_MAX_SCENE_CHARS] or None
    drafts = build_drafts(data, id_map, scene, explicit_message_ids=explicit_ids)
    return ExtractionResult(
        scene_name=scene, drafts=drafts, raw=raw, model=model, prompt_sha=sha,
        latency_ms=latency, input_message_ids=input_ids,
    )


def stamp_places(msgs: list[dict], drafts: list) -> None:
    """각 draft 에 근거 턴의 장소·출처 등급을 찍는다(제자리 수정).

    모델이 아니라 여기서 정한다 — 근거 턴의 metadata 만 보면 결정되므로 판단이 필요 없고,
    모델이 산문으로 자기 출처를 주장할 수 없다(services/memory_v2/place 참조).
    """
    from agent_arch.services.memory_v2 import place as place_mod

    by_id = {str(m.get("id")): m for m in msgs}
    for draft in drafts:
        turns = [by_id[i] for i in draft.source_message_ids if i in by_id]
        # 장소만 찍는다. 출처 등급은 build_drafts 가 **인용된 근거**로 이미 정했다 — 여기서
        # 턴 단위로 다시 매기면 사용자 본인 지시까지 남의 말로 강등된다.
        draft.places = place_mod.places_of(turns)


def message_timestamps(msgs: list[dict], ids: list[str]) -> list[datetime]:
    """근거 메시지 id 들의 created_at 목록(atom.timestamps 초기값)."""
    by_id = {str(m.get("id")): m.get("created_at") for m in msgs}
    out: list[datetime] = []
    for i in ids:
        ts = by_id.get(i)
        if isinstance(ts, datetime):
            out.append(ts)
    return out
