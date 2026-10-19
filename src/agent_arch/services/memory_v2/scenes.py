"""L2 장면 문서 — atom 을 프로젝트·루틴·관계 단위의 서사 문서로 묶는다(유저당 활성 ≤15).

LLM 에 파일 도구를 주지 않고 "현재 장면 목록 + 신규 atom → JSON ops(create/update/delete)" 로
단순화한다. 문서는 목록 나열이 아니라 이어지는 서사여야 하고, BATCH_/SUMMARY_ 같은 메타 문서는
금지한다(LLM 이 장면 대신 요약 파일을 만들어내는 흔한 실패 모드).

heat: 실행마다 0.9 배로 식히고 이번에 atom 이 붙은 장면에 그 수만큼 더한다 → 최근 관측이 위로.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from agent_arch.db.repository import memory_v2 as repo
from agent_arch.llm.base import Message
from agent_arch.services.memory_v2.llm import extract_json, invoke_text, prompt_sha
from agent_arch.utils.sanitize import normalize_math_markup

logger = logging.getLogger(__name__)

MAX_SCENES = 15
MAX_SCENE_CONTENT_CHARS = 1500
MAX_SUMMARY_CHARS = 120
MAX_TITLE_CHARS = 60
MAX_NAME_CHARS = 60
HEAT_DECAY = 0.9
MAX_INPUT_ATOMS = 120
_FORBIDDEN_NAME_PARTS = ("batch", "summary", "report", "archive", "consolidat", "integration", "misc", "기타", "요약", "정리")

SCENE_OPS_PROMPT = """\
당신은 장기 기억의 장면(scene) 편집자다. 사용자의 기억 atom 들을 프로젝트·루틴·관계처럼 **함께 떠올라야 할 맥락 단위**로 묶어
장면 문서를 만들고 유지한다. 장면 문서는 다음 대화에서 에이전트가 "이 사용자의 그 일이 어디까지 왔나"를 빠르게 복원하는 데 쓰인다.

입력:
- [현재 장면]: 지금 있는 장면들(name, title, heat, summary, content 일부).
- [신규 atom]: 이번에 새로 들어온 기억(a1, a2 … 태그, 타입, 추출 시 붙은 장면 이름 힌트, 내용).

할 일: 신규 atom 을 기존 장면에 흡수(update)하거나 새 장면을 만들고(create), 사실상 같은 맥락인 장면은 하나로 합친다(합치는 쪽 update + 흡수된 쪽 delete).

규칙:
1. 장면 수는 **최대 15개**. 넘치면 가장 관련 있는 것들을 합친다. 억지로 만들지 않는다 — atom 하나짜리 장면은 정말 독립적인 맥락일 때만.
2. content 는 **서사**다: 무엇을 위해, 누구와, 무엇을 했고, 지금 어디까지 왔는지를 이어지는 문장으로 쓴다. 목록만 나열하지 않는다. 1,500자 이내.
3. summary 는 한 줄(120자 이내). 내비게이션에 그대로 노출된다.
4. name 은 짧은 식별자(한글·영문·숫자·하이픈, 60자 이내, 예: 메일-공유-루틴, 견적-비교-조회). 기존 장면을 갱신할 때는 **기존 name 을 그대로** 쓴다.
5. "배치", "요약", "정리", "기타", "report", "archive" 류의 메타 장면은 만들지 않는다.
6. 신규 atom 중 어느 장면에도 안 맞고 장면을 만들 만큼 맥락도 없으면 그냥 두어도 된다(모든 atom 을 넣어야 하는 것은 아니다).
7. 사용자가 말하지 않은 사실을 지어내지 않는다. atom 에 있는 것만 쓴다. 실시간으로 바뀌는 값(연차·재고·오늘 일정)은 넣지 않는다.
8. 표·수식·LaTeX 표기는 쓰지 않는다(화살표는 → 문자).

출력: JSON 객체 하나만. JSON 외 텍스트 금지. 바꿀 것이 없으면 {"ops": []}.
{"ops": [
  {"op": "create", "name": "메일-공유-루틴", "title": "메일 확인과 채널 공유 루틴", "summary": "…", "content": "…", "atoms": ["a1", "a3"]},
  {"op": "update", "name": "기존-이름", "title": "…", "summary": "…", "content": "…", "atoms": ["a2"]},
  {"op": "delete", "name": "흡수된-장면"}
]}
"""

_NAME_CLEAN_RE = re.compile(r"[^0-9a-zA-Z가-힣\-]+")


@dataclass
class SceneOp:
    op: str
    name: str
    title: str = ""
    summary: str = ""
    content: str = ""
    atom_ids: list[str] = field(default_factory=list)


@dataclass
class ScenesResult:
    created: int = 0
    updated: int = 0
    deleted: int = 0
    input_atoms: int = 0
    error: Optional[str] = None
    raw: str = ""
    model: str = ""
    latency_ms: int = 0

    @property
    def changed(self) -> int:
        return self.created + self.updated + self.deleted


def normalize_name(raw: Any) -> Optional[str]:
    """장면 이름을 식별자로 정규화한다. 메타 이름·빈 값은 None."""
    s = str(raw or "").strip().lower()
    s = _NAME_CLEAN_RE.sub("-", s).strip("-")
    s = re.sub(r"-{2,}", "-", s)[:MAX_NAME_CHARS].strip("-")
    if not s:
        return None
    if any(p in s for p in _FORBIDDEN_NAME_PARTS):
        return None
    return s


def format_scene_input(scenes: list[dict], atoms: list[dict]) -> tuple[str, dict[str, str]]:
    lines = ["[현재 장면]"]
    if scenes:
        for s in scenes:
            content = (s.get("content") or "").strip().replace("\n", " ")
            lines.append(
                f"- name={s.get('name')} | title={s.get('title')} | heat={float(s.get('heat') or 0):.1f}\n"
                f"  summary: {s.get('summary') or ''}\n  content: {content[:600]}"
            )
    else:
        lines.append("(없음)")
    lines.append("")
    lines.append("[신규 atom]")
    tag_map: dict[str, str] = {}
    for i, a in enumerate(atoms):
        tag = f"a{i + 1}"
        tag_map[tag] = a["id"]
        lines.append(f"{tag} [{a.get('atom_type')}|{a.get('scene_name') or '-'}|p{a.get('priority')}] {a.get('content')}")
    return "\n".join(lines), tag_map


def parse_ops(data: Any, tag_map: dict[str, str], existing_names: set[str]) -> list[SceneOp]:
    """LLM ops → 검증된 SceneOp. 이름 정규화, 필수 필드, 존재하지 않는 장면의 update 는 create 로, delete 는 무시."""
    raw_ops = data.get("ops") if isinstance(data, dict) else data
    if not isinstance(raw_ops, list):
        return []
    out: list[SceneOp] = []
    seen: set[str] = set()
    for item in raw_ops:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "").strip().lower()
        name = normalize_name(item.get("name"))
        if op not in ("create", "update", "delete") or not name or name in seen:
            continue
        if op == "delete":
            if name in existing_names:
                out.append(SceneOp(op="delete", name=name))
                seen.add(name)
            continue
        title = normalize_math_markup(str(item.get("title") or "").strip())[:MAX_TITLE_CHARS]
        content = normalize_math_markup(str(item.get("content") or "").strip())[:MAX_SCENE_CONTENT_CHARS]
        if not title or not content:
            continue
        summary = normalize_math_markup(str(item.get("summary") or "").strip().replace("\n", " "))[:MAX_SUMMARY_CHARS]
        tags = item.get("atoms") or []
        if isinstance(tags, str):
            tags = [tags]
        atom_ids = [tag_map[t] for t in tags if isinstance(t, str) and t in tag_map]
        if op == "update" and name not in existing_names:
            op = "create"
        if op == "create" and name in existing_names:
            op = "update"
        out.append(SceneOp(op=op, name=name, title=title, summary=summary, content=content, atom_ids=atom_ids))
        seen.add(name)
    return out


def enforce_cap(ops: list[SceneOp], existing_count: int, cap: int = MAX_SCENES) -> list[SceneOp]:
    """적용 후 활성 장면 수가 cap 을 넘지 않게 create 를 뒤에서부터 잘라낸다(delete·update 는 유지)."""
    deletes = sum(1 for o in ops if o.op == "delete")
    creates = [o for o in ops if o.op == "create"]
    room = cap - (existing_count - deletes)
    if len(creates) <= room:
        return ops
    keep = set(id(o) for o in creates[:max(0, room)])
    dropped = len(creates) - max(0, room)
    logger.info("memory_v2 L2 장면 상한(%d) — create %d건 잘라냄", cap, dropped)
    return [o for o in ops if o.op != "create" or id(o) in keep]


async def rebuild_scenes(user_sub: str, *, since: Optional[datetime], company_scope: Optional[str] = None) -> ScenesResult:
    """since 이후 atom 을 장면에 반영한다. 신규 atom 이 없으면 no-op."""
    atoms = await repo.list_atoms_since(user_sub, since, limit=MAX_INPUT_ATOMS)
    if not atoms:
        return ScenesResult()
    scenes = await repo.list_scenes(user_sub, limit=50, with_content=True)
    text, tag_map = format_scene_input(scenes, atoms)
    sha = prompt_sha(SCENE_OPS_PROMPT)
    result = ScenesResult(input_atoms=len(atoms))
    try:
        raw, model, latency = await invoke_text([
            Message(role="system", content=SCENE_OPS_PROMPT), Message(role="user", content=text),
        ])
        result.raw, result.model, result.latency_ms = raw, model, latency
    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"[:300]
        logger.warning("memory_v2 L2 LLM 호출 실패(user=%s): %s", user_sub[:8], result.error)
        await _log(user_sub, result, sha, atoms, status="failed")
        return result

    data = extract_json(raw, "object")
    if data is None:
        arr = extract_json(raw, "array")
        data = {"ops": arr} if arr is not None else None
    if data is None:
        result.error = "parse_failed"
        await _log(user_sub, result, sha, atoms, status="failed")
        return result

    existing_names = {s["name"] for s in scenes}
    ops = enforce_cap(parse_ops(data, tag_map, existing_names), len(scenes))
    if ops:
        await repo.decay_scene_heat(user_sub, HEAT_DECAY)
    for op in ops:
        try:
            if op.op == "delete":
                if await repo.delete_scene(user_sub, op.name):
                    result.deleted += 1
                continue
            await repo.upsert_scene(
                user_sub, name=op.name, title=op.title, summary=op.summary, content=op.content,
                evidence_atom_ids=op.atom_ids, heat_delta=float(max(1, len(op.atom_ids))), company_scope=company_scope,
            )
            if op.op == "create":
                result.created += 1
            else:
                result.updated += 1
        except Exception:
            logger.exception("memory_v2 L2 op 적용 실패(user=%s, %s %s)", user_sub[:8], op.op, op.name)
    await _log(user_sub, result, sha, atoms, status="succeeded" if ops else "empty")
    logger.info(
        "memory_v2 L2 user=%s atoms=%d created=%d updated=%d deleted=%d",
        user_sub[:8], len(atoms), result.created, result.updated, result.deleted,
    )
    return result


SCENE_REWRITE_PROMPT = """\
당신은 장기 기억의 장면(scene) 편집자다. 사용자가 기억 일부를 지웠다. 아래 장면을 **[남은 atom] 만으로** 다시 써라.

규칙:
1. [남은 atom] 에 없는 사실은 전부 뺀다 — 지금 장면 본문에 있어도 atom 이 받쳐 주지 않으면 지운 기억이다.
2. content 는 이어지는 서사(1,500자 이내), summary 는 한 줄(120자 이내), title 은 짧게(60자 이내).
3. 사용자가 말하지 않은 사실을 지어내지 않는다. 표·수식·LaTeX 표기는 쓰지 않는다(화살표는 → 문자).

출력: JSON 객체 하나만. {"title": "…", "summary": "…", "content": "…"}
"""


@dataclass
class RepairResult:
    rewritten: int = 0
    deleted: int = 0
    failed: int = 0

    @property
    def changed(self) -> int:
        return self.rewritten + self.deleted


async def repair_scenes(user_sub: str) -> RepairResult:
    """지워진 기억을 근거로 쓴 장면을 **남은 근거만으로 다시 쓴다**(남은 근거가 없으면 장면 삭제).

    장면 갱신(rebuild_scenes)은 새 atom 만 반영하므로, 기억을 지워도 그 내용이 장면 본문에 남고 — 페르소나
    재작성이 장면 본문을 읽어 — 페르소나로 되돌아왔다(실측 2026-09-28: "경비청구 기억 다 지워줘" 뒤에도 장면·
    페르소나에 점심 경비 건들이 그대로). 근거 중 하나라도 삭제됐으면(대체는 체인을 따라가 판정) 그 장면이 대상이다.
    다시 쓰기에 실패하면 장면을 지운다 — 지운 기억이 남는 것보다 장면을 잃는 편이 낫다(atom 은 그대로라 회상은 된다).
    """
    result = RepairResult()
    scenes = await repo.list_scenes(user_sub, limit=50, with_content=True)
    all_ids = [i for sc in scenes for i in (sc.get("evidence_atom_ids") or [])]
    if not all_ids:
        return result
    live_map = await repo.resolve_evidence(user_sub, all_ids)
    for sc in scenes:
        ev = sc.get("evidence_atom_ids") or []
        if not ev or all(live_map.get(i) for i in ev):
            continue
        live_ids = list(dict.fromkeys(live_map[i] for i in ev if live_map.get(i)))
        if not live_ids:
            if await repo.delete_scene_by_id(user_sub, sc["id"]):
                result.deleted += 1
            continue
        atoms = await repo.get_active_atoms_by_ids(user_sub, live_ids)
        rewritten = await _rewrite_scene(user_sub, sc, atoms) if atoms else None
        if rewritten is None:
            result.failed += 1
            if await repo.delete_scene_by_id(user_sub, sc["id"]):
                result.deleted += 1
            continue
        title, summary, content = rewritten
        if await repo.rewrite_scene(user_sub, sc["id"], title=title, summary=summary, content=content,
                                    evidence_atom_ids=[a["id"] for a in atoms]):
            result.rewritten += 1
    if result.changed or result.failed:
        logger.info("memory_v2 L2 정리 user=%s rewritten=%d deleted=%d failed=%d",
                    user_sub[:8], result.rewritten, result.deleted, result.failed)
    return result


async def _rewrite_scene(user_sub: str, scene: dict, atoms: list[dict]) -> Optional[tuple[str, str, str]]:
    lines = [
        f"[장면] name={scene.get('name')} | title={scene.get('title')}",
        f"summary: {scene.get('summary') or ''}",
        f"content: {(scene.get('content') or '').strip()}",
        "",
        "[남은 atom]",
        *(f"- [{a.get('atom_type')}|p{a.get('priority')}] {a.get('content')}" for a in atoms),
    ]
    try:
        raw, _model, _latency = await invoke_text([
            Message(role="system", content=SCENE_REWRITE_PROMPT), Message(role="user", content="\n".join(lines)),
        ])
    except Exception as e:
        logger.warning("memory_v2 L2 장면 다시 쓰기 실패(user=%s, %s): %s", user_sub[:8], scene.get("name"), e)
        return None
    data = extract_json(raw, "object")
    if not isinstance(data, dict):
        return None
    title = normalize_math_markup(str(data.get("title") or scene.get("title") or "").strip())[:MAX_TITLE_CHARS]
    content = normalize_math_markup(str(data.get("content") or "").strip())[:MAX_SCENE_CONTENT_CHARS]
    summary = normalize_math_markup(str(data.get("summary") or "").strip().replace("\n", " "))[:MAX_SUMMARY_CHARS]
    if not title or not content:
        return None
    return title, summary, content


async def _log(user_sub: str, r: ScenesResult, sha: str, atoms: list[dict], *, status: str) -> None:
    try:
        await repo.insert_generation_log(
            user_sub=user_sub, layer="l2", status=status, model=r.model or None, prompt_sha=sha,
            input_message_ids=[], output_atom_ids=[a["id"] for a in atoms][:50],
            decision_counts={"created": r.created, "updated": r.updated, "deleted": r.deleted, "atoms": len(atoms)},
            latency_ms=r.latency_ms, error=r.error, raw_output=r.raw or None,
        )
    except Exception:
        logger.debug("memory_v2 L2 로그 실패(무시)", exc_info=True)
