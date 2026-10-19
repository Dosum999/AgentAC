"""L3 페르소나 — 사용자 한 장 요약(≤2000자). 장면 문서(L2)를 1차, atom 을 2차 근거로 종합한다.

냉시동 억제: atom 이 5개 미만이면 만들지 않는다. 정보가 없으면 "(없음)". 봇 이름·시스템 문구·
파일 경로에서 사용자 속성을 추론하지 않는다(v1 프로필에 봇 이름 "에이전트"가 별칭으로 들어갔던 실측).
fresh: 삭제·모순 갱신 뒤에는 기존 페르소나를 참조하지 않고 다시 쓴다(sticky 잔존 방지).
"""
from __future__ import annotations

import logging
from typing import Optional

from agent_arch.db.repository import memory_v2 as repo
from agent_arch.llm.base import Message
from agent_arch.services.memory_v2.llm import invoke_text, prompt_sha, strip_fence
from agent_arch.utils.sanitize import normalize_math_markup

logger = logging.getLogger(__name__)

PERSONA_MIN_ATOMS = 5
PERSONA_REBUILD_EVERY = 50           # atom 이 이만큼 쌓이면 재빌드
MAX_PERSONA_CHARS = 2000
MAX_INPUT_ATOMS = 150
EMPTY_MARKER = "(없음)"

PERSONA_PROMPT = """\
당신은 사용자 페르소나 작성자다. 장면 문서와 기억 atom 을 근거로, 에이전트가 매 대화 시작 때 읽을 "이 사용자 한 장"을 쓴다.

네 층을 순서대로 살핀다(근거 있는 것만):
1. 기본 앵커: 역할·담당 업무·소속 맥락·자주 다루는 시스템(도구가 아니라 업무 시스템).
2. 관심·업무 그래프: 반복해서 다루는 프로젝트·거래처·동료·채널. 진행 중인 것과 끝난 것을 구분.
3. 상호작용 규약: 답변 형식·길이·말투 선호, 자주 쓰는 표현→실제 대상(별칭), 반복 루틴.
4. 인지 핵심: 의사결정 방식·우선순위처럼 여러 근거가 일관될 때만 조심스럽게.

제약:
- **근거 없는 추측 금지.** 냉시동(근거 적음)에서는 비워 두는 것이 낫다. 한 번 언급된 것을 성향으로 굳히지 않는다.
- 어시스턴트 이름(에이전트), 시스템 안내문, 도구·파일 경로에서 사용자 속성을 만들지 않는다.
- 실시간으로 바뀌는 값(연차·재고·오늘 일정·메일 건수)은 넣지 않는다. 특정 날짜의 일정·회의·마감·진행 중 작업
  상태(예: "9/30 회의")도 넣지 않는다 — 페르소나는 매 대화 모든 답에 실리므로 그런 일은 장면·기록 몫이다.
- **요청·시도·초안은 결과가 아니다.** "회의 일정을 등록하도록 요청했다"를 "회의 예정"·"등록함"으로 바꿔 쓰지 않는다
  (승인되지 않은 초안일 수 있다). 근거가 요청이면 요청으로 적거나 뺀다.
- 삭제된 내용을 되살리지 않는다: [기존 페르소나]에 있어도 [장면]·[atom]에 근거가 없으면 뺀다.
- 한국어 Markdown, **2000자 이내**, 위 층 중 근거 있는 것만 소제목으로. 표는 쓰지 않는다.
- 수식·LaTeX 표기 금지(화살표는 → 문자, `$\\rightarrow$` 같은 표기 금지).
- 쓸 만한 안정 정보가 없으면 정확히 "(없음)" 한 줄만 출력한다. Markdown 외 설명 문구 금지.
"""


def format_persona_input(existing: Optional[str], scenes: list[dict], atoms: list[dict]) -> str:
    parts = [f"[기존 페르소나]\n{existing.strip() if existing else '(없음)'}"]
    if scenes:
        lines = []
        for s in scenes:
            content = (s.get("content") or "").strip().replace("\n", " ")
            lines.append(f"- {s.get('title')} (heat {float(s.get('heat') or 0):.1f}): {content[:700]}")
        parts.append("[장면 문서]\n" + "\n".join(lines))
    else:
        parts.append("[장면 문서]\n(없음)")
    atom_lines = [f"- [{a.get('atom_type')}|p{a.get('priority')}] {a.get('content')}" for a in atoms]
    parts.append(f"[atom {len(atoms)}개]\n" + ("\n".join(atom_lines) if atom_lines else "(없음)"))
    return "\n\n".join(parts)


def clean_persona(raw: str) -> Optional[str]:
    """LLM 출력 → 저장할 본문. 빈 응답·(없음) 은 None. LaTeX 화살표 등은 평문으로 정규화한다.

    페르소나는 system 프롬프트에 매 턴 들어가므로 `$\\rightarrow$` 같은 수식 표기가 남으면 답변에도
    번진다(v1 프로필 실측). 답변 경로의 정규화기(app.utils.sanitize)를 저장 시점에 미리 적용한다.
    """
    text = strip_fence(raw or "").strip()
    if not text or text.startswith(EMPTY_MARKER) or text.lower().startswith("(none)"):
        return None
    text = normalize_math_markup(text)
    return text[:MAX_PERSONA_CHARS]


async def rebuild_persona(
    user_sub: str,
    *,
    fresh: bool = False,
    company_scope: Optional[str] = None,
) -> Optional[str]:
    """페르소나를 (재)작성해 저장한다. 저장한 본문 반환, 만들지 않았으면 None.

    atom 이 최소치 미만이거나 LLM 이 (없음) 을 내면 저장하지 않는다. fresh 이고 기존 본이 있으면
    그 경우 기존 본을 지운다 — 지운 기억이 페르소나에 유령처럼 남는 것을 막는다.
    """
    atoms = await repo.list_active_atoms(user_sub, limit=MAX_INPUT_ATOMS)
    existing = await repo.get_active_persona(user_sub, company_scope)
    if len(atoms) < PERSONA_MIN_ATOMS:
        if fresh and existing:
            await repo.delete_persona(user_sub, company_scope)
            logger.info("memory_v2 L3 페르소나 삭제(근거 atom %d < %d, fresh)", len(atoms), PERSONA_MIN_ATOMS)
        return None

    scenes = await repo.list_scenes(user_sub, limit=15, with_content=True)
    text = format_persona_input(None if fresh else (existing or {}).get("content"), scenes, atoms)
    sha = prompt_sha(PERSONA_PROMPT)
    try:
        raw, model, latency = await invoke_text([
            Message(role="system", content=PERSONA_PROMPT), Message(role="user", content=text),
        ])
    except Exception as e:
        logger.warning("memory_v2 L3 LLM 호출 실패(user=%s): %s", user_sub[:8], e)
        await _log(user_sub, "failed", None, sha, len(atoms), 0, error=f"{type(e).__name__}: {e}"[:300])
        return None

    content = clean_persona(raw)
    if content is None:
        if fresh and existing:
            await repo.delete_persona(user_sub, company_scope)
        await repo.mark_l3_done(user_sub)
        await _log(user_sub, "empty", model, sha, len(atoms), latency, raw=raw)
        return None

    await repo.upsert_persona(user_sub, content, source_atom_count=len(atoms), company_scope=company_scope)
    await repo.mark_l3_done(user_sub)
    await _log(user_sub, "succeeded", model, sha, len(atoms), latency, raw=raw)
    logger.info("memory_v2 L3 user=%s chars=%d atoms=%d fresh=%s", user_sub[:8], len(content), len(atoms), fresh)
    return content


async def _log(user_sub, status, model, sha, n_atoms, latency, *, error=None, raw=None) -> None:
    try:
        await repo.insert_generation_log(
            user_sub=user_sub, layer="l3", status=status, model=model, prompt_sha=sha,
            input_message_ids=[], output_atom_ids=[], decision_counts={"atoms": n_atoms},
            latency_ms=latency, error=error, raw_output=raw,
        )
    except Exception:
        logger.debug("memory_v2 L3 로그 실패(무시)", exc_info=True)
