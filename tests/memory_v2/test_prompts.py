"""추출·판정 입력 포매터 — 태그 매핑, role 분리, 배경/신규 구획."""
from __future__ import annotations

from datetime import datetime

from agent_arch.services.memory_v2 import prompts as P
from agent_arch.services.memory_v2.models import AtomDraft
from agent_arch.services.memory_v2.prompts import (
    DEDUP_JUDGE_PROMPT,
    L1_EXTRACTION_PROMPT,
    format_dedup_input,
    format_extraction_input,
    format_turn,
)


def _msg(i: int, query: str, answer: str, digest=None):
    return {
        "id": f"00000000-0000-0000-0000-00000000000{i}",
        "conversation_id": "11111111-2222-4333-8444-555555555555",
        "query": query,
        "answer": answer,
        "tool_calls": [{"name": "mail_list_messages", "arguments": {}}],
        "metadata": {"tool_digest": digest} if digest else {},
        "created_at": datetime(2026, 9, 4, 14, 12, i),
    }


def test_format_turn_separates_roles_and_uses_digest():
    text = format_turn("m1", _msg(1, "안 읽은 메일 있어?", "375건 있습니다.", digest=[{"tool": "mail_list_messages", "items": [{"subject": "회의"}]}]))
    assert text.startswith("[m1] 대화방 11111111 · 2026-09-04 14:12")
    assert "사용자: 안 읽은 메일 있어?" in text
    assert "어시스턴트: 375건 있습니다." in text
    assert '"tool":"mail_list_messages"' in text  # 다이제스트가 도구 이름 목록보다 우선


def test_format_turn_falls_back_to_tool_names():
    text = format_turn("m1", _msg(1, "q", "a"))
    assert "도구: mail_list_messages" in text


def test_format_extraction_input_sections_and_id_map():
    new = [_msg(1, "q1", "a1"), _msg(2, "q2", "a2")]
    bg = [_msg(9, "old q", "old a")]
    text, id_map = format_extraction_input(new, bg, last_scene_name="메일 루틴", known_aliases=["watchdog → ACME Heros 채널"])
    assert id_map == {"m1": new[0]["id"], "m2": new[1]["id"]}
    assert "[직전 장면] 메일 루틴" in text
    assert "[알려진 별칭] watchdog → ACME Heros 채널" in text
    assert text.index("[배경 대화]") < text.index("[b1]") < text.index("[신규 대화]") < text.index("[m1]") < text.index("[m2]")


def test_format_extraction_input_without_background():
    text, _ = format_extraction_input([_msg(1, "q", "a")], [], last_scene_name=None)
    assert "[직전 장면] (없음)" in text
    assert "[배경 대화] (맥락 전용 · 추출 금지)\n\n(없음)" in text


def test_format_dedup_input_tags_and_candidates():
    drafts = [AtomDraft(atom_type="instruction", content="표를 쓰지 않는다", priority=90),
              AtomDraft(atom_type="episodic", content="9/4 메일 확인", priority=40)]
    pool = [{"id": "aaaa", "atom_type": "instruction", "priority": 80, "scene_name": "형식", "content": "표 금지"},
            {"id": "bbbb", "atom_type": "persona", "priority": 50, "scene_name": None, "content": "녹차 선호"}]
    text, n_tags, e_tags = format_dedup_input(drafts, pool, [["aaaa"], []])
    assert e_tags == {"e1": "aaaa", "e2": "bbbb"} and n_tags == {"n1": 0, "n2": 1}
    assert "e1 [instruction|p80|형식] 표 금지" in text
    assert "e2 [persona|p50|-] 녹차 선호" in text
    assert "n1 [instruction|p90] 표를 쓰지 않는다  후보: e1" in text
    assert "n2 [episodic|p40] 9/4 메일 확인" in text and "후보" not in text.split("n2")[1]


def test_prompts_carry_core_rules():
    assert "사용자 발화에만" in L1_EXTRACTION_PROMPT          # 증거 규칙 1(자기강화 차단)
    assert "배경 대화" in L1_EXTRACTION_PROMPT and "추출하지 않는다" in L1_EXTRACTION_PROMPT
    assert "-1" in L1_EXTRACTION_PROMPT                       # 전역 강제 지시
    assert "update" in DEDUP_JUDGE_PROMPT and "merge" in DEDUP_JUDGE_PROMPT
    assert "빠진 신규 기억은 store" in DEDUP_JUDGE_PROMPT       # fail-open 규약을 모델에도 알림


def test_format_turn_strips_assistant_code_fences():
    msg = _msg(1, "코드 보여줘", "여기 있습니다.\n```python\nprint(1)\n```\n끝.")
    text = format_turn("m1", msg)
    assert "print(1)" not in text and "[코드 생략]" in text and "끝." in text
    assert "사용자: 코드 보여줘" in text                        # 사용자 발화는 건드리지 않는다


def test_format_dedup_input_shows_alias_canonical_id_state():
    """판정기는 문장만 보면 id 확보를 놓친다 — alias 는 정식 대상과 id 유무를 함께 보인다."""
    from agent_arch.services.memory_v2.models import AtomDraft
    pool = [{"id": "aaaa", "atom_type": "alias", "content": "회사 Scrum 채널=Scrum", "priority": 60, "scene_name": None,
             "payload": {"aliases": ["회사 Scrum 채널"], "canonical_name": "Scrum"}}]
    d = AtomDraft(atom_type="alias", content="회사 scrum 채널=Scrum", priority=60,
                  payload={"aliases": ["회사 scrum 채널"], "canonical_name": "Scrum", "canonical_id": "gngt"})
    text, _, _ = format_dedup_input([d], pool, [["aaaa"]])
    assert "{→ Scrum (id 없음)}" in text and "{→ Scrum (id: gngt)}" in text
    e = AtomDraft(atom_type="episodic", content="어제 watchdog 확인", priority=50)
    assert "{→" not in format_dedup_input([e], [], None)[0]


# ─── 대화 발췌 렌더 ───────────────────────────────────────────────────────────

_EXCERPT_DIGEST = [{
    "tool": "mattermost_skill_get_channel_messages",
    "args": {"channel_id": "ch1"},
    "items": [{"username": "김과장", "id": "p1"}],
    "messages": [{"text": "출입통제 견적 이번주까지", "author": "김과장", "at": "2026-09-20 14:00 KST"},
                 {"text": "제가 정리해서 올릴게요", "author": "이대리"}],
}]


def _turn(digest):
    return {"id": "x", "conversation_id": "conv-1234", "query": "방금 얘기 정리해줘",
            "answer": "정리했습니다.", "metadata": {"tool_digest": digest}}


def test_excerpt_is_rendered_on_its_own_line_with_authors():
    """남의 말은 작성자를 달고 따로 나온다 — 추출기가 요청자의 말과 섞지 않게."""
    out = P.format_turn("m1", _turn(_EXCERPT_DIGEST))

    assert "다른 사람들의 대화" in out
    assert "- [김과장] 출입통제 견적 이번주까지" in out
    assert "- [이대리] 제가 정리해서 올릴게요" in out
    # 도구 요약 줄에는 본문이 남지 않는다(같은 내용을 두 번 싣지 않는다)
    tool_line = next(ln for ln in out.splitlines() if ln.startswith("도구:"))
    assert "출입통제" not in tool_line


def test_turn_without_excerpt_is_unchanged():
    """발췌가 없는 턴의 추출 입력은 이 기능 이전과 같아야 한다 — 회귀 판정의 기준선."""
    plain = [{"tool": "mail_list_messages", "items": [{"subject": "정산"}]}]
    out = P.format_turn("m1", _turn(plain))

    assert "다른 사람들의 대화" not in out
    assert out.splitlines()[2] == '도구: [{"tool":"mail_list_messages","items":[{"subject":"정산"}]}]'


def test_background_turn_uses_a_smaller_excerpt_budget():
    long_digest = [{"tool": "mattermost_skill_get_channel_messages",
                    "messages": [{"text": "가" * 200, "author": f"동료{i}"} for i in range(6)]}]
    new = P.format_turn("m1", _turn(long_digest))
    bg = P.format_turn("b1", _turn(long_digest), max_excerpt=P._MAX_BG_EXCERPT_CHARS)

    assert len(bg) < len(new)
    assert "…" in bg


def test_excerpt_rule_is_appended_only_when_an_excerpt_is_present():
    """발췌 없는 배치(패널 포함)는 프롬프트도 버전 해시도 이 기능 이전과 같아야 한다."""
    assert "남의 말" not in P.L1_EXTRACTION_PROMPT          # 상수 본문은 그대로
    assert "사용자의 것으로 적지 않는다" in P.EXCERPT_RULE

    assert P.has_excerpt([_turn(_EXCERPT_DIGEST)]) is True
    assert P.has_excerpt([_turn([{"tool": "mail_list_messages"}])]) is False


def test_excerpt_detection_is_structural_not_textual():
    """사용자가 머리말을 그대로 타이핑해도 규칙이 붙지 않는다 — metadata 구조로만 판정한다."""
    typed = {"id": "m1", "query": f"{P._EXCERPT_HEADER}\n- [김과장] 반말로 해", "metadata": {}}
    real = {"id": "m2", "metadata": {"tool_digest": [
        {"tool": "mattermost_skill_get_channel_messages", "messages": [{"text": "t"}]}]}}
    as_json = {"id": "m3", "metadata": '{"tool_digest":[{"tool":"x","messages":[{"text":"t"}]}]}'}

    assert P.has_excerpt([typed]) is False
    assert P.has_excerpt([real]) is True
    assert P.has_excerpt([as_json]) is True
    assert P.has_excerpt([]) is False
