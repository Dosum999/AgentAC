""""이 스레드" 질문을 그 스레드 조회로 고정한다 — 2026-09-14 회귀.

고객사 A>개발팀 스레드에서 "이 스레드에 여러 PDF 가 있는데 각각 정리해 주세요" 라고
물었더니 에이전트가 **회사 scrum 채널**의 PDF 를 읽어 와 "이 스레드(회사 scrum 채널)에 올라온
PDF 는 2개입니다" 라고 답했다. 권한 문제가 아니었다(ACL 은 fail-closed, 요청자가 실제로
볼 수 있는 채널) — **맥락을 잘못 고른 것**이다.

`[현재 위치]` 블록과 `mattermost_read_thread` 도구만으로는 모델이 그 도구를 고른다는 보장이
없어(지식검색과 의미가 겹치고, supervisor 서브쿼리에서 지시어가 빠지기도 한다) guide-first
와 같은 방식으로 첫 호출을 고정한다.
"""
import pytest

from agent_arch.agents.react.thread_first import (
    THREAD_TOOL_NAME,
    is_thread_question,
    pick_thread_tool,
)

_BOUND = {THREAD_TOOL_NAME, "mattermost_search_knowledge"}


@pytest.mark.parametrize("text", [
    "이 스레드에 여러개의 PDF 파일이 있는데 각각의 파일에 대하여 정리를 해 주세요",
    "이 쓰레드 내용 요약해줘",
    "해당 스레드에서 결정된 사항 알려줘",
    "여기 올라온 파일 정리해줘",
    "여기에 첨부된 견적서 비교해줘",
    "위에서 공유한 문서 요약해줘",
    "summarize this thread",
])
def test_thread_deixis_is_detected(text):
    assert is_thread_question(text)


@pytest.mark.parametrize("text", [
    "이 채널 최근 글 알려줘",          # 채널 질문의 답은 채널 조회다 — 스레드로 몰면 안 된다
    "PDF 파일 정리해줘",
    "지난주 품질 이슈 찾아줘",
    "회사 scrum 채널에 올라온 자료 보여줘",   # 다른 채널을 명시했으면 그쪽이 맞다
])
def test_non_thread_questions_are_left_alone(text):
    assert not is_thread_question(text)


def test_forces_thread_tool_inside_a_thread():
    assert pick_thread_tool("이 스레드 정리해줘", _BOUND, "root-9") == THREAD_TOOL_NAME


def test_no_force_outside_a_thread():
    """채널 최상위 글에는 스레드가 없다 — 강제할 대상 자체가 없다."""
    assert pick_thread_tool("이 스레드 정리해줘", _BOUND, None) is None
    assert pick_thread_tool("이 스레드 정리해줘", _BOUND, "") is None


def test_no_force_when_tool_is_not_bound():
    """도구가 안 붙은 경로(MCP 미배포 등)에서 강제하면 tool NOT FOUND 가 난다."""
    assert pick_thread_tool("이 스레드 정리해줘", {"mattermost_search_knowledge"}, "root-9") is None


def test_no_force_for_ordinary_questions():
    assert pick_thread_tool("오늘 일정 알려줘", _BOUND, "root-9") is None


# ── 조회 범위: 지시어가 있을 때만 현재 채널로 묶는다 ─────────────────────────
#
# ⚠️ 지시어가 없을 때 묶으면 **사내 지식검색이 죽는다.** "지난주 품질 이슈 찾아줘" 는
# 채널을 말하지 않지만 전사 검색이 정답이다. 이 표가 그 경계를 지킨다.

from agent_arch.agents.react.thread_first import (  # noqa: E402
    is_channel_question,
    scope_to_current,
    wants_broader_scope,
)


@pytest.mark.parametrize("text", [
    "이 채널 최근 글 알려줘",
    "해당 채널에서 논의된 내용",
    "여기 채널에 공유된 자료",
    "현재 방에서 나온 얘기",
])
def test_channel_deixis_is_detected(text):
    assert is_channel_question(text)


@pytest.mark.parametrize("text", [
    "다른 채널에서도 논의됐나?",
    "전사 기준으로 찾아줘",
    "회사 전체에서 검색해줘",
    "예전에도 비슷한 이슈 있었어?",
])
def test_broader_scope_intent_is_detected(text):
    assert wants_broader_scope(text)


@pytest.mark.parametrize("text", [
    "이 스레드에 있는 PDF 정리해줘",
    "이 채널 최근 글 알려줘",
    "여기 올라온 파일 요약",
])
def test_scoped_to_current_when_user_points_here(text):
    assert scope_to_current(text)


@pytest.mark.parametrize("text", [
    "지난주 품질 이슈 찾아줘",              # 사내 지식검색의 본래 용도 — 묶으면 죽는다
    "작년 KTL 견적 어떻게 됐지",
    "회사 scrum 채널 자료 보여줘",          # 다른 채널을 명시했으면 그쪽이 맞다
])
def test_not_scoped_without_deixis(text):
    assert not scope_to_current(text)


@pytest.mark.parametrize("text", [
    "이 스레드에서 말한 그 이슈, 다른 데서도 논의됐나?",
    "이 채널이랑 다른 채널 비교해줘",
    "여기서 나온 얘기가 예전에도 있었는지 봐줘",
])
def test_broader_intent_beats_deixis(text):
    """지시어가 있어도 '넓혀 달라'가 있으면 넓힌다 — 우선순위가 반대면 답을 못 찾는다."""
    assert not scope_to_current(text)


# ─── 스레드 맥락은 고정이 아니라 주입으로 (2026-09-22) ───────────────────────

_KNOWN = {THREAD_TOOL_NAME, "mattermost_skill_find_posts_by_author", "search_knowledge"}


def test_thread_context_no_longer_depends_on_this_pin():
    """스레드 안 멘션의 맥락은 봇이 직접 읽어 주입한다 — 여기 고정은 지름길로만 남는다.

    2026-09-22 실측: "내가 방금 뭐라했지?" 가 기억 도메인으로 라우팅돼 메신저 도구가 아예
    안 붙었고, 고정 장치는 도구가 바인딩돼야 작동하므로 발동할 수 없었다. 도구 층에서는
    막을 수 없는 종류다.
    """
    assert pick_thread_tool("내가 방금 뭐라했지?", _KNOWN, "root-9") is None
    assert pick_thread_tool("이 스레드 파일 전부 정리해줘", _KNOWN, "root-9") == THREAD_TOOL_NAME
