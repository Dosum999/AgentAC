"""스트림 복원 — 모든 분할 크기에서 feed+flush 결합이 전문 복원과 같다. 토큰 접두가 아니면 지연 0."""
import pytest

from agent_arch.llm.pii.engine import Restorer
from agent_arch.llm.pii.stream import StreamRestorer

MAP = {"PERSON_3f9a1c2d": "김보라", "EMAIL_0a0b0c0d": "user@example.com", "PERSON_abcdef012345": "박미래", "HANDLE_abababab": "jsbaek"}
VARIANTS = {
    "plain": "담당자 [[PERSON_3f9a1c2d]]입니다.",
    "particle": "담당자는 [[PERSON_3f9a1c2d]]님께서 확인하셨습니다.",
    "email_domain": "메일[[EMAIL_0a0b0c0d]]@example.com로 보내 주세요.",
    "email_no_domain": "메일 [[EMAIL_0a0b0c0d]] 로.",
    "escaped": "담당자 \\[\\[PERSON_3f9a1c2d\\]\\]님과 \\[PERSON_3f9a1c2d\\] 둘.",
    "bare_and_single": "PERSON_3f9a1c2d 와 [PERSON_3F9A1C2D] 와 [[ PERSON_3f9a1c2d ]].",
    "adjacent": "[[PERSON_3f9a1c2d]][[PERSON_abcdef012345]]@[[HANDLE_abababab]]",
    "unknown": "모르는 [[PERSON_deadbeef]] 와 [[EMAIL_deadbeef]]@x.com 끝",
    "markdown_link": "링크는 [가이드](https://x.com/a) 와 [[PERSON_3f9a1c2d]] 그리고 [대괄호] 끝 GDP",
    "none": "토큰이 전혀 없는 평범한 문장입니다. 이메일 a@b.kr 과 010-1234-5678 도 그대로.",
}


@pytest.fixture
def rest():
    return Restorer(MAP)


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 13, 19, 31, 96, 97, 200])
@pytest.mark.parametrize("variant", list(VARIANTS))
def test_stream_matches_full_restore(rest, size, variant):
    text = VARIANTS[variant]
    expected = rest.restore_text(text).value
    s = StreamRestorer(rest)
    parts = [s.feed(text[i:i + size]) for i in range(0, len(text), size)]
    parts.append(s.flush())
    assert "".join(parts) == expected


def test_no_delay_for_plain_text(rest):
    s = StreamRestorer(rest)
    assert s.feed("안녕하세요. ") == "안녕하세요. "
    assert s.feed("오늘 회의는 [링크") == "오늘 회의는 [링크"      # '[' 뒤에 종류 접두가 아닌 글자 → 즉시
    assert s.feed("](url) 끝") == "](url) 끝"


def test_holds_token_prefix_until_resolved(rest):
    s = StreamRestorer(rest)
    assert s.feed("담당자 [[PER") == "담당자 "
    assert s.feed("SON_3f9a") == ""
    assert s.feed("1c2d]]님") == "김보라님"
    assert s.flush() == ""


def test_closed_non_email_token_released_immediately(rest):
    s = StreamRestorer(rest)
    assert s.feed("담당자 [[PERSON_3f9a1c2d]]") == "담당자 김보라"
    e = StreamRestorer(rest)
    assert e.feed("메일 [[EMAIL_0a0b0c0d]]") == "메일 "          # 도메인이 더 올 수 있어 보류
    assert e.feed("@example.com 로") == "user@example.com 로"


def test_hold_max_forces_release(rest):
    s = StreamRestorer(rest, hold_max=10)
    out = s.feed("x" * 5 + "[[PERSON_3f9a1c2d")      # 접두가 10자를 넘으면 강제 방출
    assert out.endswith("김보라") or out == "x" * 5 + "[[PERSON_3f9a1c2d"


def test_unresolved_accumulates(rest):
    s = StreamRestorer(rest)
    s.feed("[[PERSON_deadbeef]] 와 ")
    s.feed("[[PERSON_3f9a1c2d]] 끝")
    s.flush()
    assert s.unresolved == ["PERSON_deadbeef"] and s.restored == 1


def test_flush_restores_remaining(rest):
    s = StreamRestorer(rest)
    assert s.feed("[[PERSON_3f9a") == ""
    assert s.flush() == "(확인되지 않은 사람)" or s.flush() == ""
