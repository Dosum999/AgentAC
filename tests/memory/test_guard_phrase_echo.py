"""가드 교정 문구가 이력을 통해 다음 턴에 전염되는 것을 막는다 — 2026-09-11 실측 회귀.

`answer_guard` 는 "일어나지 않은 액션 주장" 을 교정 문구로 대체한다. 그 문장은 **코드가
써 넣은 산출물**이지 모델의 말이 아니다. 그런데 그대로 이력에 쌓이면 다음 턴 프롬프트에
예시로 들어가고, 모델이 그걸 따라 쓴다.

실측: 가드 오발동(#351)을 고쳐 배포한 **뒤에도** 같은 문장이 계속 나왔다.
`[answer_guard]` 경고 로그가 없는데도 나왔으니 가드가 아니라 모델이 앞선 턴을 모방한
것이다(같은 스레드 00:40·00:45 → 배포 → 01:11·01:12).

사용자가 본 답변과 DB 저장본은 건드리지 않는다 — 화면과 히스토리가 어긋나면 그것대로
사고다(2026-08-10 실측). **이력으로 되돌려 주는 경로에서만** 걷어낸다.
"""
import pytest

from agent_arch.llm.base import Message
from agent_arch.memory.conversation import _strip_guard_artifact

_PHRASE = "아직 실제로 처리하지는 않았어요. 원하시면 다시 요청해 주세요."


def _clean(content: str, role: str = "assistant") -> str:
    return _strip_guard_artifact(Message(role=role, content=content)).content


class TestStripsTheArtifact:
    def test_phrase_is_removed(self):
        out = _clean(f"{_PHRASE} 아래 링크에서 다운로드하실 수 있습니다.")
        assert "아직 실제로 처리하지는 않았어요" not in out
        assert "다운로드하실 수 있습니다" in out

    def test_download_link_survives(self):
        """실제 성과물은 남아야 한다 — 그게 그 턴에 진짜 일어난 일이다."""
        out = _clean(f"{_PHRASE}\n\n[보고서.docx](https://x/dl)")
        assert "[보고서.docx](https://x/dl)" in out

    def test_trailing_period_variants(self):
        for variant in (_PHRASE, _PHRASE.replace(" 원하시면 다시 요청해 주세요.", ""),
                        "아직 실제로 처리하지는 않았어요"):
            out = _clean(f"{variant} 남은 문장입니다.")
            assert "남은 문장입니다." in out
            assert "처리하지는 않았어요" not in out


class TestLeavesEverythingElseAlone:
    def test_normal_answer_untouched(self):
        answer = "요청하신 보고서를 워드 파일로 생성했습니다.\n\n[보고서.docx](https://x)"
        assert _clean(answer) == answer

    def test_user_message_untouched(self):
        """사용자가 그 말을 했을 수도 있다 — 사용자 발화는 건드리지 않는다."""
        assert _clean(_PHRASE, role="user") == _PHRASE

    def test_phrase_only_message_is_kept(self):
        """지운 결과가 비면 원문 유지 — 빈 assistant 턴은 대화 구조를 깨뜨린다."""
        assert _clean(_PHRASE) == _PHRASE

    @pytest.mark.parametrize("content", ["", None])
    def test_empty_content_is_safe(self, content):
        msg = Message(role="assistant", content=content or "")
        assert _strip_guard_artifact(msg).content == (content or "")


# ── 초안 안내 문구도 같은 이유로 걷어낸다 (2026-09-15) ───────────────────────
#
# 실측: 전송 도구를 **한 번도 부르지 않은** 턴에서 "발송/등록 초안을 준비했어요" 라고
# 답했다. 카드도 초안도 없는데 승인을 기다리라고 말한 셈 — 하지 않은 일을 했다는 보고다.

_DRAFT = "발송/등록 초안을 준비했어요 — 아래 카드에서 승인해 주세요."


class TestDraftNoteEcho:
    def test_draft_note_is_removed(self):
        out = _clean(f"{_DRAFT}\n\n파일을 만들었습니다.")
        assert "초안을 준비했어요" not in out
        assert "파일을 만들었습니다." in out

    def test_message_with_only_the_note_is_kept(self):
        """지우면 빈 assistant 턴이 된다 — 대화 구조가 깨진다."""
        assert _clean(_DRAFT) == _DRAFT

    def test_both_artifacts_removed_together(self):
        out = _clean(f"아직 실제로 처리하지는 않았어요. {_DRAFT}\n\n결과입니다.")
        assert "아직 실제로" not in out and "초안을 준비했어요" not in out
        assert "결과입니다." in out
