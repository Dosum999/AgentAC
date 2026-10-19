from types import SimpleNamespace as M

from agent_arch.agents.react.history import recent_user_turns, FEEDBACK_NAME


def test_recent_user_turns_excludes_current_turn_tool_ctx_and_feedback():
    msgs = [
        M(role="system", content="sys"),
        M(role="user", content="점심식대 올려줘 참석자 회사벤쳐스 한서준"),
        M(role="assistant", content="초안 준비", tool_calls=None),
        M(role="user", content="[도구 결과] …"),
        M(role="assistant", content="tool", tool_calls=[{"id": "x"}]),
        M(role="user", content="결재자는 이동수"),
        M(role="assistant", content="보강 전 초안", name=FEEDBACK_NAME),
        M(role="assistant", content="확인했어요"),
        M(role="user", content="금액 2만원으로"),          # 이번 턴
    ]
    assert recent_user_turns(msgs) == ["점심식대 올려줘 참석자 회사벤쳐스 한서준", "결재자는 이동수"]
    assert recent_user_turns(msgs, max_turns=1) == ["결재자는 이동수"]
    assert recent_user_turns(msgs, max_chars=5)[0] == "점심식대 "
    assert recent_user_turns([]) == []
