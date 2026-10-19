"""합성 답변의 이름 가드는 질문자 본인 이름을 지어낸 이름으로 잡지 않는다.

실측(2026-10-02 메신저): talk+drive 합성 답이 질문자를 "이영희 이사님" 으로 불렀는데, 합성 근거에 [요청자 정보] 가
빠져 있어 그 이름을 "이름미확인" 으로 바꿨다. 일반 경로(react_graph)는 같은 블록을 이미 근거에 넣는다.
"""
from types import SimpleNamespace

from agent_arch.agents.graphs import supervisor as sup
from agent_arch.core.config import settings

REQUESTER = "[요청자 정보]\n- 이름: 이영희\n- 이메일: x@example.com\n- 소속 회사: (주)회사\n"
# 영문 ID 작성자가 섞인 워커 결과 — 이름 가드가 무장하는 조건(has_unnamed_author)
WORKERS = [("talk", SimpleNamespace(content="PROC 소통채널은 @lho 가 만든 채널입니다.", tool_results=[]))]
DRAFT = "이영희 이사님, PROC 소통채널은 발주사·공급사가 함께 쓰는 채널입니다."


def test_requester_name_is_kept(monkeypatch):
    monkeypatch.setattr(settings, "name_answer_guard", True)
    out = sup._guard_synth_names(DRAFT, "PROC 소통채널 설명해 줘", WORKERS, REQUESTER)
    assert out.startswith("이영희 이사님") and "이름미확인" not in out


def test_fabricated_name_is_still_masked(monkeypatch):
    monkeypatch.setattr(settings, "name_answer_guard", True)
    out = sup._guard_synth_names("김철수 부장님이 만든 채널입니다.", "PROC 소통채널 설명해 줘", WORKERS, REQUESTER)
    assert "김철수" not in out and "이름미확인" in out


def test_requester_block_from_system_message():
    msgs = [SimpleNamespace(role="system", content="규칙...\n\n" + REQUESTER + "\n더 규칙"),
            SimpleNamespace(role="user", content="질문")]
    assert "이영희" in sup._requester_block(msgs)
