"""첨부가 걸린 턴은 답변을 붙들었다가 고쳐서 한 번에 낸다.

왜 있는가 — 링크 가드는 **최종 본문**에서만 돈다. 붙들지 않으면 화면에는 날조된 파일
주소가 그대로 찍히고 저장본만 고쳐져, 이사님이 보는 화면은 신고 당시와 똑같다
(SSE 에는 본문 교체 이벤트가 없어 되돌릴 수 없다). 이름 가드가 hold 를 도입한 이유와 같다.
"""
from __future__ import annotations

import json

from agent_arch.agents.graphs.react_graph import _link_guard_needed
from agent_arch.agents.graphs.supervisor import _link_hold_needed, _worker_tool_results


def _file_result():
    return {"tool_call_id": "1", "name": "mattermost_search_knowledge",
            "content": json.dumps({"results": [{"file_id": "a" * 26, "doc_url": "u"}]})}


def _plain_result():
    return {"tool_call_id": "1", "name": "mattermost_search_knowledge",
            "content": json.dumps({"results": [{"channel_name": "개발팀", "text": "x"}]})}


def test_react_graph_holds_when_the_turn_has_attachment_hits():
    assert _link_guard_needed({"collected_tool_results": [_file_result()]}) is True


def test_react_graph_does_not_hold_ordinary_turns():
    assert _link_guard_needed({"collected_tool_results": [_plain_result()]}) is False
    assert _link_guard_needed({"collected_tool_results": []}) is False
    assert _link_guard_needed({}) is False


class _Worker:
    def __init__(self, results):
        self.tool_results = results


def test_supervisor_reads_worker_tool_results():
    ok = [("mattermost", _Worker([_file_result()]))]
    assert _worker_tool_results(ok) == [_file_result()]
    assert _link_hold_needed(ok) is True


def test_supervisor_does_not_hold_without_attachments():
    assert _link_hold_needed([("mattermost", _Worker([_plain_result()]))]) is False
    assert _link_hold_needed([("mattermost", _Worker(None))]) is False
    assert _link_hold_needed([]) is False
