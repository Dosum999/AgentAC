"""도구 결과 절삭 — JSON 구조를 지키고 식별자를 잃지 않는다.

예전에는 `content[:8000]` 문자 절단이었다. 도구 결과는 json.dumps 산출물이라 그렇게
자르면 **JSON 이 깨지고 뒤쪽 항목이 통째로 사라진다.** 실측(2026-08-24): 지식검색 결과가
10,331~16,909자로 상한에 걸려 다음 턴에 앞쪽 1~3건만 남았고, 뒤쪽 히트의 file_id 가
사라져 "N번째 문서 찾아줘"·"그 문서 요약해줘"가 재료 없이 실패했다.
"""
import json

from agent_arch.memory.conversation import MAX_TOOL_RESULT_CHARS, trim_tool_result_content


def _search_result(n=5, body_chars=4000):
    """지식검색 결과 모양 — 히트마다 긴 본문 + 짧은 식별자."""
    return json.dumps({
        "count": n,
        "results": [
            {"file_id": f"fid-{i}", "file_name": f"문서_{i}.pdf", "channel_id": f"c{i}",
             "channel_name": "개발팀", "score": 0.9 - i * 0.1,
             "text": f"{i}번 문서 본문 " + "가" * body_chars}
            for i in range(n)
        ],
    }, ensure_ascii=False)


def test_short_content_is_untouched():
    small = json.dumps({"a": "b"}, ensure_ascii=False)
    assert trim_tool_result_content("t", small) == small


def test_result_stays_valid_json():
    out = trim_tool_result_content("mattermost_search_knowledge", _search_result())

    json.loads(out)                      # 깨진 JSON 이면 여기서 터진다
    assert len(out) <= MAX_TOOL_RESULT_CHARS


def test_all_hits_and_file_ids_survive():
    """이게 무너지면 'N번째 문서'·'요약해줘'가 다시 재료를 잃는다."""
    out = json.loads(trim_tool_result_content("mattermost_search_knowledge", _search_result()))

    assert len(out["results"]) == 5
    assert [h["file_id"] for h in out["results"]] == [f"fid-{i}" for i in range(5)]
    assert [h["file_name"] for h in out["results"]] == [f"문서_{i}.pdf" for i in range(5)]


def test_only_long_text_is_shortened():
    out = json.loads(trim_tool_result_content("t", _search_result()))

    for i, h in enumerate(out["results"]):
        assert h["text"].startswith(f"{i}번 문서 본문")   # 머리는 남는다
        assert len(h["text"]) < 4000                     # 뒷부분만 줄었다
        assert h["channel_name"] == "개발팀"              # 짧은 필드는 그대로


def test_extreme_case_drops_items_but_records_count():
    # 히트가 아주 많으면 문자열을 최소로 줄여도 상한을 넘는다 — 그때는 항목을 덜어내되
    # 몇 개를 뺐는지 남긴다(조용히 사라지면 LLM 이 '그게 전부'라고 오해한다).
    out = json.loads(trim_tool_result_content("t", _search_result(n=400, body_chars=200)))

    assert len(out["results"]) < 400
    assert out["_생략된_항목수"] > 0


def test_non_json_falls_back_to_char_cut():
    plain = "가" * 20000
    out = trim_tool_result_content("some_plain_tool", plain)

    assert len(out) <= MAX_TOOL_RESULT_CHARS + 40
    assert out.startswith("가")


# ── 총량 상한 (한 턴의 모든 도구 결과 합계) ──────────────────────────────────
#
# 개별 상한만 두면 최악이 묶이지 않는다. `mattermost_read_file` 하나가 30,000자까지
# 나오므로 "검색 + 파일 2개 읽기" 턴이면 이월분만 9만자(≈40k 토큰)가 되는데,
# `load_history` 는 이걸 **하위 예산 없이 그대로** 프롬프트에 주입한다.

import json as _json

from agent_arch.memory.conversation import (
    MAX_TOOL_CTX_TOTAL_CHARS,
    MAX_TOOL_RESULT_CHARS,
    trim_tool_result_content,
    trim_tool_results,
)


def _res(name, payload):
    return {"tool_call_id": f"c-{name}", "name": name, "content": _json.dumps(payload, ensure_ascii=False)}


def _big_doc(n_chars, fid="f1"):
    return {"file_id": fid, "name": "계약서.pdf", "text": "가" * n_chars}


def _total(rows):
    return sum(len(r["content"]) for r in rows)


def test_small_turn_is_left_alone():
    """평상시엔 아무것도 하지 않는다 — 예산 안이면 원문 그대로 이월된다."""
    rows = [_res("search", {"results": [{"file_id": "a", "text": "짧은 본문"}]})]
    out = trim_tool_results(rows)
    assert out[0]["content"] == rows[0]["content"]


def test_per_result_cap_is_20000_not_8000():
    """top_k=10 지식검색(11k~16k자)이 눌리지 않아야 한다."""
    assert MAX_TOOL_RESULT_CHARS == 20000
    rows = [_res("search", {"results": [_big_doc(16000)]})]
    out = trim_tool_results(rows)
    assert len(out[0]["content"]) <= MAX_TOOL_RESULT_CHARS
    assert "…(이하 생략)" not in out[0]["content"]      # 축약이 걸리지 않았다


def test_total_budget_is_enforced_across_results():
    """검색 + 파일 3개 = 개별 상한은 다 지켜도 합계가 예산을 넘는 턴."""
    rows = [_res(f"read_file_{i}", _big_doc(19000, fid=f"f{i}")) for i in range(4)]
    assert _total(rows) > MAX_TOOL_CTX_TOTAL_CHARS
    out = trim_tool_results(rows)
    assert _total(out) <= MAX_TOOL_CTX_TOTAL_CHARS
    assert len(out) == 4                                  # 결과를 통째로 버리지 않는다


def test_identifiers_survive_total_shrink():
    """총량 축약이 걸려도 file_id 는 남아야 한다 — 남으면 다시 읽을 수 있다."""
    rows = [_res(f"read_file_{i}", _big_doc(19000, fid=f"f{i}")) for i in range(4)]
    out = trim_tool_results(rows)
    for i, r in enumerate(out):
        assert _json.loads(r["content"])["file_id"] == f"f{i}"


def test_small_results_are_not_punished_for_a_big_neighbor():
    """공통 상한은 그보다 짧은 결과를 건드리지 않는다 — 큰 것부터 줄어든다."""
    small = _res("calendar", {"events": ["9시 회의"]})
    bigs = [_res(f"read_file_{i}", _big_doc(19000, fid=f"f{i}")) for i in range(4)]
    out = trim_tool_results([small] + bigs)
    assert out[0]["content"] == small["content"]
    assert _total(out) <= MAX_TOOL_CTX_TOTAL_CHARS


def test_order_and_ids_are_preserved():
    rows = [_res(f"t{i}", _big_doc(19000, fid=f"f{i}")) for i in range(4)]
    out = trim_tool_results(rows)
    assert [r["name"] for r in out] == [f"t{i}" for i in range(4)]
    assert [r["tool_call_id"] for r in out] == [f"c-t{i}" for i in range(4)]


def test_pathological_tool_count_still_respects_the_budget():
    """도구가 비정상적으로 많아도 예산은 약속이다(마지막엔 문자 절단으로라도 지킨다)."""
    rows = [_res(f"t{i}", _big_doc(2000, fid=f"f{i}")) for i in range(200)]
    out = trim_tool_results(rows)
    assert _total(out) <= MAX_TOOL_CTX_TOTAL_CHARS


def test_empty_turn():
    assert trim_tool_results([]) == []


# ── 범위 오류(scope_error) payload 는 이월에서 살아남아야 한다 ───────────────────
#
# 지식검색이 팀·채널 이름을 못 풀면 검색하지 않고 scope_error + 되묻기 후보를 준다.
# 사용자가 "고객사 A요" 라고 답하는 **다음 턴**에 그 후보가 사라져 있으면 모델은
# 무엇을 물었는지 모른 채 다시 헤맨다(그리고 범위 없이 전체 검색으로 샌다).

def _scope_error_payload():
    return {
        "results": [], "count": 0,
        "scope_error": {"reason": "team_ambiguous", "message": "여러 회사에 걸쳐 있습니다."},
        "candidates": [{"team_name": "CustomerA", "channel_name": "개발팀"},
                       {"team_name": "(주)성호금속 경주2공장", "channel_name": "개발팀"}],
    }


def test_scope_error_payload_survives_per_result_trim():
    payload = _json.dumps(_scope_error_payload(), ensure_ascii=False)
    assert trim_tool_result_content("mattermost_search_knowledge", payload) == payload


def test_scope_error_survives_total_budget_shrink():
    """옆에 거대한 첨부 전문이 있어도 후보 팀명은 남아야 한다."""
    rows = [_res("mattermost_search_knowledge", _scope_error_payload())]
    rows += [_res(f"read_file_{i}", _big_doc(19000, fid=f"f{i}")) for i in range(4)]

    out = trim_tool_results(rows)
    kept = _json.loads(out[0]["content"])

    assert kept["scope_error"]["reason"] == "team_ambiguous"
    assert {c["team_name"] for c in kept["candidates"]} == {
        "CustomerA", "(주)성호금속 경주2공장",
    }
