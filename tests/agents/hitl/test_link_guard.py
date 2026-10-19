"""지어낸 메신저 딥링크 차단 (`answer_guard.redact_fabricated_permalinks`).

왜 있는가 — 2026-09-16 실사용: 답변의 「메신저 링크」가 `https://chat.example.com/dadc/pl/`
로 끝났다. **post_id 가 통째로 비어 있다.** 원래 값은
`https://chat.example.com/team/pl/hiozn4yzoibg9nwu1icjkahuzy` 였고 색인의 `doc_url`
도 멀쩡했다 — 모델이 그 값을 쓰지 않고 **현재 대화 중인 팀 슬러그**(`dadc`)에 `/pl/` 만
붙여 지어냈다. 같은 답변의 외부 문서 링크(`links` 에서 온 것)는 정상이었다.

프롬프트에 "`doc_url`·`permalink` 를 써라" 가 이미 있는데도 샜다. 링크는 눌러 보기 전엔
진위를 알 수 없어 **사용자가 검증 못 하는 거짓말**이라 이름 환각과 같은 등급으로 막는다.

판정 기준은 길이다 — Mattermost post_id 는 **정확히 26자**(운영 색인 실측: 분포 {26}).
"""
from __future__ import annotations

import json

import pytest

from unittest.mock import patch

from agent_arch.core.config import settings
from agent_arch.agents.hitl.answer_guard import (
    build_file_link_map,
    find_fabricated_permalinks,
    redact_fabricated_permalinks,
)

# 실제 사고에 나온 두 값 그대로
_REAL = "https://chat.example.com/team/pl/hiozn4yzoibg9nwu1icjkahuzy"
_FAKE = "https://chat.example.com/dadc/pl/"

_HOST = "https://chat.example.com"


@pytest.fixture(autouse=True)
def _fix_messenger_host():
    """판정은 메신저 호스트 기준이다 — 개발자 `.env` 값에 테스트가 흔들리지 않게 고정한다."""
    with patch.object(settings, "mattermost_base_url", _HOST):
        yield


def test_real_permalink_is_left_alone():
    """26자 post_id 는 실재하는 주소다 — 건드리면 정상 기능이 죽는다."""
    text = f"근거: [보기]({_REAL})"
    assert find_fabricated_permalinks(text) == []
    assert redact_fabricated_permalinks(text) == (text, [])


def test_empty_post_id_is_caught():
    assert find_fabricated_permalinks(f"[이동]({_FAKE})") == [_FAKE]


@pytest.mark.parametrize("pid", ["", "abc", "a" * 25, "a" * 27])
def test_wrong_length_ids_are_caught(pid):
    url = f"https://chat.example.com/dadc/pl/{pid}"
    assert find_fabricated_permalinks(f"[이동]({url})"), f"길이 {len(pid)} 를 놓쳤다"


def test_markdown_label_survives_so_tables_do_not_break():
    """표의 열이 통째로 비면 답이 깨져 보인다 — 링크만 풀고 라벨은 남긴다."""
    out, bad = redact_fabricated_permalinks(f"| TD4.hwp | [이동]({_FAKE}) |")
    assert out == "| TD4.hwp | 이동 |"
    assert bad == [_FAKE]


def test_bare_url_becomes_placeholder():
    out, _ = redact_fabricated_permalinks(f"메신저 글: {_FAKE} 입니다")
    assert _FAKE not in out
    assert "(링크 없음)" in out


def test_mixed_answer_keeps_the_good_and_drops_the_bad():
    """사고 당시 답변의 모양 — 외부 문서 링크는 정상이고 메신저 링크만 날조였다."""
    text = (
        f"- 링크: [에이전트의 역할과 권한](https://example.atlassian.net/wiki/x/AbCd)\n"
        f"- 메신저 링크: [이동]({_FAKE})\n"
        f"- 근거: [원문]({_REAL})"
    )
    out, bad = redact_fabricated_permalinks(text)
    assert bad == [_FAKE]
    assert "example.atlassian.net/wiki/x/AbCd" in out, "외부 문서 링크를 건드리면 안 된다"
    assert _REAL in out, "정상 permalink 를 건드리면 안 된다"
    assert "메신저 링크: 이동" in out


def test_no_links_is_a_noop():
    assert redact_fabricated_permalinks("링크 없는 답변") == ("링크 없는 답변", [])


def test_empty_input_is_safe():
    assert redact_fabricated_permalinks("") == ("", [])


def test_duplicates_are_reported_once_and_all_replaced():
    text = f"a [x]({_FAKE}) b [y]({_FAKE})"
    out, bad = redact_fabricated_permalinks(text)
    assert bad == [_FAKE]
    assert _FAKE not in out


# ── 파일 주소 날조 (2026-09-18 브라우저 패널 실사고) ──────────────────────────
#
# 답변에 나간 첨부 링크 두 개가 눌러도 아무 데도 가지 않았다. **id 는 26자로 멀쩡했다** —
# 잘린 게 아니라 경로가 가짜였다. Mattermost 웹앱에는 파일 전용 웹 경로가 없어
# `/{team}/files/{id}` 는 실재하지 않는다(그래서 색인도 `doc_url` 에 **글** permalink 를
# 싣는다). 지우기만 하면 "자료로 가는 길" 이 여전히 없으므로 진짜 permalink 로 바꿔 단다.
_FILE_ID = "pwr4bt3bopfqpbnin888qgmt3o"
_IMG_ID = "755rw55mw3rfup7yjiwqaahfcw"
_FAKE_FILE = f"https://chat.example.com/team/files/{_FILE_ID}"
_FAKE_IMG = f"https://chat.example.com/team/files/{_IMG_ID}"
_MAP = {_FILE_ID: _REAL}


@pytest.mark.parametrize("url", [_FAKE_FILE, _FAKE_IMG])
def test_file_path_is_fabricated_even_with_a_real_26_char_id(url):
    """길이만 보던 판정은 이걸 놓쳤다 — 경로 자체가 실재하지 않는다."""
    assert find_fabricated_permalinks(f"관련 파일: [문서.pdf]({url})") == [url]


def test_file_link_is_replaced_with_the_real_permalink():
    """파일 직행 링크는 덧붙이지 않는다 — 링크가 두 개씩 붙어 답변이 지저분해진다."""
    out, bad = redact_fabricated_permalinks(
        f"관련 파일: [팝빌 8월이용료.pdf]({_FAKE_FILE})", file_links=_MAP)
    assert bad == [_FAKE_FILE]
    assert out == f"관련 파일: [팝빌 8월이용료.pdf]({_REAL})"
    assert "내려받기" not in out


def test_image_syntax_loses_the_bang_because_a_permalink_is_not_an_image():
    out, _ = redact_fabricated_permalinks(
        f"관련 이미지: ![image.png]({_FAKE_FILE})", file_links=_MAP)
    assert out.startswith("관련 이미지: [image.png](")
    assert "!" not in out.split("관련 이미지: ")[1].split("]")[0]


def test_bare_file_url_becomes_the_permalink_without_the_parenthetical():
    out, _ = redact_fabricated_permalinks(
        f"자료는 {_FAKE_FILE} 입니다", file_links=_MAP)
    assert out == f"자료는 {_REAL} 입니다"


def test_trailing_punctuation_is_not_eaten():
    out, _ = redact_fabricated_permalinks(
        f"자료: {_FAKE_FILE}. 확인 바랍니다", file_links=_MAP)
    assert out == f"자료: {_REAL}. 확인 바랍니다"


def test_without_a_mapping_it_falls_back_to_the_old_behaviour():
    """매핑이 없으면 거짓 링크를 남기느니 라벨만 남긴다(기존 판단 유지)."""
    out, bad = redact_fabricated_permalinks(f"| 팝빌.pdf | [보기]({_FAKE_FILE}) |")
    assert bad == [_FAKE_FILE]
    assert out == "| 팝빌.pdf | 보기 |"


@pytest.mark.parametrize("url", [
    f"https://chat.example.com/api/v4/files/{_FILE_ID}",       # 파일 다운로드 주소(정화도 보존한다)
    "https://chat.example.com/team/channels/town-square",
    "https://chat.example.com/team/messages/@jdoe",
    f"{_REAL}?highlight=1",
    "https://chat.example.com/oauth/gitlab/login?redirect_to=%2Fx",
    "https://chat.example.com/team",
])
def test_real_routes_are_left_alone(url):
    text = f"근거: [보기]({url})"
    assert find_fabricated_permalinks(text) == []
    assert redact_fabricated_permalinks(text) == (text, [])


@pytest.mark.parametrize("url", [
    "https://drive.example.com/files/abc",
    "https://example.atlassian.net/wiki/x/AbCd",
    "https://app.example.com/main/download",
])
def test_other_hosts_are_never_judged(url):
    """allowlist 가 외부 도메인으로 새면 정상 링크를 몰살한다."""
    assert find_fabricated_permalinks(f"[링크]({url})") == []


def test_falls_back_to_pl_only_when_the_host_is_not_configured():
    with patch.object(settings, "mattermost_base_url", ""):
        assert find_fabricated_permalinks(f"[이동]({_FAKE})") == [_FAKE]
        assert find_fabricated_permalinks(f"[링크](https://example.atlassian.net/wiki/x/A)") == []


# ── file_id → permalink 매핑 ────────────────────────────────────────────────

def _hit(**kw):
    """`mattermost_search_knowledge` 파일 청크 히트 모양(retrieval/search.py)."""
    base = {
        "score": 0.5, "source": "messenger_file", "channel_name": "회사 scrum",
        "title": "팝빌 8월이용료.pdf", "text": "…본문 발췌…",
        "doc_url": _REAL, "file_id": _FILE_ID, "file_name": "팝빌 8월이용료.pdf",
        "post_id": "hiozn4yzoibg9nwu1icjkahuzy", "page": 1, "chunk_index": 0,
    }
    base.update(kw)
    return base


def _result(payload):
    return [{"tool_call_id": "1", "name": "mattermost_search_knowledge",
             "content": json.dumps(payload, ensure_ascii=False)}]


def test_map_is_built_from_search_hits():
    assert build_file_link_map(_result({"results": [_hit()], "count": 1})) == {_FILE_ID: _REAL}


def test_map_reads_read_file_results_too():
    out = build_file_link_map([{
        "tool_call_id": "1", "name": "mattermost_read_file",
        "content": json.dumps({"file_id": _FILE_ID, "name": "a.pdf", "url": _REAL}),
    }])
    assert out == {_FILE_ID: _REAL}


def test_map_skips_hits_without_a_permalink():
    """팀 슬러그를 못 구해 `doc_url` 이 빈 히트가 있다 — 우리가 주소를 지어내지 않는다."""
    assert build_file_link_map(_result({"results": [_hit(doc_url="")]})) == {}
    assert build_file_link_map(_result({"results": [_hit(doc_url=_FAKE_FILE)]})) == {}


def test_map_survives_non_json_results():
    assert build_file_link_map([
        {"tool_call_id": "1", "name": "x", "content": "[ERROR] file_id 조회 실패"},
        {"tool_call_id": "2", "name": "y", "content": None},
    ]) == {}
    assert build_file_link_map(None) == {}


def test_map_walks_nested_wrappers():
    out = build_file_link_map(_result({"items": [{"results": [_hit()]}]}))
    assert out == {_FILE_ID: _REAL}


def test_map_keeps_the_first_permalink_for_repeated_chunks():
    other = "https://chat.example.com/team/pl/aaaaaaaaaaaaaaaaaaaaaaaaaa"
    out = build_file_link_map(_result({"results": [_hit(), _hit(chunk_index=1, doc_url=other)]}))
    assert out == {_FILE_ID: _REAL}


def test_replacement_survives_the_sanitizer():
    """0e61d8c 의 순서 함정 — 가드는 정화 **앞**에서 돈다. 정화가 다시 깨면 헛일이다."""
    from agent_arch.utils.sanitize import strip_internal_ids

    out, _ = redact_fabricated_permalinks(f"[팝빌.pdf]({_FAKE_FILE})", file_links=_MAP)
    assert _REAL in strip_internal_ids(out)
