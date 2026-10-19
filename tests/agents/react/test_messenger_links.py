"""첨부 링크의 목적지는 코드가 정한다 (`react.messenger_links`).

왜 있는가 — 2026-09-18 운영: 링크는 들어갔는데 **엉뚱한 글로 갔다.** 영웅개발자들 채널의
두 항목(견적서 PDF, image.png)이 **같은 주소**를 달았고 그 글은 둘 중 어느 파일과도
무관했다. 주소 자체는 실재하는 permalink 라 날조 가드에 걸릴 것이 없다 — 틀린 것은
**짝짓기**다. 같은 이름(`image.png`)이 채널마다 있어 모델이 뒤바꿔도 눌러 보기 전엔
아무도 모른다.
"""
from __future__ import annotations

import json

from agent_arch.agents.react.messenger_links import apply_file_links, file_refs

_BASE = "https://chat.example.com"
_PDF = _BASE + "/team/pl/" + "p" * 26
_IMG = _BASE + "/team/pl/" + "i" * 26
_SCRUM = _BASE + "/team/pl/" + "s" * 26
_WRONG = _BASE + "/team/pl/" + "w" * 26
_PDF_NAME = "260522_팝빌 연동서비스 표준 견적서_주식회사 회사.pdf"


def _hit(name, fid, url, channel):
    return {"source": "messenger_file", "channel_name": channel, "title": name,
            "file_name": name, "file_id": fid, "doc_url": url, "text": "…"}


def _results(*hits):
    return [{"tool_call_id": "1", "name": "mattermost_search_knowledge",
             "content": json.dumps({"results": list(hits), "count": len(hits)},
                                   ensure_ascii=False)}]


_TR = _results(
    _hit(_PDF_NAME, "a" * 26, _PDF, "영웅개발자들"),
    _hit("image.png", "b" * 26, _IMG, "영웅개발자들"),
    _hit("image.png", "c" * 26, _SCRUM, "회사 scrum"),
)


def test_refs_carry_name_channel_and_permalink():
    refs = file_refs(_TR)
    assert [r["file_id"] for r in refs] == ["a" * 26, "b" * 26, "c" * 26]
    assert refs[1] == {"file_id": "b" * 26, "name": "image.png",
                       "channel": "영웅개발자들", "url": _IMG}


def test_refs_fold_chunks_of_the_same_file():
    tr = _results(_hit("a.pdf", "a" * 26, _PDF, "ch"), _hit("a.pdf", "a" * 26, _PDF, "ch"))
    assert len(file_refs(tr)) == 1


def test_refs_survive_non_json_results():
    assert file_refs([{"tool_call_id": "1", "name": "x", "content": "[ERROR] 실패"}]) == []
    assert file_refs(None) == []


def test_wrong_pairing_is_corrected_per_file():
    """사고 재현 — 세 항목이 같은(엉뚱한) 주소를 달았다."""
    answer = (
        f"*   **{_PDF_NAME} (영웅개발자들 채널)**\n"
        f"    *   링크: [{_PDF_NAME}]({_WRONG})\n"
        f"*   **image.png (영웅개발자들 채널)**\n"
        f"    *   링크: [image.png]({_WRONG})\n"
    )
    out, n = apply_file_links(answer, file_refs(_TR), download_base=_BASE)
    assert n == 2
    assert f"[{_PDF_NAME}]({_PDF})" in out
    assert f"[image.png]({_IMG})" in out
    assert _WRONG not in out


def test_same_file_name_is_disambiguated_by_the_nearest_channel():
    answer = (
        f"*   **image.png (회사 scrum 채널)**\n    *   링크: [image.png]({_WRONG})\n"
        f"*   **image.png (영웅개발자들 채널)**\n    *   링크: [image.png]({_WRONG})\n"
    )
    out, _ = apply_file_links(answer, file_refs(_TR), download_base=_BASE)
    lines = [ln for ln in out.split("\n") if "링크:" in ln]
    assert _SCRUM in lines[0] and _IMG in lines[1]


def test_ambiguous_name_drops_the_link_rather_than_guessing():
    """채널을 못 밝힌 줄에서 동명 파일 둘이면, 틀린 주소를 남기느니 라벨만 남긴다."""
    out, n = apply_file_links(f"첨부: [image.png]({_WRONG})", file_refs(_TR), download_base=_BASE)
    assert out == "첨부: image.png"
    assert n == 0


def test_correct_link_is_left_as_is_without_a_download_link():
    """파일 직행 링크는 덧붙이지 않는다 — 링크가 두 개씩 붙어 답변이 지저분해진다."""
    out, _ = apply_file_links(f"[image.png]({_IMG})", file_refs(_TR), download_base=_BASE)
    assert out == f"[image.png]({_IMG})"


def test_download_link_is_not_duplicated_when_the_guard_already_added_one():
    dl = f"{_BASE}/api/v4/files/{'b' * 26}"
    text = f"[image.png]({_IMG}) ([내려받기]({dl}))"
    out, _ = apply_file_links(text, file_refs(_TR), download_base=_BASE)
    assert out.count("내려받기") == 1


def test_file_without_a_permalink_still_gets_the_direct_link():
    """팀 슬러그를 못 구해 `doc_url` 이 빈 히트 — 주소를 지어내지 않고 직행 링크만 준다."""
    tr = _results(_hit("only.pdf", "d" * 26, "", "ch"))
    out, _ = apply_file_links(f"[only.pdf]({_WRONG})", file_refs(tr), download_base=_BASE)
    assert out == f"[only.pdf]({_BASE}/api/v4/files/{'d' * 26})"


def test_image_markdown_becomes_a_normal_link():
    out, _ = apply_file_links(f"![image.png]({_IMG})", file_refs(_TR), download_base=_BASE)
    assert not out.startswith("!")


def test_links_that_are_not_files_are_left_alone():
    answer = (f"대화: [원문]({_WRONG})\n외부: [위키](https://example.atlassian.net/wiki/x/A)")
    out, n = apply_file_links(answer, file_refs(_TR), download_base=_BASE)
    assert out == answer and n == 0


def test_noop_without_refs_or_text():
    assert apply_file_links("본문", [], download_base=_BASE) == ("본문", 0)
    assert apply_file_links("", file_refs(_TR), download_base=_BASE) == ("", 0)


def test_escaped_label_still_matches():
    """검색 히트의 `title` 은 escape 돼 모델에게 간다 — 라벨 글자가 원본과 어긋난다."""
    tr = _results(_hit("a_b.pdf", "e" * 26, _PDF, "ch"))
    out, n = apply_file_links(f"[a\\_b.pdf]({_WRONG})", file_refs(tr), download_base=_BASE)
    assert n == 1 and _PDF in out
