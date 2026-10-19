"""메신저 첨부파일 링크 — 파일과 주소의 짝을 **코드가 소유한다**.

문제(실측 2026-09-18 운영): 답변에 링크는 들어갔는데 **엉뚱한 글로 갔다.** 영웅개발자들
채널의 두 항목(`260522_팝빌…견적서.pdf`, `image.png`)이 **같은 주소**를 달고 있었고, 그
글은 둘 중 어느 파일과도 무관했다. 주소 자체는 실재하는 permalink 라 앞선 날조 가드
(`answer_guard.find_fabricated_permalinks`)에 걸릴 것이 없다 — 틀린 것은 **짝짓기**다.

짝짓기는 모델이 할 일이 아니다. 한 턴에 파일 히트가 여러 건 오고 이름이 겹치면
(`image.png` 가 채널마다 있다) 모델이 뒤바꿔도 **사용자는 눌러 보기 전엔 알 수 없다** —
이름 환각·날조 링크와 같은 등급의 "검증할 수 없는 거짓말" 이다. 그래서 `guide_links` 가
참고 링크 섹션을 가져온 것과 같은 방식으로, 파일 링크의 **목적지를 코드가 정한다**.

방법:
- 이번 턴 도구 결과에서 파일 히트를 뽑는다(`file_refs`) — 파일명·채널·`file_id`·
  `doc_url`(그 파일이 올라간 **글**의 permalink)이 한 dict 안에 같이 온다.
- 답변에서 **라벨이 파일명인 마크다운 링크**를 찾아 주소를 그 파일의 것으로 바로잡는다.
- 같은 이름이 여러 건이면 **그 줄 위에서 가장 최근에 언급된 채널**로 좁힌다(답변이
  "…(영웅개발자들 채널)" 처럼 채널을 밝히는 형식이라 실제로 갈린다).
- 그래도 못 좁히면 **링크를 뗀다.** 틀린 주소를 남기는 것보다 라벨만 남기는 편이 낫다.

우리가 주소를 **지어내지는 않는다** — 팀 슬러그를 못 구해 `doc_url` 이 빈 히트는 애초에
후보에서 빠진다(그 경우 파일 직행 주소만 남는다).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

#: Mattermost id 길이 — 26자. 색인 실측 분포가 {26} 단일값이다.
_MM_ID_LEN = 26
#: 파일 직행 주소 라벨. 예전에 permalink 옆에 덧붙였다가 **뺐다**(2026-09-18 사용자 요청)
#: — 링크가 두 개씩 붙어 답변이 지저분해진다. 상수는 기존 답변·히스토리에 남은 표기를
#: 접기(중복 제거) 위해 남긴다.
_DOWNLOAD_LABEL = "내려받기"
#: `[라벨](주소)` — 이미지 문법(`!`)도 같이 받는다. permalink 는 이미지가 아니라 글 주소다.
_MD_LINK = re.compile(r"(?P<bang>!?)\[(?P<label>[^\]\n]*)\]\((?P<url>[^)\s]*)\)")


def _norm_name(raw: str) -> str:
    """파일명 비교용 정규화 — 마크다운 escape(`\\_`)·공백·대소문자를 지운다.

    검색 히트의 ``title`` 은 모델에게 넘기기 전에 escape 되고(`_md_safe`), 모델은 그걸
    라벨로 옮겨 적으므로 원본 ``file_name`` 과 글자가 어긋난다.
    """
    return re.sub(r"\s+", "", (raw or "").replace("\\", "")).lower()


def _walk(node: Any, out: list) -> None:
    """도구 결과 JSON 을 훑어 파일 히트를 모은다(같은 dict 안의 값만 짝으로 본다)."""
    if isinstance(node, dict):
        fid = node.get("file_id")
        if isinstance(fid, str) and len(fid) == _MM_ID_LEN:
            # 검색 히트는 `file_name`/`title` + `doc_url`, read_file 은 `name` + `url`.
            name = node.get("file_name") or node.get("title") or node.get("name") or ""
            url = node.get("doc_url") or node.get("url") or ""
            if isinstance(name, str) and name:
                out.append({
                    "file_id": fid,
                    "name": name,
                    "channel": node.get("channel_name") or "",
                    "url": url if isinstance(url, str) and "/pl/" in url else "",
                })
        for v in node.values():
            _walk(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk(v, out)


def file_refs(tool_results: Optional[Iterable[Any]]) -> list[dict]:
    """이번 턴 도구 결과의 첨부 파일 목록.

    Args:
        tool_results: ``{"tool_call_id", "name", "content"}`` 모양의 도구 결과들
            (content 는 ``executor.serialize_result`` 가 만든 JSON 문자열).

    Returns:
        ``{"file_id", "name", "channel", "url"}`` 목록. ``file_id`` 중복은 첫 건만 남는다
        (한 파일이 청크 여러 개로 오지만 주소는 하나다).
    """
    found: list[dict] = []
    for tr in tool_results or []:
        content = tr.get("content") if isinstance(tr, dict) else getattr(tr, "content", None)
        if not isinstance(content, str) or "file_id" not in content:
            continue
        try:
            _walk(json.loads(content), found)
        except (ValueError, TypeError):
            continue  # `[ERROR] …` 처럼 JSON 이 아닌 결과
    out: list[dict] = []
    seen: set = set()
    for ref in found:
        if ref["file_id"] in seen:
            continue
        seen.add(ref["file_id"])
        out.append(ref)
    return out


def _download_url(base: str, file_id: str) -> str:
    return f"{base.rstrip('/')}/api/v4/files/{file_id}"


def _pick(cands: list[dict], channel: str, url: str) -> Optional[dict]:
    """같은 이름이 여럿일 때 하나로 좁힌다.

    ① 그 줄 위에서 마지막으로 언급된 채널 → ② 모델이 이미 적은 주소와 일치하는 후보
    (그 짝은 모델이 맞게 쓴 것이다) 순서. 둘 다 안 되면 None — 찍지 않는다.
    """
    if len(cands) == 1:
        return cands[0]
    if channel:
        narrowed = [c for c in cands if c["channel"] == channel]
        if len(narrowed) == 1:
            return narrowed[0]
    if url:
        matched = [c for c in cands if c["url"] and c["url"] == url]
        if len(matched) == 1:
            return matched[0]
    return None


def apply_file_links(
    answer: str,
    refs: list[dict],
    *,
    download_base: str = "",
) -> tuple[str, int]:
    """답변의 **파일명 라벨 링크**를 그 파일의 실제 주소로 바로잡는다.

    라벨이 이번 턴 파일 목록에 없는 링크(대화 permalink·외부 문서 등)는 건드리지 않는다.

    Args:
        answer: 가드를 거친 최종 답변 본문.
        refs: ``file_refs`` 결과.
        download_base: 메신저 base URL. 없으면 파일 직행 링크를 붙이지 않는다.

    Returns:
        ``(정리된 본문, 손댄 링크 수)``.
    """
    if not answer or not refs:
        return answer, 0
    by_name: dict[str, list[dict]] = {}
    for ref in refs:
        by_name.setdefault(_norm_name(ref["name"]), []).append(ref)
    # 긴 이름부터 본다 — 짧은 채널명이 긴 채널명의 일부일 수 있다.
    channels = sorted({r["channel"] for r in refs if r["channel"]}, key=len, reverse=True)

    touched = 0
    dropped = 0
    current = ""
    lines = (answer or "").split("\n")
    for i, line in enumerate(lines):
        for name in channels:
            if name in line:
                current = name
                break

        def _fix(m: re.Match, _ch=current) -> str:
            nonlocal touched, dropped
            label = m.group("label")
            cands = by_name.get(_norm_name(label))
            if not cands:
                return m.group(0)          # 파일 링크가 아니다 — 건드리지 않는다
            ref = _pick(cands, _ch, m.group("url"))
            if ref is None:
                # 못 좁혔다 — 틀린 주소를 남기느니 라벨만 남긴다.
                dropped += 1
                return label
            # 글 permalink 가 없으면(팀 슬러그 미해결) 파일 직행 주소로 간다 — 그 경우에만
            # 쓴다. 평소에 permalink 옆에 덧붙이는 것은 뺐다(링크가 두 개씩 붙는다).
            url = ref["url"] or (_download_url(download_base, ref["file_id"]) if download_base else "")
            if not url:
                dropped += 1
                return label
            fixed = f"[{label}]({url})"
            if fixed != m.group(0):
                touched += 1
            return fixed

        new = _MD_LINK.sub(_fix, line)
        # 가드가 이미 붙여 둔 내려받기 링크가 중복되지 않게 접는다.
        lines[i] = re.sub(r"(\(\[" + _DOWNLOAD_LABEL + r"\]\([^)]*\)\))(?:\s*\1)+", r"\1", new)

    if touched or dropped:
        logger.warning("[file_links] 파일 링크 정정 %d건 / 모호해서 링크 제거 %d건 (후보 %d개)",
                       touched, dropped, len(refs))
    return "\n".join(lines), touched
