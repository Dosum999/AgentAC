"""첨부 파일의 확장자 → MIME 추론 — **브라우저 라벨을 못 믿을 때의 보완책.**

업로드 시 형식 판정은 브라우저가 붙인 `content_type` 으로 한다. 그런데 그 값은 사실이
아니라 **보내는 쪽의 주장**이고, OS 에 타입이 등록돼 있지 않으면 `application/octet-stream`
으로 온다. 바이트는 멀쩡한 PDF 인데 라벨 때문에 거부되는 일이 실제로 있었다
(실측 2026-09-22: 고객 도면 PDF 가 octet-stream 으로 올라와 400).

**표가 여러 벌이면 한쪽만 고쳐진다.** 실제로 그랬다 — 도면 판독 엔드포인트
(`api/v1/drawing.py`)는 같은 이유로 이미지·PDF 폴백을 진작 갖고 있었는데, 에이전트 첨부
쪽은 `spreadsheet.py` 안의 엑셀 전용 표(`.xlsx`·`.xls`·`.csv`)만 보고 있었다. 함수 이름
(`get_mime_from_filename`)과 호출부 주석("불명확한 MIME 은 파일명 확장자로 보완")이
범용처럼 보여 그 어긋남이 가려져 있었다. 여기로 모아 두 사본을 없앤다.

**문서(docx·pptx·hwp·txt…)는 표를 여기 두지 않고 위임한다.** 문서는 규칙 자체가 다르다 —
폴백이 아니라 *항상* 확장자로 판정하고(브라우저가 `.py`·`.json` 에도 text/plain 을 붙인다),
`.hwp` 는 rhwp 가 로드될 때만 유효한 **런타임 조건**이 붙는다. 그 표는 생성 쪽
(`FORMAT_MIME`)과도 공유하므로 제자리에 두는 편이 어긋날 여지가 적다.
"""
from pathlib import Path
from typing import Optional

#: 확장자 → MIME. 여기 있는 것은 **업로드가 받아주는 형식**과 1:1 이다
#: (`app/models/agent.py` 의 `_MAX_SIZE_BY_MIME`·`_MAX_SIZE_BY_SPREADSHEET_MIME`).
#: 새 형식을 받기 시작하면 두 곳을 함께 고쳐야 한다.
_EXT_TO_MIME: dict[str, str] = {
    # 이미지 — `.heic`/`.heif` 는 아이폰 기본 포맷인데 윈도우에 등록이 없는 경우가 흔해
    # octet-stream 으로 오기 쉽다. 현장에서 도면을 찍어 올리는 경로가 여기에 걸린다.
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".heic": "image/heic",
    ".heif": "image/heif",
    # PDF
    ".pdf": "application/pdf",
    # 스프레드시트
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xls": "application/vnd.ms-excel",
    ".csv": "text/csv",
}


def get_mime_from_filename(filename: str) -> Optional[str]:
    """파일명 확장자로 MIME 을 추론한다. 모르면 None.

    문서 형식(docx·hwp 등)은 다루지 않는다 — 문서 파서는 이 레포 범위 밖이다.

    Args:
        filename: 파일명 (확장자 포함)

    Returns:
        추론된 MIME. 지원하지 않는 확장자면 None.
    """
    ext = Path(filename).suffix.lower()
    hit = _EXT_TO_MIME.get(ext)
    if hit:
        return hit
    # 문서 형식(docx·hwp 등) 파서는 이 레포 범위 밖이다 — 모르는 확장자는 None.
    return None
