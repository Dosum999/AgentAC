"""file-service 클라이언트 — 채팅 첨부파일을 Object Storage 에 저장한다.

presigned URL 3단계(발급 → 스토리지 직접 PUT → 완료 확인)로 업로드한다. 파일 본문은
file-service 를 거치지 않고 이 백엔드 ↔ Object Storage(NCP) 간 직접 전송된다.

FILE_SERVICE_BASE_URL + FILE_SERVICE_API_KEY 둘 다 설정돼야 활성화된다. 미설정 시
모든 함수가 조용히 no-op(None/빈값) — 채팅 흐름을 절대 막지 않는다(best-effort).
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import httpx

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)


def enabled() -> bool:
    """file-service 연동 활성화 여부."""
    return bool(settings.file_service_base_url and settings.file_service_api_key)


def _headers() -> Dict[str, str]:
    return {"X-API-Key": settings.file_service_api_key}


async def save_file(
    data: bytes,
    filename: str,
    content_type: str,
    *,
    subdir: str = "",
    timeout: float = 30.0,
) -> Optional[str]:
    """파일을 저장하고 file_id 를 반환한다. 비활성/실패 시 None.

    저장 경로는 ``{FILE_SERVICE_PREFIX}/{subdir}`` — 실제 오브젝트 키는 file-service 가
    ``{prefix}/{file_id}{ext}`` 로 만든다(파일명은 메타로만 보관).

    Args:
        data: 파일 바이트.
        filename: 원본 파일명 (메타데이터로 보관).
        content_type: MIME 타입.
        subdir: prefix 아래 하위 경로 (예: conversation_id).
        timeout: HTTP 타임아웃(초).
    """
    if not enabled() or not data:
        return None
    base = settings.file_service_base_url.rstrip("/")
    prefix = settings.file_service_prefix.strip("/")
    if subdir:
        prefix = f"{prefix}/{subdir.strip('/')}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{base}/v1/uploads",
                headers=_headers(),
                json={
                    "prefix": prefix,
                    "filename": filename or "file",
                    "content_type": content_type or "application/octet-stream",
                    "size": len(data),
                },
            )
            r.raise_for_status()
            up = r.json()

            put = await client.put(
                up["upload_url"], content=data, headers=up.get("headers") or {}
            )
            put.raise_for_status()

            done = await client.post(
                f"{base}/v1/uploads/{up['file_id']}/complete", headers=_headers()
            )
            done.raise_for_status()
            return up["file_id"]
    except Exception as e:  # noqa: BLE001 — 저장 실패가 채팅을 막으면 안 됨
        logger.warning("[file-service] 업로드 실패: %s (%s)", filename, e)
        return None


async def download(file_id: str, *, timeout: float = 60.0) -> Optional[bytes]:
    """저장된 파일의 **바이트**를 되받는다. 비활성·실패 시 None.

    presigned URL 을 한 번 발급해 그 자리에서 내려받는다 — URL 은 만료되므로 어디에도
    저장하지 않는다(migration_v28 주석과 같은 규칙).

    **왜 필요한가.** 만든 파일의 바이트는 Redis 대화 stash 에 1시간만 남는다. 그 뒤에
    "어제 만든 그 보고서 보내줘" 를 하려면 영구 저장소(여기)에서 되받아야 한다.

    Args:
        file_id: ``save_file`` 이 돌려준 식별자.
        timeout: HTTP 타임아웃(초). 본문 전송이라 넉넉히 준다.

    Returns:
        파일 바이트. 비활성·URL 미발급·다운로드 실패 시 None.
    """
    if not enabled() or not file_id:
        return None
    urls = await download_urls([file_id], timeout=timeout)
    url = urls.get(file_id)
    if not url:
        logger.warning("[file-service] 다운로드 URL 없음: file_id=%s", file_id)
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            # presigned URL 이라 인증 헤더를 붙이지 않는다(붙이면 서명과 충돌할 수 있다).
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except Exception as e:  # noqa: BLE001
        logger.warning("[file-service] 다운로드 실패 file_id=%s: %s", file_id, e)
        return None


async def download_urls(
    file_ids: List[str], *, timeout: float = 15.0
) -> Dict[str, str]:
    """file_id → presigned 다운로드 URL 매핑을 일괄 발급한다 (히스토리 표시용).

    비활성/실패 시 빈 dict. 접근 불가 항목은 매핑에서 빠진다(값 None 제외).
    """
    if not enabled() or not file_ids:
        return {}
    base = settings.file_service_base_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{base}/v1/files/download-urls",
                headers=_headers(),
                json={"file_ids": file_ids},
            )
            r.raise_for_status()
            return {
                f["file_id"]: f["download_url"]
                for f in r.json()["files"]
                if f.get("download_url")
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("[file-service] 다운로드 URL 일괄 발급 실패: %s", e)
        return {}
