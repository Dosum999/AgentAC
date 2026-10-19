import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from agent_arch.memory.redis_store import redis_store
from agent_arch.queue.enums import JobStatus

# 작업 상태를 Redis에 보관하는 TTL (초)
_JOB_TTL = 3600


class JobStore:
    """Redis 기반 작업 상태 저장소.

    작업 데이터의 생성·조회·갱신을 담당합니다.
    """

    async def create(
        self,
        job_type: str,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """새 작업을 Redis에 저장하고 job_id를 반환합니다.

        Args:
            job_type: 작업 유형 (예: "agent_run")
            input_data: 작업 실행에 필요한 입력 데이터
            metadata: 추가 메타데이터 (선택)

        Returns:
            생성된 고유 작업 ID
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        job_data: Dict[str, Any] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": JobStatus.PENDING,
            "input_data": input_data,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }

        await redis_store.enqueue_job(job_id, job_data)
        return job_id

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """작업 데이터를 조회합니다.

        Args:
            job_id: 조회할 작업 ID

        Returns:
            작업 정보 딕셔너리, 없으면 None
        """
        return await redis_store.get_job(job_id)

    async def update(
        self,
        job_id: str,
        status: JobStatus,
        result: Any = None,
        error: Optional[str] = None,
    ) -> None:
        """작업 상태를 갱신합니다.

        Args:
            job_id: 갱신할 작업 ID
            status: 새로운 작업 상태
            result: 작업 결과 (선택)
            error: 오류 메시지 (선택)
        """
        job = await redis_store.get_job(job_id)
        if not job:
            return

        job["status"] = status
        job["updated_at"] = datetime.now(timezone.utc).isoformat()

        if result is not None:
            job["result"] = result
        if error is not None:
            job["error"] = error

        await redis_store.set(f"job:{job_id}", job, expire=_JOB_TTL)
