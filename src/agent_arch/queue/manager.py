from typing import Any, Callable, Dict, Optional

from agent_arch.queue.store import JobStore
from agent_arch.queue.runner import TaskRunner


class JobManager:
    """작업 큐 파사드.

    JobStore(Redis 상태)와 TaskRunner(asyncio Task)를 조합하여
    단일 진입점(enqueue)을 제공합니다.
    """

    def __init__(self) -> None:
        self._store = JobStore()
        self._runner = TaskRunner(self._store)

    async def enqueue(
        self,
        job_type: str,
        coroutine: Callable,
        input_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """작업을 생성하고 백그라운드에서 즉시 실행합니다.

        Args:
            job_type: 작업 유형 (예: "agent_run", "scheduled_run")
            coroutine: 실행할 코루틴 함수
            input_data: Redis에 저장할 작업 입력 데이터
            metadata: 추가 메타데이터 (선택)
            **kwargs: coroutine에 전달할 키워드 인자

        Returns:
            생성된 job_id
        """
        job_id = await self._store.create(job_type, input_data, metadata)
        self._runner.run(job_id, coroutine, **kwargs)
        return job_id

    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태를 조회합니다.

        Args:
            job_id: 조회할 작업 ID

        Returns:
            작업 정보 딕셔너리, 없으면 None
        """
        return await self._store.get(job_id)

    async def cancel_job(self, job_id: str) -> bool:
        """실행 중인 작업을 취소합니다.

        Args:
            job_id: 취소할 작업 ID

        Returns:
            취소 성공 여부
        """
        return self._runner.cancel(job_id)

    def get_running_jobs(self) -> list[str]:
        """현재 실행 중인 작업 ID 목록을 반환합니다.

        Returns:
            실행 중인 job_id 리스트
        """
        return self._runner.list_running()


# 싱글톤 인스턴스
job_manager = JobManager()
