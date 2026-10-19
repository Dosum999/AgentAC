import asyncio
import logging
from typing import Callable, Dict

from agent_arch.queue.enums import JobStatus
from agent_arch.queue.store import JobStore

logger = logging.getLogger(__name__)


class TaskRunner:
    """asyncio Task 생명주기 관리자.

    실행 중인 Task를 추적하고 취소 요청을 처리합니다.
    상태 변경은 JobStore에 위임합니다.

    Attributes:
        _tasks: 실행 중인 작업의 딕셔너리 {job_id: Task}
    """

    def __init__(self, store: JobStore) -> None:
        self._tasks: Dict[str, asyncio.Task] = {}
        self._store = store

    def run(self, job_id: str, coroutine: Callable, **kwargs) -> None:
        """코루틴을 백그라운드 Task로 등록하고 즉시 반환합니다.

        Args:
            job_id: 작업 ID
            coroutine: 실행할 코루틴 함수
            **kwargs: 코루틴에 전달할 키워드 인자
        """
        task = asyncio.create_task(self._execute(job_id, coroutine, **kwargs))
        self._tasks[job_id] = task

    async def _execute(self, job_id: str, coroutine: Callable, **kwargs) -> None:
        """작업 실행을 래핑하여 상태를 관리합니다.

        Args:
            job_id: 작업 ID
            coroutine: 실행할 코루틴 함수
            **kwargs: 코루틴에 전달할 키워드 인자
        """
        try:
            await self._store.update(job_id, JobStatus.RUNNING)
            result = await coroutine(**kwargs)
            await self._store.update(job_id, JobStatus.COMPLETED, result=result)

        except asyncio.CancelledError:
            await self._store.update(job_id, JobStatus.CANCELLED)
            raise

        except Exception as e:
            logger.exception("작업 실행 중 오류 발생: job_id=%s", job_id)
            await self._store.update(job_id, JobStatus.FAILED, error=str(e))

        finally:
            self._tasks.pop(job_id, None)

    def cancel(self, job_id: str) -> bool:
        """실행 중인 작업을 취소합니다.

        Args:
            job_id: 취소할 작업 ID

        Returns:
            취소 성공 여부 (실행 중이 아니면 False)
        """
        task = self._tasks.get(job_id)
        if task:
            task.cancel()
            return True
        return False

    def list_running(self) -> list[str]:
        """현재 실행 중인 작업 ID 목록을 반환합니다.

        Returns:
            실행 중인 job_id 리스트
        """
        return list(self._tasks.keys())
