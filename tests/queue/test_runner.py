"""TaskRunner 단위 테스트.

JobStore는 AsyncMock으로 대체하여 asyncio Task 생명주기만 검증한다.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from agent_arch.queue.enums import JobStatus
from agent_arch.queue.runner import TaskRunner


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.update = AsyncMock()
    return store


@pytest.fixture
def runner(mock_store):
    return TaskRunner(mock_store)


class TestTaskRunnerRun:
    """TaskRunner.run() 검증."""

    @pytest.mark.asyncio
    async def test_successful_coroutine_transitions_to_completed(self, runner, mock_store):
        """성공적으로 완료된 코루틴은 COMPLETED 상태로 전환된다."""
        async def success_fn(**kwargs):
            return {"result": "ok"}

        runner.run("job-1", success_fn)
        await asyncio.sleep(0)  # Task 스케줄 기회 부여

        # 모든 Task가 완료될 때까지 대기
        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        calls = [c.args[1] for c in mock_store.update.call_args_list]
        assert JobStatus.RUNNING in calls
        assert JobStatus.COMPLETED in calls

    @pytest.mark.asyncio
    async def test_failed_coroutine_transitions_to_failed(self, runner, mock_store):
        """예외를 발생시키는 코루틴은 FAILED 상태로 전환된다."""
        async def failing_fn(**kwargs):
            raise ValueError("intentional error")

        runner.run("job-2", failing_fn)
        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        calls = [c.args[1] for c in mock_store.update.call_args_list]
        assert JobStatus.FAILED in calls

    @pytest.mark.asyncio
    async def test_result_passed_to_store_update(self, runner, mock_store):
        """코루틴 반환값이 store.update의 result 인자로 전달된다."""
        async def fn(**kwargs):
            return "my_result"

        runner.run("job-3", fn)
        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        # COMPLETED update call에 result 포함 여부 확인
        completed_call = next(
            c for c in mock_store.update.call_args_list
            if c.args[1] == JobStatus.COMPLETED
        )
        assert completed_call.kwargs.get("result") == "my_result"

    @pytest.mark.asyncio
    async def test_error_message_passed_to_store_update(self, runner, mock_store):
        """예외 메시지가 store.update의 error 인자로 전달된다."""
        async def fn(**kwargs):
            raise RuntimeError("boom")

        runner.run("job-4", fn)
        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        failed_call = next(
            c for c in mock_store.update.call_args_list
            if c.args[1] == JobStatus.FAILED
        )
        assert failed_call.kwargs.get("error") == "boom"

    @pytest.mark.asyncio
    async def test_kwargs_forwarded_to_coroutine(self, runner, mock_store):
        """run()에 전달된 kwargs가 코루틴에 그대로 전달된다."""
        received = {}

        async def fn(**kwargs):
            received.update(kwargs)

        runner.run("job-5", fn, input="hello", session_id="s-1")
        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        assert received == {"input": "hello", "session_id": "s-1"}

    @pytest.mark.asyncio
    async def test_task_removed_from_tracking_after_completion(self, runner, mock_store):
        """완료 후 _tasks 딕셔너리에서 job_id가 제거된다."""
        async def fn(**kwargs):
            pass

        runner.run("job-6", fn)
        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        assert "job-6" not in runner._tasks


class TestTaskRunnerCancel:
    """TaskRunner.cancel() 검증."""

    @pytest.mark.asyncio
    async def test_cancel_running_job_returns_true(self, runner, mock_store):
        """실행 중인 작업 취소 시 True를 반환한다."""
        event = asyncio.Event()

        async def long_fn(**kwargs):
            await event.wait()

        runner.run("job-cancel", long_fn)
        await asyncio.sleep(0)  # Task 시작 기회 부여

        result = runner.cancel("job-cancel")
        event.set()

        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        assert result is True

    def test_cancel_nonexistent_job_returns_false(self, runner, mock_store):
        """존재하지 않는 job_id 취소 시 False를 반환한다."""
        result = runner.cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancelled_job_transitions_to_cancelled(self, runner, mock_store):
        """취소된 작업은 CANCELLED 상태로 전환된다."""
        event = asyncio.Event()

        async def long_fn(**kwargs):
            await event.wait()

        runner.run("job-c2", long_fn)
        await asyncio.sleep(0)

        runner.cancel("job-c2")
        event.set()

        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

        calls = [c.args[1] for c in mock_store.update.call_args_list]
        assert JobStatus.CANCELLED in calls


class TestTaskRunnerListRunning:
    """TaskRunner.list_running() 검증."""

    @pytest.mark.asyncio
    async def test_returns_running_job_ids(self, runner, mock_store):
        """실행 중인 작업 ID 목록을 반환한다."""
        event = asyncio.Event()

        async def long_fn(**kwargs):
            await event.wait()

        runner.run("job-list-1", long_fn)
        runner.run("job-list-2", long_fn)
        await asyncio.sleep(0)

        running = runner.list_running()
        assert "job-list-1" in running
        assert "job-list-2" in running

        event.set()
        await asyncio.gather(*asyncio.all_tasks() - {asyncio.current_task()}, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_running_jobs(self, runner, mock_store):
        """실행 중인 작업이 없으면 빈 리스트를 반환한다."""
        assert runner.list_running() == []
