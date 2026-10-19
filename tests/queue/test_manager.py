"""JobManager 단위 테스트.

JobStore·TaskRunner를 Mock으로 대체하여 JobManager 파사드 위임 동작을 검증한다.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent_arch.queue.manager import JobManager


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.create = AsyncMock(return_value="job-abc")
    store.get = AsyncMock(return_value={"job_id": "job-abc", "status": "pending"})
    return store


@pytest.fixture
def mock_runner():
    runner = MagicMock()
    runner.run = MagicMock()
    runner.cancel = MagicMock(return_value=True)
    runner.list_running = MagicMock(return_value=["job-abc"])
    return runner


@pytest.fixture
def manager(mock_store, mock_runner):
    """JobStore·TaskRunner를 주입한 JobManager 인스턴스."""
    mgr = JobManager.__new__(JobManager)
    mgr._store = mock_store
    mgr._runner = mock_runner
    return mgr


class TestJobManagerEnqueue:
    """JobManager.enqueue() 검증."""

    @pytest.mark.asyncio
    async def test_returns_job_id(self, manager, mock_store, mock_runner):
        """enqueue()는 store.create()가 반환한 job_id를 그대로 반환한다."""
        async def dummy(**kwargs):
            pass

        job_id = await manager.enqueue("agent_run", dummy, {"input": "hi"})

        assert job_id == "job-abc"

    @pytest.mark.asyncio
    async def test_calls_store_create(self, manager, mock_store, mock_runner):
        """store.create()를 올바른 인자로 호출한다."""
        async def dummy(**kwargs):
            pass

        input_data = {"input": "hi"}
        metadata = {"user": "u-1"}

        await manager.enqueue("agent_run", dummy, input_data, metadata)

        mock_store.create.assert_called_once_with("agent_run", input_data, metadata)

    @pytest.mark.asyncio
    async def test_calls_runner_run_with_kwargs(self, manager, mock_store, mock_runner):
        """runner.run()을 job_id·coroutine·kwargs와 함께 호출한다."""
        async def dummy(**kwargs):
            pass

        await manager.enqueue("agent_run", dummy, {}, input="hello", session_id="s-1")

        mock_runner.run.assert_called_once_with("job-abc", dummy, input="hello", session_id="s-1")


class TestJobManagerGetJobStatus:
    """JobManager.get_job_status() 검증."""

    @pytest.mark.asyncio
    async def test_delegates_to_store_get(self, manager, mock_store):
        """store.get()에 job_id를 그대로 전달하고 결과를 반환한다."""
        result = await manager.get_job_status("job-abc")

        mock_store.get.assert_called_once_with("job-abc")
        assert result == {"job_id": "job-abc", "status": "pending"}

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, manager, mock_store):
        """job이 없으면 None을 반환한다."""
        mock_store.get = AsyncMock(return_value=None)

        result = await manager.get_job_status("nonexistent")

        assert result is None


class TestJobManagerCancelJob:
    """JobManager.cancel_job() 검증."""

    @pytest.mark.asyncio
    async def test_returns_true_when_cancelled(self, manager, mock_runner):
        """runner.cancel()이 True를 반환하면 그대로 True를 반환한다."""
        result = await manager.cancel_job("job-abc")

        mock_runner.cancel.assert_called_once_with("job-abc")
        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_not_found(self, manager, mock_runner):
        """runner.cancel()이 False를 반환하면 그대로 False를 반환한다."""
        mock_runner.cancel = MagicMock(return_value=False)

        result = await manager.cancel_job("nonexistent")

        assert result is False


class TestJobManagerGetRunningJobs:
    """JobManager.get_running_jobs() 검증."""

    def test_delegates_to_runner_list_running(self, manager, mock_runner):
        """runner.list_running()에 위임하고 결과를 반환한다."""
        result = manager.get_running_jobs()

        mock_runner.list_running.assert_called_once()
        assert result == ["job-abc"]

    def test_returns_empty_list_when_none_running(self, manager, mock_runner):
        """실행 중인 작업이 없으면 빈 리스트를 반환한다."""
        mock_runner.list_running = MagicMock(return_value=[])

        result = manager.get_running_jobs()

        assert result == []
