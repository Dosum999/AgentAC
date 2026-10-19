"""JobStore 단위 테스트.

Redis 의존성은 redis_store 모듈 수준 싱글톤을 AsyncMock으로 패치하여 격리한다.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from agent_arch.queue.enums import JobStatus
from agent_arch.queue.store import JobStore


@pytest.fixture
def store():
    return JobStore()


@pytest.fixture
def mock_redis():
    """redis_store의 메서드를 AsyncMock으로 대체하는 픽스처."""
    with patch("agent_arch.queue.store.redis_store") as m:
        m.enqueue_job = AsyncMock(return_value=True)
        m.get_job = AsyncMock(return_value=None)
        m.set = AsyncMock(return_value=True)
        yield m


class TestJobStoreCreate:
    """JobStore.create() 검증."""

    @pytest.mark.asyncio
    async def test_returns_uuid_string(self, store, mock_redis):
        """create()는 UUID 형식의 job_id를 반환한다."""
        import re
        job_id = await store.create("agent_run", {"input": "test"})

        assert isinstance(job_id, str)
        assert re.fullmatch(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            job_id,
        )

    @pytest.mark.asyncio
    async def test_initial_status_is_pending(self, store, mock_redis):
        """생성된 작업의 초기 상태는 PENDING이다."""
        await store.create("agent_run", {"input": "test"})

        call_args = mock_redis.enqueue_job.call_args
        job_data = call_args[0][1]  # 두 번째 위치 인자 = job_data
        assert job_data["status"] == JobStatus.PENDING

    @pytest.mark.asyncio
    async def test_job_data_fields(self, store, mock_redis):
        """생성된 job_data에 필수 필드가 모두 포함된다."""
        input_data = {"input": "hello"}
        metadata = {"user_id": "u-1"}

        await store.create("agent_run", input_data, metadata)

        job_data = mock_redis.enqueue_job.call_args[0][1]
        assert job_data["job_type"] == "agent_run"
        assert job_data["input_data"] == input_data
        assert job_data["metadata"] == metadata
        assert job_data["result"] is None
        assert job_data["error"] is None
        assert "created_at" in job_data
        assert "updated_at" in job_data

    @pytest.mark.asyncio
    async def test_metadata_defaults_to_empty_dict(self, store, mock_redis):
        """metadata를 전달하지 않으면 빈 딕셔너리로 저장된다."""
        await store.create("agent_run", {})

        job_data = mock_redis.enqueue_job.call_args[0][1]
        assert job_data["metadata"] == {}

    @pytest.mark.asyncio
    async def test_calls_enqueue_job_with_job_id(self, store, mock_redis):
        """enqueue_job은 반환된 job_id와 동일한 ID로 호출된다."""
        job_id = await store.create("agent_run", {})

        mock_redis.enqueue_job.assert_called_once()
        called_id = mock_redis.enqueue_job.call_args[0][0]
        assert called_id == job_id


class TestJobStoreGet:
    """JobStore.get() 검증."""

    @pytest.mark.asyncio
    async def test_returns_job_when_found(self, store, mock_redis):
        """존재하는 job_id 조회 시 딕셔너리를 반환한다."""
        expected = {"job_id": "abc", "status": "pending"}
        mock_redis.get_job = AsyncMock(return_value=expected)

        result = await store.get("abc")

        assert result == expected

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self, store, mock_redis):
        """존재하지 않는 job_id 조회 시 None을 반환한다."""
        mock_redis.get_job = AsyncMock(return_value=None)

        result = await store.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_delegates_to_redis_get_job(self, store, mock_redis):
        """redis_store.get_job에 job_id를 그대로 전달한다."""
        mock_redis.get_job = AsyncMock(return_value=None)

        await store.get("job-123")

        mock_redis.get_job.assert_called_once_with("job-123")


class TestJobStoreUpdate:
    """JobStore.update() 검증."""

    @pytest.mark.asyncio
    async def test_updates_status(self, store, mock_redis):
        """상태 업데이트 시 status 필드가 변경된다."""
        existing = {"job_id": "j1", "status": "pending", "result": None, "error": None}
        mock_redis.get_job = AsyncMock(return_value=existing)

        await store.update("j1", JobStatus.RUNNING)

        saved = mock_redis.set.call_args[0][1]
        assert saved["status"] == JobStatus.RUNNING

    @pytest.mark.asyncio
    async def test_updates_result_when_provided(self, store, mock_redis):
        """result 인자가 있을 때 result 필드를 업데이트한다."""
        existing = {"job_id": "j1", "status": "running", "result": None, "error": None}
        mock_redis.get_job = AsyncMock(return_value=existing)

        await store.update("j1", JobStatus.COMPLETED, result={"output": "ok"})

        saved = mock_redis.set.call_args[0][1]
        assert saved["result"] == {"output": "ok"}

    @pytest.mark.asyncio
    async def test_updates_error_when_provided(self, store, mock_redis):
        """error 인자가 있을 때 error 필드를 업데이트한다."""
        existing = {"job_id": "j1", "status": "running", "result": None, "error": None}
        mock_redis.get_job = AsyncMock(return_value=existing)

        await store.update("j1", JobStatus.FAILED, error="timeout")

        saved = mock_redis.set.call_args[0][1]
        assert saved["error"] == "timeout"

    @pytest.mark.asyncio
    async def test_sets_ttl_on_update(self, store, mock_redis):
        """update() 호출 시 TTL(expire)이 설정된다."""
        existing = {"job_id": "j1", "status": "running", "result": None, "error": None}
        mock_redis.get_job = AsyncMock(return_value=existing)

        await store.update("j1", JobStatus.COMPLETED)

        call_kwargs = mock_redis.set.call_args[1]
        assert "expire" in call_kwargs
        assert call_kwargs["expire"] > 0

    @pytest.mark.asyncio
    async def test_noop_when_job_not_found(self, store, mock_redis):
        """존재하지 않는 job_id 업데이트 시 redis_store.set을 호출하지 않는다."""
        mock_redis.get_job = AsyncMock(return_value=None)

        await store.update("nonexistent", JobStatus.FAILED)

        mock_redis.set.assert_not_called()
