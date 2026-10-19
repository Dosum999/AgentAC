from fastapi import APIRouter, HTTPException
from agent_arch.models.job import JobStatusResponse
from agent_arch.queue.manager import job_manager

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """작업 상태를 조회한다.

    Args:
        job_id (str): 조회할 작업 ID

    Returns:
        JobStatusResponse: 작업 상태 정보

    Raises:
        HTTPException: 작업을 찾을 수 없을 때 404 반환
    """
    job = await job_manager.get_job_status(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatusResponse(**job)


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """작업을 취소한다.

    Args:
        job_id (str): 취소할 작업 ID

    Returns:
        dict: 취소 성공 여부 및 메시지

    Raises:
        HTTPException: 작업을 찾을 수 없거나 이미 완료된 경우 404 반환
    """
    success = await job_manager.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or already completed"
        )

    return {"success": True, "message": f"Job {job_id} cancelled"}


@router.get("/")
async def list_running_jobs() -> dict:
    """실행 중인 작업 리스트를 반환한다.

    Returns:
        dict: 실행 중인 작업 목록 및 개수
    """
    running_jobs = job_manager.get_running_jobs()

    return {
        "running_jobs": running_jobs,
        "count": len(running_jobs)
    }
