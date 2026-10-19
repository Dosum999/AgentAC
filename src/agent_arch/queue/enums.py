from enum import Enum


class JobStatus(str, Enum):
    """비동기 작업의 수명 주기 상태.

    Attributes:
        PENDING: 대기 중
        RUNNING: 실행 중
        COMPLETED: 완료됨
        FAILED: 실패함
        CANCELLED: 취소됨
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
