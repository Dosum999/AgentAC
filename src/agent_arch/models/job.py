from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class JobStatusResponse(BaseModel):
    """작업 상태 응답 모델.

    비동기 작업의 현재 상태와 결과 정보를 반환합니다.
    """

    job_id: str = Field(description="작업 고유 ID")
    job_type: str = Field(description="작업 유형 (예: 'agent_run')")
    status: str = Field(description="작업 상태 ('pending', 'running', 'completed', 'failed', 'cancelled')")
    created_at: str = Field(description="작업 생성 시간 (ISO 8601 형식)")
    updated_at: str = Field(description="마지막 업데이트 시간 (ISO 8601 형식)")
    input_data: Optional[Dict[str, Any]] = Field(None, description="작업 입력 데이터")
    result: Optional[Any] = Field(None, description="작업 실행 결과")
    error: Optional[str] = Field(None, description="오류 메시지")
    metadata: Optional[Dict[str, Any]] = Field(None, description="작업 메타데이터")
