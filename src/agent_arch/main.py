"""FastAPI 앱 조립부(composition root).

``create_app(setup)`` 이 앱을 만든다. ``setup`` 은 기동 전에 ``agent_arch.extensions.extensions`` 를 받아
업무 도메인을 꽂는 훅이다 — 코어는 도메인을 import 하지 않고, 도메인 쪽이 슬롯(도메인 도구·승인 정책·
후처리 훅·전용 채널 모드 프롬프트 …)을 채운다. 모듈 수준 ``app`` 은 슬롯이 빈 코어만의 앱이다.

    from agent_arch.main import create_app

    def setup(ext):
        ext.domain_tools["inventory"] = lambda: [InventoryLookupTool()]

    app = create_app(setup)
"""
import logging
import logging.config
from typing import Callable, Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from agent_arch.api.v1 import agent, conversation, health, jobs, memory, preferences, user
from agent_arch.core.config import settings
from agent_arch.core.lifespan import lifespan
from agent_arch.extensions import Extensions, extensions
from agent_arch.middleware.auth import get_current_user

# 로깅 설정 적용
if settings.logging_config:
    logging.config.dictConfig(settings.logging_config)

# LLM provider 등록(import 로 데코레이터 실행)
from agent_arch.llm import anthropic_provider  # noqa: E402,F401
from agent_arch.llm import gemini  # noqa: E402,F401
from agent_arch.llm import openai_provider  # noqa: E402,F401
from agent_arch.llm import vllm  # noqa: E402,F401

logger = logging.getLogger(__name__)


def create_app(setup: Optional[Callable[[Extensions], None]] = None) -> FastAPI:
    """앱을 만든다.

    Args:
        setup: 기동 전에 확장 슬롯을 채우는 조립 훅. 앱을 만들 때 한 번 호출된다(요청마다가 아니다).
            None 이면 슬롯이 빈 코어만의 앱 — 업무 도메인 없이 대화·메모리·웹(설정 시)·MCP(설정 시)로 돈다.

    Returns:
        라우터·미들웨어·기동 훅이 붙은 FastAPI 앱.
    """
    if setup is not None:
        setup(extensions)

    # API_DOCS_ENABLED=false 면 문서 경로 자체를 만들지 않는다(없는 경로와 같은 404).
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="LLM agent backend — routing·ReAct graph·HITL·long-term memory",
        lifespan=lifespan,
        docs_url="/docs" if settings.api_docs_enabled else None,
        redoc_url="/redoc" if settings.api_docs_enabled else None,
        openapi_url="/openapi.json" if settings.api_docs_enabled else None,
    )

    cors_origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # 4xx/5xx 응답을 라우트 템플릿으로 오류 이벤트에 기록한다. 순수 ASGI 라 SSE 본문을 감싸지 않는다.
    # 마지막에 추가한 미들웨어가 가장 바깥이므로 CORS 거절 응답도 본다.
    from agent_arch.services.monitoring import ErrorEventsMiddleware

    app.add_middleware(ErrorEventsMiddleware)

    prefix = settings.api_v1_prefix
    authed = [Depends(get_current_user)]
    app.include_router(health.router, prefix=prefix)  # 인증 없음
    app.include_router(agent.router, prefix=prefix, dependencies=authed)
    app.include_router(jobs.router, prefix=prefix, dependencies=authed)
    # 내 장기 기억 셀프서비스(조회·삭제·초기화) — 메모리 v2 가 켜진 환경에서만 동작(아니면 409)
    app.include_router(memory.router, prefix=prefix, dependencies=authed)
    # 개인 설정(대화 이력 정책 — 본인 단위)
    app.include_router(preferences.router, prefix=prefix, dependencies=authed)
    # 대화방 관리(본인 대화방 목록·메시지·첨부)
    app.include_router(conversation.router, prefix=prefix, dependencies=authed)
    # 사용자 프로필 등록. 핸들러가 X-User-Sub 를 직접 검증한다 — get_current_user 를 걸면 미등록 사용자가
    # 401 에 막혀 등록 자체가 불가능해지는 닭/달걀 문제가 생긴다.
    app.include_router(user.router, prefix=prefix)

    @app.get("/")
    async def root():
        """애플리케이션 기본 정보."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs" if settings.api_docs_enabled else None,
            "health": f"{prefix}/health",
        }

    if settings.api_docs_enabled:
        async def swagger_redirect(format: str = None):
            """format=openapi 면 /openapi.json, 그 밖은 /docs 로."""
            return RedirectResponse(url="/openapi.json" if format == "openapi" else "/docs")

        app.add_api_route("/swagger/", swagger_redirect, methods=["GET"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent_arch.main:app", host=settings.host, port=settings.port, reload=settings.debug)
