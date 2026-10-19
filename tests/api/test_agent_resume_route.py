"""resume 엔드포인트 등록 가드 — 데코레이터 바로 아래 함수가 등록된다. 헬퍼를 그 사이에 끼우면 라우트가 헬퍼를
가리켜 승인/거절이 200·null 로 조용히 무시된다(2026-09-28 로컬 실측)."""
from agent_arch.api.v1.agent import router


def test_resume_route_points_at_resume_handler():
    routes = [r for r in router.routes if getattr(r, "path", "").endswith("/{conversation_id}/resume")]
    assert routes, "resume 라우트가 없다"
    assert all(r.endpoint.__name__ == "resume_agent_action" for r in routes), [r.endpoint.__name__ for r in routes]
    assert "POST" in routes[0].methods
