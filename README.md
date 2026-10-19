# agent-arch

업무용 LLM 에이전트의 **에이전트 아키텍처**와 **장기 메모리 아키텍처**를 규칙 중심으로 정리한 개인 레포.
업무 도메인은 확장 슬롯으로 분리하고, 코어 구조와 그 설계 근거만 담았다.

## 먼저 읽을 것

- [docs/rules/README.md](docs/rules/README.md): 규칙 인덱스
- [docs/rules/agent.md](docs/rules/agent.md): 라우팅·감독, 도구 노출·실행, 권한, MCP, HITL, 답변 가드, LLM 계층, PII, 알려진 결함
- [docs/rules/memory.md](docs/rules/memory.md): 장기 메모리 4층 배치 파이프라인

## 코드 지도 (`src/agent_arch/`)

- `agents/orchestrator.py`: 조립부. 모드 결정, 컨텍스트 세팅, 도구 계획, 실행 전략 선택, 저장
- `agents/graphs/`: LangGraph 기반 구성
  - `react_graph`: ReAct StateGraph
  - `router`: 도메인 라우터
  - `supervisor`: 워커 병렬 실행과 합성
  - `checkpointer`: 체크포인터 팩토리
- `agents/react/`: 루프 부품
  - 실행기와 예산: `executor`, `budget`, `evidence_budget`, `sufficiency`
  - 품질: `verify`, `provenance`, `link_policy`
  - 이력: `history`
- `agents/hitl/`: 비가역 액션 승인
  - `policy`, `graph`(interrupt), `resume`, `answer_guard`, `followups`
- `agents/tools/`: 도구 계층
  - 노출 티어와 관문: `exposure`, `tool_search`
  - MCP 원격 도구: `mcp/`
  - 기타: 동적 API 도구, 메모리 도구
- `agents/permissions/`: 도구 권한 게이트(deny-by-default)
- `llm/`: 자체 `BaseLLM` 추상과 프로바이더
  - 프로바이더: Gemini, OpenAI, Anthropic, vLLM
  - `transient`: 재시도
  - `pii/`: 외부 LLM 가명화
- `memory/`: 대화 이력, tool_ctx, 압축, Redis 저장소
- `services/memory_v2/`: 장기 메모리 파이프라인
- `services/`: 지원 서비스
  - `web/`: 웹 검색 가드와 출력 정규화
  - `glossary`: 용어 사전 교정
  - `usage/`: 벤더 경계 비용 계측
  - `monitoring/`: 오류 이벤트
  - `api_registry/`: 도구 색인
- `extensions.py`: 도메인 확장 슬롯. 코어는 업무 도메인을 import 하지 않는다.
- `main.py`·`core/lifespan.py`·`api/v1/`·`middleware/`·`queue/`: HTTP·기동 계층. `create_app(setup=)` 이 조립부다.
- `examples/helpdesk.py`: 예시 업무 도메인(코어 수정 없이 끼우기).

## 범위 밖

- **업무 도메인 도구와 서비스**: 구매·물류·결재·예약 등(헬프데스크 예시가 끼우는 방법을 보여 준다)
- **음성·문서 파서·가이드 지식베이스 수집**: 첨부는 평문(txt/csv/md)·이미지·PDF 만 받는다

업무 도메인은 `extensions.py` 슬롯으로 붙인다. 슬롯이 비어 있으면 코어는 그 기능 없이 결정적으로 동작한다.

## 업무 도메인 끼우기

`src/agent_arch/examples/helpdesk.py` 를 본보기로 삼는다. 도구를 만들고, `register_domain` 과 `extensions` 슬롯을 채우는
`install()` 을 `create_app(setup=install)` 에 넘기면 된다. 코어는 고치지 않는다. 자세한 순서는 [agent.md 1절](docs/rules/agent.md).

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # LLM 키(GEMINI_API_KEY 등)와 DB·Redis·Qdrant 주소를 채운다
set -a && . ./.env && set +a

# 테스트 (외부 인프라 없이 돈다)
.venv/bin/python -m pytest

# 서버 — 헬프데스크 예시 도메인을 끼운 앱
PYTHONPATH=src .venv/bin/uvicorn agent_arch.examples.helpdesk:create_example_app --factory --port 8000
```

- `STARTUP_CONNECT_EXTERNAL=false`: Redis·PostgreSQL·Qdrant·MCP 없이 뜬다(`/api/v1/health` 가 `degraded`). 대화를 실제로 돌리려면 LLM 키와 저장소가 필요하다.
- `AUTH_MODE=dev`: 게이트웨이 인증 헤더 없이 `DEV_USER_*` 고정 사용자로 통과한다(로컬 개발 전용).
- 주요 엔드포인트(`/api/v1`): `POST /agent/run`, `POST /agent/stream`(SSE), `POST /agent/{conversation_id}/resume`(본문 `pending_id` 필수 — 카드와 대기 항목이 다르면 409), `GET/DELETE /me/memory`, 대화 목록·검색.

설정 yaml(LLM 레인·도구 스위치·로깅)은 `config/` 에 있다.
