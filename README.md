# agent-arch

업무용 LLM 에이전트의 **에이전트 아키텍처**와 **장기 메모리 아키텍처**를 규칙 중심으로 정리한 개인 레포.
실서비스 백엔드에서 떼어 와 업무 도메인 결합을 걷어 냈다.

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

## 원본에서 뺀 것

- **업무 도메인 도구와 서비스**: 구매·물류·결재·예약 등
- **데모 모드, 외부 위탁 추론 레인**
- **HTTP API 계층과 기동(lifespan)**: FastAPI 엔드포인트와 `/resume` 핸들러 포함
- **메모리 v1**

뺀 자리는 `extensions.py` 슬롯이 대신한다. 슬롯이 비어 있으면 코어는 그 기능 없이 결정적으로 동작한다.

## 실행

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env
set -a && . ./.env && set +a
.venv/bin/python -m pytest
```

설정 yaml(LLM 레인·도구 스위치·로깅)은 `config/` 에 있다.
