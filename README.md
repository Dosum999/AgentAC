# agent-arch

업무용 LLM 에이전트의 **에이전트 아키텍처**와 **장기 메모리 아키텍처**를 정리한 레퍼런스 구현.
도구를 골라 실행하고, 근거로 답하고, 비가역 행동은 사람 승인 뒤에만 하고, 사용자에 대해 배운 것을 기억한다.
업무 도메인은 확장 슬롯으로 분리했고, 코어 구조와 그 설계 근거(실측·사고)를 함께 담았다.

## 핵심 개념

1. **Orchestrator** — 한 턴의 지휘자. 요청 컨텍스트를 세우고, 메시지를 조립하고, 도구를 계획하고, 결과를 저장한다.
2. **Router · Supervisor** — 요청을 도메인으로 나눈다. 결과는 실패 / 없음 / 지정의 세 값이고, 여러 도메인이면 워커를 병렬로 돌려 합친다.
3. **ReAct 그래프** — 도메인 워커 하나의 도구 루프. LangGraph 노드로 표현해 어느 지점이든 계측·개입할 수 있다.
4. **도구 노출과 관문** — 도구의 출처와 노출을 분리한다. 상시 노출은 도메인 관문(`find_<domain>_tools`)뿐이고, 관문이 찾은 도구는 같은 턴에 바인딩된다.
5. **권한 게이트** — deny-by-default. 노출됐다고 실행할 수 있는 것은 아니다.
6. **HITL 승인** — 비가역 도구는 항상 초안만 만든다. 그래프가 멈춰 체크포인트에 남기고, 사람이 승인하면 `/resume` 이 카드와 같은 인자로 한 번 실행한다.
7. **답변 가드** — 검증 노드, 출처 각주, 근거 있는 링크·이름만, 없는 완료 주장 교정, 답변 언어 미러링, 용어 사전 교정.
8. **LLM 계층** — 자체 `BaseLLM` 추상(Gemini·OpenAI·Anthropic·vLLM), 재시도, 스트림 반복 붕괴 차단, 외부 LLM 가명화(PII).
9. **장기 메모리** — 턴 원문 → 기억 단위(atom) → 장면 → 페르소나의 4층 배치 파이프라인. 턴 경로에서 LLM 을 부르지 않는다.
10. **확장 슬롯** — 업무 도메인이 코어에 끼어드는 모든 자리. 코어는 업무 도메인을 import 하지 않는다.

## 문서 읽는 순서

1. **[ARCHITECTURE.md](ARCHITECTURE.md)** — 전체 지도. 흐름, 진입점, 디렉터리별 역할과 불변식, 횡단 관심사.
2. **[examples/helpdesk.py](src/agent_arch/examples/helpdesk.py)** — 업무 도메인 하나를 코어 수정 없이 끼우는 본보기.
3. **[docs/rules/agent.md](docs/rules/agent.md)** — 에이전트 설계 규칙. 규칙마다 왜(실측 근거)와 어떻게(코드 위치).
4. **[docs/rules/memory.md](docs/rules/memory.md)** — 장기 메모리 설계 규칙.

## 시작하기

Python 3.12.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env            # LLM 키(GEMINI_API_KEY 등)와 DB·Redis·Qdrant 주소를 채운다
set -a && . ./.env && set +a
```

### 테스트

외부 인프라·LLM 키 없이 돈다. LLM 은 대본으로 대신한다.

```bash
.venv/bin/python -m pytest
```

### 서버

```bash
PYTHONPATH=src .venv/bin/uvicorn agent_arch.examples.helpdesk:create_example_app --factory --port 8000
```

- `STARTUP_CONNECT_EXTERNAL=false` 면 Redis·PostgreSQL·Qdrant·MCP 없이 뜬다(`/api/v1/health` 가 `degraded`). 대화를 실제로 돌리려면 LLM 키와 저장소가 필요하다.
- `AUTH_MODE=dev` 면 게이트웨이 인증 헤더 없이 `DEV_USER_*` 고정 사용자로 통과한다(로컬 개발 전용).

주요 엔드포인트(`/api/v1`):

- `POST /agent/run`, `POST /agent/stream`(SSE) — 한 턴 실행
- `POST /agent/{conversation_id}/resume` — 승인 재개. 본문 `pending_id` 가 지금 대기 항목과 다르면 409
- `POST /agent/runs/{run_id}/stop` — 실행 중단
- `GET /me/memory`, `DELETE /me/memory/atoms/{id}` — 기억 셀프서비스
- `GET /me/conversation`, `GET /me/conversation/search` — 대화 목록·검색

## 업무 도메인 끼우기

도구를 만들고, 도메인을 등록하고, 슬롯을 채운 `install` 을 앱 조립에 넘긴다. 코어는 고치지 않는다.

```python
from agent_arch.agents.tools.tool_search import register_domain
from agent_arch.main import create_app


def install(ext):
    register_domain(
        "helpdesk",
        gateway_tool="find_helpdesk_tools",               # 관문 — 상시 노출, 도메인 도구를 찾아 준다
        intent="헬프데스크 — 사내 IT 지원 티켓 조회·상세 확인·종료 처리",
        examples=["'열린 티켓 보여줘'", "'2번 티켓 해결됐으니 닫아줘'"],
        tool_prefix="helpdesk_",                           # 이 접두사의 도구는 이 도메인 소속
    )
    ext.domain_tools["helpdesk"] = lambda: [ListTickets(), GetTicket(), CloseTicket()]
    ext.approval_required.add("helpdesk_close_ticket")     # 비가역 — 항상 초안, 승인 뒤 실행
    ext.action_done_messages["helpdesk_close_ticket"] = "티켓을 종료했어요."
    ext.post_approval["helpdesk_close_ticket"] = notify_requester   # 승인 실행 뒤 후처리


app = create_app(setup=install)
```

비가역 도구는 `confirm` 인자를 받아 `false` 면 초안(draft view)만 돌려준다. 모델이 처음부터 `confirm=true` 로 불러도
코어가 초안으로 바꾸고, 실제 실행은 `/resume` 에서만 일어난다. 전체 코드는 [examples/helpdesk.py](src/agent_arch/examples/helpdesk.py),
끝까지 도는 테스트는 [tests/examples/test_helpdesk_example.py](tests/examples/test_helpdesk_example.py).

## 디렉터리

```
src/agent_arch/
├── main.py, core/, api/v1/, middleware/, queue/   HTTP·기동
├── extensions.py                                  업무 도메인 확장 슬롯
├── agents/
│   ├── orchestrator.py                            한 턴의 지휘자
│   ├── graphs/      router · supervisor · ReAct 그래프 · 체크포인터
│   ├── react/       실행기 · 예산 · 충분성 · 검증 · 출처 · 링크 정책
│   ├── tools/       노출 티어 · 관문 · MCP · 동적 API
│   ├── permissions/ 권한 게이트
│   ├── hitl/        승인 정책 · interrupt · 재개 · 답변 가드
│   └── prompts/     시스템 프롬프트 · 턴 컨텍스트
├── llm/             BaseLLM · 프로바이더 · 재시도 · 반복 붕괴 · PII
├── memory/          대화 이력 · tool_ctx · 압축
├── services/
│   ├── memory_v2/   장기 메모리 4층
│   └── web/ glossary usage/ monitoring/ api_registry/ langsmith/
└── examples/        헬프데스크 예시 도메인
config/              LLM 레인 · 도구 스위치 · 로깅 (yaml)
docs/rules/          설계 규칙 (agent.md, memory.md)
```

## 범위와 상태

- **범위 밖**: 업무 도메인 도구 자체(구매·물류·결재 등), 음성, 문서 파서, 가이드 지식베이스 수집. 첨부는 평문(txt/csv/md)·이미지·PDF 만 받는다.
- **알려진 미해결**: Gemini 런타임 503 → 다른 프로바이더 폴백 없음. 실험 단계에서 검증했지만 코드에 없는 설계(턴 락, 재라우팅, 원문 이벤트 기록 등)는 [agent.md 14절](docs/rules/agent.md)에 있다.
