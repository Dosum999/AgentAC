"""에이전트 HTTP 진입점 — 동기 실행·비동기 작업·SSE 스트림·실행 중지·HITL 재개·답변 평가.

업무 도메인(전용 채널 봇·음성·문서 파서 등)은 여기 없다. 전용 채널 모드는 ``agent_mode`` 로 받고,
허용 여부는 orchestrator 가 ``extensions.mode_prompts`` 등록 여부로 판정한다(미등록 모드는 400).
"""
import base64
import json
import logging
import re
from typing import AsyncIterator, Any, Dict, List, Literal, Optional, Tuple

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

from agent_arch.agents.cancellation import set_cancel_event
from agent_arch.agents.orchestrator import AgentOrchestrator
from agent_arch.agents.run_registry import register_run, request_stop, unregister_run
from agent_arch.agents.source_routing import apply_source_defaults
from agent_arch.core.config import settings
from agent_arch.db.repository import (
    create_conversation,
    derive_conversation_title,
    get_conversation,
    get_message_for_feedback,
    update_message_rating,
)
from agent_arch.models.agent import (
    PROGRESS_EVENT_TYPES,
    AgentResponse,
    AgentStreamChunk,
    ImageInput,
    RatingRequest,
    _MAX_FILES_PER_REQUEST,
    _SUPPORTED_MIME_TYPES,
)
from agent_arch.queue.manager import job_manager
from agent_arch.services import file_service
from agent_arch.services.langsmith.feedback import send_feedback
from agent_arch.services.llm_settings import resolve_effective_provider
from agent_arch.services.monitoring.error_events import event_extra
from agent_arch.utils.mime import get_mime_from_filename
from agent_arch.utils.request import build_user_info
from agent_arch.utils.sanitize import sanitize_tool_names

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])

# 평문 첨부 — 서버가 UTF-8 로 풀어 질문 앞에 붙인다. 이미지/PDF 는 멀티모달로 바이트째 간다.
# 문서(docx·hwp 등)·스프레드시트(xlsx) 파서는 범위 밖이다.
_TEXT_MIME_TYPES = {"text/plain", "text/csv", "text/markdown"}
_TEXT_EXTENSIONS = {".txt": "text/plain", ".csv": "text/csv", ".md": "text/markdown"}
_MAX_TEXT_ATTACHMENT_CHARS = 30_000

# 텍스트 없이 첨부만 보낸 턴에 채워 넣는 기본 지시.
_ATTACHMENT_ONLY_INPUT = "첨부한 파일을 분석해 주세요."


async def _apply_web_search(orchestrator: AgentOrchestrator, requested: Optional[bool], user_info: dict) -> bool:
    """웹 검색 opt-in — 요청 토글(web_search=true) ∧ 테넌트 설정일 때만 이 요청에 웹 도메인을 연다.

    웹 관문(find_web_tools)은 도구 설정에서 전역 OFF(fail-closed)다. 여기서만 static_tools 에 얹어 웹 도구가
    발견되게 한다. 라우터 카드·후속 상속·sticky 복원·도구 실행은 모두 요청 컨텍스트(web_search_enabled)를 보므로,
    꺼진 요청에서는 웹이 어디서도 광고·발견·실행되지 않는다.
    """
    from agent_arch.agents.context import set_web_search_enabled
    from agent_arch.agents.tools.exposure import with_domain_gateway
    from agent_arch.services.web.guard import resolve_web_search

    effective = await resolve_web_search(requested, user_info.get("company"))
    orchestrator.web_search_enabled = effective
    set_web_search_enabled(effective)
    if effective:
        orchestrator.static_tools = with_domain_gateway(orchestrator.static_tools, "find_web_tools")
    # 받은 값을 항상 남긴다 — "토글이 안 실려 온 요청"과 "회사 설정이 막은 요청"을 로그로 구분하려고.
    logger.info(
        "[web] opt-in requested=%r(%s) effective=%s company=%s",
        requested, type(requested).__name__, effective, user_info.get("company"),
    )
    return effective


async def _resolve_conversation_id(conversation_id: Optional[str], user_sub: str) -> Optional[str]:
    """진입 시점에 conversation_id 를 정리한다.

    - 본인 소유 유효 id → 그대로
    - 타인 소유 id → 404
    - 미전송 → None (동기/SSE 는 답 저장 시점에 lazy-create)
    """
    # 라우터를 직접 호출하는 테스트에서는 Form(None) 객체가 그대로 들어온다 — 문자열만 본다.
    if conversation_id and isinstance(conversation_id, str):
        if await get_conversation(conversation_id, user_sub) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        return conversation_id
    return None


def _has_attachments(files: Optional[List[UploadFile]]) -> bool:
    """폼에 실제 첨부가 실려 왔는지(빈 멀티파트 파트를 거른 기준)."""
    return bool(files) and any(f.filename or f.size for f in files)


def _compose_input(input: str, attachment_text: Optional[str], image_inputs: Optional[List[ImageInput]]) -> str:
    """첨부 평문을 질문 앞에 붙여 모델에 넘길 최종 입력을 만든다. 질문이 비면 기본 지시로 채운다."""
    text = input
    if not input.strip() and (attachment_text or image_inputs):
        text = _ATTACHMENT_ONLY_INPUT
    return (attachment_text + "\n\n" + text) if attachment_text else text


def _mark_question(orchestrator: AgentOrchestrator, input: str, augmented: str) -> None:
    """첨부 평문이 입력 앞에 붙은 턴이면 사용자가 직접 쓴 질문을 따로 넘긴다.

    모델에는 첨부까지 붙은 원문을 그대로 주고, 제목·라우터 이력·압축 질문 목록·장기 메모리는 이 값을 쓴다
    (agents/context.set_turn_question). 안 그러면 첨부 본문이 사용자 발화로 읽힌다.
    """
    if augmented != input and input.strip():
        orchestrator.turn_question = input.strip()


async def _process_files(
    files: Optional[List[UploadFile]],
) -> Tuple[Optional[List[ImageInput]], Optional[str]]:
    """첨부를 이미지/PDF(멀티모달)와 평문(질문 앞에 붙임)으로 나눈다. 지원하지 않는 형식은 400."""
    if not files:
        return None, None
    valid = [f for f in files if f.filename or f.size]
    if not valid:
        return None, None

    visual: List[Tuple[UploadFile, str]] = []
    texts: List[str] = []
    for f in valid:
        filename = f.filename or ""
        mime = f.content_type or ""
        # 브라우저 MIME 라벨은 주장일 뿐이다 — OS 에 타입이 없으면 octet-stream 으로 온다. 확장자로 보완한다.
        if mime not in _SUPPORTED_MIME_TYPES and mime not in _TEXT_MIME_TYPES:
            ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
            mime = get_mime_from_filename(filename) or _TEXT_EXTENSIONS.get(ext) or mime
        if mime in _SUPPORTED_MIME_TYPES:
            visual.append((f, mime))
        elif mime in _TEXT_MIME_TYPES:
            raw = await f.read()
            body = raw.decode("utf-8", errors="replace")[:_MAX_TEXT_ATTACHMENT_CHARS]
            texts.append(f"[첨부: {filename or '텍스트'}]\n{body}")
        else:
            raise HTTPException(
                status_code=400,
                detail=f"지원하지 않는 파일 형식입니다: {filename or mime}. "
                       f"지원 형식: 이미지(jpg/png/webp/heic/heif), PDF, 텍스트(txt/csv/md)",
            )
    return await _build_image_inputs(visual), ("\n\n".join(texts) or None)


async def _build_image_inputs(files: List[Tuple[UploadFile, str]]) -> Optional[List[ImageInput]]:
    """이미지/PDF 를 ImageInput 으로. 개수 초과·크기 검증 실패는 400."""
    if not files:
        return None
    if len(files) > _MAX_FILES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"이미지/PDF는 최대 {_MAX_FILES_PER_REQUEST}개까지 첨부할 수 있습니다.",
        )
    result = []
    for f, mime in files:
        raw = await f.read()
        try:
            result.append(ImageInput(data=base64.b64encode(raw).decode(), mime_type=mime, file_name=f.filename))
        except ValidationError as e:
            # 그대로 올리면 FastAPI 가 500 으로 내보내 사용자는 "서버 오류"만 본다.
            reason = (e.errors()[0].get("msg", "") if e.errors() else "").replace("Value error, ", "")
            raise HTTPException(status_code=400, detail=f"{f.filename or mime}: {reason or '첨부를 처리할 수 없습니다.'}")
    return result or None


async def _ensure_conversation_scope(
    conv_id: Optional[str], files: Optional[List[UploadFile]], user_sub: str, input: str,
) -> Optional[str]:
    """첨부가 있는 새 대화는 접수 시점에 conversation_id 를 확정한다.

    새 대화는 답 저장 시점 lazy-create 가 기본이지만(중단된 턴이 빈 대화방을 남기지 않게), 턴 **도중에**
    대화 스코프를 쓰는 흐름(첨부 저장 경로·대화 스코프 stash)은 그때 id 가 없으면 조용히 유실된다.
    """
    if conv_id is not None or not _has_attachments(files):
        return conv_id
    return await create_conversation(user_sub=user_sub, title=derive_conversation_title(input))


async def _save_attachments(
    files: Optional[List[UploadFile]], conv_id: Optional[str], company: Optional[str] = None,
) -> List[dict]:
    """첨부를 파일 저장소에 영속화하고 메타 목록을 돌려준다(best-effort, 미설정이면 빈 목록).

    반환값 ``[{"file_id", "filename", "content_type"}]`` 은 턴 저장 시 메시지에 기록된다(히스토리 복원용).
    """
    if not files or not file_service.enabled():
        return []
    from datetime import datetime, timezone

    company_seg = (company or "").strip().replace("/", "_") or "_unknown"
    subdir = f"{company_seg}/{conv_id or datetime.now(timezone.utc).strftime('%Y/%m')}"
    saved: List[dict] = []
    for f in files:
        try:
            await f.seek(0)
            raw = await f.read()
            await f.seek(0)
        except Exception:  # noqa: BLE001
            continue
        if not raw:
            continue
        fid = await file_service.save_file(raw, f.filename or "file", f.content_type or "application/octet-stream", subdir=subdir)
        if fid:
            saved.append({"file_id": fid, "filename": f.filename, "content_type": f.content_type})
    return saved


async def _prepare(
    http_request: Request,
    *,
    input: str,
    files: Optional[List[UploadFile]],
    conversation_id: Optional[str],
    llm_provider: Optional[str],
    agent_mode: Optional[str],
    web_search: Optional[bool],
    guide_categories: Optional[str],
    answer_language: Optional[str],
    eager_conversation: bool = False,
):
    """세 진입점(run·run/async·stream)이 공유하는 접수 처리. (orchestrator, 입력, 이미지, 대화 id, 사용자, 모드) 반환."""
    user_info = build_user_info(http_request)
    effective_provider = await resolve_effective_provider(llm_provider, user_info.get("company"))
    conv_id = await _resolve_conversation_id(conversation_id, user_info["sub"])
    if conv_id is None and eager_conversation:
        conv_id = await create_conversation(user_sub=user_info["sub"], title=derive_conversation_title(input))
    conv_id = await _ensure_conversation_scope(conv_id, files, user_info["sub"], input)
    agent_mode, guide_categories = apply_source_defaults(agent_mode, guide_categories, user_info.get("source_path"))

    # UploadFile 은 요청이 끝나면 읽을 수 없으므로 실행 전에 변환한다.
    image_inputs, attachment_text = await _process_files(files)
    saved_attachments = await _save_attachments(files, conv_id, user_info.get("company"))
    augmented_input = _compose_input(input, attachment_text, image_inputs)
    orchestrator = AgentOrchestrator(
        llm_provider=effective_provider,
        enable_pii_masking=settings.enable_pii_masking,
        agent_mode=agent_mode,
        guide_categories=guide_categories,
        language=answer_language,
    )
    _mark_question(orchestrator, input, augmented_input)
    await _apply_web_search(orchestrator, web_search, user_info)
    orchestrator.pending_attachments = saved_attachments  # 턴 저장 시 메시지에 기록된다
    return orchestrator, augmented_input, image_inputs, conv_id, user_info, agent_mode


_FORM_DOC = {
    "conversation_id": "대상 대화방 UUID. 미전송 시 신규 대화방이 생성되어 응답에 동봉된다.",
    "agent_mode": "에이전트 모드(default·guide·agent + 조립부가 extensions.mode_prompts 로 등록한 전용 채널 모드). 미전송 시 서버 기본값.",
    "web_search": "웹 검색 토글. true 이고 테넌트 설정이 허용할 때만 이 요청에 웹 도메인이 열린다.",
    "guide_categories": "Guide 모드 검색 카테고리 화이트리스트(콤마 구분). 미전송/'all' 은 전체.",
    "answer_language": "답변 언어(ko/en/zh/ja). 미전송이면 입력에서 감지 → 직전 발화 → 서버 기본값.",
}


@router.post("/run", response_model=AgentResponse, name="run_agent")
async def run_agent(
    http_request: Request,
    input: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    conversation_id: Optional[str] = Form(None, description=_FORM_DOC["conversation_id"]),
    llm_provider: Optional[str] = Form(None),
    agent_mode: Optional[str] = Form(None, description=_FORM_DOC["agent_mode"]),
    web_search: Optional[bool] = Form(None, description=_FORM_DOC["web_search"]),
    guide_categories: Optional[str] = Form(None, description=_FORM_DOC["guide_categories"]),
    answer_language: Optional[str] = Form(None, description=_FORM_DOC["answer_language"]),
):
    """에이전트를 동기 방식으로 실행한다. 확정된 conversation_id 가 응답에 동봉된다."""
    try:
        orchestrator, augmented_input, image_inputs, conv_id, user_info, _mode = await _prepare(
            http_request, input=input, files=files, conversation_id=conversation_id,
            llm_provider=llm_provider, agent_mode=agent_mode, web_search=web_search,
            guide_categories=guide_categories, answer_language=answer_language,
        )
        result = await orchestrator.run(
            input=augmented_input, conversation_id=conv_id, user_info=user_info, images=image_inputs,
        )
        return AgentResponse(**result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/async", response_model=AgentResponse, name="run_agent_async")
async def run_agent_async(
    http_request: Request,
    input: str = Form(...),
    files: Optional[List[UploadFile]] = File(None),
    conversation_id: Optional[str] = Form(None, description=_FORM_DOC["conversation_id"]),
    llm_provider: Optional[str] = Form(None),
    agent_mode: Optional[str] = Form(None, description=_FORM_DOC["agent_mode"]),
    web_search: Optional[bool] = Form(None, description=_FORM_DOC["web_search"]),
    guide_categories: Optional[str] = Form(None, description=_FORM_DOC["guide_categories"]),
    answer_language: Optional[str] = Form(None, description=_FORM_DOC["answer_language"]),
):
    """에이전트를 백그라운드 작업으로 실행하고 작업 ID 를 즉시 돌려준다(결과는 /jobs/{job_id}).

    동기 경로와 달리 **접수 시점에 대화를 만든다** — 결과보다 응답이 먼저 돌아가므로 클라이언트가
    conversation_id 를 받아 후속 요청을 같은 방으로 보낼 수 있어야 한다.
    """
    try:
        orchestrator, augmented_input, image_inputs, conv_id, user_info, _mode = await _prepare(
            http_request, input=input, files=files, conversation_id=conversation_id,
            llm_provider=llm_provider, agent_mode=agent_mode, web_search=web_search,
            guide_categories=guide_categories, answer_language=answer_language,
            eager_conversation=True,
        )
        input_data = {
            "input": augmented_input,
            "conversation_id": conv_id,
            "llm_provider": llm_provider,
            "agent_mode": _mode,
            "guide_categories": guide_categories,
            "images": [img.model_dump() for img in image_inputs] if image_inputs else [],
        }
        job_id = await job_manager.enqueue(
            job_type="agent_run",
            coroutine=orchestrator.run,
            input_data=input_data,
            metadata={
                "llm_provider": llm_provider,
                "agent_mode": _mode,
                "user_id": user_info["sub"],
                "user_email": user_info.get("email"),
                "company": user_info.get("company"),
                "conversation_id": conv_id,
            },
            input=augmented_input,
            conversation_id=conv_id,
            user_info=user_info,
            images=image_inputs,
        )
        return AgentResponse(
            success=True, job_id=job_id, conversation_id=conv_id,
            output="Job submitted successfully. Use /jobs/{job_id} to check status.",
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream", name="stream_agent")
async def stream_agent(
    http_request: Request,
    input: str = Form(""),
    files: Optional[List[UploadFile]] = File(None),
    conversation_id: Optional[str] = Form(None, description=_FORM_DOC["conversation_id"]),
    llm_provider: Optional[str] = Form(None),
    agent_mode: Optional[str] = Form(None, description=_FORM_DOC["agent_mode"]),
    web_search: Optional[bool] = Form(None, description=_FORM_DOC["web_search"]),
    guide_categories: Optional[str] = Form(None, description=_FORM_DOC["guide_categories"]),
    answer_language: Optional[str] = Form(None, description=_FORM_DOC["answer_language"]),
):
    """에이전트를 SSE 로 실행한다.

    이벤트 순서: ``run_started``(run_id — 중지 핸들) → 진행 이벤트·``token`` … → (중지 시 ``stopped``) →
    ``done``(message_id·conversation_id). 확정된 conversation_id 는 done 에 동봉된다(새 대화는 답 저장 때 생긴다).
    """
    try:
        # 첨부만 보낸 턴도 정상 요청이다 — 비어 있는 질문은 _compose_input 이 기본 지시로 채운다.
        if not input.strip() and not _has_attachments(files):
            raise HTTPException(status_code=400, detail="입력이 비어 있습니다. 텍스트나 첨부 중 하나는 보내야 합니다.")
        orchestrator, augmented_input, image_inputs, conv_id, user_info, _mode = await _prepare(
            http_request, input=input, files=files, conversation_id=conversation_id,
            llm_provider=llm_provider, agent_mode=agent_mode, web_search=web_search,
            guide_categories=guide_categories, answer_language=answer_language,
        )

        async def generate() -> AsyncIterator[str]:
            _out: Dict[str, Any] = {}
            # run 등록 후 취소 Event 를 ContextVar 에 심는다 — orchestrator.stream 아래 루프가 is_cancelled() 로 폴링한다.
            run_id, cancel_event = register_run(user_info["sub"])
            set_cancel_event(cancel_event)
            try:
                yield f"data: {AgentStreamChunk(type='run_started', data={'run_id': run_id}).model_dump_json()}\n\n"
                async for token in orchestrator.stream(
                    input=augmented_input, conversation_id=conv_id, user_info=user_info,
                    images=image_inputs, _out=_out,
                ):
                    try:
                        event = json.loads(token)
                        if isinstance(event, dict) and event.get("type") in PROGRESS_EVENT_TYPES:
                            chunk = AgentStreamChunk(
                                type=event["type"], content=event.get("name") or event.get("content"), data=event,
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"
                            continue
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    yield f"data: {AgentStreamChunk(type='token', content=token).model_dump_json()}\n\n"

                if _out.get("stopped"):
                    yield f"data: {AgentStreamChunk(type='stopped', data={'run_id': run_id}).model_dump_json()}\n\n"
                done_data = {
                    "llm_provider": orchestrator.llm_provider,
                    "message_id": _out.get("message_id"),
                    "conversation_id": _out.get("conversation_id") or conv_id,
                }
                # guide 모드가 가이드로 답할 수 없는 질문(abstain)이면 에이전트 모드 전환을 제안한다.
                if _out.get("suggested_mode"):
                    done_data["suggested_mode"] = _out["suggested_mode"]
                    done_data["abstain_reason"] = _out.get("abstain_reason")
                yield f"data: {AgentStreamChunk(type='done', data=done_data).model_dump_json()}\n\n"
            except Exception as e:
                # SSE 는 헤더가 이미 200 으로 나가 HTTP 미들웨어가 못 본다 — 여기서 남긴다.
                logger.exception("[stream] SSE error 청크로 종료: %s", e,
                                 extra=event_extra("exception", conversation_id=conv_id))
                yield f"data: {AgentStreamChunk(type='error', content=str(e)).model_dump_json()}\n\n"
            finally:
                unregister_run(run_id)
                set_cancel_event(None)

        return StreamingResponse(
            generate(), media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str, http_request: Request) -> dict:
    """진행 중인 스트리밍 run 을 중지 요청한다(협조적 취소 — 다음 체크포인트에서 멈추고 턴은 저장되지 않는다).

    Raises:
        HTTPException(404): run 이 없거나 본인 소유가 아닐 때(존재 여부 비노출).
    """
    user_info = build_user_info(http_request)
    if not request_stop(run_id, user_info["sub"]):
        raise HTTPException(status_code=404, detail="진행 중인 run 을 찾을 수 없습니다.")
    return {"success": True, "run_id": run_id}


class ResumeActionRequest(BaseModel):
    """HITL 승인/거절 시 클라이언트가 보내는 값."""

    values: Dict[str, Any] = Field(default_factory=dict)
    action: Literal["approve", "reject"] = "approve"
    # 카드가 만들어질 때 받은 대기 항목 id. 지금 대기 중인 항목과 다르면 409 — 앞 카드 버튼이 새 초안을
    # 실행하지 않게 한다(agents/hitl/resume.StalePendingAction).
    pending_id: Optional[str] = None


_TOOL_ERROR_PREFIX = re.compile(r"^\s*Error executing tool\s+\S+\s*:\s*", re.IGNORECASE)


def _failure_detail(error: Any) -> str:
    """도구 실패 문자열을 사용자에게 보여도 되는 한 문장으로 다듬는다(프레임워크 접두·도구명 제거)."""
    text = _TOOL_ERROR_PREFIX.sub("", str(error or "").strip()).strip()
    return sanitize_tool_names(text) or "실행에 실패했습니다."


async def _persist_action_outcome(
    conversation_id: str, user_sub: str, pending: dict, is_reject: bool,
    extra_note: str = "", outcome_text: Optional[str] = None, failed_text: Optional[str] = None,
) -> None:
    """실행/거절/실패한 액션을 대화 이력에 평문 확인 메시지로 남긴다.

    tool_ctx 는 직전 1턴만 남으므로, 무엇을 했는지를 assistant 메시지로 남겨야 continuation·후속 턴이
    이미 한 일을 다시 하지 않는다. 거절·실패면 초안 인자 원본을 stash 에 남겨 다음 정정 턴("9시 말고 10시")이
    같은 도구를 수정본으로 다시 부를 수 있게 하고, 승인이면 지운다(뒤늦은 정정이 유령 초안을 살리지 않게).
    이 텍스트는 대화 목록 API 로 사용자에게도 보이므로 raw 도구명을 넣지 않는다.
    """
    from agent_arch.agents.hitl import action_done_message
    from agent_arch.memory.conversation import clear_rejected_draft, save_logged_in_turn, save_rejected_draft

    tool_name = pending.get("tool_name", "작업")
    fields = (((pending.get("draft_view") or {}).get("preview") or {}).get("fields")) or []
    summary = ", ".join(f"{f.get('label')}: {f.get('value')}" for f in fields if isinstance(f, dict) and f.get("label"))
    if failed_text:
        text = f"[실패] {failed_text} — 아직 등록/발송되지 않았습니다."
        if summary:
            text += f" (초안 내용: {summary})"
    elif is_reject:
        text = "[취소됨] 사용자가 이 초안 작업을 취소했습니다 — 아직 등록/발송되지 않았습니다."
        if summary:
            text += f" (취소된 초안 내용: {summary})"
    else:
        text = (outcome_text or action_done_message(tool_name, pending.get("arguments"))) + (f" ({summary})" if summary else "")
        if extra_note:
            text += extra_note
    user_note = "(초안 거절됨)" if is_reject else "(승인 실행됨)"
    # 이 턴은 LLM 을 부르지 않는다. 그 실행을 이끈 직전 답변의 모델을 물려받는다(못 찾으면 비운다).
    provider = model = None
    try:
        from agent_arch.db.repository.messages import get_last_answer_model

        found = await get_last_answer_model(conversation_id)
        if found:
            provider, model = found
    except Exception:
        logger.exception("[hitl] 직전 답변 모델 조회 실패 — 모델 없이 저장한다")
    try:
        await save_logged_in_turn(conversation_id, user_sub, user_note, sanitize_tool_names(text),
                                  tool_calls=None, llm_provider=provider, model_name=model)
    except Exception:
        logger.exception("[hitl] 실행 결과 history 영속 실패 — continuation 컨텍스트 누락 가능")
    if is_reject or failed_text:
        await save_rejected_draft(conversation_id, tool_name, pending.get("arguments"), fields)
    else:
        await clear_rejected_draft(conversation_id)


async def _maybe_continue(conversation_id: str, user_info: dict) -> Optional[dict]:
    """큐가 비면 한 턴 더 돌려(continuation) 다음 승인 필요 액션을 계획한다. 새 큐가 나오면 그 첫 항목.

    continuation 은 메인 턴과 **같은 provider** 로 돌린다 — /resume 에는 모델 선택이 없으므로 직전 턴이 실제
    쓴 provider 를 저장소에서 복구한다. 안 그러면 서버 기본 모델로 돌아 후속 액션을 놓친다.
    """
    from agent_arch.agents.hitl import load_pending
    from agent_arch.db.repository import get_last_message_provider

    try:
        recovered = await get_last_message_provider(conversation_id)
        provider = await resolve_effective_provider(recovered, user_info.get("company"))
        orch = AgentOrchestrator(llm_provider=provider)
        cont = await orch._run_react_continuation(conversation_id, user_info)
        pq = cont.get("pending_queue")
        if pq:
            await orch._checkpoint_pending_action(conversation_id, pq)
            return await load_pending(conversation_id)
    except Exception:
        logger.exception("[hitl] continuation 실패 — 현재 액션만 완료 처리")
    return None


@router.post("/{conversation_id}/resume")
async def resume_agent_action(conversation_id: str, http_request: Request, body: ResumeActionRequest) -> Dict[str, Any]:
    """HITL 로 일시정지된 비가역 액션을 사람 승인 값으로 재개해 실제 실행한다.

    체크포인트(thread_id=conversation_id)에서 그 지점부터 ``confirm=True`` 로 이어 실행한다. 도구는 체크포인트
    에서만 읽으므로 위조할 수 없고, 승인 값은 초안 필드 화이트리스트로 검증한다. 실행은 요청자 본인 명의다.

    Raises:
        404: conversation 이 본인 소유가 아님.
        409: 재개할 대기 액션이 없음 / 카드의 pending_id 가 현재 대기 항목과 다름 / 같은 승인이 처리 중.
        400: 승인 값에 허용되지 않은 필드 포함.
        502: 도구 실행 실패.
    """
    user_info = build_user_info(http_request)
    if await get_conversation(conversation_id, user_info["sub"]) is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    from agent_arch.agents.context import set_conversation_id, set_user_context
    from agent_arch.agents.hitl import (
        ResumeInProgress,
        StalePendingAction,
        action_done_message,
        load_pending,
        resume_action,
        resume_guard,
        validate_resume_values,
    )
    from agent_arch.agents.tools.action_view import build_action_artifact

    # 승인 한 번에 액션 한 번 — 실행이 끝나기 전 두 번째 클릭이 같은 대기 항목을 집지 않게 조회부터 재개까지 잠근다.
    try:
        async with resume_guard(conversation_id):
            pending = await load_pending(conversation_id)
            if pending is None:
                raise HTTPException(status_code=409, detail="재개할 승인 대기 액션이 없습니다.")

            is_reject = body.action == "reject"
            if not is_reject:
                try:
                    validate_resume_values(pending, body.values)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))

            set_user_context(user_info)
            set_conversation_id(conversation_id)
            result = await resume_action(
                conversation_id, body.values, reject=is_reject, user_info=user_info, pending_id=body.pending_id,
            )
            if not is_reject and not result.get("success"):
                fail_text = _failure_detail(result.get("error"))
                try:
                    await _persist_action_outcome(conversation_id, user_info["sub"], pending, False, failed_text=fail_text)
                except Exception:
                    logger.exception("[hitl] 실행 실패 history 영속 실패")
                raise HTTPException(status_code=502, detail=fail_text)

            # 임시보관(held): 호출은 성공했지만 목적까지 못 간 경우 — 고정 완료 문구 대신 도구가 준 사유를 쓴다.
            held_message = str(result.get("message") or "").strip() if (not is_reject and result.get("held")) else ""
            effective_args = {**(pending.get("arguments") or {}), **(body.values or {})}
            followup_suffix = ""
            if not is_reject and not held_message:
                from agent_arch.agents.hitl.followups import followup_note

                followup_suffix = await followup_note(pending, effective_args, user_info, conversation_id)
                panel_note = str(((result.get("view") or {}).get("panel_note")) or "").strip()
                if panel_note:
                    followup_suffix = f"{followup_suffix}\n\n{panel_note}" if followup_suffix else f"\n\n{panel_note}"

            await _persist_action_outcome(
                conversation_id, user_info["sub"], pending, is_reject,
                extra_note=followup_suffix, outcome_text=held_message or None,
            )

            view = result.get("view")
            artifact = build_action_artifact(view, conversation_id, None) if isinstance(view, dict) else None

            # 같은 큐에 다음 항목이 남았으면 그걸, 없으면 continuation 으로 다음 액션을 계획한다.
            # **거절이면 continuation 하지 않는다** — 거절한 액션을 원 요청 기준으로 다시 제안하는 무한 루프가 된다.
            next_pending = result.get("next_pending")
            if next_pending is None and not is_reject:
                next_pending = await _maybe_continue(conversation_id, user_info)
            next_pending_out = None
            if next_pending:
                np_view = next_pending.get("draft_view")
                next_pending_out = {
                    "artifact": build_action_artifact(np_view, conversation_id, None) if isinstance(np_view, dict) else None,
                    "pendingId": next_pending.get("pending_id"),
                    "queueIndex": next_pending.get("queue_index"),
                    "queueTotal": next_pending.get("queue_total"),
                }

            if is_reject:
                message = ("이 작업은 취소했어요." if next_pending_out else
                           "이 작업은 취소했어요. 수정해서 다시 보내려면 어떻게 바꿀지 말씀해 주세요. (그만두려면 괜찮다고 해주세요.)")
            else:
                message = held_message or (action_done_message(pending["tool_name"], effective_args) + followup_suffix)
            return {
                "success": True,
                "message": message,
                "view": None if is_reject else view,
                "artifact": None if is_reject else artifact,
                "nextPending": next_pending_out,
            }
    except StalePendingAction:
        raise HTTPException(status_code=409, detail="이 카드는 더 이상 유효하지 않습니다 — 새 카드에서 다시 승인해 주세요.")
    except ResumeInProgress:
        raise HTTPException(status_code=409, detail="이 승인은 이미 처리 중입니다.")


@router.post("/messages/{message_id}/rating")
async def rate_message(message_id: str, body: RatingRequest, background_tasks: BackgroundTasks):
    """메시지 평가를 저장하고 LangSmith 에 피드백을 보낸다(백그라운드)."""
    msg = await get_message_for_feedback(message_id)
    if msg is None:
        raise HTTPException(status_code=404, detail="메시지를 찾을 수 없습니다.")
    if not await update_message_rating(message_id, body.rating):
        raise HTTPException(status_code=404, detail="메시지를 찾을 수 없습니다.")
    if msg["langsmith_run_id"]:
        background_tasks.add_task(
            send_feedback,
            langsmith_run_id=msg["langsmith_run_id"],
            rating=body.rating,
            query=msg["query"],
            answer=msg["answer"],
            action_view=msg.get("action_view"),
        )
    return {"success": True}


@router.get("/info")
async def get_agent_info():
    """LLM 정보·사용 가능한 도구·에이전트 설정값."""
    orchestrator = AgentOrchestrator()
    return {
        "llm_info": orchestrator.get_llm_info(),
        "available_tools": [tool.to_dict() for tool in orchestrator.get_tools()],
        "settings": {
            "agent_mode": settings.agent_mode,
            "max_iterations": settings.max_iterations,
            "agent_timeout": settings.agent_timeout,
            "pii_masking_enabled": settings.enable_pii_masking,
        },
    }
