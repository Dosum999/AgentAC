"""외부 LLM 가명화 — 식별자를 자리표시 토큰으로 치환하고 응답에서 복원한다.

컨퍼런스 "[에이전트] 외부 LLM 개인정보 보호 설계" §2.1. 호출 한 번의 ``messages`` 를
치환하면서 복원표를 만들고, 응답 텍스트·도구 인자에서 토큰을 원문으로 되돌린다.
매핑은 저장하지 않는다 — 이력이 원문으로 저장되므로 호출마다 다시 만든다.

- ``tokens``    토큰 발급(HMAC, 충돌 시 12자)·종류·라벨
- ``patterns``  정규식 규칙(이메일·전화·차단류)
- ``lexicon``   어휘(임직원·요청자·도구 결과 이름)와 한글 조사 경계 패턴
- ``engine``    단일 패스 치환(``Substituter``)과 복원(``Restorer``)
- ``stream``    스트림 복원 버퍼(토큰이 청크 경계에 걸쳐도 복원)
- ``policy``    전역 모드·대상 provider·회사 opt-in
- ``turn``      이번 턴의 어휘(요청자·회사; 디렉터리·도구 결과는 어휘 PR)
- ``audit``     shadow/enforce 기록(값 비노출)
- ``guard_llm`` ``BaseLLM`` 래퍼 — ``LLMFactory.create`` 가 외부 provider 에만 씌운다
"""
from typing import Any

from agent_arch.llm.pii.tokens import Kind, TokenIssuer, canonical_key, render, scope_id  # noqa: F401
from agent_arch.llm.pii.lexicon import Alias, Lexicon, build_lexicon, expand_person_aliases  # noqa: F401
from agent_arch.llm.pii.engine import (  # noqa: F401
    Finding,
    RestoreResult,
    Restorer,
    Substituter,
    Substitution,
    find_tokens,
)
from agent_arch.llm.pii.stream import StreamRestorer  # noqa: F401


def wrap_if_external(llm: Any, provider_name: str) -> Any:
    """대상 provider 이고 전역 모드가 off 가 아니면 ``PIIGuardLLM`` 으로 감싼다. 이미 래퍼면 그대로."""
    from agent_arch.llm.pii.guard_llm import PIIGuardLLM
    from agent_arch.llm.pii.policy import global_mode, is_guarded_provider

    if isinstance(llm, PIIGuardLLM):
        return llm
    if global_mode() == "off" or not is_guarded_provider(provider_name):
        return llm
    return PIIGuardLLM(llm, provider_name)


def unwrap_llm(llm: Any) -> Any:
    """래퍼를 벗겨 실제 provider 인스턴스를 돌려준다(provider 판별용). 래퍼가 아니면 그대로."""
    from agent_arch.llm.pii.guard_llm import PIIGuardLLM

    while isinstance(llm, PIIGuardLLM):
        llm = llm.inner
    return llm
