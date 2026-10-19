"""어휘 소스 — 임직원 디렉터리(요청자별 캐시)와 이번 대화 도구 결과에서 수확한 이름.

- **디렉터리**: ``DirectoryClient`` 로 계열사 그룹의 회사 목록을 받고 회사마다 동료를
  50건씩 페이징해 (이름, 이메일) 을 모은다. 요청자 쿠키로 조회하므로 가시성 범위가 사용자마다
  달라 **요청자별** 로 캐시한다(``pii:lexicon:user:{sub}``, 기본 1시간). 값은 Fernet 으로
  암호화해 둔다(Redis 에 임직원 명단을 평문으로 두지 않는다).
- **도구 결과 수확**: 메일·메신저·디렉터리 결과 JSON 의 ``name``/``email``/``sender`` 류 키
  값을 대화 범위 목록(``pii:lexicon:conv:{id}``, 24시간)에 합친다. 제3자 이름이 다음 호출부터
  치환된다.
- 어느 소스가 실패해도 예외를 밖으로 내지 않는다 — 이전 캐시를 유지하거나 빈 목록.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import re
from typing import Any, Iterable, Optional

from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

USER_KEY = "pii:lexicon:user:{sub}"
CONV_KEY = "pii:lexicon:conv:{conv}"
CONV_TTL = 86400
_PAGE = 50
_MAX_PAGES_PER_COMPANY = 40
_MAX_COMPANIES = 20
_MAX_HARVEST = 500

#: 이름·이메일을 담는 키. 파일명·도구명·회사명 등은 제외.
HARVEST_KEY = re.compile(
    r"(?i)(?:^|_)(?:name|full_name|display_name|user_name|username|email|sender|recipient|assignee|author|owner|reporter"
    r"|manager|requester|approver|담당자|요청자|이름|성명|작성자|결재자|수신자|발신자)(?:$|_)"
)
HARVEST_DENY = re.compile(r"(?i)(?:file|tool|company|project|template|product|team|dept|department|skill|agent|model|channel|group|host|domain)_?name")
EMAIL_KEY = re.compile(r"(?i)e?mail")
_EMAIL = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_KO_NAME = re.compile(r"^[가-힣]{2,4}$")
_LATIN_NAME = re.compile(r"^[A-Za-z][A-Za-z'.-]{0,29}(?:\s+[A-Za-z][A-Za-z'.-]{0,29}){1,3}$")


# ── 암호화 ────────────────────────────────────────────────────────────────────
def _fernet():
    from cryptography.fernet import Fernet
    from agent_arch.llm.pii.policy import secret

    key = base64.urlsafe_b64encode(hashlib.sha256((secret() + "|pii-lexicon").encode("utf-8")).digest())
    return Fernet(key)


def encrypt_json(obj: Any) -> str:
    return _fernet().encrypt(json.dumps(obj, ensure_ascii=False).encode("utf-8")).decode("ascii")


def decrypt_json(token: str) -> Any:
    return json.loads(_fernet().decrypt(token.encode("ascii")).decode("utf-8"))


# ── Redis ────────────────────────────────────────────────────────────────────
async def _cache_get(key: str) -> Optional[Any]:
    try:
        from agent_arch.memory.redis_store import redis_store
        raw = await redis_store.get(key)
        if isinstance(raw, str) and raw:
            return decrypt_json(raw)
    except Exception:  # noqa: BLE001
        logger.debug("[pii_guard] 어휘 캐시 읽기 실패 key=%s", key, exc_info=True)
    return None


async def _cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        from agent_arch.memory.redis_store import redis_store
        await redis_store.set(key, encrypt_json(value), expire=int(ttl))
    except Exception:  # noqa: BLE001
        logger.debug("[pii_guard] 어휘 캐시 쓰기 실패 key=%s", key, exc_info=True)


def lexicon_ttl() -> int:
    return int(getattr(settings, "pii_guard_lexicon_ttl", 3600) or 3600)


# ── 디렉터리 ──────────────────────────────────────────────────────────────────
def _row_person(row: dict) -> Optional[tuple[str, str]]:
    name = str(row.get("name") or "").strip()
    email = str(row.get("email") or "").strip().lower()
    if not name and not email:
        return None
    return (name, email)


async def fetch_directory(client: Any) -> dict:
    """{"persons": [[name, email], …], "companies": [회사명, …]} — 계열사 전체를 페이징."""
    persons: dict[tuple[str, str], None] = {}
    companies: list[str] = []
    rows = await client.list_companies() or []
    company_ids: list[str] = []
    for r in rows[:_MAX_COMPANIES]:
        if not isinstance(r, dict):
            continue
        cid = str(r.get("company") or "").strip()
        label = str(r.get("company_name") or "").strip()
        if cid:
            company_ids.append(cid)
        companies.extend(x for x in (cid, label) if x)
    if not company_ids:
        company_ids = [getattr(client, "_company", "") or ""]
    for cid in company_ids:
        for page in range(_MAX_PAGES_PER_COMPANY):
            batch = await client.search_colleagues(query="", limit=_PAGE, company=cid or None, offset=page * _PAGE)
            for row in batch or []:
                if isinstance(row, dict) and (p := _row_person(row)):
                    persons[p] = None
            if not batch or len(batch) < _PAGE:
                break
    return {"persons": [list(p) for p in persons], "companies": sorted(set(companies))}


async def directory_lexicon(user_sub: Optional[str], *, timeout: float = 2.0) -> dict:
    """요청자별 캐시 → 미스면 디렉터리 조회(상한 ``timeout``). 실패·초과면 빈 결과(예외 없음)."""
    empty = {"persons": [], "companies": []}
    if not user_sub:
        return empty
    key = USER_KEY.format(sub=user_sub)
    cached = await _cache_get(key)
    if isinstance(cached, dict) and "persons" in cached:
        return cached
    try:
        from agent_arch.extensions import extensions
        if extensions.directory_client is None:
            return empty
        client = extensions.directory_client()
        if not getattr(client, "enabled", False):
            return empty
        data = await asyncio.wait_for(fetch_directory(client), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("[pii_guard] 디렉터리 어휘 조회 %.1fs 초과 — 이번 턴은 부분 어휘(sub=%s)", timeout, user_sub)
        return empty
    except Exception:  # noqa: BLE001
        logger.warning("[pii_guard] 디렉터리 어휘 조회 실패 — 빈 어휘(sub=%s)", user_sub, exc_info=True)
        return empty
    await _cache_set(key, data, lexicon_ttl())
    logger.info("[pii_guard] 디렉터리 어휘 적재 sub=%s persons=%d companies=%d", user_sub, len(data["persons"]), len(data["companies"]))
    return data


async def prewarm_directory(user_sub: Optional[str]) -> None:
    """턴 시작에 백그라운드로 캐시를 채운다. 2초를 넘겨도 끝까지 받아 다음 턴부터 완전해진다."""
    try:
        await directory_lexicon(user_sub, timeout=30.0)
    except Exception:  # noqa: BLE001
        pass


# ── 도구 결과 수확 ────────────────────────────────────────────────────────────
def _looks_like_name(v: str) -> bool:
    return bool(_KO_NAME.match(v) or _LATIN_NAME.match(v))


def harvest_persons(payload: Any) -> list[tuple[str, str]]:
    """JSON 값에서 (이름, 이메일) 을 모은다. 같은 dict 안의 이름·이메일은 한 사람으로 묶는다."""
    found: dict[tuple[str, str], None] = {}

    def _walk(o: Any) -> None:
        if isinstance(o, dict):
            name = email = ""
            for k, v in o.items():
                if not isinstance(k, str):
                    continue
                if isinstance(v, str):
                    s = v.strip()
                    if EMAIL_KEY.search(k) and _EMAIL.match(s):
                        email = email or s.lower()
                    elif HARVEST_KEY.search(k) and not HARVEST_DENY.search(k) and _looks_like_name(s):
                        name = name or s
                    elif _EMAIL.match(s) and HARVEST_KEY.search(k) and not HARVEST_DENY.search(k):
                        email = email or s.lower()
                else:
                    _walk(v)
            if name or email:
                found[(name, email)] = None
        elif isinstance(o, list):
            for v in o:
                _walk(v)

    _walk(payload)
    return list(found)


def harvest_from_text(content: str) -> list[tuple[str, str]]:
    try:
        parsed = json.loads(content)
    except (ValueError, TypeError):
        return []
    return harvest_persons(parsed) if isinstance(parsed, (dict, list)) else []


async def remember_harvest(conversation_id: Optional[str], persons: Iterable[tuple[str, str]]) -> None:
    persons = [p for p in persons if p[0] or p[1]]
    if not conversation_id or not persons:
        return
    key = CONV_KEY.format(conv=conversation_id)
    cur = await _cache_get(key)
    merged: dict[tuple[str, str], None] = {}
    for p in (cur or []) if isinstance(cur, list) else []:
        if isinstance(p, list) and len(p) == 2:
            merged[(str(p[0]), str(p[1]))] = None
    for p in persons:
        merged[(str(p[0]), str(p[1]))] = None
    rows = [list(p) for p in list(merged)[-_MAX_HARVEST:]]
    await _cache_set(key, rows, CONV_TTL)


async def harvested_lexicon(conversation_id: Optional[str]) -> list[tuple[str, str]]:
    if not conversation_id:
        return []
    cur = await _cache_get(CONV_KEY.format(conv=conversation_id))
    return [(str(p[0]), str(p[1])) for p in cur if isinstance(p, list) and len(p) == 2] if isinstance(cur, list) else []


async def harvest_tool_result(conversation_id: Optional[str], content: str, company: Optional[str] = None) -> None:
    """executor 가 결과를 만든 직후 부른다(백그라운드). 회사가 켜지 않았으면 남기지 않는다. 실패는 삼킨다."""
    try:
        from agent_arch.llm.pii.policy import company_enabled

        if not await company_enabled(company):
            return
        persons = harvest_from_text(content)
        if persons:
            await remember_harvest(conversation_id, persons)
    except Exception:  # noqa: BLE001
        logger.debug("[pii_guard] 도구 결과 수확 실패", exc_info=True)
