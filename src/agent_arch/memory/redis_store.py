from typing import Optional, Any, Dict
import json
import redis.asyncio as aioredis
from agent_arch.core.config import settings


class RedisStore:
    """Redis 기반 캐시 및 세션 스토어.

    주요 용도:
        1. 단기 메모리 (대화 컨텍스트)
        2. 작업 상태 관리
        3. 캐싱
    """

    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self._initialized = False

    async def connect(self):
        """Redis에 연결한다."""
        if self._initialized:
            return

        self.redis = await aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        self._initialized = True

    async def disconnect(self):
        """Redis 연결을 종료한다."""
        if self.redis:
            await self.redis.close()
            self._initialized = False

    async def set(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None
    ) -> bool:
        """값을 저장한다.

        Args:
            key (str): 키
            value (Any): 값 (자동으로 JSON 직렬화)
            expire (Optional[int]): 만료 시간 (초). None이면 만료 없음.

        Returns:
            bool: 저장 성공 여부
        """
        try:
            serialized = json.dumps(value)
            await self.redis.set(key, serialized, ex=expire)
            return True
        except Exception as e:
            print(f"Redis set error: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """값을 조회한다.

        Args:
            key (str): 키

        Returns:
            Optional[Any]: 저장된 값 (JSON 역직렬화). 키가 없으면 None.
        """
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """키를 삭제한다.

        Args:
            key (str): 삭제할 키

        Returns:
            bool: 삭제 성공 여부
        """
        try:
            await self.redis.delete(key)
            return True
        except Exception as e:
            print(f"Redis delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """키의 존재 여부를 확인한다.

        Args:
            key (str): 확인할 키

        Returns:
            bool: 키가 존재하면 True, 아니면 False
        """
        try:
            return await self.redis.exists(key) > 0
        except Exception:
            return False

    async def set_hash(self, key: str, mapping: Dict[str, Any]) -> bool:
        """해시 값을 저장한다.

        Args:
            key (str): 키
            mapping (Dict[str, Any]): 저장할 필드-값 매핑 (자동으로 JSON 직렬화)

        Returns:
            bool: 저장 성공 여부
        """
        try:
            serialized = {k: json.dumps(v) for k, v in mapping.items()}
            await self.redis.hset(key, mapping=serialized)
            return True
        except Exception as e:
            print(f"Redis hset error: {e}")
            return False

    async def get_hash(self, key: str) -> Optional[Dict[str, Any]]:
        """해시 값을 조회한다.

        Args:
            key (str): 키

        Returns:
            Optional[Dict[str, Any]]: 필드-값 매핑 (JSON 역직렬화). 키가 없으면 None.
        """
        try:
            data = await self.redis.hgetall(key)
            if data:
                return {k: json.loads(v) for k, v in data.items()}
            return None
        except Exception as e:
            print(f"Redis hgetall error: {e}")
            return None

    async def lpush(self, key: str, *values: Any) -> bool:
        """리스트 왼쪽에 값을 추가한다.

        Args:
            key (str): 키
            *values (Any): 추가할 값 목록 (자동으로 JSON 직렬화)

        Returns:
            bool: 추가 성공 여부
        """
        try:
            serialized = [json.dumps(v) for v in values]
            await self.redis.lpush(key, *serialized)
            return True
        except Exception as e:
            print(f"Redis lpush error: {e}")
            return False

    async def rpush(self, key: str, *values: Any) -> bool:
        """리스트 오른쪽에 값을 추가한다 (삽입 순서 유지).

        Args:
            key (str): 키
            *values (Any): 추가할 값 목록 (자동으로 JSON 직렬화)

        Returns:
            bool: 추가 성공 여부
        """
        try:
            serialized = [json.dumps(v) for v in values]
            await self.redis.rpush(key, *serialized)
            return True
        except Exception as e:
            print(f"Redis rpush error: {e}")
            return False

    async def ltrim(self, key: str, start: int, end: int) -> bool:
        """리스트를 지정 범위로 잘라낸다.

        Args:
            key (str): 키
            start (int): 유지할 시작 인덱스 (음수 허용)
            end (int): 유지할 종료 인덱스 (음수 허용)

        Returns:
            bool: 성공 여부
        """
        try:
            await self.redis.ltrim(key, start, end)
            return True
        except Exception as e:
            print(f"Redis ltrim error: {e}")
            return False

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> list:
        """리스트의 범위를 조회한다.

        Args:
            key (str): 키
            start (int): 시작 인덱스. 기본값 0.
            end (int): 종료 인덱스. 기본값 -1 (전체).

        Returns:
            list: JSON 역직렬화된 값 목록
        """
        try:
            values = await self.redis.lrange(key, start, end)
            return [json.loads(v) for v in values]
        except Exception as e:
            print(f"Redis lrange error: {e}")
            return []

    async def expire(self, key: str, seconds: int) -> bool:
        """키에 만료 시간을 설정한다.

        Args:
            key (str): 키
            seconds (int): 만료 시간 (초)

        Returns:
            bool: 설정 성공 여부
        """
        try:
            await self.redis.expire(key, seconds)
            return True
        except Exception:
            return False

    async def set_nx(
        self,
        key: str,
        value: Any,
        expire: Optional[int] = None,
    ) -> bool:
        """SET NX — 키가 없을 때만 저장한다. 분산 락 용도.

        Args:
            key (str): 키
            value (Any): 저장할 값 (자동 JSON 직렬화)
            expire (Optional[int]): 만료 시간 (초). None이면 만료 없음.

        Returns:
            bool: 락 획득 성공 시 True, 이미 다른 작업이 보유 중이면 False.
        """
        try:
            serialized = json.dumps(value)
            result = await self.redis.set(key, serialized, ex=expire, nx=True)
            return result is True
        except Exception as e:
            print(f"Redis set_nx error: {e}")
            return False

    async def lindex(self, key: str, index: int) -> Optional[Any]:
        """리스트의 특정 인덱스 값을 조회한다.

        Args:
            key (str): 키
            index (int): 조회할 인덱스 (음수 허용)

        Returns:
            Optional[Any]: JSON 역직렬화된 값. 인덱스 범위 밖이거나 키가 없으면 None.
        """
        try:
            value = await self.redis.lindex(key, index)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Redis lindex error: {e}")
            return None

    async def getdel(self, key: str) -> Optional[Any]:
        """GET + DEL atomic — 압축 통지를 1회만 소비하기 위한 패턴.

        Redis 6.2+: GETDEL 명령으로 atomic 처리.
        Redis < 6.2: ResponseError 캐치 후 GET + DELETE 폴백 (atomic 아님 —
            동시 호출 시 양쪽이 값을 받을 수 있음. 호출자가 ts 기준 dedup 책임).

        Args:
            key (str): 키

        Returns:
            Optional[Any]: JSON 역직렬화된 값. 키가 없거나 오류 시 None.
        """
        value = None
        try:
            value = await self.redis.getdel(key)
        except Exception:
            # 서버 미지원 또는 일시 오류 → GET + DEL 폴백
            try:
                value = await self.redis.get(key)
                if value:
                    await self.redis.delete(key)
            except Exception as e:
                print(f"Redis getdel fallback error: {e}")
                return None
        if value:
            try:
                return json.loads(value)
            except Exception:
                return None
        return None

    # 작업 큐 관련 메서드
    async def enqueue_job(self, job_id: str, job_data: Dict[str, Any]) -> bool:
        """작업을 큐에 추가한다.

        Args:
            job_id (str): 작업 ID
            job_data (Dict[str, Any]): 저장할 작업 데이터

        Returns:
            bool: 추가 성공 여부
        """
        return await self.set(f"job:{job_id}", job_data)

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태를 조회한다.

        Args:
            job_id (str): 작업 ID

        Returns:
            Optional[Dict[str, Any]]: 작업 데이터. 없으면 None.
        """
        return await self.get(f"job:{job_id}")

    async def update_job_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Any] = None
    ) -> bool:
        """작업 상태를 업데이트한다.

        Args:
            job_id (str): 작업 ID
            status (str): 변경할 작업 상태
            result (Optional[Any]): 저장할 결과 데이터. 기본값 None.

        Returns:
            bool: 업데이트 성공 여부
        """
        job = await self.get_job(job_id)
        if job:
            job["status"] = status
            if result is not None:
                job["result"] = result
            return await self.set(f"job:{job_id}", job)
        return False


# 싱글톤 인스턴스
redis_store = RedisStore()
