import json
from pathlib import Path
from typing import Dict, List, Optional

from agent_arch.agents.permissions import normalize_service
from agent_arch.services.api_registry.models import ExternalToolDef


class ToolLoader:
    """llm_tools.json을 파싱하여 ExternalToolDef 리스트를 반환한다.

    JSON 파일에서 도구 정의를 로드하고, 활성화된 외부 서비스 도구만 필터링하여 반환한다.
    """

    def __init__(self) -> None:
        self._cache: Optional[Dict[str, ExternalToolDef]] = None

    def load_from_json(self, path: str = "data/llm_tools.json") -> List[ExternalToolDef]:
        """llm_tools.json을 로드하고 enabled 항목만 반환한다.

        parameters_schema 필드는 JSON 문자열에서 dict로 변환한다.
        로드 결과는 tool_name → ExternalToolDef dict로 캐싱하여 get_by_name()에서 재사용된다.

        Args:
            path (str): llm_tools.json 파일 경로 (기본값: "data/llm_tools.json")

        Returns:
            List[ExternalToolDef]: 활성화된 외부 서비스 도구 정의 리스트

        Raises:
            FileNotFoundError: JSON 파일을 찾을 수 없는 경우
            json.JSONDecodeError: JSON 파싱 오류 발생 시

        Note:
            enabled=True이고 is_external=True인 도구만 반환한다.
            내부 도구는 http_method나 endpoint_path가 없으므로 제외된다.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"llm_tools.json not found: {file_path.absolute()}")

        with open(file_path, encoding="utf-8") as f:
            raw: List[dict] = json.load(f)

        tools = []
        for item in raw:
            # enabled 아니거나 외부 서비스가 아닌 도구는 제외
            # (내부 도구는 http_method/endpoint_path가 없음)
            if not item.get("enabled") or not item.get("is_external"):
                continue

            # parameters_schema: 문자열이면 dict로 변환
            schema = item.get("parameters_schema", "{}")
            if isinstance(schema, str):
                item["parameters_schema"] = json.loads(schema)

            # 도메인 = 정규화된 서비스 키 (예: hr_service → hr)
            if item.get("external_service") and not item.get("domain"):
                item["domain"] = normalize_service(item["external_service"])

            tools.append(ExternalToolDef(**item))

        self._cache = {t.tool_name: t for t in tools}
        return tools

    def defs_by_domain(self, domain: str) -> List[ExternalToolDef]:
        """도메인(정규화된 서비스 키)에 속한 enabled 동적 도구 정의를 반환한다.

        채널 스코핑(특정 진입점에서만 한 도메인을 여는 노출 패턴)이 인덱스 검색을
        거치지 않고 도메인 전체를 직접 바인딩할 때 쓴다 — 검색은 top_k 근사라
        "그 도메인의 전부"를 보장하지 못한다.

        첫 호출 시 load_from_json()으로 캐시를 초기화한다 (get_by_name과 동일).

        Args:
            domain: 정규화된 서비스 키 (예: "hr"). ``normalize_service`` 결과와
                같은 체계이며 ``ExternalToolDef.domain`` 과 비교된다.

        Returns:
            해당 도메인의 ExternalToolDef 리스트. 없으면 빈 리스트.
        """
        if self._cache is None:
            self.load_from_json()
        return [d for d in self._cache.values() if d.domain == domain]

    def get_by_name(self, tool_name: str) -> Optional[ExternalToolDef]:
        """tool_name으로 ExternalToolDef를 반환한다.

        첫 호출 시 load_from_json()으로 캐시를 초기화한다.

        Args:
            tool_name: 조회할 도구 이름

        Returns:
            ExternalToolDef 인스턴스. 없으면 None.
        """
        if self._cache is None:
            self.load_from_json()
        return self._cache.get(tool_name)


tool_loader = ToolLoader()
