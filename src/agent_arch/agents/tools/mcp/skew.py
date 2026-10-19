"""MCP 도구 목록 **버전 불일치(skew)** 판정 — 배포 순서가 어긋났을 때를 잡는다.

왜 필요한가 (2026-09-15 실측, 이슈)
---------------------------------------------
같은 배포에서 백엔드가 MCP 서버보다 **1분 먼저** 떴다::

    07:28:25  agent-app 기동
    07:28:32  [mcp:mattermost] 정책 매핑 없는 도구 'mattermost_skill_find_my_posts' — 건너뜀
    07:29:21  agent-mattermost-mcp 기동 (도구 이름이 바뀐 새 버전)

백엔드는 **옛 목록**을 성공적으로 받았다. 조회가 실패한 게 아니라 성공했기 때문에 재시도
로직이 돌지 않았고, 그 뒤 MCP 가 새 이름으로 떠도 백엔드는 영영 다시 묻지 않았다. 결과는
조용한 고장이다 — 모델은 프롬프트가 가르쳐 준 새 도구를 정확히 골랐는데 실행 단계에서
``tool NOT FOUND`` 로 튕겼고, 사용자에겐 "엉뚱한 답"으로만 보였다.

판정 규칙
---------
정책표에 있는 이름이 **서버 노출 목록에도 없고 로컬에도 등록돼 있지 않으면** 그 서버는
우리가 기대하는 버전이 아니다. 로컬 등록분을 빼는 이유는 선점 래퍼 때문이다 — 같은 이름을
로컬 도구가 먼저 차지한 경우(``mattermost_post_message`` 등)는 정상이며 불일치가 아니다.

이 판정만 분리해 둔 것은 **테스트 때문**이다. ``provider`` 는 MCP 전송 계층을 끌고 들어와
일부 환경에서 import 자체가 실패한다(설치된 ``mcp`` 패키지 버전 차이). 조용한 고장을 막는
판정이 그 사정 때문에 검증 불가가 되면 안 된다.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional


class MCPToolListSkew(RuntimeError):
    """기대한 도구가 서버 노출 목록에 없다 — 아직 옛 버전이 떠 있다는 뜻.

    호출부(lifespan 재시도 래퍼)가 이 예외를 받아 백오프로 다시 조회한다. 상대가 새 버전으로
    올라오면 다음 시도에서 등록되고, 끝내 안 올라오면 재시도가 소진되며 로그가 남는다.
    """

    def __init__(self, server: str, missing: List[str]):
        self.server = server
        self.missing = missing
        super().__init__(
            f"[mcp:{server}] 기대한 도구 {len(missing)}개가 서버 목록에 없습니다: "
            f"{', '.join(missing)} — 아직 옛 버전이 떠 있을 수 있어 재조회합니다."
        )


def missing_policy_tools(
    policy_names: Iterable[str],
    exposed_names: Iterable[str],
    *,
    is_registered: Optional[Callable[[str], bool]] = None,
) -> List[str]:
    """정책이 기대하는데 서버가 안 내놓는 도구 이름들(정렬).

    Args:
        policy_names: 이 서버에 대해 우리가 알고 있는 도구 이름(정책표 키).
        exposed_names: 서버가 ``list_tools`` 로 내놓은 이름.
        is_registered: 이름이 이미 로컬 레지스트리에 있는지 판정하는 함수. 선점 래퍼가
            차지한 이름은 불일치가 아니므로 제외한다. 생략하면 아무것도 제외하지 않는다.

    Returns:
        불일치로 볼 이름 목록. 비어 있으면 기대한 도구가 전부 있다는 뜻.
    """
    exposed = set(exposed_names)
    taken = is_registered or (lambda _name: False)
    return sorted(n for n in set(policy_names) if n not in exposed and not taken(n))
