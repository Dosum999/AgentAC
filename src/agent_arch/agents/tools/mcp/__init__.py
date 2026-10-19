"""원격 MCP 서버 도구 연동 레이어.

MCP(Model Context Protocol) 서버가 노출하는 도구를 ``BaseTool``로 래핑해
``ToolRegistry``에 등록한다. 정적 스킬·동적 API 도구에 이은 세 번째 도구
소스이며, 등록 이후의 권한 게이트·tools_config 활성화·LLM 바인딩은 기존
메커니즘을 그대로 탄다.

현재 연결 대상은 mailcow 메일 서버(mailcow-mcp/) 하나지만, 어댑터 자체는
서버 비종속이라 추후 다른 MCP 서버도 provider만 추가해 연결한다.
"""
from agent_arch.agents.tools.mcp.remote_tool import RemoteMCPTool

__all__ = ["RemoteMCPTool"]
