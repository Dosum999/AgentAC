"""도메인별 도구 탐색 메타도구 — LLM 주도 도구 발견 (2단 노출 / discoverable).

설계(A안): core 에는 범용 직접호출 도구를 깔지 않고, **도메인별 탐색 도구**
(``find_messenger_tools`` 등)만 상시 노출한다. LLM 이 작업 도메인에 맞는 탐색
도구를 호출하면 해당 도메인 도구가 발견·활성화되어 다음 턴부터 직접 호출된다.

도메인별로 분리한 이유: 범용 ``tool_search(query, domain)`` 은 LLM 이 domain 을
스스로 채워 호출해야 해 경량 모델이 잘 안 부른다. ``find_messenger_tools`` 처럼
**이름이 곧 의도**면 "채널 요약 → 메신저 도구" 매핑이 자명해 호출률이 높다.
(passive 사전검색 dump 를 함께 쓰면 이 도구가 무용지물 + 잡탕 노이즈 → 제거함.)

발견 → 바인딩 흐름:
    1) LLM 이 find_<도메인>_tools(query) 호출
    2) 통합 인덱스를 그 도메인으로 좁혀 검색 → 실물 인스턴스 resolve →
       권한 게이트 통과분만 ContextVar 대기열 적재
    3) 결과(이름+설명)를 LLM 에 반환
    4) react 루프가 iteration 말미 ``drain_discovered_tools()`` 로 비워
       all_tools 에 머지·rebind → 다음 iteration 부터 네이티브 호출

레퍼런스: open-agent-sdk ToolSearch, openclaw tool-search.
"""
from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from agent_arch.agents.context import get_profile_context
from agent_arch.agents.permissions.gate import evaluate
from agent_arch.agents.permissions.spec import GateMode, PermissionRequirement
from agent_arch.agents.tools.base import BaseTool, ToolResult
from agent_arch.agents.tools.registry import ToolRegistry, register_tool
from agent_arch.core.config import settings

logger = logging.getLogger(__name__)

# 발견 도구 대기열. **리스트를 in-place 변경**해 asyncio.gather 경계를 넘긴다:
# execute_tools 는 도구를 gather(자식 Task)로 실행하는데 Task 는 context 를 복사하므로,
# 자식에서 _discovered.set(새 객체) 한 것은 부모(react 루프)로 전파되지 않는다. 대신
# 부모가 요청 시작 시 reset_discovered_tools() 로 set([]) 해두면, 자식 Task 들은 그
# **동일 리스트 객체**를 물려받아 extend() 로 in-place 적재 → 부모 drain 에서 보인다.
_discovered: ContextVar[Optional[list]] = ContextVar("tool_search_discovered", default=None)

_MAX_TOP_K = 15

# 도메인 → (도구 이름, 한줄요약, 트리거 예시들). 실제 도구가 태깅된 도메인만.
_DOMAIN_TOOLS = {
    "talk": (
        "find_messenger_tools",
        "사내 메신저 — 채널·DM 메시지 조회/요약, 사내 지식·이슈·동향 검색, "
        # ⚠️ **첨부문서 본문 읽기·요약을 반드시 남겨둘 것.** 드라이브에서 같은 실수를 했다
        # (2026-08-12: 광고 문구가 "찾고 여는" 기능만 담아 '내용 뭐야'류 질문이 오지 않았고
        # 에이전트가 파일명만 보고 본문을 지어냈다). 메신저도 이제 첨부 본문을 읽는다.
        "**첨부문서(PDF·한글·워드·엑셀·PPT·이미지 캡처) 본문 읽기·요약**, "
        "**나를 태그·멘션한 메시지 조회**, "
        "실적·매출 등 사내 수치가 오간 대화 검색, "
        "**지금 채널에 올라온 파일 찾기**(방금 올린 파일도 — 색인을 기다리지 않는다), "
        "메시지·DM 전송(**만든 파일(엑셀·워드·마크다운·텍스트)을 첨부해 보내기** 포함), 채널 검색, 리액션, "
        "화면 열기(관리자 콘솔·스레드·임시글 등)",
        ["'○○ 채널 최근 내용 요약해줘'", "'어제 회사에 무슨 일 있었어?'",
         "'나 태그한 메시지 알려줘'", "'지난 일주일간 누가 날 언급했어?'",
          "'그 견적서 내용 요약해줘'", "'공유된 사업계획서 뭐라고 써 있어?'",
         "'최근 논의된 이슈 정리해줘'", "'이번달 실적 어때?'", "'개발팀 채널에 공지 올려줘'",
         "'△△한테 DM 보내줘'", "'○○랑 나눈 DM 보여줘'", "'그 메시지에 👍 달아줘'",
         "'지금 이 채널에서 pdf 찾아줘'", "'만든 엑셀 파일 채널에 올려줘'",
         "'보고서 워드로 만들어서 채널에 올려줘'",
         "'메신저 관리자 콘솔 열어줘'"],
    ),
    "mail": (
        "find_mail_tools",
        "메일 — 메일함/받은편지함 조회, 메일 읽기, 메일 발송, 메일 작성(쓰기) 화면 열기, "
        "연락처(주소록) 저장 — 명함·문서에서 읽은 이름·전화·이메일·회사 등을 주소록에 저장, "
        "화면 열기(받은/보낸메일함·설정·주소록) "
        "(※ 발주서를 공급처에 보내는 메일은 logistics — 발주서 PDF 가 붙어야 한다)",
        ["'안 읽은 메일 있어?'", "'○○한테 제목 △△로 메일 보내줘'", "'이 명함 주소록에 저장해줘'",
         "'○○가 보낸 메일 찾아줘'", "'그 메일 열어줘'", "'연락처로 저장해줘'",
         "'메일 작성 페이지 열어줘'"],
    ),
    "calendar": (
        "find_calendar_tools",
        "일정(캘린더) — 조회, 등록, 수정, 삭제, 캘린더 화면 열기",
        ["'이번 주 일정 보여줘'", "'내일 오후 2시 회의 잡아줘'",
         "'○○ 일정 시간 바꿔줘'", "'그 일정 삭제해줘'", "'캘린더 화면 열어줘'"],
    ),
    "drive": (
        "find_drive_tools",
        # ⚠️ **본문 읽기·내용 검색을 반드시 남겨둘 것.** 드라이브 도구는 전부 retrieved 라
        # 이 문구를 보고 LLM 이 여기를 호출해야 발견·활성화된다. 예전 문구가 "찾고 여는"
        # 기능만 광고해 '내용 뭐야'류 질문이 드라이브로 오지 않았고, 그때 에이전트가
        # 파일명만 보고 본문을 지어냈다(2026-08-12 운영 실측).
        "드라이브(파일) — 폴더 조회, 파일 검색, **파일 본문 읽기·요약**, "
        "**이미지·스캔 문서 읽기**(사진·캡처·스캔 PDF 의 글자 읽기), "
        "**문서 내용 검색**(파일명을 몰라도 내용으로 찾기), 공유 링크 생성, "
        "화면 열기(드라이브 홈)",
        ["'○○ 폴더 보여줘'", "'발주서 파일 찾아줘/열어줘'",
         "'이 파일 내용 뭐야'", "'○○ 문서 요약해줘'", "'○○ 문서에 뭐라고 써 있어'",
         "'드라이브에 있는 그 사진 무슨 내용이야'", "'스캔한 견적서 접수일 언제야'",
         "'품질 관련 자료 어디 있지'",
         "'○○ 공유 링크 만들어줘'", "'드라이브 화면 열어줘'"],
    ),
    "hr": (
        "find_hr_tools",
        "인사/근태 — 근태·출결, 본인에게 배정된 휴가 잔여량, 회사에 정의된 휴가 제도·마스터, "
        "급여명세, 계약, 자산, 사내 직원 검색, "
        "화면 열기(근태·휴가·급여·관리자 화면) "
        "(※ 전자결재에서 신청 가능한 휴가 종류·요건 확인, 휴가 신청·상신, 휴가 신청 문서 "
        "상태·내역은 전자결재를 쓴다 → find_eapproval_tools) "
        "(※ 명함·외부 인물의 연락처를 '저장/추가'하는 것은 여기가 아니라 메일 주소록 → find_mail_tools)",
        ["'내 근태/출결 알려줘'", "'내 휴가 며칠 남았어'", "'회사 휴가 제도 뭐가 있어'",
         "'급여명세 보여줘'", "'○○ 사번/부서 알려줘'", "'HR 근태/관리자 화면 열어줘'"],
    ),
    "eapproval": (
        "find_eapproval_tools",
        "전자결재 — 전자결재에서 신청 가능한 휴가 종류·요건 조회, 휴가신청서 작성·상신과 "
        "휴가 신청 문서 상태·내역 조회, 영수증 기반 경비청구서 작성 등 문서를 작성·상신(제출), "
        "결재 승인/미결 확인, 문서함·결재함 조회. 사내에서 '신청/상신/올리다/제출/경비청구'하는 결재 행위 전반 "
        "(영수증 사진을 올려 경비 처리하는 것 포함). "
        "결재자 이름은 core 디렉터리 search_colleagues 로 조회해 `이름 <email>` 결재선으로 넣는다.",
        ["'보건휴가 신청 가능해?'", "'연차 휴가 신청하고 싶어'", "'이 영수증으로 경비청구 해줘'",
         "'내 휴가 신청 문서 어떻게 됐어?'", "'미결 결재 문서 보여줘'"],
    ),
    "account": (
        "find_account_tools",
        # ⚠️ 서비스 키=도메인 키="account"(Frappe tabService — "accounting" 아님).
        # provider._ACCOUNTING_REPLAY_POLICIES 의 PermissionRequirement(service="account") 와
        # 한 세트 — 도구가 전부 retrieved 라 이 문구에 기능을 광고해야 발견된다(#215 교훈).
        "회계(ERP) — 매입매출전표 입력·등록(세금계산서·건별·카드 거래 전표). "
        "일자·유형·공급가액을 받아 확인 카드로 미리 보여주고 승인하면 실제로 입력된다 "
        "(부가세는 자동 계산, 분개유형 현금/외상/혼합) "
        "(※ 경비청구서 '작성/상신' 자체는 전자결재 → find_eapproval_tools)",
        ["'매입 전표 입력해줘'", "'전표 등록해줘'", "'세금계산서 전표 처리해줘'",
         "'전표 입력해줘'"],
    ),
    # 물류(erp) — 물류 MCP(logistics-mcp) 도구 logistics_* 33종
    # (카탈로그 v2, 2026-08-26 — 단일 출처 logistics-mcp/docs/tool-catalog.md).
    # 일반 대화에서 라우팅되며(관문 enabled:true + supervisor 포함), 노출은 권한
    # 게이트(공통=logistics|mes OR, 물류 전용=logistics.access)가 사용자별로 통제.
    # ⚠️ 설명의 유형 총람을 줄이지 말 것 — 개별 도구는 전부 retrieved 라 LLM 이
    # 이 카드를 보고 재검색 쿼리를 만든다(집계형 질의의 부족분 재발견 경로).
    # ⚠️ "등록은 수주서뿐" 문구를 유지할 것 — 쓰기 도구가 없는데 LLM 이
    # list_stock_adjustments 류의 존재만 보고 "조정 초안을 만들어 드릴게요"라고
    # 없는 능력을 약속한 실측(2026-08-26, WRITE 의도 질의). 2026-10-07 수주서 등록
    # (logistics_create_sales_order, 승인 도구) 도입 — 그 외 변경은 여전히 없다.
    "logistics": (
        "find_logistics_tools",
        "물류/생산(ERP) — 문서 조회: 견적·수주·출고(의뢰/처리/마감)·수금, "
        "구매요청·발주·입고(처리/마감)·지급, 생산(계획/지시/자재불출/생산입고), "
        "외주(불출/입고), 재고(현재고·수불부(일상 입출 이력)·이동·조정·"
        "프로젝트별 현황 / 재고평가·재고자산수불부·재고자산명세는 **결산용** — "
        "기간이 아니라 평가 회차 기준, 일상 수불부와 다른 문서), "
        "채권채무(미수금/미지급 잔액·거래처 원장), 마스터(품목·단가·BOM·창고). "
        "(※ 생산 LOT·공정 실적·설비는 find_mes_tools) "
        "수주서 등록: 고객이 보낸 발주서(PDF·이미지·글로 받은 주문)를 읽어 수주서 1건을 "
        "등록 — 확인 카드로 미리 보여주고 승인하면 실제로 등록된다. "
        "수주서 수정: 등록된 수주서의 품목 수량·납기·단가 변경과 품목 추가(긴급 증량·납기 변경 요청) "
        "— 변경 전→후 카드로 확인받고 승인하면 반영된다. "
        "수주 자재 판정: 수주서 하나를 지금 재고·BOM 으로 다시 판정(품목별 재고·생산 필요, 부족 자재 수량) — "
        "'자재 부족으로 생산불가', '이 수주 자재 충분해?' 는 생산지시 잔량이 아니라 이 판정으로 답한다. "
        "판매 분석·수요 예측: 제품별 판매(출고) 순위·월별 추이·재고일수(잘 팔리는/안 팔리고 쌓이는 제품), "
        "다음 달 수요 예측과 권장 생산·구매량·안전재고·발주 시점, 그 생산량을 BOM 으로 전개한 필요 자재·발주 필요 수량 "
        "— 품목별로 따로 조회하지 말고 이 도구로 한 번에. "
        "발주서 등록: 수주 부족 자재·상품을 공급처별 발주서로 등록(수주 알림 스레드의 '해당건 발주등록') "
        "— 공급처·수량·입고요청일 카드로 확인받고 승인하면 등록된다. "
        "발주서 메일: 등록된 발주서를 공급처에 구매발주서 PDF 첨부 메일로 발송('발주서 메일 보내줘') — "
        "발주서를 보내는 메일은 메일 도메인이 아니라 여기다. "
        "생산지시 변경: 지시수량·완료예정 변경 + 투입자재 재전개, 필요 작업시간 판정('생산지시도 5,000개로 바꿔줘'). "
        "물류 이관(마이그레이션): 드라이브의 입고·출하 내역 엑셀(파일 또는 폴더)을 입고처리·출고처리 문서로 일괄 등록 "
        "('물류 마이그레이션 폴더 엑셀 물류로 마이그레이션 해줘') — 도구가 드라이브 파일을 직접 읽으므로 경로만 넘기고, "
        "건수·기간 카드로 확인받고 승인하면 등록·재고 반영된다. "
        "(※ 변경 도구는 수주서 등록·수정, 발주서 등록·메일, 생산지시 변경, 입고·출하 엑셀 이관뿐 — 수주 줄 삭제·취소, 발주 수정·삭제와 그 밖의 문서 수정·삭제·조정·마감 "
        "도구는 없다. 그런 요청에는 초안·처리를 약속하지 말고 ERP 화면에서 직접 수행하도록 안내)",
        ["'수주 들어온 거 뭐 있어?'", "'○○ 재고 얼마나 있어?'",
         "'발주 나간 거 입고됐어?'", "'○○사 미수금 얼마야?'",
         "'견적 낸 거 수주됐어?'", "'생산지시 뭐 나갔어?'",
         "'이 발주서 수주 등록해줘'", "'첨부한 발주서 수주서에 넣어줘'",
         "'H사 긴급 증량 요청 왔어 수주 수정해줘'", "'해당건으로 발주등록 해줘'", "'발주서 메일 보내줘'",
         "'자재 부족으로 생산불가, 부족 자재 확인해줘'",
         "'어떤 제품이 잘 팔리고 뭐가 재고만 쌓여?'", "'다음 달 수요 예측하고 생산량 얼마로 잡을까?'",
         "'재고 얼마나 쌓아둬야 해? 지금 발주해야 해?'", "'수요 예측대로 생산하려면 자재 얼마나 발주해야 해?'",
         "'물류 마이그레이션 폴더 엑셀 물류로 마이그레이션 해줘'"],
    ),
    # MES(제조현장, erp dams*) — 물류 MCP 의 mes_* 7종. 권한 게이트
    # mes.access. 생산 "문서"(계획·지시·불출·입고)는 물류/MES 공용이라 logistics
    # 도메인에 있다 — 여기는 현장 실행(LOT·공정) 축.
    "mes": (
        "find_mes_tools",
        "MES(제조현장) — 생산 LOT(작업실적) 현황·공정별 실적·불량내역, "
        "공정순서(라우팅·표준시간), 공정/설비/불량유형/검사항목 마스터. "
        "(※ 생산계획·생산지시·자재불출·생산입고 문서와 재고는 find_logistics_tools) "
        "(※ 전부 조회 전용 — 실적 등록·수정 등 변경 작업 도구는 없다. "
        "변경 요청에는 초안·처리를 약속하지 말고 MES 화면에서 직접 수행하도록 안내)",
        ["'오늘 생산 현황/LOT 알려줘'", "'이 LOT 불량 왜 났어?'",
         "'이 제품 공정 뭐 거쳐?'", "'설비 목록 보여줘'"],
    ),
    # MES 앱 PoC 데모 셋(레거시 물류 서비스 연동, mespoc_* — 2026-08-24 격리).
    # MES 앱 채팅/음성 채널(agent_mode=logistics) 전용: 관문은 tools_config 에서
    # enabled:false(일반 채팅 미노출), supervisor 라우팅도 제외(router.py). 앱 채널이
    # apply_logistics_bot 으로 관문·도구를 런타임 주입해서만 쓰인다.
    "mespoc": (
        "find_mespoc_tools",
        "MES 앱 데모(PoC) 물류/생산 스킬 — 생산 현황, 재고 조정, 작업지시, "
        "작업실적 전이, 품질 검사 처리 (레거시 물류 서비스)",
        ["'재고 조정해줘'", "'작업실적 시작 처리'", "'품질 검사 처리'"],
    ),
    # 웹(사외 공개 정보) — 기본 OFF(tools_config find_web_tools.enabled). 사내 데이터는 여기가 아니다:
    # 상시 바인딩 시절 '재고 현황·발주 내역·채널 찾기'를 웹으로 보낸 오호출이 대부분이었다(2026-09-08 실측).
    "web": (
        "find_web_tools",
        "웹(사외 공개 정보) — 날씨·환율·시세, 뉴스·업계 동향, 외부 업체/제품 정보, 법규·규정, 논문, 일반 상식 등 "
        "**회사 밖 공개 웹** 검색(링크·뉴스·페이지 본문 조각까지). "
        "(※ 메일·메신저·재고·발주·근태·결재 등 **사내 데이터는 웹이 아니다** — 해당 도메인 도구를 쓴다)",
        ["'오늘 서울 날씨 어때?'", "'원달러 환율 알려줘'", "'철강 가격 전망 찾아봐'",
         "'○○ 회사가 어떤 회사야?'", "'중대재해처벌법 개정 내용 알려줘'", "'최근 폴리머 시장 동향 정리해줘'"],
    ),
    # 메신저 전용·요청 단위 opt-in(router.domain_available). 꺼진 요청에는 카드가 안 나간다.
    # 회계(전표)·전자결재(상신)와 **가르는 말을 카드에 박는다** — 이 카드가 없을 때 "발주 넣어줘"
    # 가 매입 전표 입력·결재 상신으로 갔다(실측 2026-09-23). 발주요청은 SCMS 로 가는 구매 요청이지
    # 회계 전표도 결재 문서도 아니다.
    "scms_order": (
        "find_scms_order_tools",
        "SCMS 발주요청 — 메신저 대화(스레드)에서 정한 품목·수량·납기로 **SCMS 발주요청서를 작성**하고, "
        "값이 채워진 발주요청 화면 링크를 준다. 스레드에 올라온 도면을 함께 붙인다. "
        "'발주/발주요청/발주 넣어줘/발주 올려줘/발주 등록' 은 **여기다** — 회계 매입 전표 입력(account)이나 "
        "전자결재 상신(eapproval)이 아니다. 도면·PDF 를 첨부하며 발주를 요청해도 **이 도메인 하나만** 고른다 — "
        "첨부 도면은 발주요청 도구가 직접 읽는다(메신저 파일 읽기로 따로 보내지 않는다).",
        ["'SCMS 발주 등록해줘'", "'이거 발주 요청 올려줘'", "'위에 얘기한 금형 발주 넣어줘'",
         "'발주요청서 만들어줘 수량은 2SET'"],
    ),
    # 예약 작업 — ENABLE_SCHEDULED_JOBS 일 때만(router.domain_available). 도구는 schedule_tools() 가 결정적으로 붙인다.
    "schedule": (
        "find_schedule_tools",
        "예약 작업 — 정해진 때마다(매일·평일·매주 ○요일·매월 ○일·특정 시각 한 번) 에이전트가 대신 실행하고 결과를 "
        "알림으로 보내는 예약을 **만들기·목록·삭제**. '매일/평일/매주 ~에 … 알려줘', '내일 9시에 … 알려줘', "
        "'매일 아침 정리해서 보내줘' 처럼 **반복·예약 시각이 붙은 요청은 여기다** — 지금 바로 답하는 요청이 아니다"
        "(반복·시각이 붙었으면 내용이 웹·메일 등 다른 도메인이어도 이 도메인 하나만 고른다). 예약 목록·취소도 여기다. "
        "**반복·시각 표현 없이 내용만 말한 요청(단어 하나, '오늘 ~')은 예약이 아니다** — 내용의 도메인을 고른다.",
        ["'평일 아침 8시에 안 읽은 메일 요약해줘'", "'매주 월요일 아침에 이번 주 일정 정리해줘'",
         "'내 예약 뭐 있어'", "'그 알림 그만해'"],
    ),
    "memory": (
        "find_memory_tools",
        "장기 사용자 메모리 — 사용자가 과거에 밝힌 지시·선호, 별칭(표현→실제 대상), "
        "업무 사실, 지난 작업 이력을 조회. 실시간 값(근태·재고·일정)은 대상 아님.",
        ["'전에 내가 뭐라고 했지'", "'지난번에 부탁한 거'", "'내 선호/설정 뭐였지'",
         "'예전에 얘기한 그거'"],
    ),
}


def reset_discovered_tools() -> None:
    """요청 시작 시 발견 대기열을 새 리스트로 초기화한다 (react 루프가 호출).

    이 리스트 객체가 이후 gather 자식 Task 들에 복사 상속되어 in-place 적재된다.
    """
    _discovered.set([])


def drain_discovered_tools() -> List[BaseTool]:
    """대기열의 발견 도구를 비우고 반환한다 (react 루프 전용). in-place clear."""
    q = _discovered.get()
    if not q:
        return []
    drained = list(q)
    q.clear()  # 자식 Task 들과 공유하는 동일 객체를 비운다
    return drained


def _enqueue(tools: List[BaseTool]) -> None:
    q = _discovered.get()
    if q is None:
        # reset 전 호출(루프 밖 등) — 부모 전파 불가, best-effort 로 새로 set.
        q = []
        _discovered.set(q)
    q.extend(tools)


async def discover_tools(
    query: str, domain: Optional[str], top_k: int, exclude_name: str = ""
) -> ToolResult:
    """인덱스를 (도메인 스코프로) 검색 → 권한 게이트 통과분을 대기열 적재 → 요약 반환.

    도메인별 탐색 도구와 (필요 시) 시스템 안전망이 공유하는 발견 진입점이다.
    """
    # 순환 import 회피 (orchestrator ← tools ← api_registry ← agents)
    from agent_arch.agents.tools.dynamic_api_tool import DynamicAPITool
    from agent_arch.services.api_registry import api_indexer

    try:
        hits = await api_indexer.search(
            query,
            top_k=min(max(top_k, 1), _MAX_TOP_K),
            include_registry=settings.unified_tool_indexing,
            domain=domain,
        )
    except Exception as e:
        return ToolResult(success=False, output=None, error=f"도구 카탈로그 검색 실패: {e}")

    profile = get_profile_context()
    resolved: List[BaseTool] = []
    seen: set[str] = set()
    # 권한 때문에 걸러진 도구가 있었는지 — "기능이 없음"과 "권한 때문에 확인 불가"를
    # 구별해 안내하기 위한 플래그. 어느 서비스였는지는 로그로만 남긴다 (사용자 안내
    # 문구는 LLM 이 호출 맥락에서 스스로 만든다 — 서버가 한국어 기능명을 내려주지 않는다).
    denied_services: set[str] = set()
    for td in hits:
        if td.tool_name == exclude_name or td.tool_name in seen:
            continue
        if td.origin == "registry":
            instance = ToolRegistry.get(td.tool_name)
            if instance is None:
                continue
        else:
            instance = DynamicAPITool(td)
        decision = evaluate(instance.name, profile)  # deny default 일관성
        if not decision.allowed:
            if decision.is_permission_issue:
                denied_services.add(decision.service)
            continue
        resolved.append(instance)
        seen.add(td.tool_name)
    # 인덱스 밖 도메인 도구(기능 플래그·요청 단위 opt-in 으로 붙는 것). 워커에 같이 실리는 관문을
    # 모델이 먼저 불러도 빈손이 되지 않게 관문이 결정적으로 돌려준다.
    from agent_arch.extensions import domain_tools as _ext_domain_tools

    resolved += [t for t in _ext_domain_tools(domain) if t.name not in seen and evaluate(t.name, profile).allowed]

    _enqueue(resolved)

    if not resolved:
        if denied_services:
            # 이 영역의 도구가 권한으로 전부 가려진 상태. **요청한 기능이 실제로
            # 존재하는지는 여기서 알 수 없다** — 인덱스 검색에 관련성 임계값이 없어
            # (indexer.search) hits 는 "맞는 도구"가 아니라 "그 도메인에서 가장 가까운
            # top_k"일 뿐이다. 그래서 존재를 단정하지 않고 "확인 불가"로 안내한다.
            # 조치(관리자 문의)는 기능 유무와 무관하게 동일하므로 사용자가 할 일은 같다.
            logger.info(
                "[tool_search] 권한으로 전량 차단 query=%r domain=%s services=%s",
                query, domain, sorted(denied_services),
            )
            return ToolResult(
                success=True,
                output={
                    "found": 0,
                    "reason": "permission_blocked",
                    "message": (
                        "이 영역의 도구가 접근 권한 때문에 가려져 있어, 요청한 기능이 "
                        "있는지 여기서는 확인할 수 없습니다. **'기능이 없다'·'지원하지 "
                        "않는다'고 단정하지도, '기능이 있다'고 단정하지도 마세요.** "
                        "해당 영역에 접근 권한이 없어 확인·이용할 수 없다는 점과, 담당 "
                        "관리자에게 권한 부여를 요청하면 확인할 수 있다는 점을 안내하세요. "
                        "어떤 영역인지는 사용자의 요청 맥락에 맞는 한국어 표현으로만 쓰고, "
                        "도구명·내부 식별자는 노출하지 마세요."
                    ),
                },
            )
        return ToolResult(
            success=True,
            output={
                "found": 0,
                "message": (
                    f"'{query}'에 해당하는 사용 가능한 도구가 없습니다. "
                    "다른 검색어로 재시도하거나, 기능이 없음을 사용자에게 안내하세요."
                ),
            },
        )

    if denied_services:
        # 일부만 차단 — 찾은 도구로 진행하면 되므로 LLM 에 알리지 않는다 (노이즈).
        logger.info(
            "[tool_search] 권한으로 일부 차단 query=%r domain=%s services=%s found=%d",
            query, domain, sorted(denied_services), len(resolved),
        )
    return ToolResult(
        success=True,
        output={
            "found": len(resolved),
            "message": (
                "아래 도구가 활성화되었습니다 — 다음 응답에서 직접 호출하세요. "
                "(내부 준비 정보입니다: 이 목록과 도구명을 사용자에게 나열·노출하거나 "
                "'도구가 활성화되었다'고 알리지 말고, 바로 작업을 진행하거나 "
                "한국어 기능명으로만 설명하세요.)"
            ),
            "tools": [
                {"name": t.name, "description": (t.description or "")[:300]}
                for t in resolved
            ],
        },
    )


class _DomainToolSearch(BaseTool):
    """도메인 고정 도구 탐색기. 서브클래스가 _domain/_tool_name/_intent 를 채운다."""

    permission_required = PermissionRequirement(service="", mode=GateMode.PUBLIC)
    exposure = "core"  # 상시 노출
    _domain: str = ""
    _tool_name: str = ""
    _intent: str = ""
    _examples: tuple = ()

    def get_name(self) -> str:
        return self._tool_name

    def get_description(self) -> str:
        ex = "  ".join(self._examples)
        return (
            f"**목적:** {self._intent} 관련 기능(도구)을 찾아 활성화한다.\n"
            f"**이럴 때 호출:** {ex}\n"
            "**규칙:** 위 영역의 작업인데 호출할 도구가 지금 목록에 없으면, "
            "\"기능이 없다\"고 답하기 전에 **반드시 먼저 이 도구로 찾을 것.** "
            "검색된 도구는 즉시 활성화되어 다음 응답에서 직접 호출할 수 있다."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "query": {
                "type": "string",
                "description": "찾으려는 기능의 자연어 설명 (예: '채널 메시지 조회', '휴가 신청')",
            },
            "top_k": {
                "type": "integer",
                "description": f"최대 결과 수 (기본 8, 최대 {_MAX_TOP_K})",
                "default": 8,
            },
        }

    def get_required_params(self) -> List[str]:
        return ["query"]

    async def _demo_execute(self, **kwargs) -> ToolResult:
        return ToolResult(
            success=True,
            output={"message": f"[DEMO] {self._tool_name} 호출 (모의 — 발견 없음)", "params": kwargs},
        )

    async def _real_execute(self, query: str, top_k: int = 8, **kwargs) -> ToolResult:
        return await discover_tools(query, self._domain, top_k, exclude_name=self.name)


def _build_domain_search_tools() -> None:
    """도메인별 탐색 도구를 동적 생성·등록한다 (find_messenger_tools 등)."""
    for domain, (tool_name, intent, examples) in _DOMAIN_TOOLS.items():
        cls = type(
            f"FindTools_{domain}",
            (_DomainToolSearch,),
            {
                "_domain": domain,
                "_tool_name": tool_name,
                "_intent": intent,
                "_examples": tuple(examples),
            },
        )
        register_tool(cls)


_build_domain_search_tools()
