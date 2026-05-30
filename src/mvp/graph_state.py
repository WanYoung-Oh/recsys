"""LangGraph 상태 스키마."""
from __future__ import annotations

from typing import Literal, Optional
from typing_extensions import TypedDict


class GraphState(TypedDict, total=False):
    # 입력
    message: str

    # 의도 분류
    intent: Optional[Literal["shopping", "general", "user_alias"]]
    user_alias: Optional[str]
    needs_user_id: bool
    korean_name_detected: bool

    # RAG / 추천
    profile: Optional[dict]
    candidates: Optional[dict]
    context_text: Optional[str]
    requested_category: Optional[str]    # 메시지에서 추출한 요청 카테고리 (예: "shirt")
    requested_price_filter: Optional[str] # "cheaper" | "pricier" | None
    requested_section: Optional[str]     # "recency" | "revisit" | None(=content 우선)

    # Solar Pro 응답
    explanation: Optional[dict]   # {sections: {top10: [...], content: [...], ...}}
    trust_passed: Optional[bool]  # hard_gate 통과 여부
    groundedness: Optional[float] # SelfCheckGPT 점수 (0~1, None=미실행)

    # 최종 텍스트 응답 (chat display용)
    response: Optional[str]
