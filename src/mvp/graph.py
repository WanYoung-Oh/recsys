"""
LangGraph StateGraph 조립 + 전역 컴파일.

모듈 전역에서 1회 컴파일 → app (MemorySaver 포함).
세션 격리: app.invoke(..., config={"configurable": {"thread_id": "..."}})
"""
from __future__ import annotations

import sys
from pathlib import Path

# 어디서 실행하든 프로젝트 루트를 sys.path에 추가
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.mvp.graph_state import GraphState
from src.mvp.nodes import (
    alias_resolver,
    ask_user_id,
    context_builder,
    general_chat,
    intent_router,
    profile_loader,
    rec_engine,
    route_after_alias,
    route_after_intent,
    route_after_profile,
    solar_explainer,
)

# ── 그래프 조립 ───────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(GraphState)

    g.add_node("intent_router",   intent_router)
    g.add_node("alias_resolver",  alias_resolver)
    g.add_node("profile_loader",  profile_loader)
    g.add_node("rec_engine",      rec_engine)
    g.add_node("context_builder", context_builder)
    g.add_node("solar_explainer", solar_explainer)
    g.add_node("general_chat",    general_chat)
    g.add_node("ask_user_id",     ask_user_id)

    g.set_entry_point("intent_router")

    g.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {"alias_resolver": "alias_resolver", "general_chat": "general_chat"},
    )
    g.add_conditional_edges(
        "alias_resolver",
        route_after_alias,
        {"profile_loader": "profile_loader", "ask_user_id": "ask_user_id"},
    )
    g.add_conditional_edges(
        "profile_loader",
        route_after_profile,
        {"rec_engine": "rec_engine", "__end__": END},
    )

    g.add_edge("rec_engine",      "context_builder")
    g.add_edge("context_builder", "solar_explainer")
    g.add_edge("solar_explainer", END)
    g.add_edge("general_chat",    END)
    g.add_edge("ask_user_id",     END)

    return g


# ── 전역 컴파일 (Streamlit rerun에도 유지) ────────────────────────────────────
app = _build_graph().compile(checkpointer=MemorySaver())


# ── E2E 테스트 ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid

    SEP = "─" * 60

    def run(message: str, cfg: dict, label: str = "") -> dict:
        print(f"\n{SEP}")
        if label:
            print(f"[{label}]")
        print(f"입력: {message!r}")
        result = app.invoke({"message": message}, config=cfg)
        response = result.get("response", "(응답 없음)")
        print(f"intent  : {result.get('intent')}")
        print(f"user_alias: {result.get('user_alias')}")
        print(f"trust   : {result.get('trust_passed')}")
        print(f"응답 (앞 400자):\n{response[:400]}")
        return result

    # ── 시나리오 1: 쇼핑 추천 (user_alias 직접 입력) ──────────────────────────
    cfg1 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run("user_00004 상품 추천해줘", cfg1, "쇼핑 추천 — user_alias 직접 입력")

    # ── 시나리오 2: 멀티턴 — 같은 thread_id에서 user_alias 없이 후속 질문 ────
    run("추천 상품 더 보여줘", cfg1, "멀티턴 — user_alias 재입력 없이 추천 유지")

    # ── 시나리오 3: user_alias 없이 쇼핑 요청 → alias 요청 메시지 ───────────
    cfg2 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run("오늘 신발 추천해줘", cfg2, "쇼핑 요청 — user_alias 없음")

    # ── 시나리오 4: 일반 대화 ─────────────────────────────────────────────────
    cfg3 = {"configurable": {"thread_id": str(uuid.uuid4())}}
    run("날씨가 좋네요, 요즘 어때요?", cfg3, "일반 대화 (⚠️ 미검증 배지)")

    print(f"\n{SEP}")
    print("E2E 테스트 완료")
