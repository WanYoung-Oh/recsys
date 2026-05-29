"""
LangGraph 노드 함수 전체.

노드 목록:
  intent_router  → 의도 분류 (키워드 우선 → Solar Pro fallback)
  alias_resolver → user_alias 파싱 / 멀티턴 유지
  profile_loader → SQLite O(1) 프로필 조회
  rec_engine     → 4종 추천 후보 생성
  context_builder→ Solar Pro 주입 컨텍스트 조립
  solar_explainer→ Solar Pro 사유 생성 + hard_gate 검증
  general_chat   → 일반 대화 (⚠️ 미검증 배지)
  ask_user_id    → alias 입력 요청 메시지
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from src.mvp.graph_state import GraphState
from src.mvp.solar_client import solar
from src.mvp.trust_gate import hard_gate
from src.mvp.recommenders import build_all_candidates
from src.mvp.category_ko import ko_cat

ROOT = Path(__file__).resolve().parents[2]

SHOPPING_KEYWORDS = frozenset({
    "추천", "상품", "쇼핑", "구매", "브랜드", "뭐 살까", "골라줘",
    "어울리는", "왜 추천", "살까", "더 싸", "취향", "신상",
    "보여줘", "알려줘", "추천해", "뭐가 좋아",
    "더 저렴", "저렴한", "가성비", "고급", "프리미엄",
})
_USER_ALIAS_PAT = re.compile(r"\buser_\d+\b")

# 카테고리 키워드 → L2 매핑 (긴 키워드 먼저 매칭되도록 정렬해 사용)
_CAT_KEYWORDS: dict[str, str] = {
    # 신발 계열
    "운동화": "shoes", "스니커즈": "shoes", "슬리퍼": "shoes",
    "샌들": "shoes", "부츠": "shoes", "구두": "shoes", "신발": "shoes",
    # 상의 계열
    "티셔츠": "tshirt",                      # "셔츠" 보다 먼저
    "셔츠": "shirt", "블라우스": "shirt",
    # 하의 계열
    "청바지": "jeans", "진바지": "jeans",     # "바지" 보다 먼저
    "반바지": "shorts", "쇼츠": "shorts",     # "바지" 보다 먼저
    "바지": "trousers", "슬랙스": "trousers",
    "치마": "skirt", "스커트": "skirt",
    # 아우터
    "재킷": "jacket", "자켓": "jacket",
    "점퍼": "jumper", "후드": "jumper",
    # 기타
    "원피스": "dress", "드레스": "dress",
    "파자마": "pajamas", "잠옷": "pajamas",
    "속옷": "underwear", "내복": "underwear",
    "스카프": "scarf", "목도리": "scarf",
    "장갑": "glove",
    "양말": "sock",
    "벨트": "belt",
    "의상": "costume", "코스튬": "costume",
}
# 긴 키워드 우선 적용을 위해 길이 내림차순 정렬
_CAT_KEYWORDS_SORTED = sorted(_CAT_KEYWORDS.items(), key=lambda x: -len(x[0]))


def _extract_requested_category(message: str) -> str | None:
    """메시지에서 카테고리 키워드 추출 → L2 코드 반환. 없으면 None."""
    for kw, l2 in _CAT_KEYWORDS_SORTED:
        if kw in message:
            return l2
    return None


_PRICE_CHEAPER = ("더 싸", "더 저렴", "저렴한", "싸게", "저가", "가성비", "싼", "cheap")
_PRICE_PRICIER = ("더 비싸", "고급", "프리미엄", "비싼", "럭셔리", "하이엔드")


def _extract_price_filter(message: str) -> str | None:
    """메시지에서 가격 방향 추출. 'cheaper' | 'pricier' | None."""
    for kw in _PRICE_CHEAPER:
        if kw in message:
            return "cheaper"
    for kw in _PRICE_PRICIER:
        if kw in message:
            return "pricier"
    return None

# ── 모듈 전역 캐시 (Streamlit rerun에도 유지) ──────────────────────────────────
_STORE: dict = {}


def _store() -> dict:
    """RAG 데이터 lazy 로드 (최초 1회만 파일 읽기).

    user_alias.json (69.3 MB)은 로드하지 않음.
    FAISS 파일 존재 시 neighbors_data 자동 로드.
    """
    if _STORE:
        return _STORE
    rag = ROOT / "rag_data"
    _STORE["item_catalog"] = json.loads((rag / "item_catalog.json").read_text(encoding="utf-8"))
    _STORE["user_recs"]    = json.loads((rag / "user_recommendations.json").read_text(encoding="utf-8"))
    _STORE["recency_pool"] = json.loads((rag / "recency_pool.json").read_text(encoding="utf-8"))

    # FAISS 파일이 있으면 CF 활성화
    npy_path  = rag / "user_neighbors.npy"
    meta_path = rag / "user_neighbors_meta.pkl"
    row_path  = rag / "user_alias_to_row.pkl"
    if npy_path.exists() and meta_path.exists() and row_path.exists():
        import pickle
        import numpy as np
        neighbors_npy  = np.load(npy_path)
        with open(meta_path, "rb") as f:
            row_to_alias = pickle.load(f)
        with open(row_path, "rb") as f:
            alias_to_row = pickle.load(f)

        # 이웃 프로필 일괄 로드 (DB에 있는 유저만)
        neighbor_profiles: dict[str, dict] = {}
        with sqlite3.connect(_db_path()) as con:
            rows = con.execute(
                "SELECT user_alias, profile_json FROM user_profiles"
            ).fetchall()
        for ua, pj in rows:
            p = json.loads(pj)
            p["user_alias"] = ua
            neighbor_profiles[ua] = p

        _STORE["neighbors_data"] = {
            "neighbors_npy":     neighbors_npy,
            "row_to_alias":      row_to_alias,
            "alias_to_row":      alias_to_row,
            "neighbor_profiles": neighbor_profiles,
        }
    else:
        _STORE["neighbors_data"] = None   # FAISS 미빌드

    return _STORE


def _get_display_name(user_alias: str) -> str:
    """user_alias → 한국어 display_name. DB에서 O(1) 조회; 없으면 alias 그대로."""
    try:
        with sqlite3.connect(_db_path()) as con:
            # profile_text 첫 줄에서 이름 파싱: "[사용자 프로필 — 김민지 (user_00001)]"
            row = con.execute(
                "SELECT profile_text FROM user_profiles WHERE user_alias=?",
                (user_alias,),
            ).fetchone()
        if row and row[0]:
            first_line = row[0].split("\n")[0]   # "[사용자 프로필 — 김민지 (user_00001)]"
            m = re.search(r"— (.+?) \(", first_line)
            if m:
                return m.group(1)
    except Exception:
        pass
    return user_alias


def _db_path() -> Path:
    return ROOT / "rag_data" / "user_profiles.db"


# ── 노드 ──────────────────────────────────────────────────────────────────────

def intent_router(state: GraphState) -> GraphState:
    msg = state.get("message", "")

    # 요청 카테고리·가격 방향 추출 (모든 경로에서 공통 처리)
    state["requested_category"]    = _extract_requested_category(msg)
    state["requested_price_filter"] = _extract_price_filter(msg)

    # 0순위: user_alias 이미 있고 쇼핑 메시지 → 즉시 shopping (ID 재요청 없음)
    if state.get("user_alias") and any(kw in msg for kw in SHOPPING_KEYWORDS):
        state["intent"] = "shopping"
        return state

    # 1순위: user_alias 패턴 직접 감지 → API 호출 없이
    if _USER_ALIAS_PAT.search(msg):
        state["intent"] = "user_alias"
        return state

    # 2순위: 쇼핑 키워드 매칭 → API 호출 없이
    if any(kw in msg for kw in SHOPPING_KEYWORDS):
        state["intent"] = "shopping"
        return state

    # 3순위: Solar Pro 3-class 분류 (키워드 미매칭 시만)
    state["intent"] = solar.classify_intent(msg)
    return state


def alias_resolver(state: GraphState) -> GraphState:
    msg = state.get("message", "")
    match = _USER_ALIAS_PAT.search(msg)

    if match:
        state["user_alias"] = match.group(0)
        state["needs_user_id"] = False
        return state

    # 멀티턴: 이전 턴에서 이미 user_alias가 있으면 유지
    if state.get("user_alias"):
        state["needs_user_id"] = False
        return state

    state["needs_user_id"] = True
    state["korean_name_detected"] = bool(re.search(r"[가-힣]{2,4}", msg))
    return state


def profile_loader(state: GraphState) -> GraphState:
    user_alias = state.get("user_alias")
    if not user_alias:
        state["response"] = "유저 ID를 확인할 수 없습니다. 다시 입력해 주세요."
        return state

    db = _db_path()
    if not db.exists():
        state["response"] = "프로필 DB가 없습니다. build_rag_index.py를 먼저 실행해 주세요."
        return state

    try:
        with sqlite3.connect(db) as con:
            row = con.execute(
                "SELECT profile_json, profile_text FROM user_profiles WHERE user_alias = ?",
                (user_alias,),
            ).fetchone()
    except Exception as e:
        state["response"] = f"프로필 조회 오류: {e}"
        return state

    if not row:
        state["response"] = (
            f"'{user_alias}' 프로필을 찾을 수 없습니다. "
            "샘플 빌드 범위 밖 유저이거나 build_rag_index.py --sample 옵션 확인 필요."
        )
        return state

    profile = json.loads(row[0])
    profile["user_alias"]   = user_alias
    profile["profile_text"] = row[1]
    profile["display_name"] = _get_display_name(user_alias)
    state["profile"] = profile
    return state


def rec_engine(state: GraphState) -> GraphState:
    profile = state.get("profile")
    if not profile:
        return state

    s = _store()

    # 카테고리 오버라이드
    req_cat = state.get("requested_category")
    if req_cat:
        effective_profile = dict(profile)
        effective_profile["top_categories_l2"] = [req_cat]
    else:
        effective_profile = profile

    # 가격 필터: MemorySaver로 보존된 이전 candidates에서 reference 가격 계산
    price_cap = price_floor = None
    price_filter = state.get("requested_price_filter")
    if price_filter:
        catalog = s["item_catalog"]
        prev_top10 = (state.get("candidates") or {}).get("top10") or []
        prices = [
            catalog[a]["price_median"]
            for a in prev_top10[:5]
            if a in catalog and catalog[a].get("price_median")
        ]
        if not prices:
            pr = profile.get("price_range", {})
            ref = pr.get("median") or pr.get("p50") or 0.0
        else:
            prices.sort()
            ref = prices[len(prices) // 2]

        if ref > 0:
            if price_filter == "cheaper":
                price_cap = ref * 0.8
            else:
                price_floor = ref * 1.2

    state["candidates"] = build_all_candidates(
        profile=effective_profile,
        user_alias=profile["user_alias"],
        user_recs=s["user_recs"],
        item_catalog=s["item_catalog"],
        recency_pool=s["recency_pool"],
        neighbors_data=s.get("neighbors_data"),
        price_cap=price_cap,
        price_floor=price_floor,
    )
    return state


def context_builder(state: GraphState) -> GraphState:
    profile    = state.get("profile", {})
    candidates = state.get("candidates", {})
    catalog    = _store()["item_catalog"]

    _LABELS = {
        "top10":   "모델 추천 Top-10",
        "content": "내 취향",
        "recency": "새로 나온",
        "revisit": "다시 볼 만한",
    }

    # 섹션당 상위 1개만 설명 요청 — 생성 토큰 최소화로 API 응답 속도 개선
    _EXPLAIN_SECTIONS = ("content", "recency", "revisit")  # top10은 설명 불필요(대시보드 표시용)
    sections: list[str] = []
    for key in _EXPLAIN_SECTIONS:
        label = _LABELS.get(key, key)
        items = candidates.get(key, [])
        if not items:
            continue
        alias = items[0]
        meta  = catalog.get(alias, {})
        cat_str = ko_cat(meta.get("l2", ""), meta.get("l3"))
        sections.append(
            f"\n[{label}]\n"
            f"- {alias} | 카테고리: {cat_str} | "
            f"브랜드: {meta.get('brand', '')} | "
            f"가격: {meta.get('price_display', '')}"
        )

    display    = profile.get("display_name", profile.get("user_alias", ""))
    user_alias = profile.get("user_alias", "")
    req_cat    = state.get("requested_category")

    # 요청 조건 Solar Pro에 명시
    req_note = ""
    if req_cat:
        req_note += f"\n[요청 카테고리: {ko_cat(req_cat)} — 이 카테고리 위주로 사유 작성]"
    price_filter = state.get("requested_price_filter")
    if price_filter == "cheaper":
        req_note += "\n[가격 조건: 이전 추천보다 저렴 — 가격 메리트를 사유에 언급하세요]"
    elif price_filter == "pricier":
        req_note += "\n[가격 조건: 이전 추천보다 고가 — 프리미엄 가치를 사유에 언급하세요]"

    state["context_text"] = (
        f"[유저: {display} ({user_alias})]\n"
        f"{profile.get('profile_text', '')}"
        f"{req_note}\n"
        f"\n[추천 후보]"
        + "".join(sections)
        + "\n\n위 후보 상품에 대해 섹션별 추천 사유를 JSON으로 작성하세요."
    )
    return state


def solar_explainer(state: GraphState) -> GraphState:
    context_text = state.get("context_text", "")
    candidates   = state.get("candidates", {})
    catalog      = _store()["item_catalog"]
    profile      = state.get("profile", {})

    explanation = solar.explain_recommendations(context_text)
    passed, reason = hard_gate(explanation, candidates)

    if not passed:
        state["explanation"]  = None
        state["trust_passed"] = False
        state["response"]     = _format_fallback(candidates, catalog, reason)
        return state

    state["explanation"]  = explanation
    state["trust_passed"] = True
    state["response"]     = _format_response(explanation, candidates, profile, catalog)
    return state


def general_chat(state: GraphState) -> GraphState:
    msg = state.get("message", "")
    state["trust_passed"] = False
    state["response"]     = solar.general_chat(msg)
    return state


def ask_user_id(state: GraphState) -> GraphState:
    if state.get("korean_name_detected"):
        state["response"] = (
            "죄송해요, 이름만으로는 정확한 유저를 특정하기 어렵습니다. "
            "사이드바 드롭다운에서 선택하시거나 'user_00001' 형식으로 입력해 주세요."
        )
    else:
        state["response"] = (
            "추천을 위해 유저 ID가 필요합니다. "
            "'user_00001' 형식으로 입력하거나 사이드바에서 선택해 주세요."
        )
    return state


# ── 응답 포맷터 ────────────────────────────────────────────────────────────────

_SEC_EMOJI = {
    "top10":   "모델 추천 Top-10",
    "content": "내 취향",
    "recency": "새로 나온",
    "revisit": "다시 볼 만한",
}


def _item_line(alias: str, catalog: dict) -> str:
    meta = catalog.get(alias, {})
    price = meta.get("price_display", "")
    return f"{alias}  {price}" if price else alias


def _format_response(
    explanation: dict,
    candidates: dict,
    profile: dict,
    catalog: dict,
) -> str:
    display    = profile.get("display_name", profile.get("user_alias", ""))
    user_alias = profile.get("user_alias", "")
    lines      = [f"[추천 결과 — {display} ({user_alias})]\n"]

    sections = explanation.get("sections", {})
    for key, emoji in _SEC_EMOJI.items():
        cand_items = candidates.get(key, [])
        if not cand_items:
            continue
        llm_items  = sections.get(key, [])
        explained  = {
            i["item_alias"]: i["reason"]
            for i in llm_items
            if isinstance(i, dict) and "item_alias" in i
        }
        lines.append(f"{emoji}:")
        for alias in cand_items[:5]:
            lines.append(f"  • {_item_line(alias, catalog)}")
            reason = explained.get(alias, "")
            if reason:
                lines.append(f"    {reason}")
        lines.append("")

    return "\n".join(lines)


def _format_fallback(candidates: dict, catalog: dict, gate_reason: str) -> str:
    lines = [f"[추천 결과] 설명 검증 실패 ({gate_reason}) — 후보 목록만 표시\n"]
    for key, emoji in _SEC_EMOJI.items():
        items = candidates.get(key, [])
        if not items:
            continue
        lines.append(f"{emoji}:")
        for alias in items[:5]:
            lines.append(f"  • {_item_line(alias, catalog)}")
        lines.append("")
    return "\n".join(lines)


# ── 조건부 엣지 라우터 ─────────────────────────────────────────────────────────

def route_after_intent(state: GraphState) -> str:
    return "general_chat" if state.get("intent") == "general" else "alias_resolver"


def route_after_alias(state: GraphState) -> str:
    return "ask_user_id" if state.get("needs_user_id") else "profile_loader"


def route_after_profile(state: GraphState) -> str:
    # profile 없으면 response 이미 설정 → END 직행, API 낭비 방지
    return "rec_engine" if state.get("profile") else "__end__"
