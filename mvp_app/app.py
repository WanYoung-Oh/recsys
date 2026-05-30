"""
LLM 커머스 추천 MVP — Streamlit 앱

실행:
    streamlit run mvp_app/app.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import uuid
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mvp.graph import app as _lg_app  # 모듈 전역 컴파일 인스턴스
from src.mvp.category_ko import ko_cat
from src.mvp.self_check import self_check as _self_check
from src.mvp.solar_client import solar as _solar

RAG = ROOT / "rag_data"

# ── 페이지 설정 ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LLM 커머스 추천 MVP",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 데이터 캐시 ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_catalog() -> dict:
    return json.loads((RAG / "item_catalog.json").read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def _load_user_recs() -> dict:
    return json.loads((RAG / "user_recommendations.json").read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def _load_users() -> list[tuple[str, str]]:
    """[(user_alias, 'display_name (user_alias)'), ...] — DB 유저만."""
    alias_map: dict = json.loads(
        (RAG / "id_aliases" / "user_alias.json").read_text(encoding="utf-8")
    )
    alias_to_name = {v["user_alias"]: v["display_name"] for v in alias_map.values()}
    con = sqlite3.connect(RAG / "user_profiles.db")
    rows = con.execute("SELECT user_alias FROM user_profiles ORDER BY user_alias").fetchall()
    con.close()
    return [(ua, f"{alias_to_name.get(ua, ua)} ({ua})") for (ua,) in rows]

@st.cache_data(show_spinner=False)
def _load_profile(user_alias: str) -> dict | None:
    con = sqlite3.connect(RAG / "user_profiles.db")
    row = con.execute(
        "SELECT profile_json, profile_text FROM user_profiles WHERE user_alias=?",
        (user_alias,),
    ).fetchone()
    con.close()
    if not row:
        return None
    p = json.loads(row[0])
    p["profile_text"] = row[1]
    return p

@st.cache_data(show_spinner=False)
def _load_recency_pool() -> dict:
    return json.loads((RAG / "recency_pool.json").read_text(encoding="utf-8"))

@st.cache_data(show_spinner=False)
def _load_eval_report() -> dict | None:
    """rag_data/eval_report.json 로드 (없으면 None)."""
    path = RAG / "eval_report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

@st.cache_data(show_spinner=False)
def _load_calibration_temperature() -> float | None:
    """rag_data/calibration.json에서 temperature 로드."""
    path = RAG / "calibration.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("temperature")

@st.cache_resource(show_spinner=False)
def _load_neighbors_data() -> dict | None:
    """FAISS 파일 존재 시 neighbors_data 로드 (없으면 None)."""
    import pickle, numpy as np
    npy  = RAG / "user_neighbors.npy"
    meta = RAG / "user_neighbors_meta.pkl"
    row  = RAG / "user_alias_to_row.pkl"
    if not (npy.exists() and meta.exists() and row.exists()):
        return None
    import sqlite3
    row_to_alias = pickle.load(open(meta, "rb"))
    needed = list(row_to_alias.values())
    placeholders = ",".join("?" * len(needed))
    neighbor_profiles: dict[str, dict] = {}
    con = sqlite3.connect(RAG / "user_profiles.db")
    for ua, pj in con.execute(
        f"SELECT user_alias, profile_json FROM user_profiles"
        f" WHERE user_alias IN ({placeholders})",
        needed,
    ).fetchall():
        p = json.loads(pj); p["user_alias"] = ua
        neighbor_profiles[ua] = p
    con.close()
    return {
        "neighbors_npy":     np.load(npy),
        "row_to_alias":      row_to_alias,
        "alias_to_row":      pickle.load(open(row,  "rb")),
        "neighbor_profiles": neighbor_profiles,
    }

@st.cache_data(show_spinner=False)
def _compute_candidates(user_alias: str) -> dict | None:
    """유저 선택 즉시 추천 후보 계산 — Solar Pro 없이 Python 엔진만 실행."""
    from src.mvp.recommenders import build_all_candidates
    profile = _load_profile(user_alias)
    if not profile:
        return None
    p = dict(profile)
    p["user_alias"] = user_alias
    return build_all_candidates(
        profile=p,
        user_alias=user_alias,
        user_recs=_load_user_recs(),
        item_catalog=_load_catalog(),
        recency_pool=_load_recency_pool(),
        neighbors_data=_load_neighbors_data(),   # FAISS CF 활성화
    )

# ── LangGraph ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_app():
    return _lg_app

# ── 유틸 ──────────────────────────────────────────────────────────────────────
def _price_in_band(price: float, pr: dict) -> bool:
    iqr = pr["p75"] - pr["p25"]
    return (pr["p25"] - 0.5 * iqr) <= price <= (pr["p75"] + 0.5 * iqr)

def _evidence_chips(alias: str, catalog: dict, profile: dict) -> list[str]:
    meta    = catalog.get(alias, {})
    chips: list[str] = []
    pr      = profile.get("price_range", {})
    if pr and _price_in_band(meta.get("price_median", 0.0), pr):
        chips.append("💵 가격대")
    if meta.get("brand") in profile.get("top_brands", []):
        chips.append("🧭 선호 브랜드")
    elif meta.get("l2") in profile.get("top_categories_l2", []):
        chips.append("🧭 선호 카테고리")
    v = max(meta.get("view_count", 1), 1)
    if meta.get("cart_count", 0) / v > 0.005:
        chips.append("📈 인기↑")
    return chips


def _cf_chips(alias: str, catalog: dict) -> list[str]:
    """CF 섹션 전용 chips — cf_neighbor_* 신호만 사용."""
    meta  = catalog.get(alias, {})
    chips = ["👥 유사 유저"]
    v = max(meta.get("view_count", 1), 1)
    if meta.get("purchase_count", 0) / v > 0.003:
        chips.append("🛒 이웃 구매")
    elif meta.get("cart_count", 0) / v > 0.005:
        chips.append("🛒 이웃 장바구니")
    return chips

def _item_num(alias: str) -> str:
    """alias 마지막 세그먼트 = 상품품번 (예: mid_0001)."""
    return alias.split(".")[-1] if "." in alias else alias

def _item_label(alias: str, meta: dict) -> str:
    """채팅 표시용: 카테고리(한글) > 브랜드 > 상품품번  /  가격 : 000,000원"""
    cat   = ko_cat(meta.get("l2", ""), meta.get("l3"))
    brand = meta.get("brand", "")
    price = meta.get("price_display", "")
    return f"{cat} > {brand} > {_item_num(alias)}  /  가격 : {price}"

def _item_label_short(alias: str, meta: dict) -> str:
    """카드 표시용: 카테고리(한글) | 브랜드 | 상품품번"""
    cat   = ko_cat(meta.get("l2", ""), meta.get("l3"))
    brand = meta.get("brand", "")
    return f"{cat} | {brand} | {_item_num(alias)}"

def _get_best_item(result: dict) -> tuple[str, str]:
    """explanation에서 reason이 있는 첫 번째 상품 반환. 없으면 candidates 첫 번째.

    requested_section에 따라 섹션 우선순위를 조정한다.
    """
    req_sec = result.get("requested_section")
    if req_sec == "recency":
        sec_order = ("recency", "content", "revisit", "top10")
    elif req_sec == "revisit":
        sec_order = ("revisit", "content", "recency", "top10")
    else:
        sec_order = ("top10", "content", "recency", "revisit")

    for sec in sec_order:
        for item in (result.get("explanation") or {}).get("sections", {}).get(sec, []):
            if isinstance(item, dict) and item.get("item_alias") and item.get("reason", "").strip():
                return item["item_alias"], item["reason"]
    for sec in sec_order:
        items = result.get("candidates", {}).get(sec, [])
        if items:
            return items[0], ""
    return "", ""

def _trust_badge(passed: bool | None) -> None:
    # icon= 파라미터가 이미 이모지를 렌더하므로 텍스트에서 이모지 제거
    if passed is True:
        st.success("hard_gate 검증 통과", icon="✅")
    elif passed is False:
        st.warning("미검증 — 후보 목록만 표시 또는 일반 응답", icon="⚠️")


def _groundedness_badge(score: float, temperature: float | None = None) -> None:
    """SelfCheckGPT groundedness 점수 표시."""
    import math

    def _sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def _logit(p: float, eps: float = 1e-7) -> float:
        p = max(eps, min(1 - eps, p))
        return math.log(p / (1 - p))

    calibrated = score
    if temperature and temperature != 1.0:
        calibrated = round(_sigmoid(_logit(score) / temperature), 3)

    pct = int(calibrated * 100)
    if pct >= 80:
        st.success(f"Groundedness {pct}% — 설명 일관성 높음", icon="🧪")
    elif pct >= 50:
        st.info(f"Groundedness {pct}% — 설명 일관성 보통", icon="🧪")
    else:
        st.warning(f"Groundedness {pct}% — 환각 가능성 있음", icon="⚠️")

# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────
def _init_session() -> None:
    defaults = {
        "thread_id":   str(uuid.uuid4()),
        "messages":    [],        # [{"role": str, "content": str}]
        "user_alias":  None,
        "last_result": None,      # 마지막 app.invoke() 결과
        "_prev_ua":    None,      # 유저 변경 감지용
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_session()

# ── 전역 최소 폰트 사이즈 (12px) ──────────────────────────────────────────────
st.markdown("""
<style>
/* st.caption — Streamlit 기본값 0.75rem 이 12px 미만일 수 있어 명시 지정 */
.stCaptionContainer p,
[data-testid="stCaptionContainer"] p,
.stSidebar .stCaptionContainer p {
    font-size: 12px !important;
    line-height: 1.5 !important;
}
/* 기타 소형 텍스트 요소 */
small, .stSmall {
    font-size: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ── 사이드바 ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("👤 유저 선택")

    users = _load_users()
    aliases = [None] + [ua for ua, _ in users]
    labels  = ["— 선택 —"] + [lbl for _, lbl in users]

    cur_idx = next(
        (i for i, a in enumerate(aliases) if a == st.session_state.user_alias), 0
    )
    sel_idx = st.selectbox("유저", range(len(labels)), index=cur_idx,
                           format_func=lambda i: labels[i])
    new_alias = aliases[sel_idx]

    # 유저 변경 시 대화 초기화
    if new_alias != st.session_state["_prev_ua"]:
        st.session_state.user_alias  = new_alias
        st.session_state.thread_id   = str(uuid.uuid4())
        st.session_state.messages    = []
        st.session_state.last_result = None
        st.session_state["_prev_ua"] = new_alias

    # 선택된 유저 프로필 요약
    if st.session_state.user_alias:
        profile = _load_profile(st.session_state.user_alias)
        if profile:
            al = profile.get("activity_level", {})
            st.caption(
                f"이벤트 {al.get('total_events',0)}회 · "
                f"최근 14일 {al.get('recent_14d',0)}회"
            )
            st.caption("선호 카테고리: " + ", ".join(profile.get("top_categories_l2", [])[:3]))
            st.caption("선호 브랜드: "   + ", ".join(profile.get("top_brands", [])[:3]))

    st.divider()

    # 데모 시나리오 버튼
    st.subheader("🎬 데모 시나리오")
    demo_users = [
        ("user_00010", "🏃 활발한 쇼핑러"),
        ("user_00004", "👟 신발 전문 탐색"),
        ("user_00017", "🌙 파자마·셔츠 선호"),
    ]
    for _ua, _label in demo_users:
        if _ua in [a for a, _ in users]:
            if st.button(_label, key=f"demo_{_ua}", use_container_width=True):
                st.session_state.user_alias  = _ua
                st.session_state.thread_id   = str(uuid.uuid4())
                st.session_state.messages    = []
                st.session_state.last_result = None
                st.session_state["_prev_ua"] = _ua
                st.rerun()

    st.divider()
    if st.button("🔄 대화 초기화", use_container_width=True):
        st.session_state.thread_id   = str(uuid.uuid4())
        st.session_state.messages    = []
        st.session_state.last_result = None
        st.rerun()

    st.divider()

    st.subheader("🧪 품질 검증")
    enable_selfcheck = st.toggle(
        "SelfCheckGPT 활성화",
        value=False,
        help="쇼핑 추천 후 Solar Pro를 2회 추가 호출해 groundedness 점수를 측정합니다 (느림).",
        key="selfcheck_toggle",
    )

    eval_report = _load_eval_report()
    if eval_report:
        with st.expander("📊 Golden Set 평가 결과", expanded=False):
            st.metric("LLM-as-Judge 평균", f"{eval_report.get('avg_judge_score', 0):.2f}/5.0")
            st.metric("샘플 수", eval_report.get("n_entries", 0))
            g = eval_report.get("avg_groundedness", 0)
            if g > 0:
                st.metric("평균 Groundedness", f"{g:.3f}")
                st.metric("ECE (보정 전/후)",
                          f"{eval_report.get('ece_before',0):.4f} → {eval_report.get('ece_after',0):.4f}")
                st.metric("Temperature (T)", eval_report.get("temperature", 1.0))
            pass_flag = eval_report.get("pass_threshold", False)
            st.success("회귀 테스트 PASS ✅" if pass_flag else "회귀 테스트 FAIL ❌")
    else:
        st.caption("golden set 평가 결과 없음 — `evaluator.py`를 실행하세요.")

# ── 메인 ─────────────────────────────────────────────────────────────────────
st.title("🛍️ LLM 커머스 추천 MVP")
st.caption("Solar Pro 설명 · hard_gate 검증 · 5섹션 추천 · MemorySaver 멀티턴")

catalog   = _load_catalog()
user_recs = _load_user_recs()

# ── 대시보드: Top-5 카드 ──────────────────────────────────────────────────────
ua = st.session_state.user_alias
if ua:
    profile = _load_profile(ua) or {}

    # 마지막 쇼핑 결과가 있으면 우선 사용, 없으면 user_recs에서 auto-load
    last = st.session_state.last_result
    # candidates 값이 None일 수 있으므로 `or {}` 로 None 방어
    top5_source = ((last or {}).get("candidates") or {}).get("top10") or user_recs.get(ua, [])
    top5 = top5_source[:5]

    trust_passed = (last or {}).get("trust_passed")

    if top5:
        st.subheader("📊 모델 추천 Top-5")
        # 쇼핑 추천 결과가 있을 때만 검증 배지 표시 (일반 채팅 결과는 표시 안 함)
        if last and last.get("candidates"):
            _trust_badge(trust_passed)
        cols = st.columns(5)
        for i, (col, alias) in enumerate(zip(cols, top5)):
            meta   = catalog.get(alias, {})
            chips  = _evidence_chips(alias, catalog, profile)
            with col:
                st.markdown(
                    f"<div style='background:#f8f9fa;border-radius:8px;"
                    f"padding:10px;min-height:120px;font-size:15px'>"
                    f"<b>#{i+1}</b><br>"
                    f"<span style='font-size:15px;color:#555'>{_item_label_short(alias, meta)}</span><br>"
                    f"<b style='color:#1a73e8'>{meta.get('price_display','')}</b><br>"
                    f"<span style='color:#666'>{' '.join(chips)}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    st.divider()

# ── 채팅 UI ───────────────────────────────────────────────────────────────────
chat_col, rec_col = st.columns([1, 1])

with chat_col:
    st.subheader("💬 대화")

    # 채팅 히스토리 표시
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 프리셋 버튼 (유저 선택 시)
    preset: str | None = None
    if ua:
        st.caption("빠른 질문:")
        p1, p2, p3, p4, p5 = st.columns(5)
        if p1.button("❓ 왜?",    key="pr1"): preset = "방금 추천한 상품들을 왜 나한테 추천했어?"
        if p2.button("💭 살까?",  key="pr2"): preset = "추천 상품 중에 뭘 사면 좋을지 골라줘"
        if p3.button("💰 더 싼?", key="pr3"): preset = "비슷하지만 더 저렴한 상품은 없어?"
        if p4.button("💎 더 고급?",key="pr4"): preset = "더 프리미엄 고급 상품으로 추천해줘"
        if p5.button("🎯 취향?",  key="pr5"): preset = "내 취향에 딱 맞는 다른 상품 더 알려줘"

    # 채팅 입력
    if ua:
        placeholder = f"{labels[sel_idx]}님 질문하세요..."
    else:
        placeholder = "먼저 좌측에서 유저를 선택하세요"

    user_input = st.chat_input(placeholder, disabled=not ua)
    to_send = preset or user_input

    if to_send:
        # 메시지 즉시 표시
        st.session_state.messages.append({"role": "user", "content": to_send})
        with st.chat_message("user"):
            st.markdown(to_send)

        # LangGraph invoke
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        with st.chat_message("assistant"):
            with st.spinner("생성 중..."):
                try:
                    result = _get_app().invoke(
                        {
                            "message":    to_send,
                            "user_alias": st.session_state.user_alias,
                        },
                        config=config,
                    )
                except Exception as exc:
                    err_type = type(exc).__name__
                    result = {
                        "response": f"응답 생성 중 오류가 발생했습니다 ({err_type}). 잠시 후 다시 시도해 주세요.",
                        "candidates": None,
                        "trust_passed": None,
                    }

            # SelfCheckGPT (토글 활성화 + 쇼핑 추천 결과가 있을 때만)
            if (
                st.session_state.get("selfcheck_toggle", False)
                and result.get("trust_passed")
                and result.get("explanation")
            ):
                with st.spinner("🧪 SelfCheckGPT 실행 중 (2회 추가 호출)..."):
                    try:
                        sc = _self_check(
                            context=result.get("context_text", ""),
                            explanation=result["explanation"],
                            solar_client=_solar,
                            n_samples=2,
                            candidates=result.get("candidates"),
                        )
                        result["groundedness"] = sc.groundedness
                    except Exception:
                        result["groundedness"] = None

            response = result.get("response") or "(응답 없음)"

            if result.get("trust_passed") and result.get("candidates"):
                # 최고 상품 1개 + 사유 표시
                best_alias, best_reason = _get_best_item(result)
                if best_alias:
                    best_meta  = catalog.get(best_alias, {})
                    label_line = _item_label(best_alias, best_meta)
                    chat_lines = ["**추천 결과**", "", f"**{label_line}**"]
                    if best_reason:
                        chat_lines += ["", f"> {best_reason}"]
                    chat_lines += ["", "전체 추천은 오른쪽 **📋 추천 목록**에서 확인하세요."]
                    st.markdown("\n".join(chat_lines))
                else:
                    st.markdown("추천 결과가 준비됐어요! 오른쪽 **📋 추천 목록**에서 확인하세요.")

            elif result.get("candidates"):
                # 검증 실패했지만 후보는 존재 (일반적으로 발생하지 않아야 함)
                st.markdown("추천 목록을 준비했어요. 오른쪽 **📋 추천 목록**에서 확인하세요.")

            else:
                # 일반 대화 응답 (⚠️ 미검증 배지 포함)
                st.markdown(response)

        # 히스토리에 저장할 내용 결정 (화면 표시와 동일한 텍스트)
        if result.get("trust_passed") and result.get("candidates"):
            best_alias, best_reason = _get_best_item(result)
            if best_alias:
                _lbl = _item_label(best_alias, catalog.get(best_alias, {}))
                _saved = f"**추천 결과**\n\n**{_lbl}**"
                if best_reason:
                    _saved += f"\n\n> {best_reason}"
            else:
                _saved = "추천 결과가 준비됐어요!"
        elif result.get("candidates"):
            _saved = "추천 목록을 준비했어요."
        else:
            _saved = response  # 일반 채팅은 전체 저장 (잘림 없음)
        st.session_state.messages.append({"role": "assistant", "content": _saved})
        st.session_state.last_result = result
        st.rerun()

# ── 추천 결과 accordion ───────────────────────────────────────────────────────
with rec_col:
    st.subheader("📋 추천 목록")

    last = st.session_state.last_result
    has_llm = last and (last.get("candidates") is not None)

    if not ua:
        st.info("👈 좌측에서 유저를 먼저 선택하세요.")
    else:
        profile_for_chips = _load_profile(ua) or {}

        if has_llm:
            # 채팅 후 Solar Pro 결과
            candidates  = last.get("candidates") or {}
            explanation = last.get("explanation") or {}
            trust_ok    = last.get("trust_passed", False)
            show_badge  = True
        else:
            # 유저 선택 즉시: Python 엔진으로 후보 자동 계산
            with st.spinner("추천 목록 로딩 중..."):
                candidates = _compute_candidates(ua) or {}
            explanation = {}
            trust_ok    = False
            show_badge  = False  # Solar Pro 미호출 상태 — 배지 없음

        if not candidates:
            st.info("💬 왼쪽 채팅에서 추천을 요청해 보세요.")
        else:
            if show_badge:
                _trust_badge(trust_ok)
                # 가격 재질의 조건 표시
                pf = (last or {}).get("requested_price_filter")
                if pf == "cheaper":
                    st.info("💰 이전 추천 대비 저렴한 상품 필터 적용", icon="💰")
                elif pf == "pricier":
                    st.info("💎 이전 추천 대비 고가 상품 필터 적용", icon="💎")
                # Groundedness 배지 (SelfCheckGPT 결과가 있을 때)
                if has_llm and last.get("groundedness") is not None:
                    _groundedness_badge(
                        last["groundedness"],
                        temperature=_load_calibration_temperature(),
                    )

            # 설명 맵: item_alias → reason
            explained: dict[str, str] = {}
            if trust_ok:
                for sec_items in explanation.get("sections", {}).values():
                    for item in sec_items:
                        if isinstance(item, dict) and item.get("item_alias"):
                            explained[item["item_alias"]] = item.get("reason", "")

            _SEC_CONFIG = [
                ("content",       "🎯 내 취향",          True),
                ("recency",       "🆕 새로 나온",        True),
                ("revisit",       "🔄 다시 볼 만한",     True),
                ("collaborative", "👥 비슷한 분들이 산", False),
            ]

            for sec_key, label, _default_open in _SEC_CONFIG:
                items = candidates.get(sec_key, [])
                if not items:
                    if sec_key == "revisit" and len(profile_for_chips.get("seen_items", {})) < 3:
                        st.caption("🔄 다시 볼 만한: 탐색 이력이 충분하지 않아 섹션 숨김")
                    continue

                with st.expander(f"{label}  ({len(items)}개)", expanded=False):
                    for alias in items[:5]:
                        meta   = catalog.get(alias, {})
                        # CF 섹션은 cf_neighbor_* 신호만, 나머지는 content 기반 chips
                        chips  = (_cf_chips(alias, catalog) if sec_key == "collaborative"
                                  else _evidence_chips(alias, catalog, profile_for_chips))
                        reason = explained.get(alias, "")

                        c1, c2 = st.columns([2, 3])
                        with c1:
                            st.markdown(f"<span style='font-size:15px;font-weight:bold'>{alias}</span>", unsafe_allow_html=True)
                            st.caption(
                                f"{ko_cat(meta.get('l2',''), meta.get('l3'))} · "
                                f"{meta.get('brand','')} · "
                                f"{meta.get('price_display','')}"
                            )
                            if chips:
                                st.caption("  ".join(chips))
                        with c2:
                            if reason:
                                st.markdown(f"> {reason}")
                        st.divider()
