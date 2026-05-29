"""
4종 추천 엔진 + Top-10 + dedup.

섹션 0: 모델 Top-10    — submission 그대로
섹션 1: 비슷한 분들이  — CF (FAISS 미빌드 시 빈 리스트)
섹션 2: 내 취향        — content-based (카테고리·브랜드·가격대)
섹션 3: 새로 나온      — recency pool unseen 필터
섹션 4: 다시 볼 만한   — seen_items revisit score (독립)

Dedup 우선순위 (unseen 풀): submission > CF > content > recency
Revisit은 seen 풀 — dedup 대상 외
"""
from __future__ import annotations

import datetime
import re
import os

MAX_EXPAND = 2   # dedup_with_expand 최대 확장 횟수
TOP_N = 5        # 섹션별 기본 반환 수
_REVISIT_EXCLUDE_DAYS = int(os.getenv("REVISIT_EXCLUDE_DAYS", "14"))


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _price_in_band(price: float, pr: dict) -> bool:
    """사용자 가격대(IQR ±0.5배 구간) 안에 있으면 True."""
    iqr = pr["p75"] - pr["p25"]
    lo = max(pr["p25"] - 0.5 * iqr, 0.0)
    hi = pr["p75"] + 0.5 * iqr
    return lo <= price <= hi


def dedup_with_expand(
    candidates_pool: list[str],
    excluded: set[str],
    top_n: int,
    max_expand: int = MAX_EXPAND,
) -> list[str]:
    """candidates_pool 앞에서 excluded 제거 후 top_n개 추출.

    부족하면 k를 +5씩 확장. max_expand 초과 시 빈 리스트로 섹션 숨김 처리.
    """
    k = top_n
    for _ in range(max_expand + 1):
        result = [a for a in candidates_pool[:k] if a not in excluded][:top_n]
        if result:
            return result
        k += 5
    return []


# ── 섹션 0: 모델 Top-10 ───────────────────────────────────────────────────────

def recommend_top10(user_alias: str, user_recs: dict[str, list[str]]) -> list[str]:
    """submission Top-10 그대로 반환."""
    return user_recs.get(user_alias, [])


# ── 섹션 1: CF (유사 유저) ────────────────────────────────────────────────────

def recommend_cf(
    profile: dict,
    neighbors_data: dict | None,
    top_n: int = TOP_N,
) -> list[str]:
    """유사 유저 CF. neighbors_data=None이면 FAISS 미빌드 → 빈 리스트 반환.

    neighbors_data 구조:
        {
          "neighbors_npy":  np.ndarray (N, 20),
          "alias_to_row":   {user_alias: int},
          "row_to_alias":   {int: user_alias},
          "neighbor_profiles": {user_alias: profile_dict},
        }
    """
    if neighbors_data is None:
        return []

    user_alias = profile.get("user_alias", "")
    alias_to_row = neighbors_data.get("alias_to_row", {})
    row_to_alias = neighbors_data.get("row_to_alias", {})
    neighbors_npy = neighbors_data.get("neighbors_npy")
    neighbor_profiles = neighbors_data.get("neighbor_profiles", {})

    row = alias_to_row.get(user_alias)
    if row is None or neighbors_npy is None:
        return []

    seen_set = set(profile.get("seen_items", {}).keys())
    neighbor_rows = neighbors_npy[int(row)]  # (20,)

    # 이웃의 purchase > cart > view 점수 합산
    item_scores: dict[str, float] = {}
    _W = {"purchase": 50.0, "cart": 25.0, "view": 1.0}
    for r in neighbor_rows:
        n_alias = row_to_alias.get(int(r))
        if not n_alias:
            continue
        n_profile = neighbor_profiles.get(n_alias, {})
        for alias, info in n_profile.get("seen_items", {}).items():
            if alias in seen_set:
                continue
            w = _W.get(info.get("last_event_type", "view"), 1.0)
            item_scores[alias] = item_scores.get(alias, 0.0) + w

    pool = sorted(item_scores, key=lambda k: -item_scores[k])
    # 넉넉히 반환 — dedup_with_expand에서 trim
    return pool[: top_n * (MAX_EXPAND + 1) * 3]


# ── 섹션 2: 내 취향 (Content-based) ──────────────────────────────────────────

def recommend_content(
    profile: dict,
    item_catalog: dict[str, dict],
    top_n: int = TOP_N,
) -> list[str]:
    """카테고리·브랜드·가격대 affinity로 unseen 아이템 순위 매김.

    Returns:
        정렬된 후보 풀 (dedup_with_expand에서 trim). seen_items 제외 완료.
    """
    seen_set = set(profile.get("seen_items", {}).keys())
    top_cats = profile.get("top_categories_l2", [])
    top_brands = profile.get("top_brands", [])
    pr = profile.get("price_range", {})

    # 순위 기반 가중치: 1위=N, 2위=N-1, ...
    cat_w = {c: (len(top_cats) - i) for i, c in enumerate(top_cats)}
    brand_w = {b: (len(top_brands) - i) for i, b in enumerate(top_brands)}

    scored: list[tuple[str, float]] = []
    for alias, meta in item_catalog.items():
        if alias in seen_set:
            continue
        c_score = cat_w.get(meta.get("l2", ""), 0) * 2   # 카테고리 가중치 2배
        b_score = brand_w.get(meta.get("brand", ""), 0)
        p_score = 1 if (pr and _price_in_band(meta.get("price_median", 0.0), pr)) else 0
        total = c_score + b_score + p_score
        if total > 0:
            scored.append((alias, float(total)))

    if not scored:
        # fallback: unseen 전체, view_count 내림차순
        return [
            a for a, _ in sorted(
                ((a, m.get("view_count", 0)) for a, m in item_catalog.items() if a not in seen_set),
                key=lambda x: -x[1],
            )
        ]

    scored.sort(key=lambda x: (-x[1], -item_catalog.get(x[0], {}).get("view_count", 0)))
    return [a for a, _ in scored]


# ── 섹션 3: 새로 나온 (Recency + Personalized Unseen) ─────────────────────────

def _recency_candidates(
    profile: dict,
    recency_pool: dict[str, list[str]],
    mode: str,
) -> list[str]:
    """mode에 따른 recency pool 후보 목록 반환 (seen_set 필터 전)."""
    top_cats = profile.get("top_categories_l2", [])
    top_brands_set = set(profile.get("top_brands", []))

    if mode == "cat+brand":
        # Semantic ID: {l2}[.{l3}].{brand_slug}.{price_bucket}_{seq}
        # parts[-2] = brand_slug (언더스코어 포함 가능), parts[-1] = price_bucket_seq
        # profile의 top_brands는 원본 브랜드명 → slug로 변환 후 비교
        slugged_brands = {
            re.sub(r"[^a-z0-9]", "_", b.lower()).strip("_") or "unknown"
            for b in top_brands_set
        }

        result: list[str] = []
        for l2 in top_cats:
            for alias in recency_pool.get(l2, []):
                parts = alias.split(".")
                brand_slug = parts[-2] if len(parts) >= 3 else ""
                if brand_slug in slugged_brands:
                    result.append(alias)
        return result

    if mode == "cat":
        result = []
        for l2 in top_cats:
            result.extend(recency_pool.get(l2, []))
        return result

    # mode == "all": 전체 pool (카테고리 순서 없이 평탄화)
    return [alias for aliases in recency_pool.values() for alias in aliases]


def recommend_recency(
    profile: dict,
    recency_pool: dict[str, list[str]],
    top_n: int = TOP_N,
) -> list[str]:
    """신상품 풀에서 unseen 후보 반환. 4단계 fallback.

    Returns:
        정렬된 후보 풀 (dedup_with_expand에서 trim). seen_items 제외 완료.
    """
    seen_set = set(profile.get("seen_items", {}).keys())

    for mode in ("cat+brand", "cat", "all"):
        candidates = _recency_candidates(profile, recency_pool, mode)
        pool = [a for a in candidates if a not in seen_set]
        if pool:
            return pool

    # fallback ④: unseen 없으면 미구매 상품으로 대체
    purchased_set = {p["item_alias"] for p in profile.get("purchased_items", [])}
    return [
        a
        for aliases in recency_pool.values()
        for a in aliases
        if a not in purchased_set
    ]


# ── 섹션 4: 다시 볼 만한 (Revisit) ───────────────────────────────────────────

def recommend_revisit(
    profile: dict,
    top_n: int = TOP_N,
    exclude_recent_days: int = _REVISIT_EXCLUDE_DAYS,
) -> list[str]:
    """seen_items 중 revisit score 상위 아이템 반환.

    seen_items < 3이면 빈 리스트 (섹션 숨김).
    최근 exclude_recent_days 이내 구매 아이템은 제외.
    """
    seen_items = profile.get("seen_items", {})
    if len(seen_items) < 3:
        return []

    # 데이터 기준일: seen_items 내 가장 최근 last_event_date
    all_dates = [v["last_event_date"] for v in seen_items.values() if v.get("last_event_date")]
    ref_date = datetime.date.fromisoformat(max(all_dates)) if all_dates else datetime.date(2020, 2, 29)

    recently_purchased: set[str] = set()
    for p in profile.get("purchased_items", []):
        try:
            pd_date = datetime.date.fromisoformat(p["purchase_date"])
            if (ref_date - pd_date).days <= exclude_recent_days:
                recently_purchased.add(p["item_alias"])
        except (ValueError, KeyError):
            pass

    scored = [
        (alias, info["score"])
        for alias, info in seen_items.items()
        if alias not in recently_purchased
    ]
    scored.sort(key=lambda x: -x[1])
    return [alias for alias, _ in scored[:top_n * (MAX_EXPAND + 1)]]


# ── 전체 조합 + Dedup ─────────────────────────────────────────────────────────

def build_all_candidates(
    profile: dict,
    user_alias: str,
    user_recs: dict[str, list[str]],
    item_catalog: dict[str, dict],
    recency_pool: dict[str, list[str]],
    neighbors_data: dict | None = None,
    top_n: int = TOP_N,
    price_cap: float | None = None,
    price_floor: float | None = None,
) -> dict[str, list[str]]:
    """4종 추천 후보 생성 + dedup.

    Dedup 우선순위 (unseen 풀): Top10 > CF > content > recency
    Revisit은 seen 풀 독립.

    price_cap / price_floor: 재질의 가격 필터 (None이면 미적용)

    Returns:
        {
          "top10":         list[str],   # 섹션 0 (dedup 기준점, 그대로 표시)
          "collaborative": list[str],   # 섹션 1
          "content":       list[str],   # 섹션 2
          "recency":       list[str],   # 섹션 3
          "revisit":       list[str],   # 섹션 4 (독립)
        }
    """
    # 가격 필터 적용 카탈로그 (재질의 시)
    if price_cap is not None or price_floor is not None:
        effective_catalog = {
            k: v for k, v in item_catalog.items()
            if (price_cap   is None or v.get("price_median", 0) <= price_cap)
            and (price_floor is None or v.get("price_median", 0) >= price_floor)
        }
    else:
        effective_catalog = item_catalog

    # 섹션 0: top10도 가격 필터 적용
    top10_raw = recommend_top10(user_alias, user_recs)
    top10 = [a for a in top10_raw if a in effective_catalog] if effective_catalog is not item_catalog else top10_raw
    excluded = set(top10)

    # 섹션 1: CF (가격 필터: cf_pool에서 effective_catalog에 없는 아이템 제거)
    cf_pool_raw = recommend_cf(profile, neighbors_data, top_n)
    cf_pool = [a for a in cf_pool_raw if a in effective_catalog]
    cf_items = dedup_with_expand(cf_pool, excluded, top_n)
    excluded |= set(cf_items)

    # 섹션 2: Content
    content_pool = recommend_content(profile, effective_catalog, top_n)
    content_items = dedup_with_expand(content_pool, excluded, top_n)
    excluded |= set(content_items)

    # 섹션 3: Recency (recency_pool 내 아이템도 가격 필터)
    recency_pool_filtered = recommend_recency(profile, recency_pool, top_n)
    recency_pool_filtered = [a for a in recency_pool_filtered if a in effective_catalog]
    recency_items = dedup_with_expand(recency_pool_filtered, excluded, top_n)

    # 섹션 4: Revisit (독립, 가격 필터 적용)
    revisit_raw = recommend_revisit(profile, top_n)
    revisit_items = [a for a in revisit_raw if a in effective_catalog]

    return {
        "top10":         top10,
        "collaborative": cf_items,
        "content":       content_items,
        "recency":       recency_items,
        "revisit":       revisit_items,
    }


# ── 단독 테스트 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sqlite3
    from pathlib import Path

    ROOT = Path(__file__).resolve().parents[2]

    print("=== recommenders.py 단독 테스트 ===\n")

    # 데이터 로드
    with open(ROOT / "rag_data/item_catalog.json", encoding="utf-8") as f:
        item_catalog = json.load(f)
    with open(ROOT / "rag_data/user_recommendations.json", encoding="utf-8") as f:
        user_recs = json.load(f)
    with open(ROOT / "rag_data/recency_pool.json", encoding="utf-8") as f:
        recency_pool = json.load(f)

    con = sqlite3.connect(ROOT / "rag_data/user_profiles.db")

    test_users = ["user_00001", "user_00004", "user_00010"]
    for user_alias in test_users:
        row = con.execute(
            "SELECT profile_json FROM user_profiles WHERE user_alias=?", (user_alias,)
        ).fetchone()
        if not row:
            print(f"[{user_alias}] 프로필 없음 (샘플 미포함)\n")
            continue

        profile = json.loads(row[0])
        profile["user_alias"] = user_alias

        cands = build_all_candidates(
            profile, user_alias, user_recs, item_catalog, recency_pool
        )

        print(f"[{user_alias}]  seen={len(profile['seen_items'])}개  "
              f"cats={profile['top_categories_l2'][:3]}  "
              f"brands={profile['top_brands'][:3]}")
        for sec, items in cands.items():
            label = {
                "top10": "0 Top-10  ",
                "collaborative": "1 CF      ",
                "content": "2 취향    ",
                "recency": "3 신상품  ",
                "revisit": "4 재방문  ",
            }[sec]
            hidden = " [섹션 숨김]" if not items and sec in ("collaborative", "revisit") else ""
            print(f"  {label}: {items[:5]}{hidden}")
        print()

    con.close()
