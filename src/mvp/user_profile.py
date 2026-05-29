"""
유저 프로필 집계 → user_profiles.db (SQLite).

스키마:
    user_profiles(user_alias TEXT PK, user_id TEXT, profile_json TEXT, profile_text TEXT)
"""
from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

EVENT_WEIGHT = {"view": 1.0, "cart": 25.0, "purchase": 50.0}
_LAMBDA = math.log(2) / 45.0  # half-life 45일 (90일 학습 기간 기준)


def _temporal_decay(days_ago: float) -> float:
    return math.exp(-_LAMBDA * max(days_ago, 0.0))


def build_user_profile(
    user_id: str,
    user_df: pd.DataFrame,
    item_alias_map: dict[str, str],
    item_meta: dict[str, dict],
    data_end_date: pd.Timestamp,
) -> dict:
    """단일 유저 프로필 집계.

    Args:
        user_id: 원본 UUID
        user_df: 해당 유저의 이벤트 행 (event_time은 datetime with tz)
        item_alias_map: item_id → item_alias
        item_meta: item_id → {l2, l3, brand, price_median, ...}
        data_end_date: 데이터 기준일 (temporal_decay 계산 기준)
    """
    user_df = user_df.sort_values("event_time").reset_index(drop=True)

    cat_scores: dict[str, float] = {}
    brand_scores: dict[str, float] = {}
    prices: list[float] = []

    seen_items: dict[str, dict] = {}

    for _, row in user_df.iterrows():
        w = EVENT_WEIGHT.get(row["event_type"], 1.0)
        meta = item_meta.get(row["item_id"], {})
        l2 = meta.get("l2", "")
        brand = str(row["brand"]) if pd.notna(row.get("brand")) else meta.get("brand", "")

        if l2:
            cat_scores[l2] = cat_scores.get(l2, 0.0) + w
        if brand and brand != "nan":
            brand_scores[brand] = brand_scores.get(brand, 0.0) + w

        if pd.notna(row.get("price")):
            prices.append(float(row["price"]))

        alias = item_alias_map.get(row["item_id"])
        if alias:
            days_ago = (data_end_date - row["event_time"]).total_seconds() / 86400
            score_delta = w * _temporal_decay(days_ago)
            if alias not in seen_items:
                seen_items[alias] = {
                    "score": 0.0,
                    "event_count": 0,
                    "last_event_type": row["event_type"],
                    "last_event_date": str(row["event_time"].date()),
                }
            seen_items[alias]["score"] += score_delta
            seen_items[alias]["event_count"] += 1
            # user_df는 시간순 정렬이므로 마지막 값이 최신
            seen_items[alias]["last_event_type"] = row["event_type"]
            seen_items[alias]["last_event_date"] = str(row["event_time"].date())

    top_categories_l2 = sorted(cat_scores, key=lambda k: -cat_scores[k])[:5]
    top_brands = sorted(brand_scores, key=lambda k: -brand_scores[k])[:5]

    price_arr = np.array(prices) if prices else np.array([0.0])
    price_range = {
        "p25": float(np.percentile(price_arr, 25)),
        "p75": float(np.percentile(price_arr, 75)),
        "median": float(np.median(price_arr)),
    }

    recent_cutoff = data_end_date - pd.Timedelta(days=14)
    recent_14d = int((user_df["event_time"] >= recent_cutoff).sum())
    activity_level = {
        "total_events": len(user_df),
        "recent_14d": recent_14d,
    }

    recent_items: list[dict] = []
    for _, row in user_df.tail(10).iterrows():
        alias = item_alias_map.get(row["item_id"])
        if alias:
            meta = item_meta.get(row["item_id"], {})
            recent_items.append({
                "item_alias": alias,
                "category": meta.get("l2", ""),
                "brand": str(row["brand"]) if pd.notna(row.get("brand")) else meta.get("brand", ""),
                "event_type": row["event_type"],
                "event_date": str(row["event_time"].date()),
            })

    cart_ids = set(user_df[user_df["event_type"] == "cart"]["item_id"])
    purchase_ids = set(user_df[user_df["event_type"] == "purchase"]["item_id"])
    cart_items = [
        item_alias_map[i]
        for i in (cart_ids - purchase_ids)
        if i in item_alias_map
    ]

    purchased_items: list[dict] = []
    seen_purchase: set[str] = set()
    for _, row in user_df[user_df["event_type"] == "purchase"].iterrows():
        alias = item_alias_map.get(row["item_id"])
        if alias and alias not in seen_purchase:
            purchased_items.append({
                "item_alias": alias,
                "purchase_date": str(row["event_time"].date()),
            })
            seen_purchase.add(alias)

    return {
        "user_id": user_id,
        "top_categories_l2": top_categories_l2,
        "top_brands": top_brands,
        "price_range": price_range,
        "activity_level": activity_level,
        "recent_items": recent_items,
        "seen_items": seen_items,
        "cart_items": cart_items,
        "purchased_items": purchased_items,
    }


def build_profile_text(profile: dict, display_name: str, user_alias: str) -> str:
    """Solar Pro 주입용 한국어 자연어 프로필 텍스트."""
    cats = ", ".join(profile["top_categories_l2"]) or "정보 없음"
    brands = ", ".join(profile["top_brands"]) or "정보 없음"
    pr = profile["price_range"]
    al = profile["activity_level"]
    recent = [r["item_alias"] for r in profile["recent_items"][-3:]]
    recent_str = ", ".join(recent) if recent else "없음"
    cart_str = ", ".join(profile["cart_items"][:3]) if profile["cart_items"] else "없음"

    return (
        f"[사용자 프로필 — {display_name} ({user_alias})]\n"
        f"- 선호 카테고리: {cats}\n"
        f"- 선호 브랜드: {brands}\n"
        f"- 가격대: {pr['p25']:.0f}~{pr['p75']:.0f} (중앙값 {pr['median']:.0f})\n"
        f"- 최근 관심: {recent_str}\n"
        f"- 장바구니 대기: {cart_str}\n"
        f"- 활동: 총 {al['total_events']}회, 최근 14일 {al['recent_14d']}회"
    )


def build_profiles_db(
    train_df: pd.DataFrame,
    item_alias_map: dict[str, str],
    item_meta: dict[str, dict],
    user_alias_map: dict[str, dict],
    db_path: str | Path,
    sample_users: set[str] | None = None,
    batch_size: int = 500,
) -> None:
    """전체 유저 프로필 집계 후 SQLite DB 저장.

    Args:
        train_df: 전체 행동 로그 (event_time은 datetime with tz)
        item_alias_map: item_id → item_alias
        item_meta: item_id → meta dict
        user_alias_map: user_id → {user_alias, display_name}
        db_path: 저장할 SQLite DB 경로
        sample_users: 처리할 user_id 집합. None이면 전체 처리
        batch_size: DB insert 배치 크기
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    data_end_date = train_df["event_time"].max()

    target_users = [
        uid for uid in user_alias_map
        if sample_users is None or uid in sample_users
    ]

    # user_id별로 df를 미리 그룹화
    grouped = {uid: grp for uid, grp in train_df.groupby("user_id", sort=False)}

    with sqlite3.connect(db_path) as con:
        con.execute("DROP TABLE IF EXISTS user_profiles")
        con.execute("""
            CREATE TABLE user_profiles (
                user_alias   TEXT PRIMARY KEY,
                user_id      TEXT,
                profile_json TEXT,
                profile_text TEXT
            )
        """)

        rows: list[tuple] = []
        for uid in tqdm(target_users, desc="프로필 빌드", unit="user"):
            info = user_alias_map[uid]
            user_alias = info["user_alias"]
            display_name = info["display_name"]

            user_df = grouped.get(uid, pd.DataFrame(columns=train_df.columns))
            profile = build_user_profile(
                uid, user_df, item_alias_map, item_meta, data_end_date
            )
            profile_text = build_profile_text(profile, display_name, user_alias)

            rows.append((
                user_alias,
                uid,
                json.dumps(profile, ensure_ascii=False),
                profile_text,
            ))

            if len(rows) >= batch_size:
                con.executemany("INSERT INTO user_profiles VALUES (?,?,?,?)", rows)
                con.commit()
                rows.clear()

        if rows:
            con.executemany("INSERT INTO user_profiles VALUES (?,?,?,?)", rows)
            con.commit()
