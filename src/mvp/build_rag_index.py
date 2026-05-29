#!/usr/bin/env python3
"""
오프라인 RAG 인덱스 빌드 스크립트 (Phase 0).

산출물:
    rag_data/
    ├── id_aliases/user_alias.json   # user_id → {user_alias, display_name}
    ├── id_aliases/item_alias.json   # item_id → {item_alias, l2, l3, brand, ...}
    ├── item_catalog.json            # item_alias → 전체 메타 + 통계
    ├── user_profiles.db             # SQLite (user_alias PK)
    ├── user_recommendations.json    # user_alias → [item_alias, ...]
    └── recency_pool.json            # l2 → [item_alias, ...] (submission 인기도 순)

Usage:
    # 빠른 검증 (1000명, FAISS 제외)
    python src/mvp/build_rag_index.py --submission outputs/submission_reranker_lgbm.csv \\
        --skip-faiss --sample 1000

    # 전체 빌드 (FAISS 제외)
    python src/mvp/build_rag_index.py --submission outputs/submission_reranker_lgbm.csv \\
        --skip-faiss
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.mvp.id_alias import (
    build_item_aliases,
    build_user_aliases,
    parse_category,
    make_price_bucket_fn,
)
from src.mvp.user_profile import build_profiles_db

# ── 환경 변수 ────────────────────────────────────────────────────────────────
PRICE_SCALE  = int(os.getenv("PRICE_SCALE", "1000"))
RECENCY_DAYS = int(os.getenv("RECENCY_DAYS", "14"))
RAG_DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "rag_data"))


def format_price_display(price_median: float) -> str:
    krw = int(round(price_median * PRICE_SCALE, -2))
    return f"{krw:,}원"


# ── 빌드 단계 ─────────────────────────────────────────────────────────────────

def _load_train(train_path: str | Path) -> pd.DataFrame:
    print("train.parquet 로드 중...")
    t = time.time()
    df = pd.read_parquet(train_path)
    df["event_time"] = pd.to_datetime(df["event_time"], utc=True)
    df = df.sort_values(["user_id", "event_time"]).reset_index(drop=True)
    print(f"  로드 완료: {len(df):,}행, {time.time()-t:.1f}s")
    return df


def _build_item_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """item_id별 집계 → DataFrame (index=item_id)."""
    print("아이템 카탈로그 집계 중...")
    t = time.time()

    def safe_mode(x):
        m = x.dropna().mode()
        return str(m.iloc[0]) if len(m) > 0 else "unknown"

    agg = df.groupby("item_id", sort=False).agg(
        category_code=("category_code", safe_mode),
        brand=("brand", safe_mode),
        price_median=("price", "median"),
        view_count=("event_type", lambda x: int((x == "view").sum())),
        cart_count=("event_type", lambda x: int((x == "cart").sum())),
        purchase_count=("event_type", lambda x: int((x == "purchase").sum())),
        last_seen_date=("event_time", "max"),
    )
    print(f"  집계 완료: {len(agg):,}개 아이템, {time.time()-t:.1f}s")
    return agg


def _build_user_recs(
    sub_df: pd.DataFrame,
    user_alias_map: dict[str, dict],
    item_alias_map: dict[str, str],
    sample_users: set[str] | None,
) -> dict[str, list[str]]:
    """submission → {user_alias: [item_alias, ...]} (Top-10 순서 유지)."""
    print("추천 결과 alias 변환 중...")
    t = time.time()
    result: dict[str, list[str]] = {}

    for uid, group in sub_df.groupby("user_id", sort=False):
        if sample_users is not None and uid not in sample_users:
            continue
        user_info = user_alias_map.get(uid)
        if not user_info:
            continue
        aliases = [
            item_alias_map[iid]
            for iid in group["item_id"].tolist()
            if iid in item_alias_map
        ]
        result[user_info["user_alias"]] = aliases

    print(f"  변환 완료: {len(result):,}명, {time.time()-t:.1f}s")
    return result


def _build_recency_pool(
    item_catalog_dict: dict[str, dict],
    item_alias_map: dict[str, str],
    sub_df: pd.DataFrame,
    data_end_date: pd.Timestamp,
) -> dict[str, list[str]]:
    """카테고리별 신상품 풀.

    기준: last_seen_date >= data_end_date - RECENCY_DAYS
    정렬: submission 등장 횟수 내림차순 → last_seen_date 내림차순
    """
    print(f"Recency pool 빌드 중 (최근 {RECENCY_DAYS}일)...")
    cutoff_str = str((data_end_date - pd.Timedelta(days=RECENCY_DAYS)).date())

    # 아이템별 submission 등장 횟수
    sub_counts = sub_df["item_id"].value_counts().to_dict()

    pool: dict[str, list[tuple]] = {}  # l2 → [(alias, sub_count, last_seen_date)]
    for item_id, meta in item_catalog_dict.items():
        if meta["last_seen_date"] < cutoff_str:
            continue
        alias = item_alias_map.get(item_id)
        if not alias:
            continue
        l2 = meta["l2"]
        pool.setdefault(l2, []).append(
            (alias, sub_counts.get(item_id, 0), meta["last_seen_date"])
        )

    result: dict[str, list[str]] = {}
    for l2, items in pool.items():
        items.sort(key=lambda x: (x[1], x[2]), reverse=True)
        result[l2] = [alias for alias, _, _ in items]

    total = sum(len(v) for v in result.values())
    print(f"  카테고리 {len(result)}개, 아이템 {total}개 포함")
    return result


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MVP RAG 인덱스 빌드")
    parser.add_argument("--submission", required=True, help="submission CSV 경로")
    parser.add_argument("--skip-faiss", action="store_true", help="FAISS 빌드 생략")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="처리할 유저 수 (submission 순서 기준). None이면 전체"
    )
    args = parser.parse_args()

    total_start = time.time()
    out_dir = ROOT / RAG_DATA_DIR
    (out_dir / "id_aliases").mkdir(parents=True, exist_ok=True)

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    train_df = _load_train(ROOT / os.getenv("TRAIN_PARQUET", "data/train.parquet"))
    data_end_date = train_df["event_time"].max()
    print(f"  데이터 기준일: {data_end_date.date()}")

    sub_df = pd.read_csv(ROOT / args.submission)
    print(f"  submission: {len(sub_df):,}행")

    # sample_submission.csv 유저 순서로 alias 부여
    sample_sub_path = ROOT / "data" / "sample_submission.csv"
    if sample_sub_path.exists():
        ordered_users = list(pd.read_csv(sample_sub_path)["user_id"].drop_duplicates())
    else:
        ordered_users = list(sub_df["user_id"].drop_duplicates())
    print(f"  유저 순서 기준: {'sample_submission.csv' if sample_sub_path.exists() else 'submission CSV'}")

    sample_user_ids: set[str] | None = None
    if args.sample:
        sample_user_ids = set(ordered_users[: args.sample])
        print(f"  샘플 모드: {args.sample}명만 처리")

    # ── 2. 아이템 카탈로그 ────────────────────────────────────────────────────
    catalog_df = _build_item_catalog(train_df)

    # 가격 분위수 계산
    p33, p66 = catalog_df["price_median"].quantile([0.33, 0.66]).values
    print(f"  가격 버킷 기준: low < {p33:.2f} ≤ mid < {p66:.2f} ≤ high")
    price_bucket_fn = make_price_bucket_fn(p33, p66)

    # item_id → meta dict 구성
    item_catalog_dict: dict[str, dict] = {}
    for item_id, row in catalog_df.iterrows():
        l2, l3 = parse_category(row["category_code"])
        bucket = price_bucket_fn(float(row["price_median"]))
        item_catalog_dict[str(item_id)] = {
            "l2": l2,
            "l3": l3,
            "category_code": row["category_code"],
            "brand": row["brand"],
            "price_median": float(row["price_median"]),
            "price_bucket": bucket,
            "view_count": int(row["view_count"]),
            "cart_count": int(row["cart_count"]),
            "purchase_count": int(row["purchase_count"]),
            "last_seen_date": str(row["last_seen_date"].date()),
        }

    # ── 3. Alias 생성 ─────────────────────────────────────────────────────────
    print("User alias 생성 중...")
    t = time.time()
    user_alias_map = build_user_aliases(ordered_users)
    print(f"  {len(user_alias_map):,}명 alias 생성, {time.time()-t:.1f}s")

    print("Item alias (Semantic ID) 생성 중...")
    t = time.time()
    item_alias_map = build_item_aliases(catalog_df, price_bucket_fn)
    print(f"  {len(item_alias_map):,}개 alias 생성, {time.time()-t:.1f}s")

    # ── 4. id_aliases JSON 저장 ───────────────────────────────────────────────
    print("id_aliases 저장 중...")

    user_alias_path = out_dir / "id_aliases" / "user_alias.json"
    with open(user_alias_path, "w", encoding="utf-8") as f:
        json.dump(user_alias_map, f, ensure_ascii=False, indent=2)
    print(f"  → {user_alias_path}")

    item_alias_full: dict[str, dict] = {}
    for item_id, alias in item_alias_map.items():
        meta = item_catalog_dict[item_id]
        item_alias_full[item_id] = {
            "item_alias": alias,
            "l2": meta["l2"],
            "l3": meta["l3"],
            "category_code": meta["category_code"],
            "brand": meta["brand"],
            "price_bucket": meta["price_bucket"],
            "price_median": meta["price_median"],
            "price_display": format_price_display(meta["price_median"]),
        }

    item_alias_path = out_dir / "id_aliases" / "item_alias.json"
    with open(item_alias_path, "w", encoding="utf-8") as f:
        json.dump(item_alias_full, f, ensure_ascii=False, indent=2)
    print(f"  → {item_alias_path}")

    # ── 5. item_catalog.json 저장 (item_alias 기준) ───────────────────────────
    print("item_catalog.json 저장 중...")
    item_catalog_alias: dict[str, dict] = {}
    for item_id, alias in item_alias_map.items():
        meta = item_catalog_dict[item_id]
        item_catalog_alias[alias] = {
            "item_id": item_id,
            "l2": meta["l2"],
            "l3": meta["l3"],
            "category_code": meta["category_code"],
            "brand": meta["brand"],
            "price_bucket": meta["price_bucket"],
            "price_median": meta["price_median"],
            "price_display": format_price_display(meta["price_median"]),
            "view_count": meta["view_count"],
            "cart_count": meta["cart_count"],
            "purchase_count": meta["purchase_count"],
            "last_seen_date": meta["last_seen_date"],
        }

    catalog_path = out_dir / "item_catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(item_catalog_alias, f, ensure_ascii=False, indent=2)
    print(f"  → {catalog_path}")

    # ── 6. user_profiles.db 생성 ─────────────────────────────────────────────
    db_path = out_dir / "user_profiles.db"
    build_profiles_db(
        train_df=train_df,
        item_alias_map=item_alias_map,
        item_meta=item_catalog_dict,
        user_alias_map=user_alias_map,
        db_path=db_path,
        sample_users=sample_user_ids,
    )
    print(f"  → {db_path}")

    # ── 7. user_recommendations.json 저장 ────────────────────────────────────
    user_recs = _build_user_recs(sub_df, user_alias_map, item_alias_map, sample_user_ids)
    recs_path = out_dir / "user_recommendations.json"
    with open(recs_path, "w", encoding="utf-8") as f:
        json.dump(user_recs, f, ensure_ascii=False, indent=2)
    print(f"  → {recs_path}")

    # ── 8. recency_pool.json 저장 ─────────────────────────────────────────────
    recency_pool = _build_recency_pool(
        item_catalog_dict, item_alias_map, sub_df, data_end_date
    )
    pool_path = out_dir / "recency_pool.json"
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(recency_pool, f, ensure_ascii=False, indent=2)
    print(f"  → {pool_path}")

    # ── 9. FAISS (선택) ───────────────────────────────────────────────────────
    if args.skip_faiss:
        print("FAISS 빌드 생략 (--skip-faiss)")
    else:
        print("FAISS 빌드는 Phase 4에서 구현 예정")

    print(f"\n빌드 완료: 총 {time.time()-total_start:.1f}s")
    print(f"출력 디렉토리: {out_dir}")


if __name__ == "__main__":
    main()
