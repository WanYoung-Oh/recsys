"""
FAISS 유사 유저 인덱스 빌드.

유저 행동 벡터 (47차원):
  - 17 L2 카테고리 × event_weight 합산 (정규화)
  - 30 Top 브랜드  × event_weight 합산 (정규화)
  → L2 정규화 후 내적 = 코사인 유사도

산출물:
  rag_data/user_neighbors.npy          (N, 20) int32  — faiss 행 번호
  rag_data/user_neighbors_meta.pkl     {row_idx: user_alias}
  rag_data/user_alias_to_row.pkl       {user_alias: row_idx}

Usage:
    # DB 유저(1000명) 기준 빌드 (빠름, 데모용)
    python src/mvp/build_user_vectors.py

    # 전체 638K 유저 빌드 (느림, ~10분)
    python src/mvp/build_user_vectors.py --full
"""
from __future__ import annotations

import argparse
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

EVENT_WEIGHT = {"view": 1.0, "cart": 25.0, "purchase": 50.0}

L2_CATS = [
    "belt","costume","dress","glove","jacket","jeans","jumper",
    "pajamas","scarf","shirt","shoes","shorts","skirt","sock",
    "trousers","tshirt","underwear",
]
TOP_BRANDS = [
    "respect","sony","xiaomi","baden","samsung","rieker","luminarc",
    "apple","defacto","nika","nike","starline","conceptclub","greyder",
    "intex","intel","lg","huawei","amd","montale","omron","strobbs",
    "kingston","chanel","bosch","versace","lenovo","lanvin","fassen","asrock",
]

L2_IDX   = {c: i      for i, c in enumerate(L2_CATS)}
BRAND_IDX = {b: i + len(L2_CATS) for i, b in enumerate(TOP_BRANDS)}
DIM = len(L2_CATS) + len(TOP_BRANDS)   # 47


def build_user_vectors(
    df: pd.DataFrame,
    ordered_aliases: list[str],
    alias_to_uid: dict[str, str],
) -> np.ndarray:
    """유저별 47-dim 행동 벡터 구성.

    Returns:
        ndarray (N, 47) float32, 행 순서 = ordered_aliases
    """
    # user_id → 벡터 누적
    uid_set = {alias_to_uid[ua] for ua in ordered_aliases if ua in alias_to_uid}

    print(f"  유저 {len(uid_set):,}명 벡터 계산 중...")
    t0 = time.time()

    vectors: dict[str, np.ndarray] = {
        uid: np.zeros(DIM, dtype=np.float32) for uid in uid_set
    }

    for _, row in tqdm(df[df["user_id"].isin(uid_set)].iterrows(),
                       total=len(df[df["user_id"].isin(uid_set)]),
                       desc="행 처리", unit="row"):
        uid = row["user_id"]
        w   = EVENT_WEIGHT.get(row["event_type"], 1.0)

        # L2 카테고리
        cat_code = str(row.get("category_code", "") or "")
        parts = cat_code.split(".")
        l2 = parts[1] if len(parts) >= 2 else ""
        if l2 in L2_IDX:
            vectors[uid][L2_IDX[l2]] += w

        # 브랜드
        brand = str(row.get("brand", "") or "").lower()
        if brand in BRAND_IDX:
            vectors[uid][BRAND_IDX[brand]] += w

    print(f"  벡터 계산 완료: {time.time()-t0:.1f}s")

    # ordered_aliases 순서로 행렬 조립
    mat = np.zeros((len(ordered_aliases), DIM), dtype=np.float32)
    for i, ua in enumerate(ordered_aliases):
        uid = alias_to_uid.get(ua)
        if uid and uid in vectors:
            mat[i] = vectors[uid]

    # L2 정규화
    faiss.normalize_L2(mat)
    return mat


def run_faiss(
    mat: np.ndarray,
    k: int = 20,
    batch_size: int = 5_000,
) -> np.ndarray:
    """IndexFlatIP 으로 Top-k 유사 유저 검색.

    Returns:
        neighbors (N, k) int32 — row index 기준
    """
    n, d = mat.shape
    print(f"  FAISS IndexFlatIP: {n:,} × {d}차원, k={k}")
    t0 = time.time()

    index = faiss.IndexFlatIP(d)
    index.add(mat)

    all_neighbors: list[np.ndarray] = []
    for start in tqdm(range(0, n, batch_size), desc="FAISS 검색", unit="batch"):
        chunk = mat[start : start + batch_size]
        _, I = index.search(chunk, k + 1)   # k+1: 자기 자신 제외 예정
        # 자기 자신(rank=0 = 본인)이 항상 1위이므로 제거
        neighbors_chunk = I[:, 1:]          # (batch, k)
        all_neighbors.append(neighbors_chunk.astype(np.int32))

    neighbors = np.vstack(all_neighbors)    # (N, k)
    print(f"  FAISS 검색 완료: {time.time()-t0:.1f}s")
    return neighbors


def save_artifacts(
    neighbors: np.ndarray,
    ordered_aliases: list[str],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "user_neighbors.npy"
    meta_path = out_dir / "user_neighbors_meta.pkl"
    row_path  = out_dir / "user_alias_to_row.pkl"

    np.save(npy_path, neighbors)
    with open(meta_path, "wb") as f:
        pickle.dump({i: ua for i, ua in enumerate(ordered_aliases)}, f)
    with open(row_path, "wb") as f:
        pickle.dump({ua: i for i, ua in enumerate(ordered_aliases)}, f)

    print(f"  → {npy_path}  ({neighbors.shape})")
    print(f"  → {meta_path}  ({len(ordered_aliases):,}명)")
    print(f"  → {row_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS 유사 유저 인덱스 빌드")
    parser.add_argument("--full", action="store_true",
                        help="전체 638K 유저 빌드 (느림). 기본값: DB 유저(1000명) 기준")
    args = parser.parse_args()

    rag = ROOT / "rag_data"
    db_path = rag / "user_profiles.db"

    # ── 유저 순서 결정 ─────────────────────────────────────────────────────────
    import json
    alias_json = rag / "id_aliases" / "user_alias.json"

    if args.full:
        # sample_submission 순서 기준 전체 유저
        sample_sub = ROOT / "data" / "sample_submission.csv"
        if sample_sub.exists():
            sub = pd.read_csv(sample_sub)
            all_uids   = list(sub["user_id"].drop_duplicates())
        else:
            df_tmp = pd.read_parquet(ROOT / "data" / "train.parquet",
                                     columns=["user_id"])
            all_uids = sorted(df_tmp["user_id"].unique().tolist())

        alias_map  = json.loads(alias_json.read_text(encoding="utf-8"))
        uid_to_alias = {uid: info["user_alias"] for uid, info in alias_map.items()}
        ordered_aliases = [uid_to_alias[uid] for uid in all_uids if uid in uid_to_alias]
        alias_to_uid    = {v["user_alias"]: uid for uid, v in alias_map.items()}
        print(f"전체 빌드 모드: {len(ordered_aliases):,}명")
    else:
        # DB 유저만 (빠름, 데모용)
        con = sqlite3.connect(db_path)
        rows = con.execute(
            "SELECT user_alias, user_id FROM user_profiles ORDER BY user_alias"
        ).fetchall()
        con.close()
        ordered_aliases = [r[0] for r in rows]
        alias_to_uid    = {r[0]: r[1] for r in rows}
        print(f"DB 유저 모드: {len(ordered_aliases):,}명")

    # ── train.parquet 로드 ─────────────────────────────────────────────────────
    print("\ntrain.parquet 로드 중...")
    df = pd.read_parquet(
        ROOT / "data" / "train.parquet",
        columns=["user_id", "category_code", "brand", "event_type"],
    )
    print(f"  {len(df):,}행 로드 완료")

    # ── 벡터 구성 → FAISS → 저장 ──────────────────────────────────────────────
    print("\n유저 벡터 구성 중...")
    mat = build_user_vectors(df, ordered_aliases, alias_to_uid)

    print("\nFAISS 인덱스 빌드 및 검색 중...")
    neighbors = run_faiss(mat, k=20)

    print("\n결과 저장 중...")
    save_artifacts(neighbors, ordered_aliases, rag)

    print(f"\nFAISS 빌드 완료  ({len(ordered_aliases):,}명)")


if __name__ == "__main__":
    main()
