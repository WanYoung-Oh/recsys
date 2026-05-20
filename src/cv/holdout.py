import pandas as pd
import numpy as np


VAL_START = pd.Timestamp("2020-02-09", tz="UTC")
VAL_END   = pd.Timestamp("2020-02-23", tz="UTC")  # exclusive


def make_holdout(df: pd.DataFrame, val_start=VAL_START, val_end=VAL_END):
    """시간 기반 1-Fold Holdout 분할.
    반환: (train_df, val_df) — train은 val_start 이전, val은 [val_start, val_end) 구간
    """
    train_df = df[df["event_time"] < val_start].copy()
    val_df   = df[(df["event_time"] >= val_start) & (df["event_time"] < val_end)].copy()
    return train_df, val_df


def build_gt(val_df: pd.DataFrame, mode: str = "cart_purchase") -> dict:
    """val_df로부터 정답 딕셔너리 생성.

    mode:
        "cart_purchase"  — cart + purchase 이벤트 (튜닝 기준, ~1,072명)
        "purchase_only"  — purchase 이벤트만 (보조 지표, ~37명)
    """
    if mode == "purchase_only":
        src = val_df[val_df["event_type"] == "purchase"]
    else:
        src = val_df[val_df["event_type"].isin(["cart", "purchase"])]
    return src.groupby("user_id")["item_id"].apply(set).to_dict()


def build_leaderboard_proxy_gt(df: pd.DataFrame) -> dict:
    """Feb 23~29 실구매 집중 구간 GT — 참고용, 튜닝 미반영."""
    proxy_start = pd.Timestamp("2020-02-23", tz="UTC")
    proxy_end   = pd.Timestamp("2020-03-01", tz="UTC")
    src = df[
        (df["event_time"] >= proxy_start) &
        (df["event_time"] <  proxy_end) &
        (df["event_type"].isin(["cart", "purchase"]))
    ]
    return src.groupby("user_id")["item_id"].apply(set).to_dict()
