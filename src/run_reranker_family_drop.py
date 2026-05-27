import os
import re
import json
import runpy
from pathlib import Path

import pandas as pd
import lightgbm as lgb

ROOT    = Path("/data/ephemeral/home/recsys")
SRC     = ROOT / "src" / "train_reranker_lgbm.py"
OUT_DIR = ROOT / "outputs" / "ablation_reranker"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DROP_FAMILY = os.getenv("DROP_FAMILY", "").strip().lower()

DROP_PATTERNS = {
    "popularity": [
        r"item_recent\d+_pop",
        r"item_pop",
        r"popularity",
        r"freq",
        r"frequency",
        r"global_cnt",
        r"item_cnt",
        r"trend",
        r"hot",
    ],
    "tifu_rank": [
        r"tifu",
        r"tifu_knn",
        r"rank_tifu",
        r"rr_tifu",
        r"in_tifu",
    ],
    "user_activity": [
        r"user_total_cnt",
        r"user_recent\d+_cnt",
        r"user_.*cnt",
        r"active_days",
        r"n_events",
        r"n_sessions",
        r"hist_len",
        r"history_len",
        r"user_activity",
    ],
}

if not DROP_FAMILY:
    raise SystemExit(
        f"[ERROR] DROP_FAMILY 환경변수가 설정되지 않았습니다. "
        f"예: DROP_FAMILY=popularity python run_reranker_family_drop.py\n"
        f"선택 가능 값: {list(DROP_PATTERNS.keys())}"
    )
if DROP_FAMILY not in DROP_PATTERNS:
    raise SystemExit(
        f"[ERROR] DROP_FAMILY must be one of {list(DROP_PATTERNS.keys())}, got: {DROP_FAMILY!r}"
    )

regexes          = [re.compile(p, re.I) for p in DROP_PATTERNS[DROP_FAMILY]]
dropped_cols_seen = []
seen_sets         = set()
LAST_FEATURE_COLUMNS = None

OrigDataFrame = pd.core.frame.DataFrame


def _match_cols(columns):
    return [str(c) for c in columns if any(r.search(str(c)) for r in regexes)]


def _drop_if_df(X, stage="unknown"):
    global dropped_cols_seen, seen_sets
    if isinstance(X, OrigDataFrame):
        cols = _match_cols(X.columns)
        key  = (stage, tuple(sorted(cols)))
        if cols and key not in seen_sets:
            seen_sets.add(key)
            dropped_cols_seen.extend(cols)
            print(f"[ABLAT] family={DROP_FAMILY} stage={stage} drop_count={len(cols)}", flush=True)
            print(f"[ABLAT] columns={cols}", flush=True)
        if cols:
            return X.drop(columns=cols, errors="ignore")
    return X


# ── pandas.DataFrame 패치 ──────────────────────────────────────────────────

class _PatchedDFMeta(type):
    def __instancecheck__(cls, instance):
        return isinstance(instance, OrigDataFrame)

    def __subclasscheck__(cls, subclass):
        try:
            return issubclass(subclass, OrigDataFrame)
        except Exception:
            return False


class PatchedDataFrame(OrigDataFrame, metaclass=_PatchedDFMeta):
    def __init__(self, data=None, *args, **kwargs):
        global LAST_FEATURE_COLUMNS

        if isinstance(data, dict) and "feature" in data and "importance" in data:
            try:
                feat = list(data["feature"])
                imp  = list(data["importance"])
                lf, li = len(feat), len(imp)

                if lf != li:
                    if LAST_FEATURE_COLUMNS is not None and len(LAST_FEATURE_COLUMNS) == li:
                        print(
                            f"[ABLAT] align feat_imp using LAST_FEATURE_COLUMNS: "
                            f"feature={lf}, importance={li}",
                            flush=True,
                        )
                        data = dict(data)
                        data["feature"]    = list(LAST_FEATURE_COLUMNS)
                        data["importance"] = imp
                    else:
                        n = min(lf, li)
                        print(
                            f"[ABLAT] trim feat_imp lengths: feature={lf}, importance={li} -> {n}",
                            flush=True,
                        )
                        data = dict(data)
                        data["feature"]    = feat[:n]
                        data["importance"] = imp[:n]
            except Exception as e:
                print(f"[ABLAT] feat_imp patch skipped: {e}", flush=True)

        super().__init__(data, *args, **kwargs)


pd.DataFrame = PatchedDataFrame


# ── LightGBM LGBMRanker 패치 ───────────────────────────────────────────────

orig_ranker_fit     = lgb.LGBMRanker.fit
orig_ranker_predict = lgb.LGBMRanker.predict


def patched_ranker_fit(self, X, y, *args, **kwargs):
    global LAST_FEATURE_COLUMNS

    X = _drop_if_df(X, "fit")
    if isinstance(X, OrigDataFrame):
        LAST_FEATURE_COLUMNS = list(X.columns)

    if "eval_set" in kwargs and kwargs["eval_set"] is not None:
        new_eval = []
        for pair in kwargs["eval_set"]:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                ex, ey = pair
                ex = _drop_if_df(ex, "eval_set")
                new_eval.append((ex, ey))
            else:
                new_eval.append(pair)
        kwargs["eval_set"] = new_eval

    if "feature_name" in kwargs and isinstance(X, OrigDataFrame):
        kwargs["feature_name"] = list(X.columns)

    return orig_ranker_fit(self, X, y, *args, **kwargs)


def patched_ranker_predict(self, X, *args, **kwargs):
    X = _drop_if_df(X, "predict")
    return orig_ranker_predict(self, X, *args, **kwargs)


lgb.LGBMRanker.fit     = patched_ranker_fit
lgb.LGBMRanker.predict = patched_ranker_predict

# ── 실행 ───────────────────────────────────────────────────────────────────

print(f"[ABLAT] start family={DROP_FAMILY}", flush=True)
print(f"[ABLAT] source={SRC}", flush=True)

try:
    runpy.run_path(str(SRC), run_name="__main__")
finally:
    manifest = {
        "drop_family":               DROP_FAMILY,
        "patterns":                  DROP_PATTERNS[DROP_FAMILY],
        "dropped_columns_unique":    sorted(set(dropped_cols_seen)),
        "dropped_column_count_unique": len(set(dropped_cols_seen)),
        "last_feature_columns":      LAST_FEATURE_COLUMNS,
    }
    out_json = OUT_DIR / f"drop_manifest_{DROP_FAMILY}.json"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ABLAT] manifest_saved={out_json}", flush=True)
