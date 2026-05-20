"""TIFU-KNN 예측 생성 스크립트 (non-neural).

전체 학습 데이터 기준 TIFU-KNN 예측을 생성해 outputs/tifu_knn/preds.pkl로 저장.
이 파일은 ensemble_submit.py가 tifu_knn 모델로 직접 로드할 수 있다.

실행:
  python src/train_tifu.py
  python src/train_tifu.py model.tifu_group_count=7 model.tifu_decay_within=0.9
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.data.data_dir = os.path.join(orig_cwd, cfg.data.data_dir)

    from data.dataset import load_data, build_vocab, build_sequences
    from models.tifu_knn import TIFUKNN
    from omegaconf import OmegaConf

    ensemble_cfg = OmegaConf.load(ROOT / "conf" / "ensemble" / "rank.yaml")
    group_count  = int(ensemble_cfg.get("tifu_group_count", 7))
    decay_within = float(ensemble_cfg.get("tifu_decay_within", 0.9))
    decay_across = float(ensemble_cfg.get("tifu_decay_across", 0.7))

    print("▶ 데이터 로드 중...")
    df = load_data(cfg)
    item2idx, user2idx, idx2item = build_vocab(df)
    all_user_ids = list(user2idx.keys())
    print(f"  유저 {len(user2idx):,} / 아이템 {len(item2idx):,}")

    print("▶ 시퀀스 빌드 중...")
    seqs = build_sequences(df, item2idx, cfg.model.max_seq_len)

    print(f"▶ TIFU-KNN 피팅 중... (group={group_count}, "
          f"r_w={decay_within}, r_a={decay_across})")
    tifu = TIFUKNN(group_count=group_count,
                   decay_within=decay_within,
                   decay_across=decay_across)
    tifu.fit(seqs, idx2item)

    print("▶ 예측 생성 중...")
    preds = tifu.predict_all(all_user_ids, top_k=10)

    out_dir = ROOT / "outputs" / "tifu_knn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "preds.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(preds, f)
    print(f"▶ 저장 완료: {out_path}  ({len(preds):,}명)")


if __name__ == "__main__":
    main()
