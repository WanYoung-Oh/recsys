"""학습된 모델 로드 → 앙상블 추론 → 제출 CSV 저장.

실행 예시:
  python src/submit.py model=tisasrec
  python src/submit.py model=tisasrec run_id=run001
  python src/submit.py ensemble=rank  # 앙상블 모드 (ckpt_paths 지정 필요)
"""

import os
from pathlib import Path

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    from data.dataset import load_data, build_vocab, build_sequences, build_val_sequences
    from models       import build_model
    from inference    import generate_predictions, generate_submission_long, validate_submission

    orig_cwd = hydra.utils.get_original_cwd()
    cfg.data.data_dir = os.path.join(orig_cwd, cfg.data.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if cfg.train.amp == "bf16" else torch.float16

    print("▶ 데이터 로드 중...")
    df = load_data(cfg)
    item2idx, user2idx, idx2item = build_vocab(df)
    n_items = len(item2idx)
    all_user_ids = list(user2idx.keys())

    # Full-train: 전체 기간 (cv=none 시 df 전체)
    seqs     = build_sequences(df, item2idx, cfg.model.max_seq_len)
    val_seqs = build_val_sequences(seqs, cfg.model.max_seq_len)

    from util.paths import get_checkpoint_path

    ckpt_path = str(get_checkpoint_path(cfg, orig_cwd, for_training=False))
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    print(f"▶ {cfg.model.name} 로드: {ckpt_path}")
    model = build_model(cfg.model, n_items).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    is_tisasrec = cfg.model.name == "tisasrec"
    if is_tisasrec:
        from data.features import build_time_seq
        val_times = build_time_seq(seqs, cfg.model.max_seq_len)
    else:
        val_times = None

    print("▶ 추론 중...")
    preds = generate_predictions(
        model, all_user_ids, val_seqs, idx2item, device,
        batch_size=cfg.train.eval_batch_size,
        amp_dtype=amp_dtype,
        time_sequences=val_times,
    )

    sub_df = generate_submission_long(preds, all_user_ids)
    validate_submission(sub_df, n_users=len(all_user_ids))

    out_path = os.path.join(orig_cwd, f"outputs/submission_{cfg.model.name}.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"▶ 제출 파일 저장: {out_path}  ({len(sub_df):,}행)")


if __name__ == "__main__":
    main()
