"""저장된 체크포인트로 leaderboard_proxy NDCG@10만 평가 (재학습 없음).

실행 예시:
  python src/eval_proxy.py model=tisasrec
  python src/eval_proxy.py model=tisasrec
  python src/eval_proxy.py model=tisasrec ckpt_path=outputs/tisasrec/run001_260520/tuning/best.pt
  python src/eval_proxy.py model=cl4srec wandb.enabled=true wandb.name=cl4srec_proxy_eval
"""

import os
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

load_dotenv()


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    from data.dataset import load_data, build_vocab
    from models import build_model
    from util.paths import get_checkpoint_path
    from util.proxy_eval import evaluate_leaderboard_proxy

    orig_cwd = hydra.utils.get_original_cwd()
    cfg.data.data_dir = os.path.join(orig_cwd, cfg.data.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if cfg.train.amp == "bf16" else torch.float16

    # proxy는 기본 tuning ckpt (오버라이드: checkpoint.load_phase=full)
    ckpt_path = str(
        get_checkpoint_path(cfg, orig_cwd, for_training=False, load_phase="tuning")
    )
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=getattr(cfg.wandb, "name", None) or f"{cfg.model.name}_proxy_eval",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.wandb.tags) + ["proxy_eval"],
        )

    print("▶ 데이터 로드 중...")
    df = load_data(cfg)
    item2idx, _user2idx, idx2item = build_vocab(df)
    n_items = len(item2idx)
    print(f"  유저·아이템·행: {df['user_id'].nunique():,} / {n_items:,} / {len(df):,}")

    print(f"▶ 체크포인트 로드: {ckpt_path}")
    model = build_model(cfg.model, n_items).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if "ndcg" in ckpt:
        print(f"  (체크포인트 Val NDCG@10: {ckpt['ndcg']:.4f}, epoch {ckpt.get('epoch', '?')})")

    print("▶ leaderboard_proxy 평가 (Feb 23~29, cart+purchase GT)...")
    proxy_ndcg, n_gt = evaluate_leaderboard_proxy(
        model, df, item2idx, idx2item, cfg, device, amp_dtype
    )

    print(f"  GT 유저 수: {n_gt:,}")
    print(f"  Leaderboard proxy NDCG@10: {proxy_ndcg:.4f}")

    if cfg.wandb.enabled:
        wandb.log({
            "leaderboard_proxy/ndcg": proxy_ndcg,
            "leaderboard_proxy/gt_user_count": n_gt,
            "ckpt_path": ckpt_path,
        })
        wandb.finish()

    return proxy_ndcg


if __name__ == "__main__":
    main()
