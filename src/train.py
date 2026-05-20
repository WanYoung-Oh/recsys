"""단일 모델 학습 + 1-Fold Holdout CV 평가 진입점.

실행 예시:
  python src/train.py model=sasrec          # batch 4096 (conf/model/sasrec.yaml)
  python src/train.py model=tisasrec        # batch 1024 (conf/model/tisasrec.yaml)
  python src/train.py model=cl4srec train.loss_weights.cart=10
  python src/train.py model=tisasrec train.train_batch_size=512   # OOM 시 batch 오버라이드
  python src/train.py model=sasrec cv=none          # CV 없이 전체 학습
  python src/train.py model=sasrec cv.run_leaderboard_proxy=true
  python src/eval_proxy.py model=tisasrec   # 최신 run/tuning ckpt로 proxy 평가
"""

import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(model, optimizer, epoch: int, ndcg: float, path: str):
    torch.save(
        {"epoch": epoch, "ndcg": ndcg,
         "model": model.state_dict(),
         "optimizer": optimizer.state_dict()},
        path,
    )


def train_epoch(model, loader, optimizer, scaler, device, amp_dtype, model_name, epoch=0):
    model.train()
    total_loss = 0.0
    is_tisasrec = model_name == "tisasrec"
    is_saferec  = model_name == "saferec"
    is_mbstr    = model_name == "mbstr"

    pbar = tqdm(loader, desc=f"  Ep{epoch:3d}", leave=False, dynamic_ncols=True)
    for batch in pbar:
        optimizer.zero_grad()

        if is_tisasrec:
            seq, times, pos, neg, weights = [b.to(device) for b in batch]
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model.calculate_loss(seq, times, pos, neg, weights)
        elif is_saferec:
            seq, freq_seq, pos, neg, weights = [b.to(device) for b in batch]
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model.calculate_loss(seq, freq_seq, pos, neg, weights)
        elif is_mbstr:
            seq, behavior_seq, pos, neg, weights = [b.to(device) for b in batch]
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model.calculate_loss(seq, behavior_seq, pos, neg, weights)
        else:
            seq, pos, neg, weights = [b.to(device) for b in batch]
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                loss = model.calculate_loss(seq, pos, neg, weights)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{total_loss / pbar.n:.4f}")

    return total_loss / len(loader)


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if cfg.train.amp == "bf16" else torch.float16

    # ── wandb 초기화 ──────────────────────────────────────────────
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            name=f"{cfg.model.name}_seed{cfg.seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=list(cfg.wandb.tags),
        )

    # ── 데이터 로드 ───────────────────────────────────────────────
    from data.dataset import load_data, build_vocab, build_sequences, build_val_sequences
    from cv.holdout   import make_holdout, build_gt
    from models       import build_model
    from metrics      import evaluate

    # Hydra가 작업 디렉터리를 outputs/...로 바꾸므로 원본 경로는 original_cwd 사용
    orig_cwd = hydra.utils.get_original_cwd()
    cfg.data.data_dir = os.path.join(orig_cwd, cfg.data.data_dir)

    print("▶ 데이터 로드 중...")
    df = load_data(cfg)
    item2idx, user2idx, idx2item = build_vocab(df)
    n_items = len(item2idx)
    all_user_ids = list(user2idx.keys())
    print(f"  유저 {len(user2idx):,} / 아이템 {n_items:,} / 인터랙션 {len(df):,}")

    event_weights = dict(cfg.train.loss_weights)

    # ── 1-Fold Holdout CV ──────────────────────────────────────────
    if cfg.cv.enabled:
        train_df, val_df = make_holdout(df)
        gt_cp   = build_gt(val_df, mode="cart_purchase")   # 튜닝 기준
        gt_pu   = build_gt(val_df, mode="purchase_only")   # 보조 지표
        val_user_ids = np.array(list(val_df["user_id"].unique()))
        print(f"  Train: {len(train_df):,}행 / Val: {len(val_df):,}행")
        print(f"  GT 유저 수 — cart+purchase: {len(gt_cp):,} / purchase only: {len(gt_pu):,}")
    else:
        train_df = df
        gt_cp = gt_pu = {}
        val_user_ids = np.array([])

    # 시퀀스 빌드 (leakage 방지: train_df 기준)
    print("▶ 시퀀스 빌드 중...")
    seqs = build_sequences(train_df, item2idx, cfg.model.max_seq_len)
    val_seqs = build_val_sequences(seqs, cfg.model.max_seq_len)

    model_name  = cfg.model.name
    is_tisasrec = model_name == "tisasrec"
    is_saferec  = model_name == "saferec"
    is_mbstr    = model_name == "mbstr"

    val_times = val_freqs = val_behaviors = None

    if is_tisasrec:
        from data.features import SeqTrainDatasetWithTime, build_time_seq
        dataset   = SeqTrainDatasetWithTime(seqs, n_items, event_weights, cfg.model.max_seq_len)
        val_times = build_time_seq(seqs, cfg.model.max_seq_len)
    elif is_saferec:
        from data.features import (SeqTrainDatasetWithFreq, build_freq_seq,
                                   compute_item_freq)
        item_freq  = compute_item_freq(train_df, item2idx, cfg.model.n_freq_buckets)
        dataset    = SeqTrainDatasetWithFreq(seqs, n_items, event_weights,
                                             cfg.model.max_seq_len, item_freq)
        val_freqs  = build_freq_seq(seqs, cfg.model.max_seq_len, item_freq)
    elif is_mbstr:
        from data.features import SeqTrainDatasetWithBehavior, build_behavior_seq
        dataset       = SeqTrainDatasetWithBehavior(seqs, n_items, event_weights,
                                                    cfg.model.max_seq_len)
        val_behaviors = build_behavior_seq(seqs, cfg.model.max_seq_len)
    else:
        from data.dataset import SeqTrainDataset
        dataset = SeqTrainDataset(seqs, n_items, event_weights, cfg.model.max_seq_len)

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.train_batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=False,
    )
    print(f"  학습 샘플 수: {len(dataset):,} / 배치 수: {len(loader):,}")

    # ── 모델·옵티마이저 ───────────────────────────────────────────
    model = build_model(cfg.model, n_items).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"▶ {cfg.model.name} 파라미터: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler()

    from util.paths import get_checkpoint_path, get_train_phase, resolve_run_dir

    run_dir = resolve_run_dir(
        cfg, orig_cwd,
        mode="train_tuning" if cfg.cv.enabled else "train_full",
    )
    phase = get_train_phase(cfg)
    best_ckpt_path = get_checkpoint_path(cfg, orig_cwd, for_training=True)
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    best_ckpt = str(best_ckpt_path)
    print(f"  Run: {run_dir.name} / {phase} → {best_ckpt}")

    # ── 학습 루프 ──────────────────────────────────────────────────
    best_ndcg, best_epoch, patience_counter = 0.0, 0, 0
    patience = cfg.train.early_stopping_patience

    print("▶ 학습 시작")
    epoch_bar = tqdm(range(cfg.train.epochs), desc="Epochs", dynamic_ncols=True)
    for epoch in epoch_bar:
        t0 = time.time()
        loss = train_epoch(model, loader, optimizer, scaler, device, amp_dtype, cfg.model.name, epoch=epoch)
        elapsed = time.time() - t0

        log_dict = {"train_loss": loss, "epoch": epoch, "epoch_time_sec": elapsed}

        do_eval = cfg.cv.enabled  # 매 에포크 evaluate (early stopping 정확도 보장)
        if do_eval:
            eval_kwargs = dict(
                amp_dtype=amp_dtype,
                val_times=val_times,
                val_freqs=val_freqs,
                val_behaviors=val_behaviors,
            )
            ndcg_cp = evaluate(
                model, val_seqs, val_user_ids, gt_cp, idx2item,
                device, cfg.train.eval_batch_size, **eval_kwargs,
            )
            ndcg_pu = evaluate(
                model, val_seqs, val_user_ids, gt_pu, idx2item,
                device, cfg.train.eval_batch_size, **eval_kwargs,
            )
            log_dict.update({
                "val/ndcg_cart_purchase": ndcg_cp,
                "val/ndcg_purchase_only": ndcg_pu,
                "val/gt_user_count":      len(gt_cp),
            })
            tqdm.write(
                f"  Epoch {epoch:3d} | loss {loss:.4f} | "
                f"ndcg_cp {ndcg_cp:.4f} | ndcg_pu {ndcg_pu:.4f} | {elapsed:.1f}s"
            )
            epoch_bar.set_postfix(
                loss=f"{loss:.4f}", ndcg=f"{ndcg_cp:.4f}", best=f"{best_ndcg:.4f}"
            )

            if ndcg_cp > best_ndcg:
                best_ndcg, best_epoch = ndcg_cp, epoch
                patience_counter = 0
                save_checkpoint(model, optimizer, epoch, ndcg_cp, best_ckpt)
            else:
                patience_counter += 1
                if patience > 0 and patience_counter >= patience:
                    tqdm.write(f"  Early stopping at epoch {epoch} (best {best_epoch}: {best_ndcg:.4f})")
                    break
        else:
            # cv=none: 매 에포크 loss만 출력, 마지막 에포크 후 체크포인트 저장
            tqdm.write(f"  Epoch {epoch:3d} | loss {loss:.4f} | {elapsed:.1f}s")
            epoch_bar.set_postfix(loss=f"{loss:.4f}")

        if cfg.wandb.enabled:
            wandb.log(log_dict)

    # cv=none: 전체 학습 완료 후 최종 체크포인트 저장
    if not cfg.cv.enabled:
        save_checkpoint(model, optimizer, epoch, 0.0, best_ckpt)
        print(f"  체크포인트 저장: {best_ckpt}")

    # ── leaderboard proxy (선택) ───────────────────────────────────
    if cfg.cv.enabled and cfg.cv.run_leaderboard_proxy:
        from util.proxy_eval import evaluate_leaderboard_proxy
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        proxy_ndcg, n_gt = evaluate_leaderboard_proxy(
            model, df, item2idx, idx2item, cfg, device, amp_dtype
        )
        print(f"  Leaderboard proxy NDCG@10: {proxy_ndcg:.4f} (GT {n_gt:,}명)")
        if cfg.wandb.enabled:
            wandb.log({
                "leaderboard_proxy/ndcg": proxy_ndcg,
                "leaderboard_proxy/gt_user_count": n_gt,
            })

    if cfg.cv.enabled and cfg.wandb.enabled:
        wandb.log({"val/best_ndcg_cart_purchase": best_ndcg, "best_epoch": best_epoch})

    if cfg.wandb.enabled:
        wandb.finish()

    print(f"▶ 완료 — best NDCG@10 (cart+purchase): {best_ndcg:.4f} at epoch {best_epoch}")
    return best_ndcg


if __name__ == "__main__":
    main()
