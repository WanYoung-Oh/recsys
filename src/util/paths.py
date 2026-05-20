"""체크포인트 경로: outputs/<model>/runNNN_YYMMDD/{tuning|full}/best.pt"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from omegaconf import DictConfig

RUN_DIR_RE = re.compile(r"^run(\d+)_(\d{6})$")


def get_run_date(cfg: DictConfig) -> str:
    rd = cfg.get("run_date")
    if rd is not None and str(rd).strip():
        return str(rd).strip()
    return datetime.now().strftime("%y%m%d")


def _model_root(orig_cwd: str, model_name: str) -> Path:
    return Path(orig_cwd) / "outputs" / model_name


def scan_run_dirs(model_root: Path) -> list[tuple[int, Path]]:
    """(run_number, path) 오름차순."""
    if not model_root.is_dir():
        return []
    found: list[tuple[int, Path]] = []
    for p in model_root.iterdir():
        if not p.is_dir():
            continue
        m = RUN_DIR_RE.match(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found, key=lambda x: x[0])


def next_run_dir(model_root: Path, run_date: str) -> Path:
    runs = scan_run_dirs(model_root)
    n = (runs[-1][0] + 1) if runs else 1
    return model_root / f"run{n:03d}_{run_date}"


def _find_run_by_id(model_root: Path, run_id: str, run_date: str) -> Path:
    """run_id=run001 또는 run001_260520."""
    rid = str(run_id).strip()
    if RUN_DIR_RE.match(rid):
        return model_root / rid
    if not rid.startswith("run"):
        rid = f"run{rid}"
    prefix = rid if rid.endswith(run_date) else rid
    matches = [p for _, p in scan_run_dirs(model_root) if p.name.startswith(prefix + "_") or p.name == prefix]
    if matches:
        return matches[-1]
    return model_root / f"{rid}_{run_date}"


def _latest_run_with_phase(runs: list[tuple[int, Path]], phase: str) -> Path | None:
    """{phase}/best.pt 가 실제로 존재하는 가장 최신 run 반환."""
    for _, run_dir in reversed(runs):
        if (run_dir / phase / "best.pt").is_file():
            return run_dir
    return None


def resolve_run_dir(
    cfg: DictConfig,
    orig_cwd: str,
    *,
    mode: str,
) -> Path:
    """
    mode:
        train_tuning — 새 run 자동 증가
        train_full   — 새 run 자동 증가 (tuning과 동일, 별도 run 폴더)
        load         — 최신 run (없으면 FileNotFoundError)
    """
    model_root = _model_root(orig_cwd, cfg.model.name)
    run_date = get_run_date(cfg)

    if cfg.get("run_id"):
        run_dir = _find_run_by_id(model_root, cfg.run_id, run_date)
        if mode in ("train_tuning", "train_full") and not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if mode in ("train_tuning", "train_full"):
        run_dir = next_run_dir(model_root, run_date)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    # load mode
    runs = scan_run_dirs(model_root)
    if runs:
        return runs[-1][1]
    raise FileNotFoundError(
        f"run 디렉터리 없음: {model_root} (run001_YYMMDD 형식). "
        "먼저 튜닝 학습을 실행하거나 run_id= 를 지정하세요."
    )


def get_train_phase(cfg: DictConfig) -> str:
    return "tuning" if cfg.cv.enabled else "full"


def get_load_phase(cfg: DictConfig) -> str:
    """submit / eval_proxy / ensemble 기본 로드 단계."""
    cp = cfg.get("checkpoint")
    if cp is not None and cp.get("load_phase"):
        return str(cp.load_phase)
    return "full"


def legacy_checkpoint_path(orig_cwd: str, model_name: str) -> Path:
    return _model_root(orig_cwd, model_name) / "best.pt"


def get_checkpoint_path(
    cfg: DictConfig,
    orig_cwd: str,
    *,
    for_training: bool = False,
    load_phase: str | None = None,
) -> Path:
    """
    for_training=True  → cv에 따라 tuning/full 저장 경로 (항상 새 run 자동 증가)
    for_training=False → phase/best.pt 가 실제로 있는 최신 run에서 로드
    ckpt_path 설정 시 그대로 사용.
    legacy outputs/<model>/best.pt 폴백.
    """
    if cfg.get("ckpt_path"):
        p = Path(str(cfg.ckpt_path))
        return p if p.is_absolute() else Path(orig_cwd) / p

    if for_training:
        mode = "train_tuning" if cfg.cv.enabled else "train_full"
        run_dir = resolve_run_dir(cfg, orig_cwd, mode=mode)
        phase = get_train_phase(cfg)
        return run_dir / phase / "best.pt"

    # 로딩: phase/best.pt 가 실제로 존재하는 최신 run 우선 탐색
    phase = load_phase or get_load_phase(cfg)
    model_root = _model_root(orig_cwd, cfg.model.name)
    runs = scan_run_dirs(model_root)

    if cfg.get("run_id"):
        run_date = get_run_date(cfg)
        run_dir = _find_run_by_id(model_root, cfg.get("run_id"), run_date)
    else:
        run_dir = _latest_run_with_phase(runs, phase) or (runs[-1][1] if runs else None)

    if run_dir is None:
        legacy = legacy_checkpoint_path(orig_cwd, cfg.model.name)
        if legacy.is_file():
            return legacy
        raise FileNotFoundError(
            f"run 디렉터리 없음: {model_root}. "
            "먼저 학습을 실행하거나 ckpt_path= 를 지정하세요."
        )

    ckpt = run_dir / phase / "best.pt"
    if ckpt.is_file():
        return ckpt

    legacy = legacy_checkpoint_path(orig_cwd, cfg.model.name)
    if legacy.is_file():
        return legacy

    return ckpt
