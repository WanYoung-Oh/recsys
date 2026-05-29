"""
Calibration — groundedness 점수와 실제 품질(LLM-as-Judge) 간 ECE 측정 및 보정.

Guo et al. 2017 "On Calibration of Modern Neural Networks" 방식 적용:
  ECE = Σ_bins |acc(bin) - conf(bin)| × |bin| / N

여기서:
  confidence = SelfCheckGPT groundedness score (0~1)
  accuracy   = LLM-as-Judge 점수 정규화 (1~5 → 0~1)

Temperature Scaling: 단일 파라미터 T로 logit 보정
  calibrated = sigmoid(logit(conf) / T)
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


# ── ECE 계산 ─────────────────────────────────────────────────────────────────

def compute_ece(
    confidences: list[float],
    labels: list[float],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Args:
        confidences: 모델 확신도 리스트 [0, 1]
        labels:      실제 정확도 (0 또는 1, 또는 0~1 연속값)
        n_bins:      구간 수
    Returns:
        ECE (낮을수록 잘 보정됨)
    """
    if not confidences:
        return 0.0

    bins = [[] for _ in range(n_bins)]
    for conf, label in zip(confidences, labels):
        idx = min(int(conf * n_bins), n_bins - 1)
        bins[idx].append((conf, label))

    n = len(confidences)
    ece = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        avg_acc  = sum(l for _, l in b) / len(b)
        ece += abs(avg_acc - avg_conf) * len(b) / n

    return round(ece, 6)


# ── Temperature Scaling ───────────────────────────────────────────────────────

def _logit(p: float, eps: float = 1e-7) -> float:
    p = max(eps, min(1 - eps, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class TemperatureScaling:
    temperature: float = 1.0

    def fit(
        self,
        confidences: list[float],
        labels: list[float],
        lr: float = 0.01,
        steps: int = 200,
    ) -> "TemperatureScaling":
        """경사하강법으로 T 최적화 (NLL 최소화)."""
        T = self.temperature
        for _ in range(steps):
            grad = 0.0
            for conf, label in zip(confidences, labels):
                logit = _logit(conf)
                p = _sigmoid(logit / T)
                p = max(1e-7, min(1 - 1e-7, p))
                # ∂NLL/∂T = (p - y) * logit / T²
                grad += (p - label) * logit / (T ** 2)
            T -= lr * grad / len(confidences)
            T = max(0.1, T)   # T > 0 보장
        self.temperature = round(T, 4)
        return self

    def calibrate(self, confidence: float) -> float:
        """단일 점수 보정."""
        return round(_sigmoid(_logit(confidence) / self.temperature), 4)

    def calibrate_all(self, confidences: list[float]) -> list[float]:
        return [self.calibrate(c) for c in confidences]


# ── 저장 / 로드 ───────────────────────────────────────────────────────────────

def save_calibration(model: TemperatureScaling, path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(model), indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> TemperatureScaling:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TemperatureScaling(**data)


# ── 보정 리포트 ───────────────────────────────────────────────────────────────

def calibration_report(
    confidences: list[float],
    labels: list[float],
) -> dict:
    """ECE before/after temperature scaling 비교."""
    ece_before = compute_ece(confidences, labels)
    model = TemperatureScaling().fit(confidences, labels)
    calibrated = model.calibrate_all(confidences)
    ece_after = compute_ece(calibrated, labels)

    return {
        "n_samples":   len(confidences),
        "ece_before":  ece_before,
        "ece_after":   ece_after,
        "temperature": model.temperature,
        "improvement": round(ece_before - ece_after, 6),
    }
