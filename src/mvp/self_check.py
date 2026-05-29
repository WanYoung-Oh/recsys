"""
SelfCheckGPT — 추천 설명의 일관성 기반 환각 탐지.

arXiv:2303.08896 (Manakul et al., 2023) 방식을 쇼핑 도메인에 적용:
  1. 동일한 context로 N회 추가 설명 생성
  2. 원본 설명과 추가 샘플 간 item_alias / 키워드 일관성 측정
  3. 일관성 점수(0~1) = groundedness 신호로 활용

Consistency 측정:
  - item-level: Jaccard(원본 아이템 집합, 샘플 아이템 집합)
  - reason-level: 각 아이템 reason의 키워드 overlap (F1)
  - 최종: item × reason 가중 평균
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SelfCheckResult:
    groundedness: float          # 0(환각 의심) ~ 1(일관)
    n_samples: int
    item_consistency: float      # item-level Jaccard
    reason_consistency: float    # reason keyword F1 평균
    inconsistent_items: list[str] = field(default_factory=list)


def _extract_items(explanation: dict) -> dict[str, str]:
    """explanation → {item_alias: reason}."""
    result: dict[str, str] = {}
    for items in (explanation or {}).get("sections", {}).values():
        for item in items:
            if isinstance(item, dict) and item.get("item_alias"):
                result[item["item_alias"]] = item.get("reason", "")
    return result


def _keyword_f1(ref: str, hyp: str) -> float:
    """두 reason 문자열의 키워드 F1 (2자 이상 토큰 기준)."""
    def tokens(s: str) -> set[str]:
        return {t for t in re.findall(r"[가-힣a-z0-9]+", s.lower()) if len(t) >= 2}

    ref_t, hyp_t = tokens(ref), tokens(hyp)
    if not ref_t or not hyp_t:
        return 0.0
    inter = len(ref_t & hyp_t)
    prec  = inter / len(hyp_t)
    rec   = inter / len(ref_t)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _consistency(ref_explanation: dict, sample_explanation: dict) -> tuple[float, float]:
    """원본과 샘플 간 item·reason 일관성 점수 반환."""
    ref   = _extract_items(ref_explanation)
    samp  = _extract_items(sample_explanation)
    if not ref:
        return 0.0, 0.0

    # item-level Jaccard
    ref_set  = set(ref.keys())
    samp_set = set(samp.keys())
    inter    = len(ref_set & samp_set)
    union    = len(ref_set | samp_set)
    item_jac = inter / union if union > 0 else 0.0

    # reason-level F1 (공통 아이템에 대해서만)
    reason_scores: list[float] = []
    for alias in (ref_set & samp_set):
        reason_scores.append(_keyword_f1(ref[alias], samp[alias]))
    reason_f1 = sum(reason_scores) / len(reason_scores) if reason_scores else 0.0

    return item_jac, reason_f1


def self_check(
    context: str,
    explanation: dict,
    solar_client,           # src.mvp.solar_client.SolarClient
    n_samples: int = 2,
    candidates: dict | None = None,
) -> SelfCheckResult:
    """SelfCheckGPT — N 추가 샘플로 설명 일관성 평가.

    Args:
        context: Solar Pro에 보낸 원본 context_text
        explanation: 원본 Solar Pro JSON 응답
        solar_client: SolarClient 인스턴스
        n_samples: 추가 생성 샘플 수 (기본 2 — API 비용 고려)
        candidates: hard_gate용 candidates dict (None이면 건너뜀)
    Returns:
        SelfCheckResult
    """
    from src.mvp.trust_gate import hard_gate

    samples: list[dict] = []
    for _ in range(n_samples):
        try:
            sample = solar_client.explain_recommendations(context, max_retries=1)
        except Exception:
            continue
        if sample is None:
            continue
        if candidates:
            passed, _ = hard_gate(sample, candidates)
            if not passed:
                continue
        samples.append(sample)

    if not samples:
        return SelfCheckResult(
            groundedness=0.0, n_samples=0,
            item_consistency=0.0, reason_consistency=0.0,
        )

    item_scores: list[float] = []
    reason_scores: list[float] = []
    for samp in samples:
        ijac, rf1 = _consistency(explanation, samp)
        item_scores.append(ijac)
        reason_scores.append(rf1)

    avg_item   = sum(item_scores)   / len(item_scores)
    avg_reason = sum(reason_scores) / len(reason_scores)
    # groundedness: item 일관성이 더 중요 (범위 이탈 = 환각)
    groundedness = 0.7 * avg_item + 0.3 * avg_reason

    # 불일치 아이템 목록
    ref_items = set(_extract_items(explanation).keys())
    all_samp_items: set[str] = set()
    for s in samples:
        all_samp_items |= set(_extract_items(s).keys())
    inconsistent = list((ref_items | all_samp_items) - (ref_items & all_samp_items))

    return SelfCheckResult(
        groundedness=round(groundedness, 4),
        n_samples=len(samples),
        item_consistency=round(avg_item, 4),
        reason_consistency=round(avg_reason, 4),
        inconsistent_items=inconsistent,
    )
