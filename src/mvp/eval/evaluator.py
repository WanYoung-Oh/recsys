"""
P4 평가 실행기 — SelfCheckGPT + Calibration + Regression Test.

Usage:
    python src/mvp/eval/evaluator.py --golden rag_data/golden_set.json
    python src/mvp/eval/evaluator.py --golden rag_data/golden_set.json --self-check
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


@dataclass
class EvalReport:
    n_entries:          int
    avg_judge_score:    float
    judge_score_dist:   dict      # {1:n, 2:n, ...}
    avg_groundedness:   float     # SelfCheckGPT (0이면 미실행)
    ece_before:         float     # 보정 전 ECE
    ece_after:          float     # 보정 후 ECE
    temperature:        float
    pass_threshold:     bool      # avg_judge_score >= threshold(3.0)


def run_evaluation(
    golden_path: str | Path,
    run_self_check: bool = False,
    n_selfcheck_samples: int = 2,
    judge_threshold: float = 3.0,
) -> EvalReport:
    from src.mvp.eval.golden_set import load_golden_set
    from src.mvp.calibration import calibration_report

    entries = load_golden_set(golden_path)
    if not entries:
        raise ValueError(f"Golden set이 비어 있습니다: {golden_path}")

    # ── SelfCheckGPT (선택) ───────────────────────────────────────────────────
    if run_self_check:
        from src.mvp.self_check import self_check
        from src.mvp.solar_client import solar
        from src.mvp.nodes import context_builder

        print(f"\nSelfCheckGPT 실행 중 ({n_selfcheck_samples}샘플 × {len(entries)}건)...")
        for entry in entries:
            if not entry.explanation or not entry.candidates:
                continue
            state = {
                "profile":   {"user_alias": entry.user_alias,
                               "profile_text": entry.profile_text,
                               "display_name": entry.user_alias,
                               "top_categories_l2": [], "top_brands": [],
                               "price_range": {}, "activity_level": {},
                               "recent_items": [], "seen_items": {},
                               "cart_items": [], "purchased_items": []},
                "candidates": entry.candidates,
                "requested_category": None,
            }
            state = context_builder(state)
            ctx = state.get("context_text", "")
            t0 = time.time()
            try:
                sc_result = self_check(ctx, entry.explanation, solar,
                                       n_samples=n_selfcheck_samples,
                                       candidates=entry.candidates)
                entry.groundedness = sc_result.groundedness
                print(f"  {entry.user_alias}: groundedness={sc_result.groundedness:.3f} "
                      f"(item={sc_result.item_consistency:.3f}, "
                      f"reason={sc_result.reason_consistency:.3f})  {time.time()-t0:.1f}s")
            except Exception as e:
                print(f"  {entry.user_alias}: self_check 오류 — {e} (건너뜀)")

        # 업데이트된 golden set 저장
        from src.mvp.eval.golden_set import _save
        _save(entries, golden_path)
        print("SelfCheckGPT 결과 golden_set.json에 저장 완료")

    # ── 집계 ─────────────────────────────────────────────────────────────────
    scored = [e for e in entries if e.judge_score is not None and e.judge_score > 0]
    if not scored:
        raise ValueError("채점된 항목이 없습니다. --build로 golden set을 먼저 생성하세요.")

    judge_scores = [e.judge_score for e in scored]
    avg_judge    = round(sum(judge_scores) / len(judge_scores), 2)
    dist         = {}
    for s in range(1, 6):
        dist[s] = judge_scores.count(s)

    avg_groundedness = 0.0
    grounded_entries = [e for e in scored if e.groundedness > 0]
    if grounded_entries:
        avg_groundedness = round(
            sum(e.groundedness for e in grounded_entries) / len(grounded_entries), 4
        )

    # ── 보정 (groundedness score가 있는 경우) ─────────────────────────────────
    ece_before = ece_after = temperature = 0.0
    if grounded_entries:
        confidences = [e.groundedness for e in grounded_entries]
        labels      = [(e.judge_score - 1) / 4 for e in grounded_entries]  # 1~5 → 0~1
        cal_report  = calibration_report(confidences, labels)
        ece_before  = cal_report["ece_before"]
        ece_after   = cal_report["ece_after"]
        temperature = cal_report["temperature"]

        # 보정 모델 저장
        from src.mvp.calibration import TemperatureScaling, save_calibration
        model = TemperatureScaling(temperature=temperature)
        cal_path = ROOT / "rag_data" / "calibration.json"
        save_calibration(model, cal_path)
        print(f"보정 모델 저장: {cal_path}  (T={temperature})")

    report = EvalReport(
        n_entries=len(scored),
        avg_judge_score=avg_judge,
        judge_score_dist=dist,
        avg_groundedness=avg_groundedness,
        ece_before=ece_before,
        ece_after=ece_after,
        temperature=temperature,
        pass_threshold=(avg_judge >= judge_threshold),
    )

    # eval_report.json 저장 (앱 사이드바 표시용)
    import dataclasses
    report_path = ROOT / "rag_data" / "eval_report.json"
    report_path.write_text(
        json.dumps(dataclasses.asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"평가 리포트 저장: {report_path}")

    return report


def print_report(report: EvalReport) -> None:
    print("\n" + "═" * 55)
    print("  P4 평가 리포트")
    print("═" * 55)
    print(f"  샘플 수         : {report.n_entries}건")
    print(f"  LLM-as-Judge   : {report.avg_judge_score:.2f}/5.0  "
          f"{'✅ PASS' if report.pass_threshold else '❌ FAIL'}")
    print(f"  점수 분포       : {report.judge_score_dist}")
    if report.avg_groundedness > 0:
        print(f"  SelfCheckGPT   : {report.avg_groundedness:.4f}  (groundedness)")
        print(f"  ECE before/after: {report.ece_before:.4f} → {report.ece_after:.4f}  "
              f"(T={report.temperature})")
    print("═" * 55)
    status = "PASS ✅" if report.pass_threshold else "FAIL ❌"
    print(f"  회귀 테스트     : {status}  (기준 avg_judge ≥ 3.0)")
    print("═" * 55 + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="P4 평가 실행기")
    parser.add_argument("--golden",     default="rag_data/golden_set.json")
    parser.add_argument("--self-check", action="store_true", help="SelfCheckGPT 실행")
    parser.add_argument("--n-samples",  type=int, default=2, help="SelfCheck 샘플 수")
    args = parser.parse_args()

    golden_path = ROOT / args.golden
    if not golden_path.exists():
        print(f"Golden set 없음: {golden_path}")
        print("먼저 실행: python src/mvp/eval/golden_set.py --build")
        sys.exit(1)

    report = run_evaluation(
        golden_path,
        run_self_check=args.self_check,
        n_selfcheck_samples=args.n_samples,
    )
    print_report(report)

    # 회귀 실패 시 exit code 1
    sys.exit(0 if report.pass_threshold else 1)


if __name__ == "__main__":
    main()
