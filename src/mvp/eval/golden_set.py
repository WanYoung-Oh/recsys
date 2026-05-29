"""
Golden Set 빌드 + LLM-as-Judge 평가.

Golden Set:
  - 다양한 활동 수준의 유저 N명 샘플
  - 각 유저의 추천 결과 + LLM-as-Judge 점수(1~5) 저장
  - SelfCheckGPT groundedness 점수 포함
  - 회귀 테스트 기준선으로 사용

Usage:
    python src/mvp/eval/golden_set.py --build --n 20 --out rag_data/golden_set.json
    python src/mvp/eval/golden_set.py --judge --path rag_data/golden_set.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

_JUDGE_SYSTEM = """\
당신은 이커머스 추천 시스템 평가 전문가입니다.
아래 [유저 프로필]과 [추천 설명]을 보고 추천의 품질을 1~5점으로 평가하세요.

채점 기준:
5 — 프로필 취향과 완벽히 일치, 구체적·신뢰할 수 있는 사유
4 — 대체로 적절하고 관련 있는 사유
3 — 관련성 있으나 일반적·모호한 사유
2 — 프로필과 부분적으로만 맞는 사유
1 — 프로필과 무관하거나 근거 없는 사유

반드시 다음 JSON 형식으로만 응답하세요:
{"score": <1~5 정수>, "reason": "<한 문장 평가 근거>"}\
"""


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class GoldenEntry:
    user_alias:           str
    profile_text:         str
    candidates:           dict
    explanation:          dict | None
    judge_score:          int | None        # 1~5, None이면 미평가
    judge_reason:         str = ""
    groundedness:         float = 0.0      # SelfCheckGPT 점수
    built_at:             str = ""


def _save(entries: list[GoldenEntry], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(e) for e in entries], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_golden_set(path: str | Path) -> list[GoldenEntry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [GoldenEntry(**d) for d in data]


# ── LLM-as-Judge ─────────────────────────────────────────────────────────────

def llm_as_judge(
    profile_text: str,
    explanation: dict | None,
    solar_client,
) -> tuple[int, str]:
    """Solar Pro가 추천 설명 품질을 1~5로 채점.

    Returns:
        (score, reason)  score=0 이면 채점 실패
    """
    if not explanation:
        return 0, "explanation 없음"

    # 설명 요약 (첫 3개 아이템)
    items_text: list[str] = []
    for sec, items in explanation.get("sections", {}).items():
        for item in items[:2]:
            if isinstance(item, dict) and item.get("reason"):
                items_text.append(f"[{sec}] {item['item_alias']}: {item['reason']}")

    if not items_text:
        return 0, "설명 내용 없음"

    user_msg = (
        f"[유저 프로필]\n{profile_text}\n\n"
        f"[추천 설명]\n" + "\n".join(items_text[:4])
    )

    raw = solar_client.chat_complete(
        system=_JUDGE_SYSTEM,
        user=user_msg,
        max_tokens=80,
        temperature=0.0,
        json_mode=True,
        timeout=20,
    )
    if raw is None:
        return 0, "API 호출 실패"
    try:
        data   = json.loads(raw.strip())
        score  = int(data.get("score", 0))
        reason = str(data.get("reason", ""))
        return max(1, min(5, score)), reason
    except Exception as e:
        return 0, str(e)


# ── Golden Set 빌드 ───────────────────────────────────────────────────────────

def _sample_diverse_users(db_path: Path, n: int) -> list[str]:
    """활동 수준이 다양한 유저 N명 샘플링."""
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT user_alias, profile_json FROM user_profiles ORDER BY user_alias"
    ).fetchall()
    con.close()

    # 활동 수준별 그룹핑
    groups: dict[str, list[str]] = {"high": [], "mid": [], "low": []}
    for ua, pj in rows:
        p = json.loads(pj)
        total = p.get("activity_level", {}).get("total_events", 0)
        if total >= 50:
            groups["high"].append(ua)
        elif total >= 10:
            groups["mid"].append(ua)
        else:
            groups["low"].append(ua)

    # 균등 샘플 (각 그룹에서 n//3씩)
    per_group = max(1, n // 3)
    selected: list[str] = []
    for g in ("high", "mid", "low"):
        candidates = groups[g]
        step = max(1, len(candidates) // per_group)
        selected.extend(candidates[::step][:per_group])

    return selected[:n]


def build_golden_set(
    n: int,
    db_path: Path,
    out_path: Path,
    solar_client,
    run_judge: bool = True,
) -> list[GoldenEntry]:
    """Golden set 빌드 + (선택) LLM-as-Judge 채점."""
    from src.mvp.recommenders import build_all_candidates
    from src.mvp.nodes import _store, context_builder, solar_explainer

    rag = ROOT / "rag_data"
    s   = _store()

    print(f"Golden Set 빌드: 다양한 유저 {n}명 샘플링 중...")
    user_aliases = _sample_diverse_users(db_path, n)
    print(f"  선택된 유저: {user_aliases}")

    entries: list[GoldenEntry] = []
    for ua in user_aliases:
        print(f"\n  [{ua}] 추천 생성 중...")
        con = sqlite3.connect(db_path)
        row = con.execute(
            "SELECT profile_json, profile_text FROM user_profiles WHERE user_alias=?",
            (ua,),
        ).fetchone()
        con.close()
        if not row:
            continue

        profile = json.loads(row[0])
        profile.update({
            "user_alias":    ua,
            "profile_text":  row[1],
            "display_name":  ua,
        })

        # 추천 후보 생성
        candidates = build_all_candidates(
            profile=profile,
            user_alias=ua,
            user_recs=s["user_recs"],
            item_catalog=s["item_catalog"],
            recency_pool=s["recency_pool"],
        )

        # Solar Pro 설명 생성
        explanation = None
        if run_judge:
            state: dict = {
                "profile": profile,
                "candidates": candidates,
                "requested_category": None,
            }
            state = context_builder(state)
            state = solar_explainer(state)
            explanation = state.get("explanation")

        # LLM-as-Judge 채점
        judge_score, judge_reason = (None, "")
        if run_judge and explanation:
            t0 = time.time()
            judge_score, judge_reason = llm_as_judge(row[1], explanation, solar_client)
            print(f"    Judge 점수: {judge_score}/5  ({time.time()-t0:.1f}s)  {judge_reason[:60]}")

        entries.append(GoldenEntry(
            user_alias=ua,
            profile_text=row[1],
            candidates=candidates,
            explanation=explanation,
            judge_score=judge_score,
            judge_reason=judge_reason,
            groundedness=0.0,   # SelfCheckGPT는 evaluator에서 별도 실행
            built_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        ))

    _save(entries, out_path)
    print(f"\nGolden Set 저장: {out_path}  ({len(entries)}건)")
    return entries


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Golden Set 빌드 / LLM-as-Judge")
    parser.add_argument("--build", action="store_true", help="Golden Set 빌드")
    parser.add_argument("--n",    type=int, default=20,  help="샘플 유저 수")
    parser.add_argument("--out",  default="rag_data/golden_set.json", help="출력 경로")
    parser.add_argument("--no-judge", action="store_true", help="LLM 채점 생략")
    args = parser.parse_args()

    from src.mvp.solar_client import solar

    db_path  = ROOT / "rag_data" / "user_profiles.db"
    out_path = ROOT / args.out

    if args.build:
        build_golden_set(
            n=args.n,
            db_path=db_path,
            out_path=out_path,
            solar_client=solar,
            run_judge=not args.no_judge,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
