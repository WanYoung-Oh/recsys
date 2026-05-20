"""TIFU-KNN: Temporal Interest Fusion with KNN (non-neural).

논문: "Temporal-Item-Frequency-based User-KNN" 의 개인화 시간감쇠 컴포넌트.
전체 유저 KNN 없이도 유저 개인 히스토리 내 시간 감쇠 + 그룹 반복 점수화만으로
SASRec 대비 보완적 신호를 제공.

group_count: 히스토리를 g 등분해 최근 그룹에 r_within^(g-1)·r_across^0 배 가중치.
decay_within: 그룹 내 지수 감쇠 (기본 0.9)
decay_across: 그룹 간 감쇠 (기본 0.7)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List


class TIFUKNN:
    def __init__(
        self,
        group_count: int = 7,
        decay_within: float = 0.9,
        decay_across: float = 0.7,
    ):
        self.g = group_count
        self.r_w = decay_within
        self.r_a = decay_across
        self._user_scores: Dict[str, Dict[int, float]] = {}

    def fit(self, seqs: Dict[str, Dict], idx2item: Dict[int, str]) -> "TIFUKNN":
        """seqs: {uid: {'items': [...], 'times': [...], 'types': [...]}}"""
        self._idx2item = idx2item
        for uid, data in seqs.items():
            items = data["items"]   # item idx (1-based), 시간순
            n = len(items)
            if n == 0:
                self._user_scores[uid] = {}
                continue

            # n 개 아이템을 g 그룹으로 분할 (최근 그룹 = 마지막)
            group_size = max(1, math.ceil(n / self.g))
            groups: List[List[int]] = []
            for start in range(0, n, group_size):
                groups.append(items[start : start + group_size])

            scores: Dict[int, float] = defaultdict(float)
            n_groups = len(groups)
            for gi, grp in enumerate(groups):
                # gi=0 이 가장 오래된 그룹, gi=n_groups-1 이 가장 최근
                across_weight = self.r_a ** (n_groups - 1 - gi)
                grp_size = len(grp)
                for pos_in_grp, item in enumerate(reversed(grp)):
                    # 그룹 내에서도 최근일수록 높은 가중치
                    within_weight = self.r_w ** pos_in_grp
                    scores[item] += across_weight * within_weight

            self._user_scores[uid] = dict(scores)

        return self

    def predict(self, uid: str, top_k: int = 10) -> List[str]:
        scores = self._user_scores.get(uid, {})
        ranked = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [self._idx2item[idx] for idx in ranked if idx in self._idx2item]

    def predict_all(
        self, user_ids: List[str], top_k: int = 10
    ) -> Dict[str, List[str]]:
        return {uid: self.predict(uid, top_k) for uid in user_ids}
