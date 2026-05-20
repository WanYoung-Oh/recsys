from collections import defaultdict
from typing import Dict, List


def rank_ensemble(
    model_predictions: Dict[str, Dict[str, List]],
    weights: Dict[str, float],
    top_k: int = 10,
) -> Dict[str, List]:
    """랭크 기반 앙상블 (Reciprocal Rank Fusion 스타일).

    model_predictions: {model_name: {user_id: [item_id, ...]}}
    weights          : {model_name: float}
    """
    user_scores: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for model_name, preds in model_predictions.items():
        w = weights.get(model_name, 1.0)
        for uid, item_list in preds.items():
            for rank, item in enumerate(item_list):
                user_scores[uid][item] += w / (rank + 1)

    return {
        uid: sorted(scores, key=scores.get, reverse=True)[:top_k]
        for uid, scores in user_scores.items()
    }
