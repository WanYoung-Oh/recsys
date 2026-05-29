"""
User/Item alias 생성 및 역변환.

User alias : user_00001 ~ user_NNNNN  (sample_submission.csv 유저 등장 순서)
Item alias : {L2}.{L3}.{brand_slug}.{price_bucket}_{seq:04d}  (Semantic ID)
"""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd

# ── 한국어 가명 풀 ────────────────────────────────────────────────────────────
_SURNAMES = [
    "김", "이", "박", "최", "정", "강", "조", "윤", "장", "임",
    "한", "오", "서", "신", "권", "황", "안", "송", "류", "전",
    "홍", "고", "문", "양", "손", "배", "백", "허", "유", "남",
]
_GIVEN_NAMES = [
    "민준", "서준", "도윤", "주원", "시우", "예준", "하준", "지호", "준서", "준혁",
    "현우", "지훈", "건우", "우진", "민재", "현준", "선우", "진우", "동현", "승현",
    "재원", "민성", "지성", "준영", "현서", "재민", "도현", "민혁", "성민", "태민",
    "준기", "민우", "지원", "현민", "재현", "재준", "민호", "준호", "재호", "성호",
    "서연", "민서", "지우", "서현", "지유", "채원", "수아", "지아", "수빈", "하은",
    "나은", "가은", "하윤", "윤서", "지민", "채은", "서영", "예은", "아린", "민지",
    "지은", "수은", "서은", "예진", "혜진", "소현", "미현", "다현", "나현", "유현",
    "세아", "예나", "지나", "수나", "소나", "미나", "다나", "유나", "세나", "보나",
    "보미", "지미", "수미", "소미", "하미", "유미", "세미", "나미", "다미", "예미",
    "준영", "민영", "재영", "성영", "진영", "우영", "승영", "태영", "도영", "현영",
]


def _gen_display_names(n: int, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    sur = rng.integers(0, len(_SURNAMES), size=n)
    giv = rng.integers(0, len(_GIVEN_NAMES), size=n)
    return [_SURNAMES[s] + _GIVEN_NAMES[g] for s, g in zip(sur, giv)]


# ── User Alias ────────────────────────────────────────────────────────────────

def build_user_aliases(ordered_user_ids: list[str], seed: int = 42) -> dict[str, dict]:
    """user_id → {user_alias, display_name} 매핑 생성.

    Args:
        ordered_user_ids: sample_submission.csv 유저 등장 순서 기준 list
    Returns:
        {user_id: {"user_alias": "user_00001", "display_name": "김민지"}}
    """
    display_names = _gen_display_names(len(ordered_user_ids), seed)
    return {
        uid: {
            "user_alias": f"user_{i + 1:05d}",
            "display_name": display_names[i],
        }
        for i, uid in enumerate(ordered_user_ids)
    }


def build_alias_to_userid(user_alias_map: dict[str, dict]) -> dict[str, str]:
    """user_alias → user_id 역방향 맵 (O(1) lookup)."""
    return {info["user_alias"]: uid for uid, info in user_alias_map.items()}


# ── Item Alias (Semantic ID) ──────────────────────────────────────────────────

def parse_category(category_code: str) -> tuple[str, str | None]:
    """'apparel.shoes.keds' → ('shoes', 'keds'), 'apparel.shoes' → ('shoes', None)."""
    parts = str(category_code).split(".")
    l2 = parts[1] if len(parts) >= 2 else parts[0]
    l3 = parts[2] if len(parts) >= 3 else None
    return l2, l3


def make_price_bucket_fn(p33: float, p66: float):
    """price_median 기준 low/mid/high 버킷 함수 반환."""
    def price_bucket(price_median: float) -> str:
        if price_median < p33:
            return "low"
        if price_median < p66:
            return "mid"
        return "high"
    return price_bucket


def brand_to_slug(brand: str) -> str:
    """브랜드명 → ASCII 소문자 slug (영숫자·언더스코어만)."""
    slug = re.sub(r"[^a-z0-9]", "_", str(brand).lower()).strip("_")
    return slug or "unknown"


def build_item_aliases(items_df: pd.DataFrame, price_bucket_fn) -> dict[str, str]:
    """item_id → Semantic ID 매핑 생성.

    Args:
        items_df: index=item_id, columns=[category_code, brand, price_median]
        price_bucket_fn: make_price_bucket_fn() 반환값
    Returns:
        {item_id: "shoes.keds.kapika.mid_0001"}
    """
    # 그룹 키: 원본 brand 사용 (slug 충돌 방지)
    groups: dict[tuple, list[str]] = defaultdict(list)
    for item_id, row in items_df.iterrows():
        l2, l3 = parse_category(row["category_code"])
        bucket = price_bucket_fn(float(row["price_median"]))
        group_key = (l2, l3, row["brand"], bucket)
        groups[group_key].append(str(item_id))

    alias_map: dict[str, str] = {}
    seen_aliases: set[str] = set()

    for (l2, l3, brand, bucket), item_ids in groups.items():
        brand_slug = brand_to_slug(str(brand))
        prefix = (
            f"{l2}.{l3}.{brand_slug}.{bucket}"
            if l3
            else f"{l2}.{brand_slug}.{bucket}"
        )
        # hash 정렬로 재현 가능한 순서 보장
        sorted_ids = sorted(item_ids, key=lambda x: hashlib.md5(x.encode()).hexdigest())

        current_seq = 1
        for item_id in sorted_ids:
            alias = f"{prefix}_{current_seq:04d}"
            while alias in seen_aliases:
                current_seq += 1
                alias = f"{prefix}_{current_seq:04d}"
            seen_aliases.add(alias)
            alias_map[item_id] = alias
            current_seq += 1

    return alias_map


def build_alias_to_itemid(item_alias_map: dict[str, str]) -> dict[str, str]:
    """item_alias → item_id 역방향 맵."""
    return {v: k for k, v in item_alias_map.items()}


def resolve_user(user_alias: str, alias_to_userid: dict[str, str]) -> str | None:
    """user_alias → original user_id. None이면 미등록."""
    return alias_to_userid.get(user_alias)


def resolve_item(item_alias: str, alias_to_itemid: dict[str, str]) -> str | None:
    """item_alias → original item_id. None이면 미등록."""
    return alias_to_itemid.get(item_alias)
