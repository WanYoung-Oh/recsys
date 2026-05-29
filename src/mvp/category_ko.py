"""카테고리 코드 → 한국어 번역 매핑."""

_L2_KO: dict[str, str] = {
    "belt":       "벨트",
    "costume":    "의상",
    "dress":      "원피스",
    "glove":      "장갑",
    "jacket":     "재킷",
    "jeans":      "청바지",
    "jumper":     "점퍼",
    "pajamas":    "파자마",
    "scarf":      "스카프",
    "shirt":      "셔츠",
    "shoes":      "신발",
    "shorts":     "반바지",
    "skirt":      "스커트",
    "sock":       "양말",
    "trousers":   "바지",
    "tshirt":     "티셔츠",
    "underwear":  "속옷",
}

_L3_KO: dict[str, str] = {
    "ballet_shoes": "발레 슈즈",
    "espadrilles":  "에스파드리유",
    "keds":         "스니커즈",
    "moccasins":    "모카신",
    "sandals":      "샌들",
    "slipons":      "슬립온",
    "step_ins":     "스텝인",
}


def ko_cat(l2: str, l3: str | None = None) -> str:
    """l2[, l3] → 한국어 카테고리 문자열.

    예) ko_cat("shoes", "keds") → "신발 > 스니커즈"
        ko_cat("tshirt")        → "티셔츠"
    """
    l2_ko = _L2_KO.get(l2, l2)
    if l3:
        l3_ko = _L3_KO.get(l3, l3)
        return f"{l2_ko} > {l3_ko}"
    return l2_ko
