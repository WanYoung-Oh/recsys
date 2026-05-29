"""
hard_gate: Solar Pro 응답 구조·범위 검증.

검사 항목:
  1. explanation이 None이거나 'sections' 키 없음
  2. 각 섹션이 list 타입인지 확인
  3. item_alias가 candidates 범위 밖이면 환각으로 차단 (결정적 차단)

완화된 항목:
  - reason 비어 있음 → 실패 아님. Solar Pro가 일부 상품 설명을 생략해도
    나머지 유효한 설명은 그대로 사용한다. 빈 reason은 UI에서 단순히 숨김 처리.

P4(Evidence Pack 완성) 이후 추가 예정:
  - evidence_ref 화이트리스트 검사
  - truthy 값 검사
  - bool 모순 검사 (True 근거를 부정형으로 서술)
"""
from __future__ import annotations


def hard_gate(
    explanation: dict | None,
    candidates: dict[str, list[str]],
) -> tuple[bool, str]:
    """Solar Pro JSON 응답 검증.

    Returns:
        (passed, reason): passed=True면 신뢰 가능, False면 거부 사유 문자열.
    """
    if not explanation:
        return False, "응답 JSON 파싱 실패"

    if "sections" not in explanation:
        return False, "'sections' 키 없음"

    sections = explanation["sections"]
    if not isinstance(sections, dict):
        return False, "'sections'가 dict 타입이 아님"

    # 허용된 item_alias 집합 (후보 밖 = 환각)
    allowed: set[str] = set()
    for items in candidates.values():
        allowed.update(items)

    for sec, items in sections.items():
        if not isinstance(items, list):
            return False, f"섹션 '{sec}'이 list 타입이 아님"

        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                return False, f"섹션 '{sec}'[{idx}]이 dict 타입이 아님"

            alias = item.get("item_alias", "")
            # 범위 외 상품 = 환각 → 결정적 차단
            if alias and alias not in allowed:
                return False, f"범위 외 상품 감지 (환각): {alias}"
            # reason 비어 있음은 차단하지 않음 — UI에서 숨김 처리

    return True, "OK"
