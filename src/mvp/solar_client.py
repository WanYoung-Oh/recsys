"""
Solar Pro API wrapper.

의도 분류·추천 설명·일반 대화 3가지 역할만 담당.
상품 선정은 하지 않음 — 후보는 Python 규칙 엔진이 고정하여 주입.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

_API_KEY   = os.getenv("UPSTAGE_API_KEY", "")
_BASE_URL  = os.getenv("UPSTAGE_BASE_URL", "https://api.upstage.ai/v1")
_MODEL     = os.getenv("SOLAR_MODEL", "solar-pro3")
_T_EXPLAIN = float(os.getenv("SOLAR_TEMPERATURE_EXPLAIN", "0.3"))
_T_CHAT    = float(os.getenv("SOLAR_TEMPERATURE_CHAT", "0.7"))
_TOK_INTENT  = int(os.getenv("SOLAR_MAX_TOKENS_INTENT", "10"))
_TOK_EXPLAIN = int(os.getenv("SOLAR_MAX_TOKENS_EXPLAIN", "300"))

# ── 시스템 프롬프트 ───────────────────────────────────────────────────────────

_INTENT_SYSTEM = (
    "사용자 메시지의 의도를 다음 세 가지 중 하나로 분류하세요.\n"
    "- shopping   : 상품 추천, 구매 상담, 브랜드/카테고리 탐색 등 쇼핑 관련\n"
    "- user_alias : 유저 ID(user_00001 형식) 입력, 또는 본인 식별이 주목적인 메시지\n"
    "- general    : 그 외 일반 대화\n"
    '반드시 ["shopping", "user_alias", "general"] 중 하나만 출력하세요. 다른 텍스트는 포함하지 마세요.'
)

_EXPLAIN_SYSTEM = """\
당신은 이커머스 쇼핑 어드바이저입니다.

규칙:
- [추천 후보]에 있는 상품만 언급하세요. 목록 밖 상품은 언급 금지.
- item alias 문자열에서 속성을 추론하지 마세요. 제공된 메타데이터만 사용하세요.
- 각 reason은 유저 프로필 기반으로 한국어 2문장으로 작성하세요.
- 빈 섹션은 빈 배열 []로 표시하세요.

반드시 아래 JSON 구조를 그대로 사용하세요. 키 이름을 바꾸지 마세요:
{"sections":{"top10":[{"item_alias":"...","reason":"..."}],"content":[{"item_alias":"...","reason":"..."}],"recency":[{"item_alias":"...","reason":"..."}],"revisit":[{"item_alias":"...","reason":"..."}]}}\
"""

_CHAT_SYSTEM = (
    "당신은 친절한 쇼핑 도우미입니다. 일반적인 질문에 한국어로 자연스럽게 답변하세요. "
    "쇼핑 관련 질문이면 추천 기능을 안내해 주세요."
)


# ── 클라이언트 ────────────────────────────────────────────────────────────────

class SolarClient:
    def __init__(self) -> None:
        self._client = OpenAI(api_key=_API_KEY, base_url=_BASE_URL)

    def classify_intent(self, message: str) -> str:
        """3-class 의도 분류. 'shopping' | 'user_alias' | 'general' 반환."""
        resp = self._client.chat.completions.create(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user",   "content": message},
            ],
            max_tokens=_TOK_INTENT,
            temperature=0.0,
            timeout=15,
        )
        result = resp.choices[0].message.content.strip().lower()
        return result if result in ("shopping", "user_alias", "general") else "general"

    def explain_recommendations(self, context: str, max_retries: int = 1) -> dict | None:
        """추천 후보 + 유저 컨텍스트 → 섹션별 JSON 사유.

        파싱 성공 + 'sections' 키 보유 시 반환. max_retries회까지 재시도.
        """
        for attempt in range(max_retries):
            resp = self._client.chat.completions.create(
                model=_MODEL,
                messages=[
                    {"role": "system", "content": _EXPLAIN_SYSTEM},
                    {"role": "user",   "content": context},
                ],
                max_tokens=_TOK_EXPLAIN,
                temperature=_T_EXPLAIN,
                response_format={"type": "json_object"},
                timeout=120,
            )
            raw = resp.choices[0].message.content.strip()
            result = _parse_json(raw)
            # 'sections' 키가 있어야 올바른 구조
            if result is not None and isinstance(result.get("sections"), dict):
                return result
        return None

    def chat_complete(
        self,
        system: str,
        user: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        json_mode: bool = False,
        timeout: int = 20,
    ) -> str | None:
        """범용 단턴 chat completion. 실패 시 None 반환."""
        kwargs: dict = dict(
            model=_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception:
            return None

    def general_chat(self, message: str, history: list[dict] | None = None) -> str:
        """일반 대화 응답 (shopping 모드 아닌 경우)."""
        msgs: list[dict] = [{"role": "system", "content": _CHAT_SYSTEM}]
        if history:
            msgs.extend(history[-6:])   # 최근 6턴만 컨텍스트에 포함
        msgs.append({"role": "user", "content": message})

        resp = self._client.chat.completions.create(
            model=_MODEL,
            messages=msgs,
            max_tokens=512,
            temperature=_T_CHAT,
            timeout=30,
        )
        return resp.choices[0].message.content.strip()


def _parse_json(text: str) -> dict | None:
    """LLM 응답에서 JSON 추출. 코드블록으로 감싸여 있을 수 있음."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # ```json ... ``` 코드블록
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 텍스트에서 JSON 객체 추출
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# 모듈 전역 싱글턴 — nodes.py에서 import해 사용
solar = SolarClient()
