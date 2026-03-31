from __future__ import annotations

import json
import re
from typing import Any

import requests

from app.config import settings

class LLMClient:
    def __init__(self) -> None:
        self.mock = settings.use_mock_llm or not settings.gemini_api_key
        self._api_key = settings.gemini_api_key
        self._model = settings.gemini_model

    def generate_text(self, prompt: str, system: str = "You are a helpful AI assistant.") -> str:
        if self.mock:
            return self._mock_text(prompt)

        content = self._call_gemini(f"{system}\n\n{prompt}")
        return content or ""

    def generate_json(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if self.mock:
            return self._mock_json(prompt, fallback)

        raw = self._call_gemini(
            "Return strict JSON only. Do not include markdown fences.\n\n" + prompt
        )
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return fallback

    def _call_gemini(self, text: str) -> str:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {"temperature": 0.2},
        }
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return ""
        return parts[0].get("text", "")

    def _mock_json(self, prompt: str, fallback: dict[str, Any]) -> dict[str, Any]:
        if "decompose" in prompt.lower() or "拆" in prompt:
            return {
                "steps": [
                    "Analyze user request and constraints",
                    "Get weather information for target place",
                    "Search attractions and crowd insights",
                    "Generate a practical day schedule",
                    "Summarize with actionable tips",
                ]
            }
        return fallback

    def _mock_text(self, prompt: str) -> str:
        lower = prompt.lower()
        if "summary" in lower or "总结" in prompt:
            return "Here is a concise and personalized result with plan, constraints, and next actions."

        # Naive preference extraction for long-term memory.
        pref_patterns = [r"budget\s*[:=]?\s*(\d+)", r"预算\s*[:：]?\s*(\d+)"]
        prefs: list[str] = []
        for pattern in pref_patterns:
            m = re.search(pattern, lower)
            if m:
                prefs.append(f"budget={m.group(1)}")

        if prefs:
            return "Detected preferences: " + ", ".join(prefs)

        return "Generated response in mock mode based on available tool outputs and retrieved context."


llm_client = LLMClient()
