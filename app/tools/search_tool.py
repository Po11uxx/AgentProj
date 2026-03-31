from __future__ import annotations

import re
from typing import Any

import requests


class SearchTool:
    name = "search"
    description = "Search web snippets related to user query"

    def run(self, query: str) -> dict[str, Any]:
        try:
            snippets = self._instant_answer_search(query)
            if snippets:
                return {"query": query, "snippets": snippets[:5], "source": "duckduckgo_instant", "live": True}
        except Exception:
            pass

        try:
            snippets = self._html_search(query)
            if snippets:
                return {"query": query, "snippets": snippets[:5], "source": "duckduckgo_html", "live": True}
        except Exception:
            pass

        return {
            "query": query,
            "snippets": [
                f"Top attractions and crowd advice for: {query}",
                "Prioritize nearby spots to reduce transport overhead.",
                "Start outdoor activities in the morning and museums in the afternoon.",
            ],
            "source": "mock",
            "live": False,
        }

    def _instant_answer_search(self, query: str) -> list[str]:
        resp = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        abstract = data.get("AbstractText", "")
        related = [x.get("Text", "") for x in data.get("RelatedTopics", []) if isinstance(x, dict)]
        return [s.strip() for s in [abstract, *related] if s and s.strip()]

    def _html_search(self, query: str) -> list[str]:
        resp = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "agent-proj/1.0"},
            timeout=8,
        )
        resp.raise_for_status()
        html = resp.text
        raw = re.findall(r'result__snippet[^>]*>(.*?)</a>', html, flags=re.IGNORECASE | re.DOTALL)
        snippets: list[str] = []
        for item in raw:
            text = re.sub(r"<[^>]+>", " ", item)
            text = re.sub(r"\s+", " ", text).strip()
            if text and text not in snippets:
                snippets.append(text)
        return snippets
