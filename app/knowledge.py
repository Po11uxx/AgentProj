from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import KNOWLEDGE_DIR


CITY_ALIASES = {
    "hangzhou": ["hangzhou", "杭州"],
    "los angeles": ["los angeles", "la", "洛杉矶"],
    "new york": ["new york", "nyc", "纽约"],
    "san francisco": ["san francisco", "sf", "旧金山"],
    "beijing": ["beijing", "北京"],
    "shanghai": ["shanghai", "上海"],
}


FILE_HINTS = {
    "hangzhou": ["hz_travel_tips", "hangzhou_travel", "hangzhou"],
    "los angeles": ["la_travel_tips", "los_angeles_travel", "los_angeles"],
    "new york": ["new_york_travel", "newyork_travel", "new_york"],
    "san francisco": ["san_francisco_travel", "sanfrancisco_travel", "san_francisco"],
    "beijing": ["beijing_travel", "beijing"],
    "shanghai": ["shanghai_travel", "shanghai"],
}


def slugify_city(city: str) -> str:
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", city.strip().lower()).strip("_")
    return slug or "city"


def city_variants(city: str) -> list[str]:
    lower = city.strip().lower()
    variants = {city.strip(), lower, city.strip().title()}
    for canonical, aliases in CITY_ALIASES.items():
        if lower == canonical or lower in aliases:
            variants.update(aliases)
            variants.add(canonical.title())
    return [v for v in variants if v]


def canonical_city_key(city: str) -> str:
    lower = city.strip().lower()
    for canonical, aliases in CITY_ALIASES.items():
        if lower == canonical or lower in aliases:
            return canonical
    return lower


def _contains_city_token(text: str, variant: str) -> bool:
    text_lower = text.lower()
    token = variant.lower()
    if re.search(r"[\u4e00-\u9fff]", token):
        return token in text_lower
    if len(token) <= 2:
        return bool(re.search(rf"(?<![a-z]){re.escape(token)}(?![a-z])", text_lower))
    return bool(re.search(rf"\b{re.escape(token)}\b", text_lower))


class KnowledgeBaseManager:
    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR) -> None:
        self.knowledge_dir = knowledge_dir
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

    def find_city_knowledge_file(self, city: str) -> Path | None:
        canonical = canonical_city_key(city)
        variants = city_variants(city)
        city_slug = slugify_city(city)
        file_hints = FILE_HINTS.get(canonical, []) + [city_slug]
        for file_path in sorted(self.knowledge_dir.glob("*.md")):
            file_name = file_path.name.lower()
            if any(hint in file_name for hint in file_hints):
                return file_path
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            header = "\n".join(content.splitlines()[:6])
            if any(_contains_city_token(file_name, v) or _contains_city_token(header, v) for v in variants):
                return file_path
        return None

    def ensure_city_knowledge(self, city: str, search_tool: object, map_tool: object) -> dict[str, Any]:
        existing = self.find_city_knowledge_file(city)
        if existing is not None:
            return {
                "city": city,
                "created": False,
                "path": str(existing),
                "status": "cached",
            }

        content, diagnostics = self._generate_city_knowledge(city=city, search_tool=search_tool, map_tool=map_tool)
        if not diagnostics.get("has_live_data"):
            return {
                "city": city,
                "created": False,
                "path": "",
                "status": "skipped_no_live_data",
                "diagnostics": diagnostics,
            }
        file_path = self.knowledge_dir / f"{slugify_city(city)}_travel_guide.md"
        file_path.write_text(content, encoding="utf-8")
        return {
            "city": city,
            "created": True,
            "path": str(file_path),
            "status": "generated",
            "diagnostics": diagnostics,
        }

    def _generate_city_knowledge(
        self,
        city: str,
        search_tool: object,
        map_tool: object,
    ) -> tuple[str, dict[str, Any]]:
        search_queries = [
            ("Overview", f"{city} top attractions one day itinerary tips"),
            ("Local logistics", f"{city} neighborhoods transport crowd tips"),
            ("Food", f"{city} local food signature restaurants travel tips"),
        ]
        search_sections: list[tuple[str, dict[str, Any]]] = []
        for title, query in search_queries:
            result = search_tool.run(query=query)
            search_sections.append((title, result))

        poi_data = map_tool.run(
            city=city,
            intents=["landmark", "museum", "restaurant", "viewpoint", "park"],
            limit_per_intent=2,
        )
        places = poi_data.get("places", [])
        center = poi_data.get("center", {})

        lines: list[str] = [
            f"# {city} Runtime Travel Guide",
            "",
            f"- Generated automatically at {datetime.now(timezone.utc).isoformat()}",
            f"- City center reference: {center.get('display_name', city)}",
            "- Use this file as reusable RAG context for future itinerary generation.",
            "",
            "## Real POI Candidates",
        ]
        if places:
            for place in places[:10]:
                lines.append(
                    f"- {place.get('name', 'POI')} | {place.get('category', 'place')} | "
                    f"{place.get('address', city)}"
                )
        else:
            lines.append(f"- No live POIs were returned for {city}; use search hints and generic planning rules.")

        for title, result in search_sections:
            lines.extend(["", f"## {title}"])
            snippets = result.get("snippets", [])
            if snippets:
                for snippet in snippets[:5]:
                    lines.append(f"- {snippet}")
            else:
                lines.append(f"- No search snippets returned for {city}.")
            lines.append(f"- Source: {result.get('source', 'unknown')}")

        lines.extend(
            [
                "",
                "## Planning Rules",
                "- Group stops by area and avoid cross-city zig-zag routing.",
                "- Put outdoor viewpoints early morning or near sunset when possible.",
                "- Keep 20 to 30 minutes of transfer buffer between major stops.",
                "- Add at least one meal stop and one indoor fallback stop to the route.",
            ]
        )

        diagnostics = {
            "search_sources": [result.get("source", "unknown") for _, result in search_sections],
            "live_search_count": sum(1 for _, result in search_sections if result.get("live")),
            "poi_count": len(places),
            "live_map": bool(poi_data.get("live")),
        }
        diagnostics["has_live_data"] = bool(diagnostics["live_search_count"] or diagnostics["live_map"])
        return "\n".join(lines) + "\n", diagnostics
