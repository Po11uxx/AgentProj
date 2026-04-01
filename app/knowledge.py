from __future__ import annotations

from functools import lru_cache
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from app.config import KNOWLEDGE_DIR


CITY_ALIASES = {
    "hangzhou": ["hangzhou", "杭州"],
    "los angeles": ["los angeles", "la", "洛杉矶"],
    "new york": ["new york", "nyc", "纽约"],
    "san francisco": ["san francisco", "sf", "旧金山"],
    "beijing": ["beijing", "北京"],
    "nanjing": ["nanjing", "南京"],
    "shanghai": ["shanghai", "上海"],
}


FILE_HINTS = {
    "hangzhou": ["hz_travel_tips", "hangzhou_travel", "hangzhou"],
    "los angeles": ["la_travel_tips", "los_angeles_travel", "los_angeles"],
    "new york": ["new_york_travel", "newyork_travel", "new_york"],
    "san francisco": ["san_francisco_travel", "sanfrancisco_travel", "san_francisco"],
    "beijing": ["beijing_travel", "beijing"],
    "nanjing": ["nanjing_travel", "nanjing"],
    "shanghai": ["shanghai_travel", "shanghai"],
}


def slugify_city(city: str) -> str:
    city = normalize_city_name(city).replace(" ", "_")
    slug = re.sub(r"[^\w\u4e00-\u9fff]+", "_", city.strip().lower()).strip("_")
    return slug or "city"


def city_variants(city: str) -> list[str]:
    normalized = normalize_city_name(city)
    lower = city.strip().lower()
    variants = {city.strip(), lower, city.strip().title(), normalized, normalized.lower(), normalized.title()}
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


@lru_cache(maxsize=256)
def normalize_city_name(city: str) -> str:
    cleaned = city.strip()
    canonical = canonical_city_key(cleaned)
    if canonical != cleaned.lower():
        return canonical.title()
    if re.fullmatch(r"[A-Za-z][A-Za-z\s\-]{1,60}", cleaned):
        return " ".join(part.capitalize() for part in cleaned.split())
    english_name = _lookup_city_english_name(cleaned)
    return english_name or cleaned


def _lookup_city_english_name(city: str) -> str | None:
    try:
        resp = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "en", "format": "json"},
            timeout=4,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if results:
            name = str(results[0].get("name", "")).strip()
            if name:
                return name
    except Exception:
        pass

    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": city, "format": "jsonv2", "limit": 1, "accept-language": "en"},
            headers={"User-Agent": "agent-proj/1.0"},
            timeout=4,
        )
        resp.raise_for_status()
        rows = resp.json()
        if rows:
            display_name = str(rows[0].get("display_name", "")).strip()
            if display_name:
                return display_name.split(",")[0].strip()
    except Exception:
        pass
    return None


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
        if not self._looks_like_valid_city(city):
            return {
                "city": city,
                "created": False,
                "path": "",
                "status": "skipped_invalid_city",
            }
        existing = self.find_city_knowledge_file(city)
        if existing is not None:
            migrated = self._migrate_to_preferred_filename(city=city, file_path=existing)
            return {
                "city": city,
                "created": False,
                "path": str(migrated),
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

    def update_city_knowledge_from_places(self, city: str, places: list[dict[str, Any]]) -> dict[str, Any]:
        if not self._looks_like_valid_city(city):
            return {"city": city, "updated": False, "status": "skipped_invalid_city"}
        file_path = self.find_city_knowledge_file(city)
        if file_path is None:
            return {"city": city, "updated": False, "status": "missing_knowledge"}
        file_path = self._migrate_to_preferred_filename(city=city, file_path=file_path)
        if not places:
            return {"city": city, "updated": False, "status": "no_places"}

        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return {"city": city, "updated": False, "status": "read_failed", "path": str(file_path)}

        existing_lines = content.splitlines()
        poi_lines = self._extract_existing_poi_lines(existing_lines)
        existing_keys = {self._poi_key_from_line(line) for line in poi_lines}
        new_lines: list[str] = []
        for place in places:
            if str(place.get("source_type", "")).startswith("live_map") or place.get("source_type") is None:
                line = self._format_poi_line(place, city)
                key = self._poi_key_from_line(line)
                if key and key not in existing_keys:
                    existing_keys.add(key)
                    new_lines.append(line)

        if not new_lines:
            return {"city": city, "updated": False, "status": "no_new_poi", "path": str(file_path)}

        updated = self._insert_poi_lines(existing_lines, new_lines)
        file_path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")
        return {
            "city": city,
            "updated": True,
            "status": "poi_appended",
            "path": str(file_path),
            "added_count": len(new_lines),
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

    def _looks_like_valid_city(self, city: str) -> bool:
        cleaned = city.strip()
        if not cleaned:
            return False
        invalid_tokens = ["并生成", "生成", "行程表", "天气", "人流", "预算", "考虑天气", "考虑人流"]
        if any(token in cleaned for token in invalid_tokens):
            return False
        return True

    def _migrate_to_preferred_filename(self, city: str, file_path: Path) -> Path:
        preferred = self.knowledge_dir / f"{slugify_city(city)}_travel_guide.md"
        if file_path == preferred:
            return file_path
        if preferred.exists():
            return preferred
        try:
            file_path.rename(preferred)
            return preferred
        except Exception:
            return file_path

    def _extract_existing_poi_lines(self, lines: list[str]) -> list[str]:
        poi_lines: list[str] = []
        in_poi_section = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## "):
                in_poi_section = stripped.lower() == "## real poi candidates"
                continue
            if in_poi_section and stripped.startswith("- "):
                poi_lines.append(stripped)
        return poi_lines

    def _format_poi_line(self, place: dict[str, Any], city: str) -> str:
        return (
            f"- {place.get('name', 'POI')} | {place.get('category', 'place')} | "
            f"{place.get('address', city)}"
        )

    def _poi_key_from_line(self, line: str) -> str:
        parts = [part.strip().lower() for part in line[2:].split("|")] if line.startswith("- ") else []
        if len(parts) < 3:
            return ""
        return f"{parts[0]}|{parts[2]}"

    def _insert_poi_lines(self, lines: list[str], new_lines: list[str]) -> list[str]:
        has_poi_section = any(line.strip().lower() == "## real poi candidates" for line in lines)
        if not has_poi_section:
            output = list(lines)
            if output and output[-1].strip():
                output.append("")
            output.append("## Real POI Candidates")
            output.extend(new_lines)
            return output

        output: list[str] = []
        inserted = False
        in_poi_section = False
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("## "):
                if in_poi_section and not inserted:
                    output.extend(new_lines)
                    inserted = True
                in_poi_section = stripped.lower() == "## real poi candidates"
                output.append(line)
                continue
            if in_poi_section and not stripped.startswith("- ") and not inserted and stripped:
                output.extend(new_lines)
                inserted = True
            output.append(line)
        if in_poi_section and not inserted:
            output.extend(new_lines)
        return output
