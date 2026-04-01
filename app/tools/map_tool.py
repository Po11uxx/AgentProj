from __future__ import annotations

import math
import re
from typing import Any

import requests


class MapTool:
    name = "map"
    description = "Get city coordinates and live POI candidates from map search"

    def run(self, city: str, intents: list[str], limit_per_intent: int = 3) -> dict[str, Any]:
        center = self._geocode_city(city)
        places: list[dict[str, Any]] = []
        seen: set[str] = set()
        live = bool(center.get("live"))

        for intent in intents:
            candidates = self._search_places(
                city=city,
                query=intent,
                center_lat=center.get("lat"),
                center_lon=center.get("lon"),
                limit=limit_per_intent * 4,
                max_distance_km=35.0,
            )
            for c in candidates:
                key = c["name"].lower()
                if key in seen:
                    continue
                seen.add(key)
                places.append(c)
                live = True

        places.sort(key=lambda p: (p.get("_city_match_score", 0), -float(p.get("_distance_km", 9999.0))), reverse=True)

        if not places:
            places = [
                {
                    "name": f"{city} City Center Walk",
                    "lat": center.get("lat"),
                    "lon": center.get("lon"),
                    "address": center.get("display_name", city),
                    "category": "landmark",
                    "source_type": "live_map_fallback",
                }
            ]

        trimmed_places = []
        for place in places[: max(5, limit_per_intent * len(intents))]:
            trimmed_places.append({k: v for k, v in place.items() if not k.startswith("_")})

        return {
            "city": city,
            "center": center,
            "intents": intents,
            "places": trimmed_places,
            "source": "nominatim",
            "live": live,
        }

    def search_specific_place(self, city: str, place_name: str) -> list[dict[str, Any]]:
        center = self._geocode_city(city)
        places = self._search_places(
            city=city,
            query=place_name,
            center_lat=center.get("lat"),
            center_lon=center.get("lon"),
            limit=3,
            max_distance_km=35.0,
        )
        return [{k: v for k, v in place.items() if not k.startswith("_")} for place in places]

    def _geocode_city(self, city: str) -> dict[str, Any]:
        try:
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": city, "format": "jsonv2", "limit": 1},
                headers={"User-Agent": "agent-proj/1.0"},
                timeout=8,
            )
            resp.raise_for_status()
            rows = resp.json()
            if rows:
                row = rows[0]
                return {
                    "lat": float(row.get("lat", 0.0)),
                    "lon": float(row.get("lon", 0.0)),
                    "display_name": row.get("display_name", city),
                    "live": True,
                }
        except Exception:
            pass

        return {"lat": 0.0, "lon": 0.0, "display_name": city, "live": False}

    def _search_places(
        self,
        city: str,
        query: str,
        center_lat: float | None = None,
        center_lon: float | None = None,
        limit: int = 3,
        max_distance_km: float = 35.0,
    ) -> list[dict[str, Any]]:
        try:
            q = f"{query}, {city}"
            resp = requests.get(
                "https://nominatim.openstreetmap.org/search",
                params={"q": q, "format": "jsonv2", "limit": limit},
                headers={"User-Agent": "agent-proj/1.0"},
                timeout=8,
            )
            resp.raise_for_status()
            rows = resp.json()

            results: list[dict[str, Any]] = []
            city_variants = self._city_variants(city)
            generic_names = {
                "restaurant",
                "cafe",
                "museum",
                "gallery",
                "park",
                "hotel",
                "road",
                "street",
                "avenue",
                "viewpoint circle",
                "杭州路",
            }
            for row in rows:
                name = row.get("name") or row.get("display_name", "").split(",")[0]
                if not name:
                    continue
                name_lower = name.strip().lower()
                if name_lower in generic_names:
                    continue
                if self._is_generic_linear_feature(name=name, row_type=row.get("type", ""), category=row.get("class", "")):
                    continue
                lat = float(row.get("lat", 0.0))
                lon = float(row.get("lon", 0.0))
                display_name = row.get("display_name", "")
                display_lower = display_name.lower()
                address_only = display_lower.split(",", 1)[1].strip() if "," in display_lower else display_lower

                distance_km = 0.0
                if center_lat is not None and center_lon is not None:
                    distance_km = self._haversine_km(center_lat, center_lon, lat, lon)

                city_match_score = self._city_match_score(display_name=address_only, city_variants=city_variants)
                if center_lat is not None and center_lon is not None:
                    if distance_km > max_distance_km and city_match_score <= 0:
                        continue
                    if city_match_score <= 0 and distance_km > 12:
                        continue
                    if city_match_score < 2 and distance_km > 5 and self._looks_like_street_or_area(name, display_name):
                        continue
                elif city_match_score <= 0:
                    continue

                results.append(
                    {
                        "name": name,
                        "lat": lat,
                        "lon": lon,
                        "address": display_name,
                        "category": query,
                        "source_type": "live_map",
                        "_distance_km": round(distance_km, 2),
                        "_city_match_score": city_match_score,
                    }
                )
            results.sort(key=lambda item: (item["_city_match_score"], -item["_distance_km"]), reverse=True)
            return results[:limit]
        except Exception:
            return []

    def _city_variants(self, city: str) -> list[str]:
        variants = {city.lower()}
        aliases = {
            "hangzhou": ["hangzhou", "杭州", "杭州市"],
            "los angeles": ["los angeles", "洛杉矶"],
            "new york": ["new york", "nyc", "纽约"],
            "san francisco": ["san francisco", "旧金山"],
            "beijing": ["beijing", "北京", "北京市"],
            "shanghai": ["shanghai", "上海", "上海市"],
        }
        city_lower = city.lower()
        for canonical, values in aliases.items():
            if city_lower == canonical or city_lower in values:
                variants.update(v.lower() for v in values)
        return [v for v in variants if v]

    def _city_match_score(self, display_name: str, city_variants: list[str]) -> int:
        score = 0
        for variant in city_variants:
            if re.search(r"[\u4e00-\u9fff]", variant):
                if variant in display_name:
                    score = max(score, 3)
            elif re.search(rf"\b{re.escape(variant)}\b", display_name):
                score = max(score, 3)
        return score

    def _is_generic_linear_feature(self, name: str, row_type: str, category: str) -> bool:
        row_type = (row_type or "").lower()
        category = (category or "").lower()
        name_lower = name.lower()
        if row_type in {"road", "residential", "footway", "path", "pedestrian"}:
            return True
        if category == "highway":
            return True
        return bool(re.search(r"(路|街|大道|巷|弄|circle|road|street|avenue|boulevard)$", name_lower))

    def _looks_like_street_or_area(self, name: str, display_name: str) -> bool:
        blob = f"{name} {display_name}".lower()
        return bool(re.search(r"(路|街|大道|巷|弄|社区|circle|road|street|avenue|district)", blob))

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
        )
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))
