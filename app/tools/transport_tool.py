from __future__ import annotations

import math
from typing import Any


class TransportTool:
    name = "transport"
    description = "Plan point-to-point transportation with public transit details"

    def run(
        self,
        city: str,
        from_place: str,
        to_place: str,
        from_lat: float | None = None,
        from_lon: float | None = None,
        to_lat: float | None = None,
        to_lon: float | None = None,
        budget: int | None = None,
        prefer_public: bool = True,
    ) -> dict[str, Any]:
        city_key = city.lower()
        from_area = self._detect_area(city_key, from_place)
        to_area = self._detect_area(city_key, to_place)

        if "new york" in city_key:
            return self._nyc_route(from_place, to_place, from_area, to_area, budget, prefer_public)
        if "los angeles" in city_key:
            return self._la_route(from_place, to_place, from_area, to_area, budget, prefer_public)

        generic = {
            "from": from_place,
            "to": to_place,
            "mode": "public transit" if prefer_public else "taxi",
            "duration_min": 30,
            "estimated_cost_usd": 3 if prefer_public else 18,
            "route": "Use the nearest metro/bus line for 3-8 stops, then walk 5-10 minutes.",
        }
        if all(v is not None for v in [from_lat, from_lon, to_lat, to_lon]):
            distance_km = self._haversine_km(from_lat, from_lon, to_lat, to_lon)
            if prefer_public:
                generic["mode"] = "public transit"
                generic["duration_min"] = max(12, int(distance_km * 5 + 10))
                generic["estimated_cost_usd"] = round(2 + distance_km * 0.25, 1)
            else:
                generic["mode"] = "rideshare"
                generic["duration_min"] = max(8, int(distance_km * 2.6 + 8))
                generic["estimated_cost_usd"] = round(8 + distance_km * 1.8, 1)
            generic["distance_km"] = round(distance_km, 2)
        return generic

    def _detect_area(self, city: str, place: str) -> str:
        p = place.lower()
        if "new york" in city:
            if any(k in p for k in ["met", "upper east", "86"]):
                return "upper_east_side"
            if any(k in p for k in ["moma", "koreatown", "culture espresso", "midtown"]):
                return "midtown"
            if any(k in p for k in ["chelsea", "market"]):
                return "chelsea"
            if any(k in p for k in ["flatiron", "eataly"]):
                return "flatiron"
            if any(k in p for k in ["village", "soho"]):
                return "downtown"
            return "manhattan"

        if "los angeles" in city:
            if any(k in p for k in ["santa monica", "venice", "pier"]):
                return "westside"
            if any(k in p for k in ["getty", "ucla"]):
                return "brentwood"
            if any(k in p for k in ["lacma", "academy museum", "farmers market"]):
                return "miracle_mile"
            if any(k in p for k in ["koreatown", "ktown"]):
                return "koreatown"
            if any(k in p for k in ["downtown", "broad", "bookstore"]):
                return "downtown"
            return "la_core"

        return "urban"

    def _nyc_route(
        self,
        from_place: str,
        to_place: str,
        from_area: str,
        to_area: str,
        budget: int | None,
        prefer_public: bool,
    ) -> dict[str, Any]:
        if from_area == "midtown" and to_area == "upper_east_side":
            route = "Walk to 5 Av/53 St, take M1/M2/M3/M4 uptown to E 82-86 St, walk 5 minutes."
            return self._pack(from_place, to_place, "bus", 28, 2.9, route)
        if from_area == "upper_east_side" and to_area == "midtown":
            route = "Walk to 86 St station, take 4/5/6 downtown to 51 St or 59 St, walk 8-12 minutes."
            return self._pack(from_place, to_place, "subway", 30, 2.9, route)
        if from_area == "midtown" and to_area == "chelsea":
            route = "Walk to 50 St station, take C/E downtown to 23 St, walk 6-10 minutes."
            return self._pack(from_place, to_place, "subway", 22, 2.9, route)
        if from_area == "chelsea" and to_area == "flatiron":
            route = "Take M23-SBS eastbound to 5 Av, then walk 6-8 minutes to Flatiron area."
            return self._pack(from_place, to_place, "bus", 20, 2.9, route)

        if prefer_public or (budget is not None and budget <= 300):
            return self._pack(
                from_place,
                to_place,
                "subway",
                26,
                2.9,
                "Use nearby subway line (A/C/E or 4/5/6 depending on station), 4-8 stops, then walk 5-10 minutes.",
            )

        return self._pack(
            from_place,
            to_place,
            "taxi",
            18,
            22,
            "Take a yellow cab or rideshare directly between the two places.",
        )

    def _la_route(
        self,
        from_place: str,
        to_place: str,
        from_area: str,
        to_area: str,
        budget: int | None,
        prefer_public: bool,
    ) -> dict[str, Any]:
        if from_area == "westside" and to_area in {"miracle_mile", "downtown", "koreatown"}:
            route = "Take Metro E Line from Downtown Santa Monica to 7th St/Metro Center, transfer by bus/rideshare to destination."
            return self._pack(from_place, to_place, "metro+bus", 55, 3.5, route)

        if prefer_public and (budget is None or budget <= 300):
            return self._pack(
                from_place,
                to_place,
                "metro+bus",
                48,
                3.5,
                "Use LA Metro rail where available, then transfer to Metro Bus for last-mile.",
            )

        return self._pack(
            from_place,
            to_place,
            "rideshare",
            25,
            18,
            "Use Uber/Lyft for direct point-to-point transfer.",
        )

    def _pack(
        self,
        from_place: str,
        to_place: str,
        mode: str,
        duration_min: int,
        estimated_cost_usd: float,
        route: str,
    ) -> dict[str, Any]:
        return {
            "from": from_place,
            "to": to_place,
            "mode": mode,
            "duration_min": duration_min,
            "estimated_cost_usd": estimated_cost_usd,
            "route": route,
        }

    def _haversine_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
        )
        return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))
