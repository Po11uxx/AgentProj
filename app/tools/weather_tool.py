from __future__ import annotations

from datetime import date
from typing import Any

import requests


class WeatherTool:
    name = "weather"
    description = "Get weather by city and date"

    _CITY_COORDS = {
        "hangzhou": (30.27, 120.15),
        "los angeles": (34.05, -118.24),
        "new york": (40.71, -74.01),
        "san francisco": (37.77, -122.42),
        "beijing": (39.90, 116.40),
        "nanjing": (32.06, 118.79),
        "shanghai": (31.23, 121.47),
    }

    def run(self, city: str, target_date: str | None = None) -> dict[str, Any]:
        target_date = target_date or str(date.today())
        lat_lon = self._CITY_COORDS.get(city.lower()) or self._geocode_city(city)
        if lat_lon is None:
            return {
                "city": city,
                "date": target_date,
                "condition": "partly cloudy",
                "temp_max": 25,
                "temp_min": 16,
                "precipitation_probability": 10,
            }
        lat, lon = lat_lon

        try:
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,"
                "precipitation_probability_max&timezone=auto"
            )
            resp = requests.get(url, timeout=6)
            resp.raise_for_status()
            data = resp.json().get("daily", {})
            dates = data.get("time", [])
            if target_date in dates:
                idx = dates.index(target_date)
                return {
                    "city": city,
                    "date": target_date,
                    "condition": "forecast",
                    "temp_max": data.get("temperature_2m_max", [None])[idx],
                    "temp_min": data.get("temperature_2m_min", [None])[idx],
                    "precipitation_probability": data.get("precipitation_probability_max", [None])[idx],
                }
        except Exception:
            pass

        # Fallback mock weather.
        return {
            "city": city,
            "date": target_date,
            "condition": "partly cloudy",
            "temp_max": 25,
            "temp_min": 16,
            "precipitation_probability": 10,
        }

    def _geocode_city(self, city: str) -> tuple[float, float] | None:
        try:
            resp = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1, "language": "en", "format": "json"},
                timeout=6,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return None
            return float(results[0]["latitude"]), float(results[0]["longitude"])
        except Exception:
            return None
