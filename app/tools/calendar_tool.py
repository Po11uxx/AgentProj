from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any


class CalendarTool:
    name = "calendar"
    description = "Generate structured day schedule"

    def run(self, items: list[Any], start_time: str = "09:00", slot_minutes: int = 90) -> dict[str, Any]:
        cursor = datetime.strptime(start_time, "%H:%M")
        schedule: list[dict[str, Any]] = []

        for item in items:
            title = item if isinstance(item, str) else item.get("title") or item.get("name") or "Stop"
            end = cursor + timedelta(minutes=slot_minutes)
            row: dict[str, Any] = {"title": title, "start": cursor.strftime("%H:%M"), "end": end.strftime("%H:%M")}
            if isinstance(item, dict):
                for key in ["lat", "lon", "address", "category", "details", "visit_focus", "travel_tip"]:
                    if key in item:
                        row[key] = item[key]
            schedule.append(row)
            cursor = end + timedelta(minutes=20)

        return {"schedule": schedule, "slot_minutes": slot_minutes}
