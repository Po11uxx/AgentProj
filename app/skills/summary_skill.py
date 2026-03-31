from __future__ import annotations

from typing import Any


class SummarySkill:
    name = "summary"

    def run(self, payload: dict[str, Any]) -> str:
        weather = payload.get("weather", {})
        schedule = payload.get("schedule", {}).get("schedule", [])
        transport_plan = payload.get("transport_plan", [])
        prefs = payload.get("user_preferences", {})
        ctx = payload.get("retrieved_context", [])
        knowledge_status = payload.get("knowledge_status", {})

        lines: list[str] = []
        lines.append(f"Trip Plan for {payload.get('city')} on {payload.get('date')}")
        lines.append(f"Request: {payload.get('note', '')}")
        lines.append(
            "Weather: "
            f"{weather.get('condition')}, {weather.get('temp_min')}-{weather.get('temp_max')}C, "
            f"rain prob {weather.get('precipitation_probability')}%"
        )

        if prefs:
            lines.append(f"Preferences applied: {prefs}")
        if knowledge_status:
            if knowledge_status.get("created"):
                action = "generated and cached"
            elif knowledge_status.get("status") == "cached":
                action = "loaded from cache"
            else:
                action = f"not cached ({knowledge_status.get('status')})"
            lines.append(f"Knowledge: {action} ({knowledge_status.get('path', 'n/a')})")

        lines.append("Schedule:")
        for item in schedule:
            address = item.get("address")
            if address:
                lines.append(f"- {item['start']}-{item['end']} {item['title']} ({address})")
            else:
                lines.append(f"- {item['start']}-{item['end']} {item['title']}")
            if item.get("details"):
                lines.append(f"  Details: {item['details']}")
            if item.get("visit_focus"):
                lines.append(f"  Focus: {item['visit_focus']}")
            if item.get("travel_tip"):
                lines.append(f"  Tip: {item['travel_tip']}")

        if transport_plan:
            lines.append("Transport Plan:")
            for leg in transport_plan:
                lines.append(
                    f"- {leg.get('after_slot')} {leg.get('from')} -> {leg.get('to')} | "
                    f"{leg.get('mode')} | {leg.get('duration_min')} min | "
                    f"${leg.get('estimated_cost_usd')} | "
                    f"{(str(leg.get('distance_km')) + ' km | ') if leg.get('distance_km') is not None else ''}"
                    f"route: {leg.get('route')}"
                )

        if ctx:
            lines.append("RAG hints:")
            for c in ctx[:2]:
                lines.append(f"- {c[:120]}...")

        lines.append("Tips: Keep transport time buffer (20-30 mins) between major spots.")
        return "\n".join(lines)
