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
        knowledge_update = payload.get("knowledge_update", {})
        language = payload.get("language", "en")

        labels = {
            "en": {
                "title": "Trip Plan for",
                "request": "Request",
                "weather": "Weather",
                "prefs": "Preferences applied",
                "knowledge": "Knowledge",
                "schedule": "Schedule",
                "details": "Details",
                "focus": "Focus",
                "tip": "Tip",
                "transport": "Transport Plan",
                "hints": "RAG hints",
                "tips": "Tips: Keep transport time buffer (20-30 mins) between major spots.",
                "generated": "generated and cached",
                "cached": "loaded from cache",
                "not_cached": "not cached",
                "route": "route",
                "min": "min",
            },
            "zh": {
                "title": "旅行计划：",
                "request": "请求",
                "weather": "天气",
                "prefs": "已应用偏好",
                "knowledge": "知识库",
                "schedule": "行程安排",
                "details": "详情",
                "focus": "重点",
                "tip": "提示",
                "transport": "交通方案",
                "hints": "RAG 提示",
                "tips": "建议：主要景点之间预留 20-30 分钟交通缓冲。",
                "generated": "已联网生成并缓存",
                "cached": "已从缓存加载",
                "not_cached": "未缓存",
                "route": "路线",
                "min": "分钟",
            },
        }[language if language in {"zh", "en"} else "en"]

        lines: list[str] = []
        if language == "zh":
            lines.append(f"{labels['title']}{payload.get('city')} | 日期：{payload.get('date')}")
            lines.append(f"{labels['request']}：{payload.get('note', '')}")
            lines.append(
                f"{labels['weather']}：{weather.get('condition')}，"
                f"{weather.get('temp_min')}-{weather.get('temp_max')}C，"
                f"降雨概率 {weather.get('precipitation_probability')}%"
            )
        else:
            lines.append(f"{labels['title']} {payload.get('city')} on {payload.get('date')}")
            lines.append(f"{labels['request']}: {payload.get('note', '')}")
            lines.append(
                f"{labels['weather']}: "
                f"{weather.get('condition')}, {weather.get('temp_min')}-{weather.get('temp_max')}C, "
                f"rain prob {weather.get('precipitation_probability')}%"
            )

        if prefs:
            sep = "：" if language == "zh" else ": "
            lines.append(f"{labels['prefs']}{sep}{prefs}")
        if knowledge_status:
            if knowledge_status.get("created"):
                action = labels["generated"]
            elif knowledge_status.get("status") == "cached":
                action = labels["cached"]
            else:
                action = f"{labels['not_cached']} ({knowledge_status.get('status')})"
            sep = "：" if language == "zh" else ": "
            lines.append(f"{labels['knowledge']}{sep}{action} ({knowledge_status.get('path', 'n/a')})")
        if knowledge_update.get("updated"):
            if language == "zh":
                lines.append(f"知识库增量更新：已追加 {knowledge_update.get('added_count', 0)} 个实时点位")
            else:
                lines.append(
                    f"Knowledge incremental update: appended {knowledge_update.get('added_count', 0)} live POIs"
                )

        lines.append(f"{labels['schedule']}:")
        for item in schedule:
            address = item.get("address")
            if address:
                lines.append(f"- {item['start']}-{item['end']} {item['title']} ({address})")
            else:
                lines.append(f"- {item['start']}-{item['end']} {item['title']}")
            if item.get("details"):
                lines.append(f"  {labels['details']}: {item['details']}")
            if item.get("visit_focus"):
                lines.append(f"  {labels['focus']}: {item['visit_focus']}")
            if item.get("travel_tip"):
                lines.append(f"  {labels['tip']}: {item['travel_tip']}")

        if transport_plan:
            lines.append(f"{labels['transport']}:")
            for leg in transport_plan:
                lines.append(
                    f"- {leg.get('after_slot')} {leg.get('from')} -> {leg.get('to')} | "
                    f"{leg.get('mode')} | {leg.get('duration_min')} {labels['min']} | "
                    f"${leg.get('estimated_cost_usd')} | "
                    f"{(str(leg.get('distance_km')) + ' km | ') if leg.get('distance_km') is not None else ''}"
                    f"{labels['route']}: {leg.get('route')}"
                )

        if ctx:
            lines.append(f"{labels['hints']}:")
            for c in ctx[:2]:
                lines.append(f"- {c[:120]}...")

        lines.append(labels["tips"])
        return "\n".join(lines)
