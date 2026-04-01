from __future__ import annotations

import re
from typing import Any

from app.knowledge import KnowledgeBaseManager, normalize_city_name


class TravelPlanningSkill:
    name = "travel_planning"
    _INVALID_CITY_TOKENS = {
        "并生成",
        "生成",
        "行程表",
        "天气",
        "人流",
        "预算",
        "考虑天气",
        "考虑人流",
        "周末",
        "一日游",
    }

    def __init__(self) -> None:
        self.knowledge_manager = KnowledgeBaseManager()

    def run(
        self,
        user_message: str,
        toolkit: dict[str, object],
        city: str,
        date_str: str,
        retrieved_context: list[str],
        user_preferences: dict[str, Any],
    ) -> dict[str, Any]:
        weather_tool = toolkit["weather"]
        search_tool = toolkit["search"]
        map_tool = toolkit["map"]
        calendar_tool = toolkit["calendar"]
        transport_tool = toolkit["transport"]

        language = self._detect_language(user_message)
        resolved_city = self._resolve_city(user_message, city)
        weather = weather_tool.run(city=resolved_city, target_date=date_str)
        search = search_tool.run(query=f"{resolved_city} {user_message} itinerary crowd tips")
        preferred_scenes = self._normalize_preferred_scenes(user_preferences)
        intents = self._infer_intents(
            user_message=user_message,
            rainy=weather.get("precipitation_probability", 0) > 40,
            preferred_scenes=preferred_scenes,
        )
        required_terms = self._extract_required_terms(user_message)
        map_data = map_tool.run(
            city=resolved_city,
            intents=list(dict.fromkeys([*intents, *required_terms]))[:4],
            limit_per_intent=3,
        )

        attractions = self._build_attractions(
            user_message=user_message,
            resolved_city=resolved_city,
            prefs=user_preferences,
            is_rainy=weather.get("precipitation_probability", 0) > 40,
            map_data=map_data,
            required_terms=required_terms,
            preferred_scenes=preferred_scenes,
            language=language,
            retrieved_context=retrieved_context,
        )

        schedule = calendar_tool.run(items=attractions, start_time="08:30", slot_minutes=95)
        schedule["schedule"] = self._enforce_required_in_schedule(
            schedule=schedule.get("schedule", []),
            required_terms=required_terms,
            all_places=map_data.get("places", []),
            city=resolved_city,
            language=language,
        )
        transport_plan = self._build_transport_plan(
            user_message=user_message,
            city=resolved_city,
            schedule=schedule.get("schedule", []),
            budget=user_preferences.get("budget"),
            transport_tool=transport_tool,
        )

        return {
            "city": resolved_city,
            "date": date_str,
            "weather": weather,
            "search": search,
            "map_data": map_data,
            "schedule": schedule,
            "transport_plan": transport_plan,
            "retrieved_context": retrieved_context,
            "user_preferences": user_preferences,
            "language": language,
            "request": user_message,
            "note": (
                f"基于工具结果和 RAG 上下文生成，原始请求：{user_message}"
                if language == "zh"
                else f"Plan generated from tool outputs and RAG context for request: {user_message}"
            ),
        }

    def _detect_language(self, text: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"

    def _resolve_city(self, user_message: str, default_city: str) -> str:
        lower = user_message.lower()
        aliases = {
            "hangzhou": ["hangzhou", "杭州"],
            "los angeles": ["los angeles", "la", "洛杉矶"],
            "new york": ["new york", "nyc", "纽约"],
            "san francisco": ["san francisco", "sf", "旧金山"],
            "beijing": ["beijing", "北京"],
            "nanjing": ["nanjing", "南京"],
            "shanghai": ["shanghai", "上海"],
        }
        for canonical, keys in aliases.items():
            if any(self._alias_matches(text=lower, alias=k) for k in keys):
                return normalize_city_name(canonical)

        # Chinese city extraction: parse the short window before intent keywords,
        # then strip prompt filler words and keep the trailing city-like token.
        suffixes = ["一日游", "旅游", "行程", "博物馆", "美食", "徒步"]
        for suffix in suffixes:
            idx = user_message.find(suffix)
            if idx == -1:
                continue
            window = user_message[max(0, idx - 12) : idx]
            for filler in ["帮我", "请", "规划", "安排", "做", "一个", "周末", "周中", "假期", "假日", "在", "去", "到"]:
                window = window.replace(filler, "")
            m = re.search(r"([\u4e00-\u9fff]{2,8})$", window)
            if m and self._is_valid_city_candidate(m.group(1)):
                return normalize_city_name(m.group(1))

        zh_patterns = [
            r"(?:去|在|到)([\u4e00-\u9fff]{2,8})(?:市)?",
            r"(?:周末|周中|假期|假日)([\u4e00-\u9fff]{2,8})(?:市)?",
        ]
        for pattern in zh_patterns:
            zh_match = re.search(pattern, user_message)
            if zh_match:
                candidate = zh_match.group(1).strip()
                if self._is_valid_city_candidate(candidate):
                    return normalize_city_name(candidate)

        # Generic English pattern: in/for/to <city words>
        en_match = re.search(
            r"(?:in|for|to)\s+([A-Za-z][A-Za-z\-\s]{1,40})",
            lower,
        )
        if en_match:
            candidate = re.split(
                r"\b(one day|itinerary|trip|museum|food|hike|plan|budget)\b",
                en_match.group(1),
                maxsplit=1,
            )[0].strip(" ,.")
            if candidate:
                return normalize_city_name(candidate)

        # Last fallback: first Chinese token that looks like a city.
        zh_city_token = re.search(r"([\u4e00-\u9fff]{2,8})(?:市|州)", user_message)
        if zh_city_token and self._is_valid_city_candidate(zh_city_token.group(1)):
            return normalize_city_name(zh_city_token.group(1))
        return normalize_city_name(default_city)

    def _alias_matches(self, text: str, alias: str) -> bool:
        alias_lower = alias.lower()
        if re.search(r"[\u4e00-\u9fff]", alias_lower):
            return alias_lower in text
        if len(alias_lower) <= 3:
            return bool(re.search(rf"(?<![a-z]){re.escape(alias_lower)}(?![a-z])", text))
        return bool(re.search(rf"\b{re.escape(alias_lower)}\b", text))

    def _is_valid_city_candidate(self, candidate: str) -> bool:
        cleaned = re.sub(r"[，。,.\s；;:：!！?？、]", "", candidate)
        if not cleaned:
            return False
        if cleaned in self._INVALID_CITY_TOKENS:
            return False
        return not any(token in cleaned for token in self._INVALID_CITY_TOKENS)

    def _build_attractions(
        self,
        user_message: str,
        resolved_city: str,
        prefs: dict[str, Any],
        is_rainy: bool,
        map_data: dict[str, Any],
        required_terms: list[str],
        preferred_scenes: set[str],
        language: str,
        retrieved_context: list[str],
    ) -> list[dict[str, Any]]:
        text = user_message.lower()
        city = resolved_city.lower()
        live_places = list(map_data.get("places", []))
        places = self._merge_places(
            primary_places=live_places,
            secondary_places=[],
        )
        places = self._supplement_real_places(
            city=resolved_city,
            existing_places=places,
            preferred_scenes=preferred_scenes,
            retrieved_context=retrieved_context,
        )

        selected_places = self._select_places(
            places=places,
            text=text,
            rainy=is_rainy,
            preferred_scenes=preferred_scenes,
            language=language,
        )
        if selected_places:
            enriched = self._ensure_required_terms(
                selected_places=selected_places[:5],
                all_places=places,
                required_terms=required_terms,
                city=resolved_city,
                language=language,
            )
            return self._fill_to_target_slots(
                selected_places=enriched,
                all_places=places,
                target_slots=5,
                city=resolved_city,
                preferred_scenes=preferred_scenes,
                language=language,
            )

        if is_rainy:
            if "new york" in city:
                return [
                    {"name": "Daily Provisions (breakfast)", "category": "food"},
                    {"name": "The Metropolitan Museum of Art", "category": "museum"},
                    {"name": "MoMA indoor route", "category": "museum"},
                    {"name": "Chelsea Market", "category": "market"},
                    {"name": "Koreatown dinner on 32nd St", "category": "food"},
                ]
            if "los angeles" in city:
                return [
                    {"name": "Verve Coffee Roasters (breakfast)", "category": "food"},
                    {"name": "The Broad Museum", "category": "museum"},
                    {"name": "The Last Bookstore + Grand Central Market", "category": "market"},
                    {"name": "Academy Museum of Motion Pictures", "category": "museum"},
                    {"name": "Koreatown BBQ dinner", "category": "food"},
                ]
            return [
                {"name": f"{resolved_city} indoor brunch", "category": "food"},
                {"name": f"{resolved_city} flagship museum", "category": "museum"},
                {"name": f"{resolved_city} indoor market", "category": "market"},
                {"name": f"{resolved_city} gallery/theater", "category": "culture"},
                {"name": f"{resolved_city} downtown dinner", "category": "food"},
            ]

        if "nature" in preferred_scenes or "natural scenery" in preferred_scenes or re.search(
            r"自然风景|自然景观|风景|山水|公园|湖|徒步|hike|nature|scenic",
            user_message,
            flags=re.IGNORECASE,
        ):
            if "los angeles" in city:
                return [
                    {"name": "Sightglass / nature-side breakfast", "category": "food"},
                    {"name": "Griffith Park scenic trail", "category": "park"},
                    {"name": "Lake Hollywood viewpoint", "category": "viewpoint"},
                    {"name": "Botanical garden or lakeside lunch", "category": "park"},
                    {"name": "Sunset overlook dinner nearby", "category": "food"},
                ]
            return [
                {"name": f"{resolved_city} park breakfast cafe", "category": "food"},
                {"name": f"{resolved_city} major urban park or lake walk", "category": "park"},
                {"name": f"{resolved_city} scenic viewpoint", "category": "viewpoint"},
                {"name": f"{resolved_city} botanical garden / riverside stroll", "category": "park"},
                {"name": f"{resolved_city} sunset dinner near green space", "category": "food"},
            ]

        if "beach" in preferred_scenes or ("beach" in text or "海边" in text):
            if "los angeles" in city:
                return [
                    {"name": "Blue Bottle Santa Monica (breakfast)", "category": "food"},
                    {"name": "Palisades Park viewpoint", "category": "viewpoint"},
                    {"name": "Venice Beach Boardwalk lunch", "category": "food"},
                    {"name": "Santa Monica Pier activities", "category": "landmark"},
                    {"name": "Water Grill Santa Monica (sunset dinner)", "category": "food"},
                ]
            return [
                {"name": f"{resolved_city} seaside breakfast cafe", "category": "food"},
                {"name": f"{resolved_city} coastal viewpoint", "category": "viewpoint"},
                {"name": f"{resolved_city} harbor lunch", "category": "food"},
                {"name": f"{resolved_city} pier/waterfront activities", "category": "landmark"},
                {"name": f"{resolved_city} sunset seafood dinner", "category": "food"},
            ]
        if "museum" in preferred_scenes or ("museum" in text or "博物馆" in text):
            if "new york" in city:
                return [
                    {"name": "Culture Espresso (breakfast)", "category": "food"},
                    {"name": "The Met highlights route", "category": "museum"},
                    {"name": "MoMA modern art wing", "category": "museum"},
                    {"name": "Chelsea galleries walk", "category": "culture"},
                    {"name": "Eataly Flatiron dinner", "category": "food"},
                ]
            if "los angeles" in city:
                return [
                    {"name": "Maru Coffee (breakfast)", "category": "food"},
                    {"name": "The Getty Center", "category": "museum"},
                    {"name": "LACMA + Urban Light", "category": "museum"},
                    {"name": "The Academy Museum", "category": "museum"},
                    {"name": "Dinner at The Original Farmers Market", "category": "food"},
                ]
            return [
                {"name": f"{resolved_city} specialty coffee breakfast", "category": "food"},
                {"name": f"{resolved_city} flagship museum", "category": "museum"},
                {"name": f"{resolved_city} art gallery", "category": "culture"},
                {"name": f"{resolved_city} cultural district stroll", "category": "culture"},
                {"name": f"{resolved_city} museum district dinner", "category": "food"},
            ]
        if re.search(r"hike|hiking|徒步|爬山", text):
            if "los angeles" in city:
                return [
                    {"name": "Alcove Cafe (pre-hike breakfast)", "category": "food"},
                    {"name": "Runyon Canyon hike", "category": "hiking"},
                    {"name": "Los Feliz lunch stop", "category": "food"},
                    {"name": "Griffith Observatory sunset", "category": "viewpoint"},
                    {"name": "Thai Town dinner", "category": "food"},
                ]
            return [
                {"name": f"{resolved_city} trailhead breakfast", "category": "food"},
                {"name": f"{resolved_city} morning hike route", "category": "hiking"},
                {"name": f"{resolved_city} scenic lunch", "category": "food"},
                {"name": f"{resolved_city} city viewpoint", "category": "viewpoint"},
                {"name": f"{resolved_city} recovery dinner", "category": "food"},
            ]
        if re.search(r"food|美食|吃|餐厅", text):
            if "new york" in city:
                return [
                    {"name": "Ess-a-Bagel breakfast", "category": "food"},
                    {"name": "Katz's Delicatessen early lunch", "category": "food"},
                    {"name": "Chelsea Market tasting walk", "category": "food"},
                    {"name": "Levain Bakery + coffee break", "category": "food"},
                    {"name": "West Village dinner route", "category": "food"},
                ]
            return [
                {"name": f"{resolved_city} local breakfast spot", "category": "food"},
                {"name": f"{resolved_city} food market tasting", "category": "food"},
                {"name": f"{resolved_city} signature lunch", "category": "food"},
                {"name": f"{resolved_city} dessert / coffee break", "category": "food"},
                {"name": f"{resolved_city} chef-recommended dinner", "category": "food"},
            ]
        return [
            {"name": f"{resolved_city} downtown breakfast", "category": "food"},
            {"name": f"{resolved_city} city landmark", "category": "landmark"},
            {"name": f"{resolved_city} popular cultural attraction", "category": "culture"},
            {"name": f"{resolved_city} scenic sunset location", "category": "viewpoint"},
            {"name": f"{resolved_city} local hotspot dinner", "category": "food"},
        ]

    def _build_transport_plan(
        self,
        user_message: str,
        city: str,
        schedule: list[dict[str, Any]],
        budget: int | None,
        transport_tool: object,
    ) -> list[dict[str, Any]]:
        lower = user_message.lower()
        prefer_public = True
        if any(k in user_message for k in ["打车", "网约车", "出租车"]) or any(
            k in lower for k in ["taxi", "uber", "lyft", "rideshare"]
        ):
            prefer_public = False

        legs: list[dict[str, Any]] = []
        for i in range(len(schedule) - 1):
            from_stop = schedule[i]["title"]
            to_stop = schedule[i + 1]["title"]
            leg = transport_tool.run(
                city=city,
                from_place=from_stop,
                to_place=to_stop,
                from_lat=schedule[i].get("lat"),
                from_lon=schedule[i].get("lon"),
                to_lat=schedule[i + 1].get("lat"),
                to_lon=schedule[i + 1].get("lon"),
                budget=budget,
                prefer_public=prefer_public,
            )
            leg["after_slot"] = f"{schedule[i]['end']} -> {schedule[i + 1]['start']}"
            legs.append(leg)
        return legs

    def _normalize_preferred_scenes(self, prefs: dict[str, Any]) -> set[str]:
        raw = prefs.get("preferred_scene")
        values = raw if isinstance(raw, list) else [raw] if raw else []
        normalized: set[str] = set()
        for value in values:
            token = str(value).strip().lower()
            if not token:
                continue
            normalized.add(token)
            if token in {"natural scenery", "nature", "scenery", "natural", "outdoor"}:
                normalized.add("nature")
            if token in {"museum", "art", "gallery"}:
                normalized.add("museum")
            if token in {"beach", "coast", "seaside"}:
                normalized.add("beach")
        return normalized

    def _infer_intents(self, user_message: str, rainy: bool, preferred_scenes: set[str]) -> list[str]:
        text = user_message.lower()
        intents: list[str] = []
        if "nature" in preferred_scenes or re.search(
            r"自然风景|自然景观|风景|山水|公园|湖|徒步|hike|nature|scenic",
            user_message,
            flags=re.IGNORECASE,
        ):
            intents.extend(["park", "viewpoint", "garden", "lake", "trail"])
        if "museum" in preferred_scenes or "museum" in text or "博物馆" in user_message:
            intents.extend(["museum", "gallery"])
        if "food" in text or "美食" in user_message or "餐厅" in user_message:
            intents.extend(["restaurant", "cafe"])
        if "beach" in text or "海边" in user_message:
            intents.extend(["beach", "viewpoint"])
        if "hike" in text or "徒步" in user_message or "爬山" in user_message:
            intents.extend(["hiking trail", "viewpoint"])
        if rainy:
            intents.append("indoor attractions")
        if not intents:
            intents = ["landmark", "museum", "restaurant"]
        # preserve order while deduplicating
        return list(dict.fromkeys(intents))

    def _select_places(
        self,
        places: list[dict[str, Any]],
        text: str,
        rainy: bool,
        preferred_scenes: set[str],
        language: str = "en",
    ) -> list[dict[str, Any]]:
        if not places:
            return []
        keywords: list[str] = []
        if "nature" in preferred_scenes:
            keywords.extend(["park", "garden", "lake", "trail", "viewpoint", "scenic", "nature"])
        if "museum" in preferred_scenes:
            keywords.extend(["museum", "gallery", "art"])
        if "beach" in preferred_scenes:
            keywords.extend(["beach", "pier", "coast"])
        if "museum" in text or "博物馆" in text:
            keywords.extend(["museum", "gallery", "art"])
        if "food" in text or "美食" in text or "餐厅" in text:
            keywords.extend(["restaurant", "cafe", "food"])
        if "beach" in text or "海边" in text:
            keywords.extend(["beach", "pier", "coast"])
        if "hike" in text or "徒步" in text or "爬山" in text:
            keywords.extend(["hiking", "trail", "park"])
        if rainy:
            keywords.extend(["museum", "gallery", "indoor"])

        def score(place: dict[str, Any]) -> int:
            blob = f"{place.get('name','')} {place.get('category','')} {place.get('address','')}".lower()
            return sum(2 for k in keywords if k and k in blob)

        ranked = sorted(places, key=score, reverse=True)
        selected = ranked[:5]
        return [
            {
                "name": p.get("name", "Stop"),
                "lat": p.get("lat"),
                "lon": p.get("lon"),
                "address": p.get("address", ""),
                "category": p.get("category", "place"),
                "details": self._build_place_details(p, language),
                "visit_focus": self._build_visit_focus(text=text, place=p, language=language),
                "travel_tip": self._build_travel_tip(rainy=rainy, place=p, language=language),
            }
            for p in selected
        ]

    def _build_place_details(self, place: dict[str, Any], language: str) -> str:
        category = str(place.get("category", "place"))
        address = str(place.get("address", "")).strip()
        source_type = str(place.get("source_type", "live_map"))
        if source_type == "knowledge":
            if language == "zh":
                return f"该地点来自本地 travel guide 缓存。类别：{category}。地址：{address}。" if address else f"该地点来自本地 travel guide 缓存。类别：{category}。"
            return f"This place comes from the local travel guide cache. Category: {category}. Address: {address}." if address else f"This place comes from the local travel guide cache. Category: {category}."
        if language == "zh":
            if address:
                return f"该地点来自实时地图检索。类别：{category}。区域：{address}。"
            return f"该地点来自实时地图检索。类别：{category}。"
        if address:
            return f"Real POI from live map search. Category: {category}. Area: {address}."
        return f"Real POI from live map search. Category: {category}."

    def _build_visit_focus(self, text: str, place: dict[str, Any], language: str) -> str:
        category = str(place.get("category", "place")).lower()
        if "museum" in text or "博物馆" in text or "museum" in category:
            return "建议安排 60-90 分钟的重点参观。" if language == "zh" else "Book a focused 60-90 minute museum/gallery visit."
        if "food" in text or "美食" in text or "restaurant" in category or "cafe" in category:
            return "这一站适合作为用餐或咖啡休息点，下一站尽量安排在附近。" if language == "zh" else "Use this stop as a meal or coffee break and keep the next leg nearby."
        if "hike" in text or "徒步" in text or "trail" in category or "park" in category:
            return "建议穿舒适鞋履，并为步行和爬升预留额外时间。" if language == "zh" else "Wear comfortable shoes and budget extra time for elevation or walking."
        if "view" in category or "landmark" in category:
            return "适合安排拍照和周边短距离步行。" if language == "zh" else "Aim for photo time plus a short surrounding-area walk."
        return "控制停留时长，避免压缩后续交通时间。" if language == "zh" else "Keep this stop compact and avoid overextending beyond the next transfer window."

    def _build_travel_tip(self, rainy: bool, place: dict[str, Any], language: str) -> str:
        category = str(place.get("category", "place")).lower()
        if rainy and any(k in category for k in ["park", "viewpoint", "landmark"]):
            return "如遇天气变化，建议缩短户外停留并切换到附近室内点位。" if language == "zh" else "Weather fallback recommended: shorten the outdoor portion and move indoors nearby."
        if "restaurant" in category or "cafe" in category:
            return "尽量避开用餐高峰到达，以减少排队。" if language == "zh" else "Try to arrive before the peak meal rush to reduce queue time."
        if "museum" in category:
            return "若场馆需要预约或分时入场，建议提前确认。" if language == "zh" else "Check entry rules or reserve tickets if the venue requires timed admission."
        return "下一站前建议预留 20-30 分钟交通缓冲。" if language == "zh" else "Leave a 20-30 minute transfer buffer before the next stop."

    def _extract_required_terms(self, user_message: str) -> list[str]:
        terms: list[str] = []
        patterns = [
            r"(?:包含|包括|必须包含|要求包含|要求行程包含)\s*([\u4e00-\u9fffA-Za-z0-9·\-\s、，,]{2,40})",
            r"(?:must include|include)\s+([A-Za-z0-9\-\s,]{2,40})",
        ]
        for pattern in patterns:
            m = re.search(pattern, user_message, flags=re.IGNORECASE)
            if not m:
                continue
            raw = m.group(1)
            for part in re.split(r"[、，,和及/]", raw):
                token = part.strip().strip("。！？,.!?:：;；\"'“”‘’()（）[]")
                if 1 < len(token) <= 12 and token not in {"行程", "景点"}:
                    terms.append(token)
        if "西湖" in user_message and "西湖" not in terms:
            terms.append("西湖")
        return list(dict.fromkeys(terms))

    def _ensure_required_terms(
        self,
        selected_places: list[dict[str, Any]],
        all_places: list[dict[str, Any]],
        required_terms: list[str],
        city: str,
        language: str,
    ) -> list[dict[str, Any]]:
        if not required_terms:
            return selected_places

        def has_term(place: dict[str, Any], term: str) -> bool:
            name = place.get("name", "")
            address = place.get("address", "")
            blob = f"{name} {address}".lower()
            t = term.lower()
            if t == "西湖":
                # Do not treat administrative area text like "西湖区/西湖街道" as fulfilling the must-include POI.
                if "西湖" in name:
                    return True
                return any(k in blob for k in ["west lake", "西湖风景", "西湖景区", "西湖湖"])
            return t in blob

        result = list(selected_places)
        for term in required_terms:
            if any(has_term(p, term) for p in result):
                continue
            candidate = next((p for p in all_places if has_term(p, term)), None)
            if candidate is not None:
                result.insert(1, candidate)
            else:
                result.insert(1, {"name": f"{city}{term}", "category": "must_include"})

        dedup: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in result:
            key = p.get("name", "").lower()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(p)
        return dedup[:5]

    def _fill_to_target_slots(
        self,
        selected_places: list[dict[str, Any]],
        all_places: list[dict[str, Any]],
        target_slots: int,
        city: str,
        preferred_scenes: set[str],
        language: str,
    ) -> list[dict[str, Any]]:
        result = list(selected_places)
        seen = {str(p.get("name", "")).lower() for p in result}
        for place in all_places:
            key = str(place.get("name", "")).lower()
            if not key or key in seen:
                continue
            result.append(
                {
                    "name": place.get("name", "Stop"),
                    "lat": place.get("lat"),
                    "lon": place.get("lon"),
                    "address": place.get("address", ""),
                    "category": place.get("category", "place"),
                    "details": self._build_place_details(place, language),
                    "visit_focus": self._build_visit_focus(text="", place=place, language=language),
                    "travel_tip": self._build_travel_tip(rainy=False, place=place, language=language),
                }
            )
            seen.add(key)
            if len(result) >= target_slots:
                return result[:target_slots]

        fallback = (
            self._fallback_nature_slots(city, language)
            if "nature" in preferred_scenes
            else self._fallback_city_slots(city, language)
        )
        for item in fallback:
            key = str(item.get("name", "")).lower()
            if key in seen:
                continue
            result.append(item)
            seen.add(key)
            if len(result) >= target_slots:
                break
        return result[:target_slots]

    def _merge_places(
        self,
        primary_places: list[dict[str, Any]],
        secondary_places: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        merged: list[dict[str, Any]] = []
        seen: set[str] = set()
        for place in [*primary_places, *secondary_places]:
            key = str(place.get("name", "")).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(place)
        return merged

    def _supplement_real_places(
        self,
        city: str,
        existing_places: list[dict[str, Any]],
        preferred_scenes: set[str],
        retrieved_context: list[str],
    ) -> list[dict[str, Any]]:
        if len(existing_places) >= 5:
            return existing_places

        knowledge_places = self._load_places_from_knowledge(
            city=city,
            preferred_scenes=preferred_scenes,
            contexts=retrieved_context,
            strict_preference=True,
        )
        fallback_knowledge_places = self._load_places_from_knowledge(
            city=city,
            preferred_scenes=preferred_scenes,
            contexts=retrieved_context,
            strict_preference=False,
        )
        seen = {str(p.get("name", "")).lower() for p in existing_places}
        supplemented = list(existing_places)
        for place in [*knowledge_places, *fallback_knowledge_places]:
            if len(supplemented) >= 8:
                break
            name_key = str(place.get("name", "")).lower()
            if not name_key or name_key in seen:
                continue
            seen.add(name_key)
            supplemented.append(place)
        return supplemented

    def _load_places_from_knowledge(
        self,
        city: str,
        preferred_scenes: set[str],
        contexts: list[str],
        strict_preference: bool = True,
    ) -> list[dict[str, Any]]:
        file_path = self.knowledge_manager.find_city_knowledge_file(city)
        if file_path is None:
            return []
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception:
            return []

        places: list[dict[str, Any]] = []
        in_poi_section = False
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("## "):
                in_poi_section = stripped.lower() == "## real poi candidates"
                continue
            if not in_poi_section or not stripped.startswith("- "):
                continue
            parts = [part.strip() for part in stripped[2:].split("|")]
            if len(parts) < 3:
                continue
            name, category, address = parts[0], parts[1], parts[2]
            places.append({"name": name, "category": category, "address": address, "source_type": "knowledge"})

        ranked = self._rank_knowledge_places(
            places=places,
            preferred_scenes=preferred_scenes,
            contexts=contexts,
            strict_preference=strict_preference,
        )
        return ranked[:6]

    def _rank_knowledge_places(
        self,
        places: list[dict[str, Any]],
        preferred_scenes: set[str],
        contexts: list[str],
        strict_preference: bool = True,
    ) -> list[dict[str, Any]]:
        context_blob = " ".join(contexts).lower()

        def score(place: dict[str, Any]) -> int:
            blob = f"{place.get('name', '')} {place.get('category', '')} {place.get('address', '')}".lower()
            value = 0
            if "nature" in preferred_scenes:
                value += sum(3 for token in ["lake", "west lake", "garden", "park", "scenic", "view"] if token in blob)
                value += sum(2 for token in ["西湖", "公园", "植物园", "景区"] if token in blob)
                if "museum" in blob or "博物馆" in blob:
                    value -= 4
            if "museum" in preferred_scenes:
                value += sum(2 for token in ["museum", "博物馆", "art"] if token in blob)
            value += sum(2 for token in ["west lake", "西湖", "tea museum", "茶叶博物馆"] if token in blob and token in context_blob)
            return value

        ranked = sorted(places, key=score, reverse=True)
        if strict_preference and "nature" in preferred_scenes:
            ranked = [p for p in ranked if score(p) > 0]
        return ranked

    def _extract_context_place_terms(self, city: str, contexts: list[str]) -> list[str]:
        terms: list[str] = []
        city_lower = city.lower()
        replacements = {
            "west lake": "West Lake",
            "西湖": "西湖",
            "tea museums": "tea museum",
            "tea museum": "tea museum",
            "griffith observatory": "Griffith Observatory",
            "the getty center": "The Getty Center",
            "santa monica": "Santa Monica",
        }
        for chunk in contexts:
            text = chunk.strip()
            lower = text.lower()
            for needle, replacement in replacements.items():
                if needle in lower and replacement not in terms:
                    terms.append(replacement)

            for match in re.findall(r"(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})", text):
                cleaned = match.strip()
                cleaned_lower = cleaned.lower()
                if cleaned_lower in {city_lower, "travel planning rules", "one day travel notes"}:
                    continue
                if len(cleaned.split()) >= 1 and cleaned not in terms:
                    terms.append(cleaned)

            for match in re.findall(r"([\u4e00-\u9fff]{2,12}(?:湖|园|馆|寺|街|景区|公园))", text):
                if match not in terms:
                    terms.append(match)

        ordered: list[str] = []
        seen: set[str] = set()
        for term in terms:
            key = term.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(term)
        return ordered[:8]

    def _fallback_nature_slots(self, city: str, language: str) -> list[dict[str, Any]]:
        return [
            {"name": f"{city}湖边早餐" if language == "zh" else f"{city} lakeside breakfast", "category": "food"},
            {"name": f"{city}代表性公园漫步" if language == "zh" else f"{city} signature park walk", "category": "park"},
            {"name": f"{city}植物园" if language == "zh" else f"{city} botanical garden", "category": "park"},
            {"name": f"{city}风景茶歇/咖啡" if language == "zh" else f"{city} scenic tea / coffee break", "category": "food"},
            {"name": f"{city}日落观景点" if language == "zh" else f"{city} sunset viewpoint", "category": "viewpoint"},
        ]

    def _fallback_city_slots(self, city: str, language: str) -> list[dict[str, Any]]:
        return [
            {"name": f"{city}本地早餐" if language == "zh" else f"{city} local breakfast", "category": "food"},
            {"name": f"{city}核心地标" if language == "zh" else f"{city} major landmark", "category": "landmark"},
            {"name": f"{city}文化站点" if language == "zh" else f"{city} cultural stop", "category": "culture"},
            {"name": f"{city}下午休息" if language == "zh" else f"{city} afternoon break", "category": "food"},
            {"name": f"{city}傍晚观景点" if language == "zh" else f"{city} evening scenic stop", "category": "viewpoint"},
        ]

    def _enforce_required_in_schedule(
        self,
        schedule: list[dict[str, Any]],
        required_terms: list[str],
        all_places: list[dict[str, Any]],
        city: str,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        if not schedule or not required_terms:
            return schedule

        def hit(item: dict[str, Any], term: str) -> bool:
            blob = f"{item.get('title','')} {item.get('address','')}".lower()
            return term.lower() in blob

        out = list(schedule)
        for term in required_terms:
            if any(hit(item, term) for item in out):
                continue
            candidate = next(
                (p for p in all_places if term.lower() in f"{p.get('name','')} {p.get('address','')}".lower()),
                None,
            )
            replacement = {
                "title": candidate.get("name", f"{city}{term}") if candidate else f"{city}{term}",
                "address": candidate.get("address", "") if candidate else "",
                "lat": candidate.get("lat") if candidate else None,
                "lon": candidate.get("lon") if candidate else None,
                "category": candidate.get("category", "must_include") if candidate else "must_include",
                "details": self._build_place_details(candidate or {"category": "must_include", "address": ""}, language),
                "visit_focus": "这是用户明确要求包含的点位，建议固定保留在行程中。" if language == "zh" else "Required stop requested by the user; keep this visit fixed in the route.",
                "travel_tip": "调整顺序时优先围绕这个点位安排前后交通。" if language == "zh" else "Protect this stop when reordering the day and route surrounding legs around it.",
            }
            # Replace the last item to preserve 5-slot schedule.
            out[-1] = {**out[-1], **replacement}
        return out
