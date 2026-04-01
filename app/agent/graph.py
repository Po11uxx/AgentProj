from __future__ import annotations

import re
from datetime import date
from typing import Any, TypedDict

from app.config import settings
from app.knowledge import KnowledgeBaseManager
from app.memory import conversation_memory, user_pref_memory
from app.rag import build_vector_store
from app.skills.summary_skill import SummarySkill
from app.skills.task_decomposition_skill import TaskDecompositionSkill
from app.skills.travel_planning_skill import TravelPlanningSkill
from app.tools import get_toolkit

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None


class AgentState(TypedDict, total=False):
    session_id: str
    user_id: str
    message: str
    city: str
    date_str: str
    plan: list[str]
    retrieved_docs: list[str]
    retrieved_sources: list[str]
    tool_outputs: dict[str, Any]
    used_tools: list[str]
    knowledge_status: dict[str, Any]
    final_answer: str


class ProductivityAgent:
    def __init__(self) -> None:
        self.toolkit = get_toolkit()
        self.decomposer = TaskDecompositionSkill()
        self.travel_skill = TravelPlanningSkill()
        self.summary_skill = SummarySkill()
        self.knowledge_manager = KnowledgeBaseManager()
        self.store = build_vector_store()
        self.app = self._build_graph()

    def _build_graph(self):
        if StateGraph is None:
            return None

        graph = StateGraph(AgentState)
        graph.add_node("node_plan", self._plan_node)
        graph.add_node("node_retrieve", self._retrieve_node)
        graph.add_node("node_execute", self._execute_node)
        graph.add_node("node_summarize", self._summarize_node)
        graph.add_node("node_remember", self._remember_node)

        graph.add_edge(START, "node_plan")
        graph.add_edge("node_plan", "node_retrieve")
        graph.add_edge("node_retrieve", "node_execute")
        graph.add_edge("node_execute", "node_summarize")
        graph.add_edge("node_summarize", "node_remember")
        graph.add_edge("node_remember", END)
        return graph.compile()

    def run(self, session_id: str, user_id: str, message: str) -> dict[str, Any]:
        state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            "city": settings.default_city,
            "date_str": str(date.today()),
            "used_tools": [],
        }

        conversation_memory.append(session_id, "user", message)

        if self.app is not None:
            output = self.app.invoke(state)
        else:
            output = self._run_fallback(state)

        conversation_memory.append(session_id, "assistant", output["final_answer"])

        return {
            "answer": output["final_answer"],
            "plan": output.get("plan", []),
            "used_tools": output.get("used_tools", []),
            "retrieved_docs": output.get("retrieved_sources", []),
            "debug": {
                "tool_outputs": output.get("tool_outputs", {}),
                "short_memory": conversation_memory.history(session_id),
                "user_preferences": user_pref_memory.get(user_id),
            },
        }

    def _run_fallback(self, state: AgentState) -> AgentState:
        s = self._plan_node(state)
        s = self._retrieve_node(s)
        s = self._execute_node(s)
        s = self._summarize_node(s)
        s = self._remember_node(s)
        return s

    def _plan_node(self, state: AgentState) -> AgentState:
        plan = self.decomposer.run(state["message"])
        state["plan"] = plan
        return state

    def _retrieve_node(self, state: AgentState) -> AgentState:
        resolved_city = self.travel_skill._resolve_city(state["message"], state["city"])
        state["city"] = resolved_city
        knowledge_status = self.knowledge_manager.ensure_city_knowledge(
            city=resolved_city,
            search_tool=self.toolkit["search"],
            map_tool=self.toolkit["map"],
        )
        state["knowledge_status"] = knowledge_status
        self.store = build_vector_store(force=knowledge_status.get("created", False))
        top_k = max(1, settings.rag_top_k)
        candidate_k = max(top_k, settings.rag_candidate_k)
        docs = self.store.retrieve(
            f"{resolved_city} {state['message']}",
            top_k=top_k,
            candidate_k=candidate_k,
            city=resolved_city,
        )
        state["retrieved_docs"] = [d.text for d in docs]
        state["retrieved_sources"] = [d.source for d in docs]
        return state

    def _execute_node(self, state: AgentState) -> AgentState:
        saved_prefs = user_pref_memory.get(state["user_id"])
        current_prefs = self._extract_preferences(state["message"])
        prefs = {**saved_prefs, **current_prefs}
        payload = self.travel_skill.run(
            user_message=state["message"],
            toolkit=self.toolkit,
            city=state["city"],
            date_str=state["date_str"],
            retrieved_context=state.get("retrieved_docs", []),
            user_preferences=prefs,
        )
        knowledge_update = self.knowledge_manager.update_city_knowledge_from_places(
            city=state["city"],
            places=payload.get("map_data", {}).get("places", []),
        )
        if knowledge_update.get("updated"):
            self.store = build_vector_store(force=True)
        payload["knowledge_update"] = knowledge_update
        payload["knowledge_status"] = state.get("knowledge_status", {})
        state["tool_outputs"] = payload
        state["used_tools"] = ["weather", "search", "map", "calendar", "transport"]
        return state

    def _summarize_node(self, state: AgentState) -> AgentState:
        summary = self.summary_skill.run(state.get("tool_outputs", {}))
        state["final_answer"] = summary
        return state

    def _remember_node(self, state: AgentState) -> AgentState:
        updates = self._extract_preferences(state["message"])
        if updates:
            user_pref_memory.update(state["user_id"], updates)

        return state

    def _extract_preferences(self, message: str) -> dict[str, Any]:
        updates: dict[str, Any] = {}
        lower = message.lower()
        budget_match = re.search(r"(?:budget|预算)\D{0,6}(\d{2,6})", lower)
        if budget_match:
            updates["budget"] = int(budget_match.group(1))
        if "beach" in lower or "海边" in message:
            updates["preferred_scene"] = "beach"
        if re.search(r"natural scenery|nature|scenic|outdoor", lower) or re.search(
            r"自然风景|自然景观|风景|山水|户外",
            message,
        ):
            updates["preferred_scene"] = "natural scenery"
        if "museum" in lower or "博物馆" in message:
            updates["preferred_scene"] = "museum"
        return updates
