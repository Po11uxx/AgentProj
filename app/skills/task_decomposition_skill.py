from __future__ import annotations

from app.llm import llm_client


class TaskDecompositionSkill:
    name = "task_decomposition"

    def run(self, user_message: str) -> list[str]:
        prompt = (
            "Decompose the task into 4-6 actionable steps as JSON. "
            "Use key 'steps'. Task: "
            f"{user_message}"
        )
        result = llm_client.generate_json(prompt, fallback={"steps": []})
        steps = result.get("steps", [])
        if not steps:
            return [
                "Understand task goal and constraints",
                "Collect weather and place info",
                "Generate feasible timeline",
                "Summarize and provide tips",
            ]
        return steps
