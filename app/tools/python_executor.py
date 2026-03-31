from __future__ import annotations

import ast
from typing import Any


class PythonExecutorTool:
    name = "python_executor"
    description = "Run safe arithmetic/data expressions"

    _SAFE_BUILTINS = {
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "sorted": sorted,
        "round": round,
    }

    def run(self, expression: str) -> dict[str, Any]:
        try:
            tree = ast.parse(expression, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                    if isinstance(node, ast.Call):
                        continue
                    raise ValueError("Unsafe expression")
            value = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, self._SAFE_BUILTINS)
            return {"expression": expression, "result": value}
        except Exception as exc:
            return {"expression": expression, "error": str(exc)}
