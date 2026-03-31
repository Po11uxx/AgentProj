from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import MEMORY_PATH, settings


class ConversationMemory:
    def __init__(self, window_size: int = settings.short_memory_window) -> None:
        self.window_size = window_size
        self._store: dict[str, list[dict[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        messages = self._store.setdefault(session_id, [])
        messages.append({"role": role, "content": content})
        if len(messages) > self.window_size:
            self._store[session_id] = messages[-self.window_size :]

    def history(self, session_id: str) -> list[dict[str, str]]:
        return self._store.get(session_id, [])


class UserPreferenceMemory:
    def __init__(self, file_path: Path = MEMORY_PATH) -> None:
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("{}", encoding="utf-8")

    def _read(self) -> dict[str, Any]:
        return json.loads(self.file_path.read_text(encoding="utf-8"))

    def _write(self, data: dict[str, Any]) -> None:
        self.file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, user_id: str) -> dict[str, Any]:
        return self._read().get(user_id, {})

    def update(self, user_id: str, values: dict[str, Any]) -> dict[str, Any]:
        data = self._read()
        prefs = data.get(user_id, {})
        prefs.update(values)
        data[user_id] = prefs
        self._write(data)
        return prefs


conversation_memory = ConversationMemory()
user_pref_memory = UserPreferenceMemory()
