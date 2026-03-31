from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session id for short-term memory")
    user_id: str = Field(..., description="User id for long-term preference memory")
    message: str = Field(..., description="User prompt")


class ChatResponse(BaseModel):
    session_id: str
    user_id: str
    answer: str
    plan: list[str]
    used_tools: list[str]
    retrieved_docs: list[str]
    debug: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_name: str
    payload: dict[str, Any]
