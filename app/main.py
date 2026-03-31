from __future__ import annotations

from fastapi import FastAPI

from app.agent.graph import ProductivityAgent
from app.config import settings
from app.schemas import ChatRequest, ChatResponse

app = FastAPI(title=settings.app_name)
agent = ProductivityAgent()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": settings.app_name}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    result = agent.run(session_id=req.session_id, user_id=req.user_id, message=req.message)
    return ChatResponse(
        session_id=req.session_id,
        user_id=req.user_id,
        answer=result["answer"],
        plan=result["plan"],
        used_tools=result["used_tools"],
        retrieved_docs=result["retrieved_docs"],
        debug=result["debug"],
    )
