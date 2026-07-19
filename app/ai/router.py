"""AI Analyst FastAPI routes — SSE streaming chat + status."""
from __future__ import annotations
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from fastapi import Query

from app.ai import orchestrator, warehouse

router = APIRouter()


class ChatReq(BaseModel):
    query: str
    history: list = []


@router.get("/ai/status")
def ai_status():
    return {"configured": orchestrator.has_key(), "model": orchestrator.AZURE_DEPLOYMENT}


@router.get("/ai/mentions")
def ai_mentions(q: str = Query(""), type: str = Query(None), limit: int = Query(18)):
    """Entity picker source. type= filters to one kind (browse when q empty); no type + q = cross-type search."""
    try:
        return {"items": warehouse.mentions(q, type, limit)}
    except Exception:
        return {"items": []}


@router.post("/ai/chat")
def ai_chat(req: ChatReq):
    def gen():
        try:
            for ev in orchestrator.answer(req.query, req.history):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # never break the stream ungracefully
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})
