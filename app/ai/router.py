"""AI Analyst FastAPI routes — SSE streaming chat + status."""
from __future__ import annotations
import json
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.ai import orchestrator, page_context, transcribe, warehouse

router = APIRouter()


class ChatReq(BaseModel):
    query: str
    history: list = []
    # Same optional page context as POST /chat/sessions/{id}/messages/stream
    # (app/api/chat.py), reduced by the same page_context.normalise, so both ways into
    # orchestrator.answer accept the same hint and drop the same things. Never a 422.
    context: Any = None


@router.get("/ai/status")
def ai_status():
    # voice: whether POST /ai/transcribe is set up, so the chat shows its mic only then.
    return {"configured": orchestrator.has_key(), "model": orchestrator.AZURE_DEPLOYMENT,
            "voice": transcribe.configured()}


@router.get("/ai/mentions")
def ai_mentions(q: str = Query(""), type: str = Query(None), limit: int = Query(18)):
    """Entity picker source. type= filters to one kind (browse when q empty); no type + q = cross-type search."""
    try:
        return {"items": warehouse.mentions(q, type, limit)}
    except Exception:
        return {"items": []}


@router.post("/ai/chat")
def ai_chat(req: ChatReq):
    # Passed only when something valid is left, so an orchestrator.answer(query, history)
    # without the keyword (and every test fake written that way) keeps working.
    ctx = page_context.normalise(req.context)
    extra = {"context": ctx} if ctx else {}

    def gen():
        try:
            for ev in orchestrator.answer(req.query, req.history, **extra):
                yield f"data: {json.dumps(ev)}\n\n"
        except Exception as e:  # never break the stream ungracefully
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"
        yield "data: {\"type\": \"end\"}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                                      "X-Accel-Buffering": "no"})


@router.post("/ai/transcribe")
async def ai_transcribe(request: Request, lang: str = Query("en")):
    """A spoken question -> {"text": ...} (app/ai/transcribe.py). The body is the recording
    itself, sent with its own media type (audio/webm, audio/ogg, audio/mp4 ...), not a form:
    one clip needs no multipart parser. Checked before a byte is read where it can be (set up,
    format, declared size), and the read stops at the size limit, so a large upload is refused
    without being held. The Azure call is blocking, so it runs in the thread pool."""
    ctype = request.headers.get("content-type")
    try:
        if not transcribe.configured():
            raise transcribe.VoiceError(503, "Voice input is not set up on this server yet.")
        transcribe.media_type(ctype)
        if lang not in transcribe.LANGUAGES:
            raise transcribe.VoiceError(400, "Unknown language. Use en, hi or auto.")
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > transcribe.MAX_AUDIO_BYTES:
            transcribe.check_size(int(declared))
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > transcribe.MAX_AUDIO_BYTES:
                transcribe.check_size(size)
            chunks.append(chunk)
        text = await run_in_threadpool(transcribe.transcribe, b"".join(chunks), ctype, lang)
    except transcribe.VoiceError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None
    return {"text": text}
