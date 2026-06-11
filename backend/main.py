"""
Darija ASR Server
=================
Run:  python main.py

Socket events
-------------
→ server   audio_chunk      binary WebM blob
→ server   request_answer   (empty) — flush transcript and ask Gemini
→ server   reset            (empty) — clear session

← client   ready            { sid }
← client   transcription    { partial, full }
← client   llm_start        { question }
← client   llm_token        { token }  (streamed)
← client   llm_done         { answer, question }
← client   error            { message }
"""
import logging
import os
from contextlib import asynccontextmanager

import socketio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.asr_service import ASRService
from services.audio_processor import is_mostly_silence, webm_to_float32_array
from services.llm_service import LLMService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("darija-asr")

MODEL_DIR = os.getenv("MODEL_DIR", "Qwen/Qwen3-ASR-0.6B")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
DEVICE = os.getenv("DEVICE") or None

# ── Singletons ─────────────────────────────────────────────
asr = ASRService(model_dir=MODEL_DIR, device=DEVICE)
llm = LLMService(model_name=GEMINI_MODEL)

# Per-connection state
sessions: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr.load()
    llm.load()
    yield
    asr.shutdown()


# ── FastAPI ────────────────────────────────────────────────
fastapi_app = FastAPI(title="Darija ASR Server", lifespan=lifespan)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/health")
async def health():
    return {
        "status": "ok",
        "asr_device": asr.device,
        "model_dir": MODEL_DIR,
        "llm_configured": llm._client is not None,
        "active_sessions": len(sessions),
    }


# ── Socket.io ─────────────────────────────────────────────
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    max_http_buffer_size=10_000_000,
)

app = socketio.ASGIApp(
    sio, other_asgi_app=fastapi_app, socketio_path="/socket.io"
)


@sio.event
async def connect(sid, environ, auth):
    logger.info("Client connected: %s", sid)
    sessions[sid] = {"transcript_parts": [], "history": []}
    await sio.emit("ready", {"sid": sid}, to=sid)


@sio.event
async def disconnect(sid):
    logger.info("Client disconnected: %s", sid)
    sessions.pop(sid, None)


@sio.on("audio_chunk")
@sio.on("audio_chunk")
async def on_audio_chunk(sid, data):
    if sid not in sessions:
        return
    try:
        audio_bytes = data["audio"] if isinstance(data, dict) else data
        if not audio_bytes:
            return
        if not isinstance(audio_bytes, (bytes, bytearray)):
            audio_bytes = bytes(audio_bytes)

        logger.info("Received chunk: %d bytes", len(audio_bytes))

        if len(audio_bytes) < 200:
            return

        pcm = webm_to_float32_array(audio_bytes, target_sr=16000)
        if pcm.size < 8000:
            return
        if is_mostly_silence(pcm):
            return

        text = await asr.transcribe(pcm, sampling_rate=16000)
        if not text:
            return

        sessions[sid]["transcript_parts"].append(text)
        full = " ".join(sessions[sid]["transcript_parts"]).strip()
        await sio.emit("transcription", {"partial": text, "full": full}, to=sid)
    except Exception as exc:
        logger.warning("audio_chunk skipped: %s", exc)

@sio.on("request_answer")
async def on_request_answer(sid, data=None):
    if sid not in sessions:
        return

    full = " ".join(sessions[sid]["transcript_parts"]).strip()
    if not full:
        await sio.emit("llm_done", {"answer": "", "question": ""}, to=sid)
        return

    history = sessions[sid]["history"]
    await sio.emit("llm_start", {"question": full}, to=sid)

    accumulated: list[str] = []
    try:
        async for token in llm.stream(full, history=history):
            accumulated.append(token)
            await sio.emit("llm_token", {"token": token}, to=sid)

        answer = "".join(accumulated)
        sessions[sid]["history"].append({"role": "user", "parts": [full]})
        sessions[sid]["history"].append({"role": "model", "parts": [answer]})
        await sio.emit("llm_done", {"answer": answer, "question": full}, to=sid)
    except Exception as exc:
        logger.exception("LLM streaming failed")
        await sio.emit("error", {"message": str(exc)}, to=sid)
    finally:
        sessions[sid]["transcript_parts"] = []


@sio.on("reset")
async def on_reset(sid, data=None):
    if sid in sessions:
        sessions[sid]["transcript_parts"] = []
        sessions[sid]["history"] = []
    await sio.emit("reset_ok", {}, to=sid)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
