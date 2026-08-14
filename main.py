import subprocess
import sys
import os
import time
import uvicorn
import asyncio
from typing import Any
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from livekit import api
from fastapi.middleware.cors import CORSMiddleware

from fieldmate.brain.runtime import build_brain_runtime

load_dotenv()

app = FastAPI(title="FieldMate Diagnostic Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active session Brain registry for multimodal inspections.
# Bounded so a long-running server doesn't accumulate unlimited
# Groq clients / Qdrant connections, one per unique session_id.
active_runtimes: dict[str, Any] = {}
active_runtimes_last_used: dict[str, float] = {}
MAX_ACTIVE_RUNTIMES = int(os.getenv("FIELDMATE_MAX_ACTIVE_RUNTIMES", "50"))


async def _evict_stale_runtimes() -> None:
    while len(active_runtimes) > MAX_ACTIVE_RUNTIMES:
        oldest_key = min(
            active_runtimes_last_used,
            key=lambda key: active_runtimes_last_used[key],
        )
        oldest = active_runtimes.pop(oldest_key)
        active_runtimes_last_used.pop(oldest_key, None)
        try:
            await oldest.close()
        except Exception as e:
            print(f"Runtime eviction close notice: {e}")


# Generate LiveKit Token & Dispatch Agent
@app.get("/api/token")
async def get_token(room: str | None = None, username: str | None = None):
    if not room:
        room = os.getenv("FIELDMATE_ROOM_NAME", "fieldmate_dev_room")
    if not username:
        username = os.getenv("FIELDMATE_USER_ID", "tech_john_doe")

    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    url = os.getenv("LIVEKIT_URL")
    
    if not api_key or not api_secret or not url:
        return JSONResponse(status_code=500, content={"error": "LiveKit environment variables not found."})
        
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(username)
        .with_name(username)
        .with_grants(
            api.VideoGrants(
                room_join=True,
                room=room,
            )
        )
    ).to_jwt()

    # Ensure fieldmate agent worker is dispatched to the room
    try:
        lkapi = api.LiveKitAPI(url, api_key, api_secret)
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="fieldmate",
                room=room,
            )
        )
        await lkapi.aclose()
    except Exception as e:
        print(f"Agent dispatch status: {e}")

    return {"token": token, "url": url, "room": room}


# Multimodal Hardware Inspection Endpoint
@app.post("/api/inspect")
async def inspect_hardware(
    image: UploadFile = File(...),
    session_id: str = Form("fieldmate_dev_room"),
    user_utterance: str | None = Form(None),
):
    try:
        image_bytes = await image.read()
        if not image_bytes:
            return JSONResponse(status_code=400, content={"error": "Empty image provided."})

        if session_id not in active_runtimes:
            runtime = build_brain_runtime(session_id=session_id)
            try:
                await runtime.ensure_ready()
            except Exception as e:
                print(f"Qdrant bootstrap notice: {e}")
            active_runtimes[session_id] = runtime
            await _evict_stale_runtimes()

        active_runtimes_last_used[session_id] = time.monotonic()
        runtime = active_runtimes[session_id]
        result = await runtime.brain.process_image(
            image_bytes=image_bytes,
            user_utterance=user_utterance,
        )

        decision = result.decision
        return {
            "response": result.response,
            "hypothesis": decision.hypothesis if decision else None,
            "confidence": decision.confidence if decision else 0.0,
            "next_action": decision.next_action if decision else None,
            "clarification_needed": decision.clarification_needed if decision else False,
            "clarification_question": decision.clarification_question if decision else None,
            "turn": result.turn,
            "generation": result.generation,
        }
    except Exception as exc:
        print(f"Inspection error: {exc}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# Serve static frontend files
frontend_dist = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
else:
    @app.get("/")
    def no_frontend():
        return {"error": "Frontend build not found. Please run 'npm run build' in the frontend directory."}

if __name__ == "__main__":
    print("Starting FieldMate Voice Agent Worker in the background...")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")
    # "dev" mode is for local development only: it hot-reloads, logs
    # verbosely, and does not gracefully drain in-flight jobs on shutdown.
    # Production should run "start". Override with FIELDMATE_AGENT_MODE=dev
    # for local testing.
    agent_mode = os.getenv("FIELDMATE_AGENT_MODE", "start")
    agent_proc = subprocess.Popen([sys.executable, "-m", "fieldmate.voice_agent", agent_mode], env=env)
    
    try:
        print("Starting FastAPI Web Server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        print("Shutting down agent worker...")
        agent_proc.terminate()
        agent_proc.wait()
