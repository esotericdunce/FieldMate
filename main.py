import subprocess
import sys
import os
import uvicorn
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from livekit import api

load_dotenv()

app = FastAPI()

# Generate LiveKit Token & Dispatch Agent
@app.get("/api/token")
async def get_token(room: str = "fieldmate-room", username: str = "user"):
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

    # Explicitly dispatch fieldmate agent worker to this room
    lkapi = api.LiveKitAPI(url, api_key, api_secret)
    try:
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name="fieldmate",
                room=room,
            )
        )
    except Exception as e:
        print(f"Agent dispatch warning/error: {e}")
    finally:
        await lkapi.aclose()
    
    return {"token": token, "url": url}

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
    agent_proc = subprocess.Popen([sys.executable, "-m", "fieldmate.voice_agent", "dev"], env=env)
    
    try:
        print("Starting FastAPI Web Server on port 8000...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    finally:
        print("Shutting down agent worker...")
        agent_proc.terminate()
        agent_proc.wait()
