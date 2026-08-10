from __future__ import annotations

import asyncio
import os
import time
from dotenv import load_dotenv

from livekit.api import LiveKitAPI, AccessToken, VideoGrants, CreateAgentDispatchRequest
from livekit import rtc

load_dotenv()

LIVEKIT_URL = os.getenv("LIVEKIT_URL", "wss://test-6ywmxecc.livekit.cloud")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")

API_URL = LIVEKIT_URL.replace("wss://", "https://").replace("ws://", "http://")


async def main():
    print("=" * 60)
    print("TESTING LIVEKIT AGENT DISPATCH & AUDIO INTERACTION")
    print("=" * 60)

    room_name = f"test-room-{int(time.time())}"
    print(f"\n[1] Creating LiveKit API client for room: {room_name}")
    api = LiveKitAPI(url=API_URL, api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)

    # 1. Dispatch fieldmate agent to room explicitly via agent_dispatch API
    print("\n[2] Dispatching 'fieldmate' agent worker to room...")
    try:
        dispatch = await api.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name="fieldmate",
                room=room_name,
            )
        )
        print(f"    Dispatch created successfully! ID: {dispatch.id}")
    except Exception as e:
        print(f"    Dispatch failed: {e}")
        await api.aclose()
        return

    # 2. Join room as a user participant
    print("\n[3] User participant joining room...")
    token = (
        AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        .with_identity("test_technician")
        .with_name("Technician Test")
        .with_grants(VideoGrants(room_join=True, room=room_name))
        .to_jwt()
    )

    room = rtc.Room()

    agent_greeting_received = asyncio.Event()

    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        print(f"    User subscribed to track from: {participant.identity} (kind={track.kind})")
        if participant.identity != "test_technician":
            agent_greeting_received.set()

    await room.connect(LIVEKIT_URL, token)
    print(f"    User connected to room '{room_name}' as identity '{room.local_participant.identity}'")

    print("\n[4] Waiting for FieldMate agent to join and publish audio greeting...")
    try:
        await asyncio.wait_for(agent_greeting_received.wait(), timeout=20.0)
        print("    SUCCESS: FieldMate agent connected to room and published its audio track!")
    except asyncio.TimeoutError:
        print("    TIMED OUT waiting for FieldMate agent to join.")

    print("\n[5] Cleaning up...")
    await room.disconnect()
    await api.aclose()
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
