from __future__ import annotations

import asyncio
import os
import time
from dotenv import load_dotenv

from fieldmate.brain.runtime import build_brain_runtime

load_dotenv()


async def run_live_smoke_test():
    print("=" * 60)
    print("FIELDMATE — LIVE END-TO-END DIAGNOSTIC SMOKE TEST")
    print("=" * 60)

    # 1. Initialize BrainRuntime
    session_id = f"live_test_{int(time.time())}"
    print(f"\n[1] Constructing BrainRuntime for session: {session_id}")
    runtime = build_brain_runtime(session_id=session_id)

    # 2. Ensure Qdrant collection and Groq connection ready
    print("\n[2] Bootstrapping Qdrant collection & warming Groq API...")
    started = time.perf_counter()
    await runtime.ensure_ready()
    print(f"    Ready in {(time.perf_counter() - started) * 1000:.1f} ms")

    brain = runtime.brain

    # 3. Turn 1: Initial Observation (Technical)
    utterance_1 = "My Lenovo ThinkPad X1 Carbon keeps freezing and getting BSOD with WHEA_UNCORRECTABLE_ERROR"
    print(f"\n[3] TURN 1 — User: '{utterance_1}'")
    start_t1 = time.perf_counter()
    res1 = await brain.process(utterance_1, technical=True)
    elapsed_t1 = (time.perf_counter() - start_t1) * 1000.0

    print(f"    Turn 1 Response: {res1.response}")
    print(f"    Turn: {res1.turn}, Generation: {res1.generation}")
    print(f"    Retrieved: {res1.retrieved}, Retrieval Latency: {res1.retrieval_latency_ms:.1f} ms")
    print(f"    Reasoning Latency: {res1.reasoning_latency_ms:.1f} ms")
    print(f"    Total Latency: {elapsed_t1:.1f} ms")

    state1 = brain.state.session.diagnostic
    print(f"    Equipment: OEM={state1.equipment.manufacturer}, Model={state1.equipment.model}")
    print(f"    Fault Codes: {state1.fault_codes}")
    print(f"    Hypotheses Count: {len(state1.hypotheses)}")
    if state1.hypotheses:
        print(f"    Top Hypothesis: {state1.hypotheses[0].description} (confidence={state1.hypotheses[0].confidence})")
    print(f"    Next Action: {state1.next_recommended_action}")

    # 4. Turn 2: Non-technical Chit-Chat
    utterance_2 = "Thanks, can you explain what mdsched does?"
    print(f"\n[4] TURN 2 (Chat/Non-Technical) — User: '{utterance_2}'")
    start_t2 = time.perf_counter()
    res2 = await brain.process(utterance_2, technical=False)
    elapsed_t2 = (time.perf_counter() - start_t2) * 1000.0

    print(f"    Turn 2 Response: {res2.response}")
    print(f"    Retrieved: {res2.retrieved} (Fast Path Bypass)")
    print(f"    Total Latency: {elapsed_t2:.1f} ms")

    # 5. Turn 3: Diagnostic Test Completion & Resolution
    utterance_3 = "I ran mdsched and replaced the bad RAM module, now the machine boots fine and passes stress tests"
    print(f"\n[5] TURN 3 — User: '{utterance_3}'")
    start_t3 = time.perf_counter()
    res3 = await brain.process(utterance_3, technical=True)
    elapsed_t3 = (time.perf_counter() - start_t3) * 1000.0

    print(f"    Turn 3 Response: {res3.response}")
    print(f"    Total Latency: {elapsed_t3:.1f} ms")

    state3 = brain.state.session.diagnostic
    print(f"    Confirmed Resolution: {state3.confirmed_resolution}")
    print(f"    Case Status: {state3.case_status}")

    # 6. Cleanup
    print("\n[6] Closing Brain Runtime...")
    await runtime.close()
    print("\n" + "=" * 60)
    print("LIVE DIAGNOSTIC SMOKE TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_live_smoke_test())
