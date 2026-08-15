from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AsyncOpenAI

from fieldmate.brain.brain import Brain
from fieldmate.brain.memory.manager import MemoryManager
from fieldmate.brain.qdrant.config import QdrantConfig
from fieldmate.brain.qdrant.repository import QdrantMemoryRepository
from fieldmate.brain.reasoning import ReasoningManager
from fieldmate.brain.retrieval.orchestrator import RetrievalOrchestrator
from fieldmate.brain.retrieval.semantic_cache import QdrantSemanticCache
from fieldmate.brain.state.engine import StateEngine
from fieldmate.brain.state.models import FieldMateSession
from fieldmate.brain.vision import VisionEngine, DEFAULT_VISION_MODEL

load_dotenv()


DEFAULT_SYSTEM_PROMPT = """
You are FieldMate, a real-time technical diagnostic assistant for PC troubleshooting.

DOMAIN:
- Windows PCs and laptops
- Lenovo, Dell, HP, ASUS
- hardware troubleshooting
- Windows/software troubleshooting
- networking troubleshooting

You are assisting a technician.

DIAGNOSTIC & EVIDENCE DISCIPLINE:
- Hierarchy of Evidence:
  1. User Claims (source: 'user') -> Reported statements by the technician.
  2. Visual Facts (source: 'camera_vision') -> Directly observed physical evidence & OCR.
  3. Measurements (source: 'sensor') -> Numeric readings with units.
  4. Retrieved Documentation -> Manufacturer service manuals and verified case resolutions.
- CONTRADICTION RULE: When a technician's verbal statement conflicts with physical camera evidence (e.g., technician says "fan isn't spinning" but visual evidence shows "fan blades visibly rotating"), NEVER blindly accept the false premise. Preserve both in state, flag the contradiction, set "clarification_needed": true, and provide a polite clarifying response to resolve the discrepancy.
- ACTIVE INSPECTION: When diagnostic evidence is incomplete, recommend the specific next physical check or camera view in "next_action" (e.g., "Inspect motherboard diagnostic Q-LEDs" or "Move camera closer to RAM latch").
- Distinguish observations, hypotheses, recommendations, test results, and confirmed resolutions.
- Never present a hypothesis as a confirmed diagnosis.
- Do not invent facts or specs.
- Use retrieved evidence when relevant.
- Do not repeat diagnostic tests that are already completed unless there is a specific reason.
- CRITICAL RESOLUTION RULE: If current_observation states that an action, fix, setting, or step fixed, saved, or resolved the issue (e.g. "switching to power-saving mode fixed the overheating problem"), you MUST populate "resolution_confirmed" with a clear summary string of the fix.
- Keep recommendations safe and appropriate for PC troubleshooting.

OUTPUT:
Return ONLY valid JSON matching this schema:
{
  "response": "spoken response to the technician",
  "hypothesis": "current best hypothesis or null",
  "confidence": 0.0,
  "next_action": "next diagnostic action or inspection step or null",
  "clarification_needed": false,
  "clarification_question": "question or null",
  "evidence_ids": [],
  "state_updates": [],
  "resolution_proposed": "summary string or null",
  "resolution_confirmed": "summary string if technician confirmed fix/solution else null"
}

state_updates must contain only explicit diagnostic event proposals supported by the application:
- "equipment_identified" (payload: {"model": "Dell XPS 15", "oem": "Dell"})
- "symptom_recorded" (payload: {"name": "overheating"})
- "observation_recorded" (payload: {"name": "power mode", "value": "power-saving"})
- "resolution_confirmed" (payload: {"resolution": "Switching to power-saving mode resolved overheating"})
Never invent unlisted event types.
"""


@dataclass(slots=True)
class BrainRuntime:
    """
    Owns the concrete infrastructure composing one FieldMate Brain.

    Brain itself remains infrastructure-agnostic.

        StateEngine
             +
        Qdrant repository
             +
        RetrievalOrchestrator
             +
        MemoryManager
             +
        Groq ReasoningManager
             ↓
           Brain
    """

    brain: Brain
    repository: QdrantMemoryRepository
    groq: AsyncOpenAI
    semantic_cache: QdrantSemanticCache | None = None

    async def ensure_ready(self) -> None:
        """
        Prepare external infrastructure before the conversational
        hot path starts.
        """
        await self.repository.ensure_collection()
        if self.semantic_cache:
            await self.semantic_cache.ensure_collection()

    async def close(self) -> None:
        """
        Cleanly close brain-owned external resources.
        """
        await self.brain.close()
        await self.groq.close()


def build_brain_runtime(
    *,
    session_id: str,
    owner_id: str | None = None,
    system_prompt: str | None = None,
    repository: QdrantMemoryRepository | None = None,
    retrieval: RetrievalOrchestrator | None = None,
) -> BrainRuntime:
    """
    Construct the complete FieldMate diagnostic brain.

    This is dependency composition only.

    It does not perform network I/O. Call ensure_ready() during
    application startup before accepting diagnostic turns.
    """

    resolved_owner = owner_id or os.getenv("FIELDMATE_USER_ID", "tech_john_doe")

    session = FieldMateSession(
        session_id=session_id,
        owner_id=resolved_owner,
    )

    state = StateEngine(
        session,
    )

    if repository is None:
        qdrant_config = QdrantConfig.from_env()
        repository = QdrantMemoryRepository(
            qdrant_config,
        )

    if retrieval is None:
        semantic_cache_enabled = (
            os.getenv(
                "FIELDMATE_SEMANTIC_CACHE_ENABLED",
                "true",
            ).lower()
            == "true"
        )

        semantic_cache = QdrantSemanticCache(
            repository,
            collection_name=os.getenv(
                "FIELDMATE_SEMANTIC_CACHE_COLLECTION",
                "fieldmate_semantic_cache",
            ),
            threshold=float(
                os.getenv(
                    "FIELDMATE_SEMANTIC_CACHE_THRESHOLD",
                    "0.90",
                )
            ),
            ttl_seconds=float(
                os.getenv(
                    "FIELDMATE_SEMANTIC_CACHE_TTL_SECONDS",
                    "86400.0",
                )
            ),
            enabled=semantic_cache_enabled,
        )

        retrieval = RetrievalOrchestrator(
            repository,
            timeout_ms=int(
                os.getenv(
                    "FIELDMATE_RETRIEVAL_TIMEOUT_MS",
                    "1200",
                )
            ),
            prefetch_timeout_ms=int(
                os.getenv(
                    "FIELDMATE_PREFETCH_TIMEOUT_MS",
                    "3000",
                )
            ),
            prefetch_ttl_ms=int(
                os.getenv(
                    "FIELDMATE_PREFETCH_TTL_MS",
                    "5000",
                )
            ),
            semantic_cache=semantic_cache,
        )
    else:
        semantic_cache = getattr(retrieval, "semantic_cache", None)

    memory = MemoryManager()

    groq = AsyncOpenAI(
        api_key=os.environ.get("GROQ_API_KEY", "mock-key"),
        base_url=os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        ),
    )

    default_model = os.getenv(
        "GROQ_MODEL",
        # NOTE: Groq has scheduled "llama-3.1-8b-instant" for shutdown
        # on 2026-08-16. Kept as-is per explicit request.
        "llama-3.1-8b-instant",
    )

    reasoning = ReasoningManager(
        client=groq,
        model=default_model,
        system_prompt=(
            system_prompt
            or os.getenv(
                "FIELDMATE_BRAIN_SYSTEM_PROMPT",
                DEFAULT_SYSTEM_PROMPT,
            )
        ),
        timeout=float(os.getenv("FIELDMATE_GROQ_TIMEOUT_SECONDS", "20.0")),
    )

    vision = VisionEngine(
        client=groq,
        model=os.getenv(
            "GROQ_VISION_MODEL",
            "qwen/qwen3.6-27b",
        ),
    )

    brain = Brain(
        state=state,
        retrieval=retrieval,
        memory=memory,
        reasoning=reasoning,
        vision=vision,
    )

    return BrainRuntime(
        brain=brain,
        repository=repository,
        groq=groq,
        semantic_cache=semantic_cache,
    )
