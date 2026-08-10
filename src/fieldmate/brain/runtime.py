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
from fieldmate.brain.state.engine import StateEngine
from fieldmate.brain.state.models import FieldMateSession


load_dotenv()


DEFAULT_SYSTEM_PROMPT = """
You are FieldMate, a real-time technical diagnostic assistant.

DOMAIN:
- Windows PCs and laptops
- Lenovo, Dell, HP, ASUS
- hardware troubleshooting
- Windows/software troubleshooting
- networking troubleshooting

You are assisting a technician.

DIAGNOSTIC RULES:
- Distinguish observations, hypotheses, recommendations, test results,
  and confirmed resolutions.
- Never present a hypothesis as a confirmed diagnosis.
- Do not invent facts.
- Use retrieved evidence when relevant.
- Preserve contradictory evidence.
- Do not repeat diagnostic tests that are already completed unless there
  is a specific reason to repeat them.
- Prefer the next useful discriminating diagnostic action.
- If evidence is insufficient, say so and request the most useful
  next observation or test.
- Do not claim that a proposed action resolved a problem unless the
  technician explicitly confirms the resolution.
- Keep recommendations safe and appropriate for PC troubleshooting.

OUTPUT:
Return ONLY valid JSON.

The JSON object must have exactly these conceptual fields:

{
  "response": "spoken response to the technician",
  "hypothesis": "current best hypothesis or null",
  "confidence": 0.0,
  "next_action": "next diagnostic action or null",
  "clarification_needed": false,
  "clarification_question": "question or null",
  "evidence_ids": [],
  "state_updates": [],
  "resolution_proposed": null,
  "resolution_confirmed": null
}

state_updates must contain only explicit diagnostic event proposals
supported by the application. Never invent event types.
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

    async def ensure_ready(self) -> None:
        """
        Prepare external infrastructure before the conversational
        hot path starts.
        """
        await self.repository.ensure_collection()

    async def close(self) -> None:
        """
        Cleanly close brain-owned external resources.
        """
        await self.brain.close()
        await self.groq.close()


def build_brain_runtime(
    *,
    session_id: str,
    system_prompt: str | None = None,
) -> BrainRuntime:
    """
    Construct the complete FieldMate diagnostic brain.

    This is dependency composition only.

    It does not perform network I/O. Call ensure_ready() during
    application startup before accepting diagnostic turns.
    """

    session = FieldMateSession(
        session_id=session_id,
    )

    state = StateEngine(
        session,
    )

    qdrant_config = QdrantConfig.from_env()

    repository = QdrantMemoryRepository(
        qdrant_config,
    )

    retrieval = RetrievalOrchestrator(
        repository,
        timeout_ms=int(
            os.getenv(
                "FIELDMATE_RETRIEVAL_TIMEOUT_MS",
                "600",
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
    )

    memory = MemoryManager()

    groq = AsyncOpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url=os.getenv(
            "GROQ_BASE_URL",
            "https://api.groq.com/openai/v1",
        ),
    )

    reasoning = ReasoningManager(
        client=groq,
        model=os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant",
        ),
        system_prompt=(
            system_prompt
            or os.getenv(
                "FIELDMATE_BRAIN_SYSTEM_PROMPT",
                DEFAULT_SYSTEM_PROMPT,
            )
        ),
    )

    brain = Brain(
        state=state,
        retrieval=retrieval,
        memory=memory,
        reasoning=reasoning,
    )

    return BrainRuntime(
        brain=brain,
        repository=repository,
        groq=groq,
    )