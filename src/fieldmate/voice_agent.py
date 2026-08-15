from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from contextlib import suppress
from dataclasses import asdict, is_dataclass
from typing import Any, AsyncIterable, Callable

from dotenv import load_dotenv
from openai import AsyncOpenAI

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import deepgram, rime

from fieldmate.brain.qdrant.config import QdrantConfig
from fieldmate.brain.qdrant.repository import (
    QdrantMemoryRepository,
)
from fieldmate.brain.retrieval.orchestrator import (
    RetrievalOrchestrator,
)
from fieldmate.brain.retrieval.semantic_cache import (
    QdrantSemanticCache,
)

from fieldmate.brain.models import DiagnosticContext
from fieldmate.brain.runtime import build_brain_runtime
from fieldmate.brain.voice_router import (
    ParallelTurnRouter,
)
from fieldmate.brain.pronounce import tts_pronounce

# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# LOGGING
# ============================================================

LOG_LEVEL = os.getenv(
    "FIELDMATE_LOG_LEVEL",
    "INFO",
).upper()

logging.basicConfig(
    level=getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    ),
    format=(
        "%(asctime)s "
        "%(levelname)s "
        "%(name)s: "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "fieldmate.voice"
)


# ============================================================
# LIVEKIT / FLUX CONFIGURATION
# ============================================================

FLUX_MODEL = os.getenv(
    "FIELDMATE_FLUX_MODEL",
    "flux-general-en",
)

# Flux documentation allows eager EOT from 0.3–0.9.
#
# We intentionally keep this below the final EOT threshold.
#
# Eager EOT is used as an early signal for speculation.
# It does NOT mean the turn is final.
FLUX_EAGER_EOT = float(
    os.getenv(
        "FIELDMATE_FLUX_EAGER_EOT",
        "0.4",
    )
)

FLUX_EOT_THRESHOLD = float(
    os.getenv(
        "FIELDMATE_FLUX_EOT_THRESHOLD",
        "0.7",
    )
)

FLUX_EOT_TIMEOUT_MS = int(
    os.getenv(
        "FIELDMATE_FLUX_EOT_TIMEOUT_MS",
        "3000",
    )
)


# ============================================================
# GROQ CONFIGURATION
# ============================================================

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    # NOTE: Groq has scheduled "llama-3.1-8b-instant" for shutdown on
    # 2026-08-16. Kept as-is per explicit request. When Groq turns it
    # off, calls using this model will start failing until GROQ_MODEL
    # is changed (e.g. to "openai/gpt-oss-20b", Groq's recommended
    # replacement) via this env var.
    "llama-3.1-8b-instant",
)

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
)

GROQ_BASE_URL = os.getenv(
    "GROQ_BASE_URL",
    "https://api.groq.com/openai/v1",
)

GROQ_TEMPERATURE = float(
    os.getenv(
        "FIELDMATE_GROQ_TEMPERATURE",
        "0.15",
    )
)

GROQ_MAX_TOKENS = int(
    os.getenv(
        "FIELDMATE_GROQ_MAX_TOKENS",
        "220",
    )
)

# Hard upper bound for one reasoning request.
#
# This prevents a pathological backend request from holding
# the voice turn forever.
GROQ_TIMEOUT_SECONDS = float(
    os.getenv(
        "FIELDMATE_GROQ_TIMEOUT_SECONDS",
        "12.0",
    )
)


# ============================================================
# QDRANT / RETRIEVAL CONFIGURATION
# ============================================================

RETRIEVAL_TIMEOUT_MS = int(
    os.getenv(
        "FIELDMATE_RETRIEVAL_TIMEOUT_MS",
        "1200",
    )
)

PREFETCH_TIMEOUT_MS = int(
    os.getenv(
        "FIELDMATE_PREFETCH_TIMEOUT_MS",
        "3000",
    )
)

PREFETCH_TTL_MS = int(
    os.getenv(
        "FIELDMATE_PREFETCH_TTL_MS",
        "5000",
    )
)

RETRIEVAL_LIMIT = int(
    os.getenv(
        "FIELDMATE_RETRIEVAL_LIMIT",
        "8",
    )
)


# ============================================================
# VOICE / RIME CONFIGURATION
# ============================================================

RIME_MODEL = os.getenv(
    "RIME_MODEL",
    "arcana",
)

RIME_SPEAKER = os.getenv(
    "RIME_SPEAKER",
    "celeste",
)

RIME_SAMPLE_RATE = int(
    os.getenv(
        "RIME_SAMPLE_RATE",
        "24000",
    )
)


# ============================================================
# LATENCY CONFIGURATION
# ============================================================

# Partial transcripts are NOT allowed to immediately trigger
# retrieval.
#
# This is the minimum amount of useful text before speculation.
PREFETCH_MIN_CHARS = int(
    os.getenv(
        "FIELDMATE_PREFETCH_MIN_CHARS",
        "12",
    )
)

# Don't launch another speculative retrieval more often than
# this unless a high-value identifier appears.
PREFETCH_MIN_INTERVAL_MS = int(
    os.getenv(
        "FIELDMATE_PREFETCH_MIN_INTERVAL_MS",
        "150",
    )
)

# Maximum amount of evidence text sent to Groq.
#
# This protects both latency and context quality.
MAX_EVIDENCE_CHARS = int(
    os.getenv(
        "FIELDMATE_MAX_EVIDENCE_CHARS",
        "9000",
    )
)

# Number of recent conversational turns retained locally.
#
# This is NOT canonical diagnostic state.
#
# Diagnostic state belongs to the Brain/state engine.
MAX_HISTORY_MESSAGES = int(
    os.getenv(
        "FIELDMATE_MAX_HISTORY_MESSAGES",
        "8",
    )
)


# ============================================================
# FIELD MATE SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are FieldMate, a real-time technical field assistant.

DOMAIN
------
You troubleshoot Windows PCs and laptops.

INITIAL OEM SCOPE
-----------------
Lenovo
Dell
HP
ASUS

SUPPORTED AREAS
---------------
Windows hardware
Windows software
Windows networking

Examples include:

Hardware:
RAM, SSD, HDD, NVMe, SATA, thermals, fans, cooling,
display, keyboard, touchpad, USB, battery, charging,
power, motherboard-related symptoms, exposed sensors.

Software:
Windows boot problems, drivers, services, updates,
application failures, crashes, BSODs, corruption,
permissions, configuration, Event Viewer and logs.

Networking:
Wi-Fi, Ethernet, DNS, DHCP, adapters, Windows network
configuration and common client/router interaction.

DIAGNOSTIC DISCIPLINE
---------------------
You are assisting a technician.

Hierarchy of Evidence:
1. User Claims -> Reported verbal statements by the technician.
2. Visual Observations -> Directly observed physical camera evidence and OCR.
3. Measurements -> Sensor readings with units.
4. Retrieved Documentation -> Manufacturer service manuals and verified case resolutions.

CONTRADICTION RULE:
When a technician's verbal statement conflicts with physical camera evidence (e.g. technician says "fan isn't spinning" but visual evidence shows "fan blades visibly rotating"), NEVER blindly accept the false premise. Politely flag the discrepancy and ask for clarification.

ACTIVE INSPECTION:
When diagnostic evidence is incomplete, recommend the specific next physical check or camera view in your response.

Treat observations as observations.

Treat hypotheses as hypotheses.

Treat recommendations as recommendations.

Do not claim a diagnosis is confirmed without evidence.

Do not invent measurements.

Do not invent test results.

Do not invent equipment specifications.

Do not claim that a repair worked unless it has actually
been confirmed.

Prefer the next useful diagnostic action.

Prefer tests that distinguish between competing hypotheses.

Do not recommend a test that has already been completed
unless there is a specific reason to repeat it.

If retrieved evidence conflicts, preserve the uncertainty.

If evidence is insufficient, say what useful information
or diagnostic check is needed.

RETRIEVED EVIDENCE
------------------
Retrieved Qdrant material is evidence, not canonical state.

It may contain:
- previous cases
- resolutions
- procedures
- equipment-specific information
- fault relationships
- observations
- patterns

Do not blindly treat every retrieved memory as universally true.

Pay attention to provenance, verification, confidence,
contradictions and equipment scope.

VOICE STYLE
-----------
You are speaking to a technician.

Be concise.

Be direct.

Be natural.

Use spoken language.

Do not use markdown.

Do not use bullet symbols.

Do not use emojis.

Do not unnecessarily repeat the technician's words.

Prefer one useful next step over a giant troubleshooting tree.

When a diagnostic action is recommended, explain briefly
what result would mean when that is useful.

SAFETY
------
For hardware, electrical, battery, thermal, power and
disassembly work, be appropriately conservative.

Do not provide unsafe instructions.

IMPORTANT
---------
The application owns canonical diagnostic state.

You reason over the supplied state/evidence.

You do not become the source of truth for application state.
"""


# ============================================================
# TECHNICAL VOCABULARY
# ============================================================

# This classifier is deliberately cheap.
#
# It exists only to decide whether Qdrant speculation is worth
# attempting.
#
# It is NOT the diagnostic reasoning engine.

TECHNICAL_TERMS = frozenset(
    {
        # OEM
        "lenovo",
        "dell",
        "hp",
        "asus",

        # platform
        "windows",
        "laptop",
        "desktop",
        "pc",
        "computer",

        # networking
        "wifi",
        "wi-fi",
        "ethernet",
        "bluetooth",
        "network",
        "internet",
        "router",
        "dhcp",
        "dns",
        "ip",
        "adapter",
        "connection",
        "connectivity",
        "disconnect",
        "disconnecting",

        # hardware
        "cpu",
        "gpu",
        "ram",
        "memory",
        "ssd",
        "nvme",
        "sata",
        "hdd",
        "disk",
        "storage",
        "fan",
        "cooling",
        "thermal",
        "temperature",
        "overheat",
        "overheating",
        "battery",
        "charger",
        "charging",
        "power",
        "screen",
        "display",
        "keyboard",
        "touchpad",
        "mouse",
        "usb",

        # software
        "driver",
        "drivers",
        "bios",
        "uefi",
        "boot",
        "startup",
        "crash",
        "crashing",
        "freeze",
        "freezing",
        "bsod",
        "blue",
        "screen",
        "update",
        "service",
        "powershell",
        "cmd",
        "terminal",
        "device",
        "event",
        "eventviewer",

        # diagnostic language
        "fault",
        "error",
        "code",
        "failure",
        "failed",
        "problem",
        "issue",
        "diagnose",
        "diagnostic",
        "test",
        "tested",
        "measurement",
        "symptom",
        "working",
        "broken",
        "slow",
        "lag",
    }
)


# High-value identifiers deserve stronger speculative priority.
#
# These are intentionally generic enough to cover Windows/OEM
# troubleshooting without pretending to know every possible
# future fault code.

FAULT_CODE_RE = re.compile(
    r"\b(?:"
    r"(?:ERR|ERROR|E|F|WHEA|BSOD|STOP)"
    r"[-_ ]?[0-9A-F]{2,8}"
    r"|"
    r"0x[0-9A-F]{4,16}"
    r"|"
    r"[A-Z]{2,12}-[A-Z0-9]{2,12}"
    r")\b",
    re.IGNORECASE,
)


# ============================================================
# HELPERS
# ============================================================

def normalize_text(
    text: str,
) -> str:
    """
    Normalize whitespace while preserving the actual words.

    This is used for:
    - duplicate final transcript detection
    - partial stabilization
    - cache-key comparison
    """

    return " ".join(
        text.strip().split()
    )


def is_technical(
    text: str,
) -> bool:
    """
    Extremely cheap technical-domain heuristic.

    This is intentionally conservative.

    It should NOT reject technical queries merely because
    they don't contain one of our vocabulary words, because
    natural technician language is broad.

    It mainly exists to avoid obvious Qdrant work for:
        "thanks"
        "okay"
        "repeat that"
        "what can you do?"
    """

    normalized = normalize_text(
        text
    ).lower()

    if not normalized:
        return False

    if FAULT_CODE_RE.search(
        normalized
    ):
        return True

    tokens = set(
        normalized.replace(
            "/",
            " ",
        ).replace(
            "-",
            " ",
        ).split()
    )

    if tokens & TECHNICAL_TERMS:
        return True

    # Natural-language troubleshooting phrases.
    return any(
        phrase in normalized
        for phrase in (
            "not working",
            "doesn't work",
            "won't boot",
            "can't connect",
            "keeps disconnecting",
            "keeps crashing",
            "keeps freezing",
            "blue screen",
            "device manager",
            "windows update",
            "event viewer",
            "wifi issue",
            "network issue",
            "internet issue",
        )
    )


def extract_identifiers(
    text: str,
) -> tuple[str, ...]:
    """
    Extract high-value lexical identifiers.

    These are used only by the speculative query stabilizer.
    """

    normalized = normalize_text(
        text
    )

    identifiers = set(
        match.group(0).upper()
        for match in FAULT_CODE_RE.finditer(
            normalized
        )
    )

    lower = normalized.lower()

    for term in (
        "lenovo",
        "dell",
        "hp",
        "asus",
        "windows",
        "wifi",
        "wi-fi",
        "ethernet",
        "bluetooth",
        "bios",
        "uefi",
        "device manager",
        "event viewer",
    ):
        if term in lower:
            identifiers.add(term)

    return tuple(
        sorted(
            identifiers
        )
    )


def clip_text(
    text: str,
    limit: int,
) -> str:
    """
    Hard context bound.

    Never silently allow a huge Qdrant payload to consume the
    entire Groq context.
    """

    if len(text) <= limit:
        return text

    return (
        text[: max(0, limit - 80)]
        + "\n...[evidence clipped]..."
    )


# ============================================================
# SAFE SERIALIZATION
# ============================================================

def _json_safe(
    value: Any,
    *,
    depth: int = 0,
) -> Any:
    """
    Convert project objects into compact JSON-safe structures.

    The retrieval context is intentionally treated as an opaque
    domain object here.

    This lets the voice adapter work with dataclasses, pydantic
    models, dictionaries and simple objects without coupling
    voice_agent.py to every memory implementation detail.
    """

    if depth > 5:
        return "<max-depth>"

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(
        value,
        (list, tuple, set, frozenset),
    ):
        return [
            _json_safe(
                item,
                depth=depth + 1,
            )
            for item in value
        ]

    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): _json_safe(
                item,
                depth=depth + 1,
            )
            for key, item in value.items()
        }

    if is_dataclass(value):
        return _json_safe(
            asdict(value),
            depth=depth + 1,
        )

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(model_dump):
        with suppress(
            Exception
        ):
            return _json_safe(
                model_dump(),
                depth=depth + 1,
            )

    to_dict = getattr(
        value,
        "to_dict",
        None,
    )

    if callable(to_dict):
        with suppress(
            Exception
        ):
            return _json_safe(
                to_dict(),
                depth=depth + 1,
            )

    if hasattr(
        value,
        "__dict__",
    ):
        with suppress(
            Exception
        ):
            return _json_safe(
                {
                    key: item
                    for key, item
                    in vars(value).items()
                    if not key.startswith("_")
                },
                depth=depth + 1,
            )

    return str(value)


def serialize_retrieval_context(
    retrieval_result: Any,
) -> str:
    """
    Convert RetrievalResult into a bounded reasoning context.

    Important:

    Qdrant is not dumped blindly.

    We preserve:
    - routing mode
    - routing reason
    - prefetch status
    - latency
    - retrieved memories/context

    while imposing a hard size limit.
    """

    context = getattr(
        retrieval_result,
        "context",
        None,
    )

    plan = getattr(
        retrieval_result,
        "plan",
        None,
    )

    mode = getattr(
        plan,
        "mode",
        None,
    )

    mode_value = getattr(
        mode,
        "value",
        str(mode),
    )

    reason = getattr(
        plan,
        "reason",
        "unknown",
    )

    metadata = {
        "retrieval_mode": mode_value,
        "retrieval_reason": reason,
        "retrieval_latency_ms": round(
            float(
                getattr(
                    retrieval_result,
                    "latency_ms",
                    0.0,
                )
            ),
            2,
        ),
        "prefetched": bool(
            getattr(
                retrieval_result,
                "prefetched",
                False,
            )
        ),
        "timed_out": bool(
            getattr(
                retrieval_result,
                "timed_out",
                False,
            )
        ),
    }

    memories = getattr(
        context,
        "memories",
        (),
    )

    safe_memories = []

    for memory in memories:
        safe_memories.append(
            _json_safe(
                memory
            )
        )

    payload = {
        "retrieval": metadata,
        "evidence": safe_memories,
    }

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
        default=str,
    )

    return clip_text(
        serialized,
        MAX_EVIDENCE_CHARS,
    )


# ============================================================
# QUERY STABILIZER
# ============================================================

class QueryStabilizer:
    """
    Controls speculative Qdrant retrieval from Flux partials.

    Bad:

        "My"
        "My Dell"
        "My Dell laptop"
        "My Dell laptop keeps"
        "My Dell laptop keeps disconnecting"

    each independently hitting Qdrant.

    Good:

        partial
          ↓
        normalize
          ↓
        meaningful delta?
          ↓
        speculate

    High-value identifiers can bypass the normal delta threshold.
    """

    def __init__(
        self,
        *,
        min_chars: int,
        min_interval_ms: int,
    ) -> None:

        self.min_chars = min_chars

        self.min_interval_seconds = (
            min_interval_ms / 1000
        )

        self.last_query = ""

        self.last_identifiers: tuple[
            str,
            ...
        ] = ()

        self.last_launch_at = 0.0

    def consider(
        self,
        transcript: str,
    ) -> str | None:

        normalized = normalize_text(
            transcript
        )

        if len(normalized) < self.min_chars:
            return None

        if not is_technical(
            normalized
        ):
            return None

        identifiers = extract_identifiers(
            normalized
        )

        now = time.perf_counter()

        identifier_changed = (
            identifiers
            != self.last_identifiers
        )

        elapsed = (
            now - self.last_launch_at
        )

        # Always allow a newly introduced fault/model/
        # Windows identifier to trigger speculation.
        if identifier_changed:
            meaningful = True

        else:

            if elapsed < self.min_interval_seconds:
                return None

            meaningful = (
                self._meaningfully_changed(
                    self.last_query,
                    normalized,
                )
            )

        if not meaningful:
            return None

        self.last_query = normalized

        self.last_identifiers = identifiers

        self.last_launch_at = now

        return normalized

    def reset(self) -> None:
        """Reset speculative state at a finalized turn boundary."""

        self.last_query = ""
        self.last_identifiers = ()
        self.last_launch_at = 0.0

    @staticmethod
    def _meaningfully_changed(
        old: str,
        new: str,
    ) -> bool:

        if not old:
            return True

        old_tokens = old.lower().split()
        new_tokens = new.lower().split()

        if len(new_tokens) <= len(
            old_tokens
        ):
            return False

        # Ignore tiny one-word expansions.
        added = new_tokens[
            len(old_tokens):
        ]

        if len(added) <= 1:
            return False

        # New technical content is more valuable.
        if any(
            token in TECHNICAL_TERMS
            for token in added
        ):
            return True

        # Otherwise require enough new content.
        return len(added) >= 3


# ============================================================
# TURN GENERATION CONTROLLER
# ============================================================

class GenerationController:
    """
    Protects the realtime path from stale asynchronous work.

    Every finalized turn gets a monotonically increasing
    generation number.

    Any result belonging to an older generation is discarded.

    This is deliberately local to the voice layer.

    Canonical diagnostic state remains owned by the Brain/state
    engine.
    """

    def __init__(self) -> None:

        self.generation = 0

        self.active_task: (
            asyncio.Task | None
        ) = None

        self.active_speech: Any = None

        self._lock = asyncio.Lock()

    async def begin(
        self,
    ) -> int:

        current = asyncio.current_task()

        async with self._lock:

            self.generation += 1

            generation = (
                self.generation
            )

            old_task = (
                self.active_task
            )

            self.active_task = current

        if (
            old_task is not None
            and old_task is not current
            and not old_task.done()
        ):
            old_task.cancel()

            with suppress(
                asyncio.CancelledError
            ):
                await old_task

        return generation

    def is_current(
        self,
        generation: int,
    ) -> bool:

        return (
            generation
            == self.generation
        )

    def attach_task(
        self,
        task: asyncio.Task,
    ) -> None:

        self.active_task = task

    def cancel_active(
        self,
    ) -> None:

        task = self.active_task

        if (
            task is not None
            and not task.done()
        ):
            task.cancel()

        self.active_task = None

        speech = getattr(self, "active_speech", None)
        if speech is not None and not getattr(speech, "interrupted", False):
            try:
                speech.interrupt(force=True)
            except Exception:
                pass
        self.active_speech = None

    def invalidate(
        self,
    ) -> None:

        self.generation += 1

        self.cancel_active()


# ============================================================
# LOCAL CONVERSATION HISTORY
# ============================================================

class ConversationHistory:
    """
    Small bounded conversational history for Groq.

    This is not diagnostic state.

    It exists only so normal conversation and references such as
    "repeat that" remain natural.

    Canonical diagnostic state remains elsewhere.
    """

    def __init__(
        self,
        max_messages: int,
    ) -> None:

        self.messages = deque(
            maxlen=max_messages
        )

    def add_user(
        self,
        text: str,
    ) -> None:

        self.messages.append(
            {
                "role": "user",
                "content": text,
            }
        )

    def add_assistant(
        self,
        text: str,
    ) -> None:

        self.messages.append(
            {
                "role": "assistant",
                "content": text,
            }
        )

    def snapshot(
        self,
    ) -> list[dict[str, str]]:

        return list(
            self.messages
        )


# ============================================================
# VOICE BRAIN ADAPTER
# ============================================================

class VoiceBrain:
    """
    Realtime bridge between:

        LiveKit
            ↓
        RetrievalOrchestrator
            ↓
        Qdrant
            ↓
        Groq
            ↓
        LiveKit/Rime

    This is intentionally NOT the canonical Brain.

    It is an integration boundary.

    As the deeper diagnostic reasoning/state layers mature,
    this adapter can call them without changing the realtime
    transport code.
    """

    def __init__(
        self,
        *,
        repository: QdrantMemoryRepository,
        retrieval: RetrievalOrchestrator,
        groq: AsyncOpenAI,
        owner_id: str | None = None,
        canonical_brain: Any = None,
    ) -> None:

        self.repository = repository

        self.retrieval = retrieval

        self.groq = groq

        self.owner_id = owner_id

        self.canonical_brain = canonical_brain

    # --------------------------------------------------------
    # SPECULATION
    # --------------------------------------------------------

    async def speculate(
        self,
        query: str,
    ) -> None:
        """
        Fire-and-forget retrieval speculation.

        IMPORTANT:

        The caller must NEVER await this on the hot path.

        RetrievalOrchestrator itself owns:
        - duplicate prefetch protection
        - TTL
        - bounded prefetch execution
        - completed-prefetch consumption
        """

        query = normalize_text(
            query
        )

        if not query:
            return

        try:

            await self.retrieval.prefetch(
                query,
                owner_id=self.owner_id,
                limit=RETRIEVAL_LIMIT,
            )

            logger.debug(
                ">>> PREFETCH REQUESTED"
            )

        except asyncio.CancelledError:
            raise

        except Exception:
            logger.exception(
                ">>> SPECULATIVE RETRIEVAL FAILED"
            )

    # --------------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        owner_id: str | None = None,
    ) -> Any:

        return await self.retrieval.retrieve(
            query,
            owner_id=owner_id or self.owner_id,
            limit=RETRIEVAL_LIMIT,
        )

    async def warm_next(
        self,
        text: str,
    ) -> None:
        """
        Prime retrieval during TTS playback.

        This borrows Primd's warm-next idea without adding a
        second model: the already-generated assistant response
        is used as a cheap proxy for the next diagnostic topic.
        The result remains speculative and never blocks speech.
        """

        text = normalize_text(text)

        if len(text) < PREFETCH_MIN_CHARS:
            return

        if not is_technical(text):
            return

        try:
            await self.retrieval.prefetch(
                text,
                limit=max(4, min(RETRIEVAL_LIMIT, 6)),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug(
                ">>> NEXT-TURN WARM FAILED",
                exc_info=True,
            )

    async def warm(self) -> None:
        """Warm the Groq HTTP connection before the first turn."""

        try:
            started = time.perf_counter()
            await asyncio.wait_for(
                self.groq.models.list(),
                timeout=6.0,
            )
            logger.info(
                ">>> GROQ CONNECTION WARM %.1fms",
                (time.perf_counter() - started) * 1000,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Warming is opportunistic. Never prevent startup if
            # the models endpoint is unavailable.
            logger.warning(
                ">>> GROQ CONNECTION WARM FAILED",
                exc_info=True,
            )

    async def close(self) -> None:
        """Close speculative retrieval and the Groq client."""

        with suppress(Exception):
            await self.retrieval.close()

        with suppress(Exception):
            await self.groq.close()

    async def chat(
        self,
        user_text: str,
        history: list[dict[str, str]],
    ) -> str:
        """Non-streaming direct chat fallback."""
        parts: list[str] = []
        try:
            async for chunk in self.stream_response(
                user_text=user_text,
                evidence="",
                history=history,
            ):
                parts.append(chunk)
        except Exception:
            logger.exception(">>> VOICE BRAIN CHAT FAILED")
        return tts_pronounce("".join(parts).strip())

    # --------------------------------------------------------
    # GROQ STREAM
    # --------------------------------------------------------

    async def stream_response(
        self,
        *,
        user_text: str,
        evidence: str,
        history: list[dict[str, str]],
    ) -> AsyncIterable[str]:
        """
        Stream Groq output directly toward LiveKit.

        LiveKit's session.say() accepts AsyncIterable[str].

        Therefore:

        Groq token
            ↓
        session.say()
            ↓
        Rime
            ↓
        audio

        We do not wait for the entire Groq response.
        """

        retrieval_instruction = (
            """
Use the retrieved evidence below when relevant.

The evidence is not automatically authoritative.

Preserve uncertainty and contradictions.

Do not mention "Qdrant", "retrieval", "vector database",
"embedding", or internal system mechanics to the technician.

If the retrieved evidence contains a previous confirmed resolution or fix for this machine or symptom, explicitly recommend or cite it to the technician (for example: "Last time / in a previous case, switching to power-saving mode resolved this issue. Would you like to try that?").

RETRIEVED EVIDENCE:
"""
            + evidence
        )

        messages: list[
            dict[str, str]
        ] = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "system",
                "content": retrieval_instruction,
            },
        ]

        messages.extend(
            history
        )

        messages.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        started = time.perf_counter()

        first_token_at: (
            float | None
        ) = None

        stream = await self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
            stream=True,
            timeout=GROQ_TIMEOUT_SECONDS,
        )

        try:

            async for chunk in stream:

                choices = getattr(
                    chunk,
                    "choices",
                    None,
                )

                if not choices:
                    continue

                delta = getattr(
                    choices[0],
                    "delta",
                    None,
                )

                if delta is None:
                    continue

                text = getattr(
                    delta,
                    "content",
                    None,
                )

                if not text:
                    continue

                if first_token_at is None:

                    first_token_at = (
                        time.perf_counter()
                    )

                    logger.info(
                        ">>> GROQ TTFT: %.1f ms",
                        (
                            first_token_at
                            - started
                        ) * 1000,
                    )

                yield text

        except asyncio.CancelledError:

            logger.info(
                ">>> GROQ STREAM CANCELLED"
            )

            raise

        finally:

            elapsed = (
                time.perf_counter()
                - started
            ) * 1000

            logger.debug(
                ">>> GROQ STREAM CLOSED "
                "elapsed=%.1fms",
                elapsed,
            )

    # --------------------------------------------------------
    # PARALLEL GROQ + QDRANT
    # --------------------------------------------------------

    async def stream_parallel_response(
        self,
        *,
        user_text: str,
        history: list[dict[str, str]],
        router: ParallelTurnRouter,
        generation_is_current: Callable[[], bool],
    ) -> AsyncIterable[str]:
        """
        Run speculative Groq generation and Qdrant retrieval
        concurrently.

        Architecture:

            STT transcript
                  |
             +----+----+
             |         |
             v         v
           Groq     Qdrant
        speculative    |
             |         |
             |    +----+----+
             |    |         |
             | relevant  irrelevant/
             |    |      timeout/fail
             |    v         |
             | grounded     |
             |    Groq      |
             |      \\       /
             +-------+-----+
                     |
                     v
                    Rime

        Important:

        Groq is allowed to start immediately.

        Its speculative output is buffered inside
        ParallelTurnRouter and is NOT released to TTS until
        Qdrant determines whether technical grounding is
        required.

        If Qdrant says:

            RELEVANT
                -> speculative Groq is cancelled
                -> grounded Groq starts

            IRRELEVANT
                -> speculative Groq is released

            TIMEOUT
                -> speculative Groq is released

            FAILED
                -> speculative Groq is released
        """

        user_text = normalize_text(
            user_text
        )

        if not user_text:
            return

        # ----------------------------------------------------
        # QDRANT SIDE
        # ----------------------------------------------------

        # Build enriched retrieval query from conversation history if user_text is short or elliptical
        retrieval_query = user_text
        if len(user_text.split()) <= 7 and history:
            prior_user_turns = [
                m["content"] for m in history
                if m.get("role") == "user" and m.get("content")
            ]
            if prior_user_turns:
                retrieval_query = f"{prior_user_turns[-1]} {user_text}"

        async def retrieve() -> Any:
            return await self.retrieve(
                retrieval_query
            )

        # ----------------------------------------------------
        # GROQ SPECULATIVE SIDE
        # ----------------------------------------------------

        async def stream_groq(
            text: str,
        ) -> AsyncIterable[str]:
            """
            Start Groq immediately without retrieved evidence.


            The router owns buffering. Nothing is yielded to
            the caller until the routing decision is known.
            """

            async for chunk in self.stream_response(
                user_text=text,
                evidence="",
                history=history,
            ):
                if not generation_is_current():
                    return

                yield chunk

        # ----------------------------------------------------
        # GROUNDED PROMPT
        # ----------------------------------------------------

        def grounded_prompt(
            text: str,
            retrieval_result: Any,
        ) -> str:
            """
            Convert the retrieval result into additional
            technical grounding for the second Groq call.
            """

            context = getattr(
                retrieval_result,
                "context",
                None,
            )

            if context is None:
                return text

            evidence_items = getattr(
                context,
                "evidence",
                (),
            )

            if not evidence_items:
                return text

            evidence_lines: list[str] = []

            for item in evidence_items:

                content = getattr(
                    item,
                    "content",
                    None,
                )

                if not content:
                    continue

                evidence_lines.append(
                    str(content).strip()
                )

            if not evidence_lines:
                return text

            evidence_text = (
                "\n\n".join(
                    evidence_lines
                )
            )

            return (
                text
                + "\n\n"
                + "Use the following retrieved "
                + "technical evidence when answering. "
                + "Treat it as supporting evidence, "
                + "not unquestionable truth. "
                + "Preserve uncertainty or "
                + "contradictions.\n\n"
                + "RETRIEVED TECHNICAL EVIDENCE:\n"
                + evidence_text
            )

        # ----------------------------------------------------
        # ROUTER
        # ----------------------------------------------------

        async for chunk in router.stream(
            retrieve=retrieve,
            stream_groq=stream_groq,
            user_text=user_text,
            grounded_prompt=grounded_prompt,
            generation_is_current=(
                generation_is_current
            ),
        ):
            if not generation_is_current():
                logger.info(">>> ROUTER.STREAM generation_is_current FALSE, returning")
                return

            logger.info(">>> ROUTER YIELD chunk of len %d", len(chunk))
            yield chunk

# ============================================================
# FIELD MATE AGENT
# ============================================================

class FieldMate(Agent):

    def __init__(self) -> None:

        super().__init__(
            instructions=SYSTEM_PROMPT,
            allow_interruptions=True,
        )

    async def on_enter(
        self,
    ) -> None:

        logger.info(
            ">>> FIELDMATE AGENT ON_ENTER: SENDING GREETING SPEECH"
        )

        await self.session.say(
            "FieldMate is ready. What are we troubleshooting?",
            allow_interruptions=False,
        )


# ============================================================
# SERVER
# ============================================================

server = AgentServer()


# ============================================================
# SESSION ENTRYPOINT
# ============================================================

@server.rtc_session(
    agent_name="fieldmate",
)
async def entrypoint(
    ctx: JobContext,
) -> None:

    started_at = (
        time.perf_counter()
    )

    logger.info(
        ">>> FIELDMATE ENTRYPOINT"
    )

    # --------------------------------------------------------
    # VALIDATE CRITICAL CONFIG
    # --------------------------------------------------------

    if not GROQ_API_KEY:

        raise RuntimeError(
            "GROQ_API_KEY is missing."
        )

    repository: (
        QdrantMemoryRepository | None
    ) = None

    retrieval: (
        RetrievalOrchestrator | None
    ) = None

    # Local task registry exists before the try/finally so even
    # partial startup failures can shut down cleanly.
    prefetch_tasks: set[asyncio.Task] = set()

    try:

        config = (
            QdrantConfig.from_env()
        )

        repository = (
            QdrantMemoryRepository(
                config
            )
        )

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

        # ----------------------------------------------------
        # RETRIEVAL ORCHESTRATOR
        # ----------------------------------------------------

        retrieval = (
            RetrievalOrchestrator(
                repository,
                timeout_ms=(
                    RETRIEVAL_TIMEOUT_MS
                ),
                prefetch_timeout_ms=(
                    PREFETCH_TIMEOUT_MS
                ),
                prefetch_ttl_ms=(
                    PREFETCH_TTL_MS
                ),
                semantic_cache=semantic_cache,
            )
        )

        logger.info(
            ">>> RETRIEVAL ORCHESTRATOR READY "
            "hot=%dms prefetch=%dms ttl=%dms",
            RETRIEVAL_TIMEOUT_MS,
            PREFETCH_TIMEOUT_MS,
            PREFETCH_TTL_MS,
        )

        # ----------------------------------------------------
        # GROQ
        # ----------------------------------------------------

        groq = AsyncOpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )

        def resolve_user_identity() -> str:
            # 1. Environment variable override (ideal for local testing)
            env_user = os.getenv("FIELDMATE_USER_ID")
            if env_user and env_user.strip():
                return env_user.strip()

            # 2. Job Metadata JSON (e.g. {"user_id": "tech_123"})
            if getattr(ctx, "job", None) and getattr(ctx.job, "metadata", None):
                try:
                    meta = json.loads(ctx.job.metadata)
                    if isinstance(meta, dict) and meta.get("user_id"):
                        return str(meta["user_id"]).strip()
                except Exception:
                    pass

            # 3. Connected Remote Participant Identity (LiveKit JWT auth)
            if getattr(ctx, "room", None) and getattr(ctx.room, "remote_participants", None):
                for participant in ctx.room.remote_participants.values():
                    p_id = getattr(participant, "identity", None)
                    if p_id and p_id.strip() and not str(p_id).startswith("agent-") and not str(p_id).startswith("fieldmate"):
                        return str(p_id).strip()

            # 4. Room Name / Job ID fallback
            if getattr(ctx, "room", None) and getattr(ctx.room, "name", None):
                return str(ctx.room.name).strip()

            if getattr(ctx, "job", None) and getattr(ctx.job, "id", None):
                return str(ctx.job.id).strip()

            return "default_user"

        user_identity = resolve_user_identity()

        brain_runtime = build_brain_runtime(
            session_id=str(getattr(ctx.room, "name", "fieldmate_dev_room")),
            owner_id=user_identity,
            repository=repository,
            retrieval=retrieval,
        )
        canonical_brain = brain_runtime.brain

        brain = VoiceBrain(
            repository=repository,
            retrieval=retrieval,
            groq=groq,
            owner_id=user_identity,
            canonical_brain=canonical_brain,
        )

        # Prime independent remote resources concurrently.
        # This moves connection/setup work out of the first
        # conversational turn without serializing Qdrant and
        # Groq startup.
        qdrant_started = time.perf_counter()
        await asyncio.gather(
            repository.ensure_collection(),
            semantic_cache.ensure_collection(),
            brain.warm(),
        )
        logger.info(
            ">>> QDRANT + GROQ READY %.1f ms",
            (time.perf_counter() - qdrant_started) * 1000,
        )

        parallel_router = ParallelTurnRouter(
            qdrant_timeout_ms=RETRIEVAL_TIMEOUT_MS,
        )

        # ----------------------------------------------------
        # FLUX
        # ----------------------------------------------------

        stt = deepgram.STTv2(
            model=FLUX_MODEL,
            eager_eot_threshold=FLUX_EAGER_EOT,
            eot_threshold=FLUX_EOT_THRESHOLD,
            eot_timeout_ms=FLUX_EOT_TIMEOUT_MS,
        )

        logger.info(
            ">>> FLUX READY "
            "model=%s "
            "eager=%.2f "
            "eot=%.2f "
            "timeout=%dms",
            FLUX_MODEL,
            FLUX_EAGER_EOT,
            FLUX_EOT_THRESHOLD,
            FLUX_EOT_TIMEOUT_MS,
        )

        # ----------------------------------------------------
        # TTS (RIME / DEEPGRAM AURA)
        # ----------------------------------------------------

        tts_provider = os.getenv("FIELDMATE_TTS_PROVIDER", "rime").lower().strip()
        if tts_provider == "deepgram":
            tts_model = os.getenv("DEEPGRAM_TTS_MODEL", "aura-asteria-en")
            tts = deepgram.TTS(
                model=tts_model,
                sample_rate=RIME_SAMPLE_RATE,
            )
            logger.info(
                ">>> DEEPGRAM AURA TTS READY "
                "model=%s "
                "rate=%d",
                tts_model,
                RIME_SAMPLE_RATE,
            )
        else:
            tts = rime.TTS(
                model=RIME_MODEL,
                speaker=RIME_SPEAKER,
                sample_rate=RIME_SAMPLE_RATE,
                use_websocket=True,
            )
            logger.info(
                ">>> RIME READY "
                "model=%s "
                "speaker=%s "
                "rate=%d "
                "websocket=true",
                RIME_MODEL,
                RIME_SPEAKER,
                RIME_SAMPLE_RATE,
            )

        # ----------------------------------------------------
        # SESSION
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # There is deliberately NO `llm=...`.
        #
        # Groq is owned by VoiceBrain.
        #
        # AgentSession is used for:
        #
        #   LiveKit transport
        #   STT
        #   Flux endpointing
        #   interruption detection
        #   TTS
        #   speech scheduling
        #
        # This prevents LiveKit from independently generating
        # a second answer.
        #

        session = AgentSession(
            stt=stt,

            tts=tts,

            turn_handling=(
                TurnHandlingOptions(
                    turn_detection="stt",

                    endpointing={
                        "mode": "fixed",
                        "min_delay": 0.0,
                        "max_delay": 3.0,
                    },

                    interruption={
                        "enabled": True,
                        "mode": "vad",

                        # Aggressive enough for technician
                        # barge-in without requiring a long
                        # utterance.
                        "min_duration": 0.15,

                        "min_words": 0,

                        "false_interruption_timeout": 1.0,

                        "resume_false_interruption": True,

                        "discard_audio_if_uninterruptible": True,
                    },

                    # We cannot use LiveKit's built-in
                    # preemptive LLM generation because our
                    # LLM is deliberately outside AgentSession.
                    #
                    # Our own speculation happens through the
                    # Qdrant orchestrator.
                    preemptive_generation={
                        "enabled": False,
                    },
                )
            ),

            # Don't insert artificial delays between responses.
            min_consecutive_speech_delay=0.0,

            # Give AEC a short warmup period.
            #
            # This avoids immediately treating the agent's own
            # freshly-started audio as technician speech.
            aec_warmup_duration=0.5,
        )

        # ----------------------------------------------------
        # LOCAL TURN STATE
        # ----------------------------------------------------

        stabilizer = QueryStabilizer(
            min_chars=(
                PREFETCH_MIN_CHARS
            ),
            min_interval_ms=(
                PREFETCH_MIN_INTERVAL_MS
            ),
        )

        generations = (
            GenerationController()
        )

        history = ConversationHistory(
            MAX_HISTORY_MESSAGES
        )

        # ----------------------------------------------------
        # PREFETCH TASK REGISTRY
        # ----------------------------------------------------
        #
        # These tasks are only the local scheduling tasks.
        #
        # The actual Qdrant prefetch lifecycle belongs to the
        # RetrievalOrchestrator.
        #

        def schedule_prefetch(
            transcript: str,
        ) -> None:

            query = stabilizer.consider(
                transcript
            )

            if query is None:
                return

            task = asyncio.create_task(
                brain.speculate(
                    query
                )
            )

            prefetch_tasks.add(
                task
            )

            task.add_done_callback(
                prefetch_tasks.discard
            )

            logger.debug(
                ">>> SPECULATIVE QUERY: %s",
                query,
            )

        # ----------------------------------------------------
        # TURN PROCESSOR
        # ----------------------------------------------------

        async def process_final_turn(
            transcript: str,
        ) -> None:

            transcript = normalize_text(
                transcript
            )

            nonlocal user_identity
            env_user = os.getenv("FIELDMATE_USER_ID")
            if env_user and env_user.strip():
                user_identity = env_user.strip()
            elif ctx.room and getattr(ctx.room, "remote_participants", None):
                for participant in ctx.room.remote_participants.values():
                    p_id = getattr(participant, "identity", None)
                    if p_id and p_id.strip() and not str(p_id).startswith("agent-") and not str(p_id).startswith("fieldmate"):
                        user_identity = str(p_id).strip()
                        brain.owner_id = user_identity
                        if hasattr(brain, "canonical_brain") and brain.canonical_brain:
                            brain.canonical_brain.state.session.owner_id = user_identity
                        break

            # A finalized turn establishes a new speculative
            # baseline. The completed result itself remains in
            # RetrievalOrchestrator's convergence cache.
            stabilizer.reset()

            generation = (
                await generations.begin()
            )

            started = (
                time.perf_counter()
            )

            logger.info(
                ">>> TURN %d START",
                generation,
            )

            logger.debug(
                ">>> TURN TEXT: %s",
                transcript,
            )

            history.add_user(
                transcript
            )

            try:

                # ------------------------------------------------
                # PARALLEL GROQ + QDRANT
                # ------------------------------------------------
                #
                # Every finalized transcript enters the same
                # parallel path.
                #
                # Groq starts speculative generation immediately.
                #
                # Qdrant independently determines whether the
                # utterance requires technical grounding.
                #
                # Qdrant:
                #
                #   relevant
                #       -> speculative Groq is discarded
                #       -> grounded Groq is generated
                #
                #   irrelevant
                #       -> speculative Groq is released
                #
                #   timeout / failure
                #       -> speculative Groq is released
                #
                # This means general conversation does NOT need
                # a hardcoded "non-technical" classifier.
                # ------------------------------------------------

                if not generations.is_current(
                    generation
                ):
                    logger.debug(
                        ">>> STALE TURN DROPPED "
                        "generation=%d",
                        generation,
                    )
                    return

                response_parts: list[str] = []

                parallel_router = (
                    ParallelTurnRouter(
                        qdrant_timeout_ms=RETRIEVAL_TIMEOUT_MS,
                    )
                )

                warm_started = False

                async def response_stream() -> AsyncIterable[str]:
                    nonlocal warm_started
                    logger.info(">>> RESPONSE STREAM ENTERED")
                    try:
                        buffer = ""
                        async for chunk in brain.stream_parallel_response(
                            user_text=transcript,
                            history=history.snapshot(),
                            router=parallel_router,
                            generation_is_current=(
                                lambda: (
                                    generations.is_current(
                                        generation
                                    )
                                )
                            ),
                        ):
                            logger.info(">>> RESPONSE STREAM RECEIVED CHUNK")
                            if not generations.is_current(
                                generation
                            ):
                                logger.info(
                                    ">>> TURN %d DISCARDED "
                                    "(STALE GENERATION)",
                                    generation,
                                )
                                return

                            response_parts.append(
                                chunk
                            )

                            if (
                                not warm_started
                                and sum(
                                    len(part)
                                    for part in response_parts
                                ) >= 48
                            ):
                                warm_started = True
                                warm_text = "".join(
                                    response_parts
                                )
                                warm_task = asyncio.create_task(
                                    brain.warm_next(
                                        warm_text
                                    )
                                )
                                prefetch_tasks.add(
                                    warm_task
                                )
                                warm_task.add_done_callback(
                                    prefetch_tasks.discard
                                )
                                logger.debug(
                                    ">>> NEXT-TURN WARM STARTED"
                                )

                            if chunk:
                                yield chunk

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.exception(f">>> RESPONSE STREAM ERROR: {e}")
                        raise
                    finally:
                        response_text = "".join(response_parts).strip()
                        if response_text and generations.is_current(generation):
                            history.add_assistant(response_text)
                            if retrieval and getattr(retrieval, "semantic_cache", None):
                                if hasattr(retrieval.semantic_cache, "store_background"):
                                    retrieval.semantic_cache.store_background(
                                        transcript,
                                        DiagnosticContext(),
                                        response_text=response_text,
                                        owner_id=user_identity,
                                    )
                                else:
                                    asyncio.create_task(
                                        retrieval.semantic_cache.store(
                                            transcript,
                                            DiagnosticContext(),
                                            response_text=response_text,
                                            owner_id=user_identity,
                                        )
                                    )
                            if hasattr(brain, "canonical_brain") and brain.canonical_brain:
                                asyncio.create_task(
                                    brain.canonical_brain.process(
                                        transcript,
                                        technical=is_technical(transcript),
                                    )
                                )
                            total_elapsed = (
                                time.perf_counter()
                                - started
                            ) * 1000
                            logger.info(
                                ">>> TURN %d COMPLETE %.1fms: '%s...'",
                                generation,
                                total_elapsed,
                                response_text[:40],
                            )

                speech_handle = (
                    session.say(
                        response_stream(),
                        allow_interruptions=False,
                    )
                )

                generations.active_speech = (
                    speech_handle
                )

                await speech_handle

            except asyncio.CancelledError:

                logger.info(
                    ">>> TURN %d CANCELLED",
                    generation,
                )

                raise

            except Exception:

                logger.exception(
                    ">>> TURN %d FAILED",
                    generation,
                )

                # Only speak a fallback if this generation
                # is still current.
                #
                # An old turn must never inject stale speech
                # into the conversation.

                if not generations.is_current(
                    generation
                ):
                    return

                with suppress(
                    Exception
                ):

                    await session.say(
                        "I couldn't complete that check. Please repeat the issue.",
                        allow_interruptions=False,
                    )

            finally:

                generations.active_speech = None

        # ----------------------------------------------------
        # TRANSCRIPT EVENT
        # ----------------------------------------------------

        @session.on(
            "user_input_transcribed"
        )
        def on_user_input_transcribed(
            event,
        ):

            transcript = normalize_text(
                getattr(
                    event,
                    "transcript",
                    "",
                )
            )

            if not transcript:
                return

            is_final = bool(
                getattr(
                    event,
                    "is_final",
                    False,
                )
            )

            logger.info(
                ">>> USER TRANSCRIPT: '%s' (is_final=%s)",
                transcript,
                is_final,
            )

            # Interrupt active speech only on genuine recognized speech from user
            if len(transcript.strip()) >= 3 and generations.active_speech is not None:
                logger.info(">>> GENUINE USER SPEECH DETECTED: INTERRUPTING ACTIVE SPEECH")
                generations.cancel_active()

            if not is_final:

                # Speculation only.
                #
                # Never wait for this.
                schedule_prefetch(
                    transcript
                )

                return

            # Final transcript.
            #
            # Any final turn invalidates the previous response
            # generation if it is still running.
            #
            # process_final_turn() will acquire the next
            # generation safely.
            #

            turn_task = asyncio.create_task(
                process_final_turn(
                    transcript
                )
            )
            generations.attach_task(turn_task)

        # ----------------------------------------------------
        # USER STATE / FAST INTERRUPTION
        # ----------------------------------------------------
        #
        # LiveKit's bundled VAD is responsible for detecting
        # interruption.
        #
        # We additionally invalidate our own Groq stream as soon
        # as the user enters the speaking state.
        #

        @session.on(
            "user_state_changed"
        )
        def on_user_state_changed(
            event,
        ):

            new_state = getattr(
                event,
                "new_state",
                None,
            )

            logger.debug(
                ">>> USER STATE CHANGED: %s",
                new_state,
            )

            if str(new_state) == "speaking" and generations.active_speech is not None:
                logger.info(">>> USER BARGE-IN (user speaking): INTERRUPTING ACTIVE SPEECH")
                generations.cancel_active()

        # ----------------------------------------------------
        # OVERLAPPING SPEECH
        # ----------------------------------------------------

        @session.on(
            "overlapping_speech"
        )
        def on_overlapping_speech(
            event,
        ):

            is_interruption = bool(
                getattr(
                    event,
                    "is_interruption",
                    False,
                )
            )

            logger.info(
                ">>> OVERLAPPING SPEECH (is_interruption=%s): INTERRUPTING ACTIVE SPEECH",
                is_interruption,
            )
            generations.cancel_active()

        # ----------------------------------------------------
        # AGENT STATE
        # ----------------------------------------------------

        @session.on(
            "agent_state_changed"
        )
        def on_agent_state_changed(
            event,
        ):

            logger.debug(
                ">>> AGENT STATE %s -> %s",
                getattr(
                    event,
                    "old_state",
                    "?",
                ),
                getattr(
                    event,
                    "new_state",
                    "?",
                ),
            )

        # ----------------------------------------------------
        # SESSION ERROR
        # ----------------------------------------------------

        @session.on(
            "error"
        )
        def on_session_error(
            event,
        ):

            error = getattr(
                event,
                "error",
                event,
            )

            recoverable = getattr(
                error,
                "recoverable",
                True,
            )

            if recoverable:

                logger.warning(
                    ">>> RECOVERABLE SESSION ERROR: %s",
                    error,
                )

            else:

                logger.error(
                    ">>> UNRECOVERABLE SESSION ERROR: %s",
                    error,
                )

        # ----------------------------------------------------
        # CONNECT LIVEKIT ROOM & WAIT FOR PARTICIPANT
        # ----------------------------------------------------

        await ctx.connect()

        logger.info(
            ">>> LIVEKIT CONNECTED %.1f ms",
            (
                time.perf_counter()
                - started_at
            ) * 1000,
        )

        # ----------------------------------------------------
        # START SESSION
        # ----------------------------------------------------

        await session.start(
            agent=FieldMate(),
            room=ctx.room,
        )

        logger.info(
            ">>> FIELDMATE READY %.1fms",
            (
                time.perf_counter()
                - started_at
            ) * 1000,
        )

        # ----------------------------------------------------
        # KEEP SESSION ALIVE
        # ----------------------------------------------------

        await asyncio.Event().wait()

    finally:

        logger.info(
            ">>> FIELDMATE SHUTDOWN"
        )

        # ----------------------------------------------------
        # CANCEL LOCAL PREFETCH TASKS
        # ----------------------------------------------------

        for task in list(
            prefetch_tasks
        ):

            if not task.done():
                task.cancel()

        if prefetch_tasks:

            await asyncio.gather(
                *prefetch_tasks,
                return_exceptions=True,
            )

        # ----------------------------------------------------
        # CLOSE VOICE BRAIN
        # ----------------------------------------------------

        if 'brain' in locals():
            with suppress(Exception):
                await brain.close()

        # ----------------------------------------------------
        # CLOSE QDRANT
        # ----------------------------------------------------

        if repository is not None:

            with suppress(
                Exception
            ):
                await repository.close()

        logger.info(
            ">>> FIELDMATE SHUTDOWN COMPLETE"
        )


# ============================================================
# CLI ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    cli.run_app(server)