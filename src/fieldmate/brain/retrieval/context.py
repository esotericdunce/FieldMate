from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .evidence import Evidence, normalize_evidence
from fieldmate.brain.state.models import DiagnosticState


# ============================================================
# RETRIEVED MEMORY
# ============================================================

@dataclass(frozen=True)
class RetrievedMemory:
    """
    Backwards-compatible representation of retrieved memory.

    DiagnosticContext exposes these simplified memory objects
    to the reasoning layer while the richer Evidence objects
    remain available internally to retrieval/context intelligence.
    """

    memory_id: str
    memory_type: str
    content: str
    score: float
    confidence: float

    equipment_model: str | None

    fault_codes: list[str]

    # Additional retrieval metadata.
    #
    # These fields are optional from the perspective of older
    # callers, but allow the reasoning layer to understand where
    # evidence came from and whether it supports or contradicts
    # the current diagnostic state.
    source: str = "qdrant"
    verification_status: str = "unverified"
    relation: str = "neutral"
    provenance: str = "qdrant_dense"
    case_reference: str | None = None
    retrieval_mode: str = "dense"


# ============================================================
# DIAGNOSTIC CONTEXT
# ============================================================

@dataclass(frozen=True)
class DiagnosticContext:
    """
    Stable retrieval result passed toward the reasoning layer.

    The context is intentionally immutable once constructed.

    Retrieval may internally perform:

        Qdrant
          ↓
        Evidence
          ↓
        ranking
          ↓
        budgeting
          ↓
        DiagnosticContext

    Brain/reasoning code should consume this object rather than
    depending directly on Qdrant types.
    """

    memories: tuple[RetrievedMemory, ...] = ()

    # Rich evidence is retained alongside the compatibility
    # memory representation.
    #
    # This allows future reasoning code to distinguish:
    #
    #   supporting evidence
    #   contradicting evidence
    #   neutral evidence
    #
    # without breaking the existing memories contract.
    evidence: tuple[Evidence, ...] = ()

    # Useful retrieval metadata.
    query: str = ""

    retrieval_mode: str = "dense"

    # Approximate context size after retrieval/budgeting.
    total_tokens_approx: int = 0

    # Whether retrieval was completed from speculative prefetch.
    prefetched: bool = False

    # Whether retrieval exceeded the hot-path timeout.
    timed_out: bool = False


# ============================================================
# QDRANT POINT NORMALIZATION
# ============================================================

def _point_id(point: Any) -> str:
    """
    Safely extract a point ID from either a Qdrant point or
    a dictionary test double.
    """

    if isinstance(point, dict):
        return str(
            point.get(
                "id",
                "unknown",
            )
        )

    return str(
        getattr(
            point,
            "id",
            "unknown",
        )
    )


def _point_payload(
    point: Any,
) -> dict[str, Any]:
    """
    Safely extract a Qdrant payload.

    Supports both:

        ScoredPoint.payload

    and:

        {"payload": {...}}
    """

    if isinstance(point, dict):

        payload = point.get(
            "payload",
            point,
        )

        return (
            payload
            if isinstance(
                payload,
                dict,
            )
            else {}
        )

    payload = getattr(
        point,
        "payload",
        None,
    )

    return (
        payload
        if isinstance(
            payload,
            dict,
        )
        else {}
    )


def _point_score(
    point: Any,
) -> float:
    """
    Safely extract a retrieval score.
    """

    if isinstance(point, dict):
        raw_score = point.get(
            "score",
            0.0,
        )
    else:
        raw_score = getattr(
            point,
            "score",
            0.0,
        )

    try:
        score = float(
            raw_score
            if raw_score is not None
            else 0.0
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if score != score:
        return 0.0

    return score


# ============================================================
# EVIDENCE → RETRIEVED MEMORY
# ============================================================

def _memory_from_evidence(
    evidence: Evidence,
) -> RetrievedMemory:
    """
    Convert rich Evidence into the stable RetrievedMemory
    representation expected by the reasoning layer.
    """

    return RetrievedMemory(
        memory_id=evidence.memory_id,

        memory_type=evidence.memory_type,

        content=evidence.content,

        score=evidence.relevance_score,

        confidence=evidence.confidence,

        equipment_model=evidence.equipment_model,

        fault_codes=list(
            evidence.fault_codes
        ),

        source=evidence.source,

        verification_status=(
            evidence.verification_status
        ),

        relation=evidence.relation,

        provenance=evidence.provenance,

        case_reference=evidence.case_reference,

        retrieval_mode=evidence.retrieval_mode,
    )


# ============================================================
# DIRECT CONTEXT BUILDING
# ============================================================

def build_context(
    points: Sequence[Any],
    *,
    max_memories: int = 8,
    state: DiagnosticState | None = None,
    retrieval_mode: str = "dense",
    query: str = "",
) -> DiagnosticContext:
    """
    Convert raw Qdrant results into DiagnosticContext.

    This function remains intentionally lightweight.

    It performs:

        raw points
            ↓
        evidence normalization
            ↓
        basic deduplication
            ↓
        compatibility conversion

    More sophisticated ranking/budgeting belongs in
    ContextIntelligence.

    This separation keeps this function safe for direct use by
    tests and simple retrieval paths.
    """

    if max_memories <= 0:
        return DiagnosticContext(
            memories=(),
            evidence=(),
            query=query,
            retrieval_mode=retrieval_mode,
        )

    # --------------------------------------------------------
    # Normalize through the canonical Evidence layer.
    # --------------------------------------------------------

    evidence_items = normalize_evidence(
        points,
        state=state,
        retrieval_mode=retrieval_mode,
    )

    # --------------------------------------------------------
    # Deduplicate.
    #
    # Qdrant normally shouldn't return duplicate point IDs, but
    # hybrid retrieval/test doubles can make duplicates possible.
    #
    # Content deduplication also protects the LLM context from
    # receiving the same knowledge multiple times.
    # --------------------------------------------------------

    seen_memory_ids: set[str] = set()
    seen_content: set[str] = set()

    selected_evidence: list[Evidence] = []

    for item in evidence_items:

        if item.memory_id in seen_memory_ids:
            continue

        normalized_content = (
            " ".join(
                item.content
                .strip()
                .lower()
                .split()
            )
        )

        if not normalized_content:
            continue

        if normalized_content in seen_content:
            continue

        seen_memory_ids.add(
            item.memory_id
        )

        seen_content.add(
            normalized_content
        )

        selected_evidence.append(
            item
        )

        if (
            len(selected_evidence)
            >= max_memories
        ):
            break

    # --------------------------------------------------------
    # Compatibility memory representation.
    # --------------------------------------------------------

    memories = tuple(
        _memory_from_evidence(item)
        for item in selected_evidence
    )

    return DiagnosticContext(
        memories=memories,

        evidence=tuple(
            selected_evidence
        ),

        query=query,

        retrieval_mode=retrieval_mode,
    )


# ============================================================
# CONTEXT FROM EVIDENCE
# ============================================================

def context_from_evidence(
    evidence: Sequence[Evidence],
    *,
    query: str = "",
    retrieval_mode: str = "dense",
    max_memories: int = 8,
    total_tokens_approx: int = 0,
    prefetched: bool = False,
    timed_out: bool = False,
) -> DiagnosticContext:
    """
    Construct DiagnosticContext when ContextIntelligence has
    already selected the final evidence set.

    This is the preferred constructor for the production
    retrieval pipeline.
    """

    if max_memories <= 0:
        selected = ()
    else:
        selected = tuple(
            evidence[:max_memories]
        )

    memories = tuple(
        _memory_from_evidence(item)
        for item in selected
    )

    return DiagnosticContext(
        memories=memories,

        evidence=selected,

        query=query,

        retrieval_mode=retrieval_mode,

        total_tokens_approx=(
            max(
                0,
                int(
                    total_tokens_approx
                ),
            )
        ),

        prefetched=prefetched,

        timed_out=timed_out,
    )


# ============================================================
# EMPTY CONTEXT
# ============================================================

def empty_context(
    *,
    query: str = "",
    retrieval_mode: str = "dense",
    timed_out: bool = False,
) -> DiagnosticContext:
    """
    Explicit empty-context constructor.

    Useful when Qdrant is unavailable, times out, or simply has
    no matching memories.

    An empty context is a valid diagnostic state.

    The Brain must still be capable of reasoning from the current
    diagnostic state and conversation.
    """

    return DiagnosticContext(
        memories=(),
        evidence=(),
        query=query,
        retrieval_mode=retrieval_mode,
        total_tokens_approx=0,
        prefetched=False,
        timed_out=timed_out,
    )