from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from fieldmate.brain.retrieval.evidence import Evidence


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class DiagnosticContext:
    """
    Canonical context passed from retrieval into reasoning.

    Retrieval is responsible for producing Evidence objects.

    Brain/reasoning consumes this representation and does not
    know anything about Qdrant points or vector-search details.
    """

    evidence: tuple[Evidence, ...] = ()

    supporting: tuple[Evidence, ...] = ()

    contradicting: tuple[Evidence, ...] = ()

    neutral: tuple[Evidence, ...] = ()

    procedures: tuple[Evidence, ...] = ()

    past_cases: tuple[Evidence, ...] = ()

    resolutions: tuple[Evidence, ...] = ()

    token_budget: int = 4000

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)

    @property
    def has_contradictions(self) -> bool:
        return bool(self.contradicting)


@dataclass(frozen=True, slots=True)
class DiagnosticDecision:
    """
    Structured output produced by the reasoning layer.

    The reasoning model may propose these values.

    It does NOT directly mutate canonical state.

    Brain translates accepted fields into DomainEvents and
    StateEngine validates/applies them.
    """

    response: str

    hypothesis: str | None = None

    confidence: float | None = None

    next_action: str | None = None

    clarification_needed: bool = False

    clarification_question: str | None = None

    evidence_ids: tuple[str, ...] = ()

    state_updates: tuple[dict[str, Any], ...] = ()

    resolution_proposed: str | None = None

    resolution_confirmed: str | None = None

    def __post_init__(self) -> None:
        if self.confidence is not None:
            if not 0.0 <= self.confidence <= 1.0:
                raise ValueError(
                    "Decision confidence must be "
                    "between 0 and 1"
                )

        if not self.response.strip():
            raise ValueError(
                "Diagnostic decision response "
                "cannot be empty"
            )


@dataclass(frozen=True, slots=True)
class BrainResult:
    """
    Result returned by Brain.process().

    This is the application-facing result.

    It contains the reasoning response plus enough execution
    metadata for the voice/application layer to understand
    what happened during the turn.
    """

    response: str

    decision: DiagnosticDecision | None

    context: DiagnosticContext

    retrieved: bool

    prefetched: bool

    timed_out: bool

    retrieval_latency_ms: float

    reasoning_latency_ms: float

    turn: int

    generation: int

    created_at: datetime = field(
        default_factory=utc_now
    )