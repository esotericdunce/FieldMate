from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any
from uuid import uuid4


# ============================================================
# TIME
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================
# IMMUTABILITY
# ============================================================

def freeze(value: Any) -> Any:
    """
    Recursively freeze event payload data.

    DomainEvent itself is frozen, but a normal dict/list inside
    a frozen dataclass would still be mutable.

    Converting nested containers prevents callers from doing:

        event.payload["x"] = ...

    after the event has been created.

    This is important because event equality is also used for
    idempotency by StateEngine.
    """

    if isinstance(value, dict):
        return MappingProxyType(
            {
                key: freeze(item)
                for key, item in value.items()
            }
        )

    if isinstance(value, list):
        return tuple(
            freeze(item)
            for item in value
        )

    if isinstance(value, set):
        return frozenset(
            freeze(item)
            for item in value
        )

    if isinstance(value, tuple):
        return tuple(
            freeze(item)
            for item in value
        )

    return value


# ============================================================
# EVENT TYPES
# ============================================================

class EventType(str, Enum):
    """
    Complete vocabulary of state transitions understood by
    FieldMate's StateEngine.

    External systems should communicate with the domain state
    through these events rather than mutating FieldMateSession
    directly.
    """

    SESSION_STARTED = "session_started"

    USER_MESSAGE = "user_message"

    EQUIPMENT_IDENTIFIED = "equipment_identified"

    FAULT_IDENTIFIED = "fault_identified"

    SYMPTOM_RECORDED = "symptom_recorded"

    OBSERVATION_RECORDED = "observation_recorded"

    MEASUREMENT_RECORDED = "measurement_recorded"

    TEST_STARTED = "test_started"

    TEST_COMPLETED = "test_completed"

    HYPOTHESIS_CREATED = "hypothesis_created"

    HYPOTHESIS_UPDATED = "hypothesis_updated"

    RECOMMENDATION_CREATED = "recommendation_created"

    PROCEDURE_RETRIEVED = "procedure_retrieved"

    RESOLUTION_PROPOSED = "resolution_proposed"

    RESOLUTION_CONFIRMED = "resolution_confirmed"

    CASE_CLOSED = "case_closed"


# ============================================================
# DOMAIN EVENT
# ============================================================

@dataclass(frozen=True)
class DomainEvent:
    """
    Immutable command describing one attempted domain-state
    transition.

    The event contains no implementation-specific objects from
    LiveKit, Qdrant, Groq, Deepgram, etc.

    Those systems produce information; the domain receives it as
    explicit events.

    ------------------------------------------------------------
    TURN ID
    ------------------------------------------------------------

    Identifies the conversational turn that produced the event.

    StateEngine rejects events belonging to an older turn.

    ------------------------------------------------------------
    GENERATION ID
    ------------------------------------------------------------

    Identifies the reasoning generation that produced the event.

    This matters for speculative/preemptive generation:

        generation 4
             |
        user continues
             |
        generation 5
             |
        generation 4 events are stale

    ------------------------------------------------------------
    EVENT ID
    ------------------------------------------------------------

    Provides idempotency.

    Replaying the exact same event is harmless.

    Reusing an event ID for different contents is rejected.
    """

    session_id: str

    event_type: EventType

    payload: dict[str, Any] = field(
        default_factory=dict
    )

    turn_id: int | None = None

    generation_id: int = 0

    event_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    timestamp: datetime = field(
        default_factory=utc_now
    )

    source: str = "system"

    def __post_init__(self) -> None:
        # ------------------------------------------------------
        # SESSION
        # ------------------------------------------------------

        if not isinstance(
            self.session_id,
            str,
        ):
            raise ValueError(
                "session_id must be a string"
            )

        session_id = self.session_id.strip()

        if not session_id:
            raise ValueError(
                "session_id cannot be empty"
            )

        object.__setattr__(
            self,
            "session_id",
            session_id,
        )

        # ------------------------------------------------------
        # EVENT ID
        # ------------------------------------------------------

        if not isinstance(
            self.event_id,
            str,
        ):
            raise ValueError(
                "event_id must be a string"
            )

        event_id = self.event_id.strip()

        if not event_id:
            raise ValueError(
                "event_id cannot be empty"
            )

        object.__setattr__(
            self,
            "event_id",
            event_id,
        )

        # ------------------------------------------------------
        # SOURCE
        # ------------------------------------------------------

        if not isinstance(
            self.source,
            str,
        ):
            raise ValueError(
                "event source must be a string"
            )

        source = self.source.strip()

        if not source:
            raise ValueError(
                "event source cannot be empty"
            )

        object.__setattr__(
            self,
            "source",
            source,
        )

        # ------------------------------------------------------
        # TURN
        # ------------------------------------------------------

        if self.turn_id is not None:
            if not isinstance(
                self.turn_id,
                int,
            ):
                raise ValueError(
                    "turn_id must be an integer or None"
                )

            if self.turn_id < 0:
                raise ValueError(
                    "turn_id cannot be negative"
                )

        # ------------------------------------------------------
        # GENERATION
        # ------------------------------------------------------

        if not isinstance(
            self.generation_id,
            int,
        ):
            raise ValueError(
                "generation_id must be an integer"
            )

        if self.generation_id < 0:
            raise ValueError(
                "generation_id cannot be negative"
            )

        # ------------------------------------------------------
        # PAYLOAD
        # ------------------------------------------------------

        if not isinstance(
            self.payload,
            dict,
        ):
            raise ValueError(
                "event payload must be a dictionary"
            )

        object.__setattr__(
            self,
            "payload",
            freeze(self.payload),
        )


__all__ = [
    "utc_now",
    "freeze",
    "EventType",
    "DomainEvent",
]