from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ============================================================
# TIME
# ============================================================

def utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def _require_non_empty(value: str, field_name: str) -> str:
    """
    Normalize and validate a required textual field.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string"
        )

    value = value.strip()

    if not value:
        raise ValueError(
            f"{field_name} cannot be empty"
        )

    return value


def _validate_confidence(
    value: float,
    field_name: str = "confidence",
) -> float:
    """
    Confidence is always represented as a probability-like
    value in [0, 1].
    """
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be numeric"
        ) from exc

    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )

    return value


# ============================================================
# OBSERVATIONS
# ============================================================

class ObservationSource(str, Enum):
    """
    Origin of an observation.

    USER:
        Technician-reported information.

    SENSOR:
        Machine/device-generated information.

    DOCUMENT:
        Information obtained from authoritative documentation.

    MEMORY:
        Previously stored FieldMate knowledge.

    INFERENCE:
        Derived by the reasoning layer.

    StateEngine accepts these values but does not decide
    whether an observation is trustworthy.
    """

    USER = "user"
    SENSOR = "sensor"
    DOCUMENT = "document"
    MEMORY = "memory"
    INFERENCE = "inference"
    CAMERA_VISION = "camera_vision"


class ObservationStatus(str, Enum):
    REPORTED = "reported"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


@dataclass
class Observation:
    """
    A discrete diagnostic observation.

    The state layer records the observation; it does not
    determine whether the observation proves a diagnosis.
    """

    name: str
    value: Any

    source: ObservationSource

    status: ObservationStatus = (
        ObservationStatus.REPORTED
    )

    confidence: float = 1.0

    timestamp: datetime = field(
        default_factory=utc_now
    )

    notes: str | None = None

    def __post_init__(self) -> None:
        self.name = _require_non_empty(
            self.name,
            "Observation name",
        )

        self.confidence = _validate_confidence(
            self.confidence,
            "Observation confidence",
        )

        if self.notes is not None:
            self.notes = self.notes.strip() or None


# ============================================================
# MEASUREMENTS
# ============================================================

@dataclass
class Measurement:
    """
    A numeric diagnostic measurement.

    Expected ranges are optional because not every diagnostic
    measurement has a known valid range at collection time.
    """

    name: str
    value: float

    unit: str

    source: ObservationSource

    status: ObservationStatus = (
        ObservationStatus.REPORTED
    )

    min_expected: float | None = None
    max_expected: float | None = None

    timestamp: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        self.name = _require_non_empty(
            self.name,
            "Measurement name",
        )

        self.unit = _require_non_empty(
            self.unit,
            "Measurement unit",
        )

        try:
            self.value = float(self.value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Measurement value must be numeric"
            ) from exc

        if (
            self.min_expected is not None
            and self.max_expected is not None
            and self.min_expected > self.max_expected
        ):
            raise ValueError(
                "min_expected cannot exceed max_expected"
            )

        if self.min_expected is not None:
            self.min_expected = float(
                self.min_expected
            )

        if self.max_expected is not None:
            self.max_expected = float(
                self.max_expected
            )

    @property
    def is_out_of_range(self) -> bool:
        """
        Return whether the measurement is outside its known
        expected range.
        """

        if (
            self.min_expected is not None
            and self.value < self.min_expected
        ):
            return True

        if (
            self.max_expected is not None
            and self.value > self.max_expected
        ):
            return True

        return False


# ============================================================
# HYPOTHESES
# ============================================================

class HypothesisStatus(str, Enum):
    POSSIBLE = "possible"
    LIKELY = "likely"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class Hypothesis:
    """
    A diagnostic hypothesis.

    A hypothesis is not automatically a diagnosis.

    Confirmation must happen through an explicit state
    transition backed by evidence.
    """

    description: str

    status: HypothesisStatus = (
        HypothesisStatus.POSSIBLE
    )

    confidence: float = 0.5

    supporting_evidence: list[str] = field(
        default_factory=list
    )

    contradicting_evidence: list[str] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        self.description = _require_non_empty(
            self.description,
            "Hypothesis description",
        )

        self.confidence = _validate_confidence(
            self.confidence,
            "Hypothesis confidence",
        )

        self.supporting_evidence = (
            self._normalize_evidence_ids(
                self.supporting_evidence
            )
        )

        self.contradicting_evidence = (
            self._normalize_evidence_ids(
                self.contradicting_evidence
            )
        )

    @staticmethod
    def _normalize_evidence_ids(
        values: list[str],
    ) -> list[str]:
        if not values:
            return []

        result: list[str] = []

        for value in values:
            if not isinstance(value, str):
                continue

            value = value.strip()

            if value and value not in result:
                result.append(value)

        return result


# ============================================================
# DIAGNOSTIC TESTS
# ============================================================

class TestStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DiagnosticTest:
    """
    A diagnostic test performed against the equipment.

    Tests are historical facts once completed.
    """

    name: str

    status: TestStatus = (
        TestStatus.PENDING
    )

    result: str | None = None

    started_at: datetime | None = None

    completed_at: datetime | None = None

    notes: str | None = None

    def __post_init__(self) -> None:
        self.name = _require_non_empty(
            self.name,
            "Test name",
        )

        if self.result is not None:
            self.result = self.result.strip() or None

        if self.notes is not None:
            self.notes = self.notes.strip() or None

        if (
            self.completed_at is not None
            and self.started_at is not None
            and self.completed_at < self.started_at
        ):
            raise ValueError(
                "Test completion time cannot precede "
                "test start time"
            )


# ============================================================
# EQUIPMENT
# ============================================================

@dataclass
class EquipmentState:
    """
    Identity and diagnostic scope of the equipment currently
    being serviced.
    """

    manufacturer: str | None = None

    model: str | None = None

    serial_number: str | None = None

    equipment_family: str | None = None

    system: str | None = None

    subsystem: str | None = None

    component: str | None = None

    def __post_init__(self) -> None:
        self.manufacturer = self._normalize_optional(
            self.manufacturer
        )

        self.model = self._normalize_optional(
            self.model
        )

        self.serial_number = self._normalize_optional(
            self.serial_number
        )

        self.equipment_family = (
            self._normalize_optional(
                self.equipment_family
            )
        )

        self.system = self._normalize_optional(
            self.system
        )

        self.subsystem = self._normalize_optional(
            self.subsystem
        )

        self.component = self._normalize_optional(
            self.component
        )

    @staticmethod
    def _normalize_optional(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(
                "Equipment fields must be strings or None"
            )

        value = value.strip()

        return value or None


# ============================================================
# DIAGNOSTIC STATE
# ============================================================

@dataclass
class DiagnosticState:
    """
    Authoritative mutable state for the current diagnostic case.

    This is the core domain state.

    It contains facts, observations, tests, hypotheses and
    recommendations, but it does not perform reasoning itself.
    """

    equipment: EquipmentState = field(
        default_factory=EquipmentState
    )

    fault_codes: list[str] = field(
        default_factory=list
    )

    symptoms: list[Observation] = field(
        default_factory=list
    )

    observations: list[Observation] = field(
        default_factory=list
    )

    measurements: list[Measurement] = field(
        default_factory=list
    )

    tests: list[DiagnosticTest] = field(
        default_factory=list
    )

    hypotheses: list[Hypothesis] = field(
        default_factory=list
    )

    current_hypothesis: str | None = None

    next_recommended_action: str | None = None

    confirmed_resolution: str | None = None

    case_status: str = "open"

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        self.fault_codes = self._normalize_fault_codes(
            self.fault_codes
        )

        self.current_hypothesis = (
            self._normalize_optional_text(
                self.current_hypothesis
            )
        )

        self.next_recommended_action = (
            self._normalize_optional_text(
                self.next_recommended_action
            )
        )

        self.confirmed_resolution = (
            self._normalize_optional_text(
                self.confirmed_resolution
            )
        )

        self.case_status = (
            self._normalize_case_status(
                self.case_status
            )
        )

    @staticmethod
    def _normalize_fault_codes(
        values: list[str],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            if not isinstance(value, str):
                continue

            value = value.strip()

            if value and value not in result:
                result.append(value)

        return result

    @staticmethod
    def _normalize_optional_text(
        value: str | None,
    ) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError(
                "Diagnostic text fields must be "
                "strings or None"
            )

        value = value.strip()

        return value or None

    @staticmethod
    def _normalize_case_status(
        value: str,
    ) -> str:
        if not isinstance(value, str):
            raise ValueError(
                "case_status must be a string"
            )

        value = value.strip().lower()

        allowed = {
            "open",
            "resolved",
            "closed",
        }

        if value not in allowed:
            raise ValueError(
                "case_status must be one of: "
                "open, resolved, closed"
            )

        return value


# ============================================================
# CONVERSATION STATE
# ============================================================

@dataclass
class ConversationState:
    """
    Lightweight conversational history associated with the
    current diagnostic session.

    The diagnostic state remains authoritative; conversation
    history is contextual rather than permanent memory.
    """

    recent_user_messages: list[str] = field(
        default_factory=list
    )

    recent_assistant_messages: list[str] = field(
        default_factory=list
    )

    turn_id: int = 0

    # Prevent an indefinitely growing in-memory conversation.
    max_recent_messages: int = 20

    def __post_init__(self) -> None:
        if self.max_recent_messages <= 0:
            raise ValueError(
                "max_recent_messages must be positive"
            )

        self.recent_user_messages = (
            self._normalize_messages(
                self.recent_user_messages
            )
        )

        self.recent_assistant_messages = (
            self._normalize_messages(
                self.recent_assistant_messages
            )
        )

        self._trim()

    def add_user_message(
        self,
        message: str,
    ) -> None:
        message = _require_non_empty(
            message,
            "User message",
        )

        self.recent_user_messages.append(
            message
        )

        self._trim()

    def add_assistant_message(
        self,
        message: str,
    ) -> None:
        message = _require_non_empty(
            message,
            "Assistant message",
        )

        self.recent_assistant_messages.append(
            message
        )

        self._trim()

    def _trim(self) -> None:
        """
        Keep only the most recent conversational messages.
        """

        if len(self.recent_user_messages) > (
            self.max_recent_messages
        ):
            del self.recent_user_messages[
                :-
                self.max_recent_messages
            ]

        if len(self.recent_assistant_messages) > (
            self.max_recent_messages
        ):
            del self.recent_assistant_messages[
                :-
                self.max_recent_messages
            ]

    @staticmethod
    def _normalize_messages(
        values: list[str],
    ) -> list[str]:
        result: list[str] = []

        for value in values:
            if not isinstance(value, str):
                continue

            value = value.strip()

            if value:
                result.append(value)

        return result


# ============================================================
# SESSION
# ============================================================

@dataclass
class FieldMateSession:
    """
    Complete authoritative state for one FieldMate diagnostic
    session.

    StateEngine owns transitions over this object.
    """

    session_id: str
    owner_id: str | None = None

    diagnostic: DiagnosticState = field(
        default_factory=DiagnosticState
    )

    conversation: ConversationState = field(
        default_factory=ConversationState
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        self.session_id = _require_non_empty(
            self.session_id,
            "session_id",
        )
        if self.owner_id is not None and isinstance(self.owner_id, str):
            self.owner_id = self.owner_id.strip() or None