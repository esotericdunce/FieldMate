from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return the current UTC time."""
    return datetime.now(timezone.utc)


class MemoryType(str, Enum):
    """
    Long-lived knowledge categories.

    WORKING state intentionally does not exist here.
    Working diagnostic state belongs to StateEngine.
    """

    EPISODIC = "episodic"
    EQUIPMENT = "equipment"
    PROCEDURAL = "procedural"
    RESOLUTION = "resolution"
    PATTERN = "pattern"


class MemoryStatus(str, Enum):
    """
    Persistent-memory lifecycle.

    CANDIDATE:
        Newly extracted knowledge that has not yet accumulated
        enough independent evidence.

    VERIFIED:
        Memory supported by sufficient successful evidence.

    DEPRECATED:
        Memory retained for provenance/audit but excluded from
        normal retrieval.
    """

    CANDIDATE = "candidate"
    VERIFIED = "verified"
    DEPRECATED = "deprecated"


class EvidenceType(str, Enum):
    """
    Origin of evidence supporting or contradicting a memory.
    """

    MANUAL = "manual"
    TECHNICIAN = "technician"
    OBSERVATION = "observation"
    MEASUREMENT = "measurement"
    TEST = "test"
    RESOLUTION = "resolution"
    CASE = "case"
    SYSTEM = "system"


@dataclass(frozen=True)
class MemoryEvidence:
    """
    Immutable provenance attached to a memory.

    Every piece of evidence must be independently identifiable.
    """

    evidence_type: EvidenceType
    reference_id: str
    description: str

    confidence: float = 1.0

    created_at: datetime = field(
        default_factory=utc_now
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Evidence confidence must be between 0 and 1."
            )

        if not self.reference_id.strip():
            raise ValueError(
                "Evidence reference_id cannot be empty."
            )

        if not self.description.strip():
            raise ValueError(
                "Evidence description cannot be empty."
            )


@dataclass
class MemoryRecord:
    """
    Canonical application-level representation of persistent
    FieldMate memory.

    This object intentionally contains no Qdrant-specific fields.

    Qdrant is a persistence/retrieval implementation detail.
    """

    memory_type: MemoryType

    content: str

    status: MemoryStatus = MemoryStatus.CANDIDATE

    confidence: float = 0.5

    memory_id: str = field(
        default_factory=lambda: str(uuid4())
    )

    created_at: datetime = field(
        default_factory=utc_now
    )

    updated_at: datetime = field(
        default_factory=utc_now
    )

    # =========================================================
    # DOMAIN ASSOCIATIONS
    # =========================================================

    equipment_model: str | None = None
    equipment_serial: str | None = None
    equipment_family: str | None = None

    system: str | None = None
    subsystem: str | None = None
    component: str | None = None

    fault_codes: list[str] = field(
        default_factory=list
    )

    # =========================================================
    # PROVENANCE
    # =========================================================

    source_ids: list[str] = field(
        default_factory=list
    )

    evidence: list[MemoryEvidence] = field(
        default_factory=list
    )

    # =========================================================
    # EVOLUTION
    # =========================================================

    observation_count: int = 0

    successful_resolution_count: int = 0

    contradiction_count: int = 0

    last_confirmed_at: datetime | None = None

    supersedes_memory_id: str | None = None

    # =========================================================
    # METADATA
    # =========================================================

    tags: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # =========================================================
    # VALIDATION
    # =========================================================

    def __post_init__(self) -> None:

        if not self.content.strip():
            raise ValueError(
                "Memory content cannot be empty."
            )

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                "Memory confidence must be between 0 and 1."
            )

        if self.observation_count < 0:
            raise ValueError(
                "observation_count cannot be negative."
            )

        if self.successful_resolution_count < 0:
            raise ValueError(
                "successful_resolution_count cannot be negative."
            )

        if self.contradiction_count < 0:
            raise ValueError(
                "contradiction_count cannot be negative."
            )

        self.fault_codes = self._unique_clean(
            self.fault_codes
        )

        self.source_ids = self._unique_clean(
            self.source_ids
        )

        self.tags = self._unique_clean(
            self.tags
        )

        # Evidence is de-duplicated by its logical reference.
        #
        # This is important because the same case/evidence may
        # pass through the memory pipeline more than once.
        unique_evidence: list[MemoryEvidence] = []
        seen_evidence: set[str] = set()

        for item in self.evidence:

            if item.reference_id in seen_evidence:
                continue

            seen_evidence.add(
                item.reference_id
            )

            unique_evidence.append(item)

        self.evidence = unique_evidence

    # =========================================================
    # INTERNAL HELPERS
    # =========================================================

    @staticmethod
    def _unique_clean(
        values: list[str],
    ) -> list[str]:

        result: list[str] = []
        seen: set[str] = set()

        for value in values:

            cleaned = value.strip()

            if not cleaned:
                continue

            if cleaned in seen:
                continue

            seen.add(cleaned)
            result.append(cleaned)

        return result

    # =========================================================
    # EVIDENCE
    # =========================================================

    def has_evidence(
        self,
        reference_id: str,
    ) -> bool:
        """
        Return whether evidence from reference_id is already
        attached to this memory.
        """

        return any(
            item.reference_id == reference_id
            for item in self.evidence
        )

    def record_supporting_evidence(
        self,
        evidence: MemoryEvidence,
    ) -> bool:
        """
        Record supporting evidence.

        Returns:
            True  -> evidence was newly recorded
            False -> evidence already existed

        Duplicate evidence does not increase observation count.
        """

        if self.has_evidence(
            evidence.reference_id
        ):
            return False

        self.evidence.append(
            evidence
        )

        self.observation_count += 1

        self.updated_at = utc_now()

        return True

    def record_contradiction(
        self,
        evidence: MemoryEvidence | None = None,
    ) -> bool:
        """
        Record contradictory evidence.

        Contradictory evidence remains preserved rather than
        destroying the original memory.

        Returns whether a new evidence item was recorded.
        """

        if evidence is not None:

            if self.has_evidence(
                evidence.reference_id
            ):
                return False

            self.evidence.append(
                evidence
            )

        self.contradiction_count += 1

        self.updated_at = utc_now()

        return True

    # =========================================================
    # RESOLUTION
    # =========================================================

    def record_successful_resolution(
        self,
    ) -> None:
        """
        Record a confirmed successful resolution.
        """

        self.successful_resolution_count += 1

        self.last_confirmed_at = utc_now()

        self.updated_at = utc_now()

    # =========================================================
    # LIFECYCLE
    # =========================================================

    def verify(self) -> None:
        """
        Promote the memory to verified.
        """

        self.status = MemoryStatus.VERIFIED

        self.updated_at = utc_now()

    def deprecate(
        self,
        superseding_memory_id: str | None = None,
    ) -> None:
        """
        Retire memory while preserving its provenance.
        """

        self.status = MemoryStatus.DEPRECATED

        self.supersedes_memory_id = (
            superseding_memory_id
        )

        self.updated_at = utc_now()