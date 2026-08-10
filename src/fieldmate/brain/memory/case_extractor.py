from __future__ import annotations

from dataclasses import dataclass

from .identity import memory_identity
from .models import (
    EvidenceType,
    MemoryEvidence,
    MemoryRecord,
    MemoryStatus,
)


class MemoryPromotionError(Exception):
    """
    Raised when a memory lifecycle operation is invalid.
    """


@dataclass(frozen=True)
class MemoryEvolutionPolicy:
    """
    Controls how persistent memory evolves.

    The policy is intentionally conservative.

    A single observation creates a candidate.

    Repeated independent evidence increases confidence.

    Multiple successful resolutions can promote a memory
    to VERIFIED.

    Contradictions reduce confidence but do not automatically
    erase knowledge.
    """

    minimum_verification_cases: int = 2

    supporting_confidence_gain: float = 0.10

    successful_resolution_gain: float = 0.15

    contradiction_penalty: float = 0.12

    minimum_confidence: float = 0.05

    maximum_confidence: float = 0.99

    candidate_max_confidence: float = 0.60

    demotion_confidence_threshold: float = 0.40


class MemoryManager:
    """
    Owns FieldMate's persistent-memory lifecycle.

    Responsibilities:

        candidate creation
        evidence attachment
        confidence evolution
        verification
        contradiction handling
        explicit deprecation
        deterministic identity

    It does NOT:

        call Qdrant
        perform embeddings
        perform retrieval
        call an LLM
        mutate diagnostic StateEngine state
    """

    def __init__(
        self,
        policy: MemoryEvolutionPolicy | None = None,
    ) -> None:

        self.policy = (
            policy
            or MemoryEvolutionPolicy()
        )

    # =========================================================
    # IDENTITY
    # =========================================================

    @staticmethod
    def identity(
        memory: MemoryRecord,
    ) -> str:
        """
        Return the stable logical identity of a memory.
        """

        return memory_identity(
            memory
        )

    # =========================================================
    # CREATE CANDIDATE
    # =========================================================

    def create_candidate(
        self,
        memory: MemoryRecord,
    ) -> MemoryRecord:
        """
        Normalize a new memory into conservative candidate state.

        Important:

        The memory_id becomes deterministic here.

        This means repeated extraction of the same logical
        equipment/fault memory can converge onto the same
        persistent identity.
        """

        if not isinstance(
            memory,
            MemoryRecord,
        ):
            raise TypeError(
                "memory must be a MemoryRecord"
            )

        if (
            memory.status
            == MemoryStatus.DEPRECATED
        ):
            raise MemoryPromotionError(
                "A deprecated memory cannot be recreated "
                "as a candidate."
            )

        memory.status = (
            MemoryStatus.CANDIDATE
        )

        memory.confidence = max(
            self.policy.minimum_confidence,
            min(
                memory.confidence,
                self.policy.candidate_max_confidence,
            ),
        )

        memory.memory_id = (
            memory_identity(memory)
        )

        return memory

    # =========================================================
    # SUPPORTING EVIDENCE
    # =========================================================

    def add_supporting_evidence(
        self,
        memory: MemoryRecord,
        *,
        evidence_type: EvidenceType,
        reference_id: str,
        description: str,
        confidence: float = 1.0,
        confirmed_resolution: bool = False,
    ) -> MemoryRecord:
        """
        Add new supporting evidence.

        Duplicate reference IDs are ignored.

        This is critical for idempotency because the same
        diagnostic case may be persisted/replayed more than once.
        """

        self._validate_evidence_parameters(
            reference_id=reference_id,
            description=description,
            confidence=confidence,
        )

        if (
            memory.status
            == MemoryStatus.DEPRECATED
        ):
            return memory

        evidence = MemoryEvidence(
            evidence_type=evidence_type,
            reference_id=reference_id,
            description=description,
            confidence=confidence,
        )

        added = memory.record_supporting_evidence(
            evidence
        )

        # -----------------------------------------------------
        # DUPLICATE EVIDENCE
        # -----------------------------------------------------

        if not added:
            return memory

        # -----------------------------------------------------
        # CONFIDENCE
        # -----------------------------------------------------

        memory.confidence = min(
            self.policy.maximum_confidence,
            memory.confidence
            + (
                self.policy.supporting_confidence_gain
                * confidence
            ),
        )

        # -----------------------------------------------------
        # CONFIRMED RESOLUTION
        # -----------------------------------------------------

        if confirmed_resolution:

            memory.record_successful_resolution()

            memory.confidence = min(
                self.policy.maximum_confidence,
                memory.confidence
                + (
                    self.policy.successful_resolution_gain
                    * confidence
                ),
            )

        # -----------------------------------------------------
        # PROMOTION
        # -----------------------------------------------------

        self._evaluate_promotion(
            memory
        )

        return memory

    # =========================================================
    # CONTRADICTION
    # =========================================================

    def add_contradiction(
        self,
        memory: MemoryRecord,
        *,
        reference_id: str,
        description: str,
        confidence: float = 1.0,
    ) -> MemoryRecord:
        """
        Record contradictory evidence.

        Contradictions are retained as evidence.

        They do not automatically destroy a memory because
        contradictions may reveal:

            machine-specific behavior
            hardware revision differences
            Windows-version differences
            driver-version differences
            environmental conditions
            conditional diagnostic rules
        """

        self._validate_evidence_parameters(
            reference_id=reference_id,
            description=description,
            confidence=confidence,
        )

        if (
            memory.status
            == MemoryStatus.DEPRECATED
        ):
            return memory

        evidence = MemoryEvidence(
            evidence_type=EvidenceType.CASE,
            reference_id=reference_id,
            description=description,
            confidence=confidence,
        )

        added = memory.record_contradiction(
            evidence
        )

        if not added:
            return memory

        memory.confidence = max(
            self.policy.minimum_confidence,
            memory.confidence
            - (
                self.policy.contradiction_penalty
                * confidence
            ),
        )

        self._evaluate_demotion(
            memory
        )

        return memory

    # =========================================================
    # PROMOTION
    # =========================================================

    def _evaluate_promotion(
        self,
        memory: MemoryRecord,
    ) -> None:
        """
        Promote memory only when sufficient successful
        resolution evidence exists.
        """

        if (
            memory.status
            == MemoryStatus.DEPRECATED
        ):
            return

        if (
            memory.successful_resolution_count
            >= self.policy.minimum_verification_cases
        ):
            memory.verify()

    # =========================================================
    # DEMOTION
    # =========================================================

    def _evaluate_demotion(
        self,
        memory: MemoryRecord,
    ) -> None:
        """
        Handle severe contradiction.

        Verification is not immediately destroyed by one
        contradiction.

        Demotion occurs only when contradictions are at least
        as numerous as successful resolutions AND confidence
        has fallen sufficiently.
        """

        if (
            memory.status
            != MemoryStatus.VERIFIED
        ):
            return

        if (
            memory.contradiction_count
            >= memory.successful_resolution_count
            and memory.confidence
            < self.policy.demotion_confidence_threshold
        ):
            memory.status = (
                MemoryStatus.CANDIDATE
            )

            memory.updated_at = (
                memory.updated_at
            )

    # =========================================================
    # EXPLICIT DEPRECATION
    # =========================================================

    def deprecate(
        self,
        memory: MemoryRecord,
        *,
        superseding_memory_id: str | None = None,
    ) -> MemoryRecord:
        """
        Explicitly retire a memory.

        The record remains available for audit/history but
        should normally be excluded from ordinary retrieval.
        """

        if (
            memory.status
            == MemoryStatus.DEPRECATED
        ):
            return memory

        memory.deprecate(
            superseding_memory_id
        )

        return memory

    # =========================================================
    # VALIDATION
    # =========================================================

    @staticmethod
    def _validate_evidence_parameters(
        *,
        reference_id: str,
        description: str,
        confidence: float,
    ) -> None:

        if not reference_id.strip():
            raise ValueError(
                "reference_id cannot be empty."
            )

        if not description.strip():
            raise ValueError(
                "description cannot be empty."
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError(
                "Evidence confidence must be between 0 and 1."
            )