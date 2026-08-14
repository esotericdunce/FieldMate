from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from fieldmate.brain.state.models import DiagnosticState

from .context import DiagnosticContext, context_from_evidence
from .evidence import Evidence


# ============================================================
# BUDGETED CONTEXT
# ============================================================

@dataclass(frozen=True)
class BudgetedContext:
    """
    Final evidence selection produced by ContextIntelligence.

    This is an internal retrieval-layer result. The public
    representation passed toward the Brain is DiagnosticContext.
    """

    supporting_evidence: tuple[Evidence, ...]
    contradicting_evidence: tuple[Evidence, ...]
    neutral_evidence: tuple[Evidence, ...]

    total_tokens_approx: int

    formatted_prompt_text: str


# ============================================================
# CONTEXT INTELLIGENCE
# ============================================================

class ContextIntelligence:
    """
    Selects the most useful retrieved evidence for FieldMate.

    Responsibilities:

        1. deduplicate evidence
        2. score evidence against current diagnostic state
        3. preserve contradictions
        4. prioritize verified knowledge
        5. prioritize exact equipment/fault matches
        6. enforce evidence-count limits
        7. enforce approximate token limits
        8. construct a compact reasoning context

    This class does NOT:

        - call Qdrant
        - call Groq
        - modify diagnostic state
        - decide whether a diagnosis is actually correct
        - promote memory
        - delete contradictory knowledge

    It is a context-selection layer, not the reasoning engine.
    """

    def __init__(
        self,
        *,
        max_evidence_items: int = 6,
        approx_token_budget: int = 1500,
        contradiction_reserve: int = 1,
    ) -> None:

        if max_evidence_items < 1:
            raise ValueError(
                "max_evidence_items must be >= 1"
            )

        if approx_token_budget < 1:
            raise ValueError(
                "approx_token_budget must be >= 1"
            )

        if contradiction_reserve < 0:
            raise ValueError(
                "contradiction_reserve cannot be negative"
            )

        self.max_evidence_items = (
            max_evidence_items
        )

        self.approx_token_budget = (
            approx_token_budget
        )

        self.contradiction_reserve = min(
            contradiction_reserve,
            max_evidence_items,
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> str:
        """
        Normalize text for comparison/deduplication.
        """

        return " ".join(
            text.strip().lower().split()
        )

    # ========================================================
    # EVIDENCE SCORING
    # ========================================================

    def _score_evidence(
        self,
        evidence: Evidence,
        state: DiagnosticState,
    ) -> float:
        """
        Calculate a retrieval-context priority score.

        The score is NOT a probability.

        It determines which retrieved evidence should receive
        scarce LLM context budget.

        Priority sources:

            retrieval relevance
            memory confidence
            equipment match
            fault-code match
            verification status
            relation to current hypothesis

        Contradictory evidence receives a modest preservation
        boost rather than being automatically suppressed.
        """

        score = 0.0

        # ----------------------------------------------------
        # Base retrieval relevance
        # ----------------------------------------------------

        score += (
            evidence.relevance_score
            * 0.40
        )

        # ----------------------------------------------------
        # Memory confidence
        # ----------------------------------------------------

        score += (
            evidence.confidence
            * 0.25
        )

        # ----------------------------------------------------
        # Equipment match
        # ----------------------------------------------------

        state_model = getattr(
            state.equipment,
            "model",
            None,
        )

        if (
            state_model
            and evidence.equipment_model
            and (
                str(state_model).strip().lower()
                == str(
                    evidence.equipment_model
                ).strip().lower()
            )
        ):
            score += 0.20

        # ----------------------------------------------------
        # Fault-code overlap
        # ----------------------------------------------------

        state_fault_codes = {
            str(code).strip().upper()
            for code in getattr(
                state,
                "fault_codes",
                [],
            )
            if str(code).strip()
        }

        evidence_fault_codes = {
            str(code).strip().upper()
            for code in evidence.fault_codes
            if str(code).strip()
        }

        if (
            state_fault_codes
            and evidence_fault_codes
            and state_fault_codes.intersection(
                evidence_fault_codes
            )
        ):
            score += 0.25

        # ----------------------------------------------------
        # Verification status
        # ----------------------------------------------------

        verification = (
            evidence.verification_status
            .strip()
            .lower()
        )

        if verification in {
            "verified",
            "confirmed",
        }:
            score += 0.15

        elif verification in {
            "candidate",
            "provisional",
        }:
            score += 0.03

        # ----------------------------------------------------
        # Current relation
        # ----------------------------------------------------

        if evidence.relation == "supporting":
            score += 0.10

        elif evidence.relation == "contradicting":
            # Contradictions must remain visible.
            #
            # This is intentionally smaller than a normal
            # relevance bonus. We want preservation, not
            # automatic promotion.
            score += 0.08

        # ----------------------------------------------------
        # Bound score.
        # ----------------------------------------------------

        return max(
            0.0,
            min(
                2.0,
                score,
            ),
        )

    # ========================================================
    # DEDUPLICATION
    # ========================================================

    def _deduplicate(
        self,
        evidence_items: Sequence[Evidence],
    ) -> list[Evidence]:
        """
        Remove duplicate memory IDs and duplicate content.

        Memory ID is the strongest identity signal.

        Content deduplication protects the context window from
        repeated copies of effectively identical memories.
        """

        seen_memory_ids: set[str] = set()
        seen_content: set[str] = set()

        result: list[Evidence] = []

        for evidence in evidence_items:

            if evidence.memory_id in seen_memory_ids:
                continue

            normalized = self._normalize_text(
                evidence.content
            )

            if not normalized:
                continue

            if normalized in seen_content:
                continue

            seen_memory_ids.add(
                evidence.memory_id
            )

            seen_content.add(
                normalized
            )

            result.append(
                evidence
            )

        return result

    # ========================================================
    # APPROXIMATE TOKEN COUNT
    # ========================================================

    @staticmethod
    def _approx_tokens(
        text: str,
    ) -> int:
        """
        Cheap token-count approximation.

        This intentionally avoids a tokenizer on the hot path.

        A conservative approximation of roughly two tokens per
        whitespace-delimited word works well enough for context
        budgeting without introducing another dependency.
        """

        if not text.strip():
            return 0

        return max(
            1,
            len(text.split()) * 2,
        )

    # ========================================================
    # FORMAT EVIDENCE
    # ========================================================

    @staticmethod
    def _format_evidence(
        evidence: Evidence,
    ) -> str:
        """
        Convert one Evidence object into compact LLM context.

        Keep this plain text because the voice agent ultimately
        needs spoken output rather than markdown-heavy responses.
        """

        metadata: list[str] = []

        metadata.append(
            f"confidence={evidence.confidence:.2f}"
        )

        metadata.append(
            f"relevance={evidence.relevance_score:.2f}"
        )

        if evidence.verification_status:
            metadata.append(
                "status="
                + evidence.verification_status
            )

        if evidence.source:
            metadata.append(
                "source="
                + evidence.source
            )

        if evidence.equipment_model:
            metadata.append(
                "equipment="
                + evidence.equipment_model
            )

        if evidence.fault_codes:
            metadata.append(
                "fault_codes="
                + ",".join(
                    evidence.fault_codes
                )
            )

        return (
            "["
            + " | ".join(metadata)
            + "] "
            + evidence.content
        )

    # ========================================================
    # BUILD PROMPT TEXT
    # ========================================================

    def _format_prompt(
        self,
        supporting: Sequence[Evidence],
        contradicting: Sequence[Evidence],
        neutral: Sequence[Evidence],
    ) -> str:
        """
        Build structured but compact diagnostic context.

        Contradictory evidence gets its own section so the LLM
        cannot easily confuse an alternative explanation with
        supporting evidence.
        """

        blocks: list[str] = []

        if supporting:
            blocks.append(
                "SUPPORTING EVIDENCE:"
            )

            for evidence in supporting:
                blocks.append(
                    "- "
                    + self._format_evidence(
                        evidence
                    )
                )

        if contradicting:
            blocks.append(
                "CONTRADICTORY OR ALTERNATIVE EVIDENCE:"
            )

            for evidence in contradicting:
                blocks.append(
                    "- "
                    + self._format_evidence(
                        evidence
                    )
                )

        if neutral:
            blocks.append(
                "OTHER RELEVANT TECHNICAL EVIDENCE:"
            )

            for evidence in neutral:
                blocks.append(
                    "- "
                    + self._format_evidence(
                        evidence
                    )
                )

        return "\n".join(
            blocks
        )

    # ========================================================
    # BUDGET SELECTION
    # ========================================================

    def _select_with_budget(
        self,
        supporting: Sequence[Evidence],
        contradicting: Sequence[Evidence],
        neutral: Sequence[Evidence],
    ) -> tuple[
        list[Evidence],
        list[Evidence],
        list[Evidence],
    ]:
        """
        Select evidence under both item and approximate-token
        budgets.

        Invariant:

            If contradictory evidence exists and the overall
            budget permits it, at least one contradiction is
            retained.

        This prevents high-scoring supporting memories from
        completely erasing alternative explanations.
        """

        selected_supporting: list[Evidence] = []
        selected_contradicting: list[Evidence] = []
        selected_neutral: list[Evidence] = []

        remaining_items = (
            self.max_evidence_items
        )

        remaining_tokens = (
            self.approx_token_budget
        )

        # ----------------------------------------------------
        # Candidate formatter
        # ----------------------------------------------------

        def try_add(
            target: list[Evidence],
            evidence: Evidence,
        ) -> bool:

            nonlocal remaining_items
            nonlocal remaining_tokens

            if remaining_items <= 0:
                return False

            formatted = self._format_evidence(
                evidence
            )

            cost = self._approx_tokens(
                formatted
            )

            if cost > remaining_tokens:
                return False

            target.append(
                evidence
            )

            remaining_items -= 1
            remaining_tokens -= cost

            return True

        # ----------------------------------------------------
        # CONTRADICTION RESERVATION
        # ----------------------------------------------------
        #
        # Reserve contradiction capacity before filling the
        # context with supporting evidence.
        # ----------------------------------------------------

        contradiction_added = False

        if (
            contradicting
            and self.contradiction_reserve > 0
        ):
            contradiction_added = try_add(
                selected_contradicting,
                contradicting[0],
            )

        # ----------------------------------------------------
        # Remaining candidates.
        #
        # The lists are already sorted by score.
        # Merge them while preserving their category.
        # ----------------------------------------------------

        remaining_candidates: list[
            tuple[float, str, Evidence]
        ] = []

        for evidence in supporting:
            remaining_candidates.append(
                (
                    self._score_evidence_from_cached(
                        evidence
                    ),
                    "supporting",
                    evidence,
                )
            )

        for evidence in contradicting[
            1 if contradiction_added else 0:
        ]:
            remaining_candidates.append(
                (
                    self._score_evidence_from_cached(
                        evidence
                    ),
                    "contradicting",
                    evidence,
                )
            )

        for evidence in neutral:
            remaining_candidates.append(
                (
                    self._score_evidence_from_cached(
                        evidence
                    ),
                    "neutral",
                    evidence,
                )
            )

        remaining_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # ----------------------------------------------------
        # Fill remaining budget.
        # ----------------------------------------------------

        for (
            _score,
            relation,
            evidence,
        ) in remaining_candidates:

            if remaining_items <= 0:
                break

            if relation == "supporting":
                try_add(
                    selected_supporting,
                    evidence,
                )

            elif relation == "contradicting":
                try_add(
                    selected_contradicting,
                    evidence,
                )

            else:
                try_add(
                    selected_neutral,
                    evidence,
                )

        return (
            selected_supporting,
            selected_contradicting,
            selected_neutral,
        )

    # ========================================================
    # CACHED SCORE HELPER
    # ========================================================

    @staticmethod
    def _score_evidence_from_cached(
        evidence: Evidence,
    ) -> float:
        """
        Secondary score used after the primary state-aware
        ranking has already occurred.

        Evidence objects don't store their state-aware score, so
        this provides a stable fallback ordering.

        Retrieval relevance remains the dominant signal.
        """

        score = (
            evidence.relevance_score
            * 0.60
        )

        score += (
            evidence.confidence
            * 0.30
        )

        if (
            evidence.verification_status
            .strip()
            .lower()
            in {
                "verified",
                "confirmed",
            }
        ):
            score += 0.10

        return score

    # ========================================================
    # MAIN CONTEXT BUILD
    # ========================================================

    def build_context(
        self,
        evidence_items: Sequence[Evidence],
        state: DiagnosticState,
    ) -> BudgetedContext:
        """
        Produce the final budgeted evidence selection.

        Pipeline:

            raw Evidence
                ↓
            deduplicate
                ↓
            state-aware scoring
                ↓
            relation separation
                ↓
            contradiction reservation
                ↓
            token/item budgeting
                ↓
            formatted prompt
        """

        deduped = self._deduplicate(
            evidence_items
        )

        if not deduped:
            return BudgetedContext(
                supporting_evidence=(),
                contradicting_evidence=(),
                neutral_evidence=(),
                total_tokens_approx=0,
                formatted_prompt_text="",
            )

        # ----------------------------------------------------
        # Score against current state.
        # ----------------------------------------------------

        scored = [
            (
                self._score_evidence(
                    evidence,
                    state,
                ),
                evidence,
            )
            for evidence in deduped
        ]

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        # ----------------------------------------------------
        # Separate by relation.
        # ----------------------------------------------------

        supporting: list[Evidence] = []
        contradicting: list[Evidence] = []
        neutral: list[Evidence] = []

        for _score, evidence in scored:

            if evidence.relation == "supporting":
                supporting.append(
                    evidence
                )

            elif evidence.relation == "contradicting":
                contradicting.append(
                    evidence
                )

            else:
                neutral.append(
                    evidence
                )

        # ----------------------------------------------------
        # Budget.
        # ----------------------------------------------------

        (
            selected_supporting,
            selected_contradicting,
            selected_neutral,
        ) = self._select_with_budget(
            supporting,
            contradicting,
            neutral,
        )

        # ----------------------------------------------------
        # Format.
        # ----------------------------------------------------

        formatted_text = self._format_prompt(
            selected_supporting,
            selected_contradicting,
            selected_neutral,
        )

        total_tokens = self._approx_tokens(
            formatted_text
        )

        return BudgetedContext(
            supporting_evidence=tuple(
                selected_supporting
            ),

            contradicting_evidence=tuple(
                selected_contradicting
            ),

            neutral_evidence=tuple(
                selected_neutral
            ),

            total_tokens_approx=total_tokens,

            formatted_prompt_text=formatted_text,
        )

    # ========================================================
    # DIAGNOSTIC CONTEXT
    # ========================================================

    def build_diagnostic_context(
        self,
        evidence_items: Sequence[Evidence],
        state: DiagnosticState,
        *,
        query: str = "",
        retrieval_mode: str = "dense",
        prefetched: bool = False,
        timed_out: bool = False,
    ) -> DiagnosticContext:
        """
        Production helper.

        Builds budgeted evidence and immediately converts it
        into the stable DiagnosticContext contract.
        """

        budgeted = self.build_context(
            evidence_items,
            state,
        )

        selected = (
            list(
                budgeted.supporting_evidence
            )
            + list(
                budgeted.contradicting_evidence
            )
            + list(
                budgeted.neutral_evidence
            )
        )
        print(f"DEBUG ContextIntelligence: evidence_items={len(evidence_items)}, selected={len(selected)}", flush=True)

        return context_from_evidence(
            selected,

            query=query,

            retrieval_mode=retrieval_mode,

            max_memories=len(
                selected
            ),

            total_tokens_approx=(
                budgeted.total_tokens_approx
            ),

            prefetched=prefetched,

            timed_out=timed_out,
        )