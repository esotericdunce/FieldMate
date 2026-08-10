from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fieldmate.brain.memory.manager import MemoryManager
from fieldmate.brain.models import (
    BrainResult,
    DiagnosticContext,
    DiagnosticDecision,
)
from fieldmate.brain.reasoning import ReasoningManager
from fieldmate.brain.retrieval.evidence import Evidence
from fieldmate.brain.retrieval.orchestrator import (
    RetrievalOrchestrator,
)
from fieldmate.brain.state.engine import StateEngine
from fieldmate.brain.state.events import (
    DomainEvent,
    EventType,
)


@dataclass(slots=True)
class Brain:
    """
    FieldMate's application-level diagnostic brain.

    Brain is the orchestration boundary between:

        voice/application layer
                |
                v
             Brain
          /     |     \
         v      v      v
      State  Retrieval Reasoning
         |      |       |
         v      v       v
      Domain  Qdrant   Groq

    External systems never directly mutate canonical state.

    Reasoning produces a structured DiagnosticDecision.

    Brain translates supported decision fields into explicit
    DomainEvents.

    StateEngine remains the sole authority allowed to mutate
    FieldMateSession.
    """

    state: StateEngine
    retrieval: RetrievalOrchestrator
    memory: MemoryManager
    reasoning: ReasoningManager


    retrieval_enabled: bool = True

    # =========================================================
    # MAIN DIAGNOSTIC TURN
    # =========================================================

    async def process(
        self,
        user_input: str,
        *,
        technical: bool | None = None,
    ) -> BrainResult:
        """
        Process one committed technician utterance.
        """

        user_input = user_input.strip()

        if not user_input:
            return BrainResult(
                response="",
                decision=None,
                context=DiagnosticContext(),
                retrieved=False,
                prefetched=False,
                timed_out=False,
                retrieval_latency_ms=0.0,
                reasoning_latency_ms=0.0,
                turn=self.state.current_turn_id,
                generation=self.state.current_generation_id,
            )

        # =====================================================
        # NEW TURN / GENERATION
        # =====================================================

        turn = self.state.current_turn_id + 1
        generation = self.state.current_generation_id + 1

        # =====================================================
        # RECORD USER MESSAGE
        # =====================================================

        self.state.apply(
            DomainEvent(
                session_id=self.state.session.session_id,
                event_type=EventType.USER_MESSAGE,
                payload={
                    "message": user_input,
                },
                turn_id=turn,
                generation_id=generation,
                source="brain",
            )
        )


        # =====================================================
        # NON-TECHNICAL FAST PATH
        # =====================================================

        if technical is False:
            decision, reasoning_latency = await self.reasoning.chat(
                user_input=user_input,
                turn=turn,
                generation=generation,
            )

            return BrainResult(
                response=decision.response,
                decision=decision,
                context=DiagnosticContext(),
                retrieved=False,
                prefetched=False,
                timed_out=False,
                retrieval_latency_ms=0.0,
                reasoning_latency_ms=reasoning_latency,
                turn=turn,
                generation=generation,
            )

        # =====================================================
        # RETRIEVAL
        # =====================================================

        context = DiagnosticContext()
        retrieval_result = None

        if self.retrieval_enabled:

            diagnostic = self.state.session.diagnostic
            equipment = diagnostic.equipment

            retrieval_result = await self.retrieval.retrieve(
                user_input,
                equipment_model=equipment.model,
                equipment_family=equipment.equipment_family,
                equipment_serial=equipment.serial_number,
                system=equipment.system,
                subsystem=equipment.subsystem,
                component=equipment.component,
                fault_code=(
                    diagnostic.fault_codes[-1]
                    if diagnostic.fault_codes
                    else None
                ),
                limit=8,
            )

            context = self._convert_retrieval_context(
                retrieval_result
            )

        # =====================================================
        # STALE GENERATION GUARD
        # =====================================================

        if generation < self.state.current_generation_id:
            return BrainResult(
                response="",
                decision=None,
                context=context,
                retrieved=retrieval_result is not None,
                prefetched=bool(
                    getattr(
                        retrieval_result,
                        "prefetched",
                        False,
                    )
                ),
                timed_out=bool(
                    getattr(
                        retrieval_result,
                        "timed_out",
                        False,
                    )
                ),
                retrieval_latency_ms=float(
                    getattr(
                        retrieval_result,
                        "latency_ms",
                        0.0,
                    )
                ),
                reasoning_latency_ms=0.0,
                turn=turn,
                generation=generation,
            )

        # =====================================================
        # REASONING
        # =====================================================

        decision, reasoning_latency = (
            await self.reasoning.reason(
                state=self._prompt_state(),
                context=context,
                user_input=user_input,
                turn=turn,
                generation=generation,
            )
        )

        # =====================================================
        # APPLY VALIDATED DECISION
        # =====================================================

        self._apply_decision(
            decision,
            turn=turn,
            generation=generation,
        )

        return BrainResult(
            response=decision.response,
            decision=decision,
            context=context,
            retrieved=retrieval_result is not None,
            prefetched=bool(
                getattr(
                    retrieval_result,
                    "prefetched",
                    False,
                )
            ),
            timed_out=bool(
                getattr(
                    retrieval_result,
                    "timed_out",
                    False,
                )
            ),
            retrieval_latency_ms=float(
                getattr(
                    retrieval_result,
                    "latency_ms",
                    0.0,
                )
            ),
            reasoning_latency_ms=reasoning_latency,
            turn=turn,
            generation=generation,
        )

    # =========================================================
    # SPECULATIVE RETRIEVAL
    # =========================================================

    async def speculate(
        self,
        partial_transcript: str,
    ) -> None:
        """
        Called while the technician is still speaking.

        Prefetch is speculative and must never mutate state.
        """

        if not self.retrieval_enabled:
            return

        partial_transcript = partial_transcript.strip()

        if not partial_transcript:
            return

        diagnostic = self.state.session.diagnostic
        equipment = diagnostic.equipment

        await self.retrieval.prefetch(
            partial_transcript,
            equipment_model=equipment.model,
            equipment_family=equipment.equipment_family,
            equipment_serial=equipment.serial_number,
            fault_code=(
                diagnostic.fault_codes[-1]
                if diagnostic.fault_codes
                else None
            ),
        )

    # =========================================================
    # PROMPT STATE
    # =========================================================

    def _prompt_state(self) -> dict[str, Any]:
        """
        Convert authoritative domain state into a JSON-safe
        snapshot for the reasoning layer.
        """

        session = self.state.session
        diagnostic = session.diagnostic
        equipment = diagnostic.equipment
        conversation = session.conversation

        return {
            "session_id": session.session_id,
            "diagnostic": {
                "equipment": {
                    "manufacturer": equipment.manufacturer,
                    "model": equipment.model,
                    "serial_number": equipment.serial_number,
                    "equipment_family": equipment.equipment_family,
                    "system": equipment.system,
                    "subsystem": equipment.subsystem,
                    "component": equipment.component,
                },
                "fault_codes": list(
                    diagnostic.fault_codes
                ),
                "symptoms": [
                    {
                        "name": item.name,
                        "value": item.value,
                        "source": item.source.value,
                        "status": item.status.value,
                        "confidence": item.confidence,
                        "notes": item.notes,
                    }
                    for item in diagnostic.symptoms
                ],
                "observations": [
                    {
                        "name": item.name,
                        "value": item.value,
                        "source": item.source.value,
                        "status": item.status.value,
                        "confidence": item.confidence,
                        "notes": item.notes,
                    }
                    for item in diagnostic.observations
                ],
                "measurements": [
                    {
                        "name": item.name,
                        "value": item.value,
                        "unit": item.unit,
                        "source": item.source.value,
                        "status": item.status.value,
                        "min_expected": item.min_expected,
                        "max_expected": item.max_expected,
                        "is_out_of_range": item.is_out_of_range,
                    }
                    for item in diagnostic.measurements
                ],
                "tests": [
                    {
                        "name": item.name,
                        "status": item.status.value,
                        "result": item.result,
                        "started_at": (
                            item.started_at.isoformat()
                            if item.started_at
                            else None
                        ),
                        "completed_at": (
                            item.completed_at.isoformat()
                            if item.completed_at
                            else None
                        ),
                        "notes": item.notes,
                    }
                    for item in diagnostic.tests
                ],
                "hypotheses": [
                    {
                        "description": item.description,
                        "status": item.status.value,
                        "confidence": item.confidence,
                        "supporting_evidence": list(
                            item.supporting_evidence
                        ),
                        "contradicting_evidence": list(
                            item.contradicting_evidence
                        ),
                    }
                    for item in diagnostic.hypotheses
                ],
                "current_hypothesis": (
                    diagnostic.current_hypothesis
                ),
                "next_recommended_action": (
                    diagnostic.next_recommended_action
                ),
                "confirmed_resolution": (
                    diagnostic.confirmed_resolution
                ),
                "case_status": diagnostic.case_status,
            },
            "conversation": {
                "recent_user_messages": list(
                    conversation.recent_user_messages[-10:]
                ),
                "recent_assistant_messages": list(
                    conversation.recent_assistant_messages[-10:]
                ),
                "turn_id": conversation.turn_id,
            },
        }

    # =========================================================
    # DECISION → DOMAIN EVENTS
    # =========================================================

    def _apply_decision(
        self,
        decision: DiagnosticDecision,
        *,
        turn: int,
        generation: int,
    ) -> None:
        """
        Translate reasoning output into explicitly supported
        DomainEvents.
        """

        if generation < self.state.current_generation_id:
            return

        session_id = self.state.session.session_id

        # -----------------------------------------------------
        # HYPOTHESIS
        # -----------------------------------------------------

        if decision.hypothesis:
            self.state.apply(
                DomainEvent(
                    session_id=session_id,
                    event_type=EventType.HYPOTHESIS_CREATED,
                    payload={
                        "description": decision.hypothesis,
                        "confidence": (
                            decision.confidence
                            if decision.confidence is not None
                            else 0.5
                        ),
                    },
                    turn_id=turn,
                    generation_id=generation,
                    source="reasoning",
                )
            )

        # -----------------------------------------------------
        # RECOMMENDED ACTION
        # -----------------------------------------------------

        if decision.next_action:
            self.state.apply(
                DomainEvent(
                    session_id=session_id,
                    event_type=EventType.RECOMMENDATION_CREATED,
                    payload={
                        "action": decision.next_action,
                    },
                    turn_id=turn,
                    generation_id=generation,
                    source="reasoning",
                )
            )

        # -----------------------------------------------------
        # PROPOSED RESOLUTION
        # -----------------------------------------------------

        if decision.resolution_proposed:
            self.state.apply(
                DomainEvent(
                    session_id=session_id,
                    event_type=EventType.RESOLUTION_PROPOSED,
                    payload={
                        "resolution": decision.resolution_proposed,
                    },
                    turn_id=turn,
                    generation_id=generation,
                    source="reasoning",
                )
            )

        # -----------------------------------------------------
        # CONFIRMED RESOLUTION
        # -----------------------------------------------------

        if decision.resolution_confirmed:
            self.state.apply(
                DomainEvent(
                    session_id=session_id,
                    event_type=EventType.RESOLUTION_CONFIRMED,
                    payload={
                        "resolution": (
                            decision.resolution_confirmed
                        ),
                    },
                    turn_id=turn,
                    generation_id=generation,
                    source="reasoning",
                )
            )

        # -----------------------------------------------------
        # EXPLICIT STATE UPDATES
        # -----------------------------------------------------

        for update in decision.state_updates:
            if not isinstance(update, dict):
                continue

            event_type_raw = update.get("type")
            payload = update.get("payload", {})

            if not isinstance(event_type_raw, str):
                continue

            if not isinstance(payload, dict):
                continue

            event_type = self._allowed_event_type(
                event_type_raw
            )

            if event_type is None:
                continue

            self.state.apply(
                DomainEvent(
                    session_id=session_id,
                    event_type=event_type,
                    payload=payload,
                    turn_id=turn,
                    generation_id=generation,
                    source="reasoning",
                )
            )

    # =========================================================
    # EVENT ALLOWLIST
    # =========================================================

    @staticmethod
    def _allowed_event_type(
        value: str,
    ) -> EventType | None:
        """
        Convert an LLM-provided event name into a safe EventType.

        Dangerous assertions such as confirmed resolution are
        deliberately excluded from generic state_updates.
        """

        allowed = {
            EventType.EQUIPMENT_IDENTIFIED,
            EventType.FAULT_IDENTIFIED,
            EventType.SYMPTOM_RECORDED,
            EventType.OBSERVATION_RECORDED,
            EventType.MEASUREMENT_RECORDED,
            EventType.TEST_STARTED,
            EventType.TEST_COMPLETED,
            EventType.HYPOTHESIS_CREATED,
            EventType.HYPOTHESIS_UPDATED,
            EventType.RECOMMENDATION_CREATED,
            EventType.PROCEDURE_RETRIEVED,
            EventType.RESOLUTION_PROPOSED,
            EventType.CASE_CLOSED,
        }

        try:
            event_type = EventType(value)
        except ValueError:
            return None

        if event_type not in allowed:
            return None

        return event_type

    # =========================================================
    # RETRIEVAL CONTEXT ADAPTER
    # =========================================================

    @staticmethod
    def _convert_retrieval_context(
        retrieval_result: Any,
    ) -> DiagnosticContext:
        """
        Convert RetrievalOrchestrator's result into the
        canonical reasoning-layer DiagnosticContext.

        The retrieval layer may expose either:

            1. the canonical Evidence representation, or
            2. its legacy RetrievedMemory representation.

        Qdrant objects are never passed to reasoning.
        """

        if retrieval_result is None:
            return DiagnosticContext()

        raw_context = getattr(
            retrieval_result,
            "context",
            None,
        )

        if raw_context is None:
            return DiagnosticContext()

        # -----------------------------------------------------
        # Canonical context
        # -----------------------------------------------------

        if isinstance(
            raw_context,
            DiagnosticContext,
        ):
            return raw_context

        # -----------------------------------------------------
        # ContextIntelligence / BudgetedContext
        #
        # This contains Evidence directly.
        # -----------------------------------------------------

        evidence_groups = (
            getattr(
                raw_context,
                "supporting_evidence",
                (),
            ),
            getattr(
                raw_context,
                "contradicting_evidence",
                (),
            ),
            getattr(
                raw_context,
                "neutral_evidence",
                (),
            ),
        )

        evidence = []

        for group in evidence_groups:
            evidence.extend(group)

        if evidence:
            return Brain._context_from_evidence(
                evidence
            )

        # -----------------------------------------------------
        # Legacy RetrievedMemory context
        #
        # Do not leak RetrievedMemory into reasoning.
        # Convert it conservatively into Evidence.
        # -----------------------------------------------------

        memories = getattr(
            raw_context,
            "memories",
            (),
        )

        if not memories:
            return DiagnosticContext()

        converted: list[Evidence] = []

        for memory in memories:
            converted.append(
                Brain._memory_to_evidence(
                    memory
                )
            )

        return Brain._context_from_evidence(
            converted
        )

    @staticmethod
    def _context_from_evidence(
        evidence_items: list[Evidence],
    ) -> DiagnosticContext:
        """
        Build the canonical DiagnosticContext from Evidence.

        Contradictions are preserved rather than discarded.
        """

        supporting: list[Evidence] = []
        contradicting: list[Evidence] = []
        neutral: list[Evidence] = []

        procedures: list[Evidence] = []
        past_cases: list[Evidence] = []
        resolutions: list[Evidence] = []

        for evidence in evidence_items:
            relation = str(
                getattr(
                    evidence.relation,
                    "value",
                    evidence.relation,
                )
            ).lower()

            if relation in {
                "supporting",
                "supports",
            }:
                supporting.append(evidence)

            elif relation in {
                "contradicting",
                "contradicts",
            }:
                contradicting.append(evidence)

            else:
                neutral.append(evidence)

            memory_type = str(
                getattr(
                    evidence.memory_type,
                    "value",
                    evidence.memory_type,
                )
            ).lower()

            if memory_type == "procedure":
                procedures.append(evidence)

            elif memory_type in {
                "past_case",
                "case",
            }:
                past_cases.append(evidence)

            elif memory_type == "resolution":
                resolutions.append(evidence)

        return DiagnosticContext(
            evidence=tuple(evidence_items),
            supporting=tuple(supporting),
            contradicting=tuple(contradicting),
            neutral=tuple(neutral),
            procedures=tuple(procedures),
            past_cases=tuple(past_cases),
            resolutions=tuple(resolutions),
        )

    @staticmethod
    def _memory_to_evidence(
        memory: Any,
    ) -> Evidence:
        """
        Conservative compatibility adapter for legacy
        RetrievedMemory objects.
        """

        memory_id = str(
            getattr(
                memory,
                "memory_id",
                "",
            )
        )

        content = str(
            getattr(
                memory,
                "content",
                "",
            )
        )

        memory_type = str(
            getattr(
                memory,
                "memory_type",
                "unknown",
            )
        )

        if hasattr(
            memory_type,
            "value",
        ):
            memory_type = memory_type.value

        fault_codes = getattr(
            memory,
            "fault_codes",
            (),
        )

        fault_code = (
            str(fault_codes[0])
            if fault_codes
            else None
        )

        # Evidence uses retrieval.evidence.Evidence,
        # whose relation is a string in the current
        # retrieval layer.
        return Evidence(
            evidence_id=f"ev_{memory_id}",
            memory_id=memory_id or None,
            memory_type=memory_type,
            content=content,
            source="qdrant",
            equipment_model=getattr(
                memory,
                "equipment_model",
                None,
            ),
            fault_codes=tuple(
                str(code)
                for code in fault_codes
            ),
            relevance_score=float(
                getattr(
                    memory,
                    "score",
                    0.0,
                )
            ),
            confidence=float(
                getattr(
                    memory,
                    "confidence",
                    0.0,
                )
            ),
            verification_status=(
                "verified"
                if getattr(
                    memory,
                    "confidence",
                    0.0,
                )
                >= 0.8
                else "unverified"
            ),
            provenance="qdrant",
            relation="neutral",
        )

    # =========================================================
    # SHUTDOWN
    # =========================================================

    async def close(self) -> None:
        await self.retrieval.close()