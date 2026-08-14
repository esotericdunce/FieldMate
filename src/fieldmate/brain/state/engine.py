from __future__ import annotations

from copy import deepcopy

from .events import DomainEvent, EventType
from .models import (
    DiagnosticTest,
    FieldMateSession,
    Hypothesis,
    HypothesisStatus,
    Measurement,
    Observation,
    ObservationSource,
    ObservationStatus,
    TestStatus,
    utc_now,
)


class StateTransitionError(Exception):
    """Base exception for rejected state transitions."""


class StaleEventError(StateTransitionError):
    """Raised when an event belongs to an obsolete turn/generation."""


class StateEngine:
    """
    Deterministic domain-state transition engine.

    External systems such as Groq, Qdrant, STT, and retrieval
    never directly mutate FieldMateSession.

    They propose DomainEvents.

    This engine validates and applies those events.

    Guarantees:

    - invalid transitions are rejected
    - rejected events do not mutate state
    - accepted transitions are atomic
    - duplicate event IDs are idempotent
    - reusing an event ID with different contents is rejected
    - stale turns are rejected
    - stale reasoning generations are rejected
    - accepted events are preserved in the event log
    """

    def __init__(
        self,
        session: FieldMateSession,
    ) -> None:
        self.session = session

        self._events: list[DomainEvent] = []

        self._current_turn_id = 0

        self._current_generation_id = 0

        self._applied_events: dict[
            str,
            DomainEvent,
        ] = {}

    # =========================================================
    # PROPERTIES
    # =========================================================

    @property
    def events(self) -> tuple[DomainEvent, ...]:
        return tuple(self._events)

    @property
    def current_turn_id(self) -> int:
        return self._current_turn_id

    @property
    def current_generation_id(self) -> int:
        return self._current_generation_id

    # =========================================================
    # PUBLIC EVENT APPLICATION
    # =========================================================

    def apply(
        self,
        event: DomainEvent,
    ) -> None:
        """
        Atomically apply one DomainEvent.

        Processing:

            validation
                ↓
            idempotency
                ↓
            stale checks
                ↓
            handler
                ↓
            commit

        If the handler fails:

            state → rollback
            event → NOT committed
        """

        # =====================================================
        # EVENT VALIDATION
        # =====================================================

        if event.session_id != self.session.session_id:
            raise StateTransitionError(
                "Event belongs to a different session: "
                f"event={event.session_id!r}, "
                f"engine={self.session.session_id!r}"
            )

        # =====================================================
        # IDEMPOTENCY
        # =====================================================

        existing = self._applied_events.get(
            event.event_id
        )

        if existing is not None:
            if existing == event:
                return

            raise StateTransitionError(
                "Event ID reused with different contents"
            )

        # =====================================================
        # STALE TURN
        # =====================================================

        if (
            event.turn_id is not None
            and event.turn_id < self._current_turn_id
        ):
            raise StaleEventError(
                f"Stale event turn={event.turn_id}; "
                f"current turn={self._current_turn_id}"
            )

        # =====================================================
        # STALE GENERATION
        # =====================================================

        if (
            event.generation_id
            < self._current_generation_id
        ):
            raise StaleEventError(
                f"Stale generation="
                f"{event.generation_id}; "
                f"current generation="
                f"{self._current_generation_id}"
            )

        # =====================================================
        # RESOLVE HANDLER
        # =====================================================

        handler = {
            EventType.SESSION_STARTED:
                self._session_started,

            EventType.USER_MESSAGE:
                self._user_message,

            EventType.EQUIPMENT_IDENTIFIED:
                self._equipment_identified,

            EventType.FAULT_IDENTIFIED:
                self._fault_identified,

            EventType.SYMPTOM_RECORDED:
                self._symptom_recorded,

            EventType.OBSERVATION_RECORDED:
                self._observation_recorded,

            EventType.VISUAL_OBSERVATION_RECORDED:
                self._visual_observation_recorded,

            EventType.MEASUREMENT_RECORDED:
                self._measurement_recorded,

            EventType.TEST_STARTED:
                self._test_started,

            EventType.TEST_COMPLETED:
                self._test_completed,

            EventType.HYPOTHESIS_CREATED:
                self._hypothesis_created,

            EventType.HYPOTHESIS_UPDATED:
                self._hypothesis_updated,

            EventType.RECOMMENDATION_CREATED:
                self._recommendation_created,

            EventType.PROCEDURE_RETRIEVED:
                self._procedure_retrieved,

            EventType.RESOLUTION_PROPOSED:
                self._resolution_proposed,

            EventType.RESOLUTION_CONFIRMED:
                self._resolution_confirmed,

            EventType.CASE_CLOSED:
                self._case_closed,
        }.get(event.event_type)

        if handler is None:
            raise StateTransitionError(
                f"Unsupported event type: "
                f"{event.event_type}"
            )

        # =====================================================
        # TRANSACTION SNAPSHOT
        # =====================================================

        session_snapshot = deepcopy(
            self.session
        )

        events_length = len(
            self._events
        )

        applied_events_snapshot = dict(
            self._applied_events
        )

        current_turn_snapshot = (
            self._current_turn_id
        )

        current_generation_snapshot = (
            self._current_generation_id
        )

        # =====================================================
        # TRANSACTION
        # =====================================================

        try:
            handler(event)

        except Exception as exc:

            # -------------------------------------------------
            # ROLLBACK SESSION
            # -------------------------------------------------

            self.session = session_snapshot

            # -------------------------------------------------
            # ROLLBACK EVENT LOG
            # -------------------------------------------------

            del self._events[
                events_length:
            ]

            # -------------------------------------------------
            # ROLLBACK EVENT INDEX
            # -------------------------------------------------

            self._applied_events = (
                applied_events_snapshot
            )

            # -------------------------------------------------
            # ROLLBACK TURN / GENERATION
            # -------------------------------------------------

            self._current_turn_id = (
                current_turn_snapshot
            )

            self._current_generation_id = (
                current_generation_snapshot
            )

            # -------------------------------------------------
            # PRESERVE DOMAIN ERRORS
            # -------------------------------------------------

            if isinstance(
                exc,
                StateTransitionError,
            ):
                raise

            # -------------------------------------------------
            # WRAP UNEXPECTED ERRORS
            # -------------------------------------------------

            raise StateTransitionError(
                "Atomic state transition failed"
            ) from exc

        # =====================================================
        # COMMIT
        # =====================================================

        self._events.append(event)

        self._applied_events[
            event.event_id
        ] = event

        if event.turn_id is not None:
            self._current_turn_id = max(
                self._current_turn_id,
                event.turn_id,
            )

        self._current_generation_id = max(
            self._current_generation_id,
            event.generation_id,
        )

        self.session.updated_at = utc_now()

    # =========================================================
    # SESSION
    # =========================================================

    def _session_started(
        self,
        event: DomainEvent,
    ) -> None:
        if self.session.diagnostic.case_status != "open":
            self.session.diagnostic.case_status = "open"

    # =========================================================
    # CONVERSATION
    # =========================================================

    def _user_message(
        self,
        event: DomainEvent,
    ) -> None:
        message = event.payload.get(
            "message"
        )

        if not isinstance(
            message,
            str,
        ):
            raise StateTransitionError(
                "USER_MESSAGE requires "
                "string 'message'"
            )

        message = message.strip()

        if not message:
            raise StateTransitionError(
                "USER_MESSAGE cannot be empty"
            )

        self.session.conversation.recent_user_messages.append(
            message
        )

        turn_id = (
            event.turn_id
            if event.turn_id is not None
            else (
                self.session.conversation.turn_id
                + 1
            )
        )

        self.session.conversation.turn_id = turn_id

    # =========================================================
    # EQUIPMENT
    # =========================================================

    def _equipment_identified(
        self,
        event: DomainEvent,
    ) -> None:
        equipment = (
            self.session
            .diagnostic
            .equipment
        )

        allowed_fields = (
            "manufacturer",
            "model",
            "serial_number",
            "equipment_family",
            "system",
            "subsystem",
            "component",
        )

        for field_name in allowed_fields:
            if field_name not in event.payload:
                continue

            value = event.payload[field_name]

            if value is not None and not isinstance(
                value,
                str,
            ):
                raise StateTransitionError(
                    f"Equipment field "
                    f"'{field_name}' must be "
                    f"a string or None"
                )

            if isinstance(value, str):
                value = value.strip() or None

            setattr(
                equipment,
                field_name,
                value,
            )

    # =========================================================
    # FAULT
    # =========================================================

    def _fault_identified(
        self,
        event: DomainEvent,
    ) -> None:
        fault_code = event.payload.get(
            "fault_code"
        )

        if not isinstance(
            fault_code,
            str,
        ):
            raise StateTransitionError(
                "FAULT_IDENTIFIED requires "
                "'fault_code'"
            )

        fault_code = fault_code.strip()

        if not fault_code:
            raise StateTransitionError(
                "Fault code cannot be empty"
            )

        if (
            fault_code
            not in self.session.diagnostic.fault_codes
        ):
            self.session.diagnostic.fault_codes.append(
                fault_code
            )

    # =========================================================
    # SYMPTOMS
    # =========================================================

    def _symptom_recorded(
        self,
        event: DomainEvent,
    ) -> None:
        name = event.payload.get(
            "name"
        )

        value = event.payload.get(
            "value"
        )

        if not isinstance(
            name,
            str,
        ):
            raise StateTransitionError(
                "SYMPTOM_RECORDED requires "
                "'name'"
            )

        name = name.strip()

        if not name:
            raise StateTransitionError(
                "Symptom name cannot be empty"
            )

        confidence = float(
            event.payload.get(
                "confidence",
                1.0,
            )
        )

        if not 0.0 <= confidence <= 1.0:
            raise StateTransitionError(
                "Symptom confidence must "
                "be between 0 and 1"
            )

        notes = event.payload.get(
            "notes"
        )

        if notes is not None and not isinstance(
            notes,
            str,
        ):
            raise StateTransitionError(
                "Symptom notes must be "
                "a string or None"
            )

        observation = Observation(
            name=name,
            value=value,
            source=ObservationSource.USER,
            status=ObservationStatus.REPORTED,
            confidence=confidence,
            notes=notes,
        )

        self.session.diagnostic.symptoms.append(
            observation
        )

    # =========================================================
    # OBSERVATIONS
    # =========================================================

    def _observation_recorded(
        self,
        event: DomainEvent,
    ) -> None:
        name = event.payload.get(
            "name"
        )

        if not isinstance(
            name,
            str,
        ):
            raise StateTransitionError(
                "OBSERVATION_RECORDED requires "
                "'name'"
            )

        name = name.strip()

        if not name:
            raise StateTransitionError(
                "Observation name cannot be empty"
            )

        try:
            source = ObservationSource(
                event.payload.get(
                    "source",
                    ObservationSource.USER.value,
                )
            )

            status = ObservationStatus(
                event.payload.get(
                    "status",
                    ObservationStatus.REPORTED.value,
                )
            )
        except ValueError as exc:
            raise StateTransitionError(
                "Invalid observation source/status"
            ) from exc

        confidence = float(
            event.payload.get(
                "confidence",
                1.0,
            )
        )

        if not 0.0 <= confidence <= 1.0:
            raise StateTransitionError(
                "Observation confidence must "
                "be between 0 and 1"
            )

        observation = Observation(
            name=name,
            value=event.payload.get(
                "value"
            ),
            source=source,
            status=status,
            confidence=confidence,
            notes=event.payload.get(
                "notes"
            ),
        )

        self.session.diagnostic.observations.append(
            observation
        )

    def _visual_observation_recorded(
        self,
        event: DomainEvent,
    ) -> None:
        confidence = float(
            event.payload.get(
                "confidence",
                1.0,
            )
        )

        visual_facts = event.payload.get("visual_facts", [])
        for fact in visual_facts:
            self.session.diagnostic.observations.append(
                Observation(
                    name="visual_fact",
                    value=fact,
                    source=ObservationSource.CAMERA_VISION,
                    status=ObservationStatus.VERIFIED,
                    confidence=confidence,
                )
            )

        contradictions = event.payload.get("contradictions", [])
        for contradiction in contradictions:
            self.session.diagnostic.observations.append(
                Observation(
                    name="contradiction",
                    value=contradiction,
                    source=ObservationSource.CAMERA_VISION,
                    status=ObservationStatus.CONTRADICTED,
                    confidence=confidence,
                )
            )

        uncertain_observations = event.payload.get("uncertain_observations", [])
        for uncertainty in uncertain_observations:
            self.session.diagnostic.observations.append(
                Observation(
                    name="uncertain_observation",
                    value=uncertainty,
                    source=ObservationSource.CAMERA_VISION,
                    status=ObservationStatus.REPORTED,
                    confidence=0.5,
                )
            )

        hardware_identifiers = event.payload.get("hardware_identifiers", {})
        for key, value in hardware_identifiers.items():
            self.session.diagnostic.observations.append(
                Observation(
                    name=f"hardware_identifier:{key}",
                    value=value,
                    source=ObservationSource.CAMERA_VISION,
                    status=ObservationStatus.VERIFIED,
                    confidence=confidence,
                )
            )

        ocr_text = event.payload.get("ocr_text")
        if ocr_text:
            self.session.diagnostic.observations.append(
                Observation(
                    name="ocr_text",
                    value=ocr_text,
                    source=ObservationSource.CAMERA_VISION,
                    status=ObservationStatus.VERIFIED,
                    confidence=confidence,
                )
            )

    # =========================================================
    # MEASUREMENTS
    # =========================================================

    def _measurement_recorded(
        self,
        event: DomainEvent,
    ) -> None:
        name = event.payload.get(
            "name"
        )

        value = event.payload.get(
            "value"
        )

        unit = event.payload.get(
            "unit"
        )

        if not isinstance(
            name,
            str,
        ):
            raise StateTransitionError(
                "MEASUREMENT_RECORDED requires "
                "'name'"
            )

        if not isinstance(
            value,
            (int, float),
        ):
            raise StateTransitionError(
                "Measurement value must "
                "be numeric"
            )

        if isinstance(value, bool):
            raise StateTransitionError(
                "Measurement value cannot "
                "be boolean"
            )

        if not isinstance(
            unit,
            str,
        ):
            raise StateTransitionError(
                "Measurement requires "
                "'unit'"
            )

        name = name.strip()
        unit = unit.strip()

        if not name:
            raise StateTransitionError(
                "Measurement name cannot be empty"
            )

        if not unit:
            raise StateTransitionError(
                "Measurement unit cannot be empty"
            )

        try:
            source = ObservationSource(
                event.payload.get(
                    "source",
                    ObservationSource.USER.value,
                )
            )

            status = ObservationStatus(
                event.payload.get(
                    "status",
                    ObservationStatus.REPORTED.value,
                )
            )
        except ValueError as exc:
            raise StateTransitionError(
                "Invalid measurement source/status"
            ) from exc

        measurement = Measurement(
            name=name,
            value=float(value),
            unit=unit,
            source=source,
            status=status,
            min_expected=event.payload.get(
                "min_expected"
            ),
            max_expected=event.payload.get(
                "max_expected"
            ),
        )

        self.session.diagnostic.measurements.append(
            measurement
        )

    # =========================================================
    # TESTS
    # =========================================================

    def _test_started(
        self,
        event: DomainEvent,
    ) -> None:
        name = event.payload.get(
            "name"
        )

        if not isinstance(
            name,
            str,
        ):
            raise StateTransitionError(
                "TEST_STARTED requires "
                "'name'"
            )

        name = name.strip()

        if not name:
            raise StateTransitionError(
                "Test name cannot be empty"
            )

        existing = self._find_test(name)

        if existing is not None:

            if (
                existing.status
                == TestStatus.IN_PROGRESS
            ):
                raise StateTransitionError(
                    f"Test already in progress: "
                    f"{name}"
                )

            existing.status = (
                TestStatus.IN_PROGRESS
            )

            existing.started_at = utc_now()
            existing.completed_at = None
            existing.result = None
            existing.notes = None

            return

        self.session.diagnostic.tests.append(
            DiagnosticTest(
                name=name,
                status=TestStatus.IN_PROGRESS,
                started_at=utc_now(),
            )
        )

    def _test_completed(
        self,
        event: DomainEvent,
    ) -> None:
        name = event.payload.get(
            "name"
        )

        if not isinstance(
            name,
            str,
        ):
            raise StateTransitionError(
                "TEST_COMPLETED requires "
                "'name'"
            )

        name = name.strip()

        if not name:
            raise StateTransitionError(
                "Test name cannot be empty"
            )

        test = self._find_test(name)

        if test is None:
            raise StateTransitionError(
                f"Cannot complete unknown "
                f"test: {name}"
            )

        if (
            test.status
            != TestStatus.IN_PROGRESS
        ):
            raise StateTransitionError(
                f"Test is not in progress: "
                f"{name}"
            )

        try:
            status = TestStatus(
                event.payload.get(
                    "status",
                    TestStatus.PASSED.value,
                )
            )
        except ValueError as exc:
            raise StateTransitionError(
                f"Invalid test status for: {name}"
            ) from exc

        if status == TestStatus.IN_PROGRESS:
            raise StateTransitionError(
                "TEST_COMPLETED cannot leave "
                "a test in progress"
            )

        result = event.payload.get(
            "result"
        )

        if result is not None and not isinstance(
            result,
            str,
        ):
            raise StateTransitionError(
                "Test result must be "
                "a string or None"
            )

        test.status = status
        test.result = result
        test.completed_at = utc_now()
        test.notes = event.payload.get(
            "notes"
        )

    # =========================================================
    # HYPOTHESES
    # =========================================================

    def _hypothesis_created(
        self,
        event: DomainEvent,
    ) -> None:
        description = event.payload.get(
            "description"
        )

        if not isinstance(
            description,
            str,
        ):
            raise StateTransitionError(
                "HYPOTHESIS_CREATED requires "
                "'description'"
            )

        description = description.strip()

        if not description:
            raise StateTransitionError(
                "Hypothesis description "
                "cannot be empty"
            )

        confidence = float(
            event.payload.get(
                "confidence",
                0.5,
            )
        )

        if not 0.0 <= confidence <= 1.0:
            raise StateTransitionError(
                "Hypothesis confidence must "
                "be between 0 and 1"
            )

        try:
            status = HypothesisStatus(
                event.payload.get(
                    "status",
                    HypothesisStatus.POSSIBLE.value,
                )
            )
        except ValueError as exc:
            raise StateTransitionError(
                "Invalid hypothesis status"
            ) from exc

        supporting = event.payload.get(
            "supporting_evidence",
            [],
        )

        contradicting = event.payload.get(
            "contradicting_evidence",
            [],
        )

        hypothesis = Hypothesis(
            description=description,
            status=status,
            confidence=confidence,
            supporting_evidence=list(
                supporting
            ),
            contradicting_evidence=list(
                contradicting
            ),
        )

        self.session.diagnostic.hypotheses.append(
            hypothesis
        )

        if status in (
            HypothesisStatus.LIKELY,
            HypothesisStatus.CONFIRMED,
        ):
            self.session.diagnostic.current_hypothesis = (
                description
            )

    def _hypothesis_updated(
        self,
        event: DomainEvent,
    ) -> None:
        description = event.payload.get(
            "description"
        )

        if not isinstance(
            description,
            str,
        ):
            raise StateTransitionError(
                "HYPOTHESIS_UPDATED requires "
                "'description'"
            )

        hypothesis = self._find_hypothesis(
            description
        )

        if hypothesis is None:
            raise StateTransitionError(
                f"Unknown hypothesis: "
                f"{description}"
            )

        if "status" in event.payload:
            try:
                hypothesis.status = (
                    HypothesisStatus(
                        event.payload["status"]
                    )
                )
            except ValueError as exc:
                raise StateTransitionError(
                    "Invalid hypothesis status"
                ) from exc

        if "confidence" in event.payload:
            confidence = float(
                event.payload["confidence"]
            )

            if not 0.0 <= confidence <= 1.0:
                raise StateTransitionError(
                    "Hypothesis confidence must "
                    "be between 0 and 1"
                )

            hypothesis.confidence = confidence

        if "supporting_evidence" in event.payload:
            hypothesis.supporting_evidence.extend(
                event.payload[
                    "supporting_evidence"
                ]
            )

        if "contradicting_evidence" in event.payload:
            hypothesis.contradicting_evidence.extend(
                event.payload[
                    "contradicting_evidence"
                ]
            )

        if (
            hypothesis.status
            in (
                HypothesisStatus.LIKELY,
                HypothesisStatus.CONFIRMED,
            )
        ):
            self.session.diagnostic.current_hypothesis = (
                hypothesis.description
            )

        elif (
            self.session.diagnostic.current_hypothesis
            == hypothesis.description
            and hypothesis.status
            == HypothesisStatus.REJECTED
        ):
            self.session.diagnostic.current_hypothesis = None

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    def _recommendation_created(
        self,
        event: DomainEvent,
    ) -> None:
        action = event.payload.get(
            "action"
        )

        if not isinstance(
            action,
            str,
        ):
            raise StateTransitionError(
                "RECOMMENDATION_CREATED "
                "requires 'action'"
            )

        action = action.strip()

        if not action:
            raise StateTransitionError(
                "Recommendation action "
                "cannot be empty"
            )

        self.session.diagnostic.next_recommended_action = (
            action
        )

    # =========================================================
    # RETRIEVAL
    # =========================================================

    def _procedure_retrieved(
        self,
        event: DomainEvent,
    ) -> None:
        """
        Retrieval is evidence.

        Merely retrieving a procedure/memory must NOT mutate
        diagnostic state.

        If retrieved information becomes a diagnostic fact,
        hypothesis, observation, or recommendation, that must
        happen through its own explicit DomainEvent.
        """

        return

    # =========================================================
    # RESOLUTION
    # =========================================================

    def _resolution_proposed(
        self,
        event: DomainEvent,
    ) -> None:
        resolution = event.payload.get(
            "resolution"
        )

        if not isinstance(
            resolution,
            str,
        ):
            raise StateTransitionError(
                "RESOLUTION_PROPOSED requires "
                "'resolution'"
            )

        resolution = resolution.strip()

        if not resolution:
            raise StateTransitionError(
                "Proposed resolution "
                "cannot be empty"
            )

        self.session.diagnostic.next_recommended_action = (
            resolution
        )

    def _resolution_confirmed(
        self,
        event: DomainEvent,
    ) -> None:
        resolution = event.payload.get(
            "resolution"
        )

        if resolution is None:
            resolution = (
                self.session
                .diagnostic
                .next_recommended_action
            )

        if not isinstance(
            resolution,
            str,
        ):
            raise StateTransitionError(
                "RESOLUTION_CONFIRMED requires "
                "a confirmed resolution"
            )

        resolution = resolution.strip()

        if not resolution:
            raise StateTransitionError(
                "Confirmed resolution "
                "cannot be empty"
            )

        self.session.diagnostic.confirmed_resolution = (
            resolution
        )

        self.session.diagnostic.case_status = (
            "resolved"
        )

        self.session.diagnostic.next_recommended_action = None

    # =========================================================
    # CASE
    # =========================================================

    def _case_closed(
        self,
        event: DomainEvent,
    ) -> None:
        self.session.diagnostic.case_status = (
            "closed"
        )

    # =========================================================
    # HELPERS
    # =========================================================

    def _find_test(
        self,
        name: str,
    ) -> DiagnosticTest | None:
        for test in self.session.diagnostic.tests:
            if test.name == name:
                return test

        return None

    def _find_hypothesis(
        self,
        description: str,
    ) -> Hypothesis | None:
        for hypothesis in self.session.diagnostic.hypotheses:
            if hypothesis.description == description:
                return hypothesis

        return None


__all__ = [
    "StateTransitionError",
    "StaleEventError",
    "StateEngine",
]