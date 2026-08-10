from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from fieldmate.brain.models import (
    DiagnosticContext,
    DiagnosticDecision,
)


@dataclass(slots=True)
class ReasoningManager:
    """
    LLM reasoning boundary.

    The LLM can propose diagnostic decisions and state updates.

    It cannot directly mutate canonical FieldMate state.
    """

    client: Any
    model: str
    system_prompt: str
    max_tokens: int = 600

    async def reason(
        self,
        *,
        state: dict[str, Any],
        context: DiagnosticContext,
        user_input: str,
        turn: int,
        generation: int,
    ) -> tuple[DiagnosticDecision, float]:

        started = time.perf_counter()

        payload = {
            "turn": turn,
            "generation": generation,
            "current_state": state,
            "current_observation": user_input,

            "evidence": [
                self._evidence_dict(item)
                for item in context.evidence
            ],

            "supporting_evidence": [
                self._evidence_dict(item)
                for item in context.supporting
            ],

            "contradicting_evidence": [
                self._evidence_dict(item)
                for item in context.contradicting
            ],

            "neutral_evidence": [
                self._evidence_dict(item)
                for item in context.neutral
            ],

            "procedures": [
                self._evidence_dict(item)
                for item in context.procedures
            ],

            "past_cases": [
                self._evidence_dict(item)
                for item in context.past_cases
            ],

            "resolutions": [
                self._evidence_dict(item)
                for item in context.resolutions
            ],

            "instructions": {
                "do_not_invent_facts": True,
                "do_not_repeat_completed_tests": True,
                "prefer_discriminating_tests": True,
                "respect_evidence_strength": True,
                "preserve_contradictions": True,
                "do_not_claim_resolution_without_confirmation": True,
                "use_only_provided_evidence": True,
                "state_updates_must_be_explicit": True,
            },
        }

        completion = await self._complete(payload)

        decision = self._parse(completion)

        latency = (
            time.perf_counter() - started
        ) * 1000

        return decision, latency

    async def chat(
        self,
        *,
        user_input: str,
        turn: int,
        generation: int,
        conversation: list[dict[str, str]] | None = None,
    ) -> tuple[DiagnosticDecision, float]:
        """
        Direct conversational LLM path.

        This path is deliberately separate from diagnostic
        reasoning.

        It does not receive Qdrant evidence and it does not
        produce state updates.

        General conversation must not mutate diagnostic state.
        """

        started = time.perf_counter()

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are FieldMate, a natural and friendly "
                    "PC troubleshooting assistant. "
                    "For this turn the user is having a general "
                    "conversation rather than asking for a "
                    "technical diagnosis. Respond naturally and "
                    "concisely. Do not invent live information. "
                    "Do not discuss or modify diagnostic state. "
                    "Do not output JSON. "
                    "If the user asks about something outside "
                    "your capabilities, say so naturally."
                ),
            }
        ]

        if conversation:
            for message in conversation[-10:]:
                if (
                    isinstance(message, dict)
                    and message.get("role") in {
                        "user",
                        "assistant",
                    }
                    and isinstance(
                        message.get("content"),
                        str,
                    )
                ):
                    messages.append(
                        {
                            "role": message["role"],
                            "content": message["content"],
                        }
                    )

        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,
        )

        content = (
            response.choices[0]
            .message
            .content
            or ""
        ).strip()

        if not content:
            content = (
                "I'm here. What can I help you with?"
            )

        decision = DiagnosticDecision(
            response=content,
            confidence=None,
            hypothesis=None,
            next_action=None,
            clarification_needed=False,
            clarification_question=None,
            evidence_ids=(),
            state_updates=(),
            resolution_proposed=None,
            resolution_confirmed=None,
        )

        latency = (
            time.perf_counter()
            - started
        ) * 1000

        return decision, latency

    async def _complete(
        self,
        payload: dict[str, Any],
    ) -> str:

        prompt = (
            self.system_prompt
            + "\n\nDIAGNOSTIC PACKET:\n"
            + json.dumps(
                payload,
                ensure_ascii=False,
                default=str,
            )
        )

        response = (
            await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": prompt,
                    }
                ],
                temperature=0,
                max_tokens=self.max_tokens,
            )
        )

        return (
            response.choices[0]
            .message
            .content
            or ""
        )

    @staticmethod
    def _parse(
        raw: str,
    ) -> DiagnosticDecision:

        raw = raw.strip()

        if not raw:
            return DiagnosticDecision(
                response=(
                    "I could not produce a diagnostic "
                    "response."
                )
            )

        try:
            data = json.loads(raw)

        except json.JSONDecodeError:
            return DiagnosticDecision(
                response=raw
            )

        if not isinstance(data, dict):
            return DiagnosticDecision(
                response=raw
            )

        confidence = data.get("confidence")

        if confidence is not None:
            try:
                confidence = max(
                    0.0,
                    min(
                        1.0,
                        float(confidence),
                    ),
                )
            except (TypeError, ValueError):
                confidence = None

        evidence_ids = data.get(
            "evidence_ids",
            (),
        )

        if not isinstance(
            evidence_ids,
            (list, tuple),
        ):
            evidence_ids = ()

        state_updates = data.get(
            "state_updates",
            (),
        )

        if not isinstance(
            state_updates,
            (list, tuple),
        ):
            state_updates = ()

        state_updates = tuple(
            update
            for update in state_updates
            if isinstance(update, dict)
        )

        return DiagnosticDecision(
            response=str(
                data.get(
                    "response",
                    "",
                )
            ).strip(),

            hypothesis=data.get(
                "hypothesis"
            ),

            confidence=confidence,

            next_action=data.get(
                "next_action"
            ),

            clarification_needed=bool(
                data.get(
                    "clarification_needed",
                    False,
                )
            ),

            clarification_question=data.get(
                "clarification_question"
            ),

            evidence_ids=tuple(
                str(item)
                for item in evidence_ids
            ),

            state_updates=state_updates,

            resolution_proposed=data.get(
                "resolution_proposed"
            ),

            resolution_confirmed=data.get(
                "resolution_confirmed"
            ),
        )

    @staticmethod
    def _evidence_dict(
        evidence: Any,
    ) -> dict[str, Any]:
        """
        Convert canonical retrieval Evidence into the
        JSON-safe representation consumed by Groq.

        The reasoning layer knows only the Evidence contract.
        Qdrant types never cross this boundary.
        """

        def enum_value(value: Any) -> Any:
            if hasattr(value, "value"):
                return value.value
            return value

        return {
            "id": getattr(
                evidence,
                "evidence_id",
                None,
            ),
            "memory_id": getattr(
                evidence,
                "memory_id",
                None,
            ),
            "type": enum_value(
                getattr(
                    evidence,
                    "evidence_type",
                    None,
                )
            ),
            "content": getattr(
                evidence,
                "content",
                "",
            ),
            "source": getattr(
                evidence,
                "source",
                None,
            ),
            "source_type": getattr(
                evidence,
                "source_type",
                None,
            ),
            "equipment_model": getattr(
                evidence,
                "equipment_model",
                None,
            ),
            "equipment_family": getattr(
                evidence,
                "equipment_family",
                None,
            ),
            "oem": getattr(
                evidence,
                "oem",
                None,
            ),
            "system": getattr(
                evidence,
                "system",
                None,
            ),
            "subsystem": getattr(
                evidence,
                "subsystem",
                None,
            ),
            "component": getattr(
                evidence,
                "component",
                None,
            ),
            "fault_code": getattr(
                evidence,
                "fault_code",
                None,
            ),
            "symptom": getattr(
                evidence,
                "symptom",
                None,
            ),
            "confidence": getattr(
                evidence,
                "confidence",
                0.0,
            ),
            "relevance": getattr(
                evidence,
                "relevance_score",
                0.0,
            ),
            "verified": getattr(
                evidence,
                "verified",
                False,
            ),
            "relation": enum_value(
                getattr(
                    evidence,
                    "relation",
                    None,
                )
            ),
            "case_id": getattr(
                evidence,
                "case_id",
                None,
            ),
            "provenance": getattr(
                evidence,
                "provenance",
                (),
            ),
            "retrieval_mode": getattr(
                evidence,
                "retrieval_mode",
                None,
            ),
        }
