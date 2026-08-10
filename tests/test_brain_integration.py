from __future__ import annotations

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fieldmate.brain.models import BrainResult, DiagnosticContext
from fieldmate.brain.runtime import build_brain_runtime


@pytest.mark.asyncio
async def test_brain_process_technical_turn():
    """Verify that canonical Brain processes a technical turn and applies state events."""

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        runtime = build_brain_runtime(session_id="test_session_123")
        brain = runtime.brain

        mock_context_result = MagicMock(
            context=DiagnosticContext(),
            plan=MagicMock(mode="hybrid"),
            latency_ms=10.0,
            timed_out=False,
            prefetched=False,
            relevant=True,
        )

        mock_llm_json = json.dumps(
            {
                "response": "Let's check the RAM modules and run Windows Memory Diagnostic.",
                "hypothesis": "Faulty RAM module causing BSOD",
                "confidence": 0.8,
                "next_action": "Run Windows Memory Diagnostic (mdsched.exe)",
                "clarification_needed": False,
                "clarification_question": None,
                "evidence_ids": [],
                "state_updates": [
                    {
                        "type": "fault_identified",
                        "payload": {"fault_code": "WHEA_UNCORRECTABLE_ERROR"},
                    }
                ],
                "resolution_proposed": None,
                "resolution_confirmed": None,
            }
        )

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_json))]

        with patch.object(brain.retrieval, "retrieve", new_callable=AsyncMock) as mock_retrieve, \
             patch.object(brain.reasoning.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:

            mock_retrieve.return_value = mock_context_result
            mock_create.return_value = mock_completion

            result = await brain.process(
                "My Lenovo ThinkPad is crashing with WHEA_UNCORRECTABLE_ERROR",
                technical=True,
            )

            assert isinstance(result, BrainResult)
            assert result.response == "Let's check the RAM modules and run Windows Memory Diagnostic."
            assert result.turn == 1
            assert result.generation == 1

            # Verify state engine updated
            session = brain.state.session
            assert "WHEA_UNCORRECTABLE_ERROR" in session.diagnostic.fault_codes
            assert session.diagnostic.hypotheses[0].description == "Faulty RAM module causing BSOD"
            assert session.diagnostic.next_recommended_action == "Run Windows Memory Diagnostic (mdsched.exe)"


@pytest.mark.asyncio
async def test_brain_process_non_technical_turn():
    """Verify that non-technical chat turns bypass retrieval and state updates."""

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        runtime = build_brain_runtime(session_id="test_session_456")
        brain = runtime.brain

        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content="Hello! I'm FieldMate. What machine are we working on today?"))
        ]

        with patch.object(brain.reasoning.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_completion

            result = await brain.process("Hello there", technical=False)

            assert isinstance(result, BrainResult)
            assert "FieldMate" in result.response
            assert result.retrieved is False

            # Verify state engine was not mutated with diagnostic updates
            session = brain.state.session
            assert len(session.diagnostic.fault_codes) == 0
            assert session.diagnostic.current_hypothesis is None
