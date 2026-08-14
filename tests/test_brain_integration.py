from __future__ import annotations

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fieldmate.brain.models import BrainResult, DiagnosticContext
from fieldmate.brain.runtime import build_brain_runtime
from fieldmate.brain.vision import VisualAnalysis


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


@pytest.mark.asyncio
async def test_brain_process_image_bsod_ocr():
    """Verify Judge Scenario 3: Camera snapshot extracts BSOD stop-code via OCR and updates state/retrieval."""

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        runtime = build_brain_runtime(session_id="test_session_ocr")
        brain = runtime.brain

        mock_analysis = VisualAnalysis(
            visual_facts=["Blue Screen error displayed on laptop LCD"],
            hardware_identifiers={"oem": "Dell", "model": "Latitude 5420"},
            ocr_text="Your device ran into a problem. Stop code: 0x00000124 WHEA_UNCORRECTABLE_ERROR",
        )

        mock_context_result = MagicMock(
            context=DiagnosticContext(),
            plan=MagicMock(mode="hybrid"),
            latency_ms=12.0,
            timed_out=False,
            prefetched=False,
            relevant=True,
        )

        mock_llm_json = json.dumps(
            {
                "response": "I see the BSOD screen with stop code 0x00000124 (WHEA_UNCORRECTABLE_ERROR). This indicates a hardware error, often related to CPU voltage or unstable thermals.",
                "hypothesis": "Hardware parity or CPU voltage instability",
                "confidence": 0.85,
                "next_action": "Check system thermals and verify CPU voltage in BIOS",
                "clarification_needed": False,
                "clarification_question": None,
                "evidence_ids": [],
                "state_updates": [],
                "resolution_proposed": None,
                "resolution_confirmed": None,
            }
        )

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_json))]

        with patch.object(brain.vision, "analyze", new_callable=AsyncMock) as mock_analyze, \
             patch.object(brain.retrieval, "retrieve", new_callable=AsyncMock) as mock_retrieve, \
             patch.object(brain.reasoning.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:

            mock_analyze.return_value = mock_analysis
            mock_retrieve.return_value = mock_context_result
            mock_create.return_value = mock_completion

            result = await brain.process_image(
                image_bytes=b"\xff\xd8\xfffake_bsod_image",
                user_utterance="Laptop crashed with blue screen",
            )

            assert isinstance(result, BrainResult)
            assert "0x00000124" in result.response

            session = brain.state.session
            assert "0x00000124" in session.diagnostic.fault_codes
            assert session.diagnostic.equipment.manufacturer == "Dell"
            assert session.diagnostic.equipment.model == "Latitude 5420"
            assert any(obs.name == "ocr_text" for obs in session.diagnostic.observations)


@pytest.mark.asyncio
async def test_brain_process_image_contradiction():
    """Verify Judge Scenario 1: User says fan isn't spinning, camera sees fan spinning -> Clarification requested."""

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        runtime = build_brain_runtime(session_id="test_session_conflict")
        brain = runtime.brain

        mock_analysis = VisualAnalysis(
            visual_facts=["Cooling fan blades are visibly spinning at normal RPM", "Rear exhaust clear"],
            hardware_identifiers={"oem": "Lenovo", "model": "ThinkPad T14"},
            ocr_text=None,
        )

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
                "response": "I can see the cooling fan blades actively spinning in the image. Does it stop intermittently, or are you observing high temperatures despite the fan running?",
                "hypothesis": "Possible thermal sensor misreporting or intermittent fan stall",
                "confidence": 0.6,
                "next_action": "Check CPU temperature sensor in HWInfo or BIOS",
                "clarification_needed": True,
                "clarification_question": "Does the fan stop intermittently?",
                "evidence_ids": [],
                "state_updates": [],
                "resolution_proposed": None,
                "resolution_confirmed": None,
            }
        )

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_json))]

        with patch.object(brain.vision, "analyze", new_callable=AsyncMock) as mock_analyze, \
             patch.object(brain.retrieval, "retrieve", new_callable=AsyncMock) as mock_retrieve, \
             patch.object(brain.reasoning.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:

            mock_analyze.return_value = mock_analysis
            mock_retrieve.return_value = mock_context_result
            mock_create.return_value = mock_completion

            result = await brain.process_image(
                image_bytes=b"\xff\xd8\xfffake_fan_image",
                user_utterance="The cooling fan is dead and not spinning at all",
            )

            assert isinstance(result, BrainResult)
            assert "spinning" in result.response
            assert result.decision is not None
            assert result.decision.clarification_needed is True


@pytest.mark.asyncio
async def test_brain_process_image_motherboard_qled():
    """Verify Judge Scenario 2: User doesn't know problem -> Camera sees ASUS motherboard with CPU Q-LED red."""

    with patch.dict(os.environ, {"GROQ_API_KEY": "fake_key"}):
        runtime = build_brain_runtime(session_id="test_session_qled")
        brain = runtime.brain

        mock_analysis = VisualAnalysis(
            visual_facts=["Motherboard CPU diagnostic Q-LED illuminated solid RED", "RAM sticks seated in slots A2/B2"],
            hardware_identifiers={"oem": "ASUS", "model": "ROG Strix B550-F"},
            ocr_text="ASUS ROG STRIX B550-F GAMING",
        )

        mock_context_result = MagicMock(
            context=DiagnosticContext(),
            plan=MagicMock(mode="hybrid"),
            latency_ms=15.0,
            timed_out=False,
            prefetched=False,
            relevant=True,
        )

        mock_llm_json = json.dumps(
            {
                "response": "I can see the red CPU diagnostic LED is lit on your ASUS ROG Strix B550-F motherboard. This indicates the CPU failed POST. Let's verify the 8-pin CPU power cable is securely connected.",
                "hypothesis": "CPU power disconnect or seating fault",
                "confidence": 0.85,
                "next_action": "Check 8-pin EPS 12V CPU power connector at top left of motherboard",
                "clarification_needed": False,
                "clarification_question": None,
                "evidence_ids": [],
                "state_updates": [],
                "resolution_proposed": None,
                "resolution_confirmed": None,
            }
        )

        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(message=MagicMock(content=mock_llm_json))]

        with patch.object(brain.vision, "analyze", new_callable=AsyncMock) as mock_analyze, \
             patch.object(brain.retrieval, "retrieve", new_callable=AsyncMock) as mock_retrieve, \
             patch.object(brain.reasoning.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:

            mock_analyze.return_value = mock_analysis
            mock_retrieve.return_value = mock_context_result
            mock_create.return_value = mock_completion

            result = await brain.process_image(
                image_bytes=b"\xff\xd8\xfffake_motherboard_image",
                user_utterance="PC won't boot and screen is black, I don't know what's wrong",
            )

            assert isinstance(result, BrainResult)
            assert "CPU diagnostic LED" in result.response
            assert "8-pin" in result.decision.next_action

            session = brain.state.session
            assert session.diagnostic.equipment.manufacturer == "Asus"
            assert session.diagnostic.equipment.model == "ROG Strix B550-F"
            assert any("CPU diagnostic Q-LED" in str(obs.value) for obs in session.diagnostic.observations)
