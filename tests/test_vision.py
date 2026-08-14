import pytest
import io
import os
import sys
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fieldmate.brain.vision import VisionEngine, VisualAnalysis, detect_image_mime
from fieldmate.brain.models import BrainResult, DiagnosticDecision
from fieldmate.brain.runtime import build_brain_runtime
from main import app


def test_detect_image_mime():
    jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00"
    webp_bytes = b"RIFF\x20\x00\x00\x00WEBPVP8 "
    other_bytes = b"random_bytes"

    assert detect_image_mime(jpeg_bytes) == "image/jpeg"
    assert detect_image_mime(png_bytes) == "image/png"
    assert detect_image_mime(webp_bytes) == "image/webp"
    assert detect_image_mime(other_bytes) == "image/jpeg"


@pytest.mark.asyncio
async def test_vision_engine_success():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = (
        '{"visual_facts": ["CPU Q-LED is red", "Fan blades rotating"], '
        '"hardware_identifiers": {"model": "ThinkPad T14", "oem": "Lenovo"}, '
        '"ocr_text": "WHEA_UNCORRECTABLE_ERROR 0x00000124", '
        '"uncertain_observations": ["Possible dust buildup in vent"], '
        '"suggested_camera_angle": "Move closer to RAM slots"}'
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    engine = VisionEngine(client=mock_client)
    res = await engine.analyze(
        image_bytes=b"\xff\xd8\xff\xe0\x00\x10testimage",
        user_utterance="Laptop won't boot",
    )

    assert isinstance(res, VisualAnalysis)
    assert "CPU Q-LED is red" in res.visual_facts
    assert "Fan blades rotating" in res.visual_facts
    assert res.hardware_identifiers == {"model": "ThinkPad T14", "oem": "Lenovo"}
    assert res.ocr_text == "WHEA_UNCORRECTABLE_ERROR 0x00000124"
    assert res.uncertain_observations == ["Possible dust buildup in vent"]
    assert res.suggested_camera_angle == "Move closer to RAM slots"


@pytest.mark.asyncio
async def test_vision_engine_malformed_json_fallback():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Invalid not json at all"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    engine = VisionEngine(client=mock_client)
    res = await engine.analyze(image_bytes=b"\xff\xd8\xff\xe0test")

    assert isinstance(res, VisualAnalysis)
    assert len(res.uncertain_observations) > 0
    assert "failed" in res.uncertain_observations[0].lower()


@pytest.mark.asyncio
async def test_vision_engine_empty_image():
    mock_client = MagicMock()
    engine = VisionEngine(client=mock_client)
    res = await engine.analyze(image_bytes=b"")
    assert res.visual_facts == []
    assert res.ocr_text is None


@pytest.mark.asyncio
async def test_brain_process_image_end_to_end():
    runtime = build_brain_runtime(session_id="test_vision_room")
    runtime.brain.retrieval_enabled = False
    
    vision_mock = MagicMock()
    v_choice = MagicMock()
    v_choice.message.content = (
        '{"visual_facts": ["Lenovo logo visible on chassis", "Thermal paste dried out"], '
        '"hardware_identifiers": {"model": "ThinkPad T14", "oem": "Lenovo"}, '
        '"ocr_text": "0x00000124", '
        '"uncertain_observations": [], '
        '"suggested_camera_angle": null}'
    )
    v_resp = MagicMock()
    v_resp.choices = [v_choice]
    vision_mock.chat.completions.create = AsyncMock(return_value=v_resp)
    runtime.brain.vision.client = vision_mock

    # Mock reasoning manager with separate client
    reasoning_mock = MagicMock()
    r_choice = MagicMock()
    r_choice.message.content = (
        '{"response": "I see dried thermal paste on your Lenovo ThinkPad.", '
        '"confidence": 0.95, '
        '"hypothesis": "Thermal throttling from dry paste", '
        '"next_action": "Reapply thermal paste and reseat cooler", '
        '"clarification_needed": false, '
        '"clarification_question": null, '
        '"evidence_ids": [], "state_updates": []}'
    )
    r_resp = MagicMock()
    r_resp.choices = [r_choice]
    reasoning_mock.chat.completions.create = AsyncMock(return_value=r_resp)
    runtime.brain.reasoning.client = reasoning_mock

    res = await runtime.brain.process_image(
        image_bytes=b"\xff\xd8\xff\xe0\x00\x10testimage",
        user_utterance="My laptop is overheating",
    )

    assert isinstance(res, BrainResult)
    assert res.decision is not None
    assert "thermal paste" in res.decision.response.lower()
    assert res.decision.hypothesis == "Thermal throttling from dry paste"
    assert runtime.brain.state.session.diagnostic.equipment.manufacturer == "Lenovo"
    assert runtime.brain.state.session.diagnostic.equipment.model == "ThinkPad T14"
    assert "0x00000124" in runtime.brain.state.session.diagnostic.fault_codes


def test_fastapi_inspect_endpoint():
    client = TestClient(app)
    fake_img = io.BytesIO(b"\xff\xd8\xff\xe0\x00\x10JFIFfakeimagecontent")
    
    response = client.post(
        "/api/inspect",
        files={"image": ("test.jpg", fake_img, "image/jpeg")},
        data={"session_id": "test_endpoint_room", "user_utterance": "Help diagnose"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "hypothesis" in data
    assert "confidence" in data
    assert "next_action" in data


