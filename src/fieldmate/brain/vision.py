from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("fieldmate.vision")

DEFAULT_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "qwen/qwen3.6-27b",
)
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB safety limit


@dataclass
class VisualAnalysis:
    """
    Structured visual evidence extracted from a physical inspection snapshot.

    Vision observes; it does NOT formulate final diagnoses.
    """
    visual_facts: list[str] = field(default_factory=list)
    hardware_identifiers: dict[str, str] = field(default_factory=dict)
    ocr_text: str | None = None
    uncertain_observations: list[str] = field(default_factory=list)
    suggested_camera_angle: str | None = None


def detect_image_mime(image_bytes: bytes) -> str:
    """
    Detect image MIME type from magic bytes.
    """
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"RIFF") and len(image_bytes) >= 12 and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"  # Default fallback


def _parse_loose_json(text: str) -> dict[str, Any]:
    """
    Extract and parse JSON dictionary resiliently from raw LLM output.
    """
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    
    candidate = text
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if match:
        candidate = match.group(1).strip()
    elif "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        candidate = text[start:end].strip()

    # Try standard json parse
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Try stripping trailing commas before closing braces/brackets
    try:
        fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
        data = json.loads(fixed)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback regex extraction of visual facts
    facts_match = re.findall(r'"visual_facts"\s*:\s*\[(.*?)\]', text, flags=re.DOTALL)
    facts = []
    if facts_match:
        items = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', facts_match[0])
        facts = [i for i in items if i.strip()]

    uncertain = []
    if not facts and text:
        uncertain = [f"Visual extraction failed to parse JSON from output: {text[:80]}"]

    return {"visual_facts": facts, "uncertain_observations": uncertain}


class VisionEngine:
    """
    Extracts objective physical hardware observations, OCR text, and
    hardware identifiers from inspection images using Llama 4 Scout on Groq.
    """

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_VISION_MODEL,
    ):
        self.client = client
        self.model = model

    async def analyze(
        self,
        image_bytes: bytes,
        user_utterance: str | None = None,
        state: dict[str, Any] | None = None,
    ) -> VisualAnalysis:
        """
        Analyze an inspection frame and return structured visual observations.
        """
        if not image_bytes:
            return VisualAnalysis()

        if len(image_bytes) > MAX_IMAGE_BYTES:
            logger.warning(
                f"Image size {len(image_bytes)} exceeds maximum limit {MAX_IMAGE_BYTES} bytes."
            )
            return VisualAnalysis(
                uncertain_observations=["Image size exceeded 5MB limit; could not inspect full detail."]
            )

        mime_type = detect_image_mime(image_bytes)
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:{mime_type};base64,{encoded_image}"

        system_prompt = (
            "You are FieldMate's visual evidence extractor for PC hardware and software troubleshooting.\n"
            "Your job is ONLY to extract observable physical facts from the image. You do NOT make final diagnoses.\n\n"
            "INSTRUCTIONS:\n"
            "1. Extract directly visible physical facts (e.g. 'fan blades visibly spinning', 'CPU diagnostic Q-LED illuminated red', 'thermal paste dried on heatsink', 'RAM latch unclipped').\n"
            "2. Read any visible text/OCR (e.g. BSOD stop code '0x00000124', 'WHEA_UNCORRECTABLE_ERROR', BIOS error message, serial sticker).\n"
            "3. Identify visible hardware specs/models (e.g. 'motherboard: ASUS ROG Strix B550', 'laptop: Lenovo ThinkPad T14').\n"
            "4. Report any uncertain or blurry observations in 'uncertain_observations'.\n"
            "5. If the angle or lighting is poor, provide 'suggested_camera_angle'.\n"
            "6. Never infer an internal component failure solely from appearance. Never invent serials or error codes.\n\n"
            "Return valid JSON matching this schema:\n"
            "{\n"
            '  "visual_facts": ["string fact 1", "string fact 2"],\n'
            '  "hardware_identifiers": {"model": "...", "oem": "...", "serial": "..."},\n'
            '  "ocr_text": "raw or key text extracted from screen or sticker or null",\n'
            '  "uncertain_observations": ["uncertain item 1"],\n'
            '  "suggested_camera_angle": "camera advice if needed or null"\n'
            "}"
        )

        user_content: list[dict[str, Any]] = []
        text_prompt = "Extract observable facts and text from this physical inspection image."

        if user_utterance:
            text_prompt += f" The technician stated: '{user_utterance.strip()}'."

        if state:
            symptoms = state.get("symptoms", [])
            symptom_names = [
                s.get("name") if isinstance(s, dict) else getattr(s, "name", str(s))
                for s in symptoms
            ]
            fault_codes = state.get("fault_codes", [])
            light_state = {
                "reported_symptoms": symptom_names,
                "fault_codes": fault_codes,
            }
            text_prompt += f" Current diagnostic context: {json.dumps(light_state)}."

        user_content.append({"type": "text", "text": text_prompt})
        user_content.append({
            "type": "image_url",
            "image_url": {"url": image_url},
        })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        content = "{}"
        try:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=1000,
                )
                content = response.choices[0].message.content or "{}"
            except Exception:
                # Fallback for models without strict json_object grammar support
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000,
                )
                content = response.choices[0].message.content or "{}"
        except Exception as exc:
            logger.error(f"VisionEngine API call failed: {exc}")
            return VisualAnalysis(
                uncertain_observations=[f"Visual inspection failed: {str(exc)}"]
            )

        data = _parse_loose_json(content)

        # Normalize outputs safely
        visual_facts = [
            str(f).strip()
            for f in data.get("visual_facts", [])
            if isinstance(f, (str, int, float)) and str(f).strip()
        ]
        
        raw_hw = data.get("hardware_identifiers")
        hw_ids: dict[str, str] = {}
        if isinstance(raw_hw, dict):
            for k, v in raw_hw.items():
                if v and isinstance(v, (str, int, float)):
                    hw_ids[str(k).strip()] = str(v).strip()

        ocr_text = data.get("ocr_text")
        if ocr_text and isinstance(ocr_text, str):
            ocr_text = ocr_text.strip() or None
        else:
            ocr_text = None

        uncertainties = [
            str(u).strip()
            for u in data.get("uncertain_observations", [])
            if isinstance(u, (str, int, float)) and str(u).strip()
        ]

        suggested_angle = data.get("suggested_camera_angle")
        if suggested_angle and isinstance(suggested_angle, str):
            suggested_angle = suggested_angle.strip() or None
        else:
            suggested_angle = None

        return VisualAnalysis(
            visual_facts=visual_facts,
            hardware_identifiers=hw_ids,
            ocr_text=ocr_text,
            uncertain_observations=uncertainties,
            suggested_camera_angle=suggested_angle,
        )
