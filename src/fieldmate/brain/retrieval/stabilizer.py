from __future__ import annotations

import re
from dataclasses import dataclass, field


# ============================================================
# ENTITY PATTERNS
# ============================================================

OEM_PATTERNS = {
    "dell": re.compile(r"\bdell\b", re.IGNORECASE),
    "lenovo": re.compile(r"\blenovo\b", re.IGNORECASE),
    "hp": re.compile(
        r"\b(?:hp|hewlett[\s_-]*packard)\b",
        re.IGNORECASE,
    ),
    "asus": re.compile(r"\basus\b", re.IGNORECASE),
    "acer": re.compile(r"\bacer\b", re.IGNORECASE),
    "msi": re.compile(r"\bmsi\b", re.IGNORECASE),
}


# Exact diagnostic identifiers.
FAULT_PATTERNS = (
    re.compile(
        r"\b(?:"
        r"e|err|error|event|fault"
        r")[\s_-]?\d{1,5}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b0x[0-9a-f]{2,16}\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhea(?:[\s_-]+logger)?"
        r"(?:[\s_-]+\d+)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bstop[\s_-]+0x[0-9a-f]+\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b[a-z0-9]+_"
        r"(?:IRQL|MANAGEMENT|FAULT|ERROR|EXCEPTION)"
        r"\b",
        re.IGNORECASE,
    ),
)


# Keep these normalized to canonical names rather than returning
# every possible wording as a separate component.
COMPONENT_PATTERNS = {
    "wifi": (
        "wi-fi",
        "wifi",
        "wireless",
    ),
    "ethernet": (
        "ethernet",
        "lan",
    ),
    "network_adapter": (
        "network adapter",
        "network card",
        "nic",
    ),
    "bluetooth": (
        "bluetooth",
    ),
    "ram": (
        "ram",
        "memory",
    ),
    "ssd": (
        "ssd",
        "nvme",
    ),
    "hdd": (
        "hdd",
        "hard drive",
        "hard disk",
    ),
    "storage": (
        "storage",
        "disk",
        "drive",
    ),
    "battery": (
        "battery",
    ),
    "charging": (
        "charging",
        "charger",
        "power adapter",
        "ac adapter",
    ),
    "fan": (
        "fan",
    ),
    "cooling": (
        "cooling",
        "thermal",
    ),
    "cpu": (
        "cpu",
        "processor",
    ),
    "gpu": (
        "gpu",
        "graphics card",
        "graphics",
    ),
    "display": (
        "display",
        "screen",
        "monitor",
    ),
    "keyboard": (
        "keyboard",
    ),
    "touchpad": (
        "touchpad",
        "trackpad",
    ),
    "usb": (
        "usb",
    ),
    "power": (
        "power",
        "power button",
    ),
    "motherboard": (
        "motherboard",
        "mainboard",
    ),
}


SYMPTOM_PATTERNS = {
    "overheating": (
        "overheat",
        "overheating",
        "too hot",
    ),
    "crash": (
        "crash",
        "crashes",
        "crashed",
        "crashing",
    ),
    "freeze": (
        "freeze",
        "freezes",
        "freezing",
        "hung",
        "hangs",
    ),
    "blue_screen": (
        "blue screen",
        "bsod",
    ),
    "black_screen": (
        "black screen",
    ),
    "disconnect": (
        "disconnect",
        "disconnects",
        "disconnecting",
        "drops connection",
        "keeps dropping",
    ),
    "reboot": (
        "reboot",
        "reboots",
        "rebooting",
        "restart",
        "restarts",
        "restarting",
    ),
    "slow": (
        "slow",
        "sluggish",
        "laggy",
        "lagging",
    ),
    "no_power": (
        "no power",
        "won't turn on",
        "doesn't turn on",
        "does not turn on",
    ),
    "battery_drain": (
        "battery drain",
        "battery draining",
        "drains battery",
    ),
    "noise": (
        "noise",
        "noisy",
        "buzzing",
        "clicking",
    ),
    "throttling": (
        "throttling",
        "throttle",
    ),
}


# ============================================================
# ENTITY MODEL
# ============================================================


@dataclass(frozen=True)
class ExtractedEntities:
    """
    Lightweight entities extracted from an STT transcript.

    This is intentionally NOT a diagnostic interpretation.

    It is only enough information to answer:

        "Has the transcript gained something worth retrieving?"
    """

    oem: str | None = None

    model: str | None = None

    fault_codes: tuple[str, ...] = field(
        default_factory=tuple
    )

    components: tuple[str, ...] = field(
        default_factory=tuple
    )

    symptoms: tuple[str, ...] = field(
        default_factory=tuple
    )


# ============================================================
# STABILIZATION RESULT
# ============================================================


@dataclass(frozen=True)
class StabilizationResult:
    """
    Result of processing one STT partial transcript.
    """

    normalized_text: str

    is_stable: bool

    has_meaningful_delta: bool

    entities: ExtractedEntities

    # Useful for debugging/observability.
    new_entities: tuple[str, ...] = ()

    word_count: int = 0


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_transcript(
    text: str,
) -> str:
    """
    Normalize noisy streaming STT without destroying useful
    technical content.

    We deliberately do NOT aggressively spell-correct here.

    Technical identifiers such as:

        0x80070057
        E1001
        Latitude 5420
        WHEA-Logger

    must survive normalization unchanged.
    """

    return " ".join(
        text.strip().split()
    )


# ============================================================
# MODEL EXTRACTION
# ============================================================


def _extract_model(
    text: str,
) -> str | None:
    """
    Conservative laptop/model heuristic.

    Examples it can recognize:

        Latitude 5420
        ThinkPad X1
        XPS 13
        ProBook 450
        XJ-420

    This is intentionally only a retrieval hint.

    The authoritative equipment model is still established by
    the Brain/state layer.
    """

    patterns = (
        # Dell Latitude 5420
        re.compile(
            r"\b("
            r"(?:latitude|thinkpad|ideapad|"
            r"yoga|xps|elitebook|probook|"
            r"vivobook|zenbook|"
            r"precision|inspiron|vostro|"
            r"aspire|swift|predator|"
            r"rog|tuf)"
            r"[\s_-]+"
            r"[a-z0-9][a-z0-9._-]*"
            r")\b",
            re.IGNORECASE,
        ),

        # Generic XJ-420 / ABC 1234 style model.
        re.compile(
            r"\b("
            r"[a-z]{1,6}"
            r"[\s_-]"
            r"\d{2,5}[a-z0-9._-]*"
            r")\b",
            re.IGNORECASE,
        ),
    )

    for pattern in patterns:
        match = pattern.search(text)

        if match:
            return (
                " ".join(
                    match.group(1).split()
                )
                .upper()
            )

    return None


# ============================================================
# ENTITY EXTRACTION
# ============================================================


def extract_entities(
    text: str,
) -> ExtractedEntities:
    """
    Extract retrieval-relevant entities from transcript text.

    This function is deterministic and cheap enough to run on
    streaming partial transcripts.
    """

    normalized = normalize_transcript(
        text
    )

    if not normalized:
        return ExtractedEntities()

    lowered = normalized.lower()

    # --------------------------------------------------------
    # OEM
    # --------------------------------------------------------

    detected_oem: str | None = None

    for oem, pattern in OEM_PATTERNS.items():
        if pattern.search(normalized):
            detected_oem = oem
            break

    # --------------------------------------------------------
    # FAULT CODES
    # --------------------------------------------------------

    found_faults: list[str] = []

    for pattern in FAULT_PATTERNS:

        for match in pattern.finditer(
            normalized
        ):

            code = (
                match.group(0)
                .upper()
                .replace(" ", "-")
            )

            if code not in found_faults:
                found_faults.append(
                    code
                )

    # --------------------------------------------------------
    # COMPONENTS
    # --------------------------------------------------------

    found_components: list[str] = []

    for canonical, aliases in (
        COMPONENT_PATTERNS.items()
    ):

        if any(
            alias in lowered
            for alias in aliases
        ):
            found_components.append(
                canonical
            )

    # --------------------------------------------------------
    # SYMPTOMS
    # --------------------------------------------------------

    found_symptoms: list[str] = []

    for canonical, aliases in (
        SYMPTOM_PATTERNS.items()
    ):

        if any(
            alias in lowered
            for alias in aliases
        ):
            found_symptoms.append(
                canonical
            )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = _extract_model(
        normalized
    )

    return ExtractedEntities(
        oem=detected_oem,
        model=model,
        fault_codes=tuple(
            found_faults
        ),
        components=tuple(
            found_components
        ),
        symptoms=tuple(
            found_symptoms
        ),
    )


# ============================================================
# QUERY STABILIZER
# ============================================================


class QueryStabilizer:
    """
    Prevent excessive retrieval calls from streaming STT.

    The intended lifecycle is:

        partial transcript
              |
              v
        QueryStabilizer
              |
              +---- no meaningful change
              |          -> do nothing
              |
              +---- meaningful change
                         -> retrieval/prefetch allowed


    Important:

    This class does NOT call Qdrant.

    It only decides whether the transcript has changed enough
    to justify retrieval.
    """

    def __init__(
        self,
        *,
        min_words_change: int = 3,
        min_query_words: int = 3,
    ) -> None:

        if min_words_change < 1:
            raise ValueError(
                "min_words_change must be >= 1"
            )

        if min_query_words < 1:
            raise ValueError(
                "min_query_words must be >= 1"
            )

        self.min_words_change = (
            min_words_change
        )

        self.min_query_words = (
            min_query_words
        )

        self._last_stabilized_text = ""

        self._last_entities = (
            ExtractedEntities()
        )

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:
        """
        Reset state at the beginning of a new user turn.
        """

        self._last_stabilized_text = ""

        self._last_entities = (
            ExtractedEntities()
        )

    # ========================================================
    # CURRENT STATE
    # ========================================================

    @property
    def last_text(self) -> str:
        return self._last_stabilized_text

    @property
    def last_entities(
        self,
    ) -> ExtractedEntities:
        return self._last_entities

    # ========================================================
    # PROCESS PARTIAL
    # ========================================================

    def process_partial(
        self,
        partial_transcript: str,
    ) -> StabilizationResult:
        """
        Process one streaming STT transcript.

        A result with:

            is_stable=True

        means the caller may issue speculative retrieval.

        It does NOT mean the user's conversational turn has
        ended. Deepgram/LiveKit remains responsible for turn
        detection.
        """

        normalized = normalize_transcript(
            partial_transcript
        )

        if not normalized:

            return StabilizationResult(
                normalized_text="",
                is_stable=False,
                has_meaningful_delta=False,
                entities=ExtractedEntities(),
                word_count=0,
            )

        current_entities = (
            extract_entities(
                normalized
            )
        )

        # ----------------------------------------------------
        # FIRST MEANINGFUL TRANSCRIPT
        # ----------------------------------------------------

        if not self._last_stabilized_text:

            self._last_stabilized_text = (
                normalized
            )

            self._last_entities = (
                current_entities
            )

            return StabilizationResult(
                normalized_text=normalized,
                is_stable=True,
                has_meaningful_delta=True,
                entities=current_entities,
                new_entities=(
                    self._describe_new_entities(
                        current_entities,
                        ExtractedEntities(),
                    )
                ),
                word_count=len(
                    normalized.split()
                ),
            )

        # ----------------------------------------------------
        # ENTITY DELTAS
        # ----------------------------------------------------

        new_entities = (
            self._describe_new_entities(
                current_entities,
                self._last_entities,
            )
        )

        has_entity_delta = bool(
            new_entities
        )

        # ----------------------------------------------------
        # WORD DELTA
        # ----------------------------------------------------

        previous_words = len(
            self._last_stabilized_text.split()
        )

        current_words = len(
            normalized.split()
        )

        word_count_delta = (
            current_words
            - previous_words
        )

        has_length_delta = (
            word_count_delta
            >= self.min_words_change
        )

        # ----------------------------------------------------
        # TEXT EXTENSION
        # ----------------------------------------------------
        #
        # Streaming STT normally extends the previous
        # transcript. If the transcript changed without gaining
        # enough words or entities, don't retrieve again.
        #

        text_changed = (
            normalized
            != self._last_stabilized_text
        )

        has_meaningful_delta = (
            has_entity_delta
            or has_length_delta
        )

        if (
            has_meaningful_delta
            and text_changed
        ):

            self._last_stabilized_text = (
                normalized
            )

            self._last_entities = (
                current_entities
            )

            return StabilizationResult(
                normalized_text=normalized,
                is_stable=True,
                has_meaningful_delta=True,
                entities=current_entities,
                new_entities=new_entities,
                word_count=current_words,
            )

        # ----------------------------------------------------
        # NOT MEANINGFUL ENOUGH
        # ----------------------------------------------------

        return StabilizationResult(
            normalized_text=normalized,
            is_stable=False,
            has_meaningful_delta=False,
            entities=current_entities,
            new_entities=(),
            word_count=current_words,
        )

    # ========================================================
    # FINALIZE TURN
    # ========================================================

    def finalize(
        self,
        final_transcript: str,
    ) -> StabilizationResult:
        """
        Explicitly accept the final transcript of a user turn.

        This is useful because the final STT transcript should
        always become the canonical stabilizer state even if its
        final change was smaller than min_words_change.
        """

        normalized = normalize_transcript(
            final_transcript
        )

        if not normalized:

            return StabilizationResult(
                normalized_text="",
                is_stable=False,
                has_meaningful_delta=False,
                entities=ExtractedEntities(),
                word_count=0,
            )

        entities = extract_entities(
            normalized
        )

        new_entities = (
            self._describe_new_entities(
                entities,
                self._last_entities,
            )
        )

        changed = (
            normalized
            != self._last_stabilized_text
        )

        self._last_stabilized_text = (
            normalized
        )

        self._last_entities = entities

        return StabilizationResult(
            normalized_text=normalized,
            is_stable=True,
            has_meaningful_delta=changed,
            entities=entities,
            new_entities=new_entities,
            word_count=len(
                normalized.split()
            ),
        )

    # ========================================================
    # ENTITY DELTA
    # ========================================================

    @staticmethod
    def _describe_new_entities(
        current: ExtractedEntities,
        previous: ExtractedEntities,
    ) -> tuple[str, ...]:
        """
        Return a compact description of entities newly appearing
        in the transcript.

        These values are primarily useful for logging and
        retrieval-trigger decisions.
        """

        result: list[str] = []

        if (
            current.oem
            and current.oem
            != previous.oem
        ):
            result.append(
                f"oem:{current.oem}"
            )

        if (
            current.model
            and current.model
            != previous.model
        ):
            result.append(
                f"model:{current.model}"
            )

        for fault in current.fault_codes:
            if fault not in previous.fault_codes:
                result.append(
                    f"fault:{fault}"
                )

        for component in current.components:
            if component not in previous.components:
                result.append(
                    f"component:{component}"
                )

        for symptom in current.symptoms:
            if symptom not in previous.symptoms:
                result.append(
                    f"symptom:{symptom}"
                )

        return tuple(result)