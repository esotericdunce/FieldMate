from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


# ============================================================
# RETRIEVAL MODE
# ============================================================


class RetrievalMode(str, Enum):
    """
    Retrieval strategies supported by FieldMate.

    DENSE:
        Semantic similarity. Best default for natural-language
        troubleshooting descriptions.

    SPARSE:
        BM25/token-oriented retrieval. Better for exact
        identifiers such as Windows error codes, Event IDs,
        BSOD codes, model identifiers, and technical terms.

    HYBRID:
        Dense + sparse with Qdrant RRF fusion. Most expensive,
        therefore reserved for queries where both semantic
        meaning and exact identifiers matter.
    """

    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


# ============================================================
# RETRIEVAL PLAN
# ============================================================


@dataclass(frozen=True)
class RetrievalPlan:
    """
    Immutable decision produced by the retrieval planner.

    The planner does not execute retrieval.

    It only answers:

        "What retrieval strategy should we use?"
    """

    mode: RetrievalMode
    reason: str
    confidence: float


# ============================================================
# PATTERNS
# ============================================================

# Common Windows / PC diagnostic identifiers.
#
# These intentionally remain conservative. We don't want
# ordinary numbers such as "5420" in "Latitude 5420" to
# automatically trigger sparse retrieval.

FAULT_CODE_PATTERNS = (
    # E123
    # ERR123
    # ERROR 123
    # EVENT 1001
    # FAULT-42
    re.compile(
        r"\b(?:"
        r"e|err|error|event|fault"
        r")[\s_-]?\d{1,5}\b",
        re.IGNORECASE,
    ),

    # 0x80070057
    # 0x0000007E
    re.compile(
        r"\b0x[0-9a-f]{2,16}\b",
        re.IGNORECASE,
    ),

    # WHEA-Logger / WHEA Logger identifiers
    re.compile(
        r"\bwhea(?:[\s_-]+logger)?"
        r"(?:[\s_-]+\d+)?\b",
        re.IGNORECASE,
    ),

    # Classic Windows stop codes.
    re.compile(
        r"\bstop[\s_-]+0x[0-9a-f]+\b",
        re.IGNORECASE,
    ),

    # BSOD names, e.g. MEMORY_MANAGEMENT.
    re.compile(
        r"\b[a-z0-9_]+_"
        r"(?:IRQL|MANAGEMENT|FAULT|ERROR|EXCEPTION)"
        r"\b",
        re.IGNORECASE,
    ),
)


# Terms where exact lexical matching can be more valuable than
# semantic similarity.
#
# These are NOT automatically sufficient to force sparse
# retrieval; they are used as additional routing signals.
EXACT_TECHNICAL_TERMS = (
    "event viewer",
    "event id",
    "device manager",
    "stop code",
    "bsod",
    "blue screen",
    "whea",
    "ntfs",
    "dhcp",
    "dns",
    "ipconfig",
    "powershell",
    "cmd",
    "0x",
)


# ============================================================
# HELPERS
# ============================================================


def _has_fault_identifier(text: str) -> bool:
    """
    Return True when the query contains a recognizable exact
    diagnostic identifier.
    """

    return any(
        pattern.search(text)
        for pattern in FAULT_CODE_PATTERNS
    )


def _has_exact_technical_term(text: str) -> bool:
    """
    Detect terminology where lexical retrieval can be useful.

    This is deliberately weaker than a fault-code match.
    """

    lowered = text.lower()

    return any(
        term in lowered
        for term in EXACT_TECHNICAL_TERMS
    )


def _has_multiple_diagnostic_signals(
    text: str,
) -> bool:
    """
    Detect queries containing several distinct diagnostic
    signals.

    Example:

        "Dell Latitude 5420 error 0x80070057 after sleep"

    Such a query benefits more from hybrid retrieval because:

        equipment/model semantics
                +
        exact identifier
                +
        symptom context

    all matter simultaneously.
    """

    lowered = text.lower()

    signals = 0

    if _has_fault_identifier(text):
        signals += 1

    if any(
        term in lowered
        for term in (
            "after sleep",
            "after reboot",
            "after update",
            "after shutdown",
            "randomly",
            "intermittent",
            "keeps",
            "stops",
            "disconnects",
            "crashes",
            "freezes",
            "overheats",
        )
    ):
        signals += 1

    if any(
        term in lowered
        for term in (
            "dell",
            "lenovo",
            "hp",
            "asus",
            "latitude",
            "thinkpad",
            "ideapad",
            "xps",
            "elitebook",
            "probook",
            "vivobook",
            "zenbook",
            "rog",
        )
    ):
        signals += 1

    return signals >= 2


# ============================================================
# PUBLIC PLANNER
# ============================================================


def plan_retrieval(
    text: str,
    *,
    has_equipment_context: bool = False,
    has_fault_context: bool = False,
) -> RetrievalPlan:
    """
    Choose the cheapest retrieval strategy that is likely to
    preserve diagnostic recall.

    Decision hierarchy:

        1. Empty query
            -> dense

        2. Exact fault identifier + equipment
            -> hybrid

        3. Strong multi-signal diagnostic query
            -> hybrid

        4. Exact fault identifier
            -> sparse

        5. Exact technical terminology
            -> sparse

        6. Equipment + natural-language symptom
            -> dense

        7. Ordinary troubleshooting question
            -> dense


    Important:

    The planner is intentionally conservative about HYBRID.

    Hybrid retrieval costs more latency because it executes both
    retrieval branches and performs fusion. It should therefore
    only be used when the additional recall is justified.
    """

    normalized = " ".join(
        text.strip().split()
    )

    # --------------------------------------------------------
    # EMPTY QUERY
    # --------------------------------------------------------

    if not normalized:

        return RetrievalPlan(
            mode=RetrievalMode.DENSE,
            reason="empty_query",
            confidence=0.0,
        )

    fault_identifier = (
        _has_fault_identifier(
            normalized
        )
    )

    exact_term = (
        _has_exact_technical_term(
            normalized
        )
    )

    equipment_present = (
        has_equipment_context
    )

    fault_present = (
        fault_identifier
        or has_fault_context
    )

    multi_signal = (
        _has_multiple_diagnostic_signals(
            normalized
        )
    )

    # --------------------------------------------------------
    # HYBRID: EXACT FAULT + EQUIPMENT
    # --------------------------------------------------------

    if (
        fault_present
        and equipment_present
    ):
        return RetrievalPlan(
            mode=RetrievalMode.HYBRID,
            reason=(
                "equipment_and_fault_context"
            ),
            confidence=0.97,
        )

    # --------------------------------------------------------
    # HYBRID: MULTI-SIGNAL QUERY
    # --------------------------------------------------------
    #
    # This catches cases where the transcript itself contains
    # enough context even if the Brain hasn't extracted the
    # structured equipment/fault state yet.
    #
    # Example:
    #
    # "Dell Latitude 5420 gets error 0x80070057 after Windows
    #  update"
    #
    # Dense understands the overall failure scenario.
    # Sparse preserves the exact identifier.
    #

    if multi_signal and (
        fault_present
        or equipment_present
    ):
        return RetrievalPlan(
            mode=RetrievalMode.HYBRID,
            reason=(
                "multi_signal_diagnostic_query"
            ),
            confidence=0.93,
        )

    # --------------------------------------------------------
    # SPARSE: EXACT FAULT IDENTIFIER
    # --------------------------------------------------------

    if fault_present:

        return RetrievalPlan(
            mode=RetrievalMode.SPARSE,
            reason="fault_identifier",
            confidence=0.95,
        )

    # --------------------------------------------------------
    # SPARSE: EXACT TECHNICAL TERMINOLOGY
    # --------------------------------------------------------
    #
    # We don't want every query mentioning "driver" or "Wi-Fi"
    # to become sparse. Those are ordinary semantic concepts.
    #
    # These terms are selected because exact occurrences can
    # materially affect retrieval.
    #

    if exact_term:

        return RetrievalPlan(
            mode=RetrievalMode.SPARSE,
            reason="exact_technical_term",
            confidence=0.88,
        )

    # --------------------------------------------------------
    # DENSE: EQUIPMENT + NATURAL LANGUAGE
    # --------------------------------------------------------

    if equipment_present:

        return RetrievalPlan(
            mode=RetrievalMode.DENSE,
            reason=(
                "equipment_context_semantic_query"
            ),
            confidence=0.90,
        )

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return RetrievalPlan(
        mode=RetrievalMode.DENSE,
        reason="semantic_query",
        confidence=0.85,
    )