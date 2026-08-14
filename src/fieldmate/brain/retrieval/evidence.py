from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from fieldmate.brain.state.models import DiagnosticState


# ============================================================
# EVIDENCE
# ============================================================

@dataclass(frozen=True)
class Evidence:
    """
    Normalized diagnostic evidence returned by retrieval.

    Evidence is the bridge between the persistence layer
    (Qdrant) and the reasoning/context layer.

    This object deliberately contains no Qdrant-specific types.
    """

    evidence_id: str
    memory_id: str
    memory_type: str

    content: str

    source: str

    equipment_model: str | None

    fault_codes: tuple[str, ...]

    relevance_score: float
    confidence: float

    verification_status: str

    provenance: str

    # One of:
    #
    #   supporting
    #   contradicting
    #   neutral
    #
    relation: str

    case_reference: str | None = None

    retrieval_mode: str = "dense"


# ============================================================
# CONSTANTS
# ============================================================

SUPPORTING = "supporting"
CONTRADICTING = "contradicting"
NEUTRAL = "neutral"

VALID_RELATIONS = {
    SUPPORTING,
    CONTRADICTING,
    NEUTRAL,
}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Convert arbitrary numeric input into a bounded float.

    Retrieval providers should normally give us clean floats,
    but defensive normalization prevents malformed payloads
    from poisoning the reasoning layer.
    """

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if number != number:  # NaN
        return default

    if number == float("inf"):
        return 1.0

    if number == float("-inf"):
        return 0.0

    return max(
        0.0,
        min(1.0, number),
    )


def _safe_string(
    value: Any,
    default: str = "",
) -> str:
    """
    Convert a value to a normalized string.
    """

    if value is None:
        return default

    value = str(value).strip()

    return value if value else default


def _normalize_fault_codes(
    value: Any,
) -> tuple[str, ...]:
    """
    Normalize fault-code payloads.

    Supports:

        "0x80070005"

    and:

        ["0x80070005", "WHEA-18"]

    while preserving order and removing duplicates.
    """

    if value is None:
        return ()

    if isinstance(value, str):
        raw_values = [value]

    elif isinstance(value, (list, tuple, set)):
        raw_values = value

    else:
        raw_values = [value]

    result: list[str] = []

    for raw in raw_values:
        code = _safe_string(raw)

        if not code:
            continue

        normalized = code.upper()

        if normalized not in result:
            result.append(normalized)

    return tuple(result)


# ============================================================
# EXPLICIT RELATION
# ============================================================

def _explicit_relation(
    payload: dict[str, Any],
) -> str | None:
    """
    Read an explicitly stored evidence relation.

    Explicit contradiction metadata is trusted more than
    heuristic inference because it should originate from a
    deliberate domain operation.

    Accepted payload fields:

        relation
        evidence_relation

    Accepted values:

        supporting
        contradicting
        neutral

    Invalid values are ignored.
    """

    raw = payload.get(
        "relation",
        payload.get("evidence_relation"),
    )

    if raw is None:
        return None

    relation = _safe_string(raw).lower()

    if relation in VALID_RELATIONS:
        return relation

    return None


# ============================================================
# CONTRADICTION METADATA
# ============================================================

def _has_explicit_contradiction(
    payload: dict[str, Any],
) -> bool:
    """
    Detect explicit contradiction markers.

    These are intentionally narrow.

    We do NOT interpret ordinary words such as "not",
    "replace", "failed", etc. as contradictions because doing
    so produces large numbers of false positives.
    """

    if payload.get("has_contradiction") is True:
        return True

    if payload.get("contradicted") is True:
        return True

    if payload.get("is_contradiction") is True:
        return True

    status = _safe_string(
        payload.get("status")
    ).lower()

    verification_status = _safe_string(
        payload.get("verification_status")
    ).lower()

    if status == "contradicted":
        return True

    if verification_status == "contradicted":
        return True

    return False


# ============================================================
# STATE COMPARISON
# ============================================================

def _equipment_matches(
    payload: dict[str, Any],
    state: DiagnosticState,
) -> bool:
    """
    Determine whether retrieved evidence explicitly refers
    to the same equipment model as the current diagnostic case.

    This is intentionally exact after normalization.

    We do not attempt fuzzy model matching here.
    """

    state_model = getattr(
        state.equipment,
        "model",
        None,
    )

    evidence_model = payload.get(
        "equipment_model"
    )

    if not state_model or not evidence_model:
        return False

    return (
        str(state_model).strip().lower()
        == str(evidence_model).strip().lower()
    )


def _fault_codes_overlap(
    payload: dict[str, Any],
    state: DiagnosticState,
) -> bool:
    """
    Determine whether retrieved evidence and current state
    share at least one fault identifier.
    """

    state_codes = {
        str(code).strip().upper()
        for code in getattr(
            state,
            "fault_codes",
            [],
        )
        if str(code).strip()
    }

    if not state_codes:
        return False

    evidence_codes = set(
        _normalize_fault_codes(
            payload.get(
                "fault_codes",
                payload.get("fault_code"),
            )
        )
    )

    return bool(
        state_codes.intersection(
            evidence_codes
        )
    )


# ============================================================
# RELATION CLASSIFICATION
# ============================================================

def _determine_relation(
    content: str,
    payload: dict[str, Any],
    state: DiagnosticState | None = None,
) -> str:
    """
    Classify evidence relative to the current diagnostic state.

    IMPORTANT:

    This function is deliberately conservative.

    Supporting evidence may be inferred from meaningful
    overlap with the current diagnostic hypothesis.

    Contradicting evidence is NOT inferred from arbitrary
    natural-language words. It requires explicit contradiction
    metadata or a strong domain-level conflict.

    If the system cannot establish a relationship safely,
    the evidence is neutral.

    This is preferable to feeding false contradictions into
    a troubleshooting reasoning model.
    """

    # --------------------------------------------------------
    # 1. Explicit relation wins.
    # --------------------------------------------------------

    explicit = _explicit_relation(payload)

    if explicit is not None:
        return explicit

    # --------------------------------------------------------
    # 2. Explicit contradiction metadata.
    # --------------------------------------------------------

    if _has_explicit_contradiction(payload):
        return CONTRADICTING

    # --------------------------------------------------------
    # 3. No diagnostic state means no state-relative relation.
    # --------------------------------------------------------

    if state is None:
        return NEUTRAL

    lower_content = content.lower()

    # --------------------------------------------------------
    # 4. Current hypothesis support.
    #
    # Only classify as supporting when there is meaningful
    # lexical overlap with the current hypothesis.
    #
    # Tiny/common words are ignored.
    # --------------------------------------------------------

    current_hypothesis = getattr(
        state,
        "current_hypothesis",
        None,
    )

    if current_hypothesis:
        hypothesis_terms = {
            term
            for term in (
                str(current_hypothesis)
                .lower()
                .replace("/", " ")
                .replace("-", " ")
                .split()
            )
            if len(term) >= 4
        }

        meaningful_matches = sum(
            1
            for term in hypothesis_terms
            if term in lower_content
        )

        if meaningful_matches >= 2:
            return SUPPORTING

    # --------------------------------------------------------
    # 5. Same equipment + same fault code is useful contextual
    # evidence, but not automatically proof of support.
    #
    # Keep it neutral. ContextIntelligence will score it.
    # --------------------------------------------------------

    _equipment_matches(
        payload,
        state,
    )

    _fault_codes_overlap(
        payload,
        state,
    )

    # These calls deliberately do not change the relation.
    #
    # The evidence is relevant, but relevance != support.
    #
    # ContextIntelligence handles the relevance weighting.

    # --------------------------------------------------------
    # 6. Failed tests.
    #
    # A failed test appearing in a memory does NOT automatically
    # contradict the current case. It may actually be useful
    # supporting evidence.
    #
    # Therefore we leave it neutral here.
    # --------------------------------------------------------

    return NEUTRAL


# ============================================================
# POINT EXTRACTION
# ============================================================

def _extract_point(
    point: Any,
) -> tuple[str, float, dict[str, Any]]:
    """
    Extract ID, score and payload from either:

        Qdrant ScoredPoint

    or:

        dict-like test/fake representation.

    This keeps the retrieval layer easy to test without requiring
    a live Qdrant connection.
    """

    # --------------------------------------------------------
    # Dictionary representation
    # --------------------------------------------------------

    if isinstance(point, dict):

        point_id = _safe_string(
            point.get(
                "id",
                "unknown",
            ),
            default="unknown",
        )

        score = _safe_float(
            point.get(
                "score",
                0.0,
            )
        )

        raw_payload = point.get(
            "payload",
            point,
        )

        if isinstance(
            raw_payload,
            dict,
        ):
            payload = raw_payload
        else:
            payload = {}

        return (
            point_id,
            score,
            payload,
        )

    # --------------------------------------------------------
    # Qdrant ScoredPoint / compatible object
    # --------------------------------------------------------

    point_id = _safe_string(
        getattr(
            point,
            "id",
            "unknown",
        ),
        default="unknown",
    )

    score = _safe_float(
        getattr(
            point,
            "score",
            0.0,
        )
    )

    raw_payload = getattr(
        point,
        "payload",
        None,
    )

    payload = (
        raw_payload
        if isinstance(
            raw_payload,
            dict,
        )
        else {}
    )

    return (
        point_id,
        score,
        payload,
    )


# ============================================================
# NORMALIZE EVIDENCE
# ============================================================

def normalize_evidence(
    points: Sequence[Any],
    *,
    state: DiagnosticState | None = None,
    retrieval_mode: str = "dense",
) -> tuple[Evidence, ...]:
    """
    Convert raw retrieval results into domain Evidence objects.

    Supported input:

        Qdrant ScoredPoint
        dict
        compatible test doubles

    Guarantees:

        - malformed points are skipped safely
        - empty content is ignored
        - scores are bounded
        - confidence is bounded
        - fault codes are normalized
        - provenance is preserved
        - relation classification is conservative
        - evidence IDs are deterministic for a point
    """

    evidence_list: list[Evidence] = []

    for point in points:

        point_id, score, payload = _extract_point(
            point
        )

        if retrieval_mode == "hybrid" or (0.0 < score < 0.1):
            score = min(1.0, score * 30.0)

        # ----------------------------------------------------
        # CONTENT
        # ----------------------------------------------------

        content = _safe_string(
            payload.get(
                "content",
                payload.get(
                    "text",
                    payload.get(
                        "description",
                        "",
                    ),
                ),
            )
        )

        if not content:
            continue

        # ----------------------------------------------------
        # MEMORY ID
        # ----------------------------------------------------

        memory_id = _safe_string(
            payload.get(
                "memory_id",
                point_id,
            ),
            default=point_id,
        )

        # ----------------------------------------------------
        # MEMORY TYPE
        # ----------------------------------------------------

        memory_type = _safe_string(
            payload.get(
                "memory_type",
                "unknown",
            ),
            default="unknown",
        )

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        source = _safe_string(
            payload.get(
                "source",
                payload.get(
                    "owner_id",
                    "qdrant",
                ),
            ),
            default="qdrant",
        )

        # ----------------------------------------------------
        # EQUIPMENT
        # ----------------------------------------------------

        equipment_model_raw = payload.get(
            "equipment_model"
        )

        equipment_model = (
            _safe_string(
                equipment_model_raw
            )
            or None
        )

        # ----------------------------------------------------
        # FAULT CODES
        # ----------------------------------------------------

        fault_codes = _normalize_fault_codes(
            payload.get(
                "fault_codes",
                payload.get(
                    "fault_code"
                ),
            )
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidence = _safe_float(
            payload.get(
                "confidence",
                0.8,
            ),
            default=0.8,
        )

        # ----------------------------------------------------
        # VERIFICATION STATUS
        # ----------------------------------------------------

        verification_status = _safe_string(
            payload.get(
                "verification_status",
                payload.get(
                    "status",
                    "unverified",
                ),
            ),
            default="unverified",
        )

        # ----------------------------------------------------
        # PROVENANCE
        # ----------------------------------------------------

        provenance = _safe_string(
            payload.get(
                "provenance",
                f"qdrant_{retrieval_mode}",
            ),
            default=f"qdrant_{retrieval_mode}",
        )

        # ----------------------------------------------------
        # CASE REFERENCE
        # ----------------------------------------------------

        case_reference_raw = payload.get(
            "case_reference",
            payload.get(
                "case_id"
            ),
        )

        case_reference = (
            _safe_string(
                case_reference_raw
            )
            or None
        )

        # ----------------------------------------------------
        # RELATION
        # ----------------------------------------------------

        relation = _determine_relation(
            content,
            payload,
            state=state,
        )

        # ----------------------------------------------------
        # EVIDENCE ID
        # ----------------------------------------------------

        evidence_id = (
            f"ev_{point_id}"
        )

        evidence_list.append(
            Evidence(
                evidence_id=evidence_id,
                memory_id=memory_id,
                memory_type=memory_type,
                content=content,
                source=source,
                equipment_model=equipment_model,
                fault_codes=fault_codes,
                relevance_score=score,
                confidence=confidence,
                verification_status=verification_status,
                provenance=provenance,
                relation=relation,
                case_reference=case_reference,
                retrieval_mode=retrieval_mode,
            )
        )

    return tuple(evidence_list)