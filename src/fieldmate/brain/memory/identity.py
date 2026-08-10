from __future__ import annotations

import uuid

from .models import MemoryRecord


FIELDMATE_MEMORY_NAMESPACE = uuid.UUID(
    "7f4d8f4e-5e89-4d4e-9f4a-9e8d7b2e1c31"
)


def _normalize(value: str | None) -> str:
    """
    Normalize identity components so insignificant formatting
    differences do not produce different logical memories.
    """

    if value is None:
        return ""

    return " ".join(
        value.strip().lower().split()
    )


def memory_identity(
    memory: MemoryRecord,
) -> str:
    """
    Generate a deterministic logical identity for a memory.

    Two memories with the same semantic identity should map to
    the same identifier regardless of creation time or UUID4.

    This is intentionally NOT based on memory.content because
    wording changes should not automatically create a new logical
    memory.
    """

    parts = (
        memory.memory_type.value,

        _normalize(
            memory.equipment_family
        ),

        _normalize(
            memory.equipment_model
        ),

        _normalize(
            memory.system
        ),

        _normalize(
            memory.subsystem
        ),

        _normalize(
            memory.component
        ),

        "|".join(
            sorted(
                _normalize(code)
                for code in memory.fault_codes
                if _normalize(code)
            )
        ),
    )

    canonical = "::".join(parts)

    return str(
        uuid.uuid5(
            FIELDMATE_MEMORY_NAMESPACE,
            canonical,
        )
    )


def same_memory(
    first: MemoryRecord,
    second: MemoryRecord,
) -> bool:
    """
    Determine whether two records represent the same logical
    memory identity.
    """

    return (
        memory_identity(first)
        == memory_identity(second)
    )