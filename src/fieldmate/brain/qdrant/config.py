from __future__ import annotations

import os
from dataclasses import dataclass


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, str(default))

    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be an integer, got {value!r}"
        ) from exc

    if parsed <= 0:
        raise RuntimeError(
            f"{name} must be greater than zero."
        )

    return parsed


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, str(default))

    try:
        parsed = float(value)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} must be a number, got {value!r}"
        ) from exc

    if parsed <= 0:
        raise RuntimeError(
            f"{name} must be greater than zero."
        )

    return parsed


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    """
    Immutable configuration for FieldMate's Qdrant memory store.

    Qdrant Cloud is responsible for embedding generation through
    Cloud Inference.

    FieldMate therefore does not need to download or manage the
    embedding models locally.
    """

    url: str
    api_key: str

    collection_name: str = "fieldmate_memory"

    dense_model: str = (
        "sentence-transformers/"
        "all-minilm-l6-v2"
    )

    dense_vector_size: int = 384

    sparse_model: str = "qdrant/bm25"

    dense_vector_name: str = "dense"

    sparse_vector_name: str = "bm25"

    timeout_seconds: float = 30.0

    # ---------------------------------------------------------
    # ENVIRONMENT
    # ---------------------------------------------------------

    @classmethod
    def from_env(
        cls,
    ) -> "QdrantConfig":
        url = os.getenv("QDRANT_URL")

        api_key = os.getenv(
            "QDRANT_API_KEY"
        )

        if not url:
            raise RuntimeError(
                "QDRANT_URL is not set."
            )

        if not api_key:
            raise RuntimeError(
                "QDRANT_API_KEY is not set."
            )

        collection_name = os.getenv(
            "QDRANT_COLLECTION",
            "fieldmate_memory",
        ).strip()

        if not collection_name:
            raise RuntimeError(
                "QDRANT_COLLECTION cannot be empty."
            )

        dense_model = os.getenv(
            "QDRANT_DENSE_MODEL",
            (
                "sentence-transformers/"
                "all-minilm-l6-v2"
            ),
        ).strip()

        sparse_model = os.getenv(
            "QDRANT_SPARSE_MODEL",
            "qdrant/bm25",
        ).strip()

        dense_vector_name = os.getenv(
            "QDRANT_DENSE_VECTOR_NAME",
            "dense",
        ).strip()

        sparse_vector_name = os.getenv(
            "QDRANT_SPARSE_VECTOR_NAME",
            "bm25",
        ).strip()

        if not dense_model:
            raise RuntimeError(
                "QDRANT_DENSE_MODEL cannot be empty."
            )

        if not sparse_model:
            raise RuntimeError(
                "QDRANT_SPARSE_MODEL cannot be empty."
            )

        if not dense_vector_name:
            raise RuntimeError(
                "QDRANT_DENSE_VECTOR_NAME cannot be empty."
            )

        if not sparse_vector_name:
            raise RuntimeError(
                "QDRANT_SPARSE_VECTOR_NAME cannot be empty."
            )

        return cls(
            url=url.rstrip("/"),
            api_key=api_key,
            collection_name=collection_name,
            dense_model=dense_model,
            sparse_model=sparse_model,
            dense_vector_name=dense_vector_name,
            sparse_vector_name=sparse_vector_name,
            timeout_seconds=_env_float(
                "QDRANT_TIMEOUT_SECONDS",
                30.0,
            ),
        )