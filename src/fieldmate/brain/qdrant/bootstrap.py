from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from .config import QdrantConfig
from .repository import QdrantMemoryRepository


logger = logging.getLogger(
    "fieldmate.qdrant"
)


async def main() -> None:
    """
    Initialize and validate the FieldMate Qdrant memory store.

    Run this manually when setting up a deployment or when
    validating Qdrant configuration.

    The conversational application should also call
    repository.ensure_collection() during startup.
    """

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
    )

    config = QdrantConfig.from_env()

    logger.info(
        ">>> CONNECTING TO QDRANT"
    )

    repository = QdrantMemoryRepository(
        config
    )

    try:

        await repository.ensure_collection()

        logger.info(
            ">>> QDRANT MEMORY COLLECTION READY"
        )

        logger.info(
            "Collection: %s",
            config.collection_name,
        )

        logger.info(
            "Dense model: %s",
            config.dense_model,
        )

        logger.info(
            "Sparse model: %s",
            config.sparse_model,
        )

        logger.info(
            "Dense vector: %s (%d dimensions)",
            config.dense_vector_name,
            config.dense_vector_size,
        )

        logger.info(
            "Sparse vector: %s",
            config.sparse_vector_name,
        )

    except Exception:
        logger.exception(
            ">>> QDRANT INITIALIZATION FAILED"
        )
        raise

    finally:
        await repository.close()


if __name__ == "__main__":
    asyncio.run(main())