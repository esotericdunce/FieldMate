from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, AsyncIterable, Awaitable, Callable

logger = logging.getLogger(
    "fieldmate.brain.voice_router"
)


class RouteDecision(StrEnum):
    """
    Decision produced by the retrieval side of the
    parallel conversational turn.
    """

    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ParallelTurnResult:
    """
    Metadata describing how a parallel turn was routed.

    This is intentionally separate from the actual streamed
    response. The caller owns response/TTS handling.
    """

    decision: RouteDecision

    retrieval_latency_ms: float

    groq_speculative: bool

    grounded: bool


class ParallelTurnRouter:
    """
    Coordinates speculative Groq generation with Qdrant
    retrieval.

    Architecture:

        finalized / meaningful transcript
                    |
              +-----+-----+
              |           |
              v           v
            Groq       Qdrant
         speculative   retrieval
              |           |
              |      +----+----+
              |      |         |
              |   relevant   none
              |      |         |
              |      v         v
              |  grounded   release
              |    Groq     speculative
              |      |         |
              +------+---------+
                     |
                     v
                    Rime

    IMPORTANT:

    Groq is allowed to start before Qdrant finishes.

    However, speculative Groq output is NOT yielded to the
    caller until Qdrant establishes that grounding is not
    required.

    Therefore:

        Qdrant irrelevant
            -> release speculative Groq

        Qdrant timeout
            -> release speculative Groq

        Qdrant failure
            -> release speculative Groq

        Qdrant relevant
            -> cancel speculative Groq
            -> start grounded Groq
    """

    def __init__(
        self,
        *,
        qdrant_timeout_ms: int = 900,
        speculative_buffer_chunks: int = 64,
    ) -> None:

        if qdrant_timeout_ms <= 0:
            raise ValueError(
                "qdrant_timeout_ms must be > 0"
            )

        if speculative_buffer_chunks <= 0:
            raise ValueError(
                "speculative_buffer_chunks must be > 0"
            )

        self.qdrant_timeout_seconds = (
            qdrant_timeout_ms / 1000.0
        )

        self.speculative_buffer_chunks = (
            speculative_buffer_chunks
        )

    async def stream(
        self,
        *,
        retrieve: Callable[
            [],
            Awaitable[Any],
        ],
        stream_groq: Callable[
            [str],
            AsyncIterable[str],
        ],
        user_text: str,
        grounded_prompt: Callable[
            [str, Any],
            str,
        ],
        generation_is_current: Callable[
            [],
            bool,
        ],
    ) -> AsyncIterable[str]:
        """
        Execute Groq and Qdrant concurrently.

        `stream_groq()` must return an async iterable of text
        chunks.

        The first invocation is speculative.

        Its chunks are buffered until the Qdrant decision.

        If retrieval is irrelevant, times out, or fails, the
        buffered speculative response is released.

        If retrieval is relevant, speculative generation is
        cancelled and grounded Groq generation starts.
        """

        if not generation_is_current():
            return

        retrieval_task = asyncio.create_task(
            retrieve()
        )

        speculative_queue: asyncio.Queue[
            str | None
        ] = asyncio.Queue(
            maxsize=self.speculative_buffer_chunks
        )

        speculative_finished = (
            asyncio.Event()
        )

        async def run_speculative() -> None:
            """
            Consume the speculative Groq stream into a bounded
            queue.

            A bounded queue is deliberate.

            If Qdrant is slow, we don't allow unbounded model
            output to accumulate in memory.
            """

            try:

                async for chunk in stream_groq(
                    user_text
                ):

                    if not generation_is_current():
                        return

                    await speculative_queue.put(
                        chunk
                    )

            except asyncio.CancelledError:
                raise

            except Exception:
                logger.exception(
                    "Speculative Groq stream failed"
                )

            finally:

                speculative_finished.set()

                # Sentinel insertion can itself block if the
                # queue is full. Drain one item in that unusual
                # situation so the consumer can terminate.
                while True:

                    try:

                        speculative_queue.put_nowait(
                            None
                        )

                        break

                    except asyncio.QueueFull:

                        try:
                            speculative_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break

        speculative_task = asyncio.create_task(
            run_speculative()
        )

        started = time.perf_counter()

        retrieval_result: Any = None

        decision = (
            RouteDecision.TIMEOUT
        )

        try:

            # =================================================
            # WAIT FOR QDRANT DECISION
            # =================================================

            try:

                retrieval_result = (
                    await asyncio.wait_for(
                        asyncio.shield(
                            retrieval_task
                        ),
                        timeout=(
                            self.qdrant_timeout_seconds
                        ),
                    )
                )

                if self._has_relevant_evidence(
                    retrieval_result
                ):
                    decision = (
                        RouteDecision.RELEVANT
                    )
                else:
                    decision = (
                        RouteDecision.IRRELEVANT
                    )

            except asyncio.TimeoutError:

                decision = (
                    RouteDecision.TIMEOUT
                )

            except asyncio.CancelledError:
                raise

            except Exception:

                logger.exception(
                    "Qdrant retrieval failed during "
                    "parallel routing"
                )

                decision = (
                    RouteDecision.FAILED
                )

            retrieval_latency_ms = (
                time.perf_counter()
                - started
            ) * 1000.0

            logger.info(
                ">>> PARALLEL TURN ROUTE "
                "decision=%s qdrant=%.1fms",
                decision.value,
                retrieval_latency_ms,
            )

            if not generation_is_current():
                return

            # =================================================
            # SEMANTIC CACHE FAST PATH HIT (<10ms)
            # =================================================

            if (
                getattr(retrieval_result, "cached", False)
                and getattr(retrieval_result, "cached_response", None)
                and isinstance(retrieval_result.cached_response, str)
            ):
                cached_text = str(retrieval_result.cached_response)
                logger.info(
                    ">>> SEMANTIC CACHE FAST PATH HIT: yielding cached response in %.1fms",
                    retrieval_latency_ms,
                )
                speculative_task.cancel()
                try:
                    await speculative_task
                except asyncio.CancelledError:
                    pass

                if not generation_is_current():
                    return

                yield cached_text
                return

            # =================================================
            # QDRANT PASS
            # =================================================

            if decision in {
                RouteDecision.IRRELEVANT,
                RouteDecision.TIMEOUT,
                RouteDecision.FAILED,
            }:

                logger.info(
                    ">>> QDRANT PASS "
                    "releasing speculative Groq"
                )

                async for chunk in (
                    self._release_speculative(
                        speculative_queue=(
                            speculative_queue
                        ),
                        generation_is_current=(
                            generation_is_current
                        ),
                    )
                ):

                    yield chunk

                return

            # =================================================
            # QDRANT RELEVANT
            # =================================================

            logger.info(
                ">>> QDRANT RELEVANT "
                "discarding speculative Groq"
            )

            speculative_task.cancel()

            try:
                await speculative_task
            except asyncio.CancelledError:
                pass

            if not generation_is_current():
                return

            grounded_text = (
                grounded_prompt(
                    user_text,
                    retrieval_result,
                )
            )

            logger.info(
                ">>> STARTING GROUNDED GROQ"
            )

            async for chunk in stream_groq(
                grounded_text
            ):

                if not generation_is_current():
                    return

                yield chunk

        finally:

            # -------------------------------------------------
            # Retrieval lifecycle
            # -------------------------------------------------

            if (
                not retrieval_task.done()
                and decision
                in {
                    RouteDecision.TIMEOUT,
                    RouteDecision.FAILED,
                }
            ):
                # Retrieval may finish in the background.
                #
                # We deliberately do NOT await it here.
                retrieval_task.add_done_callback(
                    _consume_background_result
                )

            elif not retrieval_task.done():

                retrieval_task.cancel()

            # -------------------------------------------------
            # Speculative Groq lifecycle
            # -------------------------------------------------

            if not speculative_task.done():

                speculative_task.cancel()

                try:
                    await speculative_task
                except asyncio.CancelledError:
                    pass

    @staticmethod
    async def _release_speculative(
        *,
        speculative_queue: asyncio.Queue[
            str | None
        ],
        generation_is_current: Callable[
            [],
            bool,
        ],
    ) -> AsyncIterable[str]:
        """
        Release buffered speculative Groq output.

        Once Qdrant says the request does not require grounding,
        the speculative response becomes the response stream.
        """

        while True:

            chunk = await (
                speculative_queue.get()
            )

            if chunk is None:
                return

            if not generation_is_current():
                return

            yield chunk

    @staticmethod
    def _has_relevant_evidence(
        retrieval_result: Any,
    ) -> bool:
        """
        RetrievalOrchestrator owns relevance.

        The voice router only consumes the canonical `relevant`
        flag and keeps a conservative compatibility fallback for
        older RetrievalResult objects.
        """

        if retrieval_result is None:
            return False

        relevant = getattr(
            retrieval_result,
            "relevant",
            None,
        )

        if relevant is not None:
            return bool(relevant)

        context = getattr(
            retrieval_result,
            "context",
            None,
        )

        if context is None:
            return False

        evidence = getattr(
            context,
            "evidence",
            (),
        )

        if evidence:
            return True

        memories = getattr(
            context,
            "memories",
            (),
        )

        return bool(memories)


def _consume_background_result(
    task: asyncio.Task[Any],
) -> None:
    """
    Consume a detached retrieval task so an eventual exception
    doesn't become an unhandled-task warning.
    """

    if task.cancelled():
        return

    try:
        task.result()

    except Exception:
        logger.debug(
            "Background Qdrant retrieval failed",
            exc_info=True,
        )