"""voice_checkin.py — Proactive check-in scheduler for FieldMate.

Sends a periodic voice check-in ("Still there? Need anything?") when
the technician has been silent for a configurable interval.  Designed
to be robust against the fast event bursts that arrive from LiveKit:
only one asyncio timer is ever live at a time, and all state
transitions are synchronous so they can be called from sync event
handlers without extra locking.

States
------
VOICE_OFF         Technician has disabled proactive check-ins.
VOICE_ACTIVE      Monitoring silence; timer counting down to next check-in.
CHECKIN_SPEAKING  Agent is speaking the check-in prompt.
CHECKIN_LISTENING Waiting for the technician to reply.
CHECKIN_RETRY     No reply received; brief back-off before a second prompt.
PAUSED            Agent is busy (speaking a real response); check-ins suppressed.

All state transitions that reset the timer call _arm() which cancels
any existing handle before creating a new one — so duplicate timers
are structurally impossible.
"""

from __future__ import annotations

import asyncio
import logging
from enum import Enum, auto
from typing import Awaitable, Callable

logger = logging.getLogger("fieldmate.checkin")


# ============================================================
# STATE MACHINE
# ============================================================


class CheckinState(Enum):
    VOICE_OFF = auto()
    VOICE_ACTIVE = auto()
    CHECKIN_SPEAKING = auto()
    CHECKIN_LISTENING = auto()
    CHECKIN_RETRY = auto()
    PAUSED = auto()


class CheckinScheduler:
    """Single-timer, event-driven check-in state machine.

    Parameters
    ----------
    say_callback:
        Async callable that takes a ``str`` and speaks it via the
        existing ``session.say()`` path.  It must be non-blocking from
        the caller's perspective (i.e., it should *not* ``await
        wait_for_playout``; that is handled internally).
    interval_s:
        Seconds of silence before the first check-in prompt.
    retry_delay_s:
        Seconds to wait for a reply before issuing a retry prompt.
    max_retries:
        Number of retry prompts before giving up and going VOICE_OFF.
    min_interval_s:
        Shortest allowed interval (clamps ``interval_s`` from below so
        the agent can never spam the technician).
    prompt:
        Text spoken on the first check-in.
    retry_prompt:
        Text spoken on subsequent retries.
    """

    def __init__(
        self,
        say_callback: Callable[[str], Awaitable[None]],
        *,
        interval_s: float = 120.0,
        retry_delay_s: float = 15.0,
        max_retries: int = 2,
        min_interval_s: float = 30.0,
        prompt: str = "Still there? Let me know if you need anything.",
        retry_prompt: str = "Just checking — are you still with me?",
    ) -> None:
        self._say = say_callback

        # Clamp interval from below.
        self.interval_s = max(interval_s, min_interval_s)
        self.retry_delay_s = retry_delay_s
        self.max_retries = max_retries
        self.min_interval_s = min_interval_s
        self.prompt = prompt
        self.retry_prompt = retry_prompt

        self.state: CheckinState = CheckinState.VOICE_ACTIVE
        self._retries: int = 0
        self._timer_handle: asyncio.TimerHandle | None = None

        # Event that the background loop waits on.  Set whenever the
        # timer fires so the loop wakes and calls _on_timer().
        self._wake: asyncio.Event = asyncio.Event()

        self._loop_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Public control API (called from sync LiveKit event handlers)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background loop and arm the first timer."""
        self._loop_task = asyncio.get_event_loop().create_task(
            self._loop(), name="checkin-loop"
        )
        self._arm(self.interval_s)

    async def stop(self) -> None:
        """Cancel the loop and discard any pending timer cleanly."""
        self._disarm()
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._loop_task = None

    def on_user_spoke(self) -> None:
        """Call when the technician starts speaking (UserState → speaking).

        Any active check-in is cancelled; the silence clock resets.
        """
        if self.state is CheckinState.VOICE_OFF:
            return
        logger.debug("checkin: user spoke → resetting silence clock")
        self.state = CheckinState.VOICE_ACTIVE
        self._retries = 0
        self._arm(self.interval_s)

    def on_agent_speaking(self) -> None:
        """Call when the agent starts speaking a real response (PAUSED)."""
        if self.state is CheckinState.VOICE_OFF:
            return
        logger.debug("checkin: agent speaking → paused")
        self.state = CheckinState.PAUSED
        self._disarm()

    def on_agent_idle(self) -> None:
        """Call when the agent finishes speaking (back to monitoring)."""
        if self.state is CheckinState.VOICE_OFF:
            return
        logger.debug("checkin: agent idle → voice active")
        self.state = CheckinState.VOICE_ACTIVE
        self._retries = 0
        self._arm(self.interval_s)

    def on_settings(self, enabled: bool, interval_s: float | None = None) -> None:
        """Apply settings received from the frontend data channel.

        Parameters
        ----------
        enabled:
            ``True`` to activate check-ins, ``False`` to disable them
            permanently for this session.
        interval_s:
            New silence interval, if provided.  Clamped to
            ``min_interval_s`` from below.
        """
        if interval_s is not None:
            self.interval_s = max(interval_s, self.min_interval_s)

        if not enabled:
            logger.info("checkin: disabled by user settings")
            self.state = CheckinState.VOICE_OFF
            self._disarm()
        else:
            if self.state is CheckinState.VOICE_OFF:
                logger.info("checkin: enabled by user settings")
                self.state = CheckinState.VOICE_ACTIVE
                self._arm(self.interval_s)

    # ------------------------------------------------------------------
    # Internal timer machinery
    # ------------------------------------------------------------------

    def _arm(self, delay_s: float) -> None:
        """Cancel any existing timer and schedule a new one."""
        self._disarm()
        loop = asyncio.get_event_loop()
        self._timer_handle = loop.call_later(delay_s, self._fire)

    def _disarm(self) -> None:
        if self._timer_handle is not None:
            self._timer_handle.cancel()
            self._timer_handle = None

    def _fire(self) -> None:
        """Called by the event loop when the timer expires."""
        self._timer_handle = None
        self._wake.set()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        """Wait for timer fires and execute the appropriate check-in step."""
        while True:
            await self._wake.wait()
            self._wake.clear()

            await self._on_timer()

    async def _on_timer(self) -> None:
        """Advance the state machine on each timer tick."""

        if self.state is CheckinState.VOICE_OFF:
            return

        if self.state is CheckinState.PAUSED:
            # Agent was busy when the timer fired; re-arm and wait.
            self._arm(self.interval_s)
            return

        if self.state is CheckinState.VOICE_ACTIVE:
            logger.info("checkin: silence threshold reached → speaking prompt")
            self.state = CheckinState.CHECKIN_SPEAKING
            try:
                await self._say(self.prompt)
            except Exception:
                logger.exception("checkin: say() failed during initial prompt")
            self.state = CheckinState.CHECKIN_LISTENING
            self._arm(self.retry_delay_s)
            return

        if self.state is CheckinState.CHECKIN_LISTENING:
            # No reply came in before retry_delay_s.
            if self._retries >= self.max_retries:
                logger.info(
                    "checkin: no reply after %d retries → disabling",
                    self._retries,
                )
                self.state = CheckinState.VOICE_OFF
                self._disarm()
                return
            self._retries += 1
            logger.info(
                "checkin: no reply → retry %d/%d",
                self._retries,
                self.max_retries,
            )
            self.state = CheckinState.CHECKIN_SPEAKING
            try:
                await self._say(self.retry_prompt)
            except Exception:
                logger.exception("checkin: say() failed during retry prompt")
            self.state = CheckinState.CHECKIN_LISTENING
            self._arm(self.retry_delay_s)
            return

        if self.state is CheckinState.CHECKIN_RETRY:
            # Legacy alias — treat same as CHECKIN_LISTENING.
            self.state = CheckinState.CHECKIN_LISTENING
            await self._on_timer()
