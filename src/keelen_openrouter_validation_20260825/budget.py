"""Budget-aware retry decisions on top of a planned backoff schedule.

Pure computation only: no clock reads, no randomness, no I/O, and never a
sleep. The caller owns timing and passes ``elapsed_ms`` in.
"""

__all__ = ["RetryBudget"]


class RetryBudget:
    """Answer "should I wait at all?" for a :func:`plan_backoff` schedule.

    Wraps the delay list produced by ``plan_backoff`` plus a total time budget
    in milliseconds. Each call to :meth:`next_delay` asks for the wait before
    the next attempt; when the answer is ``None``, :meth:`stop_reason` names
    exactly one of:

    - ``"attempts"``: the schedule is exhausted (``failures >= len(schedule)``).
    - ``"deadline"``: waiting the next delay would push ``elapsed_ms`` strictly
      past ``deadline_ms`` (a deadline landing exactly on the next delay still
      allows that attempt).
    - ``"done"``: the caller already succeeded and asked again.

    The caller reports success by passing ``failures=None``.
    """

    def __init__(self, schedule: list[int], deadline_ms: int) -> None:
        self._schedule = schedule
        self._deadline_ms = deadline_ms
        self._stop_reason: str | None = None

    def next_delay(self, elapsed_ms: int, failures: int | None) -> int | None:
        """Return the integer millisecond wait before attempt ``failures + 1``.

        That is ``schedule[failures]``, or ``None`` when the caller must stop,
        in which case :meth:`stop_reason` names why. Passing ``failures=None``
        signals the caller already succeeded.

        Raises:
            ValueError: if ``failures`` is a negative int.
        """
        if failures is None:
            self._stop_reason = "done"
            return None
        if failures < 0:
            raise ValueError(f"failures must be >= 0 or None, got {failures}")
        if failures >= len(self._schedule):
            self._stop_reason = "attempts"
            return None

        delay = self._schedule[failures]
        if elapsed_ms + delay > self._deadline_ms:
            self._stop_reason = "deadline"
            return None

        self._stop_reason = None
        return delay

    def stop_reason(self) -> str | None:
        """Return why the most recent :meth:`next_delay` returned ``None``.

        ``None`` before any call and after any call that returned a delay.
        """
        return self._stop_reason
