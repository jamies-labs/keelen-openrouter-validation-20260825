"""Deterministic backoff scheduling.

Pure computation only: no clock reads, no randomness, no I/O, and never a
sleep. The caller owns timing.
"""

__all__ = ["plan_backoff"]


def plan_backoff(attempts: int, base_ms: int, cap_ms: int) -> list[int]:
    """Return the integer millisecond wait before each retry attempt.

    The first delay is ``base_ms``; each subsequent delay doubles the previous
    one, clamped at ``cap_ms``. ``plan_backoff(5, 100, 800)`` returns
    ``[100, 200, 400, 800, 800]``.

    Raises:
        ValueError: if ``attempts < 0``, ``base_ms <= 0``, or ``cap_ms < base_ms``.
    """
    if attempts < 0:
        raise ValueError(f"attempts must be >= 0, got {attempts}")
    if base_ms <= 0:
        raise ValueError(f"base_ms must be > 0, got {base_ms}")
    if cap_ms < base_ms:
        raise ValueError(f"cap_ms ({cap_ms}) must be >= base_ms ({base_ms})")

    delays: list[int] = []
    delay = base_ms
    for _ in range(attempts):
        delays.append(delay)
        delay = min(delay * 2, cap_ms)
    return delays
