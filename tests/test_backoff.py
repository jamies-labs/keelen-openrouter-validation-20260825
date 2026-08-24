"""Tests for plan_backoff: capped exponential backoff scheduling."""

import pytest

from keelen_openrouter_validation_20260825 import backoff, plan_backoff


def _assert_plain_int_delays(delays: list[int]) -> None:
    """Every element must be a plain int, never float or bool."""
    for index, delay in enumerate(delays):
        assert type(delay) is int, (
            f"delay[{index}]={delay!r} is {type(delay).__name__}, expected plain int"
        )


def test_canonical_schedule() -> None:
    delays = plan_backoff(5, 100, 800)

    assert delays == [100, 200, 400, 800, 800], f"{delays!r} != [100, 200, 400, 800, 800]"
    _assert_plain_int_delays(delays)


@pytest.mark.parametrize(
    ("attempts", "base_ms", "cap_ms", "expected"),
    [
        (3, 100, 400, [100, 200, 400]),
        (4, 100, 400, [100, 200, 400, 400]),
        (3, 250, 250, [250, 250, 250]),
    ],
)
def test_cap_boundaries(attempts: int, base_ms: int, cap_ms: int, expected: list[int]) -> None:
    delays = plan_backoff(attempts, base_ms, cap_ms)

    assert delays == expected, f"plan_backoff({attempts}, {base_ms}, {cap_ms}) -> {delays!r}"
    _assert_plain_int_delays(delays)


def test_attempts_boundary() -> None:
    empty_plan = plan_backoff(0, 100, 800)

    assert empty_plan == [], f"plan_backoff(0, 100, 800) returned {empty_plan!r}, expected []"

    with pytest.raises(ValueError) as excinfo:
        plan_backoff(-1, 100, 800)

    assert "attempts" in str(excinfo.value), (
        f"ValueError message {str(excinfo.value)!r} does not name 'attempts'"
    )


@pytest.mark.parametrize(
    ("attempts", "base_ms", "cap_ms", "raises", "message_names"),
    [
        (5, 0, 800, True, "base_ms"),
        (5, -10, 800, True, "base_ms"),
        (2, 500, 499, True, "cap_ms"),
        (1, 1, 1, False, ""),
        (2, 500, 500, False, ""),
        (7, 100, 800, False, ""),
    ],
)
def test_argument_validation(
    attempts: int, base_ms: int, cap_ms: int, raises: bool, message_names: str
) -> None:
    if not raises:
        delays = plan_backoff(attempts, base_ms, cap_ms)

        assert len(delays) == attempts, f"{delays!r} has {len(delays)} entries, want {attempts}"
        return

    with pytest.raises(ValueError) as excinfo:
        plan_backoff(attempts, base_ms, cap_ms)

    assert message_names in str(excinfo.value), (
        f"ValueError for plan_backoff({attempts}, {base_ms}, {cap_ms}) was "
        f"{str(excinfo.value)!r}, expected it to name '{message_names}'"
    )


def test_deterministic_public_export() -> None:
    first = plan_backoff(4, 50, 300)
    second = plan_backoff(4, 50, 300)

    assert first == second, f"consecutive identical calls differ: {first!r} vs {second!r}"
    assert first == [50, 100, 200, 300], f"{first!r} != [50, 100, 200, 300]"

    assert plan_backoff is backoff.plan_backoff, (
        "package-root plan_backoff is not the same callable as backoff.plan_backoff"
    )
