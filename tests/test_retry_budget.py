"""Tests for RetryBudget: budget-aware next-delay decisions with stop reasons."""

import time

import pytest

from keelen_openrouter_validation_20260825 import RetryBudget, budget, plan_backoff


def _assert_plain_int_delays(delays: list[int]) -> None:
    """Every element must be a plain int, never float or bool."""
    for index, delay in enumerate(delays):
        assert type(delay) is int, (
            f"delay[{index}]={delay!r} is {type(delay).__name__}, expected plain int"
        )


def _clock_guard(name: str):
    """Build a stand-in for a time.* function that fails loud on any use."""

    def _raise(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"purity violation: RetryBudget called time.{name}")

    return _raise


def test_walks_schedule_then_stops_on_attempts() -> None:
    subject = RetryBudget(plan_backoff(5, 100, 800), deadline_ms=10_000)

    assert subject.stop_reason() is None, (
        f"fresh budget reported stop_reason={subject.stop_reason()!r}, expected None"
    )

    delays: list[int] = []
    for failures in range(5):
        delay = subject.next_delay(0, failures)
        assert delay is not None, f"ask {failures} returned None before the schedule ran out"
        delays.append(delay)
        assert subject.stop_reason() is None, (
            f"ask {failures} was granted delay {delay!r} but set "
            f"stop_reason={subject.stop_reason()!r}"
        )
    _assert_plain_int_delays(delays)

    # Exact cap hit: the clamped entries come back exactly 800, never re-doubled to 1600.
    assert delays == [100, 200, 400, 800, 800], f"{delays!r} != [100, 200, 400, 800, 800]"

    stopped = subject.next_delay(0, 5)
    assert stopped is None, f"failures=5 ask returned {stopped!r}, expected None"
    assert subject.stop_reason() == "attempts", (
        f"exhausted schedule reported stop_reason={subject.stop_reason()!r}, expected 'attempts'"
    )


def test_deadline_landing_exactly_on_next_delay() -> None:
    schedule = plan_backoff(1, 500, 500)
    assert schedule == [500], f"{schedule!r} != [500]"

    exact = RetryBudget(schedule, deadline_ms=800)
    granted = exact.next_delay(300, 0)
    assert granted == 500, (
        f"next_delay(300, 0) returned {granted!r}, expected 500: 300 + 500 lands exactly "
        "on the 800ms deadline and equality must proceed"
    )
    assert exact.stop_reason() is None, (
        f"exact-deadline grant set stop_reason={exact.stop_reason()!r}, expected None"
    )

    over = RetryBudget(schedule, deadline_ms=800)
    denied = over.next_delay(301, 0)
    assert denied is None, f"next_delay(301, 0) returned {denied!r}, expected None"
    assert over.stop_reason() == "deadline", (
        f"past-deadline ask reported stop_reason={over.stop_reason()!r}, expected 'deadline'"
    )


def test_done_reason_and_precedence() -> None:
    succeeded = RetryBudget(plan_backoff(3, 100, 800), deadline_ms=10_000)
    assert succeeded.next_delay(0, None) is None, "success ask returned a delay"
    assert succeeded.stop_reason() == "done", (
        f"success ask reported stop_reason={succeeded.stop_reason()!r}, expected 'done'"
    )

    on_empty = RetryBudget(plan_backoff(0, 100, 800), deadline_ms=10_000)
    assert on_empty.next_delay(0, None) is None, "success ask on empty schedule returned a delay"
    assert on_empty.stop_reason() == "done", (
        "'done' lost to the empty schedule: stop_reason="
        f"{on_empty.stop_reason()!r}, expected 'done'"
    )

    past_deadline = RetryBudget(plan_backoff(3, 100, 800), deadline_ms=1_000)
    assert past_deadline.next_delay(5_000, None) is None, "success ask past deadline returned a delay"
    assert past_deadline.stop_reason() == "done", (
        "'done' lost to the spent deadline: stop_reason="
        f"{past_deadline.stop_reason()!r}, expected 'done'"
    )


def test_zero_attempts_empty_schedule() -> None:
    empty_plan = plan_backoff(0, 100, 800)
    assert empty_plan == [], f"{empty_plan!r} != []"

    subject = RetryBudget(empty_plan, deadline_ms=1_000)
    for failures in (0, 1, 7):
        result = subject.next_delay(0, failures)
        assert result is None, f"next_delay(0, {failures}) returned {result!r}, expected None"
        assert subject.stop_reason() == "attempts", (
            f"empty schedule ask at failures={failures} reported "
            f"stop_reason={subject.stop_reason()!r}, expected 'attempts'"
        )


def test_negative_failures_is_rejected_not_wrapped() -> None:
    subject = RetryBudget(plan_backoff(2, 100, 800), deadline_ms=10_000)

    with pytest.raises(ValueError) as excinfo:
        subject.next_delay(0, -1)

    assert "failures" in str(excinfo.value), (
        f"ValueError message {str(excinfo.value)!r} does not name 'failures'"
    )


def test_pure_no_clock_no_sleep_and_package_export(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", _clock_guard("time"))
    monkeypatch.setattr(time, "monotonic", _clock_guard("monotonic"))
    monkeypatch.setattr(time, "sleep", _clock_guard("sleep"))

    subject = RetryBudget(plan_backoff(4, 50, 300), deadline_ms=1_000)
    first = subject.next_delay(0, 0)
    repeat = subject.next_delay(0, 0)
    second_ask = subject.next_delay(50, 1)

    monkeypatch.undo()

    assert first == 50, f"first ask returned {first!r}, expected 50"
    assert repeat == first, f"identical arguments yielded {repeat!r} vs {first!r}"
    assert second_ask == 100, f"second ask returned {second_ask!r}, expected 100"

    from keelen_openrouter_validation_20260825 import RetryBudget as reimported_budget

    assert reimported_budget is budget.RetryBudget, (
        "package-root RetryBudget is not the same callable as budget.RetryBudget"
    )
