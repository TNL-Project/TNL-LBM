"""Unit tests for the counter bindings of pytnl_lbm.

``counter<REAL>`` (state.h:48) drives periodic actions in the simulation loop;
``action(time)`` fires when ``period > 0`` and ``time >= count * period``.
Tests cover the disabled default, the trigger boundary, and attribute
read/write roundtrips for both exported precisions.
"""

from __future__ import annotations

import pytest

import pytnl_lbm

type AnyCounter = pytnl_lbm.counter_float | pytnl_lbm.counter_double

COUNTER_CLASSES = ["counter_float", "counter_double"]


@pytest.fixture(params=COUNTER_CLASSES)
def counter(request: pytest.FixtureRequest) -> AnyCounter:
    return getattr(pytnl_lbm, request.param)()


class TestCounterDefaults:
    def test_default_values(self, counter: AnyCounter) -> None:
        assert counter.count == 0
        assert counter.period == pytest.approx(-1.0)

    def test_disabled_period_never_fires(self, counter: AnyCounter) -> None:
        assert not counter.action(0.0)
        assert not counter.action(1e9)


class TestCounterAction:
    def test_fires_at_multiple_of_period(self, counter: AnyCounter) -> None:
        counter.count = 4
        counter.period = 0.5
        assert not counter.action(1.99)
        assert counter.action(2.0)
        assert counter.action(3.14)

    def test_zero_period_disables(self, counter: AnyCounter) -> None:
        counter.count = 1
        counter.period = 0.0
        assert not counter.action(1e9)

    def test_negative_period_disables(self, counter: AnyCounter) -> None:
        counter.count = 1
        counter.period = -0.5
        assert not counter.action(1e9)


class TestCounterAttributes:
    def test_count_roundtrip(self, counter: AnyCounter) -> None:
        counter.count = 5
        assert counter.count == 5

    def test_period_roundtrip(self, counter: AnyCounter) -> None:
        counter.period = 1.25
        assert counter.period == pytest.approx(1.25)
