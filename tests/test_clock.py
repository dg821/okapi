import astropy.units as u
from astropy.time import Time

from okapi.clock import SimClock, SystemClock


def test_simclock_advances_only_when_slept():
    c = SimClock(Time("2026-01-01T00:00:00", scale="utc"))
    t0 = c.now()
    assert c.monotonic() == 0.0
    c.sleep(12.5)
    assert c.monotonic() == 12.5
    assert abs((c.now() - t0).to_value(u.s) - 12.5) < 1e-6


def test_simclock_ignores_non_positive_sleep():
    c = SimClock(Time("2026-01-01T00:00:00", scale="utc"))
    c.sleep(0)
    c.sleep(-5)
    assert c.monotonic() == 0.0


def test_systemclock_is_monotonic():
    c = SystemClock()
    a = c.monotonic()
    c.sleep(0.001)
    assert c.monotonic() > a
