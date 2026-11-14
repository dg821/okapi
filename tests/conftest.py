import astropy.units as u
import pytest
from astropy.time import Time

from okapi.clock import SimClock
from okapi.coords import site_from_geodetic


@pytest.fixture
def clock() -> SimClock:
    return SimClock(Time("2026-06-21T04:00:00", scale="utc"))


@pytest.fixture
def site():
    return site_from_geodetic(40.0, -105.0, 1600.0)


@pytest.fixture
def seconds():
    return u.s
