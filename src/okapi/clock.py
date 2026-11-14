"""Clock abstraction so sequencing logic can be tested without real time.

``now`` is the wall clock (UTC); ``monotonic`` never jumps. ``SimClock`` advances only when
slept on, so a whole observing night runs instantly in tests.
"""

import time
from typing import Protocol, cast

import astropy.units as u
from astropy.time import Time


class Clock(Protocol):
    def now(self) -> Time: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> Time:
        return Time.now()

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimClock:
    """Virtual clock: time moves only when ``sleep`` is called."""

    def __init__(self, start: Time) -> None:
        self._start = start.utc
        self._mono = 0.0

    def now(self) -> Time:
        return cast(Time, self._start + self._mono * u.s)

    def monotonic(self) -> float:
        return self._mono

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._mono += seconds
