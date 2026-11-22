"""Measure the shutter-timing offset from a satellite whose orbit is known.

The frame's start time is the system clock just before the shutter command; the shutter really
opens some unknown delay later. A streak's midpoint is where the satellite was at the true
mid-exposure, so for a satellite with a trusted orbit (a fresh TLE) the time at which it was at
the measured position, minus the recorded mid-exposure time, is that delay. For LEO (about
7 km/s) a 0.3 s delay is ~2 km, so this matters; for GEO it hardly does.
"""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from scipy.optimize import minimize_scalar

from okapi.astrometry.observation import Observation
from okapi.targets import Target


class TimingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TimingOffset:
    offset: u.Quantity  # true time minus recorded time (positive: shutter opened late)
    scatter: u.Quantity  # sample standard deviation across frames (0 for one frame)
    n: int
    per_frame: tuple[u.Quantity, ...]


def _frame_offset(o: Observation, target: Target, site: EarthLocation, window_s: float) -> float:
    def miss(dt: float) -> float:
        return float(target.predict(site, o.time + dt * u.s).separation(o.coord).to_value(u.arcsec))

    grid = np.linspace(-window_s, window_s, 41)
    best = float(grid[int(np.argmin([miss(float(g)) for g in grid]))])
    step = grid[1] - grid[0]
    res = minimize_scalar(miss, bounds=(best - step, best + step), method="bounded")
    return float(res.x)


def estimate_time_offset(
    observations: list[Observation],
    target: Target,
    site: EarthLocation,
    *,
    window: u.Quantity = 5 * u.s,
    max_residual: u.Quantity = 120 * u.arcsec,
) -> TimingOffset:
    """Offset of the recorded times, from timed points matched to ``target``.

    Frames whose best-fit position still misses by more than ``max_residual`` (a wrong target, or
    a cross-track error that no time shift explains) are rejected rather than averaged in."""
    w = float(window.to_value(u.s))
    offsets: list[float] = []
    for o in observations:
        if o.kind != "point":
            continue
        dt = _frame_offset(o, target, site, w)
        miss = target.predict(site, o.time + dt * u.s).separation(o.coord)
        if miss <= max_residual and abs(dt) < w * 0.98:
            offsets.append(dt)
    if not offsets:
        raise TimingError(
            "no timed streak matches the target's track (wrong satellite, stale TLE, "
            "or only field-crossing lines)"
        )
    arr = np.array(offsets)
    return TimingOffset(
        offset=float(np.median(arr)) * u.s,
        scatter=(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0) * u.s,
        n=len(arr),
        per_frame=tuple(float(x) * u.s for x in arr),
    )
