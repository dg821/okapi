"""Simulated plate solver: reports the fake mount's true pointing, plus noise and failures.

It does not look at the image. Rendering a synthetic star field for a real solver to chew on
is a later step; this is enough to test the centering logic.
"""

from collections.abc import Callable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from okapi.camera.base import CapturedImage
from okapi.solve.base import SolveError, SolveResult


class FakeSolver:
    def __init__(
        self,
        truth: Callable[[], SkyCoord],
        *,
        noise: u.Quantity = 0 * u.arcsec,
        fail_first: int = 0,
        always_fail: bool = False,
        seed: int = 0,
    ) -> None:
        self._truth = truth
        self._noise = noise
        self._fail_first = fail_first
        self._always_fail = always_fail
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def solve(self, image: CapturedImage) -> SolveResult:
        self.calls += 1
        if self._always_fail or self.calls <= self._fail_first:
            raise SolveError("simulated solve failure")
        center = self._truth()
        if self._noise > 0 * u.arcsec:
            sigma = float(self._noise.to_value(u.arcsec))
            d_ra, d_dec = (self._rng.normal(0.0, sigma), self._rng.normal(0.0, sigma))
            d_ra, d_dec = d_ra * u.arcsec, d_dec * u.arcsec
            center = center.spherical_offsets_by(d_ra, d_dec)
        return SolveResult(center=center, n_stars=40)
