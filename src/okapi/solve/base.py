"""Plate-solver interface: find where a frame is pointing from the stars in it."""

from dataclasses import dataclass
from typing import Protocol

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from okapi.camera.base import CapturedImage


class SolveError(RuntimeError):
    """No solution (clouds, too few stars, bad exposure, solver crash)."""


@dataclass(frozen=True)
class SolveResult:
    center: SkyCoord  # ICRS position of the image center
    n_stars: int | None = None  # stars matched, a quality indicator, when the solver reports it
    # Pixel-to-sky mapping of the array the solver itself decoded and solved (for ASTAP, the
    # binned frame), when the solver provides it. Needed to measure positions within a frame.
    wcs: WCS | None = None


class SolverProtocol(Protocol):
    def solve(self, image: CapturedImage) -> SolveResult:
        """Blind or hinted solve. Raises ``SolveError`` when no solution is found."""
        ...
