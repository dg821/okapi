"""From a captured frame to satellite observations: decode, plate-solve, find streaks, convert."""

from collections.abc import Callable
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from okapi.astrometry.observation import Observation, streak_to_observation
from okapi.astrometry.streaks import detect_streaks
from okapi.camera.base import CapturedImage
from okapi.solve.base import SolveError, SolverProtocol


@dataclass(frozen=True)
class FrameResult:
    """Everything learned from one frame. ``observations`` may legitimately be empty."""

    center: SkyCoord
    n_streaks: int
    observations: list[Observation]
    warnings: tuple[str, ...] = field(default=())


def observations_from_array(
    data: np.ndarray,
    wcs: WCS,
    *,
    start: Time,
    exposure: u.Quantity,
    timing_sigma: u.Quantity = 0.05 * u.s,
    centroid_sigma_px: float = 0.5,
    **detect_kw: float,
) -> list[Observation]:
    """Streaks in an already-decoded frame with a known pixel-to-sky mapping."""
    scale = float(np.mean(proj_plane_pixel_scales(wcs))) * u.deg
    return [
        streak_to_observation(
            s,
            wcs,
            start=start,
            exposure=exposure,
            centroid_sigma_px=centroid_sigma_px,
            pixel_scale=scale.to(u.arcsec),
            timing_sigma=timing_sigma,
        )
        for s in detect_streaks(data, **detect_kw)
    ]


def process_frame(
    image: CapturedImage,
    *,
    solver: SolverProtocol,
    decode: Callable[[CapturedImage], np.ndarray],
    timing_sigma: u.Quantity = 0.05 * u.s,
    **detect_kw: float,
) -> FrameResult:
    """Decode and plate-solve ``image``, then extract satellite streaks.

    ``decode`` must produce the same pixel grid the solver solved (for ``AstapSolver``, pass its
    ``decoder`` property). Raises ``SolveError`` if there is no solution or the solver gives no
    pixel mapping: without the stars there is no astrometry."""
    data = decode(image)
    solution = solver.solve(image)
    if solution.wcs is None:
        raise SolveError("the solver returned no pixel-to-sky mapping (WCS)")
    obs = observations_from_array(
        data,
        solution.wcs,
        start=image.start_time,
        exposure=image.settings.exposure,
        timing_sigma=timing_sigma,
        **detect_kw,
    )
    warnings: list[str] = []
    if len(obs) > 1:
        warnings.append(f"{len(obs)} streaks in one frame; each is an observation of some object")
    return FrameResult(solution.center, len(obs), obs, tuple(warnings))
