"""Synthetic frames for testing the streak chain: Gaussian stars, line streaks, noise."""

from dataclasses import dataclass

import numpy as np
from astropy.wcs import WCS


@dataclass(frozen=True)
class SimStar:
    x: float
    y: float
    peak: float


@dataclass(frozen=True)
class SimStreak:
    x0: float
    y0: float
    x1: float
    y1: float
    peak: float  # brightness per pixel along the line


def render_frame(
    shape: tuple[int, int],
    *,
    stars: list[SimStar] = (),  # type: ignore[assignment]
    streaks: list[SimStreak] = (),  # type: ignore[assignment]
    psf_sigma: float = 1.5,
    sky: float = 500.0,
    noise: float = 8.0,
    seed: int = 0,
) -> np.ndarray:
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full(shape, sky, dtype=float)
    for s in stars:
        r2 = (xx - s.x) ** 2 + (yy - s.y) ** 2
        img += s.peak * np.exp(-r2 / (2 * psf_sigma**2))
    for k in streaks:
        n = int(np.hypot(k.x1 - k.x0, k.y1 - k.y0) * 2) + 2
        for t in np.linspace(0.0, 1.0, n):
            x, y = k.x0 + t * (k.x1 - k.x0), k.y0 + t * (k.y1 - k.y0)
            lo_x, hi_x = int(max(x - 6, 0)), int(min(x + 7, w))
            lo_y, hi_y = int(max(y - 6, 0)), int(min(y + 7, h))
            if lo_x >= hi_x or lo_y >= hi_y:
                continue
            r2 = (xx[lo_y:hi_y, lo_x:hi_x] - x) ** 2 + (yy[lo_y:hi_y, lo_x:hi_x] - y) ** 2
            img[lo_y:hi_y, lo_x:hi_x] += (
                k.peak
                * np.exp(-r2 / (2 * psf_sigma**2))
                / (n / (np.hypot(k.x1 - k.x0, k.y1 - k.y0) + 1))
            )
    rng = np.random.default_rng(seed)
    return img + rng.normal(0.0, noise, shape)


def tangent_wcs(
    ra_deg: float, dec_deg: float, shape: tuple[int, int], scale_arcsec: float = 0.74
) -> WCS:
    """A plain gnomonic WCS centered on (ra, dec); x increases to the west, y to the north."""
    h, w = shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra_deg, dec_deg]
    wcs.wcs.crpix = [w / 2 + 0.5, h / 2 + 0.5]
    wcs.wcs.cdelt = [-scale_arcsec / 3600, scale_arcsec / 3600]
    return wcs
