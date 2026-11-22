"""Rough brightness calibration from catalog stars in a solved frame.

Aperture photometry at known star positions gives a zero point (magnitude of a source that
makes one count in the exposure); from it a streak's total flux becomes a magnitude, and the
frame's noise gives a limiting magnitude. This answers "can this setup see a GEO satellite in
20 s?" and doubles as a check that the telescope is looking where it thinks.

Limits: the star catalog in ``okapi.catalog`` holds only bright stars (V < ~2.5), which saturate
a T3i at long exposures; saturated stars are dropped, so a frame may have none usable. The
magnitudes are in whatever band the catalog uses (visual) against a Bayer-sensor response:
good to a few tenths of a magnitude at best.
"""

import math
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS


class PhotometryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ZeroPoint:
    value: float  # mag = value - 2.5 log10(counts)
    scatter: float  # standard deviation of the per-star values (0 for a single star)
    n_stars: int
    noise: float  # per-pixel background noise (counts)
    aperture_px: float


def aperture_flux(
    image: np.ndarray,
    x: float,
    y: float,
    radius: float = 6.0,
    annulus: tuple[float, float] = (9.0, 14.0),
) -> tuple[float, float, float]:
    """(background-subtracted flux, background noise per pixel, peak pixel) at (x, y)."""
    h, w = image.shape
    x0, x1 = int(max(x - annulus[1], 0)), int(min(x + annulus[1] + 1, w))
    y0, y1 = int(max(y - annulus[1], 0)), int(min(y + annulus[1] + 1, h))
    if x1 - x0 < 5 or y1 - y0 < 5:
        raise PhotometryError("star too close to the frame edge")
    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.hypot(xx - x, yy - y)
    cut = image[y0:y1, x0:x1]
    ring = cut[(r >= annulus[0]) & (r <= annulus[1])]
    if ring.size < 20:
        raise PhotometryError("star too close to the frame edge")
    level = float(np.median(ring))
    noise = float(1.4826 * np.median(np.abs(ring - level)))
    inside = r <= radius
    return float(np.sum(cut[inside] - level)), noise, float(cut[inside].max())


def fit_zero_point(
    image: np.ndarray,
    wcs: WCS,
    stars: list[tuple[SkyCoord, float]],
    *,
    saturation: float = 0.9 * 16383,
    radius: float = 6.0,
    min_snr: float = 10.0,
) -> ZeroPoint:
    """Zero point from (position, magnitude) reference stars; saturated and faint ones skipped."""
    values: list[float] = []
    noises: list[float] = []
    for coord, mag in stars:
        px, py = (float(v) for v in wcs.world_to_pixel(coord))
        if not (math.isfinite(px) and math.isfinite(py)):
            continue
        try:
            flux, noise, peak = aperture_flux(image, px, py, radius)
        except PhotometryError:
            continue
        if (
            peak >= saturation
            or flux <= 0
            or flux < min_snr * noise * math.sqrt(math.pi * radius**2)
        ):
            continue
        values.append(mag + 2.5 * math.log10(flux))
        noises.append(noise)
    if not values:
        raise PhotometryError(
            "no usable reference stars (all saturated, too faint, or off the frame)"
        )
    return ZeroPoint(
        value=float(np.median(values)),
        scatter=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        n_stars=len(values),
        noise=float(np.median(noises)),
        aperture_px=radius,
    )


def magnitude(flux: float, zp: ZeroPoint) -> float:
    """Magnitude of a source (point or streak) with this total background-subtracted flux."""
    if flux <= 0:
        raise PhotometryError("flux must be positive")
    return zp.value - 2.5 * math.log10(flux)


def limiting_magnitude(zp: ZeroPoint, snr: float = 5.0) -> float:
    """Magnitude of a point source at ``snr`` against the sky noise in the aperture."""
    return magnitude(snr * zp.noise * math.sqrt(math.pi * zp.aperture_px**2), zp)


def streak_limiting_magnitude(zp: ZeroPoint, length_px: float, snr: float = 5.0) -> float:
    """Total-flux magnitude of a streak of ``length_px`` detectable at ``snr``: the noise grows
    with the pixels covered, so a long streak needs more total flux than a point."""
    area = length_px * 2 * zp.aperture_px / 2  # a strip about one aperture radius wide each side
    return magnitude(snr * zp.noise * math.sqrt(area), zp)


def angular_scale(wcs: WCS) -> u.Quantity:
    from astropy.wcs.utils import proj_plane_pixel_scales

    return float(np.mean(proj_plane_pixel_scales(wcs))) * u.deg
