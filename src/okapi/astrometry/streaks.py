"""Streak detection in a sidereal-stare frame.

Stars are compact blobs; a satellite is an elongated one. We subtract the background, threshold,
label connected pixels, and keep blobs that are long and thin. Each blob's principal axis gives
its centerline and endpoints.

Direction along the streak is NOT recoverable from one frame (nothing marks which end was
first), so downstream code uses the streak midpoint, which does not depend on direction.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy import ndimage, optimize, special


@dataclass(frozen=True)
class Streak:
    x0: float  # endpoints in pixel coordinates (x right, y down), ordered arbitrarily
    y0: float
    x1: float
    y1: float
    width: float  # rms across the streak, pixels
    flux: float  # background-subtracted sum over the blob
    touches_edge: bool  # runs off the frame: endpoints are the frame edge, not the object
    # A star sits on an end of the streak, hiding it: the endpoint is only good to a few pixels.
    contaminated: bool = False

    @property
    def length(self) -> float:
        return float(np.hypot(self.x1 - self.x0, self.y1 - self.y0))

    @property
    def midpoint(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


def background(image: np.ndarray) -> tuple[float, float]:
    """Robust (level, noise) from the median and the MAD, so stars and streaks do not bias it."""
    level = float(np.median(image))
    noise = float(1.4826 * np.median(np.abs(image - level)))
    return level, max(noise, 1e-9)


def _segment_model(
    dx: np.ndarray, dy: np.ndarray, amp: float, half: float, theta: float, sigma: float
) -> np.ndarray:
    """A line segment of half-length ``half`` blurred by a Gaussian PSF, evaluated at offsets
    (dx, dy) from its centre: Gaussian across the line, an erf-edged top-hat along it."""
    c, s_ = np.cos(theta), np.sin(theta)
    along = dx * c + dy * s_
    across = -dx * s_ + dy * c
    root2 = np.sqrt(2.0) * sigma
    profile = 0.5 * (special.erf((along + half) / root2) - special.erf((along - half) / root2))
    return amp * np.exp(-0.5 * (across / sigma) ** 2) * profile


def _fit_segment(
    image: np.ndarray,
    level: float,
    noise: float,
    centre: tuple[float, float],
    half: float,
    theta: float,
    peak: float,
) -> tuple[float, float, float, float, float, bool] | None:
    """Least-squares fit of a blurred segment to the pixels around a blob, ignoring pixels that
    stand well above the model (a star on or beside the streak). Returns
    (cx, cy, half_length, theta, sigma), or None if the fit is not trustworthy."""
    h, w = image.shape
    margin = 20
    x0 = int(max(centre[0] - half * abs(np.cos(theta)) - margin, 0))
    x1 = int(min(centre[0] + half * abs(np.cos(theta)) + margin, w))
    y0 = int(max(centre[1] - half * abs(np.sin(theta)) - margin, 0))
    y1 = int(min(centre[1] + half * abs(np.sin(theta)) + margin, h))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    data = image[y0:y1, x0:x1] - level
    mask = np.ones(data.shape, dtype=bool)

    def residual(p: np.ndarray) -> np.ndarray:
        cx, cy, hl, th, amp, sg = p
        model = _segment_model(xx - cx, yy - cy, amp, hl, th, sg)
        return ((data - model) / noise)[mask]

    p0 = np.array([centre[0], centre[1], half, theta, peak, 1.5])
    lo = [centre[0] - 10, centre[1] - 10, 0.5 * half, theta - 0.1, 0.1 * peak, 0.5]
    hi = [centre[0] + 10, centre[1] + 10, 1.5 * half + 10, theta + 0.1, 10 * peak, 6.0]
    p = p0
    model = np.zeros_like(data)
    for _ in range(3):  # fit, mask what sticks up above the fit, refit
        try:
            sol = optimize.least_squares(
                residual, p, bounds=(lo, hi), x_scale=[1, 1, 5, 0.01, peak, 1]
            )
        except ValueError:
            return None
        p = sol.x
        model = _segment_model(xx - p[0], yy - p[1], p[4], p[2], p[3], p[5])
        mask = (data - model) < 5 * noise
        if mask.sum() < 100:
            return None
    # the fit must explain the data: a poor chi-squared means a curved or ragged blob
    chi2 = float(np.sum(((data - model) / noise)[mask] ** 2) / max(mask.sum() - 6, 1))
    if not np.isfinite(chi2) or chi2 > 6:
        return None
    # An end is hidden if star-light pixels (masked) lie in the zone around either endpoint.
    along = (xx - p[0]) * np.cos(p[3]) + (yy - p[1]) * np.sin(p[3])
    across = -(xx - p[0]) * np.sin(p[3]) + (yy - p[1]) * np.cos(p[3])
    zone = (np.abs(np.abs(along) - p[2]) < 3 * p[5] + 3) & (np.abs(across) < 3 * p[5] + 1)
    contaminated = int(np.sum(~mask & zone)) >= 2
    return float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[5]), contaminated


def _half_max_extent(along: np.ndarray, flux: np.ndarray) -> tuple[float, float]:
    """Ends of a blurred line: where the brightness profile along it falls to half its plateau.
    The thresholded blob extends a few pixels past the true end (the PSF halo), which would
    bias the length and hence the angular rate; the half-maximum point does not."""
    edges = np.arange(np.floor(along.min()), np.ceil(along.max()) + 2.0)
    total, _ = np.histogram(along, bins=edges, weights=flux)
    count, _ = np.histogram(along, bins=edges)
    profile = total / np.maximum(count, 1)
    centers = (edges[:-1] + edges[1:]) / 2
    n = len(profile)
    plateau = float(np.median(profile[n // 4 : max(3 * n // 4, n // 4 + 1)]))
    above = np.nonzero(profile >= plateau / 2)[0]
    if above.size == 0:
        return float(along.min()), float(along.max())
    return float(centers[above[0]] - 0.5), float(centers[above[-1]] + 0.5)


def detect_streaks(
    image: np.ndarray,
    *,
    threshold_sigma: float = 4.0,
    min_length: float = 25.0,
    min_elongation: float = 6.0,
    max_width: float = 8.0,
) -> list[Streak]:
    """Elongated blobs: length at least ``min_length`` px, length/width at least
    ``min_elongation``. Stars (a few px across) and hot pixels never qualify."""
    level, noise = background(image)
    # Smooth slightly so a faint streak, thin and noisy, stays connected.
    smooth = ndimage.gaussian_filter(image.astype(float) - level, 1.0)
    mask = smooth > threshold_sigma * noise / 2.5
    labels = cast(np.ndarray, cast(tuple, ndimage.label(mask, structure=np.ones((3, 3))))[0])
    streaks: list[Streak] = []
    h, w = image.shape
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        ys, xs = np.nonzero(labels[sl] == idx)
        if xs.size < min_length:
            continue
        xs = xs + sl[1].start
        ys = ys + sl[0].start
        weights = np.clip(smooth[ys, xs], 0, None) + 1e-12
        cx, cy = np.average(xs, weights=weights), np.average(ys, weights=weights)
        cov = np.cov(np.vstack([xs - cx, ys - cy]), aweights=weights)
        vals, vecs = np.linalg.eigh(cov)
        minor = np.sqrt(max(vals[0], 0))
        axis = vecs[:, 1]
        along = (xs - cx) * axis[0] + (ys - cy) * axis[1]
        lo, hi = _half_max_extent(along, image[ys, xs] - level)
        length = hi - lo
        width = float(minor)
        if (
            length < min_length
            or width > max_width
            or length / max(2 * width, 1.0) < min_elongation
        ):
            continue
        # a straight line is the only shape we trust: reject curved/ragged blobs
        across = -(xs - cx) * axis[1] + (ys - cy) * axis[0]
        if np.std(across) > max(2.0, 0.1 * length):
            continue
        edge = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2)
        ends = (
            float(cx + lo * axis[0]),
            float(cy + lo * axis[1]),
            float(cx + hi * axis[0]),
            float(cy + hi * axis[1]),
        )
        contaminated = False
        if not edge:
            # The blob's extent is biased by stars touching it; fit the segment itself.
            fit = _fit_segment(
                image.astype(float),
                level,
                noise,
                ((ends[0] + ends[2]) / 2, (ends[1] + ends[3]) / 2),
                length / 2,
                float(np.arctan2(axis[1], axis[0])),
                float(np.percentile(smooth[ys, xs], 90)),
            )
            if fit is not None:
                fcx, fcy, fhalf, fth, fsig, contaminated = fit
                ends = (
                    fcx - fhalf * np.cos(fth),
                    fcy - fhalf * np.sin(fth),
                    fcx + fhalf * np.cos(fth),
                    fcy + fhalf * np.sin(fth),
                )
                width = fsig
        streaks.append(
            Streak(
                x0=float(ends[0]),
                y0=float(ends[1]),
                x1=float(ends[2]),
                y1=float(ends[3]),
                width=width,
                flux=float(np.sum(image[ys, xs] - level)),
                touches_edge=edge,
                contaminated=contaminated,
            )
        )
    return sorted(streaks, key=lambda s: -s.flux)
