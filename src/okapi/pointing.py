"""Pi-side pointing model: the rotation between where the mount reports and where it points.

A Dobsonian that is not perfectly level, or whose azimuth zero is off, reports positions in a
frame that is a rotation of the true horizon frame. Two centered stars (each gives the true
position, from the sky, and the reported position, from the mount) fix that rotation. This is
the fallback for handsets that cannot accept an alignment from the Pi (PROJECT_PLAN A2); it also
composes cleanly with whatever alignment the handset already holds, provided that stays fixed.

It models a rotation only. A pure altitude-encoder zero offset is not a rotation and leaves a
residual; the calibration's verification slew measures what is left.
"""

from dataclasses import dataclass

import astropy.units as u
import numpy as np

from okapi.coords import AltAzPosition, altaz_to_vec, vec_to_altaz


class PointingModelError(ValueError):
    """Not enough or not good enough data to fit a model."""


def rotation_from_vector(rotvec: tuple[u.Quantity, u.Quantity, u.Quantity]) -> np.ndarray:
    """Rotation matrix from an axis-angle vector (Rodrigues); zero vector gives identity."""
    v = np.array([float(a.to_value(u.rad)) for a in rotvec])
    angle = np.linalg.norm(v)
    if angle == 0.0:
        return np.eye(3)
    k = v / angle
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


@dataclass(frozen=True)
class PointingModel:
    """``to_reported(true)`` is what the mount reports when it truly points at ``true``."""

    rotation: np.ndarray  # 3x3, maps true unit vectors to reported ones

    @classmethod
    def identity(cls) -> "PointingModel":
        return cls(np.eye(3))

    def to_reported(self, true: AltAzPosition) -> AltAzPosition:
        return vec_to_altaz(self.rotation @ altaz_to_vec(true))

    def to_true(self, reported: AltAzPosition) -> AltAzPosition:
        return vec_to_altaz(self.rotation.T @ altaz_to_vec(reported))

    @classmethod
    def fit(
        cls,
        pairs: list[tuple[AltAzPosition, AltAzPosition]],
        *,
        min_separation: u.Quantity = 20 * u.deg,
    ) -> "PointingModel":
        """Least-squares rotation (Kabsch) from ``(true, reported)`` pairs; needs 2 or more."""
        if len(pairs) < 2:
            raise PointingModelError("need at least two alignment stars")
        spread = max(a[0].separation(b[0]) for a in pairs for b in pairs)
        if spread < min_separation:
            raise PointingModelError(
                f"alignment stars only {spread:.1f} apart; need at least {min_separation}"
            )
        t = np.array([altaz_to_vec(p[0]) for p in pairs])  # n x 3 true
        r = np.array([altaz_to_vec(p[1]) for p in pairs])  # n x 3 reported
        h = t.T @ r
        u_, _, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u_.T))
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u_.T
        return cls(rot)

    def residuals(self, pairs: list[tuple[AltAzPosition, AltAzPosition]]) -> list[u.Quantity]:
        return [self.to_reported(true).separation(reported) for true, reported in pairs]

    def rms(self, pairs: list[tuple[AltAzPosition, AltAzPosition]]) -> u.Quantity:
        res = np.array([x.to_value(u.arcsec) for x in self.residuals(pairs)])
        return float(np.sqrt(np.mean(res**2))) * u.arcsec
