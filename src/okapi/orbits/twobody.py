"""Analytic two-body propagation with universal variables (valid for every conic), and the
conversion between a state vector and classical elements."""

from dataclasses import dataclass

import numpy as np

from okapi.orbits import MU_EARTH


def _stumpff(z: float) -> tuple[float, float]:
    if z > 1e-8:
        s = np.sqrt(z)
        return (1 - np.cos(s)) / z, (s - np.sin(s)) / s**3
    if z < -1e-8:
        s = np.sqrt(-z)
        return (np.cosh(s) - 1) / -z, (np.sinh(s) - s) / s**3
    return 0.5 - z / 24, 1 / 6 - z / 120


def propagate(
    r0: np.ndarray, v0: np.ndarray, dt: float, mu: float = MU_EARTH
) -> tuple[np.ndarray, np.ndarray]:
    """State after ``dt`` seconds (negative propagates backwards)."""
    if dt == 0:
        return r0.copy(), v0.copy()
    r0n = float(np.linalg.norm(r0))
    vr0 = float(np.dot(r0, v0)) / r0n
    alpha = 2 / r0n - float(np.dot(v0, v0)) / mu  # 1/a
    sq = np.sqrt(mu)
    chi = sq * abs(alpha) * dt if abs(alpha) > 1e-12 else sq * dt / r0n
    for _ in range(100):
        z = alpha * chi * chi
        c, s = _stumpff(z)
        f = r0n * vr0 / sq * chi * chi * c + (1 - alpha * r0n) * chi**3 * s + r0n * chi - sq * dt
        df = r0n * vr0 / sq * chi * (1 - z * s) + (1 - alpha * r0n) * chi * chi * c + r0n
        step = f / df
        chi -= step
        if abs(step) < 1e-10 * max(1.0, abs(chi)):
            break
    z = alpha * chi * chi
    c, s = _stumpff(z)
    f_ = 1 - chi * chi / r0n * c
    g = dt - chi**3 / sq * s
    r = f_ * r0 + g * v0
    rn = float(np.linalg.norm(r))
    fdot = sq / (rn * r0n) * (z * s - 1) * chi
    gdot = 1 - chi * chi / rn * c
    return r, fdot * r0 + gdot * v0


@dataclass(frozen=True)
class Elements:
    a: float  # km (negative for a hyperbola)
    e: float
    i: float  # rad
    raan: float
    argp: float
    nu: float  # true anomaly

    @property
    def period(self) -> float:
        return float(2 * np.pi * np.sqrt(abs(self.a) ** 3 / MU_EARTH))

    @property
    def perigee(self) -> float:
        return self.a * (1 - self.e)


def elements_from_state(r: np.ndarray, v: np.ndarray, mu: float = MU_EARTH) -> Elements:
    rn, vn = float(np.linalg.norm(r)), float(np.linalg.norm(v))
    h = np.cross(r, v)
    hn = float(np.linalg.norm(h))
    node = np.cross([0.0, 0.0, 1.0], h)
    nn = float(np.linalg.norm(node))
    evec = ((vn**2 - mu / rn) * r - np.dot(r, v) * v) / mu
    e = float(np.linalg.norm(evec))
    a = 1 / (2 / rn - vn**2 / mu)
    i = float(np.arccos(np.clip(h[2] / hn, -1, 1)))
    raan = float(np.arctan2(node[1], node[0])) % (2 * np.pi) if nn > 1e-12 else 0.0

    def angle(u_: np.ndarray, w: np.ndarray, ref: np.ndarray) -> float:
        ang = float(
            np.arccos(np.clip(np.dot(u_, w) / (np.linalg.norm(u_) * np.linalg.norm(w)), -1, 1))
        )
        return ang if np.dot(np.cross(u_, w), ref) >= 0 else 2 * np.pi - ang

    if nn > 1e-12 and e > 1e-10:
        argp = angle(node, evec, h)
    elif e > 1e-10:  # equatorial: measure from the x axis
        argp = angle(np.array([1.0, 0, 0]), evec, h)
    else:
        argp = 0.0
    if e > 1e-10:
        nu = angle(evec, r, h)
    elif nn > 1e-12:
        nu = angle(node, r, h)
    else:
        nu = angle(np.array([1.0, 0, 0]), r, h)
    return Elements(a, e, i, raan, argp, nu)


def state_from_elements(el: Elements, mu: float = MU_EARTH) -> tuple[np.ndarray, np.ndarray]:
    p = el.a * (1 - el.e**2)
    r_pf = p / (1 + el.e * np.cos(el.nu)) * np.array([np.cos(el.nu), np.sin(el.nu), 0.0])
    v_pf = np.sqrt(mu / p) * np.array([-np.sin(el.nu), el.e + np.cos(el.nu), 0.0])

    def rz(t: float) -> np.ndarray:
        c, s = np.cos(t), np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    def rx(t: float) -> np.ndarray:
        c, s = np.cos(t), np.sin(t)
        return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])

    q = rz(el.raan) @ rx(el.i) @ rz(el.argp)
    return q @ r_pf, q @ v_pf
