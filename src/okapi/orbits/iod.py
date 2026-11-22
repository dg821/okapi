"""Gauss's angles-only initial orbit determination from three observations (Curtis, Orbital
Mechanics for Engineering Students, Algorithm 5.5, without the final f/g iteration: the
least-squares fit in ``fit.py`` refines the result).

Good for well-separated observations of one object (an arc of a sizeable fraction of an orbit);
poor for short arcs, where range is barely constrained. Use ``determine_orbit`` in practice."""

import numpy as np

from okapi.orbits import MU_EARTH


class IodError(RuntimeError):
    pass


def gauss_iod(
    rho_hat: list[np.ndarray],
    times: list[float],
    observer: list[np.ndarray],
    mu: float = MU_EARTH,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (r2, v2, t2): the state at the middle observation time. ``rho_hat`` are unit
    line-of-sight vectors, ``observer`` the observer's geocentric positions (km), ``times`` in
    seconds, strictly increasing."""
    t1, t2, t3 = times
    if not t1 < t2 < t3:
        raise IodError("observation times must increase")
    r1o, r2o, r3o = observer
    tau1, tau3 = t1 - t2, t3 - t2
    tau = tau3 - tau1
    p1 = np.cross(rho_hat[1], rho_hat[2])
    p2 = np.cross(rho_hat[0], rho_hat[2])
    p3 = np.cross(rho_hat[0], rho_hat[1])
    d0 = float(np.dot(rho_hat[0], p1))
    if abs(d0) < 1e-12:
        raise IodError("lines of sight are coplanar with the observers: no solution")
    big = np.array([[np.dot(r, p) for p in (p1, p2, p3)] for r in (r1o, r2o, r3o)])
    a_ = (1 / d0) * (-big[0, 1] * tau3 / tau + big[1, 1] + big[2, 1] * tau1 / tau)
    b_ = (1 / (6 * d0)) * (
        big[0, 1] * (tau3**2 - tau**2) * tau3 / tau + big[2, 1] * (tau**2 - tau1**2) * tau1 / tau
    )
    e_ = float(np.dot(r2o, rho_hat[1]))
    r2sq = float(np.dot(r2o, r2o))
    coeffs = [
        1,
        0,
        -(a_**2 + 2 * a_ * e_ + r2sq),
        0,
        0,
        -2 * mu * b_ * (a_ + e_),
        0,
        0,
        -(mu**2) * b_**2,
    ]
    roots = [x.real for x in np.roots(coeffs) if abs(x.imag) < 1e-8 and x.real > 6000]
    if not roots:
        raise IodError("no physical root for the middle range")
    r2 = max(roots)  # the largest root is the usual choice; the fit disambiguates
    rho2 = a_ + mu * b_ / r2**3
    rho1 = (1 / d0) * (
        (
            6 * (big[2, 0] * tau1 / tau3 + big[1, 0] * tau / tau3) * r2**3
            + mu * big[2, 0] * (tau**2 - tau1**2) * tau1 / tau3
        )
        / (6 * r2**3 + mu * (tau**2 - tau3**2))
        - big[0, 0]
    )
    rho3 = (1 / d0) * (
        (
            6 * (big[0, 2] * tau3 / tau1 - big[1, 2] * tau / tau1) * r2**3
            + mu * big[0, 2] * (tau**2 - tau3**2) * tau3 / tau1
        )
        / (6 * r2**3 + mu * (tau**2 - tau1**2))
        - big[2, 2]
    )
    rv1 = r1o + rho1 * rho_hat[0]
    rv2 = r2o + rho2 * rho_hat[1]
    rv3 = r3o + rho3 * rho_hat[2]
    f1 = 1 - 0.5 * mu * tau1**2 / r2**3
    f3 = 1 - 0.5 * mu * tau3**2 / r2**3
    g1 = tau1 - mu * tau1**3 / (6 * r2**3)
    g3 = tau3 - mu * tau3**3 / (6 * r2**3)
    v2 = (-f3 * rv1 + f1 * rv3) / (f1 * g3 - f3 * g1)
    return rv2, v2, t2
