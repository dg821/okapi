"""Fit a two-body orbit to streak observations by weighted least squares.

Unknowns: the 6-element geocentric state at the first observation's time. Residuals, each divided
by its uncertainty:

* **point** observation (both streak ends in the frame): position at mid-exposure (2), angular
  rate (1) and direction of motion (1). The position uncertainty includes the shutter-timing
  error converted to along-track distance.
* **line** observation (streak runs off the frame): the object passed through the line at some
  unknown time inside the exposure. Residuals: cross-track miss at the moment the predicted track
  crosses the line's midpoint (1), direction of motion (1), and a penalty if that moment falls
  outside the exposure (1). Nothing constrains where along the line the object was at any
  particular time, so lines carry much less information than points.

Starting guesses come from a grid of ranges (and Gauss's method when there are three
observations); the best few are refined. The result reports the formal covariance and whether the
problem was well conditioned: from a short arc, range is often barely constrained and the honest
answer is a large uncertainty, not a confident orbit.
"""

from dataclasses import dataclass, field
from typing import Any, cast

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation
from astropy.time import Time
from scipy.optimize import least_squares

from okapi.astrometry.observation import Observation
from okapi.orbits import MU_EARTH, R_EARTH
from okapi.orbits.geometry import observer_position, tangent_offset, unit_vector
from okapi.orbits.iod import IodError, gauss_iod
from okapi.orbits.twobody import Elements, elements_from_state, propagate


class OrbitFitError(RuntimeError):
    pass


@dataclass(frozen=True)
class OrbitSolution:
    epoch: Time
    r: np.ndarray  # km, GCRS
    v: np.ndarray  # km/s
    elements: Elements
    chi2_reduced: float  # about 1 if the model and the stated uncertainties are right
    n_residuals: int
    covariance: np.ndarray  # 6x6 formal covariance of (r, v)
    condition: float  # of the weighted Jacobian; large means some direction is unconstrained
    warnings: tuple[str, ...] = field(default=())

    @property
    def r_sigma(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance)[:3])

    @property
    def v_sigma(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance)[3:])


@dataclass
class _Prepared:
    obs: Observation
    t: float  # seconds from epoch, mid-exposure
    half: float  # half the exposure, s
    r_obs: np.ndarray  # observer position at t - half, t, t + half
    ra: float
    dec: float
    sigma: float  # rad
    sigma_pos: float  # rad, including timing for points
    rate: float | None  # rad/s
    pa: float  # rad
    length: float | None  # rad


def _wrap_half_turn(x: float) -> float:
    return (x + np.pi / 2) % np.pi - np.pi / 2


def _prepare(observations: list[Observation], site: EarthLocation, epoch: Time) -> list[_Prepared]:
    return [
        _make_prepared(o, site, float((o.time - epoch).to_value(u.s)), epoch) for o in observations
    ]


def _make_prepared(o: Observation, site: EarthLocation, t: float, epoch: Time) -> _Prepared:
    # Exposure length: for lines time_sigma is half the exposure; for points it is the timing
    # error, so the exposure comes from length / rate.
    if o.kind == "line":
        half = float(o.time_sigma.to_value(u.s))
    else:
        assert o.rate is not None and o.length is not None
        half = float((o.length / o.rate).to_value(u.s)) / 2
    times = [t - half, t, t + half]
    r_obs = np.array([observer_position(site, epoch + tt * u.s) for tt in times])
    sigma = float(o.sigma.to_value(u.rad))
    rate = float(o.rate.to_value(u.rad / u.s)) if o.rate is not None else None
    sigma_pos = sigma
    if o.kind == "point" and rate is not None:
        sigma_pos = float(np.hypot(sigma, rate * float(o.time_sigma.to_value(u.s))))
    return _Prepared(
        o,
        t,
        half,
        r_obs,
        float(cast(Any, o.coord).ra.to_value(u.rad)),
        float(cast(Any, o.coord).dec.to_value(u.rad)),
        sigma,
        sigma_pos,
        rate,
        float(o.position_angle.to_value(u.rad)),
        float(o.length.to_value(u.rad)) if o.length is not None else None,
    )


def _residuals(x: np.ndarray, prepared: list[_Prepared]) -> np.ndarray:
    r0, v0 = x[:3], x[3:]
    res: list[float] = []
    for p in prepared:
        dirs = []
        for k, dt in enumerate((p.t - p.half, p.t, p.t + p.half)):
            r, _ = propagate(r0, v0, dt)
            dirs.append(r - p.r_obs[k])
        e0, n0 = tangent_offset(dirs[1], p.ra, p.dec)
        w_e, w_n = (
            a - b
            for a, b in zip(
                tangent_offset(dirs[2], p.ra, p.dec),
                tangent_offset(dirs[0], p.ra, p.dec),
                strict=True,
            )
        )
        speed = float(np.hypot(w_e, w_n))  # rad over the whole exposure
        pa_pred = float(np.arctan2(w_e, w_n)) if speed > 0 else 0.0
        if p.obs.kind == "point":
            assert p.rate is not None and p.length is not None
            # Timing error only blurs the position ALONG the track; across it the position is
            # as good as the astrometry. Weight the two directions separately.
            s_pa, c_pa = np.sin(p.pa), np.cos(p.pa)
            res += [
                (e0 * s_pa + n0 * c_pa) / p.sigma_pos,  # along track
                (e0 * c_pa - n0 * s_pa) / p.sigma,  # across track
            ]
            res.append((speed / (2 * p.half) - p.rate) / (np.sqrt(2) * p.sigma / (2 * p.half)))
            res.append(_wrap_half_turn(pa_pred - p.pa) / (np.sqrt(2) * p.sigma / p.length))
        else:
            w_hat = np.array([w_e, w_n]) / max(speed, 1e-15)
            along = -(e0 * w_hat[0] + n0 * w_hat[1])  # distance along the track to the midpoint
            cross = e0 * w_hat[1] - n0 * w_hat[0]
            tau = along / max(speed, 1e-15) * 2 * p.half  # when it gets there, s from mid-exposure
            excess = max(abs(tau) - p.half, 0.0) / (2 * p.half) * speed
            length = p.length if p.length is not None else 0.01
            res += [
                cross / p.sigma,
                _wrap_half_turn(pa_pred - p.pa) / (np.sqrt(2) * p.sigma / length),
                excess / p.sigma,
            ]
    res.append(_ground_penalty(r0, v0))
    return np.array(res)


GROUND_MARGIN = 100.0  # km above the surface that a perigee must clear (drag-free orbit)


def _ground_penalty(r0: np.ndarray, v0: np.ndarray) -> float:
    """Zero for orbits that stay above the ground, growing (per 10 km) as perigee goes under it.
    Short arcs often cannot rule out orbits that dive into the Earth; this keeps the fit physical.
    It is the LAST residual and is excluded from chi-squared and the covariance."""
    rn = float(np.linalg.norm(r0))
    h = np.cross(r0, v0)
    evec = np.cross(v0, h) / MU_EARTH - r0 / rn
    e = float(np.linalg.norm(evec))
    p = float(np.dot(h, h)) / MU_EARTH
    perigee = p / (1 + e)  # holds for ellipses and hyperbolas alike
    return max(R_EARTH + GROUND_MARGIN - perigee, 0.0) / 10.0


def _seeds(prepared: list[_Prepared]) -> list[np.ndarray]:
    seeds: list[np.ndarray] = []
    first, last = prepared[0], prepared[-1]
    u1 = unit_vector(first.ra, first.dec)
    ul = unit_vector(last.ra, last.dec)
    span = last.t - first.t
    for rho in np.geomspace(500.0, 50000.0, 14):
        r1 = first.r_obs[1] + rho * u1
        if span > 1.0:
            rl = last.r_obs[1] + rho * ul
            seeds.append(np.concatenate([r1, (rl - r1) / span]))
        if first.rate is not None:
            east = np.array([-np.sin(first.ra), np.cos(first.ra), 0.0])
            north = np.array(
                [
                    -np.sin(first.dec) * np.cos(first.ra),
                    -np.sin(first.dec) * np.sin(first.ra),
                    np.cos(first.dec),
                ]
            )
            speed = rho * first.rate
            seeds.append(
                np.concatenate([r1, speed * (np.sin(first.pa) * east + np.cos(first.pa) * north)])
            )
    return seeds


def determine_orbit(
    observations: list[Observation], site: EarthLocation, *, refine_best: int = 3
) -> OrbitSolution:
    if len(observations) < 2:
        raise OrbitFitError(
            "need at least two observations: one gives only 4 constraints for 6 unknowns"
        )
    observations = sorted(observations, key=lambda o: float(cast(Any, o.time).jd))
    epoch = observations[0].time
    prepared = _prepare(observations, site, epoch)
    n_res = len(_residuals(np.array([7000.0, 0, 0, 0, 7.5, 0]), prepared)) - 1
    if n_res < 6:
        raise OrbitFitError(f"only {n_res} constraints from these observations; need at least 6")

    seeds = _seeds(prepared)
    if len(prepared) >= 3:
        try:
            mid = len(prepared) // 2
            trio = [prepared[0], prepared[mid], prepared[-1]]
            r2, v2, t2 = gauss_iod(
                [unit_vector(p.ra, p.dec) for p in trio],
                [p.t for p in trio],
                [p.r_obs[1] for p in trio],
            )
            r0, v0 = propagate(r2, v2, trio[0].t - t2)
            seeds.append(np.concatenate([r0, v0]))
        except IodError:
            pass

    scale = np.array([1000.0] * 3 + [1.0] * 3)
    trials = []
    for s in seeds:
        try:
            with np.errstate(all="ignore"):
                sol = least_squares(_residuals, s, args=(prepared,), x_scale=scale, max_nfev=15)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        if np.all(np.isfinite(sol.fun)):
            trials.append(sol)
    if not trials:
        raise OrbitFitError("no starting guess produced a finite fit")
    trials.sort(key=lambda s: s.cost)
    best = None
    for t in trials[:refine_best]:
        with np.errstate(all="ignore"):
            sol = least_squares(_residuals, t.x, args=(prepared,), x_scale=scale, max_nfev=300)
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    el_best = elements_from_state(best.x[:3], best.x[3:])
    ground_hit = el_best.perigee < R_EARTH + GROUND_MARGIN - 1.0

    jac = best.jac[:-1]  # data only: the ground penalty is a constraint, not a measurement
    data_fun = best.fun[:-1]
    warnings: list[str] = []
    try:
        cov = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = np.full((6, 6), np.inf)
        warnings.append("normal matrix is singular: the orbit is not determined")
    cond = float(np.linalg.cond(jac * scale))
    dof = max(len(data_fun) - 6, 1)
    chi2 = float(np.sum(data_fun**2) / dof)
    if cond > 1e6:
        warnings.append(
            f"poorly conditioned (cond {cond:.1e}): some orbit direction, usually range, is "
            "barely constrained by this arc; trust the covariance, not the point estimate"
        )
    if ground_hit:
        warnings.append(
            "the best fit would dip into the Earth; the orbit is pinned to the ground limit, so it "
            "is only one of many orbits consistent with these data (range is unconstrained)"
        )
    if chi2 > 4:
        warnings.append(
            f"residuals are {chi2:.1f}x larger than the stated uncertainties: bad timing, a "
            "wrong identification, or perturbations over a long arc"
        )
    return OrbitSolution(
        epoch=epoch,
        r=best.x[:3],
        v=best.x[3:],
        elements=elements_from_state(best.x[:3], best.x[3:]),
        chi2_reduced=chi2,
        n_residuals=len(data_fun),
        covariance=cov,
        condition=cond,
        warnings=tuple(warnings),
    )
