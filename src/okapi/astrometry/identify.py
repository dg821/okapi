"""Match a measured streak to a known satellite by where it is and how it moves.

A timed point is compared on position, direction of motion and angular rate. A field-crossing
line has no usable along-track position (the object was somewhere on it at an unknown moment),
so only the cross-track offset and the direction are compared.

A match is a statement about geometry, not proof: a GEO belt holds many satellites a fraction of
a degree apart, and two objects can look alike in one frame. ``ambiguous`` is set when the
runner-up is nearly as good, and the caller should not trust the label then.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.astrometry.observation import Observation
from okapi.targets import Target, TargetError


@dataclass(frozen=True)
class Match:
    target: str
    offset: u.Quantity  # separation (points) or cross-track offset (lines)
    pa_error: u.Quantity
    rate_ratio: float | None  # observed / predicted rate; None for lines
    score: float  # 0 is perfect; 1 is at the edge of every gate
    ambiguous: bool


@dataclass(frozen=True)
class Gates:
    position: u.Quantity = 0.2 * u.deg
    cross_track: u.Quantity = 0.1 * u.deg
    position_angle: u.Quantity = 10 * u.deg
    rate_ratio: float = 1.25  # accepted from 1/ratio to ratio


def _half_exposure(o: Observation) -> u.Quantity:
    if o.kind == "line":
        return o.time_sigma  # for lines time_sigma is half the exposure
    assert o.rate is not None and o.length is not None
    return (o.length / o.rate) / 2


def _wrap_pa(x: u.Quantity) -> u.Quantity:
    return cast(u.Quantity, ((x + 90 * u.deg) % (180 * u.deg)) - 90 * u.deg)


def _predicted_motion(
    target: Target, site: EarthLocation, t: Time, half: u.Quantity
) -> tuple[SkyCoord, u.Quantity, u.Quantity]:
    a = target.predict(site, t - half)
    mid = target.predict(site, t)
    b = target.predict(site, t + half)
    return mid, (a.position_angle(b) % (180 * u.deg)).to(u.deg), a.separation(b) / (2 * half)


def _score(o: Observation, target: Target, site: EarthLocation, gates: Gates) -> Match | None:
    half = _half_exposure(o)
    try:
        pred, pa, rate = _predicted_motion(target, site, o.time, half)
    except TargetError:
        return None
    pa_err = abs(_wrap_pa(pa - o.position_angle)).to(u.deg)
    if pa_err > gates.position_angle:
        return None
    if o.kind == "point":
        offset = o.coord.separation(pred).to(u.deg)
        if offset > gates.position:
            return None
        assert o.rate is not None
        ratio = float((o.rate / rate).decompose())
        if not (1 / gates.rate_ratio <= ratio <= gates.rate_ratio):
            return None
        score = (
            (offset / gates.position).decompose().value
            + (pa_err / gates.position_angle).decompose().value
            + abs(math.log(ratio)) / math.log(gates.rate_ratio)
        ) / 3
        return Match(target.name, offset, pa_err, ratio, float(score), False)
    # line: the cross-track component of the predicted position relative to the line
    d_east, d_north = o.coord.spherical_offsets_to(pred)
    pa_rad = o.position_angle.to_value(u.rad)
    cross = (
        abs(
            float(d_east.to_value(u.deg)) * math.cos(pa_rad)
            - float(d_north.to_value(u.deg)) * math.sin(pa_rad)
        )
        * u.deg
    )
    if cross > gates.cross_track:
        return None
    score = (
        (cross / gates.cross_track).decompose().value
        + (pa_err / gates.position_angle).decompose().value
    ) / 2
    return Match(target.name, cross, pa_err, None, float(score), False)


def identify(
    observation: Observation,
    candidates: Sequence[Target],
    site: EarthLocation,
    gates: Gates = Gates(),
) -> Match | None:
    """The best-matching satellite among ``candidates``, or None if nothing fits every gate."""
    matches = [
        m
        for t in candidates
        if t.is_satellite and (m := _score(observation, t, site, gates)) is not None
    ]
    if not matches:
        return None
    matches.sort(key=lambda m: m.score)
    best = matches[0]
    ambiguous = len(matches) > 1 and matches[1].score < 2 * max(best.score, 0.05)
    return Match(best.target, best.offset, best.pa_error, best.rate_ratio, best.score, ambiguous)
