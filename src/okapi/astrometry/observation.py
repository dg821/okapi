"""A satellite observation made from one streak in one frame, ready for orbit determination."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS

from okapi.astrometry.streaks import Streak


@dataclass(frozen=True)
class Observation:
    """``kind`` is "point" when both streak ends are inside the frame, so the midpoint at
    mid-exposure is a timed position and the length gives the angular rate. It is "line" when
    the streak runs off the frame: the object crossed the field, so the line is known but not
    where along it the object was at any given time; ``time_sigma`` is then half the exposure
    and the position is only good across the track."""

    kind: str  # "point" | "line"
    time: Time  # mid-exposure
    coord: SkyCoord  # ICRS, streak midpoint
    sigma: u.Quantity  # one-axis position uncertainty
    time_sigma: u.Quantity
    rate: u.Quantity | None  # angular rate on the sky; None for a line
    position_angle: u.Quantity  # direction of the streak, east of north, modulo 180 deg
    flux: float
    length: u.Quantity | None = None  # angular length of the streak as seen


def streak_to_observation(
    streak: Streak,
    wcs: WCS,
    *,
    start: Time,
    exposure: u.Quantity,
    centroid_sigma_px: float = 0.5,
    contaminated_sigma_px: float = 4.0,
    pixel_scale: u.Quantity = 0.74 * u.arcsec,
    timing_sigma: u.Quantity = 0.05 * u.s,
) -> Observation:
    mid_time = start + TimeDelta(exposure / 2)
    xm, ym = streak.midpoint
    ra, dec = wcs.pixel_to_world_values(xm, ym)
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    # direction: offset both ends to the sky and measure the position angle between them
    a = SkyCoord(*(v * u.deg for v in wcs.pixel_to_world_values(streak.x0, streak.y0)))
    b = SkyCoord(*(v * u.deg for v in wcs.pixel_to_world_values(streak.x1, streak.y1)))
    pa = (a.position_angle(b) % (180 * u.deg)).to(u.deg)
    # a star hiding an end of the streak leaves that endpoint good to a few pixels only
    sigma = (contaminated_sigma_px if streak.contaminated else centroid_sigma_px) * pixel_scale
    if streak.touches_edge:
        return Observation(
            "line", mid_time, coord, sigma, exposure / 2, None, pa, streak.flux, a.separation(b)
        )
    rate = (a.separation(b) / exposure).to(u.arcsec / u.s)
    # both endpoints are shutter open/close (to within a PSF), so the midpoint time is good to
    # the shutter timing, not half the exposure
    return Observation(
        "point", mid_time, coord, sigma, timing_sigma, rate, pa, streak.flux, a.separation(b)
    )


def observation_to_dict(o: Observation) -> dict[str, object]:
    return {
        "kind": o.kind,
        "time_utc": o.time.utc.isot,
        "ra_deg": float(o.coord.ra.deg),  # type: ignore[attr-defined]
        "dec_deg": float(o.coord.dec.deg),  # type: ignore[attr-defined]
        "sigma_arcsec": float(o.sigma.to_value(u.arcsec)),
        "time_sigma_s": float(o.time_sigma.to_value(u.s)),
        "rate_arcsec_s": None if o.rate is None else float(o.rate.to_value(u.arcsec / u.s)),
        "position_angle_deg": float(o.position_angle.to_value(u.deg)),
        "length_arcsec": None if o.length is None else float(o.length.to_value(u.arcsec)),
        "flux": o.flux,
    }


def observation_from_dict(d: dict[str, object]) -> Observation:
    def opt(key: str, unit: u.UnitBase) -> u.Quantity | None:
        v = d[key]
        return None if v is None else float(v) * unit  # type: ignore[arg-type]

    return Observation(
        kind=str(d["kind"]),
        time=Time(str(d["time_utc"]), format="isot", scale="utc"),
        coord=SkyCoord(float(d["ra_deg"]) * u.deg, float(d["dec_deg"]) * u.deg, frame="icrs"),  # type: ignore[arg-type]
        sigma=float(d["sigma_arcsec"]) * u.arcsec,  # type: ignore[arg-type]
        time_sigma=float(d["time_sigma_s"]) * u.s,  # type: ignore[arg-type]
        rate=opt("rate_arcsec_s", u.arcsec / u.s),
        position_angle=float(d["position_angle_deg"]) * u.deg,  # type: ignore[arg-type]
        flux=float(d["flux"]),  # type: ignore[arg-type]
        length=opt("length_arcsec", u.arcsec),
    )


def write_observations(path: Path, frame_id: str, observations: list[Observation]) -> None:
    """Atomically write one frame's observations as JSON (``[]`` is a valid, meaningful file:
    the frame was processed and held no streak)."""
    payload = {"frame_id": frame_id, "observations": [observation_to_dict(o) for o in observations]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def read_observations(path: Path) -> list[Observation]:
    payload = json.loads(path.read_text())
    return [observation_from_dict(d) for d in payload["observations"]]
