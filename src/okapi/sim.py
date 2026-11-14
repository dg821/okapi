"""A complete simulated rig (clock, site, mount, camera, solver, GPS) for tests and dry runs."""

from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from okapi.camera.base import ExposureSettings
from okapi.camera.fake import FakeCamera
from okapi.clock import SimClock
from okapi.coords import site_from_geodetic
from okapi.mount.fake import FakeMount
from okapi.sensors.fake import FakeGps
from okapi.solve.fake import FakeSolver


@dataclass
class SimRig:
    clock: SimClock
    site: EarthLocation
    mount: FakeMount
    camera: FakeCamera
    solver: FakeSolver
    gps: FakeGps
    settings: ExposureSettings


def make_sim_rig(
    *,
    start: str = "2026-06-21T04:30:00",
    frame_rotation: tuple[u.Quantity, u.Quantity, u.Quantity] = (
        0.5 * u.deg,
        -0.8 * u.deg,
        0.9 * u.deg,
    ),
    alignment_error: tuple[u.Quantity, u.Quantity] = (0 * u.deg, 0 * u.deg),
    solver_noise: u.Quantity = 2 * u.arcsec,
    solver_fail_first: int = 0,
    gps_always_fail: bool = False,
    gps_clock_error: u.Quantity = 0 * u.s,
    gps_fail_first: int = 0,
    seed: int = 0,
) -> SimRig:
    clock = SimClock(Time(start, scale="utc"))
    site = site_from_geodetic(40.0, -105.0, 1600.0)
    mount = FakeMount(clock, alignment_error=alignment_error, frame_rotation=frame_rotation)
    return SimRig(
        clock=clock,
        site=site,
        mount=mount,
        camera=FakeCamera(clock),
        solver=FakeSolver(
            mount.true_radec, noise=solver_noise, fail_first=solver_fail_first, seed=seed
        ),
        gps=FakeGps(
            clock,
            site,
            clock_error=gps_clock_error,
            always_fail=gps_always_fail,
            fail_first=gps_fail_first,
        ),
        settings=ExposureSettings(2 * u.s, 1600),
    )
