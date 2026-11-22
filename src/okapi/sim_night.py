"""A simulated night: a camera that renders star fields and satellite streaks from where the
simulated mount truly points, and a solver that returns the exact pixel-to-sky mapping of what it
rendered. Lets the whole night (calibration, aiming, stares, solving, streak extraction, orbit
fitting) run on a simulated clock with no hardware."""

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.wcs import WCS

from okapi.astrometry.simulate import SimStar, SimStreak, render_frame, tangent_wcs
from okapi.camera.base import CameraError, CapturedImage, ExposureSettings
from okapi.clock import SimClock
from okapi.coords import site_from_geodetic
from okapi.mount.fake import FakeMount
from okapi.mount.synscan import SynScanMount
from okapi.mount.synscan_sim import FakeHandset
from okapi.sensors.fake import FakeGps
from okapi.solve.base import SolveError, SolveResult
from okapi.targets import Target


def decode_npy(image: CapturedImage) -> np.ndarray:
    return np.load(io.BytesIO(image.data))


def _clip_segment(
    p0: tuple[float, float], p1: tuple[float, float], width: int, height: int
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Liang-Barsky: the part of a segment inside the frame, or None."""
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in (
        (-dx, x0),
        (dx, (width - 1) - x0),
        (-dy, y0),
        (dy, (height - 1) - y0),
    ):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    a, b = (x0 + t0 * dx, y0 + t0 * dy), (x0 + t1 * dx, y0 + t1 * dy)
    if np.hypot(b[0] - a[0], b[1] - a[1]) < 1.0:  # under a pixel: nothing to render
        return None
    return a, b


class SkyCamera:
    """Renders what a sidereally tracked camera would record. Satellites are ``Target`` objects
    whose motion is the truth; whatever the mount truly points at decides the field."""

    def __init__(
        self,
        clock: SimClock,
        site: EarthLocation,
        truth_pointing: Callable[[], SkyCoord],
        satellites: Sequence[Target] = (),
        *,
        shape: tuple[int, int] = (700, 1000),
        pixel_scale_arcsec: float = 1.48,
        seed: int = 0,
        fail_on: Callable[[int], bool] | None = None,
    ) -> None:
        self._clock = clock
        self._site = site
        self._truth = truth_pointing
        self.satellites = list(satellites)
        self.shape = shape
        self._scale = pixel_scale_arcsec
        self._seed = seed
        self._fail_on = fail_on
        self.calls = 0
        self.last_wcs: WCS | None = None
        self.last_start: Time | None = None

    def capture(self, settings: ExposureSettings) -> CapturedImage:
        self.calls += 1
        if self._fail_on is not None and self._fail_on(self.calls):
            raise CameraError("simulated camera failure")
        start = self._clock.now()
        e = settings.exposure_s
        self._clock.sleep(e / 2)
        centre = self._truth()
        self._clock.sleep(e / 2)
        ra, dec = float(centre.ra.deg), float(centre.dec.deg)  # type: ignore[attr-defined]
        wcs = tangent_wcs(ra, dec, self.shape, self._scale)
        rng = np.random.default_rng((self._seed, self.calls))
        h, w = self.shape
        stars = [
            SimStar(rng.uniform(5, w - 5), rng.uniform(5, h - 5), rng.uniform(150, 3000))
            for _ in range(40)
        ]
        streaks: list[SimStreak] = []
        for sat in self.satellites:
            ends = []
            for t in (start, start + e * u.s):
                p = sat.predict(self._site, t)
                x, y = wcs.world_to_pixel_values(float(p.ra.deg), float(p.dec.deg))  # type: ignore[attr-defined]
                ends.append((float(x), float(y)))
            seg = _clip_segment(ends[0], ends[1], w, h)
            if seg is not None:
                streaks.append(SimStreak(*seg[0], *seg[1], peak=120.0))
        data = render_frame(self.shape, stars=stars, streaks=streaks, seed=self.calls)
        buf = io.BytesIO()
        np.save(buf, data.astype(np.float32))
        self.last_wcs, self.last_start = wcs, start
        return CapturedImage(buf.getvalue(), ".npy", start, settings)


class WcsSolver:
    """Returns the exact mapping of the last rendered frame, plus optional centre noise."""

    def __init__(
        self,
        camera: SkyCamera,
        *,
        noise: u.Quantity = 0 * u.arcsec,
        fail_on: Callable[[int], bool] | None = None,
        seed: int = 0,
    ) -> None:
        self._camera = camera
        self._noise = float(noise.to_value(u.arcsec))
        self._fail_on = fail_on
        self._rng = np.random.default_rng(seed)
        self.calls = 0

    def solve(self, image: CapturedImage) -> SolveResult:
        self.calls += 1
        if self._fail_on is not None and self._fail_on(self.calls):
            raise SolveError("simulated solve failure (cloud)")
        wcs = self._camera.last_wcs
        if wcs is None or self._camera.last_start != image.start_time:
            raise SolveError("solver was given a frame the simulated camera did not just take")
        wcs = wcs.deepcopy()
        if self._noise > 0:
            d_ra, d_dec = self._rng.normal(0.0, self._noise / 3600.0, 2)
            wcs.wcs.crval = [wcs.wcs.crval[0] + d_ra, wcs.wcs.crval[1] + d_dec]
        h, w = self._camera.shape
        centre = wcs.pixel_to_world((w - 1) / 2, (h - 1) / 2).icrs
        return SolveResult(center=centre, n_stars=40, wcs=wcs)


@dataclass
class NightRig:
    clock: SimClock
    site: EarthLocation
    physical: FakeMount
    handset: FakeHandset
    mount: SynScanMount
    camera: SkyCamera
    solver: WcsSolver
    gps: FakeGps
    settings: ExposureSettings
    satellites: list[Target] = field(default_factory=list)


def make_night_rig(
    satellites: Sequence[Target] = (),
    *,
    start: str = "2026-06-21T04:30:00",
    frame_rotation: tuple[u.Quantity, u.Quantity, u.Quantity] = (
        0.5 * u.deg,
        -0.8 * u.deg,
        0.9 * u.deg,
    ),
    alignment_error: tuple[u.Quantity, u.Quantity] = (0 * u.deg, 0 * u.deg),
    solver_noise: u.Quantity = 1 * u.arcsec,
    camera_fail_on: Callable[[int], bool] | None = None,
    solver_fail_on: Callable[[int], bool] | None = None,
    shape: tuple[int, int] = (700, 1000),
    pixel_scale_arcsec: float = 1.48,
    seed: int = 0,
) -> NightRig:
    clock = SimClock(Time(start, scale="utc"))
    site = site_from_geodetic(40.0, -105.0, 1600.0)
    physical = FakeMount(clock, alignment_error=alignment_error, frame_rotation=frame_rotation)
    physical.set_site(site)  # the real telescope stands at the true site whatever the handset knows
    handset = FakeHandset(physical, clock)
    camera = SkyCamera(
        clock,
        site,
        physical.true_radec,
        satellites,
        shape=shape,
        pixel_scale_arcsec=pixel_scale_arcsec,
        seed=seed,
        fail_on=camera_fail_on,
    )
    return NightRig(
        clock=clock,
        site=site,
        physical=physical,
        handset=handset,
        mount=SynScanMount(handset, clock),
        camera=camera,
        solver=WcsSolver(camera, noise=solver_noise, fail_on=solver_fail_on, seed=seed),
        gps=FakeGps(clock, site),
        settings=ExposureSettings(2 * u.s, 1600),
        satellites=list(satellites),
    )
