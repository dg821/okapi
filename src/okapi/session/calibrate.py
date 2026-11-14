"""Start-of-night calibration: site and time, two-star alignment, verification.

1. Read position and UTC from GPS (or use an explicit fallback site) and send both to the mount.
2. Center the brightest visible star, then a well-separated partner, by plate solving. If a star
   cannot be centered (cloud, below a rooftop, bad solve), move on to the next candidate.
3. Fit the pointing model from the two (true, reported) pairs and wrap the mount with it.
4. Verify: slew to a third star with the model and measure, with one solve, how far off we land.
"""

from dataclasses import dataclass, field

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.align.aligned_mount import AlignedMount
from okapi.align.center import (
    DEFAULT_TOLERANCE,
    CenteringError,
    CenterResult,
    center_on,
)
from okapi.camera.base import CameraProtocol, ExposureSettings
from okapi.catalog import Star, partner_candidates, visible_stars
from okapi.clock import Clock
from okapi.coords import AltAzPosition, radec_to_altaz
from okapi.mount.base import MountError, MountProtocol
from okapi.pointing import PointingModel
from okapi.sensors.base import GpsError, GpsProtocol
from okapi.solve.base import SolveError, SolverProtocol

VERIFY_WARN = 120 * u.arcsec  # a third of the 1 x 0.7 deg frame is ~12 arcmin; this is tight
CLOCK_WARN = 2 * u.s


class CalibrationError(RuntimeError):
    """Calibration could not complete; the message says which step failed."""


@dataclass(frozen=True)
class CalibrationReport:
    site: EarthLocation
    gps_time: Time | None
    clock_error: u.Quantity | None  # system clock minus GPS time
    used_fallback_site: bool
    stars: tuple[str, ...]
    centering: tuple[CenterResult, ...]
    model: PointingModel
    model_rms: u.Quantity
    verification_star: str | None
    verification_error: u.Quantity | None
    mount: AlignedMount
    warnings: tuple[str, ...] = field(default=())


def calibrate(
    *,
    mount: MountProtocol,
    camera: CameraProtocol,
    solver: SolverProtocol,
    gps: GpsProtocol,
    clock: Clock,
    settings: ExposureSettings,
    fallback_site: EarthLocation | None = None,
    verify: bool = True,
    tolerance: u.Quantity = DEFAULT_TOLERANCE,
    max_star_attempts: int = 4,
) -> CalibrationReport:
    warnings: list[str] = []

    # -- 1. site and time ------------------------------------------------------------------
    gps_time: Time | None = None
    clock_error: u.Quantity | None = None
    used_fallback = False
    try:
        fix = gps.read_fix()
        site, gps_time = fix.site, fix.time
        clock_error = float((clock.now() - gps_time).to_value(u.s)) * u.s
        if abs(clock_error) > CLOCK_WARN:
            warnings.append(
                f"system clock is {clock_error:+.1f} from GPS time; the mount gets GPS time, "
                "but fix the Pi's clock (chrony/gpsd) before long sessions"
            )
        mount_time = gps_time
    except GpsError as e:
        if fallback_site is None:
            raise CalibrationError(f"no GPS fix and no fallback site: {e}") from e
        site, used_fallback, mount_time = fallback_site, True, clock.now()
        warnings.append(
            f"no GPS fix ({e}); using the fallback site and the system clock, "
            "which is only as good as the Pi's last time sync"
        )
    try:
        mount.set_site(site)
        mount.set_time(mount_time)
    except MountError as e:
        raise CalibrationError(f"mount rejected site/time: {e}") from e

    # -- 2. two-star alignment, falling back to other stars on failure -----------------------
    failed: list[str] = []
    attempts = 0

    def acquire(candidates: list[Star]) -> tuple[Star, CenterResult, AltAzPosition, AltAzPosition]:
        nonlocal attempts
        for star in candidates:
            if attempts >= max_star_attempts:
                break
            attempts += 1
            now = clock.now()
            try:
                result = center_on(
                    star.coord(now),
                    mount=mount,
                    camera=camera,
                    solver=solver,
                    site=site,
                    settings=settings,
                    tolerance=tolerance,
                )
            except (CenteringError, SolveError, MountError) as e:
                failed.append(star.name)
                warnings.append(f"could not center {star.name}: {e}")
                continue
            t = clock.now()
            true = radec_to_altaz(star.coord(t), site, t)
            return star, result, true, mount.get_altaz()
        raise CalibrationError(
            f"could not center a star after {attempts} attempts (tried: {', '.join(failed)})"
        )

    first_star, first_res, true1, rep1 = acquire(visible_stars(site, clock.now()))
    partners = partner_candidates(first_star, site, clock.now(), exclude=tuple(failed))
    second_star, second_res, true2, rep2 = acquire(partners)

    # -- 3. fit ------------------------------------------------------------------------------
    pairs = [(true1, rep1), (true2, rep2)]
    model = PointingModel.fit(pairs)
    rms = model.rms(pairs)
    aligned = AlignedMount(mount, model, site=site, clock=clock)

    # -- 4. verify ---------------------------------------------------------------------------
    verify_star: Star | None = None
    verify_error: u.Quantity | None = None
    if verify:
        used = (first_star.name, second_star.name, *failed)
        others = visible_stars(site, clock.now(), exclude=used)
        if not others:
            warnings.append("no third star available for verification")
        else:
            verify_star = others[0]
            target: SkyCoord = verify_star.coord(clock.now())
            try:
                # Slew twice: the first lands near the target (our slew-time guess is rough),
                # the second is a short hop whose lead error is negligible.
                for _ in range(2):
                    aligned.slew_radec(target)
                    aligned.wait_until_settled(180 * u.s)
                aligned.start_tracking()
                solution = solver.solve(camera.capture(settings))
                err = solution.center.separation(target).to(u.arcsec)
                verify_error = err
                if err > VERIFY_WARN:
                    warnings.append(
                        f"verification on {verify_star.name} missed by {err:.0f}; "
                        "the pointing model is poor (unlevel base? altitude zero offset?)"
                    )
            except (SolveError, MountError) as e:
                warnings.append(f"verification on {verify_star.name} failed: {e}")

    return CalibrationReport(
        site=site,
        gps_time=gps_time,
        clock_error=clock_error,
        used_fallback_site=used_fallback,
        stars=(first_star.name, second_star.name),
        centering=(first_res, second_res),
        model=model,
        model_rms=rms,
        verification_star=verify_star.name if verify_star else None,
        verification_error=verify_error,
        mount=aligned,
        warnings=tuple(warnings),
    )
