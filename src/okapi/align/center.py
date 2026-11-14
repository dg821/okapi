"""Center a target in the camera frame: slew, expose, solve, nudge, repeat.

The solver tells us where the telescope truly points. The difference between that and the
target, expressed as an Az/Alt move, is exactly the nudge the mount needs, whatever its own
alignment error is. Once centered, the mount can be synced on the target so that handset
alignment is corrected as well.
"""

from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord

from okapi.camera.base import CameraProtocol, ExposureSettings
from okapi.coords import radec_to_altaz, wrap180
from okapi.mount.base import MountProtocol
from okapi.solve.base import SolveError, SolveResult, SolverProtocol

DEFAULT_TOLERANCE = 15 * u.arcsec  # about 20 T3i pixels at 1200 mm
DEFAULT_SETTLE_TIMEOUT = 180 * u.s


class CenteringError(RuntimeError):
    """Could not center the target within the iteration budget."""


@dataclass(frozen=True)
class CenterResult:
    iterations: int
    error: u.Quantity  # final sky separation between target and solved center
    solution: SolveResult


def center_on(
    target: SkyCoord,
    *,
    mount: MountProtocol,
    camera: CameraProtocol,
    solver: SolverProtocol,
    site: EarthLocation,
    settings: ExposureSettings,
    tolerance: u.Quantity = DEFAULT_TOLERANCE,
    max_iterations: int = 6,
    max_solve_failures: int = 3,
) -> CenterResult:
    """Slew to ``target`` and iterate until the solved pointing is within ``tolerance``."""
    if max_iterations < 1 or max_solve_failures < 1:
        raise ValueError("max_iterations and max_solve_failures must be at least 1")
    mount.slew_radec(target)
    mount.wait_until_settled(DEFAULT_SETTLE_TIMEOUT)
    mount.start_tracking()

    solve_failures = 0
    iteration = 0
    error = float("nan") * u.arcsec
    while iteration < max_iterations:
        image = camera.capture(settings)
        try:
            solution = solver.solve(image)
        except SolveError as e:
            solve_failures += 1
            if solve_failures >= max_solve_failures:
                raise CenteringError(f"{solve_failures} solve failures in a row: {e}") from e
            continue  # retry in place; a failed solve is not a centering iteration
        solve_failures = 0
        iteration += 1

        error = solution.center.separation(target).to(u.arcsec)
        if error <= tolerance:
            return CenterResult(iteration, error, solution)

        t_mid = image.start_time + image.settings.exposure / 2
        want = radec_to_altaz(target, site, t_mid)
        have = radec_to_altaz(solution.center, site, t_mid)
        mount.nudge(wrap180(want.az - have.az), want.alt - have.alt)
        mount.wait_until_settled(DEFAULT_SETTLE_TIMEOUT)

    raise CenteringError(
        f"not within {tolerance} of the target after {max_iterations} iterations "
        f"(last error {error:.1f})"
    )


def align_on_star(target: SkyCoord, **kwargs) -> CenterResult:
    """Center on a star, then tell the mount that is where it is pointing."""
    result = center_on(target, **kwargs)
    kwargs["mount"].sync_radec(target)
    return result
