"""Command-line entry point."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
import typer
from astropy.time import Time

from okapi import __version__
from okapi import doctor as doc
from okapi.astrometry.observation import Observation, read_observations
from okapi.astrometry.timing import TimingError, estimate_time_offset
from okapi.camera.gphoto import run_gphoto2
from okapi.clock import SystemClock
from okapi.coords import site_from_geodetic
from okapi.hardware import make_hardware_rig
from okapi.mount.base import MountError
from okapi.mount.synscan import SynScanMount
from okapi.mount.transport import SerialTransport
from okapi.orbits.fit import OrbitFitError, determine_orbit
from okapi.orbits.simulate import geo_state
from okapi.sensors.nmea import NmeaGps
from okapi.session.calibrate import CalibrationError
from okapi.session.calibrate import calibrate as run_calibration
from okapi.session.night import format_report, run_night
from okapi.session.plan import NightPlan, PlanError, load_plan
from okapi.sim import make_sim_rig, make_synscan_rig
from okapi.sim_night import decode_npy, make_night_rig
from okapi.targets import GeoSlotTarget, StateTarget, Target, TleTarget, parse_tles

app = typer.Typer(
    help="Control the Orion XT10g telescope and Canon T3i camera.", no_args_is_help=True
)


@app.callback()
def main() -> None:
    """okapi command group (subcommands below)."""


@app.command()
def version() -> None:
    """Print the okapi version."""
    typer.echo(__version__)


@app.command()
def calibrate(
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Run against a simulated telescope, camera, solver and GPS."
    ),
    mount_port: str = typer.Option("", help="Hand control serial device, e.g. /dev/ttyUSB0"),
    gps_port: str = typer.Option("", help="GPS NMEA serial device, e.g. /dev/ttyACM0"),
    exposure: float = typer.Option(2.0, help="Alignment exposure, seconds"),
    iso: int = typer.Option(1600, help="Alignment ISO"),
    astap_path: str = typer.Option("astap_cli", help="ASTAP command-line executable"),
    fallback_lat: float | None = typer.Option(None, help="Site latitude if GPS has no fix, deg"),
    fallback_lon: float | None = typer.Option(None, help="Site longitude (east +), deg"),
    fallback_height: float = typer.Option(0.0, help="Site height, metres"),
) -> None:
    """Run the start-of-night calibration: GPS site/time, two-star alignment, verification."""
    if not dry_run and not (mount_port and gps_port):
        typer.echo("give --mount-port and --gps-port, or use --dry-run", err=True)
        raise typer.Exit(code=2)
    if (fallback_lat is None) != (fallback_lon is None):
        typer.echo("give both --fallback-lat and --fallback-lon", err=True)
        raise typer.Exit(code=2)
    fallback = (
        site_from_geodetic(fallback_lat, fallback_lon, fallback_height)
        if fallback_lat is not None and fallback_lon is not None
        else None
    )
    if not dry_run:
        _calibrate_hardware(mount_port, gps_port, exposure * u.s, iso, astap_path, fallback)
        return
    rig = make_sim_rig()
    try:
        rep = run_calibration(
            mount=rig.mount,
            camera=rig.camera,
            solver=rig.solver,
            gps=rig.gps,
            clock=rig.clock,
            settings=rig.settings,
        )
    except CalibrationError as e:
        typer.echo(f"calibration failed: {e}", err=True)
        raise typer.Exit(code=1) from e
    _report(rep)


@app.command("mount-info")
def mount_info(
    port: str = typer.Argument("", help="Serial device of the hand control, e.g. /dev/ttyUSB0"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Talk to a simulated hand control."),
) -> None:
    """Read-only first contact with the hand control: version, model, position, location, time.

    Sends only queries (never a goto, rate or setting), so it is safe to run on the real mount."""
    if not dry_run and not port:
        typer.echo("give the serial port (e.g. /dev/ttyUSB0), or use --dry-run", err=True)
        raise typer.Exit(code=2)
    mount: SynScanMount | None = None
    try:
        if dry_run:
            mount = make_synscan_rig().mount
        else:
            mount = SynScanMount(SerialTransport(port), SystemClock())
        report = mount.read_report()
    except (MountError, OSError) as e:  # OSError covers a missing or busy serial port
        typer.echo(f"could not talk to the hand control: {e}", err=True)
        raise typer.Exit(code=1) from e
    finally:
        if mount is not None:
            mount._t.close()
    info = report.info
    typer.echo(f"firmware: {info.version}")
    typer.echo(f"model: {info.model} ({info.model_name})")
    typer.echo(f"aligned: {'yes' if info.aligned else 'no'}")
    typer.echo(
        f"az/alt: {report.altaz.az.to_value(u.deg):.4f}, {report.altaz.alt.to_value(u.deg):.4f} deg"
    )
    typer.echo(f"handset tracking mode: {report.tracking_mode} (0 = off)")
    typer.echo(f"handset location: {report.location[0]:.4f}, {report.location[1]:.4f} deg")
    if report.time_utc is None:
        typer.echo("handset time (UTC): not set (the handset clock holds an invalid date)")
    else:
        typer.echo(f"handset time (UTC): {report.time_utc.isot}")
    for w in mount.warnings:
        typer.echo(f"warning: {w}")


def _report(rep: Any) -> None:
    typer.echo(f"site: {rep.site.lat:.4f}, {rep.site.lon:.4f}")
    typer.echo(f"aligned on: {' + '.join(rep.stars)}")
    typer.echo(f"pointing model rms: {rep.model_rms.to_value(u.arcsec):.1f} arcsec")
    if rep.verification_error is not None:
        typer.echo(
            f"verification on {rep.verification_star}: "
            f"{rep.verification_error.to_value(u.arcsec):.0f} arcsec"
        )
    for w in rep.warnings:
        typer.echo(f"warning: {w}")


def _calibrate_hardware(
    mount_port: str,
    gps_port: str,
    exposure: Any,
    iso: int,
    astap_path: str,
    fallback: Any,
) -> None:
    try:
        rig = make_hardware_rig(
            mount_port=mount_port,
            gps_port=gps_port,
            exposure=exposure,
            iso=iso,
            astap_path=astap_path,
        )
    except (MountError, OSError) as e:
        typer.echo(f"could not open the hardware: {e}", err=True)
        raise typer.Exit(code=1) from e
    try:
        rep = run_calibration(
            mount=rig.mount,
            camera=rig.camera,
            solver=rig.solver,
            gps=rig.gps,
            clock=rig.clock,
            settings=rig.settings,
            fallback_site=fallback,
        )
        _report(rep)
    except CalibrationError as e:
        typer.echo(f"calibration failed: {e}", err=True)
        raise typer.Exit(code=1) from e
    finally:
        rig.close()  # stops the axes and closes both serial ports


def _read_plan(plan_path: Path) -> NightPlan:
    try:
        return load_plan(plan_path)
    except PlanError as e:
        typer.echo(f"plan error: {e}", err=True)
        raise typer.Exit(code=2) from e


def _sim_truth(plan: NightPlan) -> list[Target]:
    """Satellites for a dry run: each geo item gets a real object 0.02 deg of longitude from its
    nominal slot (so aiming and the pointing trim have something to do); a TLE item is its own
    truth."""
    truth: list[Target] = []
    for it in plan.items:
        if isinstance(it.target, GeoSlotTarget):
            lon = float(it.target.longitude.to_value(u.deg)) + 0.02
            when = Time("2026-06-21T04:30:00", scale="utc")
            r, v = geo_state(lon, when)
            truth.append(StateTarget(f"sim-{it.name}", r, v, when))
        elif isinstance(it.target, TleTarget):
            truth.append(it.target)
    return truth


@dataclass
class _NightSetup:
    camera: Any
    solver: Any
    decode: Callable[[Any], np.ndarray]
    clock: Any
    calibrate: Callable[[], Any]
    close: Callable[[], None]


def _simulated_setup(plan: NightPlan) -> _NightSetup:
    rig = make_night_rig(_sim_truth(plan))
    return _NightSetup(
        rig.camera,
        rig.solver,
        decode_npy,
        rig.clock,
        lambda: run_calibration(
            mount=rig.mount,
            camera=rig.camera,
            solver=rig.solver,
            gps=rig.gps,
            clock=rig.clock,
            settings=rig.settings,
        ),
        lambda: None,
    )


def _hardware_setup(mount_port: str, gps_port: str, astap_path: str, fallback: Any) -> _NightSetup:
    hw = make_hardware_rig(mount_port=mount_port, gps_port=gps_port, astap_path=astap_path)
    return _NightSetup(
        hw.camera,
        hw.solver,
        hw.solver.decoder,
        hw.clock,
        lambda: run_calibration(
            mount=hw.mount,
            camera=hw.camera,
            solver=hw.solver,
            gps=hw.gps,
            clock=hw.clock,
            settings=hw.settings,
            fallback_site=fallback,
        ),
        hw.close,
    )


@app.command()
def night(
    plan_file: Path = typer.Argument(..., help="Night plan (JSON)"),
    out: Path = typer.Option("data/night", help="Output directory (resumes if it exists)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Run the whole night in simulation."),
    mount_port: str = typer.Option("", help="Hand control serial device"),
    gps_port: str = typer.Option("", help="GPS NMEA serial device"),
    astap_path: str = typer.Option("astap_cli", help="ASTAP command-line executable"),
    fallback_lat: float | None = typer.Option(None, help="Site latitude if GPS has no fix"),
    fallback_lon: float | None = typer.Option(None, help="Site longitude (east +)"),
    fallback_height: float = typer.Option(0.0, help="Site height, metres"),
) -> None:
    """Run a night: calibrate, then stare at each target in the plan and fit orbits."""
    plan = _read_plan(plan_file)
    if not dry_run and not (mount_port and gps_port):
        typer.echo("give --mount-port and --gps-port, or use --dry-run", err=True)
        raise typer.Exit(code=2)
    if (fallback_lat is None) != (fallback_lon is None):
        typer.echo("give both --fallback-lat and --fallback-lon", err=True)
        raise typer.Exit(code=2)
    fallback = (
        site_from_geodetic(fallback_lat, fallback_lon, fallback_height)
        if fallback_lat is not None and fallback_lon is not None
        else None
    )
    try:
        setup = (
            _simulated_setup(plan)
            if dry_run
            else _hardware_setup(mount_port, gps_port, astap_path, fallback)
        )
    except (MountError, OSError) as e:
        typer.echo(f"could not open the hardware: {e}", err=True)
        raise typer.Exit(code=1) from e
    try:
        try:
            cal = setup.calibrate()
        except CalibrationError as e:
            typer.echo(f"calibration failed: {e}", err=True)
            raise typer.Exit(code=1) from e
        _report(cal)
        report = run_night(
            plan=plan,
            mount=cal.mount,
            camera=setup.camera,
            solver=setup.solver,
            decode=setup.decode,
            clock=setup.clock,
            site=cal.site,
            out_dir=out,
            recalibrate=lambda: setup.calibrate().mount,
        )
    finally:
        setup.close()
    typer.echo(format_report(report))
    if report.aborted:
        raise typer.Exit(code=1)


@app.command()
def orbit(
    files: list[Path] = typer.Argument(..., help="Observation files (*.obs.json)"),
    lat: float = typer.Option(..., help="Observer latitude, deg"),
    lon: float = typer.Option(..., help="Observer longitude, deg (east +)"),
    height: float = typer.Option(0.0, help="Observer height, metres"),
) -> None:
    """Fit a two-body orbit to observation files from `okapi night` (assumes one object)."""
    observations: list[Observation] = []
    for f in files:
        try:
            observations += read_observations(f)
        except (OSError, ValueError, KeyError) as e:
            typer.echo(f"cannot read {f}: {e}", err=True)
            raise typer.Exit(code=2) from e
    try:
        sol = determine_orbit(observations, site_from_geodetic(lat, lon, height))
    except OrbitFitError as e:
        typer.echo(f"no orbit: {e}", err=True)
        raise typer.Exit(code=1) from e
    el = sol.elements
    typer.echo(f"observations: {len(observations)}  epoch: {sol.epoch.utc.isot}")
    typer.echo(f"a = {el.a:.1f} km   e = {el.e:.5f}   i = {np.degrees(el.i):.3f} deg")
    typer.echo(
        f"raan = {np.degrees(el.raan):.3f}   argp = {np.degrees(el.argp):.3f}   "
        f"nu = {np.degrees(el.nu):.3f} deg"
    )
    typer.echo(f"position sigma (km): {', '.join(f'{x:.1f}' for x in sol.r_sigma)}")
    typer.echo(f"chi2 reduced: {sol.chi2_reduced:.2f}")
    for w in sol.warnings:
        typer.echo(f"warning: {w}")


@app.command()
def doctor(
    mount_port: str = typer.Option("", help="Handset serial port (skipped if empty)"),
    gps_port: str = typer.Option("", help="GPS serial port (skipped if empty)"),
    astap_path: str = typer.Option("astap_cli", help="ASTAP command-line executable"),
    data_dir: Path = typer.Option(Path("data"), help="Where frames will be stored"),
    skip_camera: bool = typer.Option(False, help="Do not look for the camera"),
) -> None:
    """Preflight: check each part of the rig and say what to fix. Moves nothing, takes no frames."""
    import serial

    checks = [doc.check_architecture()]
    if not skip_camera:
        checks.append(doc.check_camera(run_gphoto2))
    if gps_port:
        try:
            port = serial.serial_for_url(gps_port, baudrate=9600, timeout=2.0)
        except (OSError, serial.SerialException) as e:
            checks.append(doc.Check("gps", doc.FAIL, str(e), "check the port name and permissions"))
        else:
            gps = NmeaGps(lambda: port.readline().decode("ascii", errors="ignore"))
            checks += doc.check_gps(gps, SystemClock())
            port.close()
    if mount_port:

        def read() -> Any:
            m = SynScanMount(SerialTransport(mount_port), SystemClock())
            try:
                return m.read_report()
            finally:
                m._t.close()

        checks.append(doc.check_mount(read))
    checks.append(doc.check_astap(astap_path))
    data_dir.mkdir(parents=True, exist_ok=True)
    checks.append(doc.check_disk(data_dir))
    typer.echo(doc.format_checks(checks))
    if any(c.status == doc.FAIL for c in checks):
        raise typer.Exit(code=1)


@app.command()
def timing(
    files: list[Path] = typer.Argument(..., help="*.obs.json files of one known satellite"),
    tle: Path = typer.Option(..., help="TLE file (fresh: LEO TLEs go stale in days)"),
    name: str = typer.Option("", help="Satellite name in the TLE file (default: the only one)"),
    lat: float = typer.Option(..., help="Observer latitude, deg"),
    lon: float = typer.Option(..., help="Observer longitude, deg (east +)"),
    height: float = typer.Option(0.0, help="Observer height, metres"),
) -> None:
    """Estimate the shutter delay from streaks of a satellite with a known orbit (e.g. the ISS)."""
    try:
        targets = parse_tles(tle.read_text())
        observations: list[Observation] = []
        for f in files:
            observations += read_observations(f)
    except (OSError, ValueError, KeyError) as e:
        typer.echo(f"cannot read input: {e}", err=True)
        raise typer.Exit(code=2) from e
    chosen = [t for t in targets if not name or name.lower() in t.name.lower()]
    if len(chosen) != 1:
        typer.echo(f"need exactly one satellite in the TLE file, matched {len(chosen)}", err=True)
        raise typer.Exit(code=2)
    try:
        res = estimate_time_offset(observations, chosen[0], site_from_geodetic(lat, lon, height))
    except TimingError as e:
        typer.echo(f"no estimate: {e}", err=True)
        raise typer.Exit(code=1) from e
    typer.echo(f"frames used: {res.n}")
    typer.echo(
        f"shutter offset: {res.offset.to_value(u.s):+.3f} s "
        f"(scatter {res.scatter.to_value(u.s):.3f} s)"
    )
    typer.echo("positive = the shutter opened later than the recorded start time")
