"""Command-line entry point."""

import astropy.units as u
import typer

from okapi import __version__
from okapi.clock import SystemClock
from okapi.mount.base import MountError
from okapi.mount.synscan import SynScanMount
from okapi.mount.transport import SerialTransport
from okapi.session.calibrate import CalibrationError
from okapi.session.calibrate import calibrate as run_calibration
from okapi.sim import make_sim_rig, make_synscan_rig

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
) -> None:
    """Run the start-of-night calibration: GPS site/time, two-star alignment, verification."""
    if not dry_run:
        typer.echo(
            "Hardware drivers (mount serial, GPS, plate solver) are not implemented yet; "
            "use --dry-run.",
            err=True,
        )
        raise typer.Exit(code=2)
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
    typer.echo(f"site: {rep.site.lat:.4f}, {rep.site.lon:.4f}")
    typer.echo(f"aligned on: {' + '.join(rep.stars)}")
    typer.echo(f"pointing model rms: {rep.model_rms.to_value(u.arcsec):.1f} arcsec")
    if rep.verification_error is not None:
        typer.echo(
            f"verification on {rep.verification_star}: "
            f"{rep.verification_error.to_value(u.arcsec):.1f} arcsec"
        )
    for w in rep.warnings:
        typer.echo(f"warning: {w}")


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
