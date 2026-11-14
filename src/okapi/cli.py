"""Command-line entry point."""

import astropy.units as u
import typer

from okapi import __version__
from okapi.session.calibrate import CalibrationError
from okapi.session.calibrate import calibrate as run_calibration
from okapi.sim import make_sim_rig

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
