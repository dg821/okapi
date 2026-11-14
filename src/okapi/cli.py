"""Command-line entry point."""

import typer

from okapi import __version__

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
