# okapi

Raspberry Pi control of an Orion SkyQuest XT10g GoTo Dobsonian and a Canon EOS Rebel T3i:
slew, track, expose, and automate nightly alignment. See PROJECT_PLAN.md.

    uv sync --extra raw
    uv run pytest -m "not hardware"

First contact with the hand control (read-only, sends only queries):

    uv run okapi mount-info /dev/ttyUSB0      # or --dry-run to try it against a simulator

Real calibration (with the cable and GPS attached):

    uv run okapi calibrate --mount-port /dev/ttyUSB0 --gps-port /dev/ttyACM0 \
        --fallback-lat 40.0 --fallback-lon -105.3

A night, end to end (simulated here; drop `--dry-run` and add the ports on the real rig):

    uv run okapi night plan.json --dry-run --out data/demo
    uv run okapi orbit data/demo/raw/*/*/*/*.obs.json --lat 40 --lon -105

See PROJECT_PLAN.md for the plan-file format and what is still unverified on hardware.
