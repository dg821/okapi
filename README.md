# okapi

Raspberry Pi control of an Orion SkyQuest XT10g GoTo Dobsonian and a Canon EOS Rebel T3i:
slew, track, expose, and automate nightly alignment. See PROJECT_PLAN.md.

    uv sync --extra raw
    uv run pytest -m "not hardware"

First contact with the hand control (read-only, sends only queries):

    uv run okapi mount-info /dev/ttyUSB0      # or --dry-run to try it against a simulator
