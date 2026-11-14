# okapi project

Raspberry Pi control of an Orion XT10g GoTo Dobsonian and a Canon EOS Rebel T3i.
Read PROJECT_PLAN.md before large changes; assumptions A1-A5 there are unverified.

## Commands
- Setup: `uv sync`
- Test: `uv run pytest -m "not hardware"`   Lint: `uv run ruff check . && uv run ruff format --check .`
- Types: `uv run pyright`   CLI: `uv run okapi --help`

## Rules
- Use astropy units and Time at API boundaries; no bare-float angles or times.
- Azimuth is north through east, 0-360 deg. Positions from the mount are what it *reports*,
  which may differ from where it truly points until aligned.
- Sequencing code depends only on MountProtocol / CameraProtocol / Clock. Tests use fakes and
  SimClock; never import a hardware driver in a test of logic.
- Every new fake behavior and every driver command needs a test. Hardware-only tests get
  `@pytest.mark.hardware`.
- Record non-obvious decisions as ADRs in docs/decisions/.
- Captured data in data/ is never committed.
