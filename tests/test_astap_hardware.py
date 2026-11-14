"""Bring-up checks for the ASTAP wrapper. Need a real ASTAP, a star database and a sample frame.

Run on the Pi (or any machine with ASTAP) with::

    OKAPI_TEST_FRAME=/path/to/frame.cr2 [OKAPI_TEST_RADEC=ra_deg,dec_deg] \\
        uv run pytest -m hardware tests/test_astap_hardware.py -s

OKAPI_TEST_RADEC is where the frame is known to point; without it the test only checks that a
sensible solution comes back. These confirm PROJECT_PLAN assumption A5 and the unverified
details listed at the top of ``okapi/solve/astap.py``.
"""

import os
import shutil
from pathlib import Path

import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from okapi.camera.base import CapturedImage, ExposureSettings
from okapi.solve.astap import AstapSolver

FRAME = os.environ.get("OKAPI_TEST_FRAME")

pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(shutil.which("astap") is None, reason="astap not installed"),
    pytest.mark.skipif(not FRAME, reason="set OKAPI_TEST_FRAME to a sample RAW frame"),
]


def load() -> CapturedImage:
    path = Path(FRAME or "")
    return CapturedImage(
        path.read_bytes(),
        path.suffix.lower(),
        Time.now(),
        ExposureSettings(15 * u.s, 1600),
    )


def test_blind_solve_returns_a_position_and_matches_expected_if_given():
    result = AstapSolver().solve(load())
    assert 0 <= result.center.ra.deg < 360
    assert -90 <= result.center.dec.deg <= 90
    expected = os.environ.get("OKAPI_TEST_RADEC")
    if expected:
        ra, dec = (float(x) for x in expected.split(","))
        sep = result.center.separation(SkyCoord(ra * u.deg, dec * u.deg))
        assert sep.to_value(u.deg) < 0.1, f"solved {result.center}, expected {expected}"


def test_hinted_solve_agrees_with_blind_solve():
    blind = AstapSolver().solve(load()).center
    hinted = AstapSolver(hint=lambda: blind.directional_offset_by(0 * u.deg, 1 * u.deg)).solve(
        load()
    )
    assert hinted.center.separation(blind).to_value(u.arcsec) < 30
