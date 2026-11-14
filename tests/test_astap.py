from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from okapi.camera.base import CapturedImage, ExposureSettings
from okapi.solve.astap import (
    AstapSolver,
    Optics,
    RunResult,
    bin_mosaic,
    parse_wcs,
    rawpy_decoder,
    run_astap,
    write_fits,
)
from okapi.solve.base import SolveError

W, H = 120, 80  # decoded (binned) size used by the fake decoder
CENTER = SkyCoord(ra=210.5 * u.deg, dec=-3.25 * u.deg)


def make_wcs_text(center: SkyCoord = CENTER, *, newlines: bool = False) -> str:
    """What ASTAP would write: a TAN WCS for a W x H image, 1.48 arcsec/px (T3i binned 2x2)."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [(W + 1) / 2, (H + 1) / 2]
    w.wcs.cdelt = [-1.48 / 3600, 1.48 / 3600]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    return w.to_header().tostring(sep="\n" if newlines else "")


def image() -> CapturedImage:
    from astropy.time import Time

    return CapturedImage(
        b"RAW", ".cr2", Time("2026-06-21T04:00:00", scale="utc"), ExposureSettings(2 * u.s, 1600)
    )


def fake_decoder(img: CapturedImage) -> np.ndarray:
    return np.random.default_rng(0).normal(100, 5, (H, W)).astype(np.float32)


class FakeAstap:
    """Stands in for the astap binary: records argv, writes (or not) a .wcs, returns codes."""

    def __init__(self, *codes: int, wcs_text: str | None = None, write_on_zero: bool = True):
        self.codes = list(codes)
        self.calls: list[tuple[list[str], float]] = []
        self.wcs_text = wcs_text if wcs_text is not None else make_wcs_text()
        self.write_on_zero = write_on_zero

    def __call__(self, argv: list[str], timeout: float) -> RunResult:
        self.calls.append((argv, timeout))
        code = self.codes.pop(0) if self.codes else 0
        if code == 0 and self.write_on_zero:
            base = Path(argv[argv.index("-o") + 1])
            base.with_suffix(".wcs").write_text(self.wcs_text)
        return RunResult(code)


def solver(run, **kw) -> AstapSolver:
    return AstapSolver(run=run, decode=fake_decoder, **kw)


def flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


# -- decoding and FITS ---------------------------------------------------------------------
def test_bin_mosaic_sums_blocks_and_crops_odd_edges():
    m = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7)
    out = bin_mosaic(m, 2)
    assert out.shape == (2, 3)
    assert out[0, 0] == m[0, 0] + m[0, 1] + m[1, 0] + m[1, 1]
    assert out.dtype == np.float32


def test_bin_mosaic_does_not_overflow_uint16():
    m = np.full((4, 4), 60000, dtype=np.uint16)
    assert bin_mosaic(m, 2)[0, 0] == 240000


def test_bin_mosaic_rejects_bad_input():
    with pytest.raises(ValueError):
        bin_mosaic(np.zeros((4, 4)), 0)
    with pytest.raises(ValueError):
        bin_mosaic(np.zeros((1, 1)), 2)


def test_write_fits_records_optics_and_binned_pixel_size(tmp_path):
    p = tmp_path / "f.fits"
    write_fits(p, np.zeros((8, 6), np.float32), Optics(), 2, image())
    h = fits.getheader(p)
    assert h["FOCALLEN"] == 1200.0
    assert h["XPIXSZ"] == pytest.approx(8.6)
    assert h["EXPTIME"] == 2.0
    assert fits.getdata(p).shape == (8, 6)


def test_rawpy_decoder_reports_garbage_as_solve_error():
    with pytest.raises(SolveError, match="decode"):
        rawpy_decoder(2)(image())


def test_rawpy_decoder_reports_missing_rawpy(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_rawpy(name, *a, **k):
        if name == "rawpy":
            raise ImportError("nope")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_rawpy)
    with pytest.raises(SolveError, match="uv sync --extra raw"):
        rawpy_decoder(2)(image())


# -- WCS parsing ---------------------------------------------------------------------------
@pytest.mark.parametrize("newlines", [False, True])
def test_parse_wcs_both_header_layouts(newlines):
    c = parse_wcs(make_wcs_text(newlines=newlines), W, H)
    assert c.separation(CENTER).to_value(u.arcsec) < 0.01


def test_parse_wcs_uses_the_true_image_center_not_crpix():
    # CRPIX deliberately not at the center: the answer must still be the center's position.
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [10.0, 10.0]
    w.wcs.cdelt = [-1.48 / 3600, 1.48 / 3600]
    w.wcs.crval = [210.0, -3.0]
    expected = w.pixel_to_world((W - 1) / 2, (H - 1) / 2)
    got = parse_wcs(w.to_header().tostring(sep=""), W, H)
    assert got.separation(expected).to_value(u.arcsec) < 0.01
    assert got.separation(SkyCoord(210 * u.deg, -3 * u.deg)).to_value(u.arcsec) > 60


@pytest.mark.parametrize("bad", ["", "garbage", "SIMPLE  =                    T"])
def test_parse_wcs_rejects_non_wcs(bad):
    with pytest.raises(SolveError, match="WCS"):
        parse_wcs(bad, W, H)


# -- command line and control flow ---------------------------------------------------------
def test_hinted_solve_command_line_and_result():
    run = FakeAstap()
    hint = SkyCoord(ra=15 * u.deg, dec=-20 * u.deg)  # 1 h, spd 70
    result = solver(run, hint=lambda: hint).solve(image())
    argv, timeout = run.calls[0]
    assert argv[0] == "astap"
    assert flag(argv, "-ra") == "1.00000"
    assert flag(argv, "-spd") == "70.0000"
    assert flag(argv, "-r") == "10"
    assert flag(argv, "-fov") == "0"  # let ASTAP use the FITS header
    assert "-wcs" in argv
    assert timeout == 60.0
    assert result.center.separation(CENTER).to_value(u.arcsec) < 0.01
    assert len(run.calls) == 1


def test_spd_is_never_negative_even_at_the_south_pole():
    run = FakeAstap()
    solver(run, hint=lambda: SkyCoord(0 * u.deg, -89.9 * u.deg)).solve(image())
    assert float(flag(run.calls[0][0], "-spd")) > 0


def test_explicit_fov_is_passed_through():
    run = FakeAstap()
    solver(run, fov_height=0.71 * u.deg).solve(image())
    assert flag(run.calls[0][0], "-fov") == "0.71"


def test_no_hint_means_a_single_blind_solve():
    run = FakeAstap()
    solver(run).solve(image())
    argv, timeout = run.calls[0]
    assert "-ra" not in argv and "-spd" not in argv
    assert flag(argv, "-r") == "180"
    assert timeout == 300.0
    assert len(run.calls) == 1


def test_failed_hinted_solve_falls_back_to_blind():
    run = FakeAstap(1, 0)
    solver(run, hint=lambda: SkyCoord(15 * u.deg, 0 * u.deg)).solve(image())
    assert len(run.calls) == 2
    assert "-ra" in run.calls[0][0]
    assert "-ra" not in run.calls[1][0]


def test_blind_fallback_can_be_disabled():
    run = FakeAstap(1)
    with pytest.raises(SolveError, match="no solution"):
        solver(run, hint=lambda: SkyCoord(15 * u.deg, 0 * u.deg), blind_fallback=False).solve(
            image()
        )
    assert len(run.calls) == 1


def test_mount_with_no_position_yet_solves_blind():
    run = FakeAstap()
    solver(run, hint=lambda: None).solve(image())
    assert "-ra" not in run.calls[0][0]


@pytest.mark.parametrize(
    ("code", "text"),
    [
        (2, "not enough stars"),
        (16, "could not read the image"),
        (32, "star database"),
        (33, "read its star database"),
    ],
)
def test_unrecoverable_exit_codes_do_not_retry_blind(code, text):
    run = FakeAstap(code)
    with pytest.raises(SolveError, match=text):
        solver(run, hint=lambda: SkyCoord(15 * u.deg, 0 * u.deg)).solve(image())
    assert len(run.calls) == 1


def test_unknown_exit_code_is_reported():
    with pytest.raises(SolveError, match="code 99"):
        solver(FakeAstap(99)).solve(image())


def test_success_without_wcs_file_is_an_error():
    with pytest.raises(SolveError, match=r"no \.wcs"):
        solver(FakeAstap(0, write_on_zero=False)).solve(image())


def test_garbage_wcs_is_an_error():
    with pytest.raises(SolveError, match="WCS"):
        solver(FakeAstap(0, wcs_text="not a header")).solve(image())


def test_custom_binary_path_is_used():
    run = FakeAstap()
    solver(run, astap_path="/opt/astap/astap").solve(image())
    assert run.calls[0][0][0] == "/opt/astap/astap"


def test_input_fits_exists_while_astap_runs(tmp_path):
    seen = {}

    def run(argv, timeout):
        p = Path(flag(argv, "-f"))
        seen["exists"], seen["shape"] = p.exists(), fits.getdata(p).shape
        Path(flag(argv, "-o")).with_suffix(".wcs").write_text(make_wcs_text())
        return RunResult(0)

    solver(run).solve(image())
    assert seen == {"exists": True, "shape": (H, W)}


def test_real_runner_reports_missing_binary():
    with pytest.raises(SolveError, match="not found"):
        run_astap(["/nonexistent/astap", "-f", "x"], 5)


def test_astap_solver_drives_the_centering_loop_end_to_end():
    """AstapSolver plugs into center_on: hint from the mount, solution from 'ASTAP'."""
    from okapi.align.center import center_on
    from okapi.coords import AltAzPosition, altaz_to_radec
    from okapi.sim import make_sim_rig

    rig = make_sim_rig()
    rig.mount.set_site(rig.site)
    argvs: list[list[str]] = []

    def run(argv, timeout):
        argvs.append(argv)
        text = make_wcs_text(rig.mount.true_radec())  # 'ASTAP' reports where we truly point
        Path(flag(argv, "-o")).with_suffix(".wcs").write_text(text)
        return RunResult(0)

    target = altaz_to_radec(AltAzPosition(150 * u.deg, 55 * u.deg), rig.site, rig.clock.now())
    s = AstapSolver(run=run, decode=fake_decoder, hint=rig.mount.get_radec)
    result = center_on(
        target, mount=rig.mount, camera=rig.camera, solver=s, site=rig.site, settings=rig.settings
    )
    assert rig.mount.true_radec().separation(target).to_value(u.arcsec) < 15
    assert result.iterations <= 3
    assert all("-ra" in a for a in argvs)  # every solve was hinted with the mount's position
