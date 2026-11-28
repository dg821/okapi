"""Plate solving with the ASTAP command-line program.

ASTAP cannot read Canon RAW, so each frame is decoded here, binned 2x2 into a mono image, and
written as FITS. ASTAP is then run with the mount's position as a hint (fast); if that fails it
can retry blind. The solution comes back as a ``.wcs`` file, from which we take the sky position
of the image center.

Facts taken from ASTAP's own command-line source (astap-cli 2024.05.01): ``-ra`` is in hours,
``-spd`` is south-pole distance (Dec + 90 deg), ``-r`` is the search radius in degrees, ``-fov 0``
means "work it out from the FITS header", the ``.wcs`` file is written only on success, and the
exit codes are 0 solved, 1 no solution, 2 too few stars, 16 bad image, 32 no star database,
33 database read error.

NOT yet verified against a real ASTAP (PROJECT_PLAN A5): that the header keywords FOCALLEN and
XPIXSZ give the right field of view, the exact layout of the ``.wcs`` file, and that omitting
``-ra/-spd`` with ``-r 180`` performs a true blind solve. ``tests/test_astap_hardware.py``
checks all three on a machine with ASTAP and a sample frame.
"""

import io
import subprocess
import tempfile
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS

from okapi.camera.base import CapturedImage
from okapi.solve.base import SolveError, SolveResult

Decoder = Callable[[CapturedImage], np.ndarray]

EXIT_MESSAGES = {
    1: "no solution found",
    2: "not enough stars detected (exposure too short, clouds, or out of focus?)",
    16: "ASTAP could not read the image file",
    32: "ASTAP found no star database (install one, e.g. D50, next to the astap binary)",
    33: "ASTAP could not read its star database",
}
# Searching wider or blind cannot fix these.
UNRECOVERABLE = {2, 16, 32, 33}


@dataclass(frozen=True)
class RunResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


Runner = Callable[[list[str], float], RunResult]


def run_astap(argv: list[str], timeout: float) -> RunResult:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as e:
        raise SolveError(f"ASTAP not found ({argv[0]}); install the astap CLI") from e
    except subprocess.TimeoutExpired as e:
        raise SolveError(f"ASTAP timed out after {timeout:.0f} s") from e
    return RunResult(proc.returncode, proc.stdout, proc.stderr)


@dataclass(frozen=True)
class Optics:
    """Defaults: Orion XT10g (1200 mm focal length) with a Canon T3i (22.3 mm / 5184 px)."""

    focal_length: u.Quantity = 1200 * u.mm
    pixel_size: u.Quantity = 4.3 * u.um  # unbinned pixel pitch


def bin_mosaic(mosaic: np.ndarray, binning: int = 2) -> np.ndarray:
    """Sum ``binning`` x ``binning`` blocks of a Bayer mosaic into a mono float32 image.

    With 2x2 each block holds one red, two green and one blue sample, so the sum is a clean
    luminance and the color pattern disappears. Odd edges are cropped.
    """
    if binning < 1:
        raise ValueError("binning must be at least 1")
    h, w = (mosaic.shape[0] // binning) * binning, (mosaic.shape[1] // binning) * binning
    if h == 0 or w == 0:
        raise ValueError("image smaller than one bin")
    blocks = mosaic[:h, :w].astype(np.float32).reshape(h // binning, binning, w // binning, binning)
    return blocks.sum(axis=(1, 3), dtype=np.float32)


def rawpy_decoder(binning: int = 2) -> Decoder:
    """Decode a RAW frame with libraw (``uv sync --extra raw``) and bin it to mono."""

    def decode(image: CapturedImage) -> np.ndarray:
        try:
            import rawpy
        except ImportError as e:
            raise SolveError("rawpy is not installed; run `uv sync --extra raw`") from e
        try:
            with rawpy.imread(io.BytesIO(image.data)) as raw:
                return bin_mosaic(np.asarray(raw.raw_image_visible), binning)
        except Exception as e:  # libraw raises several unrelated types on bad data
            raise SolveError(f"could not decode the RAW frame: {e}") from e

    return decode


def write_fits(
    path: Path, data: np.ndarray, optics: Optics, binning: int, image: CapturedImage
) -> None:
    hdu = fits.PrimaryHDU(data=data.astype(np.float32))
    pixel_um = float((optics.pixel_size * binning).to_value(u.um))
    hdu.header["FOCALLEN"] = (float(optics.focal_length.to_value(u.mm)), "focal length [mm]")
    hdu.header["XPIXSZ"] = (pixel_um, "pixel size after binning [um]")
    hdu.header["YPIXSZ"] = (pixel_um, "pixel size after binning [um]")
    hdu.header["EXPTIME"] = (image.settings.exposure_s, "exposure [s]")
    hdu.header["DATE-OBS"] = (image.start_time.utc.isot, "UTC at shutter command")
    hdu.writeto(path)


def read_wcs(text: str) -> WCS:
    """The celestial WCS in the text of an ASTAP ``.wcs`` file."""
    text = text.strip("\x00 \r\n")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", AstropyUserWarning)  # we validate below instead
            header = fits.Header.fromstring(text, sep="\n" if "\n" in text else "")
            wcs = WCS(header, naxis=2)
        if not wcs.has_celestial:
            raise ValueError("no celestial WCS")
        return wcs
    except Exception as e:
        raise SolveError(f"could not read ASTAP's WCS output: {e}") from e


def parse_wcs(text: str, width: int, height: int) -> SkyCoord:
    """Sky position of the image center from the text of an ASTAP ``.wcs`` file."""
    wcs = read_wcs(text)
    try:
        return wcs.pixel_to_world((width - 1) / 2, (height - 1) / 2).icrs
    except Exception as e:
        raise SolveError(f"could not read ASTAP's WCS output: {e}") from e


class AstapSolver:
    def __init__(
        self,
        *,
        hint: Callable[[], SkyCoord | None] | None = None,
        run: Runner = run_astap,
        decode: Decoder | None = None,
        astap_path: str = "astap_cli",
        optics: Optics | None = None,
        binning: int = 2,
        fov_height: u.Quantity | None = None,
        search_radius: u.Quantity = 10 * u.deg,
        blind_fallback: bool = True,
        max_stars: int = 500,
        hinted_timeout: float = 60.0,
        blind_timeout: float = 300.0,
    ) -> None:
        """``hint`` returns the mount's reported position (or None). ``fov_height`` overrides
        the field of view; leave it None to let ASTAP derive it from the FITS header."""
        self._hint = hint
        self._run = run
        self._binning = binning
        self._decode = decode or rawpy_decoder(binning)
        self._astap = astap_path
        self._optics = optics or Optics()
        self._fov = fov_height
        self._radius = search_radius
        self._blind = blind_fallback
        self._max_stars = max_stars
        self._timeouts = (hinted_timeout, blind_timeout)

    def _argv(self, fits_path: Path, base: Path, hint: SkyCoord | None) -> list[str]:
        fov = "0" if self._fov is None else f"{self._fov.to_value(u.deg):g}"
        argv = [self._astap, "-f", str(fits_path), "-o", str(base), "-wcs"]
        argv += ["-fov", fov, "-s", str(self._max_stars)]
        if hint is not None:
            ra, dec = cast(u.Quantity, hint.ra), cast(u.Quantity, hint.dec)
            ra_h = float(ra.to_value(u.hourangle))
            spd = float(dec.to_value(u.deg)) + 90.0
            argv += ["-ra", f"{ra_h:.5f}", "-spd", f"{spd:.4f}"]
            argv += ["-r", f"{self._radius.to_value(u.deg):g}"]
        else:
            argv += ["-r", "180"]
        return argv

    @property
    def decoder(self) -> Decoder:
        """The frame decoder; its output is the pixel grid that ``SolveResult.wcs`` describes."""
        return self._decode

    def solve(self, image: CapturedImage) -> SolveResult:
        data = self._decode(image)
        height, width = data.shape
        hint = self._hint() if self._hint else None
        plans: list[tuple[SkyCoord | None, float]] = []
        if hint is not None:
            plans.append((hint, self._timeouts[0]))
        if hint is None or self._blind:
            plans.append((None, self._timeouts[1]))

        with tempfile.TemporaryDirectory(prefix="okapi-astap-") as tmp:
            fits_path, base = Path(tmp) / "frame.fits", Path(tmp) / "solve"
            write_fits(fits_path, data, self._optics, self._binning, image)
            last = RunResult(1)
            for plan_hint, timeout in plans:
                last = self._run(self._argv(fits_path, base, plan_hint), timeout)
                wcs_path = base.with_suffix(".wcs")
                if last.returncode == 0:
                    if not wcs_path.exists():
                        raise SolveError("ASTAP reported success but wrote no .wcs file")
                    text = wcs_path.read_text(errors="replace")
                    center = parse_wcs(text, width, height)
                    return SolveResult(center=center, wcs=read_wcs(text))
                if last.returncode in UNRECOVERABLE:
                    break
        reason = EXIT_MESSAGES.get(last.returncode, f"ASTAP exited with code {last.returncode}")
        raise SolveError(reason)
