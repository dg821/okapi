from pathlib import Path

import astropy.units as u
import pytest

from okapi.camera.base import CameraError, ExposureSettings
from okapi.camera.fake import FakeCamera
from okapi.camera.gphoto import (
    GPhotoCamera,
    nearest_shutter,
    parse_shutter_choices,
)

CANON_CONFIG = """Label: Shutter Speed
Type: RADIO
Current: 1/60
Choice: 0 bulb
Choice: 1 30
Choice: 2 25
Choice: 3 20
Choice: 4 15
Choice: 5 13
Choice: 6 10
Choice: 7 1.3
Choice: 8 1/60
END
"""


def test_exposure_settings_validation():
    with pytest.raises(ValueError):
        ExposureSettings(0 * u.s, 800)
    with pytest.raises(ValueError):
        ExposureSettings(10 * u.s, 0)
    assert ExposureSettings(2 * u.min, 800).exposure_s == 120.0


def test_fake_camera_advances_clock_and_records(clock):
    cam = FakeCamera(clock)
    t0 = clock.now()
    img = cam.capture(ExposureSettings(15 * u.s, 1600))
    assert clock.monotonic() == 15
    assert img.start_time == t0
    assert img.extension == ".cr2"
    assert len(cam.captures) == 1


def test_fake_camera_failure(clock):
    cam = FakeCamera(clock, fail_after=1)
    cam.capture(ExposureSettings(1 * u.s, 100))
    with pytest.raises(CameraError):
        cam.capture(ExposureSettings(1 * u.s, 100))


def test_parse_shutter_choices_handles_fractions_and_skips_bulb():
    parsed = dict(parse_shutter_choices(CANON_CONFIG))
    assert parsed["30"] == 30.0
    assert parsed["1/60"] == pytest.approx(1 / 60)
    assert "bulb" not in parsed


def test_nearest_shutter_exact_and_refuses_far_matches():
    choices = parse_shutter_choices(CANON_CONFIG)
    assert nearest_shutter(choices, 15.0) == "15"
    with pytest.raises(CameraError, match="no shutter speed near"):
        nearest_shutter(choices, 17.0)  # between 15 and 20: don't silently pick the wrong one
    with pytest.raises(CameraError, match="M mode"):
        nearest_shutter([], 15.0)


class RecordingRunner:
    def __init__(self, *, write_files: tuple[str, ...] = ("frame.cr2",)) -> None:
        self.calls: list[list[str]] = []
        self._write_files = write_files

    def __call__(self, args: list[str]) -> str:
        self.calls.append(args)
        if args[:2] == ["--get-config", "shutterspeed"]:
            return CANON_CONFIG
        if "--filename" in args:
            template = args[args.index("--filename") + 1]
            for name in self._write_files:
                Path(template).with_name(name).write_bytes(b"RAWBYTES")
        return ""


def test_gphoto_timed_exposure_sets_iso_and_shutter(clock):
    run = RecordingRunner()
    img = GPhotoCamera(clock, run=run).capture(ExposureSettings(15 * u.s, 1600))
    assert ["--set-config", "iso=1600"] in run.calls
    assert ["--set-config", "shutterspeed=15"] in run.calls
    assert any("--capture-image-and-download" in c for c in run.calls)
    assert not any("bulb=1" in " ".join(c) for c in run.calls)
    assert img.data == b"RAWBYTES"
    assert img.extension == ".cr2"


def test_gphoto_long_exposure_uses_bulb(clock):
    run = RecordingRunner()
    GPhotoCamera(clock, run=run).capture(ExposureSettings(45 * u.s, 800))
    assert ["--set-config", "shutterspeed=bulb"] in run.calls
    bulb = next(c for c in run.calls if "bulb=1" in c)
    assert "--wait-event=45s" in bulb
    assert bulb.index("bulb=1") < bulb.index("--wait-event=45s") < bulb.index("bulb=0")


def test_gphoto_missing_download_is_an_error(clock):
    run = RecordingRunner(write_files=())
    with pytest.raises(CameraError, match="found 0"):
        GPhotoCamera(clock, run=run).capture(ExposureSettings(15 * u.s, 800))


def test_gphoto_runner_error_propagates(clock):
    def boom(args: list[str]) -> str:
        raise CameraError("no camera")

    with pytest.raises(CameraError, match="no camera"):
        GPhotoCamera(clock, run=boom).capture(ExposureSettings(15 * u.s, 800))
