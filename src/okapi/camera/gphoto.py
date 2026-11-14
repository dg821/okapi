"""Canon EOS Rebel T3i (EOS 600D) driver using the ``gphoto2`` command-line tool.

We shell out to the CLI rather than binding libgphoto2: it is what the gphoto2 docs describe,
it is easy to try by hand on the Pi, and the subprocess boundary is trivial to fake in tests.

Exposures of 30 s or less pick a shutter speed from the camera's own list. Longer exposures use
bulb timing. The bulb sequence follows the gphoto2 remote-control docs and has NOT yet been
verified against a real T3i (see PROJECT_PLAN.md, hardware bring-up); the camera must be in M mode.
"""

import re
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from okapi.camera.base import CameraError, CapturedImage, ExposureSettings
from okapi.clock import Clock

MAX_TIMED_EXPOSURE_S = 30.0
Runner = Callable[[list[str]], str]


def run_gphoto2(args: list[str]) -> str:
    try:
        proc = subprocess.run(
            ["gphoto2", *args], capture_output=True, text=True, timeout=600, check=False
        )
    except FileNotFoundError as e:
        raise CameraError("gphoto2 is not installed (sudo apt install gphoto2)") from e
    except subprocess.TimeoutExpired as e:
        raise CameraError(f"gphoto2 timed out: {args}") from e
    if proc.returncode != 0:
        raise CameraError(f"gphoto2 {' '.join(args)} failed: {proc.stderr.strip() or proc.stdout}")
    return proc.stdout


def parse_shutter_choices(config_output: str) -> list[tuple[str, float]]:
    """Parse ``--get-config shutterspeed`` output into (label, seconds) pairs.

    Canon labels look like ``30``, ``1.3``, ``1/250``, ``bulb``. Non-numeric labels are skipped.
    """
    choices: list[tuple[str, float]] = []
    for m in re.finditer(r"^Choice: \d+ (\S+)$", config_output, re.MULTILINE):
        label = m.group(1)
        try:
            if "/" in label:
                num, den = label.split("/")
                seconds = float(num) / float(den)
            else:
                seconds = float(label)
        except ValueError:
            continue
        choices.append((label, seconds))
    return choices


def nearest_shutter(choices: list[tuple[str, float]], seconds: float) -> str:
    """Label of the choice within 5% of ``seconds``; raise rather than silently expose wrong."""
    if not choices:
        raise CameraError("camera reported no numeric shutter speeds (is it in M mode?)")
    label, value = min(choices, key=lambda c: abs(c[1] - seconds))
    if abs(value - seconds) > 0.05 * seconds:
        raise CameraError(
            f"no shutter speed near {seconds} s (nearest {label} s); "
            f"use one of the camera's speeds or exceed {MAX_TIMED_EXPOSURE_S:.0f} s for bulb"
        )
    return label


class GPhotoCamera:
    def __init__(self, clock: Clock, *, run: Runner = run_gphoto2) -> None:
        self._clock = clock
        self._run = run

    def capture(self, settings: ExposureSettings) -> CapturedImage:
        with tempfile.TemporaryDirectory(prefix="okapi-") as tmp:
            template = str(Path(tmp) / "frame.%C")
            self._run(["--set-config", f"iso={settings.iso}"])
            seconds = settings.exposure_s
            if seconds <= MAX_TIMED_EXPOSURE_S:
                choices = parse_shutter_choices(self._run(["--get-config", "shutterspeed"]))
                self._run(["--set-config", f"shutterspeed={nearest_shutter(choices, seconds)}"])
                start = self._clock.now()
                self._run(
                    ["--capture-image-and-download", "--filename", template, "--force-overwrite"]
                )
            else:
                self._run(["--set-config", "shutterspeed=bulb"])
                start = self._clock.now()
                self._run(
                    [
                        "--set-config",
                        "bulb=1",
                        f"--wait-event={seconds:g}s",
                        "--set-config",
                        "bulb=0",
                        "--wait-event-and-download=10s",
                        "--filename",
                        template,
                        "--force-overwrite",
                    ]
                )
            files = [p for p in Path(tmp).glob("frame.*") if p.is_file()]
            if len(files) != 1:
                raise CameraError(f"expected one downloaded frame, found {len(files)}")
            return CapturedImage(
                files[0].read_bytes(), "." + files[0].suffix.lstrip(".").lower(), start, settings
            )
