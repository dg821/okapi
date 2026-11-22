"""Preflight checks for the real rig: each returns pass/warn/fail with a likely fix.

Every probe is injected, so the checks are tested with fakes and the CLI wires the real ones.
Nothing here moves the mount or fires the shutter.
"""

import platform
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u

from okapi.camera.gphoto import Runner, parse_shutter_choices
from okapi.clock import Clock
from okapi.mount.synscan import HandsetReport
from okapi.sensors.base import GpsProtocol

PASS, WARN, FAIL = "pass", "warn", "fail"
CLOCK_WARN = 0.5 * u.s
CLOCK_FAIL = 2.0 * u.s
MIN_FREE_GB = 5.0
STAR_DB_GLOBS = ("[dDhHvVwWgG][0-9][0-9]_*.290", "[dDhHvVwWgG][0-9][0-9]_*.1476")


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


def check_architecture(machine: str | None = None) -> Check:
    m = machine or platform.machine()
    if m in ("aarch64", "arm64", "x86_64", "AMD64"):
        return Check("os", PASS, f"{m} (64-bit)")
    return Check(
        "os",
        FAIL if m.startswith("armv") else WARN,
        f"{m}: 32-bit or unknown",
        "reflash 64-bit Raspberry Pi OS (scientific wheels are not built for 32-bit)",
    )


def check_camera(run: Runner) -> Check:
    try:
        listing = run(["--auto-detect"])
    except Exception as e:
        return Check("camera", FAIL, str(e), "sudo apt install gphoto2")
    rows = [ln for ln in listing.splitlines()[2:] if ln.strip()]
    if not rows:
        return Check(
            "camera",
            FAIL,
            "gphoto2 sees no camera",
            "USB cable in, camera on, not mounted by the desktop (use Pi OS Lite, or "
            "`gio mount -s gphoto2`); try another USB port",
        )
    try:
        choices = parse_shutter_choices(run(["--get-config", "shutterspeed"]))
    except Exception as e:
        return Check("camera", FAIL, f"found {rows[0].split('  ')[0].strip()} but: {e}")
    name = rows[0].split("  ")[0].strip()
    if not choices:
        return Check(
            "camera", FAIL, f"{name}: no numeric shutter speeds", "set the mode dial to M (manual)"
        )
    if 30.0 not in [s for _, s in choices]:
        return Check("camera", WARN, f"{name}: no 30 s shutter speed", "set the mode dial to M")
    return Check("camera", PASS, f"{name}; shutter speeds up to 30 s available")


def check_gps(gps: GpsProtocol, clock: Clock) -> list[Check]:
    try:
        fix = gps.read_fix()
    except Exception as e:
        return [
            Check(
                "gps",
                FAIL,
                str(e),
                "needs sky view and a few minutes for a cold start; check `cat /dev/ttyACM0` "
                "shows $..GGA/$..RMC; stop gpsd/ModemManager if they hold the port",
            )
        ]
    out = [
        Check(
            "gps",
            PASS if fix.satellites >= 4 else WARN,
            f"fix at {fix.lat.to_value(u.deg):.4f}, {fix.lon.to_value(u.deg):.4f}, "
            f"{fix.height.to_value(u.m):.0f} m, {fix.satellites} satellites",
            "" if fix.satellites >= 4 else "move the puck to see more sky",
        )
    ]
    off = abs((clock.now() - fix.time).to_value(u.s))
    if off > CLOCK_FAIL.value:
        status, fix_hint = FAIL, "sync the Pi clock: enable NTP or set up chrony (docs/pi-setup.md)"
    elif off > CLOCK_WARN.value:
        status, fix_hint = WARN, "clock is loose; chrony with GPS or NTP will tighten it"
    else:
        status, fix_hint = PASS, ""
    out.append(Check("clock", status, f"system clock is {off:.2f} s from GPS time", fix_hint))
    return out


def check_mount(read_report: Callable[[], HandsetReport]) -> Check:
    try:
        rep = read_report()
    except Exception as e:
        return Check(
            "mount",
            FAIL,
            str(e),
            "handset on and past its start-up prompts; cable in the small serial jack; "
            "see `ls -l /dev/serial/by-id/`; user in the dialout group",
        )
    info = rep.info
    return Check(
        "mount",
        PASS,
        f"{info.model_name}, firmware {info.version}, "
        f"az/alt {rep.altaz.az.to_value(u.deg):.2f}/{rep.altaz.alt.to_value(u.deg):.2f}",
    )


def check_astap(astap_path: str, db_dirs: Sequence[Path] | None = None) -> Check:
    exe = shutil.which(astap_path)
    if exe is None:
        return Check(
            "astap", FAIL, f"{astap_path} not found", "install the ASTAP CLI (docs/pi-setup.md)"
        )
    dirs = (
        list(db_dirs)
        if db_dirs is not None
        else [
            Path(exe).resolve().parent,
            Path("/opt/astap"),
            Path("/usr/share/astap"),
            Path.home() / ".astap",
        ]
    )
    for d in dirs:
        for pat in STAR_DB_GLOBS:
            if d.is_dir() and any(d.glob(pat)):
                return Check("astap", PASS, f"{exe}; star database in {d}")
    return Check(
        "astap",
        WARN,
        f"{exe} found but no star database in {len(dirs)} usual places",
        "install the D50 database next to the astap binary",
    )


def check_disk(path: Path, free_gb: float | None = None) -> Check:
    gb = free_gb if free_gb is not None else shutil.disk_usage(path).free / 1e9
    if gb < MIN_FREE_GB:
        return Check(
            "disk", FAIL, f"{gb:.1f} GB free at {path}", "RAW frames are ~25 MB; use a bigger drive"
        )
    return Check("disk", PASS, f"{gb:.0f} GB free at {path} (about {int(gb * 1000 / 25)} frames)")


def format_checks(checks: Sequence[Check]) -> str:
    mark = {PASS: "ok  ", WARN: "WARN", FAIL: "FAIL"}
    lines = []
    for c in checks:
        lines.append(f"[{mark[c.status]}] {c.name}: {c.detail}")
        if c.fix and c.status != PASS:
            lines.append(f"       -> {c.fix}")
    bad = sum(c.status == FAIL for c in checks)
    lines.append(f"{len(checks)} checks, {bad} failed" if bad else f"{len(checks)} checks passed")
    return "\n".join(lines)
