# okapi project plan

Control an Orion SkyQuest XT10g GoTo Dobsonian (10", f/4.7, 1200 mm, alt-az) and a Canon EOS
Rebel T3i (EOS 600D) from a Raspberry Pi, and automate the nightly setup.

## Goals

1. Slew to Az/Alt and to ICRS RA/Dec.
2. Track at the sidereal rate (alt-az mount: both axes move) for 10-20 s exposures.
3. Take exposures on the T3i (timed up to 30 s, bulb beyond).
4. Automate start-of-night calibration: geodetic position and UTC from GPS, push both to the
   mount, then two-star alignment with a search-and-center loop.

## Assumptions to verify (hardware not available yet)

| # | Assumption | How to verify | Consequence if wrong |
|---|---|---|---|
| A1 | The 2013 handset is a SynScan-family controller with an RS-232 port, 9600 8N1 | **Confirmed 2026-11-21** from the XTg manual (IN 388 Rev. D 06/13) and the photos: SynScan AZ handset; the wide RJ-45 jack takes the coiled cable to the mount, the smaller RJ-12 jack is the RS-232 PC port; an RS-232 computer cable was in the box; protocol v3.3 says 9600 8N1. Still unknown: firmware version (read it with `okapi mount-info`; needs 3.38+ for protocol v3.3, 3.03+ for rate slews) and the supplied cable's exact pinout. See docs/design/synscan-protocol-notes.md | Older firmware lacks commands; update it over the same port. Fallback: drive the motor controller directly (ADR 0004) |
| A2 | The protocol allows sync/alignment from the Pi | **Resolved (no)**: there is no command to start the handset's alignment; sync exists but is documented for an aligned handset only. The Pi pointing model is therefore the primary correction, driven through Az/Alt commands | Done |
| A6 | Sidereal tracking works on an unaligned handset | Decided: the Pi drives tracking with variable-rate slews (ADR 0004). Verify on the real mount: rates move the axes, handset tracking stays off, drift stays small | If it does not work, align the handset by keypad each night and use its own tracking (mode 1) |
| A3 | gphoto2 bulb timing works on the T3i | `gphoto2 --get-config shutterspeed`; run the bulb sequence by hand | Keep exposures at or under 30 s, or add a bulb remote cable via a GPIO relay |
| A4 | T3i reaches focus at the Dobsonian's prime focus | Try it at the focuser | Need a Barlow or other optics; changes the field of view |
| A5 | Plate solving works on the Pi with ASTAP at ~1.1 x 0.7 deg | Install ASTAP + D50 database; run `OKAPI_TEST_FRAME=... uv run pytest -m hardware tests/test_astap_hardware.py -s`. Also confirms: FOCALLEN/XPIXSZ give the right field, the `.wcs` layout parses, `-r 180` solves blind | Pass `fov_height` explicitly, fix the parser, or switch solver |

## Architecture

One installable package, `okapi`, all of it running on the Pi. Capture code depends only on
protocols (`MountProtocol`, `CameraProtocol`, `Clock`); hardware drivers and fakes both satisfy
them, so sequencing logic is tested without hardware. Pattern borrowed from the dugong repo.

```
src/okapi/
  clock.py         Clock, SystemClock, SimClock
  coords.py        Az/Alt <-> ICRS conversions (astropy), AltAzPosition, site helpers
  catalog.py       bright-star catalog, alignment-pair picker
  pointing.py      PointingModel: rotation fitted from two or more centered stars
  frames.py        RAW + JSON sidecar writer
  mount/           MountProtocol, FakeMount, SynScanMount (serial driver), FakeHandset, protocol codec
  camera/          CameraProtocol, FakeCamera, GPhotoCamera
  solve/           SolverProtocol, FakeSolver, AstapSolver (RAW decode, FITS, ASTAP CLI)
  sensors/         GpsProtocol, FakeGps, NMEA parser
  align/           center_on, align_on_star, AlignedMount
  session/         calibrate: the nightly sequence
  sim.py           SimRig: a whole simulated telescope for tests and dry runs
  cli.py           Typer entry point (`okapi calibrate --dry-run`)
```

Planned addition: `session/` night-long orchestration.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1 Foundation | uv project, clock, coordinates, protocols, fakes, CI | done |
| 2 Camera | gphoto2 driver with fake-runner tests; frame writer with metadata sidecar | done (driver untested on real T3i) |
| 3 Mount | SynScan serial driver, simulated handset, Pi-driven tracking, `okapi mount-info` | done against the simulator (ADR 0004); needs the real handset to confirm the items in the bring-up checklist |
| 4 Calibration | GPS time and position; push to mount; two-star alignment (named bright star: slew, solve, offset, nudge, sync) | done against fakes: catalog (35 stars), GPS/NMEA parser, Pi-side pointing model, two-star alignment with verification (`session/calibrate.py`); real GPS device binding and push-to-mount over serial pending |
| 5 Recenter | Solve-and-recenter loop | done (`align/center.py`). A spiral search was considered and dropped, see ADR 0003 |
| 6 Orchestration | `okapi night` runs the whole sequence; logs and resumable state | |
| 7 Bring-up | Hardware-in-the-loop tests (`-m hardware`), tune timing, verify A1-A5 | |

## Test strategy

- **Unit**: coordinate math against known answers (Polaris altitude is about the latitude; an
  object at Dec = latitude on the meridian is at the zenith); hypothesis round-trips.
- **Fakes**: `FakeMount` models finite slew rate, a hidden alignment error that `sync_radec`
  removes, and sidereal tracking; `SimClock` makes a night run instantly. Every behavior the
  real hardware has that the software must handle gets a fake behavior and a test.
- **Boundary**: gphoto2 is exercised through an injected runner, so command construction and
  parsing are tested without a camera.
- **Hardware**: tests marked `hardware` need the real devices and never run in CI.
- CI runs ruff, pyright and pytest on every push and PR.

## Parts list (verify before buying)

Compute and power: Raspberry Pi 5 (4 GB) or Pi 4, official power supply, 64 GB+ A1/A2 microSD
or NVMe hat, case with fan, 12 V battery (7-12 Ah) or jump pack, 12 V to 5 V USB-C buck converter
(5 A).
Mount link: the **RS-232 computer cable that came in the XT10g box** (check you still have it),
plus a USB-RS232 adapter (FTDI chip) for the Pi, and a gender changer or null-modem only if the
cable's connector ends need it. The port is true RS-232: never wire it straight to the Pi's GPIO
pins. Mount power is 12 V DC, 2.1 A, tip positive (manual specification).
Camera: T-ring for Canon EF/EOS and a 1.25"/2" T-adapter, DC coupler plus AC adapter (ACK-E8
class) so the camera does not run on its battery, a good USB cable.
Plate solving (software, free): ASTAP command-line build for Linux arm64 plus the D50 star database (fits fields from about 6 to 0.2 deg) in the same folder as the binary; `rawpy` (`uv sync --extra raw`) to decode the T3i's RAW frames.
Time and location: USB GPS puck (u-blox), optional DS3231 RTC.
Field use: Pi Wi-Fi hotspot or Ethernet cable, optional dew heater, red light, powered USB hub.

## Bring-up checklist (when the serial cable arrives)

Do these in order; stop and report at the first surprise.

1. Connect the USB-RS232 adapter to the Pi and the handset's serial jack. Find the device
   (`ls /dev/ttyUSB*`).
2. `uv run okapi mount-info /dev/ttyUSB0` (read-only). Note the firmware version, model, whether it
   is aligned, and what time and location the handset holds. Needs firmware 3.38+ (V3) for protocol
   v3.3 and 3.03+ for rate slews; update the firmware over the same port if it is older.
3. `OKAPI_TEST_PORT=/dev/ttyUSB0 uv run pytest -m hardware tests/test_synscan_hardware.py -s`
   (stage 1: read-only plus writing site and time).
4. With the area clear, the tube balanced and a hand on the power switch, add
   `OKAPI_TEST_MOTION=1` (stage 2) to find each axis's sign and try a small goto and back. If an
   axis is backwards, construct `SynScanMount(..., az_rate_sign=-1)` (or `alt_rate_sign`).
5. Point the telescope at a bright star, run `okapi calibrate` once the real drivers are wired in
   (not yet), and watch whether tracking holds. This answers A6.

