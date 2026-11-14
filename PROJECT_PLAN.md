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
| A1 | The 2013 handset is a SynScan-family controller with an RJ-12 RS-232 port, 9600 8N1 | Read the handset serial number/version (Utility menu) and port labels; compare to the SynScan serial protocol docs | Mount driver (Phase 3) changes; everything above it does not |
| A2 | The protocol allows sync/alignment from the Pi | Try it on the real mount | Keep our own pointing model on the Pi (Phase 4 fallback) |
| A3 | gphoto2 bulb timing works on the T3i | `gphoto2 --get-config shutterspeed`; run the bulb sequence by hand | Keep exposures at or under 30 s, or add a bulb remote cable via a GPIO relay |
| A4 | T3i reaches focus at the Dobsonian's prime focus | Try it at the focuser | Need a Barlow or other optics; changes the field of view |
| A5 | Plate solving works on the Pi (ASTAP or astrometry.net) at ~1.1 x 0.7 deg | Solve test frames | Fall back to a larger field or a faster solver |

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
  mount/           MountProtocol, FakeMount, (later) serial driver
  camera/          CameraProtocol, FakeCamera, GPhotoCamera
  solve/           SolverProtocol, FakeSolver, (later) ASTAP wrapper
  sensors/         GpsProtocol, FakeGps, NMEA parser
  align/           center_on, align_on_star, AlignedMount
  session/         calibrate: the nightly sequence
  sim.py           SimRig: a whole simulated telescope for tests and dry runs
  cli.py           Typer entry point (`okapi calibrate --dry-run`)
```

Planned additions: spiral search for failed solves, and `session/` night-long orchestration.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 1 Foundation | uv project, clock, coordinates, protocols, fakes, CI | done |
| 2 Camera | gphoto2 driver with fake-runner tests; frame writer with metadata sidecar | done (driver untested on real T3i) |
| 3 Mount | Serial driver, protocol-level simulator, slew/track/sync over serial | blocked on A1 |
| 4 Calibration | GPS time and position; push to mount; two-star alignment (named bright star: slew, solve, offset, nudge, sync) | done against fakes: catalog (35 stars), GPS/NMEA parser, Pi-side pointing model, two-star alignment with verification (`session/calibrate.py`); real GPS device binding and push-to-mount over serial pending |
| 5 Search | Spiral search and solve-and-recenter loop | recenter loop done (`align/center.py`); spiral search for failed solves next |
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
Mount link: USB-RS232 adapter (FTDI chip) and an RJ-12 cable (or the Sky-Watcher SynScan USB
adapter if A1 holds).
Camera: T-ring for Canon EF/EOS and a 1.25"/2" T-adapter, DC coupler plus AC adapter (ACK-E8
class) so the camera does not run on its battery, a good USB cable.
Time and location: USB GPS puck (u-blox), optional DS3231 RTC.
Field use: Pi Wi-Fi hotspot or Ethernet cable, optional dew heater, red light, powered USB hub.
