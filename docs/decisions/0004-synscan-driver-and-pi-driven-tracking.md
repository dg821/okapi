# 0004: SynScan driver, Az/Alt only, Pi-driven tracking

Status: accepted (2026-11-20); the tracking approach is unverified on the real mount (A6)

## Context

The handset is an Orion SynScan AZ. Sky-Watcher's serial protocol v3.3 gives Az/Alt goto and
readout, location, time, tracking mode and variable-rate slews, but nothing to start the
handset's alignment, and RA/Dec is meaningless until aligned (docs/design/synscan-protocol-notes.md).

## Decisions

1. **Az/Alt only.** RA/Dec is computed in the driver from Az/Alt, the site and the clock. The
   Pi-side pointing model (ADR 0002) maps the handset's frame to the true one.
2. **The Pi drives tracking.** Handset tracking is switched off (`T 0`). Every second the Pi
   reads the position and sends a variable-rate slew on each axis aimed at where the target will
   be one second later, through the pointing model. Closed loop, so the 0.25 arcsec/s rate
   resolution and model errors do not accumulate. Chosen over requiring a manual keypad alignment
   each night, which would defeat the point of automating the night.
3. **Why the model matters for tracking.** Measured in simulation over 120 s at 1 Hz: 0.1 arcsec
   with the model, 5-27 arcsec without it for a 1.3 deg tilt, 60-85 arcsec for a 3.7 deg tilt.
   Open loop over a 20 s exposure: 0.5-1.9 arcsec with the model, up to 15 arcsec without.
4. **Safety.** The handset has no dead-man timer: axis rates keep running if the Pi dies. So the
   driver (a) stops the motors and raises if the measured position is more than 0.25 deg from the
   aim ("tracking lost"), which also catches a backwards axis sign, (b) refuses rates above
   1 deg/s, (c) zeroes the rates on `stop_tracking`, `close` and on leaving a `with` block, and
   (d) never sends a movement command without waiting for its acknowledgement.
5. **Binary replies are read by length.** A byte of value 35 is `#`; a 35 degree latitude would
   truncate a read-to-`#`. Tested by deliberately breaking it and watching the test fail.
6. **Read-back checks.** Location and time are read back after being set and compared.
7. **First contact is read-only.** `okapi mount-info PORT` sends only queries.

## Considered: bypass the handset and drive the motor controller directly (2026-11-21)

Not chosen now. The handset's RS-232 port is documented, level-safe through an ordinary
USB-RS232 adapter, the cable came in the box, and the driver for it is written and tested. The
direct route needs a custom 3.3/5 V half-duplex interface with a Drop line, an unknown pinout on
a jack that may carry supply voltage, a new driver, and our own altitude limits and safety
(docs/design/synscan-protocol-notes.md). Revisit if bring-up shows the handset route cannot do
what we need: firmware too old for rate slews, handset tracking that cannot be switched off, or
latency that makes the tracking loop unreliable. The `MountProtocol` boundary means a motor-
controller driver could replace `SynScanMount` without touching anything above it.

## Not verified (needs the real handset; see PROJECT_PLAN bring-up checklist)

Which serial jack; the firmware version and whether it accepts variable-rate slews; the sign of
each axis; that handset tracking really is off after `T 0` and does not fight our rates; the
DST/offset convention of the time command (we send UTC with offset 0, so it should not matter);
how negative altitudes are encoded; whether the handset's power-on Az/Alt frame is close enough
to north and level for the first goto to land in the right part of the sky.
