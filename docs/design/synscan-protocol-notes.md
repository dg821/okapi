# SynScan hand-control serial protocol: working notes

Source: Sky-Watcher "SynScan Serial Communication Protocol, Version 3.3" (SSAP-F-160414V1-EN,
https://inter-static.skywatcher.com/downloads/synscanserialcommunicationprotocol_version33.pdf).
These are our notes, not a copy; re-read the PDF before changing driver behavior.
Our handset is an Orion-branded **SynScan AZ** (photographed 2026-11-20). Firmware version is
unknown: the document says it applies to V3 firmware 3.38 or later (V4: 4.38.06 or later), and
individual commands list older minimums (sync from 3.37, variable-rate slew from 3.03).

## Link

9600 baud, 8 data bits, no parity, 1 stop bit, on the RS-232 port at the base of the handset.
This is true RS-232 levels: use a USB-RS232 adapter, never the Pi's GPIO UART pins directly.
Wait up to 5 s for a reply. Never send commands blindly without reading the response, or
replies get matched to the wrong command. Every reply ends with `#`.

## Commands we expect to use

| Purpose | Send | Reply |
|---|---|---|
| Echo / link check | `K` + 1 char | that char + `#` |
| Firmware version | `V` | 6 hex digits + `#` (e.g. `042507#` = 4.37.07) |
| Mount model | `m` | 1 byte + `#` (128-143 AZ GOTO, 144-159 DOB GOTO) |
| Alignment done? | `J` | byte 1 or 0 + `#` |
| Get Az/Alt (precise) | `z` | `AAAAAA00,LLLLLL00#` (24-bit fraction of a turn, hex) |
| Go to Az/Alt (precise) | `b` + same format | `#` |
| Get / go to RA/Dec | `e` / `r` | same format; **needs alignment**, J2000 |
| Sync RA/Dec | `s` + RA/Dec | `#` (handset 3.37+) |
| Goto in progress? | `L` | ASCII `0` or `1` + `#` |
| Cancel goto | `M` | `#` |
| Tracking mode | `t` get, `T`+chr(mode) set | 0 off, 1 alt/az, 2 equatorial, 3 PEC |
| Set / get location | `W` / `w` | 8 bytes: lat deg, min, sec, N=0/S=1, lon deg, min, sec, E=0/W=1 |
| Set / get time | `H` / `h` | 8 bytes: h, m, s, month, day, 2-digit year, GMT offset (256-n if negative), DST flag |
| Variable-rate slew | `P` + 7 bytes | `#` (rate in quarter arcsec/s; per axis, either direction) |

Values are binary bytes for location and time (not ASCII). Angles are fractions of a revolution:
`degrees = int(hex24) / 2**24 * 360` for precise commands (the last two hex digits are ignored).

## What this means for okapi

1. **Un-aligned Az/Alt is relative to where the mount was powered on**, and RA/Dec values and RA/Dec
   goto are meaningless until the handset is aligned. Our Pi-side pointing model (ADR 0002) fits
   this exactly: drive everything through Az/Alt (`z`, `b`) and let the model map reported
   positions to true ones. We never need the handset's own RA/Dec.
2. **No command starts the handset's alignment procedure.** Sync (`s`) exists, but the document
   only describes it for an aligned handset, so we cannot rely on it to automate the handset's
   alignment. This resolves assumption A2 the other way: keep the Pi model as the primary
   correction.
3. **Tracking is the open question (A6).** Mode 1 (alt/az tracking) probably needs the handset to
   know its orientation on the sky, which an unaligned handset does not. Two options: require one
   handset keypad alignment per night and use mode 1, or turn handset tracking off and have the
   Pi drive both axes with variable-rate slews (`P`) at rates computed from the pointing model
   each second. The second keeps everything automated; the first is simpler. Needs a test on the
   real mount.
4. **Time is local time plus an offset.** We send UTC with offset 0 and DST 0 and read it back
   with `h` to confirm. Location has 1-arcsecond resolution.
5. **Altitude below the horizon** is presumably encoded as a negative fraction (two's complement);
   the document does not say. Decode values above 180 deg as negative, and confirm on hardware.

## Physical connections (Orion XTg manual IN 388 Rev. D 06/13, covers XT8g/XT10g/XT12g)

- The base electronics sit in the **altitude motor housing**. It has a jack labeled **AZ MOTOR**
  (8-pin RJ-45, flat cable to the azimuth motor) and a jack labeled **HC** (the smaller RJ-12).
- The **coiled hand-controller cable** has a wide RJ-45 plug (into the handset) and a smaller
  RJ-12 plug (into **HC** on the altitude motor housing).
- So on the handset, the wide RJ-45 jack is the coiled cable to the mount, and the other, smaller
  jack is the **RS-232 PC port**. The box includes an **"RS-232 computer cable"** for it. This
  matches the photographs taken on 2026-11-20 (cable in the left jack, a free smaller middle jack,
  and a DC jack).
- Mount power: **12 V DC, 2.1 A, tip positive** (the manual's specification).
- The handset's own firmware version is not in the manual; read it with `okapi mount-info`.

Still unverified: the exact connector ends and pinout of the supplied RS-232 cable (likely RJ-12
to DB9), and whether a USB-RS232 adapter plugs straight onto it or needs a gender changer or
null-modem. If `okapi mount-info` gets no reply, check that first.

## The alternative: skip the handset and talk to the motor controller

Sky-Watcher publishes the motor controller's command set
(https://inter-static.skywatcher.com/downloads/skywatcher_motor_controller_command_set.pdf).
It is a simple ASCII protocol (`:` command, `=` or `!` response, carriage return), with positions
in encoder counts offset by 0x800000, goto and constant-speed modes, and timer-preset speed
control. Section 6 of that document is the catch:

- 9600 baud 8N1, but **5 V or 3.3 V logic levels, not RS-232**.
- "On most the Alt/Az mount, TX and RX lines are connected together, and there is another line
  (Drop) to indicate that the TX/RX bus is busy": a half-duplex single-wire bus where the master
  must pull the Drop line low while sending and until the reply arrives or a timeout passes.
- The pinout of the **HC** RJ-12 jack is not in the documents we have, and that jack may carry
  supply voltage to power the handset. Plugging a Pi into it without knowing the pinout can
  destroy the Pi.
- The handset would have to be unplugged (one master on the bus), and the handset's altitude
  limits and goto safety would be gone; we would own them.

What it would buy: encoder-count positions with no firmware-version or alignment dependence,
exact speed control for tracking instead of 0.25 arcsec/s steps, and no 5 s handset latency.
Decision: not now, see ADR 0004.

