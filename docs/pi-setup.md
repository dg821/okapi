# Raspberry Pi setup

Written without access to a Pi: every command is standard but untested here. When a step
differs on your system, fix this file.

## 1. Operating system
- Flash **Raspberry Pi OS Lite, 64-bit** with Raspberry Pi Imager. In the Imager's settings set the
  hostname, a user, your Wi-Fi, and enable SSH. Lite avoids the desktop auto-mounting the camera.
- After first boot: `uname -m` must print `aarch64`. `sudo apt update && sudo apt full-upgrade`.

## 2. Packages
```
sudo apt install -y git gphoto2 chrony libgphoto2-dev
curl -LsSf https://astral.sh/uv/install.sh | sh   # then reopen the shell
git clone git@github.com:dg821/okapi.git && cd okapi
uv sync --extra raw
```

## 3. Permissions and stable names
- `sudo usermod -aG dialout $USER`, then log out and in. Serial ports need this group.
- Plug in devices one at a time and note `ls -l /dev/serial/by-id/`. Use those paths (they
  survive reboots and re-plugging) for `--mount-port` and `--gps-port`.
- Stop programs that grab ports: `sudo systemctl disable --now gpsd.socket gpsd` if installed,
  and `sudo apt remove modemmanager` if present.

## 4. ASTAP (plate solver)
- From hnsky.org/astap.htm, row "Linux 64 bit arm", download the Debian **.deb** (GTK2) and the
  **D50 (debian)** star database `.deb` (about 830 MB: download on the Mac and `scp` it over, or
  use `wget -c` inside tmux).
- `sudo apt install ./astap.deb ./d50_star_database.deb` (keep the `./`).
- The GUI program `/opt/astap/astap` needs desktop libraries and will not run on Lite. The package
  also installs **`/opt/astap/astap_cli`**, which is what okapi uses (its default is `astap_cli`):
  `sudo ln -sf /opt/astap/astap_cli /usr/local/bin/astap_cli`.
- Check: `astap_cli -h` prints usage. `uv run okapi doctor` reports the database.

## 5. Clock
Frame timestamps come from the Pi's system clock, and a Pi has no battery clock. Needs either
internet time or GPS time:
- With a network that reaches the internet, `chrony` (installed above) syncs automatically.
  Check `chronyc tracking`: "System time" within ~0.05 s.
- Without internet at the telescope, feed chrony from the GPS (a USB GPS gives NMEA time only,
  good to about a second, which is not good enough for LEO). Prefer a network.
- `okapi doctor --gps-port ...` compares the clock with the GPS time and warns above 0.5 s.

## 6. Camera
- T3i: mode dial **M**, auto power-off **off**, image quality RAW only, long-exposure noise
  reduction **off**. USB cable direct to the Pi.
- `gphoto2 --auto-detect` lists it; `gphoto2 --get-config shutterspeed` lists speeds.
- Bulb mode (exposures over 30 s) is untested: plan files cap exposures at 30 s.

## 7. First run
```
uv run okapi doctor --mount-port /dev/serial/by-id/<handset> --gps-port /dev/serial/by-id/<gps>
uv run okapi mount-info /dev/serial/by-id/<handset>
```
Then follow the bring-up checklist in PROJECT_PLAN.md. Use `tmux` so a dropped SSH session does
not stop a night: `tmux new -s night`, run `okapi night ...`, detach with Ctrl-B then D.

## 8. Measuring the shutter delay (do this once, on a clear night)
1. Get a fresh TLE for a bright LEO satellite (the ISS, from celestrak.org) and a pass time.
2. Run a plan that stares at the pass with short exposures (5-10 s) so each streak is a timed
   point, not a line. Keep the Pi clock synced.
3. `uv run okapi timing data/<night>/raw/*/*/*/*.obs.json --tle iss.tle --lat .. --lon ..`
   prints the shutter offset. A consistent value (scatter well under 0.1 s) is the delay to
   add to `FrameMetadata.start_time`; large scatter means a stale TLE or poor timing.
