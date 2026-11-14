# 0003: plate solving with the ASTAP command-line program

Status: accepted (2026-11-14)

## Decisions

1. **ASTAP, run as a subprocess.** Fast on small fields, has a Linux arm64 build, and works from
   a command line like the gphoto2 driver, so the process boundary is faked the same way in tests
   (`Runner`). Alternative considered: astrometry.net, heavier to install and slower on a Pi.
2. **We decode RAW ourselves.** ASTAP's command-line source lists FITS, TIFF, PNG and JPG as
   inputs and says nothing about CR2. `rawpy` (libraw) reads the mosaic; we sum 2x2 blocks into a
   mono image (clean luminance, no demosaic, no color pattern) and write FITS. `rawpy` is an
   optional extra so a laptop checkout does not need libraw; CI installs it.
3. **Field of view from FITS header keywords by default** (`FOCALLEN`, `XPIXSZ`, `-fov 0`),
   because ASTAP's help describes `-fov` as a "diameter" without saying whether that is image
   height or diagonal, and we could not confirm it. `fov_height=` overrides it once known.
4. **Hint first, blind second.** The mount's reported position goes in as `-ra` (hours) and
   `-spd` (Dec + 90 deg, never negative) with a 10 deg radius; if that finds nothing we retry
   blind (`-r 180`). Exit codes 2, 16, 32, 33 (too few stars, unreadable image, no or unreadable
   star database) are not retried because searching wider cannot help.
5. **Image center from the WCS, not CRPIX.** The `.wcs` reference pixel need not be the center.
   The solver converts the true center pixel through the WCS.

## Not verified (needs ASTAP on the Pi)

The header-keyword field of view, the `.wcs` layout (parsed both as 80-column cards and as
newline-separated text), and blind solving with `-r 180` and no `-ra/-spd`. The exit-code
meanings come from comments in ASTAP's source; only 0, 1 and 16 are set in the file we could read.
`tests/test_astap_hardware.py` checks these on real hardware.
