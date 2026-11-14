# 0001: single package; plate-solve-based alignment

Status: accepted (2026-11-14)

## Decisions

1. **One package, not a uv workspace.** Unlike dugong, all okapi code runs on the Pi with the
   same dependencies. A workspace would add per-package config for no gain; split later if an
   offline analysis tool with different dependencies appears.
2. **Align by plate solving, not star-brightness matching.** To align on a named star, slew to
   it, solve a short exposure to find the true pointing, nudge until the star is at the frame
   center, then sync the mount. Brightness estimation would be fragile and is unnecessary.
3. **Pointing model on the Pi as fallback.** If the handset protocol cannot accept sync or
   alignment from the Pi (assumption A2), the Pi corrects every slew with its own offset model.
4. **No refraction, degraded IERS accuracy.** The Pi is offline in the field, so astropy's
   Earth-orientation download is disabled. The error is a few arcseconds, below mount accuracy.
