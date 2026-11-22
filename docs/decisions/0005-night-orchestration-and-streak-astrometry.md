# 0005: Night orchestration, streak astrometry and orbit determination

Status: accepted (2026-11-21); exercised only against the simulator

## Context

The telescope exists to take 10-20 s sidereal stares and turn satellite streaks into orbits, for
GEO (a long streak inside the frame) and LEO (a line across it), unattended for hours.

## Decisions

1. **One loop, restartable.** `session/night.py` runs due frames in order, writes the RAW frame, its
   sidecar and its observation file, then updates `night_state.json` atomically. Re-running the same
   command resumes: finished frames are not repeated, and the orbit fit reads every saved
   observation, including those from earlier runs. The log is `night.log.jsonl`.
2. **Failures are classified, not swallowed.** An unsolvable frame is kept (the RAW is evidence)
   and counts as taken. Three consecutive solve failures trigger a recalibration if one is
   supplied, otherwise a clean abort with state saved. Camera and mount failures abort after three
   in a row. A target below the altitude limit is retried later and eventually given up on.
3. **Aim ahead, then check.** A GEO object drifts 15 arcsec/s against the stars, so the frame
   centre is aimed at where the target will be mid-exposure. Slews take unknown time, so the aim is
   re-checked (up to three passes) and re-slewed when the target has moved more than 30 arcsec.
4. **Pointing trim.** The pointing model is a rotation and leaves a residual (up to ~100 arcsec in
   simulation). After each solved frame the runner compares where the frame landed with where the
   target was predicted and shifts the next aim for that target by minus the error. Trims are not
   saved: a restart recalibrates, which fits a new model, and a trim measured against the old one
   would be wrong (this was caught by a test: the first resumed frame was 61 arcsec off).
5. **Streaks are fitted, not thresholded.** The blob extent is biased by stars touching it, so the
   endpoints come from fitting a PSF-blurred segment with star light masked. When a star sits on an
   end, that end is hidden and unmeasurable; the streak is flagged `contaminated` and its
   observation carries a 4 px uncertainty (instead of 0.5 px) so the orbit fit discounts it.
6. **Timed points versus lines.** A streak inside the frame gives a position at mid-exposure, a rate
   and a direction. A streak running off the frame gives only a line: no along-track position. The
   orbit fit uses lines for cross-track position and direction only.
7. **Orbit determination.** Two-body analytic propagation, weighted least squares on the state, seeded
   from a range grid and Gauss's method, with a penalty that keeps the orbit above the ground. It
   returns a formal covariance and a conditioning number, and warns when range is not constrained.
8. **Identification** compares predicted and measured position, direction and rate against the
   plan's satellites and any TLE catalog, and marks near-ties ambiguous. An unmatched streak is kept
   as `unidentified:<item>`; its orbit assumes all such streaks from that item are one object.

## Consequences and limits

- Two-body only: valid for arcs of hours. Multi-night tracking needs J2 and drag.
- A single short LEO pass does not determine an orbit (range is unconstrained); the fit says so.
- Shutter timing is the largest unmeasured error for LEO; `FrameMetadata.start_time` is the system
  clock just before the shutter command. Measure the real delay before trusting LEO orbits.
- None of this has run on a real camera, solver or mount. See the bring-up checklist.
