# 0002: Pi-side pointing model and the calibration sequence

Status: accepted (2026-11-14)

## Context

We do not yet know whether the 2013 XT10g handset accepts an alignment from the Pi
(PROJECT_PLAN assumption A2). The Dobsonian's main pointing error is a mis-leveled or
mis-oriented base, which makes the mount's reported frame a *rotation* of the true horizon frame.

## Decisions

1. **The Pi owns the pointing model.** `PointingModel` is a 3-D rotation fitted (Kabsch/SVD)
   from `(true, reported)` pairs, where "true" comes from the sky via plate solving and
   "reported" from the mount. `AlignedMount` wraps any `MountProtocol` and translates positions,
   so sequencing code is unchanged. The handset's own alignment is left untouched; the model
   composes with it as long as it does not change during the night. `sync_radec` on the wrapper is
   refused for that reason.
2. **Rotation only.** An altitude-encoder zero error is not a rotation. A test shows a 4 deg
   altitude offset is flagged by the verification step instead of being silently mis-modeled.
3. **Verify every calibration.** After the fit, slew to a third star and measure the miss with
   one solve; warn above 2 arcmin. The slew is done twice because our slew-time estimate for
   RA/Dec targets is rough (we measured 67 arcsec miss with a wrong assumed rate versus 2-6
   arcsec with a second short hop).
4. **Failures move on, they do not retry forever.** A star that cannot be centered is skipped
   in favor of the next candidate, up to 4 attempts. A GPS failure aborts unless the caller gave
   an explicit fallback site (then the system clock is used and a warning is recorded).
5. **GPS time goes to the mount; the system clock disagreement is reported, not fixed.** The
   Pi's own clock should be disciplined by chrony/gpsd; calibration warns if it is off by more
   than 2 s.
6. **Star catalog is hand-entered.** Positions were cross-checked against CDS Sesame (all
   within 1.6 arcsec); proper motions are not independently verified.

## Not decided yet

How to push an alignment into the handset, if its protocol allows (A2). If it does, the Pi model
can become a cross-check rather than the primary correction.
