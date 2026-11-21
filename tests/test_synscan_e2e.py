"""The whole nightly calibration, driven through the SynScan serial driver and the simulated
handset: protocol bytes on the wire, tilt in the mount, plate solves in the loop."""

import astropy.units as u
import numpy as np
import pytest

from okapi.coords import AltAzPosition, altaz_to_radec
from okapi.mount.synscan_protocol import Version
from okapi.pointing import rotation_from_vector
from okapi.session.calibrate import CalibrationError, calibrate
from okapi.sim import SynScanRig, make_synscan_rig

TILT = (0.5 * u.deg, -0.8 * u.deg, 0.9 * u.deg)


def run(rig: SynScanRig, **kw):
    return calibrate(
        mount=rig.mount,
        camera=rig.camera,
        solver=rig.solver,
        gps=rig.gps,
        clock=rig.clock,
        settings=rig.settings,
        **kw,
    )


def test_calibration_recovers_the_tilt_through_the_serial_stack():
    rig = make_synscan_rig(frame_rotation=TILT)
    rep = run(rig)
    assert rep.warnings == ()
    assert rep.model_rms.to_value(u.arcsec) < 15
    assert rep.verification_error.to_value(u.arcsec) < 30
    # the fitted model is the tilt we built into the simulated telescope
    assert np.allclose(rep.model.rotation, rotation_from_vector(TILT), atol=2e-4)


def test_site_and_gps_time_were_sent_as_protocol_bytes():
    rig = make_synscan_rig(frame_rotation=TILT)
    run(rig)
    sent = {c[:1] for c in rig.handset.commands}
    assert {b"W", b"H", b"b", b"P", b"T"} <= sent
    # handset tracking was switched off, and never switched on (the Pi drives the axes)
    assert [c for c in rig.handset.commands if c[:1] == b"T"] == [b"T\x00"] * len(
        [c for c in rig.handset.commands if c[:1] == b"T"]
    )
    assert rig.handset.tracking_mode == 0


def test_after_calibration_the_aligned_mount_lands_on_new_targets_and_tracks():
    rig = make_synscan_rig(frame_rotation=TILT)
    rep = run(rig)
    target = altaz_to_radec(AltAzPosition(250 * u.deg, 45 * u.deg), rig.site, rig.clock.now())
    for _ in range(2):
        rep.mount.slew_radec(target)
        rep.mount.wait_until_settled(180 * u.s)
    rep.mount.start_tracking()
    start = rig.physical.true_radec()
    assert start.separation(target).to_value(u.arcsec) < 15
    rig.clock.sleep(20)  # one exposure's worth, no re-aiming
    assert rig.physical.true_radec().separation(start).to_value(u.arcsec) < 3


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_other_noise_seeds_calibrate_too(seed):
    rep = run(make_synscan_rig(frame_rotation=TILT, seed=seed, solver_noise=4 * u.arcsec))
    assert rep.verification_error.to_value(u.arcsec) < 40


def test_firmware_too_old_for_rate_slews_fails_with_a_clear_reason():
    rig = make_synscan_rig(frame_rotation=TILT, firmware=Version(3, 2, 0))
    with pytest.raises(CalibrationError, match="update the hand control firmware"):
        run(rig, verify=False)
