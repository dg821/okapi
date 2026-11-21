import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from okapi.coords import AltAzPosition, altaz_to_radec, radec_to_altaz
from okapi.mount.base import MountError, MountLimitError
from okapi.mount.fake import FakeMount
from okapi.mount.synscan import SynScanMount
from okapi.mount.synscan_protocol import ProtocolError, Version
from okapi.mount.synscan_sim import FakeHandset
from okapi.sim import make_synscan_rig


@pytest.fixture
def rig():
    r = make_synscan_rig(frame_rotation=(0 * u.deg,) * 3)
    r.mount.set_site(r.site)
    r.mount.set_time(r.clock.now())
    return r


def sky_target(rig, az=130.0, alt=50.0) -> SkyCoord:
    return altaz_to_radec(AltAzPosition(az * u.deg, alt * u.deg), rig.site, rig.clock.now())


def go_to_and_track(rig, target: SkyCoord) -> None:
    for _ in range(2):
        rig.mount.slew_radec(target)
        rig.mount.wait_until_settled(180 * u.s)
    rig.mount.start_tracking()


def drift_arcsec(rig, target: SkyCoord) -> float:
    return rig.physical.true_radec().separation(target).to_value(u.arcsec)


# -- handshake -----------------------------------------------------------------------------------
def test_handshake_reads_version_model_and_alignment():
    r = make_synscan_rig(firmware=Version(4, 37, 7), model=130, aligned=True)
    assert r.mount.info.version == Version(4, 37, 7)
    assert r.mount.info.model == 130
    assert r.mount.info.aligned
    assert "AZ GOTO" in r.mount.info.model_name


def test_old_firmware_and_wrong_model_produce_warnings():
    r = make_synscan_rig(firmware=Version(3, 20, 0), model=10)
    assert any("older" in w for w in r.mount.warnings)
    assert any("not an alt-az" in w for w in r.mount.warnings)


def test_a_silent_handset_is_an_error_not_a_hang():
    clock_rig = make_synscan_rig()
    deaf = FakeHandset(clock_rig.physical, clock_rig.clock, silent=True)
    with pytest.raises(MountError, match="did not answer"):
        SynScanMount(deaf, clock_rig.clock)


def test_garbage_echo_is_rejected():
    class WrongEcho(FakeHandset):
        def _handle(self, cmd: bytes) -> bytes:
            return b"y#" if cmd[:1] == b"K" else super()._handle(cmd)

    r = make_synscan_rig()
    with pytest.raises(ProtocolError, match="echo"):
        SynScanMount(WrongEcho(r.physical, r.clock), r.clock)


# -- setup ---------------------------------------------------------------------------------------
def test_site_and_time_reach_the_handset_and_read_back(rig):
    sent = [c for c in rig.handset.commands if c[:1] in (b"W", b"H")]
    assert [c[:1] for c in sent] == [b"W", b"H"]
    assert rig.physical.mount_time is not None


def test_location_with_a_35_degree_latitude_survives_the_hash_byte(rig):
    # 35 is the ASCII code of '#'; the binary reply must be read by length, not up to '#'.
    from okapi.coords import site_from_geodetic

    rig.mount.set_site(site_from_geodetic(35.0, -35.0, 100.0))
    assert rig.handset._location[0] == 35
    assert rig.handset._location[4] == 35


def test_location_mismatch_is_detected(rig):
    class BadLocation(FakeHandset):
        def _handle(self, cmd: bytes) -> bytes:
            return (
                bytes([1, 0, 0, 0, 2, 0, 0, 0]) + b"#" if cmd[:1] == b"w" else super()._handle(cmd)
            )

    r = make_synscan_rig()
    m = SynScanMount(BadLocation(r.physical, r.clock), r.clock)
    with pytest.raises(MountError, match="read-back"):
        m.set_site(r.site)


def test_time_mismatch_is_detected():
    class WrongClock(FakeHandset):
        def _get_time(self) -> bytes:
            return bytes([1, 2, 3, 4, 5, 6, 0, 0]) + b"#"  # nowhere near the time we set

    r = make_synscan_rig()
    m = SynScanMount(WrongClock(r.physical, r.clock), r.clock)
    with pytest.raises(MountError, match="time read-back mismatch"):
        m.set_time(r.clock.now())


# -- position and motion -------------------------------------------------------------------------
def test_position_round_trip_is_accurate(rig):
    rig.mount.slew_altaz(AltAzPosition(213.456 * u.deg, 61.789 * u.deg))
    rig.mount.wait_until_settled(60 * u.s)
    pos = rig.mount.get_altaz()
    assert pos.separation(AltAzPosition(213.456 * u.deg, 61.789 * u.deg)).to_value(u.arcsec) < 0.5


def test_negative_altitude_is_read_back_as_negative():
    r = make_synscan_rig(frame_rotation=(0 * u.deg,) * 3)
    r.physical._true = AltAzPosition(10 * u.deg, -3 * u.deg)
    assert r.mount.get_altaz().alt.to_value(u.deg) == pytest.approx(-3.0, abs=1e-3)


def test_altitude_limits_are_enforced_before_anything_is_sent(rig):
    before = len(rig.handset.commands)
    with pytest.raises(MountLimitError):
        rig.mount.slew_altaz(AltAzPosition(0 * u.deg, 2 * u.deg))
    assert len(rig.handset.commands) == before


def test_wait_until_settled_times_out(rig):
    rig.mount.slew_altaz(AltAzPosition(200 * u.deg, 80 * u.deg))
    with pytest.raises(MountError, match="timed out"):
        rig.mount.wait_until_settled(1 * u.s)


def test_abort_cancels_the_goto(rig):
    rig.mount.slew_altaz(AltAzPosition(200 * u.deg, 80 * u.deg))
    rig.clock.sleep(2)
    rig.mount.abort()
    assert not rig.mount.is_slewing()
    here = rig.mount.get_altaz()
    rig.clock.sleep(30)
    assert rig.mount.get_altaz().separation(here).to_value(u.arcsec) < 1e-3


def test_nudge_moves_by_the_requested_amount(rig):
    rig.mount.slew_altaz(AltAzPosition(100 * u.deg, 40 * u.deg))
    rig.mount.wait_until_settled(60 * u.s)
    rig.mount.nudge(0.1 * u.deg, -0.05 * u.deg)
    rig.mount.wait_until_settled(60 * u.s)
    pos = rig.mount.get_altaz()
    assert pos.az.to_value(u.deg) == pytest.approx(100.1, abs=1e-4)
    assert pos.alt.to_value(u.deg) == pytest.approx(39.95, abs=1e-4)


def test_radec_is_computed_from_altaz(rig):
    target = sky_target(rig)
    for _ in range(2):
        rig.mount.slew_radec(target)
        rig.mount.wait_until_settled(180 * u.s)
    assert rig.mount.get_radec().separation(target).to_value(u.arcsec) < 30


def test_radec_needs_a_site():
    r = make_synscan_rig()
    with pytest.raises(MountError, match="site"):
        r.mount.get_radec()


def test_sync_is_refused_with_an_explanation(rig):
    with pytest.raises(MountError, match="pointing model"):
        rig.mount.sync_radec(sky_target(rig))


# -- tracking ------------------------------------------------------------------------------------
def test_without_tracking_the_target_drifts_away(rig):
    target = sky_target(rig)
    go_to_and_track(rig, target)
    rig.mount.stop_tracking()
    rig.clock.sleep(60)
    assert drift_arcsec(rig, target) > 500  # sky moves 15 arcsec/s


def test_pi_driven_tracking_holds_a_star_with_one_second_updates(rig):
    go_to_and_track(rig, sky_target(rig))
    start = rig.physical.true_radec()  # measure from where tracking began, not from the target
    worst = 0.0
    for _ in range(120):
        rig.clock.sleep(1)
        rig.mount.track_step()
        worst = max(worst, rig.physical.true_radec().separation(start).to_value(u.arcsec))
    assert worst < 3.0  # two minutes of tracking


def test_open_loop_tracking_is_good_enough_for_one_exposure(rig):
    """In simulation the camera sleeps through a whole exposure with no track_step calls, so
    the rates set at the start carry it. Measured 0.5-1.9 arcsec over 20 s, mostly the 0.25
    arcsec/s rate resolution."""
    go_to_and_track(rig, sky_target(rig))
    start = rig.physical.true_radec()
    rig.clock.sleep(20)
    assert rig.physical.true_radec().separation(start).to_value(u.arcsec) < 3.0


def test_tracking_resumes_after_a_goto(rig):
    target = sky_target(rig)
    go_to_and_track(rig, target)
    rig.mount.nudge(0.2 * u.deg, 0.1 * u.deg)
    rig.mount.wait_until_settled(60 * u.s)  # resumes tracking at the new position
    new = rig.physical.true_radec()
    rig.clock.sleep(20)
    assert rig.physical.true_radec().separation(new).to_value(u.arcsec) < 5.0


def test_stop_tracking_zeroes_the_axis_rates(rig):
    go_to_and_track(rig, sky_target(rig))
    rig.mount.stop_tracking()
    assert not rig.mount.is_tracking
    here = rig.physical.true_altaz()
    rig.clock.sleep(30)
    assert rig.physical.true_altaz().separation(here).to_value(u.arcsec) < 1e-3
    assert rig.handset.commands[-1][:2] == b"P\x03"  # the last thing we sent zeroed an axis


def test_a_backwards_axis_is_caught_and_the_motors_stopped():
    r = make_synscan_rig(frame_rotation=(0 * u.deg,) * 3)
    r.mount._signs[16] = -1
    r.mount.set_site(r.site)
    target = sky_target(r)
    for _ in range(2):
        r.mount.slew_radec(target)
        r.mount.wait_until_settled(180 * u.s)
    r.mount.start_tracking()
    with pytest.raises(MountError, match="tracking lost"):
        for _ in range(300):
            r.clock.sleep(1)
            r.mount.track_step()
    assert not r.mount.is_tracking
    here = r.physical.true_altaz()
    r.clock.sleep(30)
    assert r.physical.true_altaz().separation(here).to_value(u.arcsec) < 1e-3  # stopped


def test_tracking_needs_firmware_that_can_do_rate_slews():
    r = make_synscan_rig(firmware=Version(3, 2, 0))
    r.mount.set_site(r.site)
    with pytest.raises(MountError, match="variable-rate"):
        r.mount.start_tracking()


def test_tracking_needs_a_site():
    r = make_synscan_rig()
    with pytest.raises(MountError, match="site"):
        r.mount.start_tracking()


def test_handset_tracking_is_switched_off_so_we_do_not_fight_it(rig):
    go_to_and_track(rig, sky_target(rig))
    assert b"T\x00" in rig.handset.commands
    assert rig.handset.tracking_mode == 0


def test_close_stops_tracking_and_closes_the_port(rig):
    go_to_and_track(rig, sky_target(rig))
    rig.mount.close()
    assert rig.handset.closed
    assert not rig.mount.is_tracking


def test_context_manager_closes():
    r = make_synscan_rig()
    with r.mount:
        pass
    assert r.handset.closed


def test_background_thread_keeps_tracking_in_real_time():
    """Real clock, real thread: the loop re-aims on its own and stops cleanly."""
    from okapi.clock import SystemClock

    clock = SystemClock()
    physical = FakeMount(clock, slew_rate=200 * u.deg / u.s)  # fast, so the test is not slow
    handset = FakeHandset(physical, clock)
    mount = SynScanMount(handset, clock, track_period=0.05 * u.s, background_tracking=True)
    from okapi.coords import site_from_geodetic

    mount.set_site(site_from_geodetic(40.0, -105.0, 1600.0))
    mount.slew_altaz(AltAzPosition(150 * u.deg, 50 * u.deg))
    mount.wait_until_settled(30 * u.s)
    mount.start_tracking()
    n_before = sum(1 for c in handset.commands if c[:2] == b"P\x03")
    import time

    time.sleep(0.4)
    mount.stop_tracking()
    assert sum(1 for c in handset.commands if c[:2] == b"P\x03") > n_before + 4
    assert mount.tracking_error is None
    assert not mount.is_tracking


def test_tracking_through_the_pointing_model_follows_a_tilted_mount():
    """A 1.3 degree tilt makes the star path in the mount's own frame differ from the true one.
    Measured over 120 s with 1 Hz updates: 0.1 arcsec with the model, tens of arcsec without."""
    from okapi.pointing import PointingModel, rotation_from_vector

    rot = (0.5 * u.deg, -0.8 * u.deg, 0.9 * u.deg)
    model = PointingModel(rotation_from_vector(rot))
    drift = {}
    for use_model in (True, False):
        r = make_synscan_rig(frame_rotation=rot)
        r.mount.set_site(r.site)
        if use_model:
            r.mount.set_pointing_model(model)
        target = sky_target(r, az=250.0, alt=35.0)
        # put the star exactly under the telescope using the true model, either way
        r.mount.slew_altaz(model.to_reported(radec_to_altaz(target, r.site, r.clock.now())))
        r.mount.wait_until_settled(120 * u.s)
        r.mount.start_tracking()
        start = r.physical.true_radec()
        for _ in range(120):
            r.clock.sleep(1)
            r.mount.track_step()
        drift[use_model] = r.physical.true_radec().separation(start).to_value(u.arcsec)
    assert drift[True] < 1.0
    assert drift[False] > 10.0


def test_setting_the_model_while_tracking_re_aims_the_track():
    from okapi.pointing import PointingModel, rotation_from_vector

    rot = (0.5 * u.deg, -0.8 * u.deg, 0.9 * u.deg)
    r = make_synscan_rig(frame_rotation=rot)
    r.mount.set_site(r.site)
    model = PointingModel(rotation_from_vector(rot))
    target = sky_target(r, az=250.0, alt=35.0)
    r.mount.slew_altaz(model.to_reported(radec_to_altaz(target, r.site, r.clock.now())))
    r.mount.wait_until_settled(120 * u.s)
    r.mount.start_tracking()  # no model yet: aimed along the wrong path
    r.mount.set_pointing_model(model)  # calibration finishes while tracking is running
    start = r.physical.true_radec()
    for _ in range(120):
        r.clock.sleep(1)
        r.mount.track_step()
    assert r.physical.true_radec().separation(start).to_value(u.arcsec) < 1.0


def test_report_of_a_handset_with_an_unset_clock_says_so_instead_of_failing():
    r = make_synscan_rig()
    report = r.mount.read_report()
    assert report.time_utc is None


def test_report_shows_the_time_once_it_has_been_set(rig):
    report = rig.mount.read_report()
    assert report.time_utc is not None
    assert abs((report.time_utc - rig.clock.now()).to_value(u.s)) < 2
    assert report.location[0] == pytest.approx(40.0, abs=1 / 3600)
