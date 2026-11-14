import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from okapi.coords import AltAzPosition, altaz_to_radec, radec_to_altaz
from okapi.mount.base import MountError, MountLimitError
from okapi.mount.fake import FakeMount


def target_in_sky(site, clock, az=120.0, alt=50.0) -> SkyCoord:
    return altaz_to_radec(AltAzPosition(az * u.deg, alt * u.deg), site, clock.now())


@pytest.fixture
def mount(clock, site):
    m = FakeMount(clock)
    m.set_site(site)
    return m


def test_slew_takes_time_proportional_to_distance(mount, clock):
    mount.slew_altaz(AltAzPosition(0 * u.deg, 60 * u.deg))  # 30 deg of altitude at 3 deg/s
    assert mount.is_slewing()
    clock.sleep(5)
    assert mount.is_slewing()
    clock.sleep(6)
    assert not mount.is_slewing()
    assert mount.get_altaz().separation(AltAzPosition(0 * u.deg, 60 * u.deg)).to_value(u.deg) < 1e-6


def test_azimuth_slew_takes_the_short_way_round(mount, clock):
    mount.slew_altaz(AltAzPosition(350 * u.deg, 30 * u.deg))  # 10 deg west of north, not 350
    mount.wait_until_settled(5 * u.s)
    assert clock.monotonic() < 5


def test_wait_until_settled_times_out(mount):
    mount.slew_altaz(AltAzPosition(180 * u.deg, 80 * u.deg))
    with pytest.raises(MountError, match="timed out"):
        mount.wait_until_settled(1 * u.s)


@pytest.mark.parametrize("alt", [2.0, 89.5])
def test_altitude_limits_rejected(mount, alt):
    with pytest.raises(MountLimitError):
        mount.slew_altaz(AltAzPosition(0 * u.deg, alt * u.deg))
    assert not mount.is_slewing()


def test_radec_slew_requires_site(clock):
    m = FakeMount(clock)
    with pytest.raises(MountError, match="site"):
        m.slew_radec(SkyCoord(ra=10 * u.deg, dec=20 * u.deg))


def test_radec_slew_lands_on_target_even_though_sky_moves(mount, clock, site):
    star = target_in_sky(site, clock, az=200.0, alt=45.0)
    mount.slew_radec(star)
    mount.wait_until_settled(120 * u.s)
    # wait_until_settled polls every 0.1 s and the untracked sky moves 15 arcsec/s, so up to
    # ~1.5 arcsec of the miss is the polling granularity, not the slew.
    assert mount.get_radec().separation(star).to_value(u.arcsec) < 2.0


def test_tracking_holds_radec_while_altaz_changes(mount, clock, site):
    star = target_in_sky(site, clock)
    mount.slew_radec(star)
    mount.wait_until_settled(120 * u.s)
    mount.start_tracking()
    az0 = mount.get_altaz()
    clock.sleep(600)
    assert mount.get_altaz().separation(az0).to_value(u.deg) > 1.0  # the sky moved
    assert mount.get_radec().separation(star).to_value(u.arcsec) < 1.0  # we followed it


def test_without_tracking_the_target_drifts_away(mount, clock, site):
    star = target_in_sky(site, clock)
    mount.slew_radec(star)
    mount.wait_until_settled(120 * u.s)
    clock.sleep(600)  # 10 minutes: ~2.5 deg of sky rotation
    drift = mount.get_radec().separation(star).to_value(u.deg)
    assert 2.0 < drift < 3.0


def test_stop_tracking_freezes_position(mount, clock, site):
    mount.slew_radec(target_in_sky(site, clock))
    mount.wait_until_settled(120 * u.s)
    mount.start_tracking()
    mount.stop_tracking()
    assert not mount.is_tracking
    before = mount.get_altaz()
    clock.sleep(300)
    assert mount.get_altaz().separation(before).to_value(u.arcsec) < 1e-3


def test_abort_stops_mid_slew(mount, clock):
    mount.slew_altaz(AltAzPosition(0 * u.deg, 80 * u.deg))
    clock.sleep(3)
    mount.abort()
    assert not mount.is_slewing()
    here = mount.get_altaz()
    clock.sleep(30)
    assert mount.get_altaz().separation(here).to_value(u.arcsec) < 1e-3
    assert 30 < here.alt.to_value(u.deg) < 80


def test_alignment_error_makes_slew_miss_until_synced(clock, site):
    m = FakeMount(clock, alignment_error=(0.6 * u.deg, -0.4 * u.deg))
    m.set_site(site)
    star = target_in_sky(site, clock)
    m.slew_radec(star)
    m.wait_until_settled(120 * u.s)
    # The mount *reports* it is on the star, but truly it is ~0.7 deg away.
    true_sep = radec_to_altaz(star, site, clock.now()).separation(m._true_altaz())
    assert 0.5 < true_sep.to_value(u.deg) < 0.8
    # One sync on the star it is truly looking at removes the offset for the next slew.
    seen = altaz_to_radec(m._true_altaz(), site, clock.now())
    m.sync_radec(seen)
    m.slew_radec(star)
    m.wait_until_settled(120 * u.s)
    true_sep = radec_to_altaz(star, site, clock.now()).separation(m._true_altaz())
    assert true_sep.to_value(u.arcsec) < 2.0


def test_nudge_moves_by_requested_amount(mount, clock):
    mount.slew_altaz(AltAzPosition(100 * u.deg, 40 * u.deg))
    mount.wait_until_settled(60 * u.s)
    before = mount.get_altaz()
    mount.nudge(0.1 * u.deg, -0.05 * u.deg)
    mount.wait_until_settled(5 * u.s)
    after = mount.get_altaz()
    assert (after.az - before.az).to_value(u.deg) == pytest.approx(0.1, abs=1e-6)
    assert (after.alt - before.alt).to_value(u.deg) == pytest.approx(-0.05, abs=1e-6)


def test_set_time_is_recorded(mount, clock):
    mount.set_time(clock.now())
    assert mount.mount_time == clock.now()
