import astropy.units as u
import pytest
from astropy.coordinates import SkyCoord

from okapi.align.aligned_mount import AlignedMount
from okapi.coords import AltAzPosition, altaz_to_radec
from okapi.mount.base import MountError
from okapi.pointing import PointingModel, rotation_from_vector
from okapi.sim import make_sim_rig

ROTVEC = (0.5 * u.deg, -0.8 * u.deg, 0.9 * u.deg)


def target(rig, az=130.0, alt=50.0) -> SkyCoord:
    return altaz_to_radec(AltAzPosition(az * u.deg, alt * u.deg), rig.site, rig.clock.now())


def aligned(rig) -> AlignedMount:
    model = PointingModel(rotation_from_vector(ROTVEC))  # the true model of the fake's tilt
    return AlignedMount(rig.mount, model, site=rig.site, clock=rig.clock)


def settle_twice(m: AlignedMount, t: SkyCoord) -> None:
    for _ in range(2):
        m.slew_radec(t)
        m.wait_until_settled(180 * u.s)


def test_raw_mount_misses_by_the_tilt():
    rig = make_sim_rig()
    rig.mount.set_site(rig.site)
    t = target(rig)
    rig.mount.slew_radec(t)
    rig.mount.wait_until_settled(180 * u.s)
    # the unlevel base throws a raw slew off by about a degree
    assert rig.mount.true_radec().separation(t).to_value(u.deg) > 0.5


def test_aligned_mount_lands_on_target_despite_the_tilt():
    rig = make_sim_rig()
    rig.mount.set_site(rig.site)
    t = target(rig)
    settle_twice(aligned(rig), t)
    assert rig.mount.true_radec().separation(t).to_value(u.arcsec) < 10


def test_reads_are_translated_to_true_positions():
    rig = make_sim_rig()
    rig.mount.set_site(rig.site)
    m = aligned(rig)
    t = target(rig)
    settle_twice(m, t)
    assert m.get_radec().separation(rig.mount.true_radec()).to_value(u.arcsec) < 1.0


def test_sync_is_refused():
    rig = make_sim_rig()
    rig.mount.set_site(rig.site)
    with pytest.raises(MountError, match="refit"):
        aligned(rig).sync_radec(target(rig))


def test_passthrough_tracking_and_abort():
    rig = make_sim_rig()
    rig.mount.set_site(rig.site)
    m = aligned(rig)
    assert not m.is_tracking
    m.start_tracking()
    assert m.is_tracking and rig.mount.is_tracking
    m.stop_tracking()
    assert not rig.mount.is_tracking
    m.slew_altaz(AltAzPosition(10 * u.deg, 60 * u.deg))
    assert m.is_slewing()
    m.abort()
    assert not m.is_slewing()
