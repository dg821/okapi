import astropy.units as u
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from okapi.coords import AltAzPosition
from okapi.pointing import PointingModel, PointingModelError, rotation_from_vector


def pos(az, alt):
    return AltAzPosition(az * u.deg, alt * u.deg)


ROT = rotation_from_vector((1.0 * u.deg, -0.5 * u.deg, 0.8 * u.deg))


def pair_through(rot, true):
    return true, PointingModel(rot).to_reported(true)


def test_zero_rotation_is_identity():
    assert np.allclose(rotation_from_vector((0 * u.deg,) * 3), np.eye(3))


def test_quarter_turn_about_z_moves_north_to_east():
    r = rotation_from_vector((0 * u.deg, 0 * u.deg, 90 * u.deg))
    assert np.allclose(r @ np.array([1.0, 0, 0]), [0, 1, 0])


def test_rotation_is_proper():
    assert np.linalg.det(ROT) == pytest.approx(1.0)
    assert np.allclose(ROT @ ROT.T, np.eye(3))


def test_fit_recovers_rotation_exactly_from_two_stars():
    pairs = [pair_through(ROT, pos(30, 40)), pair_through(ROT, pos(200, 55))]
    model = PointingModel.fit(pairs)
    assert np.allclose(model.rotation, ROT, atol=1e-12)
    assert model.rms(pairs).to_value(u.arcsec) < 1e-6


def test_fit_with_three_noisy_stars_is_close_and_proper():
    rng = np.random.default_rng(1)
    pairs = []
    for az, alt in [(30, 40), (200, 55), (120, 25)]:
        true, rep = pair_through(ROT, pos(az, alt))
        noisy = AltAzPosition(
            rep.az + rng.normal(0, 5) * u.arcsec, rep.alt + rng.normal(0, 5) * u.arcsec
        )
        pairs.append((true, noisy))
    model = PointingModel.fit(pairs)
    assert np.linalg.det(model.rotation) == pytest.approx(1.0)
    assert model.rms(pairs).to_value(u.arcsec) < 15
    check = pos(75, 60)
    err = model.to_reported(check).separation(PointingModel(ROT).to_reported(check))
    assert err.to_value(u.arcsec) < 20


def test_fit_needs_two_stars():
    with pytest.raises(PointingModelError, match="two"):
        PointingModel.fit([pair_through(ROT, pos(30, 40))])


def test_fit_rejects_close_stars():
    pairs = [pair_through(ROT, pos(30, 40)), pair_through(ROT, pos(33, 41))]
    with pytest.raises(PointingModelError, match="apart"):
        PointingModel.fit(pairs)


def test_identity_model_changes_nothing():
    p = pos(123, 45)
    assert PointingModel.identity().to_reported(p).separation(p).to_value(u.arcsec) < 1e-6


@given(az=st.floats(0, 359), alt=st.floats(-80, 85))
@settings(max_examples=40, deadline=None)
def test_to_true_inverts_to_reported(az, alt):
    m = PointingModel(ROT)
    p = pos(az, alt)
    assert m.to_true(m.to_reported(p)).separation(p).to_value(u.arcsec) < 1e-4
