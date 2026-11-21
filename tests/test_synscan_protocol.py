"""Codec tests. Several cases are the worked examples printed in Sky-Watcher's protocol v3.3."""

import pytest
from astropy.time import Time
from hypothesis import given
from hypothesis import strategies as st

from okapi.mount.synscan_protocol import (
    ALT_AXIS,
    AZ_AXIS,
    ProtocolError,
    Version,
    command_length,
    decode_angle,
    decode_location,
    decode_rate_slew,
    decode_time,
    encode_angle,
    encode_location,
    encode_position_command,
    encode_rate_slew,
    encode_time,
    encode_time_fields,
    parse_position,
    parse_version,
    signed,
)


# -- angles --------------------------------------------------------------------------------------
def test_standard_example_from_the_document():
    # "34AB,12CE": DEC is 0x12CE = 4814/65536 of a turn = 26.4441 degrees
    assert decode_angle("12CE") == pytest.approx(26.4441, abs=1e-4)


def test_precise_example_from_the_document():
    # "12AB0500,40000500": AZM 0x12AB05 / 2**24 * 360 = 26.252 degrees; ALT is a quarter turn
    az, alt = parse_position(b"12AB0500,40000500#")
    assert az == pytest.approx(26.252, abs=1e-3)
    assert alt == pytest.approx(90.0, abs=1e-3)


def test_precision_of_the_precise_form_is_about_0_08_arcsec():
    one_unit = (
        decode_angle("00000100") * 3600
    )  # 1 / 2**24 of a turn (the last two digits are ignored)
    assert one_unit == pytest.approx(0.0772, abs=0.001)
    assert decode_angle("00000000") == 0.0
    assert decode_angle("000005AB") == decode_angle("00000500")  # last two digits are ignored


@given(st.floats(0, 359.9999))
def test_angle_round_trip_is_accurate_to_a_tenth_of_an_arcsecond(deg):
    assert abs(decode_angle(encode_angle(deg)) - deg) * 3600 < 0.08


def test_encode_wraps_negative_and_large_angles():
    assert encode_angle(-90) == encode_angle(270)
    assert encode_angle(360) == encode_angle(0) == "00000000"
    assert encode_angle(450) == encode_angle(90)


def test_signed_maps_large_fractions_to_negative_altitudes():
    assert signed(decode_angle(encode_angle(-5.0))) == pytest.approx(-5.0, abs=1e-4)
    assert signed(20.0) == 20.0


def test_position_command_format():
    cmd = encode_position_command("b", 26.252, 90.0)
    assert cmd.startswith(b"b") and cmd.count(b",") == 1 and len(cmd) == 18


@pytest.mark.parametrize("bad", ["", "XYZ", "12AB0", "12AB05000"])
def test_bad_angle_fields_rejected(bad):
    with pytest.raises(ProtocolError):
        decode_angle(bad)


@pytest.mark.parametrize("bad", [b"", b"12AB0500,40000500", b"12AB0500#", b"zzzzzzzz,00000000#"])
def test_bad_position_replies_rejected(bad):
    with pytest.raises(ProtocolError):
        parse_position(bad)


# -- version -------------------------------------------------------------------------------------
def test_version_example_from_the_document():
    # version 04.37.07 comes back as "042507#"
    assert parse_version(b"042507#") == Version(4, 37, 7)
    assert str(parse_version(b"042507#")) == "4.37.07"


def test_versions_order_correctly():
    assert Version(3, 38, 0) > Version(3, 3, 0) > Version(3, 0, 9)


@pytest.mark.parametrize("bad", [b"", b"042507", b"04250#", b"04X507#"])
def test_bad_version_rejected(bad):
    with pytest.raises(ProtocolError):
        parse_version(bad)


# -- location and time ---------------------------------------------------------------------------
def test_location_example_from_the_document():
    # 33 deg 50' 41" N, 118 deg 20' 17" W
    cmd = encode_location(33 + 50 / 60 + 41 / 3600, -(118 + 20 / 60 + 17 / 3600))
    assert cmd == b"W" + bytes([33, 50, 41, 0, 118, 20, 17, 1])


def test_location_round_trip_to_one_arcsecond():
    lat, lon = decode_location(encode_location(40.0123, -105.2745)[1:] + b"#")
    assert lat == pytest.approx(40.0123, abs=1 / 3600)
    assert lon == pytest.approx(-105.2745, abs=1 / 3600)


def test_location_hemispheres_and_longitude_wrap():
    cmd = encode_location(-33.5, 250.0)  # 250 E is 110 W
    assert cmd[4] == 1 and cmd[8] == 1  # south, west
    assert cmd[5] == 110


def test_location_seconds_carry_into_minutes():
    cmd = encode_location(10 + 59 / 60 + 59.7 / 3600, 0.0)  # 59.7 s rounds to 60 -> carry
    assert cmd[1:4] == bytes([11, 0, 0])


def test_location_rejects_bad_latitude():
    with pytest.raises(ValueError):
        encode_location(91.0, 0.0)


def test_time_example_from_the_document():
    # 3:26:00 PM on April 6, 2005, Eastern time (-5, so 256 - 5 = 251), daylight saving on
    cmd = encode_time_fields(15, 26, 0, 4, 6, 2005, -5, 1)
    assert cmd == b"H" + bytes([15, 26, 0, 4, 6, 5, 251, 1])


def test_utc_is_sent_with_zero_offset_and_no_dst():
    cmd = encode_time(Time("2026-06-21T04:30:15.2", scale="utc"))
    assert cmd == b"H" + bytes([4, 30, 15, 6, 21, 26, 0, 0])


def test_time_is_rounded_to_the_nearest_second():
    assert encode_time(Time("2026-06-21T04:30:15.7", scale="utc"))[3] == 16
    assert encode_time(Time("2026-06-21T23:59:59.8", scale="utc"))[1:4] == bytes([0, 0, 0])


def test_decode_time_converts_back_to_utc():
    reply = encode_time_fields(15, 26, 0, 4, 6, 2005, -5, 1)[1:] + b"#"
    t = decode_time(reply)
    assert t.offset_h == -5 and t.dst
    # April 6 is daylight saving time, so 15:26 local (offset -5, DST on) is 19:26 UTC. That the
    # handset reads the DST flag this way is our interpretation of the document, unverified; we
    # always send offset 0 and DST 0, so it does not affect us.
    assert t.to_utc().isot.startswith("2005-04-06T19:26:00")


# -- rates ---------------------------------------------------------------------------------------
def test_rate_example_from_the_document():
    # 150 arcsec/s: high = (150*4) \ 256 = 2, low = (150*4) mod 256 = 88
    cmd = encode_rate_slew(AZ_AXIS, 150.0)
    assert cmd == b"P" + bytes([3, 16, 6, 2, 88, 0, 0])


def test_negative_rates_and_alt_axis():
    assert encode_rate_slew(ALT_AXIS, -15.0) == b"P" + bytes([3, 17, 7, 0, 60, 0, 0])


def test_rate_resolution_is_a_quarter_arcsec():
    assert decode_rate_slew(encode_rate_slew(AZ_AXIS, 10.13)) == (AZ_AXIS, 10.25)
    assert decode_rate_slew(encode_rate_slew(AZ_AXIS, 10.12)) == (AZ_AXIS, 10.0)


@given(st.floats(-5000, 5000), st.sampled_from([AZ_AXIS, ALT_AXIS]))
def test_rate_round_trip_within_an_eighth(rate, axis):
    got_axis, got = decode_rate_slew(encode_rate_slew(axis, rate))
    assert got_axis == axis and abs(got - rate) <= 0.125 + 1e-9


def test_rate_limits_and_bad_axis():
    with pytest.raises(ProtocolError):
        encode_rate_slew(AZ_AXIS, 20000.0)
    with pytest.raises(ValueError):
        encode_rate_slew(18, 1.0)
    with pytest.raises(ProtocolError):
        decode_rate_slew(b"P\x02\x10\x24\x05\x00\x00\x00")  # a fixed-rate slew, not variable


def test_command_lengths_for_framing():
    assert command_length(ord("z")) == 1
    assert command_length(ord("K")) == 2
    assert command_length(ord("b")) == 18
    assert command_length(ord("P")) == 8
    assert command_length(ord("W")) == command_length(ord("H")) == 9
    assert command_length(ord("?")) == 1


def test_an_unset_handset_clock_is_a_protocol_error_not_a_crash():
    with pytest.raises(ProtocolError, match="invalid date"):
        decode_time(bytes(8) + b"#").to_utc()
