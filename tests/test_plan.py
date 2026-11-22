import json

import astropy.units as u
import pytest
from tests.tle_helpers import make_tle

from okapi.session.plan import PlanError, load_plan, parse_plan
from okapi.targets import FixedTarget, GeoSlotTarget, TleTarget

GOOD = {
    "min_altitude_deg": 20,
    "targets": [
        {
            "name": "GEO-100W",
            "kind": "geo",
            "longitude_deg": -100,
            "count": 5,
            "interval_s": 1800,
            "exposure_s": 20,
            "iso": 3200,
            "start_offset_s": 60,
        },
        {"name": "Vega", "kind": "star", "star": "Vega"},
        {"name": "M31", "kind": "fixed", "ra_deg": 10.68, "dec_deg": 41.27, "exposure_s": 15},
    ],
}


def with_target(**changes):
    t = dict(GOOD["targets"][0])
    t.update(changes)
    return {"targets": [t]}


def test_a_complete_plan_parses_with_every_field():
    plan = parse_plan(GOOD)
    geo, vega, m31 = plan.items
    assert isinstance(geo.target, GeoSlotTarget) and geo.target.is_satellite
    assert (geo.count, geo.iso) == (5, 3200)
    assert geo.interval == 1800 * u.s and geo.exposure == 20 * u.s and geo.start_offset == 60 * u.s
    assert isinstance(vega.target, FixedTarget) and vega.count == 1
    assert vega.exposure == 20 * u.s  # default
    assert m31.exposure == 15 * u.s
    assert plan.min_altitude == 20 * u.deg


def test_defaults_are_a_single_20s_frame_at_iso_1600_above_15_degrees():
    plan = parse_plan({"targets": [{"name": "x", "kind": "geo", "longitude_deg": 0}]})
    it = plan.items[0]
    assert (it.count, it.iso, it.exposure) == (1, 1600, 20 * u.s)
    assert plan.min_altitude == 15 * u.deg


def test_tle_targets_and_a_tle_catalog_file(tmp_path):
    l1, l2 = make_tle()
    c1, c2 = make_tle(catalog=11111)
    (tmp_path / "cat.tle").write_text(f"CATSAT\n{c1}\n{c2}\n")
    plan = parse_plan(
        {
            "catalog_tle_file": "cat.tle",
            "targets": [{"name": "ISS", "kind": "tle", "line1": l1, "line2": l2}],
        },
        base_dir=tmp_path,
    )
    assert isinstance(plan.items[0].target, TleTarget)
    assert [t.name for t in plan.catalog] == ["CATSAT"]


def test_load_plan_reads_a_file_and_resolves_the_catalog_relative_to_it(tmp_path):
    c1, c2 = make_tle(catalog=22222)
    (tmp_path / "cat.tle").write_text(f"{c1}\n{c2}\n")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({**GOOD, "catalog_tle_file": "cat.tle"}))
    plan = load_plan(path)
    assert len(plan.items) == 3 and len(plan.catalog) == 1


@pytest.mark.parametrize(
    "bad,match",
    [
        ({}, "non-empty 'targets'"),
        ({"targets": []}, "non-empty 'targets'"),
        ({"targets": ["x"]}, "must be an object"),
        ({"targets": [{"kind": "geo", "longitude_deg": 1}]}, "missing 'name'"),
        (with_target(kind="comet"), "unknown kind"),
        (with_target(kind=None), "unknown kind"),
        (with_target(longitude_deg="west"), "must be a number"),
        (with_target(longitude_deg=True), "must be a number"),
        ({"targets": [{"name": "g", "kind": "geo"}]}, "missing 'longitude_deg'"),
        (with_target(count=0), "at least 1"),
        (with_target(count=2.5), "whole number"),
        (with_target(exposure_s=0), "exposure_s"),
        (with_target(exposure_s=45), "exposure_s"),
        (with_target(interval_s=5), "interval_s"),  # shorter than the 20 s exposure, count 5
        (with_target(iso=0), "iso"),
        (with_target(iso=1600.5), "iso"),
        (with_target(start_offset_s=-1), "start_offset_s"),
        ({"targets": [{"name": "f", "kind": "fixed", "ra_deg": 400, "dec_deg": 0}]}, "range"),
        ({"targets": [{"name": "f", "kind": "fixed", "ra_deg": 10, "dec_deg": 95}]}, "range"),
        ({"targets": [{"name": "s", "kind": "star", "star": "Nonexistent"}]}, "unknown star"),
        ({"targets": [{"name": "t", "kind": "tle", "line1": "1 x"}]}, "line1' and 'line2"),
        ({"targets": [{"name": "t", "kind": "tle", "line1": "1 x", "line2": "2 y"}]}, "69"),
        ({**GOOD, "min_altitude_deg": 95}, "min_altitude_deg"),
        ({**GOOD, "catalog_tle_file": "missing.tle"}, "catalog_tle_file"),
    ],
)
def test_invalid_plans_say_what_is_wrong_and_where(bad, match):
    with pytest.raises(PlanError, match=match):
        parse_plan(bad)


def test_duplicate_names_are_rejected():
    t = GOOD["targets"][0]
    with pytest.raises(PlanError, match="duplicate"):
        parse_plan({"targets": [t, dict(t)]})


def test_error_messages_name_the_offending_entry():
    with pytest.raises(PlanError, match=r"targets\[0\] \(GEO-100W\)"):
        parse_plan(with_target(count=0))


def test_load_plan_reports_missing_and_malformed_files(tmp_path):
    with pytest.raises(PlanError, match="cannot read"):
        load_plan(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(PlanError, match="not valid JSON"):
        load_plan(bad)
    arr = tmp_path / "arr.json"
    arr.write_text("[1, 2]")
    with pytest.raises(PlanError, match="JSON object"):
        load_plan(arr)
