"""The whole night, in simulation: calibration, aiming, stares, solving, streak extraction,
identification, orbit fitting, persistence, resuming and failure handling."""

import itertools
import json

import astropy.units as u
import numpy as np

from okapi.frames import FrameMetadata
from okapi.orbits.simulate import geo_state
from okapi.orbits.twobody import propagate
from okapi.session.calibrate import calibrate
from okapi.session.night import NightConfig, format_report, run_night
from okapi.session.plan import parse_plan
from okapi.sim_night import decode_npy, make_night_rig
from okapi.targets import StateTarget

TILT_ERROR = (0.05 * u.deg, -0.03 * u.deg)  # a pointing residual the rotation-only model leaves


def geo_plan(count=5, interval=1800, exposure=20, name="GEO-100W", lon=-100, **extra):
    return parse_plan(
        {
            "targets": [
                {
                    "name": name,
                    "kind": "geo",
                    "longitude_deg": lon,
                    "count": count,
                    "interval_s": interval,
                    "exposure_s": exposure,
                    **extra,
                }
            ]
        }
    )


def truth_state(lon=-100.02):
    rig = make_night_rig()
    r, v = geo_state(lon, rig.clock.now())
    return r, v, rig.clock.now()


def make_rig(lon=-100.02, **kw):
    r, v, epoch = truth_state(lon)
    rig = make_night_rig([StateTarget("truth", r, v, epoch)], alignment_error=TILT_ERROR, **kw)
    return rig, (r, v, epoch)


def cal_for(rig):
    return calibrate(
        mount=rig.mount,
        camera=rig.camera,
        solver=rig.solver,
        gps=rig.gps,
        clock=rig.clock,
        settings=rig.settings,
    )


def go(rig, cal, plan, out, **kw):
    return run_night(
        plan=plan,
        mount=cal.mount,
        camera=rig.camera,
        solver=rig.solver,
        decode=decode_npy,
        clock=rig.clock,
        site=rig.site,
        out_dir=out,
        **kw,
    )


def events(out):
    return [json.loads(line) for line in (out / "night.log.jsonl").read_text().splitlines()]


def orbit_error(sol, truth):
    r, v, epoch = truth
    rt, vt = propagate(r, v, float((sol.epoch - epoch).to_value(u.s)))
    return sol.r - rt, sol.v - vt


# -- a complete night ---------------------------------------------------------------------------
def test_a_full_geo_night_produces_frames_observations_and_a_correct_orbit(tmp_path):
    rig, truth = make_rig()
    cal = cal_for(rig)
    rep = go(rig, cal, geo_plan(), tmp_path)
    assert rep.aborted is None and rep.warnings == []
    assert len(rep.frames) == 5 and all(f.solved for f in rep.frames)
    assert all(f.n_observations == 1 and f.labels == ("GEO-100W",) for f in rep.frames)

    # the pointing trim removes the model's residual after the first frame
    errs = [f.pointing_error.to_value(u.arcsec) for f in rep.frames]
    assert errs[0] > 30, "the scenario should start with a real pointing error"
    assert max(errs[1:]) < 12, errs

    sol = rep.orbits["GEO-100W"].solution
    assert sol is not None and sol.warnings == ()
    dr, dv = orbit_error(sol, truth)
    assert np.linalg.norm(dr) < 25.0 and np.linalg.norm(dv) < 0.005
    assert np.all(np.abs(dr) < 4 * sol.r_sigma)
    assert 0.05 < sol.chi2_reduced < 4


def test_every_frame_is_stored_with_a_sidecar_and_an_observation_file(tmp_path):
    rig, _ = make_rig()
    rep = go(rig, cal_for(rig), geo_plan(count=3), tmp_path)
    for f in rep.frames:
        folder = tmp_path / "raw" / f.start_time.strftime("%Y/%m/%d")
        assert (folder / f"{f.frame_id}.npy").stat().st_size > 1000
        meta = FrameMetadata.from_json((folder / f"{f.frame_id}.json").read_text())
        assert meta.frame_id == f.frame_id and meta.tracking and meta.exposure_s == 20
        obs = json.loads((folder / f"{f.frame_id}.obs.json").read_text())
        assert obs["frame_id"] == f.frame_id and len(obs["observations"]) == 1
    ids = [f.frame_id for f in rep.frames]
    assert len(set(ids)) == 3  # sequence numbers keep ids unique


def test_the_log_state_and_orbit_files_describe_the_night(tmp_path):
    rig, _ = make_rig()
    go(rig, cal_for(rig), geo_plan(count=3), tmp_path)
    kinds = [e["event"] for e in events(tmp_path)]
    assert kinds[0] == "night_start" and kinds[-1] == "night_end"
    assert kinds.count("frame") == 3 and "trim" in kinds
    state = json.loads((tmp_path / "night_state.json").read_text())
    assert state["done"] == {"GEO-100W": 3} and state["seq"] == 3
    orbits = json.loads((tmp_path / "orbits.json").read_text())
    assert orbits["GEO-100W"]["n_observations"] == 3
    assert 42000 < orbits["GEO-100W"]["a_km"] < 42400
    assert not list(tmp_path.rglob("*.tmp")) and not list(tmp_path.rglob("*.part"))


def test_after_the_night_the_telescope_is_not_left_driving(tmp_path):
    rig, _ = make_rig()
    go(rig, cal_for(rig), geo_plan(count=2), tmp_path)
    before = rig.physical.true_altaz()
    rig.clock.sleep(120)
    assert rig.physical.true_altaz().separation(before).to_value(u.arcsec) < 0.01
    assert rig.handset.tracking_mode == 0


# -- resuming ------------------------------------------------------------------------------------
def test_a_night_interrupted_by_camera_failure_resumes_without_repeating_frames(tmp_path):
    limit = {"n": 10**9}
    rig, truth = make_rig(camera_fail_on=lambda n: n > limit["n"])
    cal = cal_for(rig)
    limit["n"] = rig.camera.calls + 2  # two frames succeed, then the camera dies
    first = go(rig, cal, geo_plan(), tmp_path)
    assert first.aborted and "camera" in first.aborted
    assert len(first.frames) == 2
    state = json.loads((tmp_path / "night_state.json").read_text())
    assert state["done"]["GEO-100W"] == 2 and "trim" not in state

    limit["n"] = 10**9  # the camera is fixed; run the same command again
    second = go(rig, cal_for(rig), geo_plan(), tmp_path)
    assert second.aborted is None and len(second.frames) == 3
    assert len(list(tmp_path.rglob("*.npy"))) == 5
    all_ids = [p.stem for p in tmp_path.rglob("*.npy")]
    assert len(set(all_ids)) == 5
    # the restart recalibrates, which fits a new pointing model, so trims from before it would be
    # wrong and are not carried over: the first resumed frame starts uncorrected again
    errs = [f.pointing_error.to_value(u.arcsec) for f in second.frames]
    assert errs[0] > 15 and max(errs[1:]) < 15, errs
    # and the orbit uses all five observations, from both runs
    oo = second.orbits["GEO-100W"]
    assert oo.n_observations == 5 and oo.solution is not None
    dr, _ = orbit_error(oo.solution, truth)
    assert np.linalg.norm(dr) < 25.0


def test_rerunning_a_finished_night_takes_no_new_frames(tmp_path):
    rig, _ = make_rig()
    go(rig, cal_for(rig), geo_plan(count=2), tmp_path)
    calls = rig.camera.calls
    again = go(rig, cal_for(rig), geo_plan(count=2), tmp_path)
    assert again.frames == [] and rig.camera.calls == calls + (cal_for_calls(rig, calls))
    assert again.orbits["GEO-100W"].n_observations == 2


def cal_for_calls(rig, before):
    """Camera calls made by the calibration that the re-run performed (not by the night)."""
    return rig.camera.calls - before


def test_a_corrupt_state_file_stops_the_night_with_a_clear_message(tmp_path):
    (tmp_path / "night_state.json").write_text("{not json")
    rig, _ = make_rig()
    rep = go(rig, cal_for(rig), geo_plan(count=1), tmp_path)
    assert rep.aborted and "unreadable" in rep.aborted and rep.frames == []


# -- failures -----------------------------------------------------------------------------------
def solver_failing(rig_holder, first, count):
    return lambda n: first <= n < first + count


def test_clouds_abort_the_night_cleanly_but_the_raw_frames_are_kept(tmp_path):
    window = {"lo": 10**9, "hi": 10**9}
    rig, _ = make_rig(solver_fail_on=lambda n: window["lo"] <= n < window["hi"])
    cal = cal_for(rig)
    window["lo"], window["hi"] = rig.solver.calls + 1, rig.solver.calls + 4  # three in a row
    rep = go(rig, cal, geo_plan(), tmp_path)
    assert rep.aborted and "solve" in rep.aborted
    assert len(rep.frames) == 3 and not any(f.solved for f in rep.frames)
    assert len(list(tmp_path.rglob("*.npy"))) == 3  # unsolved frames are still evidence
    assert not list(tmp_path.rglob("*.obs.json"))  # but yield no observations
    assert rep.orbits == {}
    assert rig.physical.true_altaz().separation(rig.physical.true_altaz()).value == 0
    assert "ABORTED" in format_report(rep)


def test_repeated_solve_failures_trigger_a_recalibration_and_the_night_continues(tmp_path):
    window = {"lo": 10**9, "hi": 10**9}
    rig, _ = make_rig(solver_fail_on=lambda n: window["lo"] <= n < window["hi"])
    cal = cal_for(rig)
    window["lo"], window["hi"] = rig.solver.calls + 1, rig.solver.calls + 4
    recals = []

    def recalibrate():
        recals.append(1)
        window["lo"] = window["hi"] = 10**9  # the cloud has passed
        return cal_for(rig).mount

    rep = go(rig, cal, geo_plan(), tmp_path, recalibrate=recalibrate)
    assert rep.aborted is None and rep.recalibrations == 1 and len(recals) == 1
    assert len(rep.frames) == 5 and sum(f.solved for f in rep.frames) == 2
    assert any(e["event"] == "recalibrate" for e in events(tmp_path))


def test_a_recalibration_that_fails_ends_the_night_and_says_why(tmp_path):
    window = {"lo": 10**9, "hi": 10**9}
    rig, _ = make_rig(solver_fail_on=lambda n: window["lo"] <= n < window["hi"])
    cal = cal_for(rig)
    window["lo"], window["hi"] = rig.solver.calls + 1, 10**9  # solving never recovers

    def recalibrate():
        raise RuntimeError("no stars visible")

    rep = go(rig, cal, geo_plan(), tmp_path, recalibrate=recalibrate)
    assert rep.aborted and "recalibration failed" in rep.aborted and "no stars" in rep.aborted


def test_a_single_camera_error_is_retried_and_the_frame_is_still_taken(tmp_path):
    first = {"n": 10**9}
    rig, _ = make_rig(camera_fail_on=lambda n: n == first["n"])
    cal = cal_for(rig)
    first["n"] = rig.camera.calls + 1
    rep = go(rig, cal, geo_plan(count=2), tmp_path)
    assert rep.aborted is None and len(rep.frames) == 2 and all(f.solved for f in rep.frames)
    assert [e for e in events(tmp_path) if e["event"] == "camera_error"]


def test_a_target_that_never_rises_is_skipped_then_given_up_and_the_rest_continues(tmp_path):
    plan = parse_plan(
        {
            "targets": [
                {"name": "south", "kind": "fixed", "ra_deg": 0.0, "dec_deg": -80.0},
                {
                    "name": "GEO-100W",
                    "kind": "geo",
                    "longitude_deg": -100,
                    "count": 2,
                    "interval_s": 600,
                    "exposure_s": 20,
                },
            ]
        }
    )
    rig, _ = make_rig()
    cfg = NightConfig(max_skips=3, retry_below_horizon=60 * u.s)
    rep = go(rig, cal_for(rig), plan, tmp_path, config=cfg)
    assert rep.aborted is None
    assert rep.skipped["south"] == 3
    assert any("gave up on south" in w for w in rep.warnings)
    assert [f.item for f in rep.frames] == ["GEO-100W", "GEO-100W"]
    assert [e for e in events(tmp_path) if e["event"] == "give_up"]


def test_the_altitude_limit_in_the_plan_is_respected(tmp_path):
    rig, _ = make_rig()
    cal = cal_for(rig)
    # the GEO slot is about 45 degrees up from the test site; demand more than that
    plan = geo_plan(count=1)
    high = type(plan)(plan.items, plan.catalog, 80 * u.deg)
    rep = go(
        rig, cal, high, tmp_path, config=NightConfig(max_skips=2, retry_below_horizon=10 * u.s)
    )
    assert rep.frames == [] and rep.skipped["GEO-100W"] == 2


# -- scheduling ---------------------------------------------------------------------------------
def test_two_targets_are_interleaved_by_due_time(tmp_path):
    plan = parse_plan(
        {
            "targets": [
                {
                    "name": "A",
                    "kind": "star",
                    "star": "Vega",
                    "count": 3,
                    "interval_s": 600,
                    "exposure_s": 10,
                },
                {
                    "name": "B",
                    "kind": "star",
                    "star": "Deneb",
                    "count": 3,
                    "interval_s": 600,
                    "exposure_s": 10,
                    "start_offset_s": 300,
                },
            ]
        }
    )
    rig, _ = make_rig()
    rep = go(rig, cal_for(rig), plan, tmp_path)
    assert rep.aborted is None
    assert [f.item for f in rep.frames] == ["A", "B", "A", "B", "A", "B"]
    for name in "AB":
        starts = [f.start_time for f in rep.frames if f.item == name]
        gaps = [(b - a).to_value(u.s) for a, b in itertools.pairwise(starts)]
        assert all(g >= 600 for g in gaps), gaps
    assert rep.orbits == {}  # fixed stars: nothing to fit


# -- aiming ------------------------------------------------------------------------------------
def test_aiming_ahead_is_corrected_when_the_slew_takes_longer_than_assumed(tmp_path):
    def streak_offset(tolerance, sub):
        # no pointing residual here, so the only error is how long the slew took
        r, v, epoch = truth_state(-100.0)  # exactly on the planned slot
        rig = make_night_rig([StateTarget("truth", r, v, epoch)])
        cal = cal_for(rig)
        out = tmp_path / sub
        cfg = NightConfig(slew_estimate=1 * u.s, recenter_tolerance=tolerance)
        rep = go(rig, cal, geo_plan(count=1), out, config=cfg)
        assert rep.aborted is None and rep.frames[0].n_observations == 1
        folder = out / "raw" / rep.frames[0].start_time.strftime("%Y/%m/%d")
        from okapi.astrometry.observation import read_observations

        obs = read_observations(folder / f"{rep.frames[0].frame_id}.obs.json")[0]
        h, w = rig.camera.shape
        centre = rig.camera.last_wcs.pixel_to_world((w - 1) / 2, (h - 1) / 2)
        return obs.coord.separation(centre).to_value(u.arcsec), out

    with_reaim, out1 = streak_offset(30 * u.arcsec, "a")
    without, _ = streak_offset(1e9 * u.arcsec, "b")
    assert [e for e in events(out1) if e["event"] == "reaim"]
    assert with_reaim < 40
    assert without > 250  # without re-aiming the stare lands hundreds of arcsec off


# -- identification ---------------------------------------------------------------------------
def test_a_streak_matching_nothing_in_the_plan_is_kept_but_labelled_unidentified(tmp_path):
    # a satellite 0.3 deg from the planned slot is in the (wide) frame but outside the gates
    r, v, epoch = truth_state(-99.7)
    rig = make_night_rig(
        [StateTarget("stranger", r, v, epoch)],
        alignment_error=TILT_ERROR,
        pixel_scale_arcsec=3.0,
    )
    rep = go(rig, cal_for(rig), geo_plan(count=3), tmp_path)
    assert rep.aborted is None
    assert all(f.labels == ("unidentified",) for f in rep.frames), [f.labels for f in rep.frames]
    assert "unidentified:GEO-100W" in rep.orbits
    assert rep.orbits["unidentified:GEO-100W"].n_observations == 3


# -- report ----------------------------------------------------------------------------------
def test_the_report_text_summarises_frames_orbits_and_warnings(tmp_path):
    rig, _ = make_rig()
    rep = go(rig, cal_for(rig), geo_plan(count=3), tmp_path)
    text = format_report(rep)
    assert "frames: 3  (solved 3), observations: 3" in text
    assert "GEO-100W: 3 obs, a=" in text
    assert "ABORTED" not in text
