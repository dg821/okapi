import json

from typer.testing import CliRunner

from okapi.cli import app

PLAN = {
    "targets": [
        {
            "name": "GEO-100W",
            "kind": "geo",
            "longitude_deg": -100,
            "count": 3,
            "interval_s": 1800,
            "exposure_s": 20,
        }
    ]
}


def write_plan(tmp_path, plan=PLAN):
    p = tmp_path / "plan.json"
    p.write_text(json.dumps(plan))
    return p


def test_night_dry_run_completes_and_writes_everything(tmp_path):
    out = tmp_path / "out"
    r = CliRunner().invoke(
        app, ["night", str(write_plan(tmp_path)), "--dry-run", "--out", str(out)]
    )
    assert r.exit_code == 0, r.output
    assert "aligned on:" in r.stdout and "frames: 3  (solved 3), observations: 3" in r.stdout
    assert "GEO-100W: 3 obs, a=" in r.stdout
    assert (out / "orbits.json").exists() and (out / "night.log.jsonl").exists()
    assert len(list(out.rglob("*.obs.json"))) == 3


def test_night_dry_run_resumes_when_run_again(tmp_path):
    out = tmp_path / "out"
    plan = write_plan(tmp_path)
    CliRunner().invoke(app, ["night", str(plan), "--dry-run", "--out", str(out)])
    r = CliRunner().invoke(app, ["night", str(plan), "--dry-run", "--out", str(out)])
    assert r.exit_code == 0, r.output
    assert "frames: 0" in r.stdout  # nothing left to take
    assert "GEO-100W: 3 obs" in r.stdout  # but the orbit is still reported from the saved files
    assert len(list(out.rglob("*.npy"))) == 3


def test_night_rejects_a_bad_plan_with_exit_code_2(tmp_path):
    bad = write_plan(tmp_path, {"targets": [{"name": "x", "kind": "geo"}]})
    r = CliRunner().invoke(app, ["night", str(bad), "--dry-run", "--out", str(tmp_path / "o")])
    assert r.exit_code == 2 and "plan error" in r.output and "longitude_deg" in r.output


def test_night_reports_a_missing_plan_file(tmp_path):
    r = CliRunner().invoke(app, ["night", str(tmp_path / "nope.json"), "--dry-run"])
    assert r.exit_code == 2 and "plan error" in r.output


def test_night_needs_ports_or_dry_run(tmp_path):
    r = CliRunner().invoke(app, ["night", str(write_plan(tmp_path))])
    assert r.exit_code == 2 and "--mount-port" in r.output


def test_night_reports_unopenable_serial_ports_cleanly(tmp_path):
    r = CliRunner().invoke(
        app,
        [
            "night",
            str(write_plan(tmp_path)),
            "--mount-port",
            "/dev/nope1",
            "--gps-port",
            "/dev/nope2",
            "--out",
            str(tmp_path / "o"),
        ],
    )
    assert r.exit_code == 1 and "could not open the hardware" in r.output


def test_orbit_command_fits_files_written_by_a_night(tmp_path):
    out = tmp_path / "out"
    CliRunner().invoke(app, ["night", str(write_plan(tmp_path)), "--dry-run", "--out", str(out)])
    files = sorted(str(p) for p in out.rglob("*.obs.json"))
    r = CliRunner().invoke(
        app, ["orbit", *files, "--lat", "40", "--lon", "-105", "--height", "1600"]
    )
    assert r.exit_code == 0, r.output
    assert "observations: 3" in r.stdout and "a = 42" in r.stdout


def test_orbit_command_refuses_a_single_observation(tmp_path):
    out = tmp_path / "out"
    CliRunner().invoke(app, ["night", str(write_plan(tmp_path)), "--dry-run", "--out", str(out)])
    one = sorted(str(p) for p in out.rglob("*.obs.json"))[:1]
    r = CliRunner().invoke(app, ["orbit", *one, "--lat", "40", "--lon", "-105"])
    assert r.exit_code == 1 and "at least two" in r.output


def test_orbit_command_reports_unreadable_files(tmp_path):
    bad = tmp_path / "bad.obs.json"
    bad.write_text("not json")
    r = CliRunner().invoke(app, ["orbit", str(bad), "--lat", "40", "--lon", "-105"])
    assert r.exit_code == 2 and "cannot read" in r.output
