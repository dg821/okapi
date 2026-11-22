"""The night: aim, stare, capture, extract streaks, identify, and fit orbits, resumably.

For each due frame of the plan: predict where the target will be mid-exposure, slew there (through
the pointing model), track sidereally, expose, store the RAW frame with its sidecar, plate-solve,
extract streaks, write the observations next to the frame, and compare the solved pointing with
the aim. A persistent per-target pointing trim feeds that error back into the next aim, because
the pointing model is only a rotation and leaves a residual.

Failure policy: a frame that cannot be solved is still kept (the RAW is evidence) but yields no
observation. Repeated solve failures trigger a recalibration if one is available, otherwise the
night aborts cleanly with its state saved. Targets below the altitude limit are retried later.
Progress is written after every frame, so a crash (or Ctrl-C) loses at most the current frame and
re-running the same command picks up where it stopped.
"""

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.astrometry.identify import identify
from okapi.astrometry.observation import Observation, read_observations, write_observations
from okapi.astrometry.pipeline import process_frame
from okapi.camera.base import CameraError, CameraProtocol, CapturedImage, ExposureSettings
from okapi.clock import Clock
from okapi.coords import radec_to_altaz
from okapi.frames import FrameWriter, make_frame_id, metadata_for
from okapi.mount.base import MountError, MountProtocol
from okapi.orbits.fit import OrbitFitError, OrbitSolution, determine_orbit
from okapi.session.plan import NightPlan, PlanItem
from okapi.solve.base import SolveError, SolverProtocol
from okapi.targets import Target, TargetError


@dataclass(frozen=True)
class NightConfig:
    slew_estimate: u.Quantity = 20 * u.s  # how long we assume a slew takes when aiming ahead
    settle_timeout: u.Quantity = 180 * u.s
    recenter_tolerance: u.Quantity = 30 * u.arcsec  # re-aim if the target moved more than this
    trim_threshold: u.Quantity = 10 * u.arcsec  # pointing errors below this are solver noise
    max_consecutive_failures: int = 3
    max_recalibrations: int = 2
    retry_below_horizon: u.Quantity = 300 * u.s
    max_skips: int = 12
    idle_stop_tracking: u.Quantity = 60 * u.s


class _AbortError(RuntimeError):
    pass


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    item: str
    start_time: Time
    solved: bool
    pointing_error: u.Quantity | None
    n_observations: int
    labels: tuple[str, ...]  # per observation: matched object, or "unidentified"


@dataclass(frozen=True)
class ObjectOrbit:
    name: str
    n_observations: int
    solution: OrbitSolution | None
    error: str | None = None


@dataclass
class NightReport:
    out_dir: Path
    frames: list[FrameRecord] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    orbits: dict[str, ObjectOrbit] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    recalibrations: int = 0
    aborted: str | None = None

    @property
    def n_observations(self) -> int:
        return sum(f.n_observations for f in self.frames)


# -- persistent state and event log ----------------------------------------------------------------
@dataclass
class _State:
    seq: int = 0
    done: dict[str, int] = field(default_factory=dict)
    last_start: dict[str, str] = field(default_factory=dict)
    # one entry per observation: item, label, path (relative to out_dir), index within that file
    observations: list[dict[str, Any]] = field(default_factory=list)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _load_state(path: Path) -> _State:
    if not path.exists():
        return _State()
    try:
        raw = json.loads(path.read_text())
        return _State(**raw)
    except (OSError, ValueError, TypeError) as e:
        raise _AbortError(
            f"{path} is unreadable ({e}); move it aside to start the night afresh, or repair it"
        ) from e


class _EventLog:
    def __init__(self, path: Path, clock: Clock) -> None:
        self._path, self._clock = path, clock

    def write(self, event: str, **fields: Any) -> None:
        line = json.dumps(
            {"time_utc": self._clock.now().utc.isot, "event": event, **fields}, default=str
        )
        with open(self._path, "a") as f:
            f.write(line + "\n")


# -- the run ----------------------------------------------------------------------------------
def run_night(
    *,
    plan: NightPlan,
    mount: MountProtocol,
    camera: CameraProtocol,
    solver: SolverProtocol,
    decode: Callable[[CapturedImage], np.ndarray],
    clock: Clock,
    site: EarthLocation,
    out_dir: Path,
    recalibrate: Callable[[], MountProtocol] | None = None,
    config: NightConfig = NightConfig(),
) -> NightReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = NightReport(out_dir)
    log = _EventLog(out_dir / "night.log.jsonl", clock)
    state_path = out_dir / "night_state.json"
    try:
        state = _load_state(state_path)
    except _AbortError as e:
        report.aborted = str(e)
        return report
    writer = FrameWriter(out_dir)
    run_start = clock.now()
    satellites: list[Target] = [it.target for it in plan.items if it.target.is_satellite]
    candidates = [*satellites, *plan.catalog]
    skips: dict[str, int] = {}
    # Pointing trims (arcsec: d_ra_cosdec, d_dec) are relative to the pointing model fitted at the
    # start of THIS run, so they are deliberately not saved: a resumed night recalibrates.
    trims: dict[str, list[float]] = {}
    retry_at: dict[str, Time] = {}
    exhausted: set[str] = set()
    consecutive = {"solve": 0, "camera": 0, "mount": 0}
    log.write("night_start", items=[it.name for it in plan.items], resumed=state.seq > 0)

    def due_time(it: PlanItem) -> Time:
        if state.done.get(it.name, 0) == 0:
            t = run_start + it.start_offset
        else:
            t = Time(state.last_start[it.name], format="isot", scale="utc") + it.interval
        return max(t, retry_at[it.name]) if it.name in retry_at else t

    def aim_for(it: PlanItem, t: Time) -> tuple[SkyCoord, SkyCoord]:
        """(where to command the mount, where the target is). They differ by the pointing trim:
        the commanded aim is the prediction shifted by minus the mount's residual error."""
        trim = trims.get(it.name, [0.0, 0.0])
        pred = it.target.predict(site, t)
        return pred.spherical_offsets_by(trim[0] * u.arcsec, trim[1] * u.arcsec), pred

    def save() -> None:
        _atomic_write(state_path, json.dumps(state.__dict__, indent=2, sort_keys=True))

    def skip(it: PlanItem, why: str) -> None:
        skips[it.name] = skips.get(it.name, 0) + 1
        report.skipped[it.name] = skips[it.name]
        retry_at[it.name] = clock.now() + config.retry_below_horizon
        log.write("skip", item=it.name, reason=why, count=skips[it.name])
        if skips[it.name] >= config.max_skips:
            exhausted.add(it.name)
            report.warnings.append(f"gave up on {it.name}: skipped {skips[it.name]} times ({why})")
            log.write("give_up", item=it.name)

    def failed(kind: str, why: str) -> None:
        nonlocal mount
        consecutive[kind] += 1
        if consecutive[kind] < config.max_consecutive_failures:
            return
        if (
            kind == "solve"
            and recalibrate is not None
            and (report.recalibrations < config.max_recalibrations)
        ):
            report.recalibrations += 1
            log.write("recalibrate", reason=why, attempt=report.recalibrations)
            try:
                mount = recalibrate()
            except (
                Exception
            ) as e:  # calibration reports its own failures; any of them ends the night
                raise _AbortError(f"recalibration failed: {e}") from e
            trims.clear()  # the pointing model changed, so the old trims are stale
            consecutive["solve"] = 0
            return
        raise _AbortError(f"{consecutive[kind]} consecutive {kind} failures: {why}")

    def do_frame(it: PlanItem) -> None:
        nonlocal mount
        exposure = it.exposure
        t_est = clock.now() + config.slew_estimate + exposure / 2
        try:
            aim, aim_pred = aim_for(it, t_est)
        except TargetError as e:
            return skip(it, f"cannot predict position: {e}")
        alt = radec_to_altaz(aim, site, t_est).alt
        if alt < plan.min_altitude:
            return skip(it, f"altitude {alt.to_value(u.deg):.1f} deg below the limit")
        try:
            for attempt in range(3):  # later passes correct for how long the slews really took
                mount.slew_radec(aim)
                mount.wait_until_settled(config.settle_timeout)
                mount.start_tracking()
                need, need_pred = aim_for(it, clock.now() + exposure / 2)
                if attempt == 2 or need.separation(aim) <= config.recenter_tolerance:
                    break
                log.write(
                    "reaim",
                    item=it.name,
                    moved_arcsec=float(need.separation(aim).to_value(u.arcsec)),
                )
                aim, aim_pred = need, need_pred
            consecutive["mount"] = 0
        except MountError as e:
            log.write("mount_error", item=it.name, error=str(e))
            return failed("mount", str(e))
        settings = ExposureSettings(exposure, it.iso)
        image = None
        for attempt in range(2):
            try:
                image = camera.capture(settings)
                break
            except CameraError as e:
                log.write("camera_error", item=it.name, error=str(e), attempt=attempt)
        if image is None:
            return failed("camera", "the camera would not take an exposure")
        consecutive["camera"] = 0

        frame_id = make_frame_id(image.start_time, state.seq)
        state.seq += 1
        image_path = writer.write(
            image, metadata_for(image, frame_id=frame_id, mount=mount, site=site)
        )
        obs_path = image_path.with_name(f"{frame_id}.obs.json")
        pointing_error: u.Quantity | None = None
        observations: list[Observation] = []
        try:
            result = process_frame(image, solver=solver, decode=decode)
        except SolveError as e:
            log.write("solve_failed", item=it.name, frame=frame_id, error=str(e))
            report.frames.append(
                FrameRecord(frame_id, it.name, image.start_time, False, None, 0, ())
            )
            state.done[it.name] = state.done.get(it.name, 0) + 1
            state.last_start[it.name] = image.start_time.utc.isot
            save()
            return failed("solve", str(e))
        consecutive["solve"] = 0
        observations = result.observations
        # How far the frame centre landed from where the target was predicted (the stare holds
        # the centre fixed in RA/Dec). Feed it back: the trim is minus the mount's residual error.
        pointing_error = result.center.separation(aim_pred).to(u.arcsec)
        if pointing_error > config.trim_threshold:
            d_lon, d_lat = aim_pred.spherical_offsets_to(result.center)
            cur = trims.get(it.name, [0.0, 0.0])
            trims[it.name] = [
                cur[0] - float(d_lon.to_value(u.arcsec)),
                cur[1] - float(d_lat.to_value(u.arcsec)),
            ]
            log.write(
                "trim",
                item=it.name,
                error_arcsec=float(pointing_error.value),
                trim=trims[it.name],
            )
        labels: list[str] = []
        for o in observations:
            m = identify(o, candidates, site)
            labels.append("unidentified" if m is None else m.target)
            if m is not None and m.ambiguous:
                report.warnings.append(f"{frame_id}: identification of {m.target} is ambiguous")
        write_observations(obs_path, frame_id, observations)
        rel = str(obs_path.relative_to(out_dir))
        for idx, lab in enumerate(labels):
            state.observations.append({"item": it.name, "label": lab, "path": rel, "index": idx})
        state.done[it.name] = state.done.get(it.name, 0) + 1
        state.last_start[it.name] = image.start_time.utc.isot
        save()
        report.frames.append(
            FrameRecord(
                frame_id,
                it.name,
                image.start_time,
                True,
                pointing_error,
                len(observations),
                tuple(labels),
            )
        )
        log.write(
            "frame",
            item=it.name,
            frame=frame_id,
            observations=len(observations),
            labels=labels,
            pointing_error_arcsec=float(pointing_error.value),
        )

    try:
        while True:
            pending = [
                (i, it)
                for i, it in enumerate(plan.items)
                if state.done.get(it.name, 0) < it.count and it.name not in exhausted
            ]
            if not pending:
                break
            _, it = min(pending, key=lambda p: (due_time(p[1]), p[0]))
            wait = float((due_time(it) - clock.now()).to_value(u.s))
            if wait > 0:
                if wait > config.idle_stop_tracking.to_value(u.s):
                    with contextlib.suppress(MountError):
                        mount.stop_tracking()
                clock.sleep(wait)
            do_frame(it)
    except _AbortError as e:
        report.aborted = str(e)
        log.write("abort", reason=str(e))
    finally:
        with contextlib.suppress(MountError):
            mount.stop_tracking()
    return _finish(report, state, site, out_dir, log)


def _finish(
    report: NightReport, state: _State, site: EarthLocation, out_dir: Path, log: _EventLog
) -> NightReport:
    groups: dict[str, list[Observation]] = {}
    cache: dict[str, list[Observation]] = {}
    for entry in state.observations:
        if entry["path"] not in cache:
            cache[entry["path"]] = read_observations(out_dir / entry["path"])
        key = (
            entry["label"] if entry["label"] != "unidentified" else f"unidentified:{entry['item']}"
        )
        groups.setdefault(key, []).append(cache[entry["path"]][entry["index"]])
    out: dict[str, Any] = {}
    for name, obs in sorted(groups.items()):
        try:
            sol = determine_orbit(obs, site)
            report.orbits[name] = ObjectOrbit(name, len(obs), sol)
            out[name] = _solution_json(len(obs), sol)
        except (OrbitFitError, ValueError, FloatingPointError) as e:
            report.orbits[name] = ObjectOrbit(name, len(obs), None, str(e))
            out[name] = {"n_observations": len(obs), "error": str(e)}
    _atomic_write(out_dir / "orbits.json", json.dumps(out, indent=2, sort_keys=True))
    log.write("night_end", frames=len(report.frames), aborted=report.aborted)
    return report


def _solution_json(n: int, sol: OrbitSolution) -> dict[str, Any]:
    el = sol.elements
    return {
        "n_observations": n,
        "epoch_utc": sol.epoch.utc.isot,
        "r_km": [float(x) for x in sol.r],
        "v_km_s": [float(x) for x in sol.v],
        "r_sigma_km": [float(x) for x in sol.r_sigma],
        "a_km": float(el.a),
        "e": float(el.e),
        "i_deg": float(np.degrees(el.i)),
        "raan_deg": float(np.degrees(el.raan)),
        "argp_deg": float(np.degrees(el.argp)),
        "nu_deg": float(np.degrees(el.nu)),
        "chi2_reduced": sol.chi2_reduced,
        "condition": sol.condition,
        "warnings": list(sol.warnings),
    }


def format_report(report: NightReport) -> str:
    """A short human-readable summary of a night."""
    lines = [
        f"frames: {len(report.frames)}  "
        f"(solved {sum(f.solved for f in report.frames)}), "
        f"observations: {report.n_observations}",
    ]
    for item, n in sorted(report.skipped.items()):
        lines.append(f"skipped {item}: {n} time(s)")
    if report.recalibrations:
        lines.append(f"recalibrations: {report.recalibrations}")
    for name, oo in report.orbits.items():
        if oo.solution is None:
            lines.append(f"{name}: {oo.n_observations} observation(s), no orbit ({oo.error})")
            continue
        el, sol = oo.solution.elements, oo.solution
        lines.append(
            f"{name}: {oo.n_observations} obs, a={el.a:.0f} km e={el.e:.4f} "
            f"i={np.degrees(el.i):.2f} deg, position sigma "
            f"{np.max(sol.r_sigma):.0f} km, chi2 {sol.chi2_reduced:.2f}"
        )
        lines.extend(f"  warning: {w}" for w in sol.warnings)
    lines.extend(f"warning: {w}" for w in report.warnings)
    if report.aborted:
        lines.append(f"ABORTED: {report.aborted}")
    return "\n".join(lines)
