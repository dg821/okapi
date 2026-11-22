"""A night plan: what to observe, how often, with what exposure. Loaded from JSON.

{
  "min_altitude_deg": 15,
  "catalog_tle_file": "tles.txt",
  "targets": [
    {"name": "GEO-100W", "kind": "geo", "longitude_deg": -100,
     "count": 5, "interval_s": 1800, "exposure_s": 20, "iso": 1600},
    {"name": "Vega", "kind": "star", "star": "Vega", "count": 1, "exposure_s": 10},
    {"name": "ISS", "kind": "tle", "line1": "1 ...", "line2": "2 ...",
     "count": 3, "interval_s": 120, "exposure_s": 10},
    {"name": "M31", "kind": "fixed", "ra_deg": 10.68, "dec_deg": 41.27, "exposure_s": 20}
  ]
}
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import astropy.units as u
from astropy.coordinates import SkyCoord

from okapi.catalog import get_star
from okapi.targets import (
    FixedTarget,
    GeoSlotTarget,
    Target,
    TargetError,
    TleTarget,
    parse_tles,
)

MAX_EXPOSURE = 30 * u.s  # beyond this the T3i needs bulb mode (assumption A3, unverified)


class PlanError(ValueError):
    """The plan file is invalid; the message names the entry and the field."""


@dataclass(frozen=True)
class PlanItem:
    name: str
    target: Target
    count: int
    interval: u.Quantity  # between starts of consecutive frames of this item
    exposure: u.Quantity
    iso: int
    start_offset: u.Quantity  # delay of the first frame after the run starts


@dataclass(frozen=True)
class NightPlan:
    items: tuple[PlanItem, ...]
    catalog: tuple[Target, ...] = ()  # extra satellites to identify streaks against
    min_altitude: u.Quantity = 15 * u.deg


def _number(
    entry: Mapping[str, Any], key: str, where: str, *, default: float | None = None
) -> float:
    if key not in entry:
        if default is None:
            raise PlanError(f"{where}: missing '{key}'")
        return default
    v = entry[key]
    if isinstance(v, bool) or not isinstance(v, int | float):
        raise PlanError(f"{where}: '{key}' must be a number, got {v!r}")
    return float(v)


def _target(entry: Mapping[str, Any], where: str) -> Target:
    kind = entry.get("kind")
    name = str(entry.get("name", ""))
    try:
        if kind == "geo":
            lon = _number(entry, "longitude_deg", where)
            if not -180 <= lon <= 360:
                raise PlanError(f"{where}: longitude_deg {lon} out of range")
            return GeoSlotTarget(name, lon * u.deg)
        if kind == "fixed":
            ra, dec = _number(entry, "ra_deg", where), _number(entry, "dec_deg", where)
            if not (0 <= ra < 360 and -90 <= dec <= 90):
                raise PlanError(f"{where}: ra_deg/dec_deg out of range")
            return FixedTarget(name, SkyCoord(ra * u.deg, dec * u.deg, frame="icrs"))
        if kind == "star":
            star = get_star(str(entry.get("star", name)))
            return FixedTarget(name, star.coord())
        if kind == "tle":
            if "line1" not in entry or "line2" not in entry:
                raise PlanError(f"{where}: a tle target needs 'line1' and 'line2'")
            return TleTarget(name, str(entry["line1"]), str(entry["line2"]))
    except (TargetError, KeyError) as e:
        raise PlanError(f"{where}: {e}") from e
    raise PlanError(f"{where}: unknown kind {kind!r} (use geo, fixed, star or tle)")


def parse_plan(data: Mapping[str, Any], *, base_dir: Path | None = None) -> NightPlan:
    targets = data.get("targets")
    if not isinstance(targets, list) or not targets:
        raise PlanError("plan needs a non-empty 'targets' list")
    items: list[PlanItem] = []
    seen: set[str] = set()
    for i, entry in enumerate(targets):
        if not isinstance(entry, Mapping):
            raise PlanError(f"targets[{i}]: must be an object")
        name = str(entry.get("name", "")).strip()
        where = f"targets[{i}] ({name or 'unnamed'})"
        if not name:
            raise PlanError(f"{where}: missing 'name'")
        if name in seen:
            raise PlanError(f"{where}: duplicate name")
        seen.add(name)
        count = _number(entry, "count", where, default=1)
        if count != int(count) or count < 1:
            raise PlanError(f"{where}: 'count' must be a whole number of at least 1")
        exposure = _number(entry, "exposure_s", where, default=20.0)
        if not 0 < exposure <= MAX_EXPOSURE.to_value(u.s):
            raise PlanError(f"{where}: 'exposure_s' must be in (0, {MAX_EXPOSURE.to_value(u.s):g}]")
        interval = _number(entry, "interval_s", where, default=0.0)
        if interval < 0 or (count > 1 and interval < exposure):
            raise PlanError(f"{where}: 'interval_s' must be at least the exposure for repeats")
        iso = _number(entry, "iso", where, default=1600)
        if iso != int(iso) or iso <= 0:
            raise PlanError(f"{where}: 'iso' must be a positive whole number")
        offset = _number(entry, "start_offset_s", where, default=0.0)
        if offset < 0:
            raise PlanError(f"{where}: 'start_offset_s' cannot be negative")
        items.append(
            PlanItem(
                name,
                _target(entry, where),
                int(count),
                interval * u.s,
                exposure * u.s,
                int(iso),
                offset * u.s,
            )
        )
    catalog: list[Target] = []
    tle_file = data.get("catalog_tle_file")
    if tle_file:
        path = Path(str(tle_file))
        if not path.is_absolute() and base_dir is not None:
            path = base_dir / path
        try:
            catalog = list(parse_tles(path.read_text()))
        except (OSError, TargetError) as e:
            raise PlanError(f"catalog_tle_file {tle_file}: {e}") from e
    min_alt = _number(data, "min_altitude_deg", "plan", default=15.0)
    if not 0 <= min_alt < 89:
        raise PlanError("plan: 'min_altitude_deg' must be in [0, 89)")
    return NightPlan(tuple(items), tuple(catalog), min_alt * u.deg)


def load_plan(path: Path) -> NightPlan:
    try:
        data = json.loads(path.read_text())
    except OSError as e:
        raise PlanError(f"cannot read {path}: {e}") from e
    except json.JSONDecodeError as e:
        raise PlanError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, Mapping):
        raise PlanError("plan must be a JSON object")
    return parse_plan(data, base_dir=path.parent)
