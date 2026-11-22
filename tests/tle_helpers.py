"""Build valid TLE lines (with checksums) from orbital elements, so tests do not depend on
remembered element sets."""

import numpy as np

from okapi.orbits import MU_EARTH


def _checksum(line: str) -> int:
    total = 0
    for ch in line:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def make_tle(
    *,
    catalog: int = 99999,
    epoch_year: int = 26,
    epoch_day: float = 325.5,  # day of year, 1-based
    inclination_deg: float = 51.6,
    raan_deg: float = 100.0,
    eccentricity: float = 0.0005,
    argp_deg: float = 30.0,
    mean_anomaly_deg: float = 0.0,
    revs_per_day: float = 15.5,
    bstar: str = " 00000-0",
) -> tuple[str, str]:
    l1 = (
        f"1 {catalog:05d}U 26001A   {epoch_year:02d}{epoch_day:012.8f}  .00000000  00000-0"
        f" {bstar} 0  999"
    )
    l1 += str(_checksum(l1))
    l2 = (
        f"2 {catalog:05d} {inclination_deg:8.4f} {raan_deg:8.4f}"
        f" {round(eccentricity * 1e7):07d} {argp_deg:8.4f} {mean_anomaly_deg:8.4f}"
        f" {revs_per_day:11.8f}    1"
    )
    l2 = l2[:68] + str(_checksum(l2[:68]))
    return l1, l2


def revs_per_day_for(a_km: float) -> float:
    return 86400.0 / (2 * np.pi * np.sqrt(a_km**3 / MU_EARTH))
