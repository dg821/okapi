"""Bright-star catalog for alignment, and a picker for a well-spread pair.

Positions are ICRS at epoch J2000, hand-entered and cross-checked against the CDS Sesame
resolver (all 35 agree within 1.6 arcsec; Castor, a close binary, is the worst). Proper
motions (mas/yr) are from memory and NOT independently verified; they matter at the tens of
arcseconds level for Arcturus, Sirius and Procyon only. Both are far better than the
centering loop needs. tests/test_catalog.py cross-checks known angular separations.
"""

import math
from dataclasses import dataclass

import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from okapi.coords import radec_to_altaz

J2000 = Time("J2000")


@dataclass(frozen=True)
class Star:
    name: str
    ra: str  # J2000, e.g. "18h36m56.3s"
    dec: str  # J2000, e.g. "+38d47m01s"
    vmag: float
    pm_ra_cosdec_mas: float = 0.0  # proper motion in RA, already multiplied by cos(dec)
    pm_dec_mas: float = 0.0

    def coord(self, at: Time | None = None) -> SkyCoord:
        """ICRS position, moved to ``at`` by linear proper motion (J2000 if omitted)."""
        c = SkyCoord(ra=self.ra, dec=self.dec, frame="icrs")
        if at is None:
            return c
        years = float(at.tt.jyear - J2000.tt.jyear)
        d_dec = self.pm_dec_mas * years * u.mas
        d_ra = self.pm_ra_cosdec_mas * years * u.mas / math.cos(c.dec.radian)
        return SkyCoord(ra=c.ra + d_ra, dec=c.dec + d_dec, frame="icrs")


# fmt: off
STARS: tuple[Star, ...] = (
    Star("Sirius",      "06h45m08.9s", "-16d42m58s", -1.46, -546.0, -1223.0),
    Star("Arcturus",    "14h15m39.7s", "+19d10m57s", -0.05, -1093.0, -2000.0),
    Star("Vega",        "18h36m56.3s", "+38d47m01s",  0.03,   201.0,   286.0),
    Star("Capella",     "05h16m41.4s", "+45d59m53s",  0.08,    76.0,  -427.0),
    Star("Rigel",       "05h14m32.3s", "-08d12m06s",  0.13,     1.0,    -0.0),
    Star("Procyon",     "07h39m18.1s", "+05d13m30s",  0.34,  -715.0, -1037.0),
    Star("Betelgeuse",  "05h55m10.3s", "+07d24m25s",  0.42,    27.0,    11.0),
    Star("Altair",      "19h50m47.0s", "+08d52m06s",  0.76,   537.0,   385.0),
    Star("Aldebaran",   "04h35m55.2s", "+16d30m33s",  0.86,    63.0,  -190.0),
    Star("Antares",     "16h29m24.5s", "-26d25m55s",  1.06,   -10.0,   -24.0),
    Star("Spica",       "13h25m11.6s", "-11d09m41s",  0.97,   -42.0,   -31.0),
    Star("Pollux",      "07h45m18.9s", "+28d01m34s",  1.14,  -626.0,   -46.0),
    Star("Fomalhaut",   "22h57m39.0s", "-29d37m20s",  1.16,   329.0,  -165.0),
    Star("Deneb",       "20h41m25.9s", "+45d16m49s",  1.25,     2.0,     1.0),
    Star("Regulus",     "10h08m22.3s", "+11d58m02s",  1.35,  -249.0,     5.0),
    Star("Castor",      "07h34m36.0s", "+31d53m18s",  1.58,  -206.0,  -149.0),
    Star("Bellatrix",   "05h25m07.9s", "+06d20m59s",  1.64,   -9.0,   -13.0),
    Star("Dubhe",       "11h03m43.7s", "+61d45m03s",  1.79,   -136.0,   -35.0),
    Star("Alioth",      "12h54m01.7s", "+55d57m35s",  1.77,   111.0,    -9.0),
    Star("Mirfak",      "03h24m19.4s", "+49d51m40s",  1.79,    24.0,   -26.0),
    Star("Polaris",     "02h31m49.1s", "+89d15m51s",  1.98,    44.0,   -12.0),
    Star("Hamal",       "02h07m10.4s", "+23d27m45s",  2.00,   190.0,  -148.0),
    Star("Alphard",     "09h27m35.2s", "-08d39m31s",  2.00,   -15.0,    34.0),
    Star("Alpheratz",   "00h08m23.3s", "+29d05m26s",  2.06,   136.0,   -163.0),
    Star("Rasalhague",  "17h34m56.1s", "+12d33m36s",  2.07,   110.0,  -222.0),
    Star("Kochab",      "14h50m42.3s", "+74d09m20s",  2.08,   -32.0,    12.0),
    Star("Denebola",    "11h49m03.6s", "+14d34m19s",  2.13,  -499.0,  -114.0),
    Star("Schedar",     "00h40m30.4s", "+56d32m14s",  2.23,    50.0,   -32.0),
    Star("Sadr",        "20h22m13.7s", "+40d15m24s",  2.23,     2.0,    -0.0),
    Star("Eltanin",     "17h56m36.4s", "+51d29m20s",  2.23,   -8.0,   -23.0),
    Star("Alphecca",    "15h34m41.3s", "+26d42m53s",  2.23,   120.0,    -89.0),
    Star("Mizar",       "13h23m55.5s", "+54d55m31s",  2.23,   121.0,   -22.0),
    Star("Scheat",      "23h03m46.5s", "+28d04m58s",  2.42,    187.0,    137.0),
    Star("Enif",        "21h44m11.2s", "+09d52m30s",  2.39,    30.0,     0.0),
    Star("Markab",      "23h04m45.7s", "+15d12m19s",  2.49,    61.0,   -42.0),
)
# fmt: on
BY_NAME = {s.name: s for s in STARS}


def get_star(name: str) -> Star:
    try:
        return BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown star {name!r}") from None


def visible_stars(
    site: EarthLocation,
    t: Time,
    *,
    min_alt: u.Quantity = 25 * u.deg,
    max_alt: u.Quantity = 80 * u.deg,
    exclude: tuple[str, ...] = (),
) -> list[Star]:
    """Catalog stars usable for alignment now, brightest first.

    The ceiling avoids the zenith, where an alt-az mount's azimuth axis is ill-conditioned.
    """
    out = []
    for s in STARS:
        if s.name in exclude:
            continue
        alt = radec_to_altaz(s.coord(t), site, t).alt
        if min_alt <= alt <= max_alt:
            out.append(s)
    return sorted(out, key=lambda s: s.vmag)


def partner_candidates(
    first: Star,
    site: EarthLocation,
    t: Time,
    *,
    min_separation: u.Quantity = 60 * u.deg,
    max_separation: u.Quantity = 150 * u.deg,
    exclude: tuple[str, ...] = (),
) -> list[Star]:
    """Visible stars 60-150 deg from ``first``, brightest first.

    Widely separated stars make the fitted pointing model well-conditioned; stars nearly
    opposite each other are as bad as close ones, hence the upper limit.
    """
    c1 = first.coord(t)
    return [
        s
        for s in visible_stars(site, t, exclude=(*exclude, first.name))
        if min_separation <= c1.separation(s.coord(t)) <= max_separation
    ]


def pick_alignment_pair(
    site: EarthLocation, t: Time, *, exclude: tuple[str, ...] = ()
) -> tuple[Star, Star]:
    """Brightest visible star, then its brightest well-separated partner."""
    for first in visible_stars(site, t, exclude=exclude):
        partners = partner_candidates(first, site, t, exclude=exclude)
        if partners:
            return first, partners[0]
    raise LookupError("no pair of visible bright stars with a usable separation")
