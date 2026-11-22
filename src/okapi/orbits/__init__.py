"""Orbit determination from sidereal-stare observations (analytic two-body dynamics).

Internals use plain floats: kilometres, seconds, radians, in the GCRS (geocentric, axes aligned
with ICRS). Astropy units and Time appear at the public boundary. Not modelled: J2 and other
perturbations (fine over an arc of hours for IOD, wrong over days), light time, aberration,
atmospheric refraction, and the observer's velocity (only positions are needed)."""

MU_EARTH = 398600.4418  # km^3/s^2
R_EARTH = 6378.137  # km
