"""Write each frame as RAW + sidecar JSON, atomically, under ``raw/YYYY/MM/DD/``.

The image is written before its sidecar, so a sidecar always has a complete image beside it;
an image without a sidecar is an interrupted write and should be ignored by later processing.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import astropy.units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from okapi import __version__
from okapi.camera.base import CapturedImage
from okapi.mount.base import MountProtocol


@dataclass(frozen=True)
class FrameMetadata:
    frame_id: str
    start_time_utc: str  # ISO 8601, UTC
    exposure_s: float
    iso: int
    mount_ra_deg: float  # what the mount *reports*; may differ from the true pointing
    mount_dec_deg: float
    mount_az_deg: float
    mount_alt_deg: float
    tracking: bool
    site_lat_deg: float
    site_lon_deg: float
    site_height_m: float
    okapi_version: str = __version__

    @property
    def start_time(self) -> Time:
        return Time(self.start_time_utc, format="isot", scale="utc")

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "FrameMetadata":
        return cls(**json.loads(text))


def make_frame_id(start_time: Time, sequence: int) -> str:
    """``YYYYMMDDTHHMMSS_NNNN``; the sequence number keeps frames in the same second distinct."""
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    return f"{start_time.utc.strftime('%Y%m%dT%H%M%S')}_{sequence:04d}"


def metadata_for(
    image: CapturedImage, *, frame_id: str, mount: MountProtocol, site: EarthLocation
) -> FrameMetadata:
    """Snapshot of the mount and site for one frame. Call right after the exposure."""
    radec = mount.get_radec()
    altaz = mount.get_altaz()
    ra, dec = cast(u.Quantity, radec.ra), cast(u.Quantity, radec.dec)
    return FrameMetadata(
        frame_id=frame_id,
        start_time_utc=image.start_time.utc.isot,
        exposure_s=image.settings.exposure_s,
        iso=image.settings.iso,
        mount_ra_deg=float(ra.to_value(u.deg)),
        mount_dec_deg=float(dec.to_value(u.deg)),
        mount_az_deg=float(altaz.az.to_value(u.deg)),
        mount_alt_deg=float(altaz.alt.to_value(u.deg)),
        tracking=mount.is_tracking,
        site_lat_deg=float(site.lat.to_value(u.deg)),
        site_lon_deg=float(site.lon.to_value(u.deg)),
        site_height_m=float(site.height.to_value(u.m)),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class FrameWriter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def directory_for(self, metadata: FrameMetadata) -> Path:
        d = metadata.start_time
        return self.root / "raw" / d.strftime("%Y") / d.strftime("%m") / d.strftime("%d")

    def write(self, image: CapturedImage, metadata: FrameMetadata) -> Path:
        directory = self.directory_for(metadata)
        directory.mkdir(parents=True, exist_ok=True)
        image_path = directory / f"{metadata.frame_id}{image.extension}"
        _atomic_write(image_path, image.data)
        _atomic_write(image_path.with_suffix(".json"), metadata.to_json().encode())
        return image_path
