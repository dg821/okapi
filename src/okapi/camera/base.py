"""Camera interface. All sequencing code depends on this, never on gphoto2 directly."""

from dataclasses import dataclass
from typing import Protocol

import astropy.units as u
from astropy.time import Time


class CameraError(RuntimeError):
    """The camera failed to take or deliver an exposure."""


@dataclass(frozen=True)
class ExposureSettings:
    exposure: u.Quantity
    iso: int

    def __post_init__(self) -> None:
        if self.exposure.to_value(u.s) <= 0:
            raise ValueError("exposure must be positive")
        if self.iso <= 0:
            raise ValueError("iso must be positive")

    @property
    def exposure_s(self) -> float:
        return float(self.exposure.to_value(u.s))


@dataclass(frozen=True)
class CapturedImage:
    data: bytes
    extension: str  # e.g. ".cr2"
    start_time: Time  # system clock just before the shutter command
    settings: ExposureSettings


class CameraProtocol(Protocol):
    def capture(self, settings: ExposureSettings) -> CapturedImage:
        """Take one exposure and return the RAW bytes. Raises ``CameraError`` on failure."""
        ...
