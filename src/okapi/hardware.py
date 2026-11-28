"""Wire the real drivers (SynScan serial, gphoto2, ASTAP, NMEA GPS) into one rig."""

from dataclasses import dataclass
from typing import Any

import astropy.units as u
import serial  # type: ignore[import-untyped]
from astropy.coordinates import SkyCoord

from okapi.camera.base import ExposureSettings
from okapi.camera.gphoto import GPhotoCamera
from okapi.clock import SystemClock
from okapi.mount.synscan import SynScanMount
from okapi.mount.transport import SerialTransport
from okapi.sensors.nmea import NmeaGps
from okapi.solve.astap import AstapSolver


class TargetRecorder:
    """Proxy for a mount that remembers the last RA/Dec it was sent to, so the plate solver can
    be hinted with where the telescope should be (the unaligned handset's own RA/Dec is
    meaningless). Everything else passes straight through."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.last_target: SkyCoord | None = None

    def slew_radec(self, target: SkyCoord) -> None:
        self.last_target = target
        self._inner.slew_radec(target)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class HardwareRig:
    clock: SystemClock
    mount: Any  # TargetRecorder around SynScanMount; satisfies MountProtocol
    driver: SynScanMount
    camera: GPhotoCamera
    solver: AstapSolver
    gps: NmeaGps
    settings: ExposureSettings
    _gps_serial: Any

    def close(self) -> None:
        self.driver.close()
        self._gps_serial.close()


def make_hardware_rig(
    *,
    mount_port: str,
    gps_port: str,
    exposure: u.Quantity = 2 * u.s,
    iso: int = 1600,
    astap_path: str = "astap_cli",
    gps_baud: int = 9600,
) -> HardwareRig:
    clock = SystemClock()
    driver = SynScanMount(SerialTransport(mount_port), clock)
    mount = TargetRecorder(driver)
    gps_serial = serial.serial_for_url(gps_port, baudrate=gps_baud, timeout=2.0)

    def next_line() -> str:
        return gps_serial.readline().decode("ascii", errors="ignore")

    return HardwareRig(
        clock=clock,
        mount=mount,
        driver=driver,
        camera=GPhotoCamera(clock),
        solver=AstapSolver(hint=lambda: mount.last_target, astap_path=astap_path),
        gps=NmeaGps(next_line),
        settings=ExposureSettings(exposure, iso),
        _gps_serial=gps_serial,
    )
