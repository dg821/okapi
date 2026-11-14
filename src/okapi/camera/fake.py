"""Simulated camera for tests and dry runs."""

from collections.abc import Callable

from okapi.camera.base import CameraError, CapturedImage, ExposureSettings
from okapi.clock import Clock


class FakeCamera:
    def __init__(
        self,
        clock: Clock,
        *,
        image_for: Callable[[ExposureSettings], bytes] | None = None,
        fail_after: int | None = None,
    ) -> None:
        self._clock = clock
        self._image_for = image_for or (lambda s: b"FAKE-CR2")
        self._fail_after = fail_after
        self.captures: list[CapturedImage] = []

    def capture(self, settings: ExposureSettings) -> CapturedImage:
        if self._fail_after is not None and len(self.captures) >= self._fail_after:
            raise CameraError("simulated camera failure")
        start = self._clock.now()
        self._clock.sleep(settings.exposure_s)
        image = CapturedImage(self._image_for(settings), ".cr2", start, settings)
        self.captures.append(image)
        return image
