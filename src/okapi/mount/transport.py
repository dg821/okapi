"""Byte transport to the hand control: a real serial port, or a simulator in tests.

Replies end with ``#``, but some replies carry binary payloads (location, time, model) in which
a byte of value 35 *is* ``#``: a latitude of 35 degrees would end the read early. So binary
replies are read by exact length, and only ASCII replies by "up to ``#``".
"""

from typing import Protocol

from okapi.mount.base import MountError


class TransportTimeoutError(MountError):
    """The handset did not answer in time."""


class Transport(Protocol):
    def write(self, data: bytes) -> None:
        """Send one command. Anything left unread from before is discarded first."""
        ...

    def read_reply(self, timeout: float, length: int | None = None) -> bytes:
        """The reply: exactly ``length`` bytes if given, else everything up to and including ``#``.
        Raises ``TransportTimeoutError``."""
        ...

    def close(self) -> None: ...


class SerialTransport:
    """9600 8N1, as the protocol document specifies. ``port`` may be a device path or a
    pyserial URL such as ``loop://`` (used in tests)."""

    def __init__(self, port: str, *, baudrate: int = 9600) -> None:
        import serial

        self._serial = serial.serial_for_url(
            port, baudrate=baudrate, bytesize=8, parity="N", stopbits=1, timeout=1.0
        )

    def write(self, data: bytes) -> None:
        self._serial.reset_input_buffer()
        self._serial.write(data)
        self._serial.flush()

    def read_reply(self, timeout: float, length: int | None = None) -> bytes:
        self._serial.timeout = timeout
        data = self._serial.read(length) if length else self._serial.read_until(b"#")
        if (length and len(data) < length) or (not length and not data.endswith(b"#")):
            raise TransportTimeoutError(f"no complete reply within {timeout:.1f} s (got {data!r})")
        return bytes(data)

    def close(self) -> None:
        self._serial.close()
