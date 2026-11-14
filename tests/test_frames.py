import json
import os

import astropy.units as u
import pytest
from astropy.time import Time

from okapi.camera.base import CapturedImage, ExposureSettings
from okapi.coords import AltAzPosition
from okapi.frames import FrameMetadata, FrameWriter, make_frame_id, metadata_for
from okapi.mount.fake import FakeMount


def make_image(clock) -> CapturedImage:
    return CapturedImage(b"RAWDATA", ".cr2", clock.now(), ExposureSettings(15 * u.s, 1600))


@pytest.fixture
def mount(clock, site):
    m = FakeMount(clock)
    m.set_site(site)
    m.slew_altaz(AltAzPosition(120 * u.deg, 50 * u.deg))
    m.wait_until_settled(60 * u.s)
    return m


def test_frame_id_format_and_sequence():
    t = Time("2026-06-21T04:05:06.7", scale="utc")
    assert make_frame_id(t, 0) == "20260621T040506_0000"
    assert make_frame_id(t, 12) == "20260621T040506_0012"
    with pytest.raises(ValueError):
        make_frame_id(t, -1)


def test_metadata_snapshots_mount_and_site(clock, site, mount):
    meta = metadata_for(make_image(clock), frame_id="f1", mount=mount, site=site)
    assert meta.exposure_s == 15.0
    assert meta.iso == 1600
    assert meta.mount_az_deg == pytest.approx(120.0, abs=1e-6)
    assert meta.mount_alt_deg == pytest.approx(50.0, abs=1e-6)
    assert meta.site_lat_deg == pytest.approx(40.0, abs=1e-6)
    assert meta.site_lon_deg == pytest.approx(-105.0, abs=1e-6)
    assert meta.tracking is False


def test_metadata_json_roundtrip(clock, site, mount):
    meta = metadata_for(make_image(clock), frame_id="f1", mount=mount, site=site)
    assert FrameMetadata.from_json(meta.to_json()) == meta


def test_write_layout_and_contents(tmp_path, clock, site, mount):
    image = make_image(clock)
    meta = metadata_for(image, frame_id="f1", mount=mount, site=site)
    path = FrameWriter(tmp_path).write(image, meta)
    assert path == tmp_path / "raw" / "2026" / "06" / "21" / "f1.cr2"
    assert path.read_bytes() == b"RAWDATA"
    sidecar = json.loads(path.with_suffix(".json").read_text())
    assert sidecar["frame_id"] == "f1"
    assert sidecar["start_time_utc"] == image.start_time.utc.isot
    assert not list(path.parent.glob("*.part"))


def test_interrupted_sidecar_write_leaves_image_without_sidecar(
    tmp_path, clock, site, mount, monkeypatch
):
    image = make_image(clock)
    meta = metadata_for(image, frame_id="f1", mount=mount, site=site)
    real_replace = os.replace
    calls = []

    def flaky_replace(src, dst):
        calls.append(dst)
        if str(dst).endswith(".json"):
            raise OSError("disk pulled")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError):
        FrameWriter(tmp_path).write(image, meta)
    out = tmp_path / "raw" / "2026" / "06" / "21"
    assert (out / "f1.cr2").exists()  # image is complete...
    assert not (out / "f1.json").exists()  # ...and the missing sidecar marks it as interrupted
    assert [str(c).rsplit(".", 1)[-1] for c in calls] == ["cr2", "json"]  # image first
