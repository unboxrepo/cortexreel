import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cortexreel import overlay
from cortexreel.media import ffmpeg_exe, probe

REGIONS = ("visual", "auditory", "language", "attention", "face_body")


# --------------------------------------------------------------------- layout


@pytest.mark.parametrize("compare", [False, True])
def test_bands_fill_the_frame(compare):
    bands = list(overlay.layout(compare).values())
    assert sum(b.h for b in bands) == overlay.H == 1920
    assert bands[0].y == 0
    for above, below in zip(bands, bands[1:]):
        assert below.y == above.y + above.h
    assert ("brain" in overlay.layout(compare)) is not compare


def test_fit_portrait_becomes_centered_column():
    assert overlay.fit(1080, 1920, 1080, 1080) == (608, 1080, 236, 0)


def test_fit_landscape_fills_width():
    assert overlay.fit(1920, 1080, 1080, 1080) == (1080, 608, 0, 236)


def test_fit_compare_column_and_even_sizes():
    assert overlay.fit(1080, 1920, 540, 1080) == (540, 960, 0, 60)
    w, h, x, y = overlay.fit(1001, 777, 540, 1080)
    assert w % 2 == 0 and h % 2 == 0 and w <= 540 and h <= 1080


# --------------------------------------------------------------------- timing


def test_value_at_interpolates_between_tr_centers():
    centers = np.array([0.5, 1.5, 2.5])
    values = np.array([10.0, 30.0, 20.0])
    assert overlay.value_at(centers, values, 1.0) == pytest.approx(20.0)
    assert overlay.value_at(centers, values, 2.0) == pytest.approx(25.0)
    assert overlay.value_at(centers, values, 0.0) == 10.0  # held before the first center
    assert overlay.value_at(centers, values, 9.0) == 20.0  # and after the last


def test_tr_blend_crossfades_between_centers():
    assert overlay.tr_blend(4, 0.0, 1.0, 0.5) == (0, 1, 0.0)
    i, j, a = overlay.tr_blend(4, 0.0, 1.0, 1.25)
    assert (i, j) == (0, 1) and a == pytest.approx(0.75)
    assert overlay.tr_blend(4, 0.0, 1.0, 0.0) == (0, 1, 0.0)
    assert overlay.tr_blend(4, 0.0, 1.0, 99.0) == (3, 3, 0.0)


def test_moment_at_prefers_dips_and_ignores_other_kinds():
    moments = [
        {"kind": "hook_weak", "start": 0.0, "end": 3.0, "reason": "x"},
        {"kind": "peak", "start": 4.0, "end": 6.0, "reason": "p"},
        {"kind": "dip", "start": 5.0, "end": 8.0, "reason": "d"},
    ]
    assert overlay.moment_at(moments, 1.0) is None
    assert overlay.moment_at(moments, 4.5)["kind"] == "peak"
    assert overlay.moment_at(moments, 5.5)["kind"] == "dip"
    assert overlay.moment_at(moments, 8.0) is None  # end is exclusive


def test_headline():
    assert overlay.headline("Visual activity drops: the frame may be static.") == "Visual activity drops"
    assert overlay.headline("Sound/voice energy drops.") == "Sound/voice energy drops"
    assert (
        overlay.headline("Overall response is lower here; no single region stands out.")
        == "Overall response is lower here"
    )


# --------------------------------------------------------------------- smoke


def _make_video(path, size, seconds):
    subprocess.run(
        [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        check=True,
    )  # fmt: skip


def _make_analysis(stem, n_tr):
    out_dir = Path("out") / stem
    out_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    times = np.arange(n_tr, dtype=float)
    analysis = {
        "video": {"name": f"{stem}.mp4", "sha256": None, "tr_seconds": 1.0},
        "summary": {"overall": 60.0, "hook": 55.0, "hold": 62.0, "ending": 58.0},
        "moments": [
            {"kind": "dip", "start": 1.0, "end": 2.0, "score": 40.0, "delta": -12.0,
             "reason": "Visual activity drops: the frame may be static."},
        ],  # fmt: skip
        "curves": {
            "times": times.tolist(),
            "engagement": rng.uniform(30, 90, n_tr).round(3).tolist(),
            "regions": {r: rng.normal(size=n_tr).tolist() for r in REGIONS},
        },
    }
    (out_dir / "analysis.json").write_text(json.dumps(analysis), encoding="utf-8")
    np.savez(out_dir / "preds.npz", preds=rng.normal(size=(n_tr, 8)), times=times, tr_seconds=1.0)


@pytest.fixture
def fake_brain(monkeypatch):
    def frames(preds, cache_dir):
        return [Image.new("RGB", (120, 60), (40 * i % 255, 80, 160)) for i in range(len(preds))]

    monkeypatch.setattr(overlay, "brain_frames", frames)


def test_overlay_smoke(tmp_path, monkeypatch, fake_brain):
    monkeypatch.chdir(tmp_path)
    _make_video(tmp_path / "clip.mp4", "360x640", 3)
    _make_analysis("clip", 3)
    result = overlay.render(tmp_path / "clip.mp4")
    assert result.output == (tmp_path / "out" / "clip" / "clip-overlay.mp4").resolve()
    info = probe(result.output)
    assert (info.width, info.height) == (1080, 1920)
    assert abs(info.duration - 3.0) <= 0.15
    assert info.audio_codec == "aac"


def test_compare_runs_until_the_longer_video_ends(tmp_path, monkeypatch, fake_brain):
    monkeypatch.chdir(tmp_path)
    _make_video(tmp_path / "before.mp4", "360x640", 3)
    _make_video(tmp_path / "after.mp4", "640x360", 2)  # shorter and landscape
    _make_analysis("before", 3)
    _make_analysis("after", 2)
    result = overlay.render(tmp_path / "before.mp4", other=tmp_path / "after.mp4", watermark=False)
    assert result.output.name == "before-compare.mp4"
    info = probe(result.output)
    assert (info.width, info.height) == (1080, 1920)
    assert abs(info.duration - 3.0) <= 0.15


def test_missing_analysis_names_the_analyze_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    with pytest.raises(overlay.OverlayError, match="cortexreel analyze"):
        overlay.render(video)


@pytest.mark.slow
def test_brain_images_render_with_nilearn():
    rng = np.random.default_rng(1)
    preds = rng.normal(scale=0.2, size=(2, 20484)).astype(np.float32)
    images = list(overlay.render_brain_images(preds, vmax=0.5))
    assert len(images) == 2
    for img in images:
        assert img.width <= overlay.BRAIN_BOX[0] and img.height <= overlay.BRAIN_BOX[1]
        assert max(img.size[0] / overlay.BRAIN_BOX[0], img.size[1] / overlay.BRAIN_BOX[1]) > 0.99
    assert np.asarray(images[0]).std() > 10  # drawn, not a blank background
    assert images[0].tobytes() != images[1].tobytes()
