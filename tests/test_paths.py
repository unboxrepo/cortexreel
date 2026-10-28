from pathlib import Path

import numpy as np
import pytest

from cortexreel.predict import Result, load_cached, output_dir, save_result


@pytest.fixture(autouse=True)
def in_tmp(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def test_output_dir_under_out(tmp_path):
    assert output_dir("videos/my clip.mp4") == (tmp_path / "out" / "my clip").resolve()
    assert output_dir(Path("/elsewhere/x.mov")) == (tmp_path / "out" / "x").resolve()


@pytest.mark.parametrize("bad", ["", ".", "..", "/"])
def test_output_dir_rejects_degenerate_names(bad):
    with pytest.raises(ValueError):
        output_dir(bad)


def test_save_refuses_outside_out(tmp_path):
    result = Result(preds=np.zeros((2, 3), np.float32), tr_seconds=1.0, times=np.arange(2.0))
    with pytest.raises(ValueError):
        save_result(result, tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        save_result(result, tmp_path / "out" / ".." / "escape")
    assert not (tmp_path / "elsewhere").exists()


def test_save_writes_npz_and_metrics(tmp_path):
    result = Result(
        preds=np.ones((2, 3), np.float32),
        tr_seconds=1.0,
        times=np.array([0.0, 1.0]),
        metrics={"gpu": "test"},
    )
    target = output_dir("clip.mp4")
    save_result(result, target)
    data = np.load(target / "preds.npz")
    assert data["preds"].dtype == np.float32
    assert data["preds"].shape == (2, 3)
    assert float(data["tr_seconds"]) == 1.0
    assert (target / "metrics.json").read_text(encoding="utf-8").strip().startswith("{")


def test_cache_requires_same_video_and_settings():
    import pandas as pd

    words = pd.DataFrame({"text": ["hi"], "start": [0.5], "duration": [0.2], "sentence": ["hi"]})
    metrics = {"video_sha256": "abc", "use_text": True, "precision": "fp16", "n_words": 1}
    result = Result(
        preds=np.ones((2, 3), np.float32),
        tr_seconds=1.0,
        times=np.array([0.0, 1.0]),
        metrics=metrics,
        words=words,
    )
    target = output_dir("clip.mp4")
    save_result(result, target)

    hit = load_cached(target, "abc", True, "fp16")
    assert hit is not None and hit.cached
    assert hit.preds.shape == (2, 3) and list(hit.words["text"]) == ["hi"]
    assert load_cached(target, "other", True, "fp16") is None
    assert load_cached(target, "abc", False, "fp16") is None
    assert load_cached(target, "abc", True, "fp32") is None
    (target / "words.tsv").unlink()
    assert load_cached(target, "abc", True, "fp16") is None
