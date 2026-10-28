"""Video -> TRIBE v2 cortical predictions, with timing and VRAM metrics."""

from __future__ import annotations

import json
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from .outputs import OUT_ROOT, OutputError, is_link, out_root, prepare_file, rel_path, write_text

VIDEO_SUFFIXES = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
WORD_COLUMNS = ["text", "start", "duration", "sentence"]
TRIBE_SIDECARS = (".wav", ".tsv")  # audio and transcript tribev2 saves next to the proxy


@dataclass
class Result:
    preds: np.ndarray  # (n_TR, n_vertices) float32
    tr_seconds: float
    times: np.ndarray  # start time (s) of each predicted TR
    metrics: dict = field(default_factory=dict)
    words: pd.DataFrame | None = None  # word events fed to TRIBE (WORD_COLUMNS)
    cached: bool = False


def output_dir(video: str | Path, out_root: Path = OUT_ROOT) -> Path:
    """``out/<video-stem>`` under the CWD; refuses anything that escapes ``out/`` or is a link."""
    if is_link(Path.cwd() / out_root):
        raise OutputError(f"Refusing to write through a link: {Path.cwd() / out_root}")
    root = (Path.cwd() / out_root).resolve()
    stem = Path(video).stem
    if not stem or stem in (".", ".."):
        raise ValueError(f"Cannot derive an output name from {video!r}")
    if is_link(root / stem):
        raise OutputError(f"Refusing to write through a link: {root / stem}")
    target = (root / stem).resolve()
    if target.parent != root:
        raise ValueError(f"Refusing to write outside {root}: {target}")
    return target


@contextmanager
def _patched_transcriber(language: str | None):
    from tribev2.eventstransforms import ExtractWordsFromAudio

    from .transcribe import make_tribe_transcriber

    original = ExtractWordsFromAudio.__dict__["_get_transcript_from_audio"]
    ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(
        make_tribe_transcriber(language)
    )
    try:
        yield
    finally:
        ExtractWordsFromAudio._get_transcript_from_audio = original


def _build_events(video: Path, use_text: bool, language: str | None):
    import pandas as pd
    from tribev2.demo_utils import get_audio_and_text_events

    event = {
        "type": "Video",
        "filepath": str(video),
        "start": 0,
        "timeline": "default",
        "subject": "default",
    }
    with _patched_transcriber(language):
        return get_audio_and_text_events(
            pd.DataFrame([event]), audio_only=not use_text
        )


def _word_table(events):
    import pandas as pd

    words = events.loc[events.type == "Word"]
    if words.empty:
        return None
    table = pd.DataFrame({col: words[col] if col in words else "" for col in WORD_COLUMNS})
    return table.sort_values("start", kind="stable").reset_index(drop=True)


def load_cached(
    out_dir: Path, video_sha256: str, use_text: bool, precision: str
) -> Result | None:
    """The saved result in ``out_dir`` if it was made from the same video and settings."""
    import pandas as pd

    preds_file, metrics_file = out_dir / "preds.npz", out_dir / "metrics.json"
    if not (preds_file.is_file() and metrics_file.is_file()):
        return None
    try:
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if (
        metrics.get("video_sha256") != video_sha256
        or metrics.get("use_text") != use_text
        or metrics.get("precision") != precision
    ):
        return None
    words_file = out_dir / "words.tsv"
    words = None
    if metrics.get("n_words"):
        if not words_file.is_file():
            return None
        words = pd.read_csv(words_file, sep="\t", keep_default_na=False)
    with np.load(preds_file) as data:
        return Result(
            preds=data["preds"].astype(np.float32, copy=False),
            tr_seconds=float(data["tr_seconds"]),
            times=data["times"].astype(np.float64),
            metrics=metrics,
            words=words,
            cached=True,
        )


def predict_video(
    path: str | Path,
    use_text: bool = True,
    language: str | None = None,
    device: str = "auto",
    save: bool = True,
    precision: str = "fp16",
    force: bool = False,
) -> Result:
    """Predict the brain response to ``path``.

    With ``save``, results go to ``out/<stem>/`` and are reused on the next call
    for the same video bytes, ``use_text`` and ``precision`` unless ``force``.
    """
    from .media import make_proxy, sha256_file

    video = Path(path).resolve()
    if video.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"Unsupported video type {video.suffix!r}: {sorted(VIDEO_SUFFIXES)}")
    if not video.is_file():
        raise FileNotFoundError(video)
    out_dir = output_dir(video) if save else None
    video_sha256 = sha256_file(video)
    if save and not force:
        cached = load_cached(out_dir, video_sha256, use_text, precision)
        if cached is not None:
            return cached

    import torch

    from .model import load_tribe

    cuda = torch.cuda.is_available()
    metrics: dict = {
        "video": rel_path(video),
        "video_sha256": video_sha256,
        "use_text": use_text,
        "precision": precision,
        "language": language or "auto",
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "torch": torch.__version__,
    }

    t0 = time.perf_counter()
    model = load_tribe(device=device, use_text=use_text, precision=precision)
    metrics["seconds_load"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    # Small proxy: faster decoding, and tribev2's wav/tsv sidecars land next to
    # it in out/ instead of next to the user's video.
    proxy_dir = out_dir if out_dir is not None else Path(tempfile.mkdtemp(prefix="cortexreel-"))
    proxy = make_proxy(video, proxy_dir / "proxy.mp4")
    for sidecar in TRIBE_SIDECARS:  # tribev2 writes (and would reuse) these: start clean
        prepare_file(proxy.with_suffix(sidecar))
    metrics["seconds_proxy"] = round(time.perf_counter() - t0, 2)

    t0 = time.perf_counter()
    events = _build_events(proxy, use_text, language)
    metrics["seconds_events_asr"] = round(time.perf_counter() - t0, 2)
    words = _word_table(events)
    metrics["n_words"] = 0 if words is None else len(words)

    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    preds, segments = model.predict(events, verbose=True)
    metrics["seconds_features_predict"] = round(time.perf_counter() - t0, 2)
    if cuda:
        # Peak of torch allocations during features+predict. Whisper runs in
        # CTranslate2 before this and is not counted here.
        metrics["cuda_max_memory_allocated_gb"] = round(
            torch.cuda.max_memory_allocated() / 1024**3, 3
        )

    result = Result(
        preds=np.asarray(preds, dtype=np.float32),
        tr_seconds=float(model.data.TR),
        times=np.array([float(s.start) for s in segments], dtype=np.float64),
        metrics=metrics,
        words=words,
    )
    metrics["n_tr"], metrics["n_vertices"] = map(int, result.preds.shape)
    if save:
        save_result(result, out_dir)
    return result


def save_result(result: Result, out_dir: Path) -> None:
    root = out_root()
    out_dir = out_dir.resolve()
    if not out_dir.is_relative_to(root):
        raise ValueError(f"Refusing to write outside {root}: {out_dir}")
    metrics_file = out_dir / "metrics.json"
    prepare_file(metrics_file)  # an old metrics.json must not vouch for half-written files
    np.savez_compressed(
        prepare_file(out_dir / "preds.npz"),
        preds=result.preds,
        times=result.times,
        tr_seconds=np.float64(result.tr_seconds),
    )
    words_file = prepare_file(out_dir / "words.tsv")  # also drops a stale one from a text run
    if result.words is not None:
        result.words.to_csv(words_file, sep="\t", index=False)
    # Written last: a metrics.json with this video's hash marks a complete cache.
    write_text(metrics_file, json.dumps(result.metrics, indent=2))
