"""Local word-level speech recognition with faster-whisper.

Produces the same table tribev2's ``ExtractWordsFromAudio`` expects from its
transcription step, so it can stand in for tribev2's ``uvx whisperx`` call.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from . import pins

logger = logging.getLogger(__name__)

COLUMNS = ["text", "start", "duration", "sequence_id", "sentence"]


def _model_dir() -> str:
    from .model import local_snapshot

    return local_snapshot(pins.WHISPER)  # cache only: `cortexreel setup` downloads it


def _load_model():
    import torch
    from faster_whisper import WhisperModel

    path = _model_dir()
    if sys.platform == "win32":
        # CTranslate2 needs cuBLAS 12 / cuDNN 9 DLLs; the CUDA torch wheel ships them.
        os.add_dll_directory(str(Path(torch.__file__).parent / "lib"))
    if torch.cuda.is_available():
        try:
            return WhisperModel(path, device="cuda", compute_type="float16")
        except Exception as exc:  # missing cuDNN/cuBLAS for CTranslate2, etc.
            logger.warning("Whisper on GPU failed (%s); falling back to CPU", exc)
    return WhisperModel(path, device="cpu", compute_type="int8")


def _free(model) -> None:
    del model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def segments_to_words(segments) -> pd.DataFrame:
    """Flatten faster-whisper segments into one row per timed word."""
    rows = []
    for seq_id, segment in enumerate(segments):
        sentence = segment.text.strip().replace('"', "")
        for word in segment.words or ():
            if word.start is None or word.end is None:
                continue
            text = word.word.strip().replace('"', "")
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "start": float(word.start),
                    "duration": float(word.end - word.start),
                    "sequence_id": seq_id,
                    "sentence": sentence,
                }
            )
    return pd.DataFrame(rows, columns=COLUMNS)


def transcribe(audio_path: str | Path, language: str | None = None) -> pd.DataFrame:
    """Transcribe ``audio_path`` locally. ``language=None`` auto-detects.

    The Whisper model is released (and the CUDA cache emptied) before returning,
    so TRIBE's feature extractors get the whole GPU.
    """
    model = _load_model()
    try:
        segments, info = model.transcribe(
            str(audio_path), language=language, word_timestamps=True
        )
        words = segments_to_words(list(segments))  # generator: decoding runs here
        logger.info(
            "Transcribed %d words (language=%s, p=%.2f)",
            len(words),
            info.language,
            info.language_probability,
        )
        return words
    finally:
        _free(model)


def make_tribe_transcriber(language: str | None):
    """Return a drop-in for ``ExtractWordsFromAudio._get_transcript_from_audio``.

    tribev2 always passes its own language name (default ``"english"``); it is
    ignored in favour of ``language`` (an ISO code, or None to auto-detect).
    """

    def _get_transcript_from_audio(wav_filename, _tribe_language) -> pd.DataFrame:
        return transcribe(wav_filename, language)

    return _get_transcript_from_audio
