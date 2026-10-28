"""ffmpeg helpers (the binary ships with imageio-ffmpeg)."""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


# Inputs are user files (or files derived from them). A crafted "video" must
# not make ffmpeg read URLs or other local files: only the file and pipe
# protocols, and only real video demuxers (a playlist or an ffconcat list
# named .mp4 would otherwise open other files). Nested opens inherit both.
PROTOCOL_WHITELIST = "file,pipe"
VIDEO_FORMATS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi"  # the suffixes predict accepts


def input_args(source: str | Path, formats: str = VIDEO_FORMATS) -> list[str]:
    """``-protocol_whitelist ... -format_whitelist ... -i <source>``; every ffmpeg input goes through this."""
    return ["-protocol_whitelist", PROTOCOL_WHITELIST, "-format_whitelist", formats, "-i", str(source)]


@dataclass
class Probe:
    duration: float  # container duration, seconds
    fps: float | None
    width: int | None
    height: int | None
    has_audio: bool
    audio_codec: str | None = None  # first audio stream, e.g. "aac"


def probe(video: str | Path) -> Probe:
    """Duration, frame rate, size and audio stream, parsed from ``ffmpeg -i``."""
    done = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", *input_args(video)], capture_output=True, check=False
    )
    text = done.stderr.decode(errors="replace")
    m = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not m:
        raise RuntimeError(f"ffmpeg could not read {video}: {text.strip()[-500:]}")
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
    fps = width = height = None
    video_line = re.search(r"Stream #\S+.*?: Video: (.*)", text)
    if video_line:
        line = video_line[1]
        rate = re.search(r"(\d+(?:\.\d+)?) fps", line) or re.search(r"(\d+(?:\.\d+)?) tbr", line)
        fps = float(rate[1]) if rate else None
        size = re.search(r", (\d{2,5})x(\d{2,5})", line)
        if size:
            width, height = int(size[1]), int(size[2])
    audio = re.search(r"Stream #\S+.*?: Audio: (\w+)", text)
    return Probe(duration, fps, width, height, audio is not None, audio[1] if audio else None)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


PROXY_SHORT_EDGE = 320


def make_proxy(video: str | Path, dest: str | Path) -> Path:
    """Downscale ``video`` so its short edge is PROXY_SHORT_EDGE px, audio untouched.

    V-JEPA2 resizes to a 292 px short edge before cropping anyway, so the model
    sees almost the same pixels (spatial corr 0.996 on a test reel), while
    decoding 64 small frames per clip instead of 1080p ones saves ~25% time.
    """
    from .outputs import prepare_file

    dest = prepare_file(dest)
    edge = PROXY_SHORT_EDGE
    scale = f"scale='if(lt(iw,ih),{edge},-2)':'if(lt(iw,ih),-2,{edge})'"
    base = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", *input_args(video), "-vf", scale]
    video_args = ["-c:v", "libx264", "-crf", "16", "-preset", "veryfast"]
    for audio_args in (["-c:a", "copy"], ["-c:a", "aac", "-b:a", "192k"]):
        done = subprocess.run(base + video_args + audio_args + [str(dest)], capture_output=True)
        if done.returncode == 0:
            return dest
    raise RuntimeError(f"ffmpeg could not make a proxy of {video}: {done.stderr.decode(errors='replace')}")


def frame_jpeg(video: str | Path, seconds: float, width: int = 320) -> bytes | None:
    """One JPEG frame at ``seconds``, scaled to ``width`` px; None if ffmpeg fails."""
    cmd = [
        ffmpeg_exe(),
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{max(seconds, 0.0):.3f}",
        *input_args(video),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2",
        "-q:v", "4",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-",
    ]  # fmt: skip
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=60, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0 or not done.stdout.startswith(b"\xff\xd8"):
        return None
    return done.stdout
