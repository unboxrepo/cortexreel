"""``splice``: one video from the strongest blocks of the compared versions.

splice plan (``"schema": 1``):

- ``schema``: int, 1.
- ``compare``: path of the ``compare.json`` the plan belongs to.
- ``segments``: ordered list of ``{source, start, end, why}``; ``source`` is
  the path of one of the compared versions, times are in that source's own
  timeline (seconds).
- ``notes``: list of str.

``splice.json`` (written next to ``splice-<a>-vs-<b>.mp4``) holds the plan, the output
path, its duration and ``segment_map``: one ``{source, name, start, end,
out_start, out_end}`` row per rendered segment.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .edit import END_TOLERANCE, MIN_SEGMENT_SECONDS, PlanError, finite, snap
from .outputs import prepare_file, same_file

SCHEMA = 1
MIN_TOTAL_SECONDS = 1.0
MIN_PIECE_SECONDS = 0.8  # a switch that leaves a shorter piece keeps the previous source
MAX_MISMATCH_SECONDS = 0.4  # content repeated or skipped at a switch before it falls back
JOIN_TOLERANCE = 0.05
DEFAULT_FPS = 30.0
AUDIO_FORMAT = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
AUTO_NOTE = (
    "Auto splicing assumes the versions share structure and script (the same lines at about "
    "the same times, endings aligned to each version's end). Review every boundary against each "
    "source's words.tsv and watch the result: nothing may be cut mid-word, repeated or dropped."
)


def _key(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


# --------------------------------------------------------------------- auto plan


@dataclass
class Piece:
    name: str
    start: float  # source time
    end: float
    off_start: float  # source time = absolute time + offset (non-zero for an end-aligned ending)
    off_end: float
    ending: bool  # ends with the end-aligned ending block
    blocks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _block_pieces(cmp: dict, durations: dict[str, float]) -> list[Piece]:
    """One piece per block from its winner (a tie stays with the previous source)."""
    longest = max(durations.values())
    pieces: list[Piece] = []
    prev = None
    for b in cmp["blocks"]:
        name = b.get("winner")
        if not name:
            continue
        if b.get("tie") and prev in b.get("tied", []):
            name = prev
        d = durations[name]
        if b["align"] == "end":
            off = d - longest
            start, end = max(d - (b["end"] - b["start"]), 0.0), d
        else:
            off = 0.0
            start, end = b["start"], min(b["end"], d)
        if end - start < MIN_SEGMENT_SECONDS:
            continue
        pieces.append(Piece(name, start, end, off, off, b["align"] == "end", [b["id"]]))
        prev = name
    return pieces


def _merge_same_source(pieces: list[Piece]) -> list[Piece]:
    out: list[Piece] = []
    for p in pieces:
        q = out[-1] if out else None
        if q and q.name == p.name and p.start <= q.end + JOIN_TOLERANCE:
            q.end, q.off_end, q.ending = max(q.end, p.end), p.off_end, p.ending
            q.blocks += p.blocks
            q.notes += p.notes
        else:
            out.append(p)
    return out


def _fallback_end(q: Piece, p: Piece, durations: dict[str, float]) -> float | None:
    """Where ``q``'s source would end if it also covered ``p``; None if it is too short."""
    d = durations[q.name]
    if p.ending:  # q takes its own ending
        return d
    return min(p.end, d) if p.end <= d + END_TOLERANCE else None


def auto_plan(cmp: dict, words: dict[str, list]) -> dict:
    """Take each block from its winner, merge runs, snap every switch to a word gap.

    ``words`` maps version name to a ``edit.word_list`` (empty if unknown). A
    switch falls back to the previous source when snapping leaves a piece
    shorter than MIN_PIECE_SECONDS or repeats/skips more than
    MAX_MISMATCH_SECONDS of the shared timeline.
    """
    versions = {v["name"]: v for v in cmp["versions"]}
    durations = {n: float(v["duration_seconds"]) for n, v in versions.items()}
    pieces = _merge_same_source(_block_pieces(cmp, durations))
    notes = [AUTO_NOTE]
    out: list[Piece] = []
    for p in pieces:
        if not out:
            out.append(p)
            continue
        q = out[-1]
        if q.name == p.name and p.start <= q.end + JOIN_TOLERANCE:
            q.end, q.off_end, q.ending = max(q.end, p.end), p.off_end, p.ending
            q.blocks += p.blocks
            continue
        wq, wp = words.get(q.name) or [], words.get(p.name) or []
        t_q = snap(q.end, wq, durations[q.name], lo=q.start, hi=durations[q.name])
        target = min(max(t_q - q.off_end + p.off_start, 0.0), p.end)
        t_p = snap(target, wp, durations[p.name], lo=0.0, hi=p.end)
        mismatch = (t_q - q.off_end) - (t_p - p.off_start)  # > 0 repeats, < 0 skips
        problem = None
        if abs(mismatch) > MAX_MISMATCH_SECONDS:
            problem = f"{'repeats' if mismatch > 0 else 'skips'} {abs(mismatch):.2f} s at the switch"
        elif p.end - t_p < MIN_PIECE_SECONDS or t_q - q.start < MIN_PIECE_SECONDS:
            problem = "piece too short after snapping to word gaps"
        if problem:
            end = _fallback_end(q, p, durations)
            if end is not None:
                q.end, q.ending = end, p.ending
                q.off_end = durations[q.name] - max(durations.values()) if p.ending else 0.0
                q.notes.append(f"kept {q.name} for {', '.join(p.blocks)} (won by {p.name}): {problem}")
                q.blocks += p.blocks
                continue
            notes.append(
                f"Switch to {p.name} at {p.start:.2f} s {problem}, but {q.name} is too short to "
                "keep: review this boundary."
            )
        q.end, p.start = t_q, t_p
        out.append(p)
    last = out[-1] if out else None
    if last and last.end < durations[last.name] - END_TOLERANCE:
        wl = words.get(last.name) or []
        last.end = snap(last.end, wl, durations[last.name], lo=last.start)

    segments = []
    for p in out:
        scores = []
        for bid in p.blocks:
            b = next(b for b in cmp["blocks"] if b["id"] == bid)
            s = b["scores"].get(p.name)
            scores.append(f"{bid} {'–' if s is None else f'{s:.0f}'}")
        why = f"{p.name}: " + ", ".join(scores)
        if p.notes:
            why += "; " + "; ".join(p.notes)
        segments.append(
            {"source": versions[p.name]["path"], "start": round(p.start, 3),
             "end": round(p.end, 3), "why": why}
        )  # fmt: skip
    if len({s["source"] for s in segments}) == 1 and segments:
        notes.append(
            f"Every block comes from {out[0].name}: a splice adds nothing over that version."
        )
    return {"schema": SCHEMA, "segments": segments, "notes": notes}


# --------------------------------------------------------------------- validate


def validate_splice(plan: dict, versions: list[dict]) -> list[dict]:
    """Checked segments (``name`` added, ends clamped to the source duration); raises PlanError.

    ``versions`` are compare.json versions: ``name``, ``path``, ``duration_seconds``.
    """
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise PlanError(f'splice plan must be a JSON object with "schema": {SCHEMA}')
    raw = plan.get("segments")
    if not isinstance(raw, list) or not raw:
        raise PlanError('splice plan needs a non-empty "segments" list')
    by_path = {_key(v["path"]): v for v in versions}
    out, total = [], 0.0
    for i, seg in enumerate(raw):
        where = f"segment {i + 1}"
        if not isinstance(seg, dict):
            raise PlanError(f"{where}: must be an object")
        source = seg.get("source")
        version = by_path.get(_key(source)) if isinstance(source, str) and source else None
        if version is None:
            names = ", ".join(v["name"] for v in versions)
            raise PlanError(f"{where}: source must be one of the compared versions ({names})")
        try:
            start, end = finite(seg["start"]), finite(seg["end"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"{where}: needs finite numeric start and end (seconds)") from None
        duration = float(version["duration_seconds"])
        if start < 0 or end <= start:
            raise PlanError(f"{where}: needs 0 <= start < end, got {start:g}-{end:g}")
        if end > duration + END_TOLERANCE:
            raise PlanError(
                f"{where}: ends at {end:g} s, past {version['name']}'s {duration:g} s"
            )
        end = min(end, duration)
        if end - start < MIN_SEGMENT_SECONDS:
            raise PlanError(f"{where}: shorter than {MIN_SEGMENT_SECONDS:g} s")
        out.append({**seg, "source": version["path"], "name": version["name"], "start": start, "end": end})
        total += end - start
    if total < MIN_TOTAL_SECONDS:
        raise PlanError(f"splice is {total:.2f} s long; at least {MIN_TOTAL_SECONDS:g} s needed")
    return out


# --------------------------------------------------------------------- render


@dataclass
class Source:
    path: str
    fps: float | None
    has_audio: bool


def quantize(segments: list[dict], sources: dict[str, Source]) -> list[dict]:
    """Segment edges on each source's frame grid so video and audio pieces match."""
    out = []
    for seg in segments:
        fps = sources[_key(seg["source"])].fps
        a, b = seg["start"], seg["end"]
        if fps:
            a, b = round(a * fps) / fps, round(b * fps) / fps
        if b - a >= MIN_SEGMENT_SECONDS:
            out.append({**seg, "start": a, "end": b})
    return out


def segment_map(segments: list[dict]) -> list[dict]:
    rows, t = [], 0.0
    for seg in segments:
        length = seg["end"] - seg["start"]
        rows.append(
            {"source": seg["source"], "name": seg.get("name"), "start": round(seg["start"], 3),
             "end": round(seg["end"], 3), "out_start": round(t, 3), "out_end": round(t + length, 3)}
        )  # fmt: skip
        t += length
    return rows


def build_filter(
    segments: list[dict], inputs: list[Source], size: tuple[int, int], fps: float
) -> tuple[str, bool]:
    """``-filter_complex`` trimming each segment from its input, scaled/padded to ``size``.

    Returns the graph (outputs ``[outv]`` and, if any input has audio, ``[outa]``)
    and whether it has audio. Inputs without audio contribute silence.
    """
    n = len(segments)
    if n == 0:
        raise PlanError("splice keeps nothing")
    w, h = size
    index = {_key(s.path): k for k, s in enumerate(inputs)}
    with_audio = any(s.has_audio for s in inputs)
    uses: dict[int, list[int]] = {}
    for i, seg in enumerate(segments):
        uses.setdefault(index[_key(seg["source"])], []).append(i)
    parts = []
    for k, idx in uses.items():
        parts.append(f"[{k}:v]split={len(idx)}" + "".join(f"[vs{i}]" for i in idx))
        if with_audio and inputs[k].has_audio:
            parts.append(f"[{k}:a]asplit={len(idx)}" + "".join(f"[as{i}]" for i in idx))
    for i, seg in enumerate(segments):
        k = index[_key(seg["source"])]
        a, b = seg["start"], seg["end"]
        parts.append(
            f"[vs{i}]trim=start={a:.6f}:end={b:.6f},setpts=PTS-STARTPTS,"
            f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps:g},format=yuv420p[v{i}]"
        )
        if with_audio:
            if inputs[k].has_audio:
                parts.append(f"[as{i}]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS,{AUDIO_FORMAT}[a{i}]")
            else:
                parts.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={b - a:.6f},{AUDIO_FORMAT}[a{i}]")
    pads = "".join(f"[v{i}][a{i}]" if with_audio else f"[v{i}]" for i in range(n))
    outs = "[outv][outa]" if with_audio else "[outv]"
    parts.append(f"{pads}concat=n={n}:v=1:a={1 if with_audio else 0}{outs}")
    return ";".join(parts), with_audio


def render_command(
    segments: list[dict], inputs: list[Source], size: tuple[int, int], fps: float, out_path: str | Path
) -> list[str]:
    from .media import ffmpeg_exe, input_args

    graph, with_audio = build_filter(segments, inputs, size, fps)
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    for s in inputs:
        cmd += input_args(s.path)
    cmd += [
        "-filter_complex", graph,
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
    ]  # fmt: skip
    if with_audio:
        cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "192k"]
    return cmd + ["-movflags", "+faststart", str(out_path)]


@dataclass
class SpliceResult:
    output: Path | None
    duration: float
    segment_map: list[dict]


def _even(x: int) -> int:
    return x + (x % 2)


def render(
    plan: dict, versions: list[dict], out_path: str | Path | None, dry_run: bool = False
) -> SpliceResult:
    """Validate ``plan`` and render it in one ffmpeg run (or only map it, with ``dry_run``).

    Output size and frame rate come from the first segment's source.
    """
    from .media import probe

    segments = validate_splice(plan, versions)
    sources: dict[str, Source] = {}
    infos = {}
    for seg in segments:
        key = _key(seg["source"])
        if key not in sources:
            path = str(Path(seg["source"]).resolve())
            info = probe(path)
            infos[key] = info
            sources[key] = Source(path, info.fps, info.has_audio)
    segments = quantize(segments, sources)
    if sum(s["end"] - s["start"] for s in segments) < MIN_TOTAL_SECONDS:
        raise PlanError(f"splice is shorter than {MIN_TOTAL_SECONDS:g} s")
    rows = segment_map(segments)
    duration = rows[-1]["out_end"] if rows else 0.0
    if dry_run or out_path is None:
        return SpliceResult(None, duration, rows)

    out_path = Path(out_path).resolve()
    if any(same_file(out_path, p) for p in [s.path for s in sources.values()] + [v["path"] for v in versions]):
        raise PlanError("refusing to overwrite a source video")
    first = infos[_key(segments[0]["source"])]
    if not (first.width and first.height):
        raise PlanError(f"{segments[0]['source']} has no video stream")
    size = (_even(first.width), _even(first.height))
    fps = first.fps or DEFAULT_FPS
    prepare_file(out_path)
    done = subprocess.run(
        render_command(segments, list(sources.values()), size, fps, out_path),
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.decode(errors='replace').strip()[-1500:]}")
    return SpliceResult(out_path, duration, rows)
