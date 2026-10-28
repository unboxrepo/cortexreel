"""Edit plans: suggest cuts/speed-ups from an analysis, validate them, render with ffmpeg.

plan.json (``"schema": 1``):

- ``schema``: int, 1.
- ``source``: path of the video the plan applies to; ``source_sha256``.
- ``source_duration``: seconds (probed from the file, not n_tr * tr).
- ``ops``: list of ``{start, end, action, factor?, why}`` sorted by ``start``,
  non-overlapping. ``action`` is ``cut`` (remove) or ``speed`` (play
  ``factor`` times faster, 1.05-2.0, pitch kept with ``atempo``).
- ``notes``: list of str, suggestions the plan does not apply (e.g. a weak hook).

Times are in the source's timeline. ``timemap`` rows are
``[old_start, old_end, new_start, new_end]`` for every kept segment.
"""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

SCHEMA = 1
ACTIONS = ("cut", "speed")
MIN_FACTOR, MAX_FACTOR = 1.05, 2.0
MIN_KEEP_SECONDS = 1.0
MAX_REMOVED_FRACTION = 0.5
END_TOLERANCE = 0.05  # ops may end this far past the probed duration (clamped)

HOOK_PROTECT_SECONDS = 3.0  # suggest_plan never touches the opening
WORD_PAD = 0.12  # keep op boundaries this far from a word edge when the gap allows
SENTENCE_SNAP_SECONDS = 1.0
DIP_SPEED, SHORT_DIP_SPEED, SHORT_DIP_SECONDS = 1.25, 1.15, 3.0
ENDING_SPEED = 1.2
ENDING_SILENCE_MIN, ENDING_TAIL_KEEP = 0.6, 0.4
MIN_OP_SECONDS = {"cut": 0.3, "speed": 0.5}
MIN_SEGMENT_SECONDS = 0.01
SENTENCE_END = (".", "?", "!")


class PlanError(ValueError):
    pass


# --------------------------------------------------------------------- words


@dataclass
class Word:
    text: str
    start: float
    end: float
    sentence: str = ""


def word_list(words) -> list[Word]:
    """Words from a ``words.tsv`` DataFrame or a list of dicts, sorted by start."""
    if words is None:
        return []
    rows = words.to_dict("records") if hasattr(words, "to_dict") else list(words)
    out = [
        Word(
            text=str(r.get("text", "")),
            start=float(r["start"]),
            end=float(r["start"]) + float(r["duration"]),
            sentence=str(r.get("sentence", "")),
        )
        for r in rows
    ]
    return sorted(out, key=lambda w: w.start)


@dataclass
class Gap:
    lo: float  # earliest allowed boundary inside the gap
    hi: float
    sentence: bool  # a sentence ends before this gap (or it is a video edge)


def word_gaps(words: list[Word], duration: float) -> list[Gap]:
    """Silent stretches where a boundary cuts no word, padded away from word edges."""
    if not words:
        return [Gap(0.0, duration, True)]

    def gap(a: float, b: float, pad_a: bool, pad_b: bool, sentence: bool) -> Gap:
        lo, hi = a + (WORD_PAD if pad_a else 0.0), b - (WORD_PAD if pad_b else 0.0)
        if lo > hi:  # narrow (or overlapping) gap: its middle is the only option
            lo = hi = (a + b) / 2
        return Gap(lo, hi, sentence)

    gaps = [gap(0.0, words[0].start, False, True, True)]
    for w, nxt in zip(words, words[1:]):
        ends_sentence = w.text.rstrip().endswith(SENTENCE_END) or (
            bool(w.sentence) and w.sentence != nxt.sentence
        )
        gaps.append(gap(w.end, nxt.start, True, True, ends_sentence))
    gaps.append(gap(words[-1].end, max(duration, words[-1].end), True, False, True))
    return gaps


def snap(t: float, words: list[Word], duration: float, lo: float = 0.0, hi: float | None = None) -> float:
    """Move ``t`` off any spoken word, staying within [lo, hi].

    A time already in a gap only moves away from a word edge by up to WORD_PAD.
    A time inside a word goes to the nearest gap, or to a sentence boundary if
    one is within SENTENCE_SNAP_SECONDS.
    """
    hi = duration if hi is None else hi
    gaps = [g for g in word_gaps(words, duration) if g.hi >= lo and g.lo <= hi]
    if not gaps:
        return min(max(t, lo), hi)

    def point(g: Gap) -> float:
        return min(max(t, g.lo, lo), g.hi, hi)

    inside_word = any(w.start < t < w.end for w in words)
    if not inside_word:
        for g in gaps:
            if g.lo - WORD_PAD <= t <= g.hi + WORD_PAD:
                return point(g)
    sentence = [g for g in gaps if g.sentence and abs(point(g) - t) <= SENTENCE_SNAP_SECONDS]
    return point(min(sentence or gaps, key=lambda g: abs(point(g) - t)))


def words_in(words: list[Word], start: float, end: float) -> list[Word]:
    return [w for w in words if w.start < end and w.end > start]


# --------------------------------------------------------------------- suggest


def _subtract(spans: list[tuple[float, float]], holes: list[tuple[float, float]]):
    out = []
    for a, b in spans:
        pieces = [(a, b)]
        for h0, h1 in holes:
            next_pieces = []
            for p0, p1 in pieces:
                if h1 <= p0 or h0 >= p1:
                    next_pieces.append((p0, p1))
                    continue
                if h0 > p0:
                    next_pieces.append((p0, h0))
                if h1 < p1:
                    next_pieces.append((h1, p1))
            pieces = next_pieces
        out.extend(pieces)
    return out


def _moment_ref(m: dict) -> str:
    return (
        f"{m['kind']} {m['start']:.1f}-{m['end']:.1f} s, score {m['score']:.0f} "
        f"({m['delta']:+.1f})"
    )


def suggest_plan(analysis: dict, words=None, duration: float | None = None) -> dict:
    """A conservative, deterministic plan from ``analysis.json`` and ``words.tsv``.

    Silent dips are cut, dips with speech are sped up, a silent tail after the
    last word is cut. The first HOOK_PROTECT_SECONDS and every peak are left
    alone; a weak hook only produces a note. ``words=None`` means the speech is
    unknown, so dips are sped up rather than cut.
    """
    duration = float(duration if duration is not None else analysis["video"]["duration_seconds"])
    known_speech = words is not None
    wl = word_list(words)
    moments = analysis.get("moments", [])
    peaks = [(float(m["start"]), float(m["end"])) for m in moments if m["kind"] == "peak"]
    notes: list[str] = []
    cuts: list[dict] = []
    speeds: list[dict] = []

    for m in moments:
        if m["kind"] == "hook_weak":
            gap = -float(m["delta"])
            notes.append(
                f"Hook scores {gap:.0f} points under the middle ({m['score']:.0f} vs "
                f"{m['score'] + gap:.0f}). Not changed automatically: consider opening on the "
                "demo footage or the strongest moment instead of a title card."
            )
            continue
        start, end = max(float(m["start"]), HOOK_PROTECT_SECONDS), min(float(m["end"]), duration)
        if end <= start:
            continue
        if m["kind"] == "dip":
            spoken = words_in(wl, start, end)
            if known_speech and not spoken:
                cuts.append({"start": start, "end": end, "why": f"{_moment_ref(m)}; no speech"})
            else:
                factor = SHORT_DIP_SPEED if end - start < SHORT_DIP_SECONDS else DIP_SPEED
                speeds.append(
                    {
                        "start": start,
                        "end": end,
                        "factor": factor,
                        "why": f"{_moment_ref(m)}; "
                        + ("has speech, sped up instead of cut" if spoken else "speech unknown"),
                    }
                )
        elif m["kind"] == "ending_drop":
            last_end = wl[-1].end if wl else None
            if last_end is not None and duration - last_end >= ENDING_SILENCE_MIN:
                cut_start = max(last_end + ENDING_TAIL_KEEP, HOOK_PROTECT_SECONDS)
                if cut_start < duration:
                    cuts.append(
                        {
                            "start": cut_start,
                            "end": duration,
                            "why": f"{_moment_ref(m)}; silent tail after the last word "
                            f"({last_end:.2f} s)",
                        }
                    )
            else:
                speeds.append(
                    {
                        "start": start,
                        "end": end,
                        "factor": ENDING_SPEED,
                        "why": f"{_moment_ref(m)}; speech runs to the end, sped up",
                    }
                )

    # Snap to word gaps, keep peaks, let cuts win over speed-ups, drop slivers.
    ops: list[dict] = []
    taken: list[tuple[float, float]] = []
    for group in (cuts, speeds):
        for op in group:
            a = snap(op["start"], wl, duration, lo=HOOK_PROTECT_SECONDS)
            b = op["end"] if op["end"] >= duration else snap(op["end"], wl, duration, lo=a)
            for p0, p1 in _subtract([(a, b)], peaks + taken):
                action = "cut" if group is cuts else "speed"
                if p1 - p0 < MIN_OP_SECONDS[action]:
                    continue
                new = {"start": round(p0, 3), "end": round(p1, 3), "action": action}
                if action == "speed":
                    new["factor"] = op["factor"]
                new["why"] = op["why"]
                ops.append(new)
                taken.append((p0, p1))
    ops.sort(key=lambda o: o["start"])
    if not known_speech:
        notes.append("No transcript (words.tsv) found: dips were sped up, not cut.")
    if not ops:
        notes.append("Nothing to change automatically.")

    video = analysis.get("video", {})
    return {
        "schema": SCHEMA,
        "source": video.get("path"),
        "source_sha256": video.get("sha256"),
        "source_duration": round(duration, 3),
        "ops": ops,
        "notes": notes,
    }


# --------------------------------------------------------------------- validate


def finite(value) -> float:
    """``float(value)``; ValueError for NaN/inf (``json.loads`` accepts them) and bools."""
    if isinstance(value, bool):
        raise ValueError(f"not a number: {value!r}")
    x = float(value)
    if not math.isfinite(x):
        raise ValueError(f"not a finite number: {value!r}")
    return x


def validate_plan(plan: dict, duration: float, allow_heavy: bool = False) -> list[dict]:
    """Checked copy of ``plan['ops']`` (ends clamped to ``duration``); raises PlanError."""
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise PlanError(f'plan must be a JSON object with "schema": {SCHEMA}')
    raw_ops = plan.get("ops")
    if not isinstance(raw_ops, list):
        raise PlanError('plan needs an "ops" list')
    ops = []
    for i, op in enumerate(raw_ops):
        where = f"op {i + 1}"
        if not isinstance(op, dict):
            raise PlanError(f"{where}: must be an object")
        try:
            start, end = finite(op["start"]), finite(op["end"])
        except (KeyError, TypeError, ValueError):
            raise PlanError(f"{where}: needs finite numeric start and end (seconds)") from None
        action = op.get("action")
        if action not in ACTIONS:
            raise PlanError(f"{where}: action must be one of {', '.join(ACTIONS)}, got {action!r}")
        if start < 0 or end <= start:
            raise PlanError(f"{where}: needs 0 <= start < end, got {start:g}-{end:g}")
        if end > duration + END_TOLERANCE:
            raise PlanError(f"{where}: ends at {end:g} s, past the video's {duration:g} s")
        end = min(end, duration)
        if start >= duration:
            raise PlanError(f"{where}: starts at {start:g} s, past the video's {duration:g} s")
        checked = {**op, "start": start, "end": end}
        if action == "speed":
            try:
                factor = finite(op["factor"])
            except (KeyError, TypeError, ValueError):
                raise PlanError(f"{where}: speed needs a finite numeric factor") from None
            if not MIN_FACTOR <= factor <= MAX_FACTOR:
                raise PlanError(
                    f"{where}: speed factor {factor:g} outside {MIN_FACTOR:g}-{MAX_FACTOR:g}"
                )
            checked["factor"] = factor
        if ops:
            prev = ops[-1]
            if start < prev["start"]:
                raise PlanError(f"{where}: ops must be sorted by start ({start:g} < {prev['start']:g})")
            if start < prev["end"] - 1e-6:
                raise PlanError(
                    f"{where}: overlaps op {i} ({start:g}-{end:g} vs {prev['start']:g}-{prev['end']:g})"
                )
        ops.append(checked)

    new_duration = sum((b - a) / f for a, b, f in segments(ops, duration))
    if new_duration < MIN_KEEP_SECONDS:
        raise PlanError(f"plan leaves {new_duration:.2f} s; at least {MIN_KEEP_SECONDS:g} s must remain")
    removed = 1 - new_duration / duration
    if removed > MAX_REMOVED_FRACTION and not allow_heavy:
        raise PlanError(
            f"plan removes {removed:.0%} of the video (limit {MAX_REMOVED_FRACTION:.0%}); "
            "pass --allow-heavy to allow it"
        )
    return ops


# --------------------------------------------------------------------- segments


def segments(ops: list[dict], duration: float) -> list[tuple[float, float, float]]:
    """Kept ``(start, end, factor)`` pieces of the source, in order."""
    out = []
    t = 0.0
    for op in ops:
        if op["start"] - t >= MIN_SEGMENT_SECONDS:
            out.append((t, op["start"], 1.0))
        if op["action"] == "speed" and op["end"] - op["start"] >= MIN_SEGMENT_SECONDS:
            out.append((op["start"], op["end"], float(op["factor"])))
        t = max(t, op["end"])
    if duration - t >= MIN_SEGMENT_SECONDS:
        out.append((t, duration, 1.0))
    return out


def timemap(segs: list[tuple[float, float, float]]) -> list[list[float]]:
    rows, new_t = [], 0.0
    for a, b, f in segs:
        length = (b - a) / f
        rows.append([round(a, 3), round(b, 3), round(new_t, 3), round(new_t + length, 3)])
        new_t += length
    return rows


def map_time(tmap: list[list[float]], t: float) -> float | None:
    """Source time -> recut time; None if ``t`` was cut."""
    for a, b, na, nb in tmap:
        if a <= t <= b:
            return na + (t - a) * (nb - na) / (b - a) if b > a else na
    return None


def _quantize(segs, fps: float | None):
    """Snap segment edges to the frame grid so video and audio pieces match in length."""
    if not fps:
        return segs
    out = []
    for a, b, f in segs:
        a, b = round(a * fps) / fps, round(b * fps) / fps
        if b - a >= MIN_SEGMENT_SECONDS:
            out.append((a, b, f))
    return out


def build_filter(
    segs: list[tuple[float, float, float]], has_audio: bool, fps: float | None = None
) -> str:
    """``-filter_complex`` that trims ``segs`` (speeding where factor > 1) and concats them.

    Outputs ``[outv]`` and, with audio, ``[outa]``.
    """
    n = len(segs)
    if n == 0:
        raise PlanError("plan keeps nothing of the video")
    parts = [f"[0:v]split={n}" + "".join(f"[vs{i}]" for i in range(n))]
    if has_audio:
        parts.append(f"[0:a]asplit={n}" + "".join(f"[as{i}]" for i in range(n)))
    for i, (a, b, f) in enumerate(segs):
        pts = "PTS-STARTPTS" if f == 1.0 else f"(PTS-STARTPTS)/{f:g}"
        parts.append(f"[vs{i}]trim=start={a:.6f}:end={b:.6f},setpts={pts}[v{i}]")
        if has_audio:
            tempo = f",atempo={f:g}" if f != 1.0 else ""
            parts.append(
                f"[as{i}]atrim=start={a:.6f}:end={b:.6f},asetpts=PTS-STARTPTS{tempo}[a{i}]"
            )
    inputs = "".join(f"[v{i}][a{i}]" if has_audio else f"[v{i}]" for i in range(n))
    concat_out = "[cv][outa]" if has_audio else "[cv]"
    parts.append(f"{inputs}concat=n={n}:v=1:a={1 if has_audio else 0}{concat_out}")
    parts.append(f"[cv]fps={fps:g}[outv]" if fps else "[cv]null[outv]")
    return ";".join(parts)


def render_command(
    source: str | Path, out_path: str | Path, segs, has_audio: bool, fps: float | None
) -> list[str]:
    from .media import ffmpeg_exe, input_args

    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        *input_args(source),
        "-filter_complex", build_filter(segs, has_audio, fps),
        "-map", "[outv]",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
    ]  # fmt: skip
    if has_audio:
        cmd += ["-map", "[outa]", "-c:a", "aac", "-b:a", "192k"]
    return cmd + ["-movflags", "+faststart", str(out_path)]


@dataclass
class RenderResult:
    output: Path
    old_duration: float
    new_duration: float
    timemap: list[list[float]]


def render(plan: dict, out_path: str | Path, allow_heavy: bool = False) -> RenderResult:
    """Apply ``plan`` to ``plan['source']`` in one ffmpeg run; returns the timemap."""
    from .media import probe
    from .outputs import prepare_file, same_file

    source = Path(plan["source"]).resolve()
    out_path = Path(out_path)
    if same_file(out_path, source):
        raise PlanError("refusing to overwrite the source video")
    info = probe(source)
    ops = validate_plan(plan, info.duration, allow_heavy=allow_heavy)
    segs = _quantize(segments(ops, info.duration), info.fps)
    prepare_file(out_path)
    done = subprocess.run(
        render_command(source, out_path, segs, info.has_audio, info.fps),
        capture_output=True,
        check=False,
    )
    if done.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {done.stderr.decode(errors='replace').strip()[-1500:]}")
    tmap = timemap(segs)
    return RenderResult(out_path, info.duration, tmap[-1][3] if tmap else 0.0, tmap)


def plan_summary(
    plan: dict, duration: float, fps: float | None = None, allow_heavy: bool = False
) -> tuple[float, list]:
    """(new duration, timemap) of a plan without rendering; matches ``render``."""
    ops = validate_plan(plan, duration, allow_heavy=allow_heavy)
    tmap = timemap(_quantize(segments(ops, duration), fps))
    return (tmap[-1][3] if tmap else 0.0), tmap
