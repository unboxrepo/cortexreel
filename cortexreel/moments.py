"""Find dips, peaks, a weak hook and an ending drop in an engagement curve.

Thresholds are in engagement points (0-100) or seconds. MAD is the plain
median absolute deviation of the smoothed curve (not scaled to a sigma).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .score import ENDING_SECONDS, HOOK_SECONDS, Scores

DIP_MAD_K = 0.5  # dip: smoothed engagement < median - DIP_MAD_K * MAD
PEAK_MAD_K = 1.0  # peak: local max > median + PEAK_MAD_K * MAD
MAD_FLOOR = 1.0  # engagement points; keeps flat curves from flagging noise
DIP_MIN_SECONDS = 2.0
DIP_IGNORE_START_SECONDS = 1.0
DIP_MERGE_GAP_SECONDS = 1.0  # runs closer than this are merged
MAX_PEAKS = 5
PEAK_MIN_SEPARATION_SECONDS = 3.0
HOOK_MARGIN = 5.0  # hook_weak if hook < hold - HOOK_MARGIN
ENDING_MARGIN = 5.0  # ending_drop if ending < hold - ENDING_MARGIN
REGION_STD_FLOOR = 1e-6

DROP_REASONS = {
    "visual": "Visual activity drops: the frame may be static or low-contrast here.",
    "auditory": "Sound/voice energy drops.",
    "language": "Language response drops: speech may be thin or repetitive here.",
    "attention": "Attention network drops: nothing new to track.",
    "face_body": "Face/body response drops: no person or gesture to follow here.",
}
RISE_REASONS = {
    "visual": "Visual activity rises: strong motion, contrast or a scene change.",
    "auditory": "Sound/voice energy rises.",
    "language": "Language response rises: dense or meaningful speech.",
    "attention": "Attention network rises: something new to track.",
    "face_body": "Face/body response rises: a person or gesture on screen.",
}
NO_REGION_DROP = "Overall response is lower here; no single region stands out."
NO_REGION_RISE = "Overall response is higher here; no single region stands out."


@dataclass
class Moment:
    kind: str  # dip | peak | hook_weak | ending_drop
    start: float  # seconds
    end: float
    score: float  # mean engagement in the moment (peak: smoothed value at the peak)
    delta: float  # score minus the reference (video median, or hold for hook/ending)
    reason: str
    regions_delta: dict[str, float] = field(default_factory=dict)  # z vs region's video mean
    words: str = ""  # speech overlapping the moment

    def to_dict(self, ndigits: int = 2) -> dict:
        d = asdict(self)
        for key in ("start", "end", "score", "delta"):
            d[key] = round(float(d[key]), ndigits)
        d["regions_delta"] = {k: round(float(v), ndigits) for k, v in self.regions_delta.items()}
        return d


def load_words(path: str | Path):
    """``words.tsv`` as a DataFrame, or None if there is none."""
    import pandas as pd

    path = Path(path)
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    except pd.errors.EmptyDataError:
        return None


def words_between(words, start: float, end: float) -> str:
    if words is None or len(words) == 0:
        return ""
    w_start = words["start"].astype(float)
    w_end = w_start + words["duration"].astype(float)
    hit = (w_start < end) & (w_end > start)
    return " ".join(str(t) for t in words.loc[hit, "text"])


def _mad(x: np.ndarray) -> float:
    return max(float(np.median(np.abs(x - np.median(x)))), MAD_FLOOR)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive (first, last) index pairs of True runs."""
    edges = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), stops.tolist()))


def _regions_delta(scores: Scores, sel: np.ndarray) -> dict[str, float]:
    out = {}
    for name, curve in scores.region_curves.items():
        std = max(float(curve.std()), REGION_STD_FLOOR)
        out[name] = (float(curve[sel].mean()) - float(curve.mean())) / std
    return out


def _reason(scores: Scores, deltas: dict[str, float], falling: bool) -> str:
    # Only groups that feed engagement can explain a change in it.
    pool = {k: v for k, v in deltas.items() if scores.weights.get(k, 0) > 0} or deltas
    if falling:
        name = min(pool, key=pool.get)
        return DROP_REASONS.get(name, NO_REGION_DROP) if pool[name] < 0 else NO_REGION_DROP
    name = max(pool, key=pool.get)
    return RISE_REASONS.get(name, NO_REGION_RISE) if pool[name] > 0 else NO_REGION_RISE


def _window_moment(kind, scores, sel, start, end, reference, words) -> Moment:
    deltas = _regions_delta(scores, sel)
    score = float(scores.engagement[sel].mean())
    return Moment(
        kind=kind,
        start=start,
        end=end,
        score=score,
        delta=score - reference,
        reason=_reason(scores, deltas, falling=True),
        regions_delta=deltas,
        words=words_between(words, start, end),
    )


def find_dips(scores: Scores, words=None) -> list[Moment]:
    """Runs where the smoothed curve sits below median - DIP_MAD_K * MAD.

    Candidate runs come from the smoothed curve; their edges are then trimmed
    back to TRs whose unsmoothed engagement is also below the threshold, so
    smoothing does not widen a dip into its neighbours.
    """
    t, tr = scores.times, scores.tr_seconds
    smooth, eng = scores.engagement_smooth, scores.engagement
    median = float(np.median(smooth))
    threshold = median - DIP_MAD_K * _mad(smooth)
    below = (smooth < threshold) & (t >= DIP_IGNORE_START_SECONDS)

    spans: list[list[int]] = []
    for i0, i1 in _runs(below):
        while i0 < i1 and eng[i0] >= threshold:
            i0 += 1
        while i1 > i0 and eng[i1] >= threshold:
            i1 -= 1
        if spans and t[i0] - (t[spans[-1][1]] + tr) < DIP_MERGE_GAP_SECONDS:
            spans[-1][1] = i1
        else:
            spans.append([i0, i1])

    dips = []
    for i0, i1 in spans:
        start, end = float(t[i0]), float(t[i1] + tr)
        if end - start < DIP_MIN_SECONDS - 1e-9:
            continue
        sel = np.zeros(len(t), dtype=bool)
        sel[i0 : i1 + 1] = True
        dips.append(_window_moment("dip", scores, sel, start, end, median, words))
    return dips


def find_peaks(scores: Scores, words=None) -> list[Moment]:
    """Up to MAX_PEAKS local maxima above median + PEAK_MAD_K * MAD, spaced apart."""
    t, tr, smooth = scores.times, scores.tr_seconds, scores.engagement_smooth
    median = float(np.median(smooth))
    threshold = median + PEAK_MAD_K * _mad(smooth)
    n = len(smooth)
    candidates = []
    for i in range(n):
        left = smooth[i - 1] if i > 0 else -np.inf
        right = smooth[i + 1] if i < n - 1 else -np.inf
        if smooth[i] > threshold and smooth[i] >= left and smooth[i] >= right and (
            smooth[i] > left or smooth[i] > right
        ):
            candidates.append(i)

    chosen: list[int] = []
    for i in sorted(candidates, key=lambda i: -smooth[i]):
        if all(abs(t[i] - t[j]) >= PEAK_MIN_SEPARATION_SECONDS for j in chosen):
            chosen.append(i)
        if len(chosen) == MAX_PEAKS:
            break

    peaks = []
    for i in sorted(chosen):
        sel = np.zeros(n, dtype=bool)
        sel[i] = True
        deltas = _regions_delta(scores, sel)
        start, end = float(t[i]), float(t[i] + tr)
        peaks.append(
            Moment(
                kind="peak",
                start=start,
                end=end,
                score=float(smooth[i]),
                delta=float(smooth[i]) - median,
                reason=_reason(scores, deltas, falling=False),
                regions_delta=deltas,
                words=words_between(words, start, end),
            )
        )
    return peaks


def find_hook_and_ending(scores: Scores, words=None) -> list[Moment]:
    t, duration = scores.times, scores.duration
    out = []
    if scores.hook < scores.hold - HOOK_MARGIN:
        sel = t < HOOK_SECONDS
        out.append(
            _window_moment(
                "hook_weak", scores, sel, 0.0, min(HOOK_SECONDS, duration), scores.hold, words
            )
        )
    if scores.ending < scores.hold - ENDING_MARGIN:
        start = max(duration - ENDING_SECONDS, 0.0)
        sel = t >= start
        out.append(_window_moment("ending_drop", scores, sel, start, duration, scores.hold, words))
    return out


def find_moments(scores: Scores, words=None) -> list[Moment]:
    """All moments, sorted by start time."""
    moments = (
        find_hook_and_ending(scores, words) + find_dips(scores, words) + find_peaks(scores, words)
    )
    return sorted(moments, key=lambda m: (m.start, m.kind))
