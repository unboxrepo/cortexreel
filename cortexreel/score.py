"""TRIBE predictions -> per-region curves and a 0-100 engagement curve.

Pure numpy. ``times[i]`` is the stimulus time of TR ``i``: TRIBE already
compensates the hemodynamic lag, so no shift is applied here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOOK_SECONDS = 3.0
ENDING_SECONDS = 3.0
SMOOTH_SIGMA_TR = 1.0


@dataclass
class Scores:
    times: np.ndarray  # (n_TR,) seconds
    tr_seconds: float
    region_curves: dict[str, np.ndarray]  # mean predicted activation per TR
    weights: dict[str, float]
    engagement_raw: np.ndarray  # weighted sum of region curves
    engagement: np.ndarray  # 0-100, fixed mapping, onset-corrected
    engagement_uncorrected: np.ndarray  # 0-100, before onset correction
    engagement_smooth: np.ndarray  # for moment detection
    hook: float  # mean engagement, first HOOK_SECONDS
    hold: float  # mean engagement between the hook and ending windows
    ending: float  # mean engagement, last ENDING_SECONDS
    overall: float

    @property
    def duration(self) -> float:
        return float(self.times[-1] + self.tr_seconds) if len(self.times) else 0.0

    def summary(self) -> dict[str, float]:
        return {
            "overall": self.overall,
            "hook": self.hook,
            "hold": self.hold,
            "ending": self.ending,
        }


def to_engagement(raw: np.ndarray, center: float, scale: float) -> np.ndarray:
    """``100 * sigmoid((raw - center) / scale)``; same raw value -> same score."""
    z = (np.asarray(raw, dtype=np.float64) - center) / scale
    return 100.0 / (1.0 + np.exp(-z))


def correct_onset(engagement: np.ndarray, points: list[float]) -> np.ndarray:
    """Add back the model's warm-up deficit over the first ``len(points)`` TRs."""
    out = np.asarray(engagement, dtype=np.float64).copy()
    n = min(len(out), len(points))
    out[:n] += np.asarray(points[:n], dtype=np.float64)
    return np.clip(out, 0.0, 100.0)


def gaussian_smooth(x: np.ndarray, sigma: float) -> np.ndarray:
    """Centered Gaussian smoothing with edge values repeated past both ends."""
    x = np.asarray(x, dtype=np.float64)
    if sigma <= 0 or len(x) < 2:
        return x.copy()
    radius = int(np.ceil(3 * sigma))
    k = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (k / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(x, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _mean(x: np.ndarray, mask: np.ndarray) -> float:
    return float(x[mask].mean()) if mask.any() else float("nan")


def window_means(
    engagement: np.ndarray, times: np.ndarray, tr_seconds: float
) -> dict[str, float]:
    """Hook / hold / ending / overall means.

    A TR belongs to a window if its start time falls inside it. ``hold`` is the
    stretch between the hook and the ending; for videos too short to have one
    it falls back to everything after the hook, then to the whole video.
    """
    engagement = np.asarray(engagement, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    duration = float(times[-1] + tr_seconds)
    in_hook = times < HOOK_SECONDS
    in_ending = times >= duration - ENDING_SECONDS
    in_hold = ~in_hook & ~in_ending
    if not in_hold.any():
        in_hold = ~in_hook
    if not in_hold.any():
        in_hold = np.ones_like(in_hook)
    return {
        "overall": float(engagement.mean()),
        "hook": _mean(engagement, in_hook),
        "hold": _mean(engagement, in_hold),
        "ending": _mean(engagement, in_ending),
    }


def compute_scores(
    preds: np.ndarray,
    times: np.ndarray,
    tr_seconds: float = 1.0,
    masks: dict[str, np.ndarray] | None = None,
    config: dict | None = None,
) -> Scores:
    """Score ``preds`` (n_TR, n_vertices). ``masks``/``config`` default to the atlas and regions.json."""
    from . import regions

    config = config or regions.load_config()
    if masks is None:
        masks = regions.region_masks(config)
    preds = np.asarray(preds, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    if preds.ndim != 2 or preds.shape[0] != len(times) or len(times) == 0:
        raise ValueError(f"preds {preds.shape} and times {times.shape} do not match")

    weights = regions.weights(config)
    curves = {name: preds[:, mask].mean(axis=1) for name, mask in masks.items()}
    raw = np.zeros(len(times))
    for name, w in weights.items():
        if w:
            raw += w * curves[name]

    mapping = config["engagement"]
    uncorrected = to_engagement(raw, float(mapping["center"]), float(mapping["scale"]))
    engagement = correct_onset(uncorrected, config.get("onset_correction", {}).get("points", []))
    smooth = gaussian_smooth(engagement, SMOOTH_SIGMA_TR)
    summary = window_means(engagement, times, tr_seconds)
    return Scores(
        times=times,
        tr_seconds=float(tr_seconds),
        region_curves=curves,
        weights=weights,
        engagement_raw=raw,
        engagement=engagement,
        engagement_uncorrected=uncorrected,
        engagement_smooth=smooth,
        **summary,
    )
