import numpy as np
import pandas as pd
import pytest

from cortexreel import moments as mom
from cortexreel.regions import load_config
from cortexreel.score import compute_scores, to_engagement, window_means

CONFIG = load_config()
# Synthetic curves test the mapping and moment logic; the measured onset
# correction is tested on its own below.
CONFIG = {**CONFIG, "onset_correction": {"points": []}}
GROUPS = list(CONFIG["groups"])
PER_GROUP = 4  # synthetic vertices per group, disjoint


def _masks():
    n = PER_GROUP * len(GROUPS)
    masks = {}
    for i, name in enumerate(GROUPS):
        m = np.zeros(n, dtype=bool)
        m[i * PER_GROUP : (i + 1) * PER_GROUP] = True
        masks[name] = m
    return masks


def _scores(group_curves: dict[str, np.ndarray], n_tr: int):
    """Scores from per-group activation curves (groups not given stay at 0)."""
    preds = np.zeros((n_tr, PER_GROUP * len(GROUPS)), np.float32)
    for i, name in enumerate(GROUPS):
        if name in group_curves:
            preds[:, i * PER_GROUP : (i + 1) * PER_GROUP] = group_curves[name][:, None]
    return compute_scores(preds, np.arange(n_tr, dtype=float), 1.0, _masks(), CONFIG)


def _scores_for_engagement(engagement: np.ndarray):
    """Every weighted group gets the activation that maps to ``engagement``."""
    c, s = CONFIG["engagement"]["center"], CONFIG["engagement"]["scale"]
    p = np.asarray(engagement, float) / 100
    raw = c + s * np.log(p / (1 - p))
    return _scores({name: raw for name in GROUPS}, len(raw))


def test_weights_sum_to_one_so_shared_activation_is_raw():
    scores = _scores_for_engagement(np.full(10, 70.0))
    assert np.allclose(scores.engagement, 70.0)


def test_planted_dip_found_with_exact_bounds():
    e = np.full(40, 60.0)
    e[10:14] = 25.0  # TRs starting at 10..13 -> stimulus 10-14 s
    dips = mom.find_dips(_scores_for_engagement(e))
    assert len(dips) == 1
    assert (dips[0].start, dips[0].end) == (10.0, 14.0)
    assert dips[0].delta < -20


def test_short_and_early_dips_ignored():
    e = np.full(40, 60.0)
    e[20] = 20.0  # 1 s only
    e[0] = 20.0  # inside the ignored first second
    assert mom.find_dips(_scores_for_engagement(e)) == []


def test_dip_reason_names_the_region_that_dropped():
    n = 30
    curves = {name: np.full(n, 0.05) for name in GROUPS}
    curves["visual"] = curves["visual"].copy()
    curves["visual"][12:16] = -0.6
    dips = mom.find_dips(_scores(curves, n))
    assert len(dips) == 1
    assert dips[0].reason == mom.DROP_REASONS["visual"]
    assert min(dips[0].regions_delta, key=dips[0].regions_delta.get) == "visual"


def test_planted_peaks():
    e = np.full(40, 50.0)
    e[8] = e[20] = e[30] = 85.0
    e[22] = 75.0  # within 3 s of the stronger peak at 20
    peaks = mom.find_peaks(_scores_for_engagement(e))
    assert [p.start for p in peaks] == [8.0, 20.0, 30.0]
    assert all(p.kind == "peak" and p.delta > 0 for p in peaks)


def test_at_most_five_peaks():
    e = np.full(60, 50.0)
    e[2::4] = 90.0
    peaks = mom.find_peaks(_scores_for_engagement(e))
    assert len(peaks) == mom.MAX_PEAKS
    starts = [p.start for p in peaks]
    assert all(b - a >= mom.PEAK_MIN_SEPARATION_SECONDS for a, b in zip(starts, starts[1:]))


def test_fixed_mapping_is_comparable_across_videos():
    a = np.array([0.0, 0.1, -0.2, 0.05, 0.3])
    b = np.array([0.0, -0.4, 0.5, 0.05, 0.9])  # different range, same values at 0 and 3
    ea = _scores({n: a for n in GROUPS}, len(a)).engagement
    eb = _scores({n: b for n in GROUPS}, len(b)).engagement
    assert ea[0] == pytest.approx(eb[0]) == pytest.approx(50.0)
    assert ea[3] == pytest.approx(eb[3])
    # Not min-max: the top of each video is not forced to 100.
    assert ea.max() < 99 and eb.max() != pytest.approx(ea.max())
    assert to_engagement(np.array([0.1]), 0.0, 0.1)[0] == pytest.approx(100 / (1 + np.exp(-1)))


def test_hook_hold_ending_math():
    e = np.full(20, 50.0)
    e[:3] = 10.0
    e[-3:] = 90.0
    w = window_means(e, np.arange(20.0), 1.0)
    assert w == pytest.approx({"hook": 10.0, "hold": 50.0, "ending": 90.0, "overall": e.mean()})

    scores = _scores_for_engagement(e)
    kinds = {m.kind for m in mom.find_hook_and_ending(scores)}
    assert kinds == {"hook_weak"}

    e2 = np.full(20, 50.0)
    e2[-3:] = 20.0
    found = mom.find_hook_and_ending(_scores_for_engagement(e2))
    assert [(m.kind, m.start, m.end) for m in found] == [("ending_drop", 17.0, 20.0)]


def test_moment_carries_spoken_words():
    e = np.full(30, 60.0)
    e[10:14] = 25.0
    words = pd.DataFrame(
        {
            "text": ["before", "inside", "edge", "after"],
            "start": [2.0, 11.0, 13.8, 20.0],
            "duration": [0.5, 0.5, 0.5, 0.5],
            "sentence": ["s"] * 4,
        }
    )
    (dip,) = mom.find_dips(_scores_for_engagement(e), words)
    assert dip.words == "inside edge"


def test_onset_correction_only_touches_the_first_points():
    from cortexreel.score import correct_onset

    eng = np.full(20, 50.0)
    out = correct_onset(eng, [10.0, 5.0, 2.0])
    assert out[:3].tolist() == [60.0, 55.0, 52.0]
    assert np.all(out[3:] == 50.0)
    assert correct_onset(np.full(2, 95.0), [10.0, 5.0, 2.0]).tolist() == [100.0, 100.0]
