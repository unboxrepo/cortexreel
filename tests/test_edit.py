import subprocess

import pytest

from cortexreel import edit
from cortexreel.media import ffmpeg_exe, probe


def _w(text, start, end, sentence="s"):
    return {"text": text, "start": start, "duration": end - start, "sentence": sentence}


# "one two three." | "four five six" with a 0.3 s pause between the sentences.
WORDS = [
    _w("one", 4.0, 4.5, "a"),
    _w("two", 4.6, 5.2, "a"),
    _w("three.", 5.3, 6.0, "a"),
    _w("four", 6.3, 6.8, "b"),
    _w("five", 6.9, 7.6, "b"),
    _w("six", 7.7, 8.4, "b"),
]


def _inside_word(t, words):
    return any(w.start < t < w.end for w in edit.word_list(words))


# --------------------------------------------------------------------- snapping


def test_boundary_inside_word_moves_to_nearest_gap():
    # One long sentence: 1.5 s words with 0.1 s gaps from 2.0 s on.
    words = [_w(f"w{i}", 2 + i * 1.6, 3.5 + i * 1.6) for i in range(6)]
    t = edit.snap(7.0, edit.word_list(words), 20.0)  # inside w3 (6.8-8.3)
    assert not _inside_word(t, words)
    assert t == pytest.approx(6.75)  # middle of the narrow w2|w3 gap


def test_sentence_boundary_preferred_within_one_second():
    wl = edit.word_list(WORDS)
    # Inside "four": nearest gap is four|five (6.85), the sentence gap (6.0-6.3) is ~0.6 s away.
    t = edit.snap(6.7, wl, 20.0)
    assert 6.0 < t < 6.3
    assert not _inside_word(t, WORDS)


def test_time_in_gap_kept_but_padded_from_words():
    wl = edit.word_list(WORDS)
    assert edit.snap(2.0, wl, 20.0) == pytest.approx(2.0)
    assert edit.snap(8.45, wl, 20.0) == pytest.approx(8.4 + edit.WORD_PAD)


def test_snap_respects_lower_bound():
    wl = edit.word_list([_w("hello", 2.5, 3.4)])
    assert edit.snap(3.0, wl, 20.0, lo=3.0) >= 3.4


# --------------------------------------------------------------------- suggest_plan


def _analysis(moments, duration=20.0):
    return {
        "video": {"path": "x.mp4", "sha256": "abc", "duration_seconds": duration},
        "summary": {"overall": 60, "hook": 50, "hold": 62, "ending": 40},
        "moments": moments,
    }


def _m(kind, start, end, score=50.0, delta=-10.0):
    return {"kind": kind, "start": start, "end": end, "score": score, "delta": delta}


def test_silent_dip_is_cut_and_spoken_dip_is_sped_up():
    words = WORDS + [_w("tail", 14.0, 15.0, "c")]
    plan = edit.suggest_plan(
        _analysis([_m("dip", 4.5, 8.0), _m("dip", 9.5, 12.0)]), words, duration=20.0
    )
    speed, cut = plan["ops"]
    assert speed["action"] == "speed" and speed["factor"] == edit.DIP_SPEED
    assert not _inside_word(speed["start"], words) and not _inside_word(speed["end"], words)
    assert cut == {**cut, "action": "cut", "start": 9.5, "end": 12.0}
    assert "dip 9.5-12.0" in cut["why"] and "no speech" in cut["why"]


def test_short_spoken_dip_gets_gentler_speed():
    plan = edit.suggest_plan(_analysis([_m("dip", 4.5, 7.0)]), WORDS, duration=20.0)
    assert plan["ops"][0]["factor"] == edit.SHORT_DIP_SPEED


def test_hook_untouched_and_weak_hook_only_noted():
    plan = edit.suggest_plan(
        _analysis([_m("hook_weak", 0.0, 3.0, 50, -9.0), _m("dip", 1.0, 3.8)]),
        [_w("hi", 4.0, 4.5)],
        duration=20.0,
    )
    assert all(op["start"] >= edit.HOOK_PROTECT_SECONDS for op in plan["ops"])
    assert plan["ops"][0]["start"] == 3.0 and plan["ops"][0]["action"] == "cut"
    assert any(n.startswith("Hook scores 9 points under the middle") for n in plan["notes"])


def test_ending_tail_cut_after_last_word():
    plan = edit.suggest_plan(_analysis([_m("ending_drop", 17.0, 20.0)]), WORDS, duration=20.0)
    (op,) = plan["ops"]
    assert op["action"] == "cut"
    assert op["start"] == pytest.approx(8.4 + edit.ENDING_TAIL_KEEP)
    assert op["end"] == 20.0


def test_ending_with_speech_to_the_end_is_sped_up():
    words = [_w("talk", 17.0, 19.8)]
    plan = edit.suggest_plan(_analysis([_m("ending_drop", 17.0, 20.0)]), words, duration=20.0)
    (op,) = plan["ops"]
    assert op["action"] == "speed" and op["factor"] == edit.ENDING_SPEED


def test_cut_wins_over_overlapping_speed_and_peaks_are_kept():
    words = [_w("a", 10.0, 11.0), _w("b.", 11.5, 12.0)]
    moments = [_m("dip", 9.0, 20.0), _m("ending_drop", 17.0, 20.0), _m("peak", 10.0, 11.0, 80, 20)]
    plan = edit.suggest_plan(_analysis(moments), words, duration=20.0)
    ops = plan["ops"]
    edit.validate_plan(plan, 20.0)  # non-overlapping
    for op in ops:
        assert op["end"] <= 10.0 or op["start"] >= 11.0
    assert ops[-1]["action"] == "cut" and ops[-1]["end"] == 20.0
    assert ops[-2]["action"] == "speed" and ops[-2]["end"] == ops[-1]["start"]


def test_no_transcript_never_cuts_dips():
    plan = edit.suggest_plan(_analysis([_m("dip", 5.0, 9.0)]), None, duration=20.0)
    assert [op["action"] for op in plan["ops"]] == ["speed"]


# --------------------------------------------------------------------- validation


def _plan(*ops):
    return {"schema": 1, "ops": list(ops)}


def test_validation_errors():
    cut = lambda a, b: {"start": a, "end": b, "action": "cut"}  # noqa: E731
    with pytest.raises(edit.PlanError, match="overlaps"):
        edit.validate_plan(_plan(cut(1, 3), cut(2, 4)), 10)
    with pytest.raises(edit.PlanError, match="sorted"):
        edit.validate_plan(_plan(cut(5, 6), cut(1, 2)), 10)
    with pytest.raises(edit.PlanError, match="past the video"):
        edit.validate_plan(_plan(cut(8, 12)), 10)
    with pytest.raises(edit.PlanError, match="removes 60%"):
        edit.validate_plan(_plan(cut(0, 6)), 10)
    edit.validate_plan(_plan(cut(0, 6)), 10, allow_heavy=True)
    with pytest.raises(edit.PlanError, match="at least 1 s"):
        edit.validate_plan(_plan(cut(0, 9.5)), 10, allow_heavy=True)
    with pytest.raises(edit.PlanError, match="factor"):
        edit.validate_plan(_plan({"start": 1, "end": 2, "action": "speed", "factor": 3}), 10)
    with pytest.raises(edit.PlanError, match="action"):
        edit.validate_plan(_plan({"start": 1, "end": 2, "action": "fade"}), 10)
    with pytest.raises(edit.PlanError, match="schema"):
        edit.validate_plan({"ops": []}, 10)


def test_end_slightly_past_duration_is_clamped():
    (op,) = edit.validate_plan(_plan({"start": 8, "end": 10.03, "action": "cut"}), 10)
    assert op["end"] == 10


# --------------------------------------------------------------------- filter + timemap


OPS = [
    {"start": 2.0, "end": 4.0, "action": "cut"},
    {"start": 6.0, "end": 8.0, "action": "speed", "factor": 2.0},
]


def test_segments_and_filter():
    segs = edit.segments(OPS, 10.0)
    assert segs == [(0.0, 2.0, 1.0), (4.0, 6.0, 1.0), (6.0, 8.0, 2.0), (8.0, 10.0, 1.0)]
    f = edit.build_filter(segs, has_audio=True, fps=30)
    assert f.count("]trim=start=") == 4 and f.count("]atrim=start=") == 4
    assert f.count("atempo=") == 1 and "atempo=2" in f
    assert "setpts=(PTS-STARTPTS)/2" in f
    assert "concat=n=4:v=1:a=1[cv][outa]" in f and "[cv]fps=30[outv]" in f


def test_filter_without_audio_has_no_audio_chain():
    f = edit.build_filter(edit.segments(OPS, 10.0), has_audio=False, fps=None)
    assert "atrim" not in f and "atempo" not in f and "[0:a]" not in f
    assert "concat=n=4:v=1:a=0[cv]" in f


def test_timemap():
    tmap = edit.timemap(edit.segments(OPS, 10.0))
    assert tmap == [[0, 2, 0, 2], [4, 6, 2, 4], [6, 8, 4, 5], [8, 10, 5, 7]]
    assert edit.map_time(tmap, 1.0) == 1.0
    assert edit.map_time(tmap, 3.0) is None
    assert edit.map_time(tmap, 7.0) == pytest.approx(4.5)
    assert edit.map_time(tmap, 9.0) == pytest.approx(6.0)


# --------------------------------------------------------------------- smoke


def test_render_smoke(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=160x288:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src),
        ],  # fmt: skip
        check=True,
        capture_output=True,
    )
    plan = {
        "schema": 1,
        "source": str(src),
        "ops": [
            {"start": 0.5, "end": 1.0, "action": "cut"},
            {"start": 1.5, "end": 2.5, "action": "speed", "factor": 2.0},
        ],
    }
    result = edit.render(plan, tmp_path / "out.mp4")
    assert result.new_duration == pytest.approx(2.0, abs=0.05)
    info = probe(result.output)
    assert info.has_audio and (info.width, info.height) == (160, 288)
    assert info.duration == pytest.approx(result.new_duration, abs=0.15)
    with pytest.raises(edit.PlanError, match="overwrite"):
        edit.render(plan, src)
