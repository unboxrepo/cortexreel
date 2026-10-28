import math
import subprocess

import pytest

from cortexreel import compare, edit, splice
from cortexreel.media import ffmpeg_exe, probe

# --------------------------------------------------------------------- blocks


def _ids(blocks):
    return [b["id"] for b in blocks]


def test_blocks_hook_middle_ending():
    blocks = compare.make_blocks(15.0, 3.0)
    assert _ids(blocks) == ["hook", "b1", "b2", "b3", "ending"]
    assert [(b["start"], b["end"]) for b in blocks[:-1]] == [(0, 3), (3, 6), (6, 9), (9, 12)]
    assert blocks[-1] == {"id": "ending", "start": 12.0, "end": 15.0, "align": "end"}


def test_short_remainder_joins_previous_block():
    blocks = compare.make_blocks(16.0, 3.0)  # middle 3-13: 3,6,9 + a 1 s remainder
    assert [(b["start"], b["end"]) for b in blocks if b["id"].startswith("b")] == [
        (3, 6), (6, 9), (9, 13)
    ]  # fmt: skip
    blocks = compare.make_blocks(17.0, 3.0)  # 2 s remainder stays its own block
    assert blocks[-2]["start"] == 12 and blocks[-2]["end"] == 14


def test_very_short_video_has_hook_and_ending_only():
    assert _ids(compare.make_blocks(5.0)) == ["hook", "ending"]
    assert _ids(compare.make_blocks(2.0)) == ["hook"]
    with pytest.raises(compare.CompareError):
        compare.make_blocks(10.0, 0)


def test_block_scores_for_versions_of_different_length():
    times = [float(t) for t in range(8)]  # 8 s video, TR 1
    eng = [10, 10, 10, 50, 50, 50, 90, 90]
    blocks = {b["id"]: b for b in compare.make_blocks(15.0)}
    assert compare.block_score(times, eng, 1.0, blocks["hook"]) == 10
    assert compare.block_score(times, eng, 1.0, blocks["b1"]) == 50
    assert compare.block_score(times, eng, 1.0, blocks["b2"]) == 90  # partial: TRs 6, 7
    assert math.isnan(compare.block_score(times, eng, 1.0, blocks["b3"]))  # past the end
    # the ending is this version's own last 3 s (TRs 5, 6, 7), not 12-15 s
    assert compare.block_score(times, eng, 1.0, blocks["ending"]) == pytest.approx(230 / 3)


# --------------------------------------------------------------------- winners


def test_winner_tie_and_missing():
    assert compare.pick_winner({"a": 70.0, "b": 72.5}) == {"winner": "b", "tie": False, "tied": ["b"]}
    w = compare.pick_winner({"a": 72.0, "b": 72.5, "c": 60.0})
    assert w["winner"] == "b" and w["tie"] and w["tied"] == ["b", "a"]
    assert compare.pick_winner({"a": 80.0, "b": math.nan})["winner"] == "a"
    assert compare.pick_winner({"a": math.nan})["winner"] is None


def _analysis(eng, tr=1.0):
    times = [i * tr for i in range(len(eng))]
    return {
        "video": {"duration_seconds": len(eng) * tr, "tr_seconds": tr},
        "curves": {"times": times, "engagement": eng, "engagement_smooth": eng},
        "summary": {"overall": sum(eng) / len(eng), "hook": 0, "hold": 0, "ending": 0},
    }


def test_build_compare_leader_and_best_of():
    a = _analysis([60] * 3 + [80] * 3 + [40] * 3 + [50] * 3)  # 12 s
    b = _analysis([70] * 3 + [70] * 3 + [70] * 3)  # 9 s
    versions = [
        {"name": "a", "path": "a.mp4", "duration_seconds": 12.0},
        {"name": "b", "path": "b.mp4", "duration_seconds": 9.0},
    ]
    cmp = compare.build_compare(versions, [a, b], 3.0)
    by = {blk["id"]: blk for blk in cmp["blocks"]}
    assert _ids(cmp["blocks"]) == ["hook", "b1", "b2", "ending"]
    assert by["hook"]["winner"] == "b" and by["b1"]["winner"] == "a"
    assert by["b2"]["scores"] == {"a": 40.0, "b": 70.0} and by["b2"]["winner"] == "b"
    assert by["ending"]["scores"] == {"a": 50.0, "b": 70.0}
    assert cmp["leader"] == "b"  # overall 70 vs 57.5
    assert cmp["best_of"] == pytest.approx((70 + 80 + 70 + 70) / 4, abs=0.05)


def test_compare_html_is_offline_and_escaped(tmp_path):
    a, b = _analysis([60] * 9), _analysis([70] * 6)
    versions = [
        {"name": "<b>x</b>", "path": "a.mp4", "duration_seconds": 9.0, "report": str(tmp_path / "r.html")},
        {"name": "y", "path": "b.mp4", "duration_seconds": 6.0},
    ]
    cmp = compare.build_compare(versions, [a, b], 3.0)
    json_path, html_path = compare.write_compare(tmp_path / "c", cmp, [a, b])
    text = html_path.read_text(encoding="utf-8")
    assert json_path.is_file() and "<b>x</b>" not in text and "&lt;b&gt;x&lt;/b&gt;" in text
    assert "http://" not in text.replace("http://www.w3.org", "") and "<script src" not in text
    assert 'href="../r.html"' in text and "var(--v1)" in text


# --------------------------------------------------------------------- auto splice


def _w(text, start, end):
    return edit.Word(text, start, end, "")


def _cmp(blocks, durations):
    return {
        "versions": [{"name": n, "path": f"/v/{n}.mp4", "duration_seconds": d} for n, d in durations.items()],
        "blocks": blocks,
    }


def _blk(bid, start, end, scores, align="start"):
    blk = {"id": bid, "start": start, "end": end, "align": align, "scores": scores}
    blk.update(compare.pick_winner(scores))
    return blk


def test_auto_merges_runs_and_snaps_switches_to_word_gaps():
    blocks = [
        _blk("hook", 0, 3, {"a": 70, "b": 60}),
        _blk("b1", 3, 6, {"a": 70, "b": 60}),
        _blk("b2", 6, 9, {"a": 60, "b": 70}),
        _blk("ending", 9, 12, {"a": 60, "b": 70}, "end"),
    ]
    # A word runs across 6.0 s in both versions: the switch moves to a gap.
    words = [_w("long", 5.5, 6.5), _w("next", 6.7, 7.5)]
    plan = splice.auto_plan(_cmp(blocks, {"a": 12.0, "b": 12.0}), {"a": words, "b": words})
    segs = plan["segments"]
    assert [s["source"] for s in segs] == ["/v/a.mp4", "/v/b.mp4"]
    assert segs[0]["start"] == 0 and segs[1]["end"] == 12.0
    switch = segs[0]["end"]
    assert segs[1]["start"] == pytest.approx(switch)
    assert not any(w.start < switch < w.end for w in words)
    assert switch != 6.0
    assert "hook" in segs[0]["why"] and "ending" in segs[1]["why"]
    assert any("share structure" in n for n in plan["notes"])


def test_auto_falls_back_when_snapping_would_repeat_content():
    blocks = [
        _blk("hook", 0, 3, {"a": 70, "b": 60}),
        _blk("b1", 3, 6, {"a": 60, "b": 70}),
        _blk("b2", 6, 9, {"a": 60, "b": 70}),
    ]
    # "a" is silent around 3 s; "b" says one long word there, so its start would
    # snap ~1.5 s early and replay what "a" already showed.
    wa = [_w("x", 0.5, 1.0)]
    wb = [_w("y", 0.5, 1.0), _w("drawn-out", 1.6, 4.6)]
    plan = splice.auto_plan(_cmp(blocks, {"a": 9.0, "b": 9.0}), {"a": wa, "b": wb})
    (seg,) = plan["segments"]
    assert seg["source"] == "/v/a.mp4" and seg["end"] == 9.0
    assert "kept a for b1, b2 (won by b)" in seg["why"]


def test_auto_falls_back_when_piece_too_short():
    blocks = [
        _blk("hook", 0, 3, {"a": 70, "b": 60}),
        _blk("b1", 3, 3.6, {"a": 60, "b": 70}),
        _blk("b2", 3.6, 9, {"a": 70, "b": 60}),
    ]
    plan = splice.auto_plan(_cmp(blocks, {"a": 9.0, "b": 9.0}), {})
    assert [s["source"] for s in plan["segments"]] == ["/v/a.mp4"]


def test_auto_tie_stays_with_previous_source():
    blocks = [
        _blk("hook", 0, 3, {"a": 70, "b": 60}),
        _blk("b1", 3, 6, {"a": 70, "b": 70.5}),  # tie, b slightly higher
    ]
    plan = splice.auto_plan(_cmp(blocks, {"a": 6.0, "b": 6.0}), {})
    assert [s["source"] for s in plan["segments"]] == ["/v/a.mp4"]
    assert any("adds nothing" in n for n in plan["notes"])


def test_auto_ending_comes_from_the_shorter_versions_own_end():
    blocks = [
        _blk("hook", 0, 3, {"a": 70, "b": 60}),
        _blk("b1", 3, 9, {"a": 70, "b": 60}),
        _blk("ending", 9, 12, {"a": 50, "b": 70}, "end"),
    ]
    plan = splice.auto_plan(_cmp(blocks, {"a": 12.0, "b": 10.0}), {})
    a, b = plan["segments"]
    assert (a["start"], a["end"]) == (0, 9.0)
    assert (b["source"], b["start"], b["end"]) == ("/v/b.mp4", 7.0, 10.0)


# --------------------------------------------------------------------- validation


VERSIONS = [
    {"name": "a", "path": "/v/a.mp4", "duration_seconds": 10.0},
    {"name": "b", "path": "/v/b.mp4", "duration_seconds": 5.0},
]


def _plan(*segs):
    return {"schema": 1, "segments": [{"source": s, "start": a, "end": b} for s, a, b in segs]}


def test_splice_validation():
    ok = splice.validate_splice(_plan(("/v/a.mp4", 0, 3), ("/v/b.mp4", 3, 5.03)), VERSIONS)
    assert [s["name"] for s in ok] == ["a", "b"] and ok[1]["end"] == 5.0
    with pytest.raises(edit.PlanError, match="compared versions"):
        splice.validate_splice(_plan(("/v/c.mp4", 0, 3)), VERSIONS)
    with pytest.raises(edit.PlanError, match="past b's 5 s"):
        splice.validate_splice(_plan(("/v/b.mp4", 3, 6)), VERSIONS)
    with pytest.raises(edit.PlanError, match="start < end"):
        splice.validate_splice(_plan(("/v/a.mp4", 4, 3)), VERSIONS)
    with pytest.raises(edit.PlanError, match="at least 1 s"):
        splice.validate_splice(_plan(("/v/a.mp4", 0, 0.5), ("/v/b.mp4", 1, 1.4)), VERSIONS)
    with pytest.raises(edit.PlanError, match="non-empty"):
        splice.validate_splice({"schema": 1, "segments": []}, VERSIONS)
    with pytest.raises(edit.PlanError, match="schema"):
        splice.validate_splice({"segments": []}, VERSIONS)


# --------------------------------------------------------------------- filter + smoke


def test_filter_two_inputs_different_sizes():
    inputs = [splice.Source("/v/a.mp4", 30.0, True), splice.Source("/v/b.mp4", 25.0, False)]
    segs = [
        {"source": "/v/a.mp4", "start": 0.0, "end": 2.0},
        {"source": "/v/b.mp4", "start": 2.0, "end": 3.0},
        {"source": "/v/a.mp4", "start": 3.0, "end": 4.0},
    ]
    graph, audio = splice.build_filter(segs, inputs, (160, 288), 30.0)
    assert audio
    assert "[0:v]split=2[vs0][vs2]" in graph and "[1:v]split=1[vs1]" in graph
    assert "[0:a]asplit=2[as0][as2]" in graph and "[1:a]" not in graph
    assert graph.count("scale=160:288:force_original_aspect_ratio=decrease") == 3
    assert graph.count("pad=160:288") == 3 and graph.count("fps=30") == 3
    assert "anullsrc" in graph and "atrim=duration=1.000000" in graph  # b has no audio
    assert "[v0][a0][v1][a1][v2][a2]concat=n=3:v=1:a=1[outv][outa]" in graph
    graph, audio = splice.build_filter(segs, [splice.Source(s.path, 30.0, False) for s in inputs], (160, 288), 30.0)
    assert not audio and "atrim" not in graph and "concat=n=3:v=1:a=0[outv]" in graph


def _clip(path, size, freq):
    subprocess.run(
        [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration=3:size={size}:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency={freq}:duration=3",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path),
        ],  # fmt: skip
        check=True,
        capture_output=True,
    )


def test_splice_render_smoke(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    _clip(a, "160x288", 440)
    _clip(b, "320x180", 660)
    versions = [
        {"name": "a", "path": str(a), "duration_seconds": probe(a).duration},
        {"name": "b", "path": str(b), "duration_seconds": probe(b).duration},
    ]
    plan = {
        "schema": 1,
        "segments": [
            {"source": str(a), "start": 0.0, "end": 1.5},
            {"source": str(b), "start": 1.5, "end": 3.0},
        ],
    }
    result = splice.render(plan, versions, tmp_path / "splice.mp4")
    assert result.duration == pytest.approx(3.0, abs=0.05)
    info = probe(result.output)
    assert info.has_audio and (info.width, info.height) == (160, 288)
    assert info.duration == pytest.approx(3.0, abs=0.15)
    assert [r["name"] for r in result.segment_map] == ["a", "b"]
    with pytest.raises(edit.PlanError, match="overwrite"):
        splice.render(plan, versions, a)
