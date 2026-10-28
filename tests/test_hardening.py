"""Security hardening: ffmpeg inputs, link-safe writes, source protection, plan numbers, output text."""

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cortexreel import cli, edit, media, outputs, overlay, splice
from cortexreel.media import PROTOCOL_WHITELIST, VIDEO_FORMATS, ffmpeg_exe, input_args, probe

PACKAGE = Path(cli.__file__).parent


@pytest.fixture(autouse=True)
def restore_offline_vars(monkeypatch):
    """cli.main forces offline mode; keep that from leaking into other test files."""
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, os.environ.get(key, ""))


def _clip(path, seconds=2):
    subprocess.run(
        [
            ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={seconds}:size=160x288:rate=30",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(path),
        ],  # fmt: skip
        check=True,
        capture_output=True,
    )


# --------------------------------------------------------------------- M1: ffmpeg inputs


def _assert_guarded(cmd):
    """Every ``-i`` in ``cmd`` is preceded by both whitelists."""
    inputs = [i for i, a in enumerate(cmd) if a == "-i"]
    assert inputs, cmd
    for i in inputs:
        assert cmd[i - 4 : i] == ["-protocol_whitelist", PROTOCOL_WHITELIST, "-format_whitelist", cmd[i - 1]]
        assert cmd[i - 1] in (VIDEO_FORMATS, "rawvideo")


@pytest.fixture
def captured(monkeypatch):
    """Record ffmpeg command lines run from media.py instead of running them."""
    cmds = []

    class Done:
        returncode, stdout, stderr = 0, b"\xff\xd8", b"Duration: 00:00:01.00, start"

    real_run = subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if not (isinstance(cmd, list) and cmd[0] == "fake-ffmpeg"):
            return real_run(cmd, *args, **kwargs)
        cmds.append(cmd)
        return Done()

    monkeypatch.setattr(media, "ffmpeg_exe", lambda: "fake-ffmpeg")
    monkeypatch.setattr(media.subprocess, "run", fake_run)
    return cmds


def test_media_commands_whitelist_inputs(captured, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    media.probe("v.mp4")
    media.make_proxy("v.mp4", tmp_path / "out" / "x" / "proxy.mp4")
    media.frame_jpeg("v.mp4", 1.0)
    assert len(captured) == 3
    for cmd in captured:
        _assert_guarded(cmd)


def test_render_commands_whitelist_inputs():
    _assert_guarded(edit.render_command("in.mp4", "out.mp4", [(0.0, 1.0, 1.0)], True, 30.0))
    inputs = [splice.Source("a.mp4", 30.0, True), splice.Source("b.mp4", 30.0, True)]
    segs = [{"source": "a.mp4", "start": 0.0, "end": 1.0}, {"source": "b.mp4", "start": 0.0, "end": 1.0}]
    cmd = splice.render_command(segs, inputs, (160, 288), 30.0, "out.mp4")
    _assert_guarded(cmd)
    assert cmd.count("-i") == 2
    track = overlay.Track(
        "a", Path("a.mp4"), Path("out/a"), 1.0, (160, 288), "aac", np.zeros(1), 1.0,
        np.zeros(1), {}, [], {},
    )  # fmt: skip
    cmd = overlay._ffmpeg_cmd([track], [(160, 288, 0, 0)], 30, Path("o.mp4"))
    _assert_guarded(cmd)
    assert cmd[cmd.index("-i") + 1] == "pipe:0"


def test_no_ffmpeg_input_bypasses_the_helper():
    for path in PACKAGE.glob("*.py"):
        if path.name == "media.py":
            continue
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"""["']-i["']""", source), f"{path.name} builds an ffmpeg -i by hand"


def test_ffconcat_list_named_mp4_is_refused(tmp_path):
    secret = tmp_path / "secret.mp4"
    _clip(secret)
    fake = tmp_path / "list.mp4"
    fake.write_text("ffconcat version 1.0\nfile secret.mp4\n", encoding="utf-8")
    assert probe(secret).duration > 1.5
    with pytest.raises(RuntimeError, match="could not read"):
        probe(fake)


# --------------------------------------------------------------------- L4: links under out/


def _symlink_dir(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        # No symlink privilege on Windows: a junction is the same attack.
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def test_refuses_linked_out_and_stem(tmp_path, monkeypatch):
    from cortexreel.compare import compare_dir
    from cortexreel.predict import output_dir

    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    _symlink_dir(tmp_path / "out", elsewhere)
    assert outputs.is_link(tmp_path / "out")
    with pytest.raises(outputs.OutputError):
        output_dir("clip.mp4")
    with pytest.raises(outputs.OutputError):
        compare_dir(["a", "b"])
    with pytest.raises(outputs.OutputError):
        outputs.write_text(tmp_path / "out" / "clip" / "analysis.json", "{}")
    assert not any(elsewhere.iterdir())

    if os.name == "nt":
        os.rmdir(tmp_path / "out")  # removes a junction or directory symlink, not its target
    else:
        (tmp_path / "out").unlink()
    (tmp_path / "out").mkdir()
    _symlink_dir(tmp_path / "out" / "clip", elsewhere)
    with pytest.raises(outputs.OutputError):
        output_dir("clip.mp4")
    with pytest.raises(outputs.OutputError):
        outputs.write_text(tmp_path / "out" / "clip" / "report.html", "x")
    assert not any(elsewhere.iterdir())


def test_existing_output_is_replaced_not_written_through(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    victim = tmp_path / "victim.txt"
    victim.write_text("keep", encoding="utf-8")
    target = tmp_path / "out" / "clip" / "analysis.json"
    target.parent.mkdir(parents=True)
    os.link(victim, target)  # hardlinks need no privilege
    outputs.write_text(target, "{}")
    assert victim.read_text(encoding="utf-8") == "keep"
    assert target.read_text(encoding="utf-8") == "{}"
    try:
        os.symlink(victim, tmp_path / "out" / "clip" / "report.html")
    except OSError:
        return  # no symlink privilege (Windows); the hardlink case covers unlink-first
    outputs.write_text(tmp_path / "out" / "clip" / "report.html", "x")
    assert victim.read_text(encoding="utf-8") == "keep"


def test_save_result_refuses_linked_out(tmp_path, monkeypatch):
    from cortexreel.predict import Result, save_result

    monkeypatch.chdir(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "clip").mkdir(parents=True)
    _symlink_dir(tmp_path / "out", elsewhere)
    result = Result(preds=np.zeros((2, 3), np.float32), tr_seconds=1.0, times=np.arange(2.0))
    with pytest.raises(ValueError):
        save_result(result, tmp_path / "out" / "clip")
    assert not any((elsewhere / "clip").iterdir())


# --------------------------------------------------------------------- L5: never overwrite a source


def test_same_file_sees_hardlinks(tmp_path):
    a = tmp_path / "a.mp4"
    a.write_bytes(b"x")
    os.link(a, tmp_path / "b.mp4")
    assert outputs.same_file(a, tmp_path / "b.mp4")
    assert outputs.same_file(a, tmp_path / "." / "a.mp4")
    assert not outputs.same_file(a, tmp_path / "missing.mp4")


def test_recut_and_splice_refuse_a_hardlink_to_the_source(tmp_path):
    src = tmp_path / "src.mp4"
    _clip(src)
    size = src.stat().st_size
    link = tmp_path / "link.mp4"
    os.link(src, link)
    plan = {"schema": 1, "source": str(src), "ops": []}
    with pytest.raises(edit.PlanError, match="overwrite"):
        edit.render(plan, link)
    versions = [{"name": "src", "path": str(src), "duration_seconds": probe(src).duration}]
    splice_plan = {"schema": 1, "segments": [{"source": str(src), "start": 0.0, "end": 1.5}]}
    with pytest.raises(edit.PlanError, match="overwrite"):
        splice.render(splice_plan, versions, link)
    assert src.stat().st_size == size


def test_overlay_refuses_a_hardlink_to_the_source(tmp_path, monkeypatch):
    src = tmp_path / "src.mp4"
    src.write_bytes(b"x")
    os.link(src, tmp_path / "link.mp4")
    track = overlay.Track(
        "src", src, tmp_path, 1.0, (160, 288), None, np.zeros(1), 1.0, np.zeros(1), {}, [], {}
    )  # fmt: skip
    monkeypatch.setattr(overlay, "load_track", lambda v: track)
    with pytest.raises(overlay.OverlayError, match="overwrite"):
        overlay.render(src, out=tmp_path / "link.mp4")
    assert src.read_bytes() == b"x"


# --------------------------------------------------------------------- L6: finite numbers


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), True, "NaN"])
def test_plans_reject_non_finite_numbers(bad):
    ops = [{"start": 4.0, "end": 5.0, "action": "cut"}]
    for key in ("start", "end"):
        with pytest.raises(edit.PlanError, match="finite"):
            edit.validate_plan({"schema": 1, "ops": [{**ops[0], key: bad}]}, 10.0)
    with pytest.raises(edit.PlanError, match="finite"):
        edit.validate_plan(
            {"schema": 1, "ops": [{"start": 4.0, "end": 5.0, "action": "speed", "factor": bad}]}, 10.0
        )
    versions = [{"name": "a", "path": "/v/a.mp4", "duration_seconds": 10.0}]
    for key in ("start", "end"):
        seg = {"source": "/v/a.mp4", "start": 0.0, "end": 5.0, key: bad}
        with pytest.raises(edit.PlanError, match="finite"):
            splice.validate_splice({"schema": 1, "segments": [seg]}, versions)


def test_nan_from_json_is_rejected():
    plan = json.loads('{"schema": 1, "ops": [{"start": NaN, "end": 5, "action": "cut"}]}')
    with pytest.raises(edit.PlanError, match="finite"):
        edit.validate_plan(plan, 10.0)


# --------------------------------------------------------------------- L3: compare.json integrity


def _compare_json(tmp_path, versions):
    cmp_dir = tmp_path / "out" / "compare-a-vs-b"
    cmp_dir.mkdir(parents=True)
    cmp = {"schema": 1, "versions": versions, "blocks": [{"id": "hook"}]}
    path = cmp_dir / "compare.json"
    path.write_text(json.dumps(cmp), encoding="utf-8")
    return path


@pytest.mark.parametrize("sha", [None, "", "0" * 64])
def test_splice_requires_matching_sha256(tmp_path, monkeypatch, capsys, sha):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.mp4").write_bytes(b"a")
    versions = [{"name": "a", "path": "a.mp4", "duration_seconds": 1.0}]
    if sha is not None:
        versions[0]["sha256"] = sha
    path = _compare_json(tmp_path, versions)
    assert cli.main(["splice", "--compare", str(path), "--dry-run"]) == 2
    assert "changed since the comparison" in capsys.readouterr().err


def test_splice_resolves_relative_version_paths(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.mp4").write_bytes(b"a")
    versions = [{"name": "a", "path": "a.mp4", "duration_seconds": 1.0, "sha256": media.sha256_file(tmp_path / "a.mp4")}]
    path = _compare_json(tmp_path, versions)
    seen = []

    def fake_render(plan, vs, out, dry_run):
        seen.extend(vs)
        raise edit.PlanError("stop here")

    monkeypatch.setattr(splice, "render", fake_render)
    assert cli.main(["splice", "--compare", str(path), "--plan", str(path), "--dry-run"]) == 2
    assert seen and Path(seen[0]["path"]).is_absolute()
    assert Path(seen[0]["path"]) == (tmp_path / "a.mp4").resolve()


# --------------------------------------------------------------------- L7: shareable paths


def test_rel_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert outputs.rel_path(tmp_path / "samples" / "a.mp4") == "samples/a.mp4"
    assert outputs.rel_path("out/x/analysis.json") == "out/x/analysis.json"


def test_analysis_metrics_drop_gpu(tmp_path, monkeypatch):
    from cortexreel.report import shareable_metrics

    monkeypatch.chdir(tmp_path)
    metrics = {"video": str(tmp_path / "v.mp4"), "gpu": "NVIDIA RTX", "use_text": True}
    assert shareable_metrics(metrics, tmp_path / "v.mp4") == {"video": "v.mp4", "use_text": True}


# --------------------------------------------------------------------- L8: terminal output


def test_clean_strips_control_characters():
    assert cli._clean("ok \x1b[2J\x1b]0;pwned\x07 done\r\n") == "ok [2J]0;pwned done"
    assert cli._clean("türkçe · ünïcode – ok") == "türkçe · ünïcode – ok"


def test_diff_prints_names_without_escapes(tmp_path, capsys):
    def analysis(name):
        return {
            "video": {"name": name, "path": name, "duration_seconds": 3.0, "tr_seconds": 1.0},
            "summary": {k: 50.0 for k in cli.DIFF_KEYS},
            "curves": {"engagement": [50.0, 50.0, 50.0]},
            "moments": [],
        }

    for name in ("a", "b"):
        (tmp_path / f"{name}.json").write_text(json.dumps(analysis(f"{name}\x1b[2J.mp4")), encoding="utf-8")
    assert cli.main(["diff", str(tmp_path / "a.json"), str(tmp_path / "b.json")]) == 0
    assert "\x1b" not in capsys.readouterr().out


# --------------------------------------------------------------------- L1/L2: offline, pinned only


def test_model_commands_run_offline(monkeypatch):
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setattr(cli, "cmd_diff", lambda args: 0)
    assert cli.main(["diff", "a", "b"]) == 0
    assert os.environ["HF_HUB_OFFLINE"] == os.environ["TRANSFORMERS_OFFLINE"] == "1"


def test_unpinned_repo_fails_closed():
    import transformers.utils.hub as tf_hub

    from cortexreel.model import UnpinnedModel, pin_transformers_revisions

    pin_transformers_revisions()
    with pytest.raises(UnpinnedModel):
        tf_hub.resolve_revision("someone/not-pinned")
    with pytest.raises(UnpinnedModel):
        tf_hub.resolve_revision("facebook/dinov2-large", "refs/pr/1")


def test_extended_path_unc():
    from cortexreel.model import extended_path

    assert extended_path(r"C:\cache\x") == "\\\\?\\C:\\cache\\x"
    assert extended_path(r"\\server\share\x") == "\\\\?\\UNC\\server\\share\\x"
    assert extended_path("\\\\?\\C:\\x") == "\\\\?\\C:\\x"
