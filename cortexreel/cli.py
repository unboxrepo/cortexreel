"""cortexreel command line."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

# Only stdlib and our own env module above this line: env.configure() must run
# before torch / transformers / huggingface_hub are imported.
from . import env

OK, FAIL, WARN = "✔", "✘", "!"


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _clean(text) -> str:
    """``text`` without control characters, for printing strings from JSON or transcripts.

    Keeps an ANSI escape sequence in a plan's ``why`` or a video name from reaching the terminal.
    """
    return "".join(c for c in str(text) if c.isprintable())


def _line(mark: str, label: str, detail: str = "") -> None:
    print(f" {mark} {label}" + (f": {detail}" if detail else ""))


def _free_space(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def _cached(pin) -> bool:
    from .model import local_snapshot

    try:
        local_snapshot(pin)
        return True
    except env.SetupNeeded:
        return False


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import pins

    todo: list[str] = []
    print("cortexreel doctor\n")

    py = sys.version.split()[0]
    py_ok = sys.version_info[:2] == (3, 11)
    _line(OK if py_ok else FAIL, "Python", py)
    if not py_ok:
        todo.append("Use Python 3.11: run `uv sync` and invoke cortexreel via `uv run`.")

    try:
        import torch

        cuda = torch.cuda.is_available()
        _line(OK, "PyTorch", f"{torch.__version__} (CUDA build {torch.version.cuda})")
        if cuda:
            props = torch.cuda.get_device_properties(0)
            vram = props.total_memory / 1024**3
            _line(OK if vram >= 7.5 else WARN, "GPU", f"{props.name}, {vram:.1f} GB VRAM")
            if vram < 7.5:
                todo.append("Less than 8 GB VRAM: TRIBE feature extraction may run out of memory.")
        else:
            _line(FAIL, "GPU", "CUDA not available")
            todo.append("Install an NVIDIA driver supporting CUDA 12.4; CPU runs are very slow.")
    except Exception as exc:  # broken torch install
        _line(FAIL, "PyTorch", repr(exc))
        todo.append("Reinstall dependencies with `uv sync`.")

    try:
        import imageio_ffmpeg

        _line(OK, "ffmpeg", imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        _line(FAIL, "ffmpeg", repr(exc))
        todo.append("ffmpeg missing: run `uv sync` (it ships with imageio-ffmpeg).")

    import huggingface_hub as hf
    from huggingface_hub.errors import GatedRepoError

    token = hf.get_token()
    _line(OK if token else FAIL, "Hugging Face token", "found" if token else "not found")
    if not token:
        todo.append(
            "Create a read token at https://huggingface.co/settings/tokens and run "
            "`uv run hf auth login` (or set HF_TOKEN)."
        )

    llama = pins.LLAMA
    if env.is_offline():
        _line(WARN, f"{llama.repo_id} access", "not checked (offline mode)")
    elif not token:
        _line(FAIL, f"{llama.repo_id} access", "needs a token")
        todo.append(
            f"Request access at https://huggingface.co/{llama.repo_id} "
            "(or use --no-text to skip the text channel)."
        )
    else:
        try:
            hf.model_info(llama.repo_id, revision=llama.revision, token=token)
            hf.auth_check(llama.repo_id, token=token)
            _line(OK, f"{llama.repo_id} access", "granted")
        except GatedRepoError:
            _line(FAIL, f"{llama.repo_id} access", "not granted (gated)")
            todo.append(
                f"Request access at https://huggingface.co/{llama.repo_id} and wait for "
                "approval (or use --no-text meanwhile)."
            )
        except Exception as exc:
            _line(WARN, f"{llama.repo_id} access", f"could not check: {exc.__class__.__name__}")

    root = env.cache_root()
    free_gb = _free_space(root) / 1024**3
    _line(OK if free_gb >= 30 else WARN, "Disk free at cache", f"{free_gb:.0f} GB ({root})")
    if free_gb < 30:
        todo.append("Models need roughly 20-30 GB; free up space or set CORTEXREEL_HOME.")

    telemetry = {k: os.environ.get(k) for k in env.PRIVACY_DEFAULTS}
    tel_ok = all(os.environ.get(k) == v for k, v in env.PRIVACY_DEFAULTS.items())
    _line(OK if tel_ok else WARN, "Telemetry off", ", ".join(f"{k}={v}" for k, v in telemetry.items()))
    _line(OK, "Offline mode", "analyze, predict and the other model commands always run offline")
    missing = [p for p in pins.needed(True) if not _cached(p)]
    if missing:
        _line(WARN, "Models", "not downloaded yet (`cortexreel setup`): " + ", ".join(p.repo_id for p in missing))
    else:
        _line(OK, "Models", "all pinned revisions in the cache")

    print()
    if todo:
        print("Next steps:")
        for i, step in enumerate(todo, 1):
            print(f" {i}. {step}")
        return 1
    print("All checks passed. Next: `cortexreel setup`, then `cortexreel predict VIDEO`.")
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    from . import regions
    from .model import download_models

    for pin, path in download_models(use_text=not args.no_text):
        print(f" {OK} {pin.repo_id}@{pin.revision[:12]} -> {path}")
    regions.fetch_atlas()
    print(f" {OK} Destrieux atlas -> {regions.atlas_dir() / regions.DESTRIEUX_DIR}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    from .predict import output_dir, predict_video

    result = predict_video(
        args.video,
        use_text=not args.no_text,
        language=args.lang,
        precision=args.precision,
        force=args.force,
    )
    m = result.metrics
    print(
        f"{result.preds.shape[0]} TRs x {result.preds.shape[1]} vertices "
        f"(TR={result.tr_seconds:g}s) -> {output_dir(args.video)}"
    )
    if result.cached:
        print("reused cached predictions (--force to recompute)")
        return 0
    print(
        f"events/ASR {m['seconds_events_asr']}s, features+predict "
        f"{m['seconds_features_predict']}s, peak VRAM "
        f"{m.get('cuda_max_memory_allocated_gb', 'n/a')} GB on {m['gpu']}"
    )
    return 0


@contextmanager
def _stdout_to_stderr(enabled: bool = True):
    """Send everything written to stdout, including by child processes, to stderr."""
    if not enabled:
        yield
        return
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _fmt_seconds(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


@dataclass
class _Analyzed:
    analysis: dict
    html_path: Path
    scores: object
    moments: list
    cached: bool


def _analyze(video: Path, args: argparse.Namespace) -> _Analyzed:
    """Predict (cached when possible), score, find moments, write analysis.json + report.html."""
    from . import moments as mom
    from . import regions, report
    from .predict import output_dir, predict_video
    from .score import compute_scores

    result = predict_video(
        video,
        use_text=not args.no_text,
        language=args.lang,
        precision=args.precision,
        force=getattr(args, "force", False),
    )
    config = regions.load_config()
    masks = regions.region_masks(config)
    scores = compute_scores(result.preds, result.times, result.tr_seconds, masks, config)
    found = mom.find_moments(scores, result.words)
    analysis = report.build_analysis(video, result.metrics, scores, found, config, masks)
    _, html_path = report.write_report(
        output_dir(video), video, analysis, scores, found, mean_map=result.preds.mean(axis=0)
    )
    return _Analyzed(analysis, html_path, scores, found, result.cached)


def cmd_analyze(args: argparse.Namespace) -> int:
    from . import report

    with _stdout_to_stderr(args.json):
        video = Path(args.video).resolve()
        done = _analyze(video, args)
    analysis, scores, found, html_path = done.analysis, done.scores, done.moments, done.html_path

    if args.json:
        print(json.dumps(report.summary_json(analysis), indent=2, ensure_ascii=False))
        return 0

    s = analysis["summary"]
    print(f"{_clean(video.name)}: {scores.duration:g} s" + (" (cached predictions)" if done.cached else ""))
    print(
        f"engagement  overall {s['overall']:.0f}  hook {s['hook']:.0f}  "
        f"hold {s['hold']:.0f}  ending {s['ending']:.0f}   (0-100)"
    )
    flags = [m for m in found if m.kind in ("hook_weak", "ending_drop")]
    for m in flags:
        print(f" ! {m.kind.replace('_', ' ')} ({m.delta:+.1f} vs hold): {_clean(m.reason)}")
    dips = sorted((m for m in found if m.kind == "dip"), key=lambda m: m.delta)[:3]
    if dips:
        print("weakest dips:")
        for m in dips:
            print(f" - {_fmt_seconds(m.start)}-{_fmt_seconds(m.end)}  {m.score:.0f} ({m.delta:+.1f})  {_clean(m.reason)}")
    else:
        print("no dips found")
    print(f"report: {_clean(html_path)}")
    return 0


def _error(message: str) -> int:
    print(f"error: {_clean(message)}", file=sys.stderr)
    return 2


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cmd_recut(args: argparse.Namespace) -> int:
    from . import edit
    from .media import probe, sha256_file
    from .moments import load_words
    from .outputs import rel_path, same_file, write_text
    from .predict import output_dir

    if args.plan and args.auto:
        return _error("use either --plan or --auto, not both")
    video = Path(args.video).resolve()
    if not video.is_file():
        return _error(f"no such video: {args.video}")
    out_dir = output_dir(video)
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.is_file():
        return _error(f"no analysis for {video.name}. Run first: cortexreel analyze {args.video}")
    analysis = _load_json(analysis_path)
    sha = sha256_file(video)
    if analysis.get("video", {}).get("sha256") not in (None, sha):
        return _error(
            f"{analysis_path} was made from a different version of {video.name}. "
            f"Run again: cortexreel analyze {args.video}"
        )
    info = probe(video)

    if args.plan:
        plan_path = Path(args.plan)
        try:
            plan = _load_json(plan_path)
        except (OSError, json.JSONDecodeError) as exc:
            return _error(f"cannot read plan {plan_path}: {exc}")
        if plan.get("source_sha256") not in (None, sha):
            return _error(f"{plan_path} was written for a different video (source_sha256 differs)")
    else:
        plan = edit.suggest_plan(analysis, load_words(out_dir / "words.tsv"), info.duration)
    # In memory the plan points at the real file; plan.json gets a relative path.
    plan.update(source=str(video), source_sha256=sha, source_duration=round(info.duration, 3))
    try:
        new_duration, tmap = edit.plan_summary(plan, info.duration, info.fps, args.allow_heavy)
    except edit.PlanError as exc:
        return _error(f"invalid plan: {exc}")
    shared_plan = {**plan, "source": rel_path(video)}
    plan_file = write_text(out_dir / "plan.json", json.dumps(shared_plan, indent=2, ensure_ascii=False))

    output = None
    if not args.dry_run:
        out_path = Path(args.out).resolve() if args.out else out_dir / f"{video.stem}-recut.mp4"
        if same_file(out_path, video):
            return _error("refusing to overwrite the source video; pick another --out")
        try:
            result = edit.render(plan, out_path, allow_heavy=args.allow_heavy)
        except edit.PlanError as exc:
            return _error(f"invalid plan: {exc}")
        output, new_duration, tmap = result.output, result.new_duration, result.timemap

    payload = {
        "plan": shared_plan,
        "output": rel_path(output) if output else None,
        "old_duration": round(info.duration, 3),
        "new_duration": round(new_duration, 3),
        "timemap": tmap,
    }
    if output:
        write_text(output.with_suffix(".json"), json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"plan: {_clean(plan_file)}")
    for op in plan["ops"]:
        what = op["action"] if op["action"] == "cut" else f"speed x{op['factor']:g}"
        print(
            f" - {_fmt_seconds(op['start'])}-{_fmt_seconds(op['end'])}  {what:<11} "
            f"{_clean(op.get('why', ''))}"
        )
    if not plan["ops"]:
        print(" (no ops)")
    for note in plan.get("notes", []):
        print(f" note: {_clean(note)}")
    print(f"duration {info.duration:.2f} s -> {new_duration:.2f} s")
    print(f"output: {_clean(output)}" if output else "dry run: nothing rendered")
    return 0


DIFF_KEYS = ("overall", "hook", "hold", "ending")


def _diff(before: dict, after: dict) -> dict:
    import numpy as np

    def dips(a: dict) -> list[dict]:
        return [m for m in a.get("moments", []) if m["kind"] == "dip"]

    def flags(a: dict) -> list[str]:
        return [m["kind"] for m in a.get("moments", []) if m["kind"] in ("hook_weak", "ending_drop")]

    def row(x: float, y: float) -> dict:
        return {"before": x, "after": y, "delta": round(y - x, 2)}

    # Dips are relative to each video's own median, so a better cut can show more
    # of them. "low seconds" measures both videos against the BEFORE median instead.
    floor = float(np.median(before["curves"]["engagement"]))

    def low_seconds(a: dict) -> float:
        curve = np.asarray(a["curves"]["engagement"], dtype=float)
        return round(float((curve < floor).sum() * a["video"]["tr_seconds"]), 2)

    return {
        "before": before["video"]["path"],
        "after": after["video"]["path"],
        "summary": {k: row(before["summary"][k], after["summary"][k]) for k in DIFF_KEYS},
        "duration": row(before["video"]["duration_seconds"], after["video"]["duration_seconds"]),
        "dips": row(len(dips(before)), len(dips(after))),
        "dip_seconds": row(
            round(sum(m["end"] - m["start"] for m in dips(before)), 2),
            round(sum(m["end"] - m["start"] for m in dips(after)), 2),
        ),
        "low_seconds": row(low_seconds(before), low_seconds(after)),
        "low_threshold": round(floor, 1),
        "flags": {"before": flags(before), "after": flags(after)},
    }


def cmd_diff(args: argparse.Namespace) -> int:
    try:
        before, after = _load_json(Path(args.before)), _load_json(Path(args.after))
    except (OSError, json.JSONDecodeError) as exc:
        return _error(f"cannot read analysis: {exc}")
    d = _diff(before, after)
    if args.json:
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return 0
    print(f"before: {_clean(before['video']['name'])}   after: {_clean(after['video']['name'])}\n")
    print(f" {'':<12}{'before':>8}{'after':>8}{'delta':>8}")
    rows = [(k, d["summary"][k]) for k in DIFF_KEYS] + [
        ("duration s", d["duration"]),
        ("dips", d["dips"]),
        ("dip seconds", d["dip_seconds"]),
        (f"s < {d['low_threshold']:g}", d["low_seconds"]),
    ]
    for name, r in rows:
        print(f" {name:<12}{r['before']:>8g}{r['after']:>8g}{r['delta']:>+8g}")
    for side in ("before", "after"):
        print(f" flags {side}: {_clean(', '.join(d['flags'][side])) or 'none'}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    from . import compare
    from .media import probe, sha256_file
    from .outputs import rel_path
    from .predict import output_dir

    videos = [Path(v).resolve() for v in args.videos]
    if not compare.MIN_VERSIONS <= len(videos) <= compare.MAX_VERSIONS:
        return _error(
            f"compare takes {compare.MIN_VERSIONS}-{compare.MAX_VERSIONS} videos, got {len(videos)}"
        )
    for video, arg in zip(videos, args.videos):
        if not video.is_file():
            return _error(f"no such video: {arg}")
    names = [v.stem for v in videos]
    if len(set(names)) != len(names):
        return _error("videos need different file names (their outputs share out/<name>/)")
    if args.block <= 0:
        return _error(f"--block must be positive, got {args.block:g}")

    versions, analyses = [], []
    with _stdout_to_stderr(args.json):
        for video in videos:
            print(f"analyzing {video.name}", file=sys.stderr)
            done = _analyze(video, args)
            sha = sha256_file(video)
            if done.analysis["video"]["sha256"] != sha:
                return _error(f"{video.name} changed while it was analyzed. Run cortexreel compare again.")
            versions.append(
                {
                    "name": video.stem,
                    "path": rel_path(video),
                    "sha256": sha,
                    "duration_seconds": round(probe(video).duration, 3),
                    "analysis": rel_path(output_dir(video) / "analysis.json"),
                    "report": rel_path(done.html_path),
                }
            )
            analyses.append(done.analysis)
        cmp = compare.build_compare(versions, analyses, args.block)
        json_path, html_path = compare.write_compare(compare.compare_dir(names), cmp, analyses)

    if args.json:
        print(json.dumps(cmp, indent=2, ensure_ascii=False))
        return 0

    keys = ("overall", "hook", "hold", "ending")
    width = max(len(_clean(n)) for n in names) + 2
    print(f" {'':<{width}}{'dur s':>7}" + "".join(f"{k:>9}" for k in keys))
    for v in versions:
        lead = "  <- leader" if v["name"] == cmp["leader"] else ""
        cells = "".join(f"{v['summary'][k]:>9.1f}" for k in keys)
        print(f" {_clean(v['name']):<{width}}{v['duration_seconds']:>7.1f}{cells}{lead}")
    print("\nblock winners:")
    for b in cmp["blocks"]:
        when = "last 3 s" if b["align"] == "end" else f"{_fmt_seconds(b['start'])}-{_fmt_seconds(b['end'])}"
        if b["winner"] is None:
            verdict = "no data"
        elif b["tie"]:
            verdict = "tie: " + ", ".join(b["tied"])
        else:
            verdict = b["winner"]
        scores = "  ".join(f"{n} {'-' if s is None else f'{s:.1f}'}" for n, s in b["scores"].items())
        print(_clean(f" {b['id']:<7}{when:<15}{verdict:<{2 * width + 8}}{scores}"))
    leader = next(v for v in versions if v["name"] == cmp["leader"])
    best = cmp["best_of"]
    print(
        f"\nleader: {_clean(leader['name'])} (overall {leader['summary']['overall']:.1f})"
        + (f"; best of blocks ~{best:.1f} (rough estimate)" if best is not None else "")
    )
    print(f"compare: {_clean(json_path)}\nreport: {_clean(html_path)}")
    return 0


def cmd_splice(args: argparse.Namespace) -> int:
    from . import edit, splice
    from .media import sha256_file
    from .moments import load_words
    from .outputs import OUT_ROOT, out_root, rel_path, write_text
    from .predict import output_dir

    if args.plan and args.auto:
        return _error("use either --plan or --auto, not both")
    compare_path = Path(args.compare).resolve()
    try:
        cmp = _load_json(compare_path)
    except (OSError, json.JSONDecodeError) as exc:
        return _error(f"cannot read {args.compare}: {exc}")
    if cmp.get("schema") != 1 or not cmp.get("versions") or not cmp.get("blocks"):
        return _error(f"{args.compare} is not a cortexreel compare.json")
    out_dir = compare_path.parent
    if not out_dir.is_relative_to(out_root()):
        return _error(f"compare.json must be under ./{OUT_ROOT}/ (outputs go next to it)")
    for v in cmp["versions"]:
        # Paths are relative to the CWD the comparison ran in; resolve before any use.
        path = Path(v["path"]).resolve()
        if not path.is_file():
            return _error(f"compared version is missing: {v['path']}")
        if not v.get("sha256") or v["sha256"] != sha256_file(path):
            return _error(f"{path.name} changed since the comparison. Run cortexreel compare again.")
        v["path"] = str(path)

    if args.plan:
        try:
            plan = _load_json(Path(args.plan))
        except (OSError, json.JSONDecodeError) as exc:
            return _error(f"cannot read plan {args.plan}: {exc}")
    else:
        words = {
            v["name"]: edit.word_list(load_words(output_dir(v["path"]) / "words.tsv"))
            for v in cmp["versions"]
        }
        plan = splice.auto_plan(cmp, words)
    if isinstance(plan, dict):
        plan["compare"] = rel_path(compare_path)
    try:
        # Unique stem so analyzing two different splices never shares out/<stem>/.
        video_name = f"splice-{out_dir.name.removeprefix('compare-')}.mp4"
        result = splice.render(plan, cmp["versions"], out_dir / video_name, dry_run=args.dry_run)
    except edit.PlanError as exc:
        return _error(f"invalid splice plan: {exc}")

    def shared(rows: list[dict]) -> list[dict]:  # absolute source paths only in memory
        return [{**r, "source": rel_path(r["source"])} for r in rows]

    shared_plan = {**plan, "segments": shared(plan["segments"])}
    plan_file = write_text(
        out_dir / "splice-plan.json", json.dumps(shared_plan, indent=2, ensure_ascii=False)
    )

    payload = {
        "plan": shared_plan,
        "output": rel_path(result.output) if result.output else None,
        "duration": round(result.duration, 3),
        "segment_map": shared(result.segment_map),
    }
    if result.output:
        write_text(out_dir / "splice.json", json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    print(f"plan: {_clean(plan_file)}")
    for row, seg in zip(result.segment_map, plan["segments"]):
        print(
            f" - {_fmt_seconds(row['out_start'])}-{_fmt_seconds(row['out_end'])}  {_clean(row['name'])} "
            f"{_fmt_seconds(row['start'])}-{_fmt_seconds(row['end'])}  {_clean(seg.get('why', ''))}"
        )
    for note in plan.get("notes", []):
        print(f" note: {_clean(note)}")
    print(f"duration {result.duration:.2f} s")
    print(f"output: {_clean(result.output)}" if result.output else "dry run: nothing rendered")
    return 0


def cmd_overlay(args: argparse.Namespace) -> int:
    from .overlay import OverlayError, render

    try:
        result = render(
            args.video,
            out=args.out,
            other=args.compare,
            fps=args.fps,
            watermark=not args.no_watermark,
        )
    except OverlayError as exc:
        return _error(str(exc))
    brain = f", brain frames {result.seconds_brain:.1f} s" if result.seconds_brain else ""
    print(
        f"{result.frames} frames ({result.duration:.2f} s video) in "
        f"{result.seconds_total:.1f} s{brain}"
    )
    print(f"output: {_clean(result.output)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cortexreel", description=__doc__)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="never touch the network; only matters for doctor and setup, every other command "
        "always runs offline from the model cache",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check GPU, ffmpeg and Hugging Face access")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("setup", help="download all pinned models")
    p.add_argument("--no-text", action="store_true", help="skip Llama and Whisper")
    p.set_defaults(func=cmd_setup)

    for name, func, help_text in (
        ("predict", cmd_predict, "predict brain response to a video"),
        ("analyze", cmd_analyze, "score a video, find weak moments, write the HTML report"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("video")
        p.add_argument("--no-text", action="store_true", help="video+audio only, no Llama/ASR")
        p.add_argument("--lang", default=None, help="speech language code, e.g. en (default: auto)")
        p.add_argument(
            "--precision",
            choices=["fp16", "fp32"],
            default="fp16",
            help="video extractor precision; fp32 needs well over 8 GB VRAM (default: fp16)",
        )
        p.add_argument(
            "--force", action="store_true", help="recompute even if cached predictions match"
        )
        p.add_argument(
            "--offline", action="store_true", default=argparse.SUPPRESS, help="same as global --offline"
        )
        if name == "analyze":
            p.add_argument(
                "--json",
                action="store_true",
                help="print only the analysis summary as JSON on stdout (logs go to stderr)",
            )
        p.set_defaults(func=func)

    p = sub.add_parser("recut", help="cut or speed up weak moments (needs `analyze` first)")
    p.add_argument("video")
    p.add_argument("--plan", help="edit plan JSON to apply (default: an automatic plan)")
    p.add_argument(
        "--auto", action="store_true", help="suggest a plan from analysis.json (default without --plan)"
    )
    p.add_argument("--dry-run", action="store_true", help="write and print the plan, render nothing")
    p.add_argument("--out", help="output MP4 (default: out/<name>/<name>-recut.mp4)")
    p.add_argument(
        "--allow-heavy", action="store_true", help="allow plans that remove more than half the video"
    )
    p.add_argument("--json", action="store_true", help="print only JSON on stdout")
    p.set_defaults(func=cmd_recut)

    p = sub.add_parser("diff", help="compare two analysis.json files (before/after)")
    p.add_argument("before")
    p.add_argument("after")
    p.add_argument("--json", action="store_true", help="print only JSON on stdout")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("compare", help="score 2-4 versions block by block, pick each block's winner")
    p.add_argument("videos", nargs="+", metavar="VIDEO")
    p.add_argument(
        "--block", type=float, default=3.0, help="length of the middle blocks in seconds (default: 3)"
    )
    p.add_argument("--no-text", action="store_true", help="video+audio only, no Llama/ASR")
    p.add_argument("--lang", default=None, help="speech language code, e.g. en (default: auto)")
    p.add_argument(
        "--precision", choices=["fp16", "fp32"], default="fp16", help="video extractor precision"
    )
    p.add_argument("--json", action="store_true", help="print only compare.json on stdout")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("splice", help="render one video from a comparison's block winners")
    p.add_argument(
        "--compare", required=True, metavar="COMPARE_JSON", help="out/compare-.../compare.json"
    )
    p.add_argument("--plan", help="splice plan JSON to apply (default: an automatic plan)")
    p.add_argument(
        "--auto", action="store_true", help="take each block from its winner (default without --plan)"
    )
    p.add_argument("--dry-run", action="store_true", help="write and print the plan, render nothing")
    p.add_argument("--json", action="store_true", help="print only JSON on stdout")
    p.set_defaults(func=cmd_splice)

    p = sub.add_parser(
        "overlay", help="render a 9:16 video with the engagement curve and brain map (needs `analyze`)"
    )
    p.add_argument("video")
    p.add_argument("--out", help="output MP4 (default: out/<name>/<name>-overlay.mp4, -compare.mp4)")
    p.add_argument(
        "--compare",
        metavar="OTHER_VIDEO",
        help="play OTHER_VIDEO (after) next to VIDEO (before) with both curves; no brain panel",
    )
    p.add_argument("--no-watermark", action="store_true", help="leave out the unboxrepo wordmark")
    p.add_argument("--fps", type=int, default=30, help="output frame rate (default: 30)")
    p.set_defaults(func=cmd_overlay)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Only setup (and doctor's access check) may go online; everything else reads the cache.
    env.configure(offline=args.offline or args.command not in env.NETWORK_COMMANDS)
    _ensure_utf8_stdout()
    from .model import UnpinnedModel
    from .outputs import OutputError

    try:
        return args.func(args)
    except (env.SetupNeeded, OutputError, UnpinnedModel) as exc:
        return _error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
