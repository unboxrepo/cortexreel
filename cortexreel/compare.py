"""``compare``: score 2-4 versions of a video block by block, ``compare.json`` + ``compare.html``.

compare.json (``"schema": 1``):

- ``schema``: int, 1. ``cortexreel_version``, ``created``.
- ``block_seconds``: length of the middle blocks.
- ``versions``: list of ``{name, path, sha256, duration_seconds, analysis,
  report, summary}`` in command-line order. ``name`` is the file stem (unique),
  ``duration_seconds`` is probed from the file, ``summary`` is copied from its
  ``analysis.json``.
- ``blocks``: list of ``{id, start, end, align, scores, winner, tie, tied}``
  on the absolute timeline. ``id`` is ``hook`` (0-3 s), ``b1``, ``b2``... or
  ``ending``. ``align`` is ``start``, or ``end`` for the ending: each version's
  own last 3 s (``start``/``end`` then place it at the end of the longest
  version). ``scores`` maps each name to its mean onset-corrected engagement
  in the block, or null when the version does not reach it. ``winner`` is the
  highest score; ``tie`` is true when others are within TIE_POINTS of it, and
  ``tied`` lists every name that close (winner first).
- ``leader``: name of the version with the highest ``summary.overall``.
- ``best_of``: mean of the block winners' scores, a rough ceiling for a splice.
- ``disclaimer``: str.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

from . import __version__
from .outputs import OUT_ROOT, OutputError, check_no_links, is_link, write_text
from .report import CSS, DISCLAIMER, LICENSE_URL, E, _fmt_time, _polyline, _tick_step, csp
from .score import ENDING_SECONDS, HOOK_SECONDS

SCHEMA = 1
MIN_VERSIONS, MAX_VERSIONS = 2, 4
DEFAULT_BLOCK_SECONDS = 3.0
MIN_BLOCK_SECONDS = 1.0  # a remainder shorter than this (or half a block) joins the block before it
TIE_POINTS = 1.0


class CompareError(ValueError):
    pass


# --------------------------------------------------------------------- blocks


def make_blocks(longest: float, block_seconds: float = DEFAULT_BLOCK_SECONDS) -> list[dict]:
    """``hook``, middle blocks up to ``longest - 3`` and ``ending`` (end-aligned)."""
    if block_seconds <= 0:
        raise CompareError(f"--block must be positive, got {block_seconds:g}")
    blocks = [{"id": "hook", "start": 0.0, "end": min(HOOK_SECONDS, longest), "align": "start"}]
    mid_end = longest - ENDING_SECONDS
    edges = [HOOK_SECONDS]
    while edges[-1] + block_seconds < mid_end - 1e-9:
        edges.append(edges[-1] + block_seconds)
    if mid_end > HOOK_SECONDS:
        edges.append(mid_end)
        if len(edges) > 2 and edges[-1] - edges[-2] < max(MIN_BLOCK_SECONDS, block_seconds / 2):
            del edges[-2]
        for i, (a, b) in enumerate(zip(edges, edges[1:]), 1):
            blocks.append({"id": f"b{i}", "start": round(a, 3), "end": round(b, 3), "align": "start"})
    if longest > HOOK_SECONDS:
        blocks.append(
            {"id": "ending", "start": round(max(longest - ENDING_SECONDS, 0.0), 3),
             "end": round(longest, 3), "align": "end"}
        )  # fmt: skip
    return blocks


def block_score(times, engagement, tr: float, block: dict) -> float:
    """Mean engagement of the TRs starting inside ``block``; NaN if there are none.

    Matches ``score.window_means``: hook = TRs starting before 3 s, ending = TRs
    starting in the version's own last 3 s.
    """
    times = np.asarray(times, dtype=np.float64)
    engagement = np.asarray(engagement, dtype=np.float64)
    if block["align"] == "end":
        duration = float(times[-1] + tr) if len(times) else 0.0
        mask = times >= duration - ENDING_SECONDS
    else:
        mask = (times >= block["start"]) & (times < block["end"])
    return float(engagement[mask].mean()) if mask.any() else float("nan")


def pick_winner(scores: dict[str, float]) -> dict:
    """``{winner, tie, tied}`` from ``{name: score}`` (NaN = not reached)."""
    ranked = sorted(
        ((s, n) for n, s in scores.items() if s is not None and not math.isnan(s)),
        key=lambda p: -p[0],
    )
    if not ranked:
        return {"winner": None, "tie": False, "tied": []}
    best = ranked[0][0]
    tied = [n for s, n in ranked if best - s < TIE_POINTS]
    return {"winner": ranked[0][1], "tie": len(tied) > 1, "tied": tied}


def best_of(blocks: list[dict]) -> float | None:
    """Mean of the winning scores, one per block that has a winner."""
    won = [b["scores"][b["winner"]] for b in blocks if b.get("winner")]
    return round(float(np.mean(won)), 1) if won else None


def build_compare(versions: list[dict], analyses: list[dict], block_seconds: float) -> dict:
    """compare.json from per-version metadata (name, path, sha256, duration...) and analyses."""
    longest = max(a["video"]["duration_seconds"] for a in analyses)
    blocks = make_blocks(longest, block_seconds)
    for block in blocks:
        scores = {}
        for v, a in zip(versions, analyses):
            s = block_score(
                a["curves"]["times"], a["curves"]["engagement"], a["video"]["tr_seconds"], block
            )
            scores[v["name"]] = None if math.isnan(s) else round(s, 1)
        block["scores"] = scores
        block.update(pick_winner({k: (math.nan if s is None else s) for k, s in scores.items()}))
    for v, a in zip(versions, analyses):
        v["summary"] = dict(a["summary"])
    leader = max(versions, key=lambda v: v["summary"]["overall"])["name"]
    return {
        "schema": SCHEMA,
        "cortexreel_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "block_seconds": block_seconds,
        "versions": versions,
        "blocks": blocks,
        "leader": leader,
        "best_of": best_of(blocks),
        "disclaimer": DISCLAIMER,
    }


def compare_dir(names: list[str], out_root: Path | None = None) -> Path:
    """``out/compare-<a>-vs-<b>[...]`` under the CWD; refuses anything that escapes ``out/``."""
    base = Path.cwd() / (out_root or OUT_ROOT)
    if is_link(base):
        raise OutputError(f"Refusing to write through a link: {base}")
    root = base.resolve()
    folder = root / ("compare-" + "-vs-".join(names))
    if is_link(folder):
        raise OutputError(f"Refusing to write through a link: {folder}")
    target = folder.resolve()
    if target.parent != root:
        raise CompareError(f"Refusing to write outside {root}: {target}")
    return target


# --------------------------------------------------------------------- html

COMPARE_CSS = """
:root{--v0:#2f6fdb;--v1:#d9771e;--v2:#1f9d63;--v3:#9a4fd0}
@media (prefers-color-scheme:dark){:root{--v0:#6f9cf0;--v1:#f0a050;--v2:#45c98a;--v3:#c08af0}}
.curve{fill:none;stroke-width:2.5;stroke-linejoin:round}
.blk{opacity:.85}.blk.tie{opacity:.35}.blk.none{fill:var(--line)}
.blabel{fill:#fff;font-size:11px;font-weight:600}
.blabel.dim{fill:var(--fg)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:.9rem}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500;font-size:.8rem}
td a{color:inherit}
.lead{font-size:.72rem;font-weight:600;color:#fff;background:var(--peak);border-radius:99px;
padding:1px 7px;margin-left:6px;vertical-align:1px}
.win{font-weight:700}
.scroll{overflow-x:auto}
"""


def _rel(target: Path, out_dir: Path) -> str | None:
    try:
        return quote(os.path.relpath(target, out_dir).replace(os.sep, "/"))
    except ValueError:  # different drive on Windows
        return None


def _chart_svg(cmp: dict, analyses: list[dict]) -> str:
    W, H, L, R, T, B = 800, 280, 36, 10, 10, 26
    pw, ph = W - L - R, H - T - B
    dur = max(a["video"]["duration_seconds"] for a in analyses) or 1.0

    def x(t):
        return L + pw * t / dur

    def y(v):
        return T + ph * (1 - v / 100)

    parts = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Engagement of each version">']
    for v in (0, 25, 50, 75, 100):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="axis" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    step = _tick_step(dur)
    t = 0.0
    while t <= dur + 1e-9:
        parts.append(f'<text class="axis" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t:g}s</text>')
        t += step
    for i, a in enumerate(analyses):
        c = a["curves"]
        tr = a["video"]["tr_seconds"]
        xs = [x(s + tr / 2) for s in c["times"]]
        parts.append(
            _polyline(xs, [y(v) for v in c["engagement_smooth"]], "curve").replace(
                "<polyline", f'<polyline style="stroke:var(--v{i})"', 1
            )
        )
    parts.append("</svg>")
    return "".join(parts)


def _strip_svg(cmp: dict) -> str:
    W, H, L, R = 800, 34, 36, 10
    pw = W - L - R
    blocks = cmp["blocks"]
    dur = blocks[-1]["end"] or 1.0
    index = {v["name"]: i for i, v in enumerate(cmp["versions"])}

    def x(t):
        return L + pw * t / dur

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Winner of each block">']
    for b in blocks:
        x0, x1 = x(b["start"]), x(b["end"])
        scores = ", ".join(
            f"{n} {'–' if s is None else f'{s:.1f}'}" for n, s in b["scores"].items()
        )
        tip = f"{b['id']} {_fmt_time(b['start'])}–{_fmt_time(b['end'])}: {scores}"
        if b["align"] == "end":
            tip += " (each version's own last 3 s)"
        if b["winner"] is None:
            cls, style, label = "blk none", "", "–"
        else:
            cls = "blk tie" if b["tie"] else "blk"
            style = f' style="fill:var(--v{index[b["winner"]]})"'
            label = "tie" if b["tie"] else b["id"]
        parts.append(
            f'<rect class="{cls}"{style} x="{x0 + 1:.1f}" y="2" width="{max(x1 - x0 - 2, 1):.1f}" '
            f'height="{H - 4}" rx="4"><title>{E(tip)}</title></rect>'
        )
        if x1 - x0 >= 30:
            dim = " dim" if b["tie"] or b["winner"] is None else ""
            parts.append(
                f'<text class="blabel{dim}" x="{(x0 + x1) / 2:.1f}" y="{H / 2 + 4:.1f}" '
                f'text-anchor="middle">{E(label)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def render_html(cmp: dict, analyses: list[dict], out_dir: Path) -> str:
    versions = cmp["versions"]
    wins = {v["name"]: 0 for v in versions}
    for b in cmp["blocks"]:
        if b["winner"] and not b["tie"]:
            wins[b["winner"]] += 1
    keys = ("overall", "hook", "hold", "ending")
    top = {k: max(v["summary"][k] for v in versions) for k in keys}

    rows = []
    for i, v in enumerate(versions):
        href = _rel(Path(v["report"]), out_dir) if v.get("report") else None
        name = f'<a href="{E(href)}">{E(v["name"])}</a>' if href else E(v["name"])
        lead = '<span class="lead">leader</span>' if v["name"] == cmp["leader"] else ""
        cells = "".join(
            f'<td class="{"win" if v["summary"][k] == top[k] else ""}">{v["summary"][k]:.0f}</td>'
            for k in keys
        )
        rows.append(
            f'<tr><td><span class="sw" style="background:var(--v{i})"></span>{name}{lead}</td>'
            f'<td>{v["duration_seconds"]:.1f} s</td>{cells}<td>{wins[v["name"]]}</td></tr>'
        )
    summary_table = (
        '<div class="scroll"><table><tr><th>version</th><th>duration</th>'
        + "".join(f"<th>{k}</th>" for k in keys)
        + "<th>blocks won</th></tr>"
        + "".join(rows)
        + "</table></div>"
    )

    brows = []
    for b in cmp["blocks"]:
        cells = "".join(
            f'<td class="{"win" if n == b["winner"] else ""}">{"–" if s is None else f"{s:.1f}"}</td>'
            for n, s in b["scores"].items()
        )
        verdict = "–" if b["winner"] is None else ("tie: " + ", ".join(b["tied"]) if b["tie"] else b["winner"])
        when = "last 3 s" if b["align"] == "end" else f"{_fmt_time(b['start'])}–{_fmt_time(b['end'])}"
        brows.append(f"<tr><td>{E(b['id'])}</td><td>{when}</td>{cells}<td>{E(verdict)}</td></tr>")
    block_table = (
        '<div class="scroll"><table><tr><th>block</th><th>time</th>'
        + "".join(f"<th>{E(v['name'])}</th>" for v in versions)
        + "<th>winner</th></tr>"
        + "".join(brows)
        + "</table></div>"
    )

    legend = "".join(
        f'<span><span class="sw" style="background:var(--v{i})"></span>{E(v["name"])}</span>'
        for i, v in enumerate(versions)
    )
    leader = next(v for v in versions if v["name"] == cmp["leader"])
    best = cmp["best_of"]
    tiles = (
        f'<div class="tile"><div class="k">leader</div><div class="v">{leader["summary"]["overall"]:.0f}</div>'
        f'<div class="h">{E(leader["name"])} overall</div></div>'
        f'<div class="tile"><div class="k">best of blocks</div><div class="v">'
        f'{"–" if best is None else f"{best:.0f}"}</div><div class="h">mean of block winners</div></div>'
        f'<div class="tile"><div class="k">versions</div><div class="v">{len(versions)}</div>'
        f'<div class="h">{cmp["block_seconds"]:g} s blocks</div></div>'
        f'<div class="tile"><div class="k">blocks</div><div class="v">{len(cmp["blocks"])}</div>'
        f'<div class="h">hook, middle, ending</div></div>'
    )
    title = " vs ".join(v["name"] for v in versions)

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{csp()}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{E(title)} · cortexreel</title>
<style>{CSS}{COMPARE_CSS}</style>
</head>
<body>
<main>
<h1>{E(title)}</h1>
<p class="sub">{len(versions)} versions · cortexreel {E(cmp["cortexreel_version"])}</p>
<div class="tiles">{tiles}</div>

<h2>Engagement</h2>
<div class="panel">
{_chart_svg(cmp, analyses)}
{_strip_svg(cmp)}
<div class="legend">{legend}<span>strip: winner of each block (faded = tie within {TIE_POINTS:g} point)</span></div>
</div>

<h2>Summary</h2>
<div class="panel">{summary_table}</div>

<h2>Blocks</h2>
<div class="panel">{block_table}</div>
<p class="sub">The ending block is each version's own last 3 s. "Best of blocks" is a rough
estimate of a splice of the winners; render it with <code>cortexreel splice</code> and
analyze the result to measure it.</p>

<footer>
Predictions from Meta TRIBE v2 (<a rel="license" href="{LICENSE_URL}">CC BY-NC 4.0</a>).
Model estimate of an average viewer, not a measurement and not a virality guarantee.
</footer>
</main>
</body>
</html>
"""


def write_compare(out_dir: Path, cmp: dict, analyses: list[dict]) -> tuple[Path, Path]:
    """Write ``compare.json`` and ``compare.html``; returns both paths."""
    check_no_links(out_dir)
    json_path = write_text(out_dir / "compare.json", json.dumps(cmp, indent=2, ensure_ascii=False))
    html_path = write_text(out_dir / "compare.html", render_html(cmp, analyses, out_dir))
    return json_path, html_path
