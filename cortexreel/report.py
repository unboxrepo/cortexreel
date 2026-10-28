"""``analysis.json`` and a self-contained offline ``report.html``.

analysis.json (``"schema": 1``) — keys are stable; add, never rename:

- ``schema``: int, 1.
- ``cortexreel_version``: str.
- ``created``: ISO-8601 UTC timestamp.
- ``video``: ``name`` (file name), ``path`` (relative to the CWD where possible), ``sha256``,
  ``duration_seconds`` (n_tr * tr_seconds), ``n_tr``, ``tr_seconds``.
- ``metrics``: the run's ``metrics.json`` (timings, use_text, precision...) without the GPU name.
- ``model_pins``: ``{hf repo id: commit}`` for the models this run used.
- ``engagement_mapping``: ``center``, ``scale``, ``formula``.
- ``regions``: ``{group: {weight, n_vertices, labels}}``.
- ``summary``: ``overall``, ``hook``, ``hold``, ``ending`` (0-100, 1 decimal).
  hook = first 3 s, ending = last 3 s, hold = between them.
- ``moments``: list of ``{kind, start, end, score, delta, reason,
  regions_delta, words}``, sorted by ``start``. ``kind`` is ``dip``, ``peak``,
  ``hook_weak`` or ``ending_drop``; times in seconds; ``score`` is 0-100;
  ``delta`` is score minus the video median (dip/peak) or minus ``hold``
  (hook_weak/ending_drop); ``regions_delta`` is each group's change during the
  moment in units of that group's standard deviation over the video.
- ``curves``: ``times`` (TR start, s), ``engagement`` (onset-corrected, see
  ``onset_correction`` in regions.json), ``engagement_uncorrected``,
  ``engagement_smooth``, ``engagement_raw`` and ``regions`` (``{group: mean activation}``), all
  rounded to 3 decimals. ``summary_json()`` drops this key.
- ``disclaimer``: str.
"""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np

from . import __version__, pins
from .moments import Moment
from .outputs import check_no_links, rel_path, write_text
from .score import Scores

SCHEMA = 1
DISCLAIMER = (
    "Predictions from Meta TRIBE v2 (CC BY-NC 4.0). Model estimate of an average "
    "viewer, not a measurement and not a virality guarantee."
)
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"
PRIVATE_METRICS = ("gpu",)
KIND_LABELS = {
    "dip": "Dip",
    "peak": "Peak",
    "hook_weak": "Weak hook",
    "ending_drop": "Ending drop",
}


def _r(values, ndigits: int = 3) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def shareable_metrics(metrics: dict, video: Path) -> dict:
    """``metrics`` for analysis.json: no GPU name, a relative video path.

    analysis.json is the file people share; the GPU name stays in the local
    metrics.json cache, where ``predict`` prints it.
    """
    out = {k: v for k, v in metrics.items() if k not in PRIVATE_METRICS}
    if "video" in out:
        out["video"] = rel_path(video)
    return out


def build_analysis(
    video: Path,
    metrics: dict,
    scores: Scores,
    moments: list[Moment],
    config: dict,
    masks: dict[str, np.ndarray],
) -> dict:
    use_text = bool(metrics.get("use_text", True))
    mapping = config["engagement"]
    return {
        "schema": SCHEMA,
        "cortexreel_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {
            "name": video.name,
            "path": rel_path(video),
            "sha256": metrics.get("video_sha256"),
            "duration_seconds": round(scores.duration, 3),
            "n_tr": int(len(scores.times)),
            "tr_seconds": scores.tr_seconds,
        },
        "metrics": shareable_metrics(metrics, video),
        "model_pins": {p.repo_id: p.revision for p in pins.needed(use_text)},
        "engagement_mapping": {
            "center": float(mapping["center"]),
            "scale": float(mapping["scale"]),
            "formula": "100 * sigmoid((raw - center) / scale)",
        },
        "regions": {
            name: {
                "weight": scores.weights[name],
                "n_vertices": int(masks[name].sum()),
                "labels": list(group["labels"]),
            }
            for name, group in config["groups"].items()
        },
        "summary": {k: round(v, 1) for k, v in scores.summary().items()},
        "moments": [m.to_dict() for m in moments],
        "curves": {
            "times": _r(scores.times),
            "engagement": _r(scores.engagement),
            "engagement_uncorrected": _r(scores.engagement_uncorrected),
            "engagement_smooth": _r(scores.engagement_smooth),
            "engagement_raw": _r(scores.engagement_raw),
            "regions": {k: _r(v) for k, v in scores.region_curves.items()},
        },
        "disclaimer": DISCLAIMER,
    }


def summary_json(analysis: dict) -> dict:
    """``analysis`` without the per-TR curves."""
    return {k: v for k, v in analysis.items() if k != "curves"}


# --------------------------------------------------------------------- images


def brain_png(mean_map: np.ndarray) -> bytes | None:
    """Lateral left + right inflated fsaverage5 views of ``mean_map`` as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from nilearn import datasets, plotting

        mesh = datasets.fetch_surf_fsaverage("fsaverage5")
        n = len(mean_map) // 2
        vmax = float(np.percentile(np.abs(mean_map), 99)) or 1.0
        fig = plt.figure(figsize=(8.6, 3.2), dpi=110)
        for i, (hemi, values) in enumerate((("left", mean_map[:n]), ("right", mean_map[n:]))):
            ax = fig.add_axes((0.46 * i, 0.0, 0.46, 1.0), projection="3d")
            plotting.plot_surf_stat_map(
                surf_mesh=mesh[f"infl_{hemi}"],
                stat_map=np.asarray(values, dtype=np.float64),
                bg_map=mesh[f"sulc_{hemi}"],
                hemi=hemi,
                view="lateral",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
                colorbar=False,
                axes=ax,
                figure=fig,
            )
        cax = fig.add_axes((0.94, 0.25, 0.015, 0.5))
        fig.colorbar(
            matplotlib.cm.ScalarMappable(
                matplotlib.colors.Normalize(-vmax, vmax), cmap="RdBu_r"
            ),
            cax=cax,
        ).ax.tick_params(labelsize=8)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor="white")
        plt.close(fig)
        return buf.getvalue()
    except Exception:  # the report is still useful without the brain picture
        return None


def thumbnails(video: Path, moments: list[Moment], duration: float) -> list[bytes | None]:
    from .media import frame_jpeg

    out = []
    for m in moments:
        t = min((m.start + m.end) / 2, max(duration - 0.1, 0.0))
        out.append(frame_jpeg(video, t) if video.is_file() else None)
    return out


def _data_uri(mime: str, data: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


# --------------------------------------------------------------------- charts

E = html.escape


def _fmt_time(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def _tick_step(duration: float) -> float:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if duration / step <= 10:
            return step
    return 600


def _polyline(xs, ys, cls: str) -> str:
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    return f'<polyline class="{cls}" points="{points}"/>'


def _engagement_svg(scores: Scores, moments: list[Moment]) -> str:
    W, H, L, R, T, B = 800, 280, 36, 10, 10, 26
    pw, ph = W - L - R, H - T - B
    dur = scores.duration or 1.0
    tr = scores.tr_seconds

    def x(t):
        return L + pw * t / dur

    def y(v):
        return T + ph * (1 - v / 100)

    parts = [
        f'<svg id="chart" class="chart" viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="Engagement over time" data-l="{L}" data-w="{pw}" data-dur="{dur:.3f}">'
    ]
    for v in (0, 25, 50, 75, 100):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        parts.append(f'<text class="axis" x="{L - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    step = _tick_step(dur)
    t = 0.0
    while t <= dur + 1e-9:
        parts.append(f'<text class="axis" x="{x(t):.1f}" y="{H - 8}" text-anchor="middle">{t:g}s</text>')
        t += step
    for m in moments:
        if m.kind != "peak":
            cls = "dip" if m.kind == "dip" else "dip soft"
            parts.append(
                f'<rect class="{cls}" x="{x(m.start):.1f}" y="{T}" '
                f'width="{max(x(m.end) - x(m.start), 1):.1f}" height="{ph}">'
                f"<title>{E(KIND_LABELS[m.kind])} {_fmt_time(m.start)}–{_fmt_time(m.end)}</title></rect>"
            )
    centers = scores.times + tr / 2
    xs = [x(c) for c in centers]
    parts.append(_polyline(xs, [y(v) for v in scores.engagement], "raw"))
    parts.append(_polyline(xs, [y(v) for v in scores.engagement_smooth], "smooth"))
    for m in moments:
        if m.kind == "peak":
            cx, cy = x((m.start + m.end) / 2), y(m.score)
            parts.append(
                f'<circle class="peak" cx="{cx:.1f}" cy="{cy:.1f}" r="5">'
                f"<title>Peak {_fmt_time(m.start)} · {m.score:.0f}</title></circle>"
            )
    parts.append(f'<line id="playhead" class="playhead" x1="{L}" x2="{L}" y1="{T}" y2="{T + ph}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _mini_svg(name: str, curve: np.ndarray, scores: Scores, moments: list[Moment]) -> str:
    W, H, P = 240, 70, 4
    dur = scores.duration or 1.0
    lo, hi = float(curve.min()), float(curve.max())
    span = (hi - lo) or 1.0

    def x(t):
        return P + (W - 2 * P) * t / dur

    def y(v):
        return P + (H - 2 * P) * (1 - (v - lo) / span)

    parts = [f'<svg class="mini" viewBox="0 0 {W} {H}" role="img" aria-label="{E(name)} over time">']
    for m in moments:
        if m.kind == "dip":
            parts.append(
                f'<rect class="dip" x="{x(m.start):.1f}" y="0" '
                f'width="{max(x(m.end) - x(m.start), 1):.1f}" height="{H}"/>'
            )
    xs = [x(c) for c in scores.times + scores.tr_seconds / 2]
    parts.append(_polyline(xs, [y(v) for v in curve], "smooth"))
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------- html

CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1f;--muted:#6b6b70;--line:#e3e3e0;
--accent:#2f6fdb;--raw:#9db8e8;--dip:#d64545;--peak:#1f9d63;--warn:#c27a12}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1e1e21;--fg:#ececee;
--muted:#9a9aa1;--line:#2e2e33;--accent:#6f9cf0;--raw:#3c5687;--dip:#f06a6a;
--peak:#45c98a;--warn:#e0a040}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:1.4rem;margin:0 0 4px;overflow-wrap:anywhere}
h2{font-size:1.05rem;margin:32px 0 12px}
.sub{color:var(--muted);font-size:.85rem;margin:0}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:20px}
.tile,.card{background:var(--card);border:1px solid var(--line);border-radius:10px}
.tile{padding:12px 14px}
.tile .k{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:1.8rem;font-weight:600;font-variant-numeric:tabular-nums}
.tile .h{color:var(--muted);font-size:.75rem}
.panel{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}
video{width:100%;max-height:60vh;border-radius:8px;background:#000;display:block;margin-bottom:12px}
svg{display:block;width:100%;height:auto}
.chart{cursor:crosshair}
.grid{stroke:var(--line);stroke-width:1}
.axis{fill:var(--muted);font-size:11px}
.raw{fill:none;stroke:var(--raw);stroke-width:1.5}
.smooth{fill:none;stroke:var(--accent);stroke-width:2.5;stroke-linejoin:round}
.dip{fill:var(--dip);opacity:.16}
.dip.soft{fill:var(--warn);opacity:.12}
.peak{fill:var(--peak);stroke:var(--card);stroke-width:2}
.playhead{stroke:var(--fg);stroke-width:1;opacity:0}
.legend{color:var(--muted);font-size:.8rem;margin:8px 0 0;display:flex;gap:16px;flex-wrap:wrap}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
.minis{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}
.minis .card{padding:10px 12px}
.minis .t{font-weight:600;font-size:.9rem}
.minis .w{color:var(--muted);font-size:.75rem;float:right}
.mini .smooth{stroke-width:1.8}
.moments{display:grid;gap:12px}
.moment{display:flex;gap:14px;padding:12px}
.moment img{width:auto;height:auto;max-width:160px;max-height:200px;border-radius:6px;flex:none;
align-self:flex-start;background:#000}
.moment .meta{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap}
.badge{font-size:.72rem;font-weight:600;text-transform:uppercase;letter-spacing:.04em;
padding:2px 8px;border-radius:99px;color:#fff;background:var(--muted)}
.badge.dip{background:var(--dip);opacity:1}.badge.peak{background:var(--peak)}
.badge.hook_weak,.badge.ending_drop{background:var(--warn)}
.when{font-variant-numeric:tabular-nums;font-weight:600}
.num{color:var(--muted);font-size:.85rem}
.reason{margin:4px 0}
.words{color:var(--muted);font-style:italic;margin:0;overflow-wrap:anywhere}
.seek{font:inherit;font-size:.8rem;color:var(--accent);background:none;border:1px solid var(--line);
border-radius:6px;padding:2px 8px;cursor:pointer;margin-top:6px}
.brain{background:#fff;border-radius:10px;padding:8px}
.brain img{width:100%;height:auto;display:block}
footer{margin-top:40px;color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);padding-top:12px}
footer a{color:inherit}
@media (max-width:640px){.tiles{grid-template-columns:repeat(2,1fr)}
.moment{flex-direction:column}.moment img{max-width:100%;max-height:260px}}
"""

JS = """
(function(){
var v=document.getElementById('player'),svg=document.getElementById('chart'),
ph=document.getElementById('playhead');
if(!svg)return;
var L=+svg.dataset.l,W=+svg.dataset.w,D=+svg.dataset.dur;
function seek(t){if(!v)return;v.currentTime=Math.max(0,Math.min(t,D));v.play();}
svg.addEventListener('click',function(e){
var p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
var q=p.matrixTransform(svg.getScreenCTM().inverse());
seek((q.x-L)/W*D);});
document.querySelectorAll('[data-t]').forEach(function(b){
b.addEventListener('click',function(){seek(+b.dataset.t);});});
if(v&&ph)v.addEventListener('timeupdate',function(){
var x=L+W*Math.min(v.currentTime/D,1);ph.setAttribute('x1',x);ph.setAttribute('x2',x);
ph.style.opacity=1;});
})();
"""


def csp(script: str | None = None) -> str:
    """Content-Security-Policy for our offline pages: inline CSS, data: images, the local video.

    Only ``script`` (exact text of the page's one inline script) may run, by hash.
    """
    scripts = (
        "'sha256-" + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode() + "'"
        if script
        else "'none'"
    )
    return (
        "default-src 'none'; img-src data:; media-src 'self' file:; style-src 'unsafe-inline'; "
        f"script-src {scripts}; base-uri 'none'; form-action 'none'"
    )


TILE_HINTS = {
    "overall": "whole video",
    "hook": "first 3 s",
    "hold": "middle",
    "ending": "last 3 s",
}


def _video_src(video: Path, out_dir: Path) -> str | None:
    if not video.is_file():
        return None
    try:
        rel = os.path.relpath(video, out_dir)
    except ValueError:  # different drive on Windows
        return None
    return quote(rel.replace(os.sep, "/"))


def render_html(
    analysis: dict,
    scores: Scores,
    moments: list[Moment],
    out_dir: Path,
    video: Path,
    thumbs: list[bytes | None] | None = None,
    brain: bytes | None = None,
) -> str:
    name = analysis["video"]["name"]
    summary = analysis["summary"]
    thumbs = thumbs or [None] * len(moments)
    src = _video_src(video, out_dir)

    tiles = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{summary[k]:.0f}</div>'
        f'<div class="h">{TILE_HINTS[k]}</div></div>'
        for k in ("overall", "hook", "hold", "ending")
    )
    player = (
        f'<video id="player" controls preload="metadata" src="{E(src)}"></video>' if src else ""
    )

    minis = []
    for group, curve in scores.region_curves.items():
        weight = scores.weights.get(group, 0.0)
        minis.append(
            f'<div class="card"><span class="w">weight {weight:.2f}</span>'
            f'<div class="t">{E(group.replace("_", " / "))}</div>'
            f"{_mini_svg(group, curve, scores, moments)}</div>"
        )

    cards = []
    for m, thumb in zip(moments, thumbs):
        img = (
            f'<img src="{_data_uri("image/jpeg", thumb)}" alt="Frame at {_fmt_time((m.start + m.end) / 2)}">'
            if thumb
            else ""
        )
        words = f'<p class="words">“{E(m.words)}”</p>' if m.words else ""
        seek = (
            f'<button class="seek" type="button" data-t="{m.start:.2f}">Play from {_fmt_time(m.start)}</button>'
            if src
            else ""
        )
        cards.append(
            f'<div class="card moment">{img}<div>'
            f'<div class="meta"><span class="badge {E(m.kind)}">{E(KIND_LABELS.get(m.kind, m.kind))}</span>'
            f'<span class="when">{_fmt_time(m.start)}–{_fmt_time(m.end)}</span>'
            f'<span class="num">score {m.score:.0f} ({m.delta:+.1f})</span></div>'
            f'<p class="reason">{E(m.reason)}</p>{words}{seek}</div></div>'
        )
    moments_html = (
        f'<div class="moments">{"".join(cards)}</div>'
        if cards
        else '<p class="sub">No dips or peaks stood out.</p>'
    )
    brain_html = (
        '<h2>Mean predicted activation</h2><div class="brain">'
        f'<img src="{_data_uri("image/png", brain)}" alt="Mean predicted activation, '
        'left and right hemispheres, lateral view"></div>'
        '<p class="sub">Inflated fsaverage5 surface, lateral views. Red: above the '
        "model's baseline, blue: below.</p>"
        if brain
        else ""
    )
    meta = analysis["metrics"]
    sub = (
        f"{analysis['video']['duration_seconds']:g} s · {analysis['video']['n_tr']} TRs · "
        f"text channel {'on' if meta.get('use_text') else 'off'} · "
        f"cortexreel {E(analysis['cortexreel_version'])}"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{csp(JS)}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{E(name)} · cortexreel</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>{E(name)}</h1>
<p class="sub">{sub}</p>
<div class="tiles">{tiles}</div>

<h2>Engagement</h2>
<div class="panel">
{player}
{_engagement_svg(scores, moments)}
<div class="legend">
<span><span class="sw" style="background:var(--accent)"></span>smoothed</span>
<span><span class="sw" style="background:var(--raw)"></span>per second</span>
<span><span class="sw" style="background:var(--dip);opacity:.5"></span>dip</span>
<span><span class="sw" style="background:var(--warn);opacity:.5"></span>weak hook / ending</span>
<span><span class="sw" style="background:var(--peak)"></span>peak</span>
{'<span>click the chart to jump the video</span>' if src else ''}
</div>
</div>

<h2>Regions</h2>
<div class="minis">{"".join(minis)}</div>

<h2>Moments</h2>
{moments_html}

{brain_html}

<footer>
Predictions from Meta TRIBE v2 (<a rel="license" href="{LICENSE_URL}">CC BY-NC 4.0</a>).
Model estimate of an average viewer, not a measurement and not a virality guarantee.
</footer>
</main>
<script>{JS}</script>
</body>
</html>
"""


def write_report(
    out_dir: Path,
    video: Path,
    analysis: dict,
    scores: Scores,
    moments: list[Moment],
    mean_map: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Write ``analysis.json`` and ``report.html``; returns both paths."""
    check_no_links(out_dir)
    json_path = write_text(out_dir / "analysis.json", json.dumps(analysis, indent=2, ensure_ascii=False))
    thumbs = thumbnails(video, moments, scores.duration)
    brain = brain_png(mean_map) if mean_map is not None else None
    html_path = write_text(
        out_dir / "report.html", render_html(analysis, scores, moments, out_dir, video, thumbs, brain)
    )
    return json_path, html_path
