"""``overlay``: a 1080x1920 (9:16) showcase video of a cortexreel analysis.

Bands, top to bottom: the source video, the engagement strip, the brain panel
(lateral views of the predicted response, plus region bars) and a footer.
Static layers are drawn once with Pillow; each output frame only redraws the
parts that move. Frames go to one ffmpeg process as raw RGB; ffmpeg overlays
the source video onto the top band itself (so it is never decoded in Python)
and copies the source audio.

With ``other`` (``--compare``) two videos play side by side, both curves share
the strip and a before/after summary replaces the brain panel.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .outputs import prepare_file, same_file, write_text

W, H = 1080, 1920
FPS = 30
VIDEO_H, STRIP_H, BRAIN_H, FOOTER_H = 1080, 380, 380, 80
FONT_PATH = Path(__file__).parent / "data" / "fonts" / "InterVariable.ttf"
SS = 3  # supersampling factor for anti-aliased shapes
MARGIN = 40

BG = (11, 13, 16)
GRID = (40, 45, 54)
TEXT = (238, 240, 243)
MUTED = (156, 163, 174)
ACCENT = (92, 156, 255)
DIP = (255, 94, 94)
PEAK = (61, 220, 132)
COMPARE_COLORS = ((255, 166, 77), (76, 201, 240))  # before, after
BRAIN_CMAP = "RdBu_r"
BRAIN_BOX = (680, 274)  # brain image size inside the panel
BRAIN_RENDER_VERSION = 1  # bump to invalidate cached brain_frames/
MOMENT_KINDS = ("dip", "peak")  # earlier wins when both cover a moment
SUMMARY_KEYS = ("overall", "hook", "hold", "ending")
FOOTER_TEXT = "cortexreel · Meta TRIBE v2 (CC BY-NC 4.0) · model estimate, not a measurement"
WATERMARK = "unboxrepo"


class OverlayError(Exception):
    """A user-facing problem (missing analysis, bad arguments)."""


# --------------------------------------------------------------------- layout


@dataclass(frozen=True)
class Band:
    y: int
    h: int


def layout(compare: bool = False) -> dict[str, Band]:
    """Horizontal bands top to bottom; compare mode gives the brain's height to the strip."""
    if compare:
        heights = {"video": VIDEO_H, "strip": STRIP_H + BRAIN_H, "footer": FOOTER_H}
    else:
        heights = {"video": VIDEO_H, "strip": STRIP_H, "brain": BRAIN_H, "footer": FOOTER_H}
    bands, y = {}, 0
    for name, h in heights.items():
        bands[name] = Band(y, h)
        y += h
    return bands


def fit(src_w: int, src_h: int, box_w: int, box_h: int) -> tuple[int, int, int, int]:
    """``(w, h, x, y)``: the source scaled to fit the box, even-sized and centered."""
    scale = min(box_w / src_w, box_h / src_h)
    w = min(box_w, max(2, round(src_w * scale / 2) * 2))
    h = min(box_h, max(2, round(src_h * scale / 2) * 2))
    return w, h, (box_w - w) // 2, (box_h - h) // 2


# --------------------------------------------------------------------- timing


def value_at(centers: np.ndarray, values: np.ndarray, t: float) -> float:
    """``values`` (one per TR center) linearly interpolated at ``t``, held flat past both ends."""
    return float(np.interp(t, centers, values))


def moment_at(moments: list[dict], t: float) -> dict | None:
    """The dip or peak covering ``t`` (a dip wins if both do)."""
    hits = [m for m in moments if m["kind"] in MOMENT_KINDS and m["start"] <= t < m["end"]]
    hits.sort(key=lambda m: MOMENT_KINDS.index(m["kind"]))
    return hits[0] if hits else None


def tr_blend(n_tr: int, first: float, tr: float, t: float) -> tuple[int, int, float]:
    """TRs ``(i, j)`` and the weight of ``j`` for a cross-fade at ``t``.

    TR ``i`` shows alone at its center, ``first + (i + 0.5) * tr``.
    """
    p = min(max((t - first) / tr - 0.5, 0.0), n_tr - 1)
    i = int(p)
    return i, min(i + 1, n_tr - 1), p - i


def headline(reason: str) -> str:
    """First clause of a moment reason: "Visual activity drops: ..." -> "Visual activity drops"."""
    for sep in (":", ";"):
        reason = reason.split(sep, 1)[0]
    return reason.strip().rstrip(".")


# --------------------------------------------------------------------- data


@dataclass
class Track:
    name: str  # file stem, used as the label
    video: Path
    out_dir: Path
    duration: float  # container duration, seconds
    size: tuple[int, int]
    audio_codec: str | None
    times: np.ndarray  # TR start times
    tr: float
    engagement: np.ndarray  # 0-100, onset-corrected
    regions: dict[str, np.ndarray]
    moments: list[dict]
    summary: dict[str, float]

    @property
    def centers(self) -> np.ndarray:
        return self.times + self.tr / 2

    def score(self, t: float) -> float:
        return value_at(self.centers, self.engagement, t)


def load_track(video: str | Path) -> Track:
    from .media import probe, sha256_file
    from .predict import output_dir

    path = Path(video).resolve()
    if not path.is_file():
        raise OverlayError(f"no such video: {video}")
    out_dir = output_dir(path)
    analysis_path = out_dir / "analysis.json"
    if not analysis_path.is_file():
        raise OverlayError(f"no analysis for {path.name}. Run first: cortexreel analyze {video}")
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    if analysis.get("video", {}).get("sha256") not in (None, sha256_file(path)):
        raise OverlayError(
            f"{analysis_path} was made from a different version of {path.name}. "
            f"Run again: cortexreel analyze {video}"
        )
    info = probe(path)
    if not (info.width and info.height):
        raise OverlayError(f"{path.name} has no video stream")
    curves = analysis["curves"]
    return Track(
        name=path.stem,
        video=path,
        out_dir=out_dir,
        duration=info.duration,
        size=(info.width, info.height),
        audio_codec=info.audio_codec,
        times=np.asarray(curves["times"], dtype=np.float64),
        tr=float(analysis["video"]["tr_seconds"]),
        engagement=np.asarray(curves["engagement"], dtype=np.float64),
        regions={k: np.asarray(v, dtype=np.float64) for k, v in curves["regions"].items()},
        moments=list(analysis.get("moments", [])),
        summary=dict(analysis["summary"]),
    )


# --------------------------------------------------------------------- drawing


@lru_cache(maxsize=None)
def font(size: int, weight: int = 400) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(str(FONT_PATH), size)
    f.set_variation_by_axes([min(max(size, 14), 32), weight])  # optical size, weight
    return f


def _mix(color: tuple, base: tuple, amount: float) -> tuple:
    return tuple(round(c * amount + b * (1 - amount)) for c, b in zip(color, base))


def _clip_text(text: str, f: ImageFont.FreeTypeFont, max_w: float) -> str:
    if f.getlength(text) <= max_w:
        return text
    while text and f.getlength(text + "…") > max_w:
        text = text[:-1]
    return text.rstrip() + "…"


def _supersampled(size: tuple[int, int], paint) -> Image.Image:
    """BG-filled RGB image whose shapes ``paint(draw, s)`` draws at ``s``x, then downsampled."""
    big = Image.new("RGB", (size[0] * SS, size[1] * SS), BG)
    paint(ImageDraw.Draw(big, "RGBA"), SS)
    return big.resize(size, Image.LANCZOS)


@lru_cache(maxsize=None)
def _dot(color: tuple, radius: int = 9) -> Image.Image:
    d = 2 * radius + 6
    big = Image.new("RGBA", (d * SS, d * SS), (0, 0, 0, 0))
    c = d * SS / 2
    ImageDraw.Draw(big).ellipse(
        (c - radius * SS, c - radius * SS, c + radius * SS, c + radius * SS),
        fill=color, outline=BG, width=3 * SS,
    )  # fmt: skip
    return big.resize((d, d), Image.LANCZOS)


def _paste_dot(canvas: Image.Image, color: tuple, x: float, y: float) -> None:
    dot = _dot(color)
    canvas.paste(dot, (round(x - dot.width / 2), round(y - dot.height / 2)), dot)


def _fit_image(img: Image.Image, box: tuple[int, int]) -> Image.Image:
    scale = min(box[0] / img.width, box[1] / img.height)
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img if size == img.size else img.resize(size, Image.LANCZOS)


def _tick_step(duration: float) -> float:
    for step in (1, 2, 5, 10, 15, 30, 60, 120, 300):
        if duration / step <= 9:
            return step
    return 600


@dataclass(frozen=True)
class Chart:
    """Plot rectangle in band-local pixels; time axis 0..duration, value axis 0..100."""

    x0: int
    y0: int
    x1: int
    y1: int
    duration: float

    def x(self, t: float) -> float:
        return self.x0 + (self.x1 - self.x0) * min(max(t / self.duration, 0.0), 1.0)

    def y(self, v: float) -> float:
        return self.y1 - (self.y1 - self.y0) * min(max(v, 0.0), 100.0) / 100

    def paint_grid(self, d: ImageDraw.ImageDraw, s: int) -> None:
        for v in (0, 25, 50, 75, 100):
            d.line((self.x0 * s, self.y(v) * s, self.x1 * s, self.y(v) * s), fill=GRID, width=s)

    def paint_curve(self, d, s: int, track: Track, color: tuple, area: bool) -> None:
        pts = [(self.x(c) * s, self.y(v) * s) for c, v in zip(track.centers, track.engagement)]
        if area and len(pts) > 1:
            floor = [(pts[-1][0], self.y1 * s), (pts[0][0], self.y1 * s)]
            d.polygon(pts + floor, fill=color + (40,))
        d.line(pts, fill=color, width=4 * s, joint="curve")

    def draw_axes(self, img: Image.Image) -> None:
        d = ImageDraw.Draw(img)
        f = font(22)
        for v in (0, 50, 100):
            d.text((self.x0 - 12, self.y(v)), str(v), font=f, fill=MUTED, anchor="rm")
        step = _tick_step(self.duration)
        t = 0.0
        while t <= self.duration + 1e-9:
            label = f"{t:g}s"
            half = f.getlength(label) / 2
            x = min(max(self.x(t), self.x0 + half), self.x1 - half)
            d.text((x, self.y1 + 12), label, font=f, fill=MUTED, anchor="ma")
            t += step

    def playhead(self, canvas: Image.Image, oy: int, t: float) -> float:
        x = round(self.x(t))
        ImageDraw.Draw(canvas).line((x, oy + self.y0 - 8, x, oy + self.y1), fill=TEXT, width=3)
        return x


def _paste_played(canvas, dim: Image.Image, bright: Image.Image, band: Band, x0: int, px: float):
    """The strip with the curve bright up to the playhead and dimmed after it."""
    canvas.paste(dim, (0, band.y))
    if px > x0:
        canvas.paste(bright.crop((x0, 0, int(px) + 1, band.h)), (x0, band.y))


def _badge(d: ImageDraw.ImageDraw, x: float, y: float, label: str, color: tuple) -> float:
    """A filled label; returns its right edge."""
    f = font(22, 700)
    w = f.getlength(label)
    d.rounded_rectangle((x, y, x + w + 20, y + 34), radius=8, fill=color)
    d.text((x + 10, y + 17), label, font=f, fill=BG, anchor="lm")
    return x + w + 20


# --------------------------------------------------------------------- panels


class StripPanel:
    """One engagement curve, a big current score and the current dip/peak."""

    NUM_RIGHT = MARGIN + 190  # right edge of the big number ("100" fits)

    def __init__(self, band: Band, track: Track, duration: float):
        self.band, self.track = band, track
        self.chart = Chart(78, 140, W - MARGIN, 322, duration)
        chart = self.chart

        def shared(d, s):
            chart.paint_grid(d, s)
            for m in track.moments:
                if m["kind"] == "dip":
                    box = (chart.x(m["start"]) * s, chart.y0 * s, chart.x(m["end"]) * s, chart.y1 * s)
                    d.rectangle(box, fill=DIP + (48,))
                    d.rectangle((box[0], box[3] - 5 * s, box[2], box[3]), fill=DIP)

        def peaks(d, s):
            for m in track.moments:
                if m["kind"] == "peak":
                    mid = (m["start"] + m["end"]) / 2
                    x, y, r = chart.x(mid) * s, chart.y(track.score(mid)) * s, 8 * s
                    d.ellipse((x - r, y - r, x + r, y + r), fill=PEAK, outline=BG, width=3 * s)

        def layer(color):
            def paint(d, s):
                shared(d, s)
                chart.paint_curve(d, s, track, color, area=True)
                peaks(d, s)

            img = _supersampled((W, band.h), paint)
            chart.draw_axes(img)
            d = ImageDraw.Draw(img)
            d.text((self.NUM_RIGHT + 20, 26), "predicted engagement", font=font(24), fill=MUTED)
            for i, key in enumerate(("hook", "overall")):
                d.text(
                    (W - MARGIN, 26 + 34 * i), f"{key} {track.summary[key]:.0f}",
                    font=font(24, 500), fill=MUTED, anchor="ra",
                )  # fmt: skip
            return img

        self.dim, self.bright = layer(_mix(ACCENT, BG, 0.38)), layer(ACCENT)

    def draw(self, canvas: Image.Image, t: float) -> None:
        band, chart, track = self.band, self.chart, self.track
        px = chart.x(t)
        _paste_played(canvas, self.dim, self.bright, band, chart.x0 - 6, px)
        moment = moment_at(track.moments, t)
        color = {"dip": DIP, "peak": PEAK}.get(moment["kind"]) if moment else None
        value = track.score(t)
        chart.playhead(canvas, band.y, t)
        _paste_dot(canvas, color or ACCENT, px, band.y + chart.y(value))
        d = ImageDraw.Draw(canvas)
        d.text((self.NUM_RIGHT, band.y + 10), f"{value:.0f}", font=font(104, 700), fill=color or TEXT, anchor="ra")
        if moment:
            x = _badge(d, self.NUM_RIGHT + 20, band.y + 68, moment["kind"].upper(), color)
            text = _clip_text(headline(moment["reason"]), font(28, 600), W - MARGIN - 150 - x - 14)
            d.text((x + 14, band.y + 85), text, font=font(28, 600), fill=TEXT, anchor="lm")


class CompareStripPanel:
    """Two curves on one time axis, both current scores and a static before/after summary."""

    def __init__(self, band: Band, tracks: list[Track], duration: float):
        self.band, self.tracks = band, tracks
        self.chart = Chart(78, 170, W - MARGIN, 470, duration)
        chart = self.chart

        def layer(amount):
            def paint(d, s):
                chart.paint_grid(d, s)
                for track, color in zip(tracks, COMPARE_COLORS):
                    chart.paint_curve(d, s, track, _mix(color, BG, amount), area=False)

            img = _supersampled((W, band.h), paint)
            chart.draw_axes(img)
            self._draw_static(img)
            return img

        self.dim, self.bright = layer(0.4), layer(1.0)

    def _draw_static(self, img: Image.Image) -> None:
        d = ImageDraw.Draw(img)
        before, after = self.tracks
        for i, (track, color, role) in enumerate(zip(self.tracks, COMPARE_COLORS, ("before", "after"))):
            x = MARGIN + i * W // 2
            d.text((x + 170, 34), _clip_text(track.name, font(26, 600), 300), font=font(26, 600), fill=TEXT)
            d.text((x + 170, 70), role, font=font(22), fill=color)
        # legend under the chart
        x, y = MARGIN, self.chart.y1 + 58
        for track, color, role in zip(self.tracks, COMPARE_COLORS, ("before", "after")):
            d.rounded_rectangle((x, y + 4, x + 28, y + 12), radius=4, fill=color)
            label = f"{track.name} ({role})"
            d.text((x + 38, y + 8), label, font=font(22), fill=MUTED, anchor="lm")
            x += 38 + font(22).getlength(label) + 36
        # summary tiles
        y = self.chart.y1 + 110
        d.text((MARGIN, y), "summary · predicted engagement 0-100", font=font(22), fill=MUTED)
        tile_w, gap = (W - 2 * MARGIN - 3 * 16) // 4, 16
        for i, key in enumerate(SUMMARY_KEYS):
            x0, y0 = MARGIN + i * (tile_w + gap), y + 40
            d.rounded_rectangle((x0, y0, x0 + tile_w, y0 + 120), radius=12, fill=(20, 23, 29))
            d.text((x0 + 18, y0 + 14), key, font=font(22), fill=MUTED)
            a, b = before.summary[key], after.summary[key]
            delta = round(b) - round(a)
            d.text((x0 + 18, y0 + 46), f"{a:.0f} → {b:.0f}", font=font(34, 700), fill=TEXT)
            d.text(
                (x0 + 18, y0 + 88), f"{delta:+d}", font=font(24, 600),
                fill=PEAK if delta > 0 else DIP if delta < 0 else MUTED,
            )  # fmt: skip

    def draw(self, canvas: Image.Image, t: float) -> None:
        band, chart = self.band, self.chart
        px = chart.x(t)
        _paste_played(canvas, self.dim, self.bright, band, chart.x0 - 6, px)
        chart.playhead(canvas, band.y, t)
        d = ImageDraw.Draw(canvas)
        for i, (track, color) in enumerate(zip(self.tracks, COMPARE_COLORS)):
            value = track.score(t)
            # past its last TR a track's dot stays at the end of its curve
            _paste_dot(canvas, color, chart.x(min(t, track.centers[-1])), band.y + chart.y(value))
            d.text(
                (MARGIN + i * W // 2 + 150, band.y + 14), f"{value:.0f}",
                font=font(104, 700), fill=color, anchor="ra",
            )  # fmt: skip


class BrainPanel:
    """Cross-faded brain frames and five region bars."""

    BOX_X, BOX_Y = 20, 56
    BAR_X0, BAR_X1 = 740, W - MARGIN

    def __init__(self, band: Band, track: Track, frames: list[Image.Image], times, tr: float):
        self.band, self.track = band, track
        self.frames = [_fit_image(f.convert("RGB"), BRAIN_BOX) for f in frames]
        self.first, self.tr = float(times[0]), float(tr)
        self.ranges = {k: (float(v.min()), float(v.max())) for k, v in track.regions.items()}
        img = Image.new("RGB", (W, band.h), BG)
        d = ImageDraw.Draw(img)
        d.text((MARGIN, 18), "predicted cortical response", font=font(24, 600), fill=TEXT)
        self._legend(img, MARGIN, band.h - 36)
        d.text((self.BAR_X0, 18), "regions · vs. own range", font=font(24, 600), fill=TEXT)
        for i, name in enumerate(track.regions):
            y = self._row_y(i)
            d.text((self.BAR_X0, y), name.replace("_", " / "), font=font(22), fill=MUTED)
            d.rounded_rectangle((self.BAR_X0, y + 32, self.BAR_X1, y + 44), radius=6, fill=GRID)
        self.static = img
        self._cache: tuple | None = None

    @staticmethod
    def _row_y(i: int) -> int:
        return 66 + 58 * i

    @staticmethod
    def _legend(img: Image.Image, x: int, y: int) -> None:
        import matplotlib

        d = ImageDraw.Draw(img)
        f = font(22)
        d.text((x, y + 6), "lower", font=f, fill=MUTED, anchor="lm")
        x0 = x + f.getlength("lower") + 12
        cmap = matplotlib.colormaps[BRAIN_CMAP]
        width = 180
        for k in range(width):
            r, g, b, _ = cmap(k / (width - 1))
            d.line((x0 + k, y, x0 + k, y + 12), fill=(round(r * 255), round(g * 255), round(b * 255)))
        d.text((x0 + width + 12, y + 6), "higher than baseline", font=f, fill=MUTED, anchor="lm")

    def _brain(self, t: float) -> Image.Image:
        i, j, a = tr_blend(len(self.frames), self.first, self.tr, t)
        key = (i, j, round(a, 2))
        if self._cache is None or self._cache[0] != key:
            img = self.frames[i] if i == j or key[2] == 0 else Image.blend(self.frames[i], self.frames[j], key[2])
            self._cache = (key, img)
        return self._cache[1]

    def draw(self, canvas: Image.Image, t: float) -> None:
        band = self.band
        canvas.paste(self.static, (0, band.y))
        brain = self._brain(t)
        canvas.paste(
            brain,
            (self.BOX_X + (BRAIN_BOX[0] - brain.width) // 2, band.y + self.BOX_Y + (BRAIN_BOX[1] - brain.height) // 2),
        )
        d = ImageDraw.Draw(canvas)
        span = self.BAR_X1 - self.BAR_X0
        for i, (name, curve) in enumerate(self.track.regions.items()):
            lo, hi = self.ranges[name]
            frac = (value_at(self.track.centers, curve, t) - lo) / (hi - lo) if hi > lo else 0.5
            y = band.y + self._row_y(i)
            x1 = self.BAR_X0 + max(12, round(span * min(max(frac, 0.0), 1.0)))
            d.rounded_rectangle((self.BAR_X0, y + 32, x1, y + 44), radius=6, fill=ACCENT)


def _footer(watermark: bool) -> Image.Image:
    img = Image.new("RGB", (W, FOOTER_H), BG)
    d = ImageDraw.Draw(img)
    d.line((0, 0, W, 0), fill=GRID, width=1)
    f = font(22)
    right = W - MARGIN
    if watermark:
        wm = font(32, 700)
        d.text((right, FOOTER_H / 2), WATERMARK, font=wm, fill=TEXT, anchor="rm")
        right -= wm.getlength(WATERMARK) + 32
    if f.getlength(FOOTER_TEXT) <= right - MARGIN:
        d.text((MARGIN, FOOTER_H / 2), FOOTER_TEXT, font=f, fill=MUTED, anchor="lm")
    else:
        first, second = FOOTER_TEXT.rsplit(" · ", 1)
        d.text((MARGIN, FOOTER_H / 2 - 2), first, font=f, fill=MUTED, anchor="ld")
        d.text((MARGIN, FOOTER_H / 2 + 4), second, font=f, fill=MUTED, anchor="la")
    return img


# --------------------------------------------------------------------- brain frames


def _content_box(img: Image.Image, pad: int = 6) -> tuple[int, int, int, int]:
    arr = np.asarray(img, dtype=np.int16)
    mask = np.abs(arr - np.array(BG, dtype=np.int16)).sum(axis=-1) > 24
    if not mask.any():
        return (0, 0, img.width, img.height)
    ys, xs = np.nonzero(mask)
    return (
        max(int(xs.min()) - pad, 0),
        max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + pad + 1, img.width),
        min(int(ys.max()) + pad + 1, img.height),
    )


def render_brain_images(preds: np.ndarray, vmax: float):
    """Yield one image per TR: left + right lateral inflated fsaverage5 views."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nilearn import datasets, plotting

    mesh = datasets.fetch_surf_fsaverage("fsaverage5")
    n = preds.shape[1] // 2
    bg = "#%02x%02x%02x" % BG
    crop = None
    for frame in preds:
        fig = plt.figure(figsize=(10, 4), dpi=100, facecolor=bg)
        for i, (hemi, values) in enumerate((("left", frame[:n]), ("right", frame[n:]))):
            ax = fig.add_axes((0.5 * i, 0.0, 0.5, 1.0), projection="3d")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotting.plot_surf_stat_map(
                    surf_mesh=mesh[f"infl_{hemi}"],
                    stat_map=np.asarray(values, dtype=np.float64),
                    bg_map=mesh[f"sulc_{hemi}"],
                    hemi=hemi,
                    view="lateral",
                    cmap=BRAIN_CMAP,
                    vmin=-vmax,
                    vmax=vmax,
                    colorbar=False,
                    axes=ax,
                    figure=fig,
                )
            ax.set_facecolor(bg)
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
        crop = crop or _content_box(img)  # same mesh and view every TR: one crop for all
        yield _fit_image(img.crop(crop), BRAIN_BOX)


def brain_frames(preds: np.ndarray, cache_dir: Path) -> list[Image.Image]:
    """One brain image per TR, cached as PNGs in ``cache_dir``.

    The colour scale is symmetric around 0 and shared by the whole video
    (99th percentile of ``|preds|``).
    """
    preds = np.ascontiguousarray(preds, dtype=np.float32)
    vmax = float(np.percentile(np.abs(preds), 99)) or 1.0
    key = hashlib.sha256(
        preds.tobytes() + f"{BRAIN_RENDER_VERSION}:{BRAIN_BOX}:{BRAIN_CMAP}".encode()
    ).hexdigest()
    paths = [cache_dir / f"tr{i:04d}.png" for i in range(len(preds))]
    meta_path = cache_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}
    if meta.get("key") != key or not all(p.is_file() for p in paths):
        prepare_file(meta_path)
        for path, img in zip(paths, render_brain_images(preds, vmax)):
            img.save(prepare_file(path))
        write_text(meta_path, json.dumps({"key": key, "n_tr": len(preds), "vmax": vmax}))
    frames = []
    for path in paths:
        with Image.open(path) as img:
            frames.append(img.convert("RGB"))
    return frames


# --------------------------------------------------------------------- render


@dataclass
class OverlayResult:
    output: Path
    frames: int
    duration: float
    seconds_brain: float
    seconds_total: float


def _ffmpeg_cmd(tracks: list[Track], placements, fps: int, out: Path) -> list[str]:
    from .media import ffmpeg_exe, input_args

    chains, last = [], "0:v"
    for k, (w, h, x, y) in enumerate(placements, 1):
        chains.append(
            f"[{k}:v]setpts=PTS-STARTPTS,fps={fps},scale={w}:{h}:flags=lanczos,setsar=1,"
            f"tpad=stop_mode=clone:stop=-1[s{k}]"  # the shorter video freezes on its last frame
        )
        chains.append(f"[{last}][s{k}]overlay=x={x}:y={y}:shortest=1[o{k}]")
        last = f"o{k}"
    # Limited-range BT.709 yuv420p plays the same everywhere, even from full-range sources.
    chains.append(f"[{last}]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[out]")
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), *input_args("pipe:0", formats="rawvideo"),
    ]  # fmt: skip
    for track in tracks:
        cmd += input_args(track.video)
    cmd += ["-filter_complex", ";".join(chains), "-map", "[out]"]
    codec = tracks[0].audio_codec
    if codec:
        cmd += ["-map", "1:a:0", "-c:a"] + (["copy"] if codec == "aac" else ["aac", "-b:a", "192k"])
    cmd += [
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-r", str(fps),
        "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart", str(out),
    ]  # fmt: skip
    return cmd


def render(
    video: str | Path,
    out: str | Path | None = None,
    other: str | Path | None = None,
    fps: int = FPS,
    watermark: bool = True,
) -> OverlayResult:
    """Render the overlay of ``video`` (or ``video`` vs ``other``) to an MP4."""
    started = time.perf_counter()
    if not 1 <= fps <= 60:
        raise OverlayError(f"--fps must be between 1 and 60, got {fps}")
    tracks = [load_track(video)] + ([load_track(other)] if other is not None else [])
    compare = len(tracks) == 2
    first = tracks[0]
    suffix = "compare" if compare else "overlay"
    out = Path(out).resolve() if out else first.out_dir / f"{first.name}-{suffix}.mp4"
    if any(same_file(out, t.video) for t in tracks):
        raise OverlayError("refusing to overwrite a source video; pick another --out")
    prepare_file(out)

    duration = max(t.duration for t in tracks)
    bands = layout(compare)
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    seconds_brain = 0.0
    if compare:
        col = W // 2
        placements = []
        for i, (track, color, role) in enumerate(zip(tracks, COMPARE_COLORS, ("before", "after"))):
            w, h, x, y = fit(*track.size, col, VIDEO_H)
            placements.append((w, h, x + i * col, y))
            if y >= 52:  # label above the column when there is room
                d.rounded_rectangle((i * col + 24, 20, i * col + 44, 32), radius=4, fill=color)
                label = _clip_text(f"{role} · {track.name}", font(22, 600), col - 80)
                d.text((i * col + 54, 26), label, font=font(22, 600), fill=TEXT, anchor="lm")
        panels = [CompareStripPanel(bands["strip"], tracks, duration)]
    else:
        placements = [fit(*first.size, W, VIDEO_H)]
        preds_path = first.out_dir / "preds.npz"
        if not preds_path.is_file():
            raise OverlayError(f"no predictions for {first.video.name}. Run first: cortexreel analyze {video}")
        with np.load(preds_path) as data:
            preds, times, tr = data["preds"], data["times"], float(data["tr_seconds"])
        t0 = time.perf_counter()
        frames = brain_frames(preds, first.out_dir / "brain_frames")
        seconds_brain = time.perf_counter() - t0
        panels = [
            StripPanel(bands["strip"], first, duration),
            BrainPanel(bands["brain"], first, frames, times, tr),
        ]
    canvas.paste(_footer(watermark), (0, bands["footer"].y))

    n_frames = max(1, round(duration * fps))
    cmd = _ffmpeg_cmd(tracks, placements, fps, out)
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors)
        try:
            for k in range(n_frames):
                t = k / fps
                for panel in panels:
                    panel.draw(canvas, t)
                proc.stdin.write(canvas.tobytes())
        except OSError:
            pass  # ffmpeg died (broken pipe); its exit code and stderr say why
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass
            code = proc.wait()
        if code != 0:
            errors.seek(0)
            raise RuntimeError(f"ffmpeg failed ({code}): {errors.read().decode(errors='replace')[-2000:]}")
    return OverlayResult(out, n_frames, n_frames / fps, seconds_brain, time.perf_counter() - started)
