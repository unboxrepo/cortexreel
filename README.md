# cortexreel

> Work in progress (milestone M5).

cortexreel runs Meta's [TRIBE v2](https://huggingface.co/facebook/tribev2) brain-encoding
model on a video to predict how an average viewer's cortex responds second by second,
turns that into an engagement curve (0-100), points out weak moments, and can cut or
speed them up for you.

**What it is not:** it is not a virality predictor and it does not measure a real
audience. It predicts the average response of a model brain. Treat the scores as an
editing hint, then trust your own eyes.

## Install

Requires an NVIDIA GPU (8 GB VRAM target), Python 3.11 (uv installs it) and
[uv](https://docs.astral.sh/uv/). ffmpeg ships with the package.

**Recommended: a locked install from a release tag.** `uv sync --locked` installs
exactly what `uv.lock` pins (hashes included): the CUDA 12.4 build of torch, Pillow 13
and every other dependency at a reviewed version.

```sh
git clone --branch v0.1.1 https://github.com/unboxrepo/cortexreel
cd cortexreel
uv sync --locked
uv run cortexreel doctor     # checks GPU, ffmpeg, Hugging Face access; prints next steps
uv run cortexreel setup      # downloads pinned models (~20-30 GB)
uv run cortexreel analyze path/to/video.mp4
```

`uv run` works from inside the clone. To call `cortexreel` from any folder (the Claude Code
plugin needs this), put the clone's `.venv/bin` (`.venv\Scripts` on Windows) on your `PATH`.

**Alternative: `uv tool install` from the tag.** This resolves the dependencies fresh
instead of using `uv.lock`, so only the direct pins in `pyproject.toml` hold. It does use
the project's CUDA torch index, but not its `[tool.uv]` override for Pillow: without
`--overrides` the install fails, because moviepy (via tribev2) caps Pillow below 12.

```sh
echo "pillow>=13" > overrides.txt
uv tool install --overrides overrides.txt git+https://github.com/unboxrepo/cortexreel@v0.1.1
cortexreel doctor
cortexreel setup
```

The text channel uses `meta-llama/Llama-3.2-3B`, a gated model: request access on its
Hugging Face page and log in with `hf auth login` (or set `HF_TOKEN`). Until access is
granted, pass `--no-text` to run on video and audio only.

Only `setup` (and the access check in `doctor`) connects to Hugging Face. Every other
command runs offline from the model cache and asks you to run `cortexreel setup` if a
pinned model is missing.

## Commands

| Command | What it does |
|---|---|
| `cortexreel doctor` | Checks GPU/VRAM, ffmpeg, Hugging Face token and Llama access, disk space. |
| `cortexreel setup [--no-text]` | Downloads every model at a pinned revision. |
| `cortexreel analyze VIDEO [--json]` | Engagement curve, hook/hold/ending scores, dips and peaks. Writes `out/<name>/analysis.json`, `words.tsv` and an offline `report.html`. |
| `cortexreel recut VIDEO [--plan plan.json] [--dry-run] [--out PATH] [--json]` | Applies an edit plan (cuts and speed-ups). Without `--plan`, suggests one from the analysis and saves it to `out/<name>/plan.json` so you can edit it and rerun. Output: `out/<name>/<name>-recut.mp4`. |
| `cortexreel diff A/analysis.json B/analysis.json [--json]` | Before/after table: overall, hook, hold, ending, duration, dips. |
| `cortexreel overlay VIDEO [--compare OTHER] [--no-watermark] [--out PATH] [--fps 30]` | 1080x1920 showcase video: the source on top, the engagement curve with a moving playhead, the predicted brain response and region bars below. `--compare` plays two cuts side by side with both curves and a before/after summary. Output: `out/<name>/<name>-overlay.mp4` (`-compare.mp4`). |
| `cortexreel compare V1 V2 [V3 V4] [--block 3] [--json]` | Scores 2-4 versions block by block (hook 0-3 s, 3 s blocks, each version's own last 3 s) and names each block's winner and the overall leader. Output: `out/compare-<a>-vs-<b>/compare.json` and `compare.html`. |
| `cortexreel splice --compare COMPARE_JSON [--plan splice.json] [--dry-run] [--json]` | Renders one video from the block winners (or from your plan): each switch snaps to a word gap, every piece is scaled to the first one's size. Output: `splice-<a>-vs-<b>.mp4` and `splice.json` next to `compare.json`. |
| `cortexreel predict VIDEO` | Raw TRIBE predictions only (`preds.npz`). |

`--json` prints machine-readable JSON on stdout only; logs go to stderr. Put `--` before
video paths that might start with `-` (`cortexreel analyze --json -- -take2.mp4`); every
command accepts it.

The automatic plan is conservative: it never cuts through a spoken word (boundaries snap
to gaps between words, preferring sentence ends), never touches the first 3 s or a peak,
cuts silent dips and the silent tail after the last word, and speeds up dips with speech
(x1.15-1.25, pitch preserved). A weak hook only produces a written suggestion. Plans
removing more than half the video are refused unless you pass `--allow-heavy`. The
source video is never overwritten.

### Compare and splice

`compare` puts the versions on one absolute timeline, so it works best for cuts of the
same script (an original and its recut, two takes). The automatic splice takes each block
from its winner, merges runs from the same source and moves every switch into a gap
between words; if snapping would leave a piece under 0.8 s or repeat/skip more than 0.4 s
of the shared timeline, it keeps the previous source instead. It assumes the versions share
structure, so review each boundary (the plan is saved as `splice-plan.json` for editing),
then `analyze` the splice and `diff` it against the leader.

## Claude Code plugin

```text
/plugin marketplace add unboxrepo/cortexreel
/plugin install cortexreel@cortexreel
```

- `/cortexreel:analyze <video>` explains the scores and weakest moments in plain language.
- `/cortexreel:improve <video>` runs the loop: analyze, write a plan, recut, re-analyze,
  `diff`; keeps the new cut only if overall and hold did not drop and the ending or dips
  improved, otherwise reverts the worst op and retries (max 3 rounds).
- `/cortexreel:compare <v1> <v2> [v3 v4]` names the leader and the winner of each part,
  then offers to splice the strongest parts, re-analyzes the splice and compares it with
  the leader.

The plugin calls the `cortexreel` command on your PATH, so install it first. Its skills
pre-approve only `cortexreel ...` commands, reading files and writing the edit-plan files
under `out/`; anything else asks you first. Transcripts, plan notes and file names come
from the video, and the skills treat them as data, never as instructions.

## Privacy

- Command line only: no server, no open ports, no telemetry (`HF_HUB_DISABLE_TELEMETRY`,
  `DO_NOT_TRACK`, `WANDB_MODE=disabled` are set on start).
- Speech is transcribed locally with faster-whisper; your video never leaves the machine.
- Models are downloaded at pinned revisions by `cortexreel setup`, the only command that
  downloads them. Everything else runs offline and refuses to load a model that is not
  pinned. `--offline` also keeps `doctor` and `setup` off the network.
- TRIBE's config is parsed with a restricted YAML loader; weights load with `weights_only=True`.
- ffmpeg only opens local files with real video demuxers (`-protocol_whitelist file,pipe`,
  `-format_whitelist`), so a playlist or concat list disguised as a video can't pull in other files.
- Outputs are written only under `./out/` (except an explicit `recut --out` or `overlay --out`).
  A symlinked or junctioned `out/` or `out/<name>/` is refused, and an existing output file
  is replaced rather than written through.
- JSON outputs store paths relative to the current folder, and `analysis.json` leaves out
  the GPU name (only the local `metrics.json` cache keeps it).
- The HTML reports carry a Content-Security-Policy: no network requests, only their own
  inline script.

## Performance and accuracy notes

Measured on an RTX 3070 Ti (8 GB):

- First analysis of a 40 s reel: about 2.5-3 minutes. Re-analyzing the same file reuses
  cached predictions and takes seconds. A recut renders in a few seconds; an overlay of a
  40 s reel in about 25 s (brain frames are cached in `out/<name>/brain_frames/`).
- The video extractor (V-JEPA2) runs in fp16 to fit in 8 GB; fp16 and fp32 scores agree
  within 0.1 points. Analysis uses a 320 px proxy of the video (the model downsizes to
  292 px anyway).
- Onset correction: TRIBE scores the first ~12 s of any video 6-14 points lower than the
  same footage placed mid-video (a model warm-up effect, not content). This gap was
  measured on 5 reels and is added back; without it every hook would be flagged weak.
  The uncorrected curve stays in `analysis.json`. It was calibrated on short screen-recording
  reels with voice-over; other genres may differ.

## License

cortexreel's code is MIT licensed. The TRIBE v2 model is CC BY-NC 4.0: you download it
yourself from Hugging Face, and anything produced with it (predictions, scores, reports)
is for non-commercial use only. See [NOTICE.md](NOTICE.md).
