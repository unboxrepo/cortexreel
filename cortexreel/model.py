"""Loading TRIBE v2 safely and at pinned revisions.

Three things differ from calling ``TribeModel.from_pretrained("facebook/tribev2")``:

1. ``config.yaml`` and ``best.ckpt`` are downloaded at a pinned commit.
2. tribev2 parses ``config.yaml`` with ``yaml.UnsafeLoader``. We swap the ``yaml``
   name inside ``tribev2.demo_utils`` for a shim backed by
   :class:`cortexreel.safe_yaml.TribeConfigLoader` while it loads.
   (The checkpoint itself is read by tribev2 with ``torch.load(weights_only=True)``.)
3. The feature extractors (V-JEPA2, DINOv2, w2v-BERT, Llama) are loaded by
   neuralset through ``transformers`` ``from_pretrained(model_name)`` with no
   revision. In transformers 5.x every public ``from_pretrained`` entry point
   first calls ``transformers.utils.hub.resolve_revision``, which goes through
   one shared ``HfApi`` instance; we override ``resolve_revision`` on that
   instance so our repos resolve to their pinned commit and everything after
   (including offline cache lookups) uses that commit.
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from pathlib import Path

from . import env, pins

logger = logging.getLogger(__name__)

# Only weights in safetensors format plus configs/tokenizers; skips duplicate
# ``original/*.pth`` and ``pytorch_model.bin`` copies.
EXTRACTOR_FILES = ["*.json", "*.safetensors", "LICENSE*", "USE_POLICY.md"]
TRIBE_FILES = ["config.yaml", "best.ckpt"]


def extended_path(path: str | Path) -> str:
    r"""Windows extended-length form: ``C:\x`` -> ``\\?\C:\x``, ``\\server\share`` -> ``\\?\UNC\server\share``."""
    s = str(path)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def feature_cache_dir(precision: str = "fp32") -> Path:
    # Extractor cache keys ignore dtype, so each precision gets its own folder.
    folder = env.cache_root() / ("features" if precision == "fp32" else f"features-{precision}")
    if sys.platform == "win32":
        # neuralset/exca build cache folder names from the full extractor config,
        # which exceeds MAX_PATH (260). The extended-length prefix lifts that limit
        # without touching the system-wide LongPathsEnabled setting.
        folder = Path(extended_path(folder.resolve()))
    return folder


def _patterns(pin: pins.Pin):
    if pin is pins.TRIBE:
        return TRIBE_FILES
    if pin is pins.WHISPER:
        return None
    return EXTRACTOR_FILES


def _missing(pin: pins.Pin) -> env.SetupNeeded:
    return env.SetupNeeded(
        f"{pin.repo_id}@{pin.revision[:12]} is not downloaded; run `cortexreel setup`"
        + (" (or pass --no-text)" if pin.text_only else "")
    )


def download_models(use_text: bool = True):
    """Download every model cortexreel needs at its pinned revision.

    The only place cortexreel goes online for models. Yields ``(pin, local_path)``
    as each finishes.
    """
    from huggingface_hub import snapshot_download

    for pin in pins.needed(use_text):
        yield pin, snapshot_download(pin.repo_id, revision=pin.revision, allow_patterns=_patterns(pin))


def local_snapshot(pin: pins.Pin) -> str:
    """The cached snapshot of ``pin``, never downloading; SetupNeeded if it is missing."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return snapshot_download(
            pin.repo_id, revision=pin.revision, allow_patterns=_patterns(pin), local_files_only=True
        )
    except LocalEntryNotFoundError:
        raise _missing(pin) from None


def require_models(use_text: bool = True) -> None:
    """Fail early, with a clear message, if any pinned model is not in the cache."""
    for pin in pins.needed(use_text):
        local_snapshot(pin)


def tribe_checkpoint_dir() -> Path:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    pin = pins.TRIBE
    try:
        config, ckpt = [
            hf_hub_download(pin.repo_id, name, revision=pin.revision, local_files_only=True)
            for name in TRIBE_FILES
        ]
    except LocalEntryNotFoundError:
        raise _missing(pin) from None
    folder = Path(config).parent
    if Path(ckpt).parent != folder:  # both live in snapshots/<revision>/
        raise RuntimeError(f"Unexpected Hugging Face cache layout: {config}, {ckpt}")
    return folder


@contextmanager
def safe_tribe_yaml():
    """Make ``tribev2.demo_utils`` parse YAML with our restricted loader."""
    import tribev2.demo_utils as demo_utils

    from .safe_yaml import yaml_shim

    original = demo_utils.yaml
    demo_utils.yaml = yaml_shim
    try:
        yield
    finally:
        demo_utils.yaml = original


class UnpinnedModel(BaseException):
    """A model outside ``pins`` was requested.

    A BaseException on purpose: transformers calls the revision hook inside
    ``except Exception`` and falls back to the default branch, which would make
    an ordinary error fail open.
    """


def pin_transformers_revisions() -> None:
    """Force transformers to load our extractor repos at their pinned commits.

    Idempotent. Fails closed: a repo that is not pinned, or a pinned repo asked
    for at another revision, raises instead of loading something unreviewed.
    """
    import transformers.utils.hub as tf_hub

    if not hasattr(tf_hub, "hf_api") or not hasattr(tf_hub, "resolve_revision"):
        raise RuntimeError(
            "This transformers version has no resolve_revision hook; cannot pin "
            "feature-extractor revisions. Use the versions in uv.lock."
        )
    api = tf_hub.hf_api()
    if getattr(api, "_cortexreel_pinned", False):
        return
    def resolve_revision(repo_id, *args, revision=None, **kwargs):
        pin = pins.BY_REPO.get(str(repo_id))
        if pin is None:
            raise UnpinnedModel(f"Refusing to load {repo_id}: not one of cortexreel's pinned models")
        if revision not in (None, "main", pin.revision):
            raise UnpinnedModel(f"Refusing to load {repo_id}@{revision}: pinned to {pin.revision}")
        return pin.revision

    api.resolve_revision = resolve_revision
    api._cortexreel_pinned = True


def half_precision_video() -> None:
    """Run the V-JEPA2 video extractor in float16.

    In float32 the 1B-parameter model plus its 41 cached hidden-state layers
    overflow 8 GB cards, and Windows silently spills into shared system memory,
    which makes encoding many times slower. Idempotent.
    """
    import torch
    from neuralset.extractors import video as ns_video

    cls = ns_video._HFVideoModel
    if getattr(cls, "_cortexreel_half", False):
        return
    original_init, original_predict = cls.__init__, cls.predict

    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.model.half()

    def predict(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.float16):
            return original_predict(self, *args, **kwargs)

    cls.__init__, cls.predict = __init__, predict
    cls._cortexreel_half = True


def half_precision_text() -> None:
    """Load the Llama text extractor in float16 (fp32 would need ~13 GB). Idempotent."""
    import torch
    from neuralset.extractors import text as ns_text

    cls = ns_text.HuggingFaceText
    if getattr(cls, "_cortexreel_half", False):
        return
    original = cls._load_model

    def _load_model(self, **kwargs):
        kwargs.setdefault("dtype", torch.float16)
        return original(self, **kwargs)

    cls._load_model = _load_model
    cls._cortexreel_half = True


def load_tribe(
    device: str = "auto",
    use_text: bool = True,
    cache_folder: str | Path | None = None,
    precision: str = "fp16",
):
    """Load the pretrained TRIBE v2 model.

    ``use_text=False`` drops the Llama text channel: only the audio and video
    extractors are prepared, so Llama is never downloaded or loaded. The brain
    model itself is unchanged (its architecture comes from the checkpoint); its
    text projector input is filled with zeros, the same state it saw during
    training via modality dropout and on clips without speech.
    """
    from tribev2.demo_utils import TribeModel

    require_models(use_text)
    pin_transformers_revisions()
    if precision == "fp16":
        half_precision_video()
        half_precision_text()
    elif precision != "fp32":
        raise ValueError(f"precision must be fp16 or fp32, got {precision!r}")
    cache_folder = Path(cache_folder) if cache_folder else feature_cache_dir(precision)
    config_update: dict = {
        # Windows-safe: no worker processes re-importing torch for the loader.
        "data.num_workers": 0,
    }
    if not use_text:
        config_update["data.features_to_use"] = ["audio", "video"]

    checkpoint_dir = tribe_checkpoint_dir()
    with safe_tribe_yaml():
        model = TribeModel.from_pretrained(
            checkpoint_dir,
            cache_folder=cache_folder,
            device=device,
            config_update=config_update,
        )
    logger.info("Loaded TRIBE v2 @ %s (use_text=%s)", pins.TRIBE.revision[:12], use_text)
    return model
