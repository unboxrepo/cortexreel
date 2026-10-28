"""Process environment: telemetry off, cache location, offline mode.

``configure()`` must run before torch, transformers or huggingface_hub are
imported, because those libraries read these variables at import time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PRIVACY_DEFAULTS = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "WANDB_MODE": "disabled",
    "TOKENIZERS_PARALLELISM": "false",
}

OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
# Only these commands may reach Hugging Face; every other one runs offline from the cache.
NETWORK_COMMANDS = ("setup", "doctor")


class SetupNeeded(RuntimeError):
    """A pinned model or the atlas is missing from the cache: run ``cortexreel setup``."""


def cache_root() -> Path:
    """Root folder for downloaded models and feature caches."""
    custom = os.environ.get("CORTEXREEL_HOME")
    if custom:
        return Path(custom).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "cortexreel"
    return Path.home() / ".cache" / "cortexreel"


def configure(offline: bool = False) -> Path:
    """Set privacy, cache and offline variables. Returns the cache root.

    Existing values are kept (``setdefault``) so users can override them,
    except that ``offline=True`` always forces offline mode on.
    """
    for key, value in PRIVACY_DEFAULTS.items():
        os.environ.setdefault(key, value)
    root = cache_root()
    os.environ.setdefault("HF_HOME", str(root / "huggingface"))
    # A token saved by a plain `hf auth login` lives under the default HF home;
    # reuse it when our own HF_HOME has none.
    own_token = Path(os.environ["HF_HOME"]) / "token"
    default_token = Path.home() / ".cache" / "huggingface" / "token"
    if not own_token.exists() and default_token.exists():
        os.environ.setdefault("HF_TOKEN_PATH", str(default_token))
    if offline:
        for key in OFFLINE_VARS:
            os.environ[key] = "1"
    return root


def is_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes", "on")
