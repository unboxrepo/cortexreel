"""Named cortical region groups on fsaverage5, built from the Destrieux atlas.

Group definitions and engagement weights live in ``data/regions.json``. The
atlas itself is downloaded once by ``cortexreel setup`` into
``<cache root>/nilearn_data`` (never nilearn's default under the home folder)
and read from there afterwards, so it works offline.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import env

CONFIG_PATH = Path(__file__).parent / "data" / "regions.json"
N_PER_HEMI = 10242  # fsaverage5 vertices per hemisphere
N_VERTICES = 2 * N_PER_HEMI
DESTRIEUX_DIR = "destrieux_surface"  # nilearn's dataset folder name
ANNOT_FILES = ("left.aparc.a2009s.annot", "right.aparc.a2009s.annot")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def weights(config: dict | None = None) -> dict[str, float]:
    config = config or load_config()
    return {name: float(g["weight"]) for name, g in config["groups"].items()}


def atlas_dir() -> Path:
    return env.cache_root() / "nilearn_data"


def atlas_cached() -> bool:
    folder = atlas_dir() / DESTRIEUX_DIR
    return all((folder / name).is_file() for name in ANNOT_FILES)


def fetch_atlas():
    """Destrieux atlas bunch (``labels``, ``map_left``, ``map_right``).

    Downloads into our cache root on first use; later calls only read the
    cached ``.annot`` files.
    """
    from nilearn.datasets import fetch_atlas_surf_destrieux

    if not atlas_cached() and env.is_offline():
        raise env.SetupNeeded(
            f"Destrieux atlas not found in {atlas_dir()}; run `cortexreel setup` once online."
        )
    return fetch_atlas_surf_destrieux(data_dir=str(atlas_dir()), verbose=0)


@lru_cache(maxsize=1)
def _atlas_maps() -> tuple[tuple[str, ...], np.ndarray]:
    atlas = fetch_atlas()
    labels = tuple(str(name) for name in atlas.labels)
    # Left hemisphere first, matching TRIBE's vertex order.
    vertex_label = np.concatenate([atlas.map_left, atlas.map_right]).astype(np.int64)
    if vertex_label.shape != (N_VERTICES,):
        raise RuntimeError(f"Unexpected atlas size {vertex_label.shape}, want ({N_VERTICES},)")
    return labels, vertex_label


def region_masks(config: dict | None = None) -> dict[str, np.ndarray]:
    """``{group: bool[20484]}`` for every group in ``regions.json``."""
    config = config or load_config()
    labels, vertex_label = _atlas_maps()
    index = {name: i for i, name in enumerate(labels)}
    masks = {}
    for group, spec in config["groups"].items():
        missing = [name for name in spec["labels"] if name not in index]
        if missing:
            raise ValueError(f"regions.json group {group!r}: unknown Destrieux labels {missing}")
        ids = [index[name] for name in spec["labels"]]
        masks[group] = np.isin(vertex_label, ids)
    return masks
