import json

import pytest

from cortexreel import regions


def _no_duplicate_keys(pairs):
    keys = [k for k, _ in pairs]
    assert len(keys) == len(set(keys)), f"duplicate keys: {keys}"
    return dict(pairs)


def test_regions_json_schema():
    config = json.loads(
        regions.CONFIG_PATH.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys
    )
    groups = config["groups"]
    assert set(groups) == {"visual", "auditory", "language", "attention", "face_body"}
    weights = regions.weights(config)
    assert sum(w for w in weights.values() if w) == pytest.approx(1.0)
    assert all(w >= 0 for w in weights.values())
    for name, group in groups.items():
        assert group["labels"], name
        assert len(group["labels"]) == len(set(group["labels"])), name
    mapping = config["engagement"]
    assert mapping["scale"] > 0 and "center" in mapping


@pytest.mark.skipif(not regions.atlas_cached(), reason="Destrieux atlas not cached")
def test_labels_exist_and_groups_are_large_enough():
    labels, _ = regions._atlas_maps()
    for name, group in regions.load_config()["groups"].items():
        missing = set(group["labels"]) - set(labels)
        assert not missing, (name, missing)
    masks = regions.region_masks()
    for name, mask in masks.items():
        assert mask.shape == (regions.N_VERTICES,)
        assert mask.sum() >= 200, (name, int(mask.sum()))
        # Both hemispheres are covered.
        assert mask[: regions.N_PER_HEMI].any() and mask[regions.N_PER_HEMI :].any()
