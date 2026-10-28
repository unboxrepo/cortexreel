"""Hugging Face model ids and the exact revisions cortexreel downloads."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pin:
    repo_id: str
    revision: str
    gated: bool = False
    text_only: bool = False  # only needed for the Llama text channel


TRIBE = Pin("facebook/tribev2", "f894e783020944dcd96e5568550afe2aa9743f9f")
VJEPA2 = Pin("facebook/vjepa2-vitg-fpc64-256", "875c192b7b704b87d1e1d99345769632dd5f739a")
DINOV2 = Pin("facebook/dinov2-large", "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c")
W2V_BERT = Pin("facebook/w2v-bert-2.0", "da985ba0987f70aaeb84a80f2851cfac8c697a7b")
LLAMA = Pin(
    "meta-llama/Llama-3.2-3B",
    "13afe5124825b4f3751f836b40dafda64c1ed062",
    gated=True,
    text_only=True,
)
WHISPER = Pin(
    "Systran/faster-whisper-large-v3",
    "edaa852ec7e145841d8ffdb056a99866b5f0a478",
    text_only=True,
)

# Feature extractors referenced by model_name in TRIBE's config.yaml.
EXTRACTORS = (VJEPA2, DINOV2, W2V_BERT, LLAMA)
ALL = (TRIBE, *EXTRACTORS, WHISPER)
BY_REPO = {pin.repo_id: pin for pin in ALL}


def needed(use_text: bool = True) -> tuple[Pin, ...]:
    return tuple(p for p in ALL if use_text or not p.text_only)
