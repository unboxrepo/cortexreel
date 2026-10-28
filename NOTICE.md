# Notices and third-party attribution

cortexreel's own source code is released under the MIT License (see `LICENSE`).
It depends on, and downloads at runtime, third-party models and code that are
under their own licenses. **No model weights are included in or redistributed
by this repository**; each user downloads them directly from Hugging Face.

cortexreel is an independent project. It is **not affiliated with, endorsed by,
or sponsored by Meta Platforms, Inc.**

## TRIBE v2 (Meta)

- Model: https://huggingface.co/facebook/tribev2
- Code: https://github.com/facebookresearch/tribev2
- Paper: https://arxiv.org/abs/2605.04326
- License: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)

TRIBE v2 was created by Meta FAIR. Its weights and code may be used for
non-commercial purposes only, with attribution. cortexreel imports the `tribev2`
Python package as a dependency (pinned to a specific commit) and calls its public
API; it does not copy TRIBE v2 source code into this repository. Because the model
is non-commercial, outputs produced with cortexreel inherit that restriction in
practice: do not use cortexreel with TRIBE v2 for commercial purposes.

## Feature-extractor models used by TRIBE v2

| Model | License |
|---|---|
| `meta-llama/Llama-3.2-3B` | Llama 3.2 Community License (gated; accept on Hugging Face) |
| `facebook/vjepa2-vitg-fpc64-256` | see its model card |
| `facebook/dinov2-large` | see its model card |
| `facebook/w2v-bert-2.0` | see its model card |

**Llama 3.2:** "Llama 3.2 is licensed under the Llama 3.2 Community License,
Copyright © Meta Platforms, Inc. All Rights Reserved." Use of Llama 3.2 is subject
to that license and Meta's Acceptable Use Policy, which each user accepts when
requesting access on Hugging Face. cortexreel can run without Llama (`--no-text`).

## Speech recognition

Local transcription uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
(MIT) with the `Systran/faster-whisper-large-v3` conversion of OpenAI Whisper
large-v3 (MIT).

## Bundled font

`cortexreel/data/fonts/InterVariable.ttf` is Inter 4.1 by The Inter Project Authors
(https://github.com/rsms/inter), taken unmodified from the official `Inter-4.1.zip`
release. It is licensed under the SIL Open Font License 1.1; the full license text
is in `cortexreel/data/fonts/OFL.txt`. The overlay video uses it for all text.
