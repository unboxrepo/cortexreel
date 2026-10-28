from types import SimpleNamespace as NS

from cortexreel import pins
from cortexreel.transcribe import COLUMNS, segments_to_words


def test_segments_to_words_matches_tribe_columns():
    segments = [
        NS(
            text=' Hello "world".',
            words=[
                NS(word=" Hello", start=0.0, end=0.4),
                NS(word=' "world".', start=0.5, end=0.9),
                NS(word=" ", start=0.9, end=1.0),
                NS(word=" lost", start=None, end=None),
            ],
        ),
        NS(text=" Again", words=[NS(word=" Again", start=2.0, end=2.5)]),
    ]
    df = segments_to_words(segments)
    assert list(df.columns) == COLUMNS == ["text", "start", "duration", "sequence_id", "sentence"]
    assert df.text.tolist() == ["Hello", "world.", "Again"]
    assert df.sequence_id.tolist() == [0, 0, 1]
    assert df.sentence.tolist() == ["Hello world.", "Hello world.", "Again"]
    assert abs(df.duration.iloc[1] - 0.4) < 1e-9


def test_segments_to_words_empty():
    df = segments_to_words([])
    assert list(df.columns) == COLUMNS and len(df) == 0


def test_pins_are_full_shas():
    for pin in pins.ALL:
        assert len(pin.revision) == 40 and int(pin.revision, 16) >= 0
    assert pins.LLAMA not in pins.needed(use_text=False)
    assert pins.WHISPER not in pins.needed(use_text=False)
    assert pins.VJEPA2 in pins.needed(use_text=False)


def test_transformers_resolves_to_pinned_revision():
    import transformers.utils.hub as tf_hub

    from cortexreel.model import pin_transformers_revisions

    pin_transformers_revisions()
    pin_transformers_revisions()  # idempotent
    for pin in pins.EXTRACTORS:
        assert tf_hub.resolve_revision(pin.repo_id) == pin.revision
        assert tf_hub.resolve_revision(pin.repo_id, "main") == pin.revision
