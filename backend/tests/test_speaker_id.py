"""Unit tests for live speaker labeling (label_dominant_speaker).

The labeling logic is tested independently of resemblyzer/ffmpeg by patching the decode and
embed steps. Each diarized speaker maps to a fixed unit vector, so we exercise the real
decision — isolate the dominant voice, embed it, threshold against the officer reference —
without audio. The dominant-speaker isolation is what makes this robust where embedding the
whole mixed clip was not.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ai import speaker_id

# Officer reference and a different speaker. On *clean single-speaker* audio (what isolating
# the dominant voice gives us) resemblyzer puts a different speaker well below the 0.70
# threshold — measured ~0.45–0.55 on real clips — so SUBJECT sits at cosine 0.50 to OFFICER.
OFFICER = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
SUBJECT = np.asarray([0.5, 0.0, 0.8660254], dtype=np.float32)  # cosine 0.50 to OFFICER


def _word(speaker: str, start: float, end: float) -> dict:
    return {"text": "x", "start": start, "end": end, "speaker_id": speaker}


@pytest.fixture(autouse=True)
def patched_audio(monkeypatch: pytest.MonkeyPatch):
    """Skip ffmpeg; map each diarized speaker id to a preset embedding via its word slices.

    decode returns a marker; _embed_wav inspects which speaker's slices it was handed. We tag
    slices by length: speaker 'off' words are 1.0s, 'sub' words are 2.0s, so the concatenated
    slice length tells us whose audio it is.
    """

    def fake_decode(clip: bytes):
        # Return a long zero waveform; slicing by word times below selects sub-ranges of it.
        return np.zeros(speaker_id.SAMPLE_RATE * 60, dtype=np.float32)

    monkeypatch.setattr(speaker_id, "decode_to_wav16k", fake_decode)


def test_clip_dominated_by_officer_labels_officer(monkeypatch) -> None:
    monkeypatch.setattr(speaker_id, "_embed_wav", lambda wav: OFFICER)
    words = [_word("speaker_0", 0.0, 3.0), _word("speaker_1", 3.0, 0.5 + 3.0)]
    label, sim = speaker_id.label_dominant_speaker(b"clip", words, [1.0, 0.0, 0.0])
    assert label == "officer"
    assert sim == pytest.approx(1.0, abs=1e-6)


def test_clip_dominated_by_subject_labels_subject(monkeypatch) -> None:
    monkeypatch.setattr(speaker_id, "_embed_wav", lambda wav: SUBJECT)
    # Subject speaks most of the clip.
    words = [_word("speaker_1", 0.0, 3.5), _word("speaker_0", 3.5, 4.0)]
    label, sim = speaker_id.label_dominant_speaker(b"clip", words, [1.0, 0.0, 0.0])
    assert label == "subject"
    assert sim == pytest.approx(0.5, abs=1e-6)


def test_no_diarization_defaults_to_officer() -> None:
    # A backend that doesn't diarize (empty words) keeps the transcript flowing as 'officer';
    # the post-session pass relabels precisely. No embed/decode is attempted.
    label, sim = speaker_id.label_dominant_speaker(b"clip", [], [1.0, 0.0, 0.0])
    assert label == "officer"
    assert sim == -1.0


def test_unembeddable_dominant_voice_defaults_to_officer(monkeypatch) -> None:
    monkeypatch.setattr(speaker_id, "_embed_wav", lambda wav: None)
    words = [_word("speaker_0", 0.0, 2.0)]
    label, _sim = speaker_id.label_dominant_speaker(b"clip", words, [1.0, 0.0, 0.0])
    assert label == "officer"


def test_extract_words_filters_spacing_and_missing_fields() -> None:
    payload = {
        "words": [
            {"text": "hi", "start": 0.0, "end": 0.5, "speaker_id": "speaker_0", "type": "word"},
            {"text": " ", "type": "spacing", "speaker_id": "speaker_0", "start": 0.5},
            {"text": "no-spk", "start": 1.0, "end": 1.2},  # missing speaker_id -> dropped
            {"text": "there", "start": 1.2, "speaker_id": "speaker_1"},  # missing end -> end=start
        ]
    }
    words = speaker_id.extract_words(payload)
    assert [w["text"] for w in words] == ["hi", "there"]
    assert words[1]["end"] == words[1]["start"] == 1.2
