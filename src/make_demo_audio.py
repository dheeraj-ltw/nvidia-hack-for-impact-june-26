"""Build a synthetic, noisy multi-speaker demo for the officer-id pipeline.

Generates a conversation between a police officer and THREE other people using
distinct ElevenLabs voices, mixes in background noise at a target SNR, and writes
a separate officer reference clip. Useful as a repeatable test fixture for
`src/speaker_id.py` (no real field recording needed).

    python src/make_demo_audio.py --snr-db 15 --num-others 3
    make identify              # then run the officer-id pipeline on the output
"""
from __future__ import annotations

import argparse
import pathlib

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
AUDIO = ROOT / "data" / "audio"
SAMPLE_RATE = 16_000

load_dotenv(ROOT / ".env")

# Distinct ElevenLabs voices: the officer + three other people.
VOICE_OFFICER = "aN0SWmSqDdwgj0ww8TIu"        # configured "officer" voice
OTHER_VOICES = [
    "21m00Tcm4TlvDq8ikWAM",  # Rachel (female)
    "ErXwobaYiN019PkySvjV",  # Antoni (male)
    "VR6AewLTigWG4xSOukaG",  # Arnold (male)
]

# (speaker_key, text) in conversation order. "officer" => officer voice;
# "other0/1/2" => the three OTHER_VOICES.
SCRIPT = [
    ("officer", "Step back from the vehicle and keep your hands where I can see them."),
    ("other0", "Officer, he's my brother, can you please tell us what is going on here?"),
    ("other1", "I already told you, we were just standing on the pavement, this is ridiculous."),
    ("officer", "I understand you are upset. You are being detained while we check a report."),
    ("other2", "You cannot search us without a reason, that is against our rights."),
    ("officer", "Under section one of PACE I have reasonable grounds to carry out this search."),
    ("other0", "Everyone please just stay calm, let's not make this situation any worse."),
    ("other1", "Fine, but I want all of your badge numbers written down right now."),
]

OFFICER_REFERENCE = "This is the duty officer speaking, please confirm your location, over."


def _synth_turn(text: str, voice_id: str) -> np.ndarray:
    """TTS one line and return it as a 16 kHz mono waveform."""
    from audio import text_to_speech  # reuse the existing ElevenLabs TTS

    tmp = AUDIO / "_turns"
    tmp.mkdir(parents=True, exist_ok=True)
    # Stable-ish filename per voice so repeated runs overwrite rather than pile up.
    path = tmp / f"{voice_id[:8]}_{abs(hash(text)) % 10**8}.mp3"
    text_to_speech(text, voice_id=voice_id, out_path=path)
    wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return wav


def _background_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Ambient-ish noise: low-frequency rumble + faint hiss (numpy only)."""
    white = rng.standard_normal(n)
    brown = np.cumsum(white)                       # integrate -> low-frequency rumble
    brown = brown - np.linspace(brown[0], brown[-1], n)  # remove DC drift
    brown /= (np.max(np.abs(brown)) or 1.0)
    hiss = 0.15 * white / (np.max(np.abs(white)) or 1.0)
    noise = brown + hiss
    return noise / (np.max(np.abs(noise)) or 1.0)


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale `noise` to the target SNR (vs speech RMS) and add it to `speech`."""
    speech_rms = np.sqrt(np.mean(speech ** 2)) or 1e-9
    noise_rms = np.sqrt(np.mean(noise ** 2)) or 1e-9
    target_noise_rms = speech_rms / (10 ** (snr_db / 20))
    mixed = speech + noise * (target_noise_rms / noise_rms)
    peak = np.max(np.abs(mixed)) or 1.0
    return (mixed / peak * 0.95).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a noisy multi-speaker demo conversation")
    parser.add_argument("--snr-db", type=float, default=15.0,
                        help="speech-to-noise ratio in dB (lower = noisier; default 15)")
    parser.add_argument("--num-others", type=int, default=3, choices=[1, 2, 3],
                        help="how many non-officer speakers to include (default 3)")
    parser.add_argument("--gap", type=float, default=0.4, help="silence between turns, seconds")
    parser.add_argument("--out", default=str(AUDIO / "conversation.wav"))
    parser.add_argument("--ref", default=str(AUDIO / "officer_ref.mp3"))
    parser.add_argument("--seed", type=int, default=7, help="noise RNG seed (reproducible)")
    args = parser.parse_args()

    AUDIO.mkdir(parents=True, exist_ok=True)
    allowed_others = {f"other{i}" for i in range(args.num_others)}

    print(f"Synthesizing conversation (officer + {args.num_others} others)...")
    gap = np.zeros(int(args.gap * SAMPLE_RATE), dtype=np.float32)
    chunks: list[np.ndarray] = []
    used_speakers: list[str] = []
    for key, text in SCRIPT:
        if key.startswith("other") and key not in allowed_others:
            continue
        voice = VOICE_OFFICER if key == "officer" else OTHER_VOICES[int(key[-1])]
        chunks.append(_synth_turn(text, voice))
        chunks.append(gap)
        if key not in used_speakers:
            used_speakers.append(key)
        print(f"  [{key:7}] {text[:60]}...")

    speech = np.concatenate(chunks)
    rng = np.random.default_rng(args.seed)
    mixed = _mix_at_snr(speech, _background_noise(len(speech), rng), args.snr_db)

    sf.write(args.out, mixed, SAMPLE_RATE)
    dur = len(mixed) / SAMPLE_RATE
    print(f"\nwrote {args.out} ({dur:.1f}s, {len(used_speakers)} speakers, SNR={args.snr_db} dB)")

    print("Synthesizing officer reference clip...")
    from audio import text_to_speech
    text_to_speech(OFFICER_REFERENCE, voice_id=VOICE_OFFICER, out_path=args.ref)
    print(f"wrote {args.ref}")

    total_speakers = 1 + args.num_others
    print(f"\nNext:  make identify CONV={args.out} REF={args.ref}")
    print(f"       (or: python src/speaker_id.py identify --conversation {args.out} "
          f"--officer {args.ref} --num-speakers {total_speakers})")


if __name__ == "__main__":
    main()
