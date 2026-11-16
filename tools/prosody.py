"""Prosody of a take compared with the real speaker: is it flat (robot-like) or overdone (exaggerated)?

Measures, on voiced speech only:
- pitch_st: spread of the pitch in semitones (std around the median). Low = monotone/robotic, high = theatrical.
- pitch_range_st: 5th..95th percentile pitch range in semitones.
- rate: vowels per second of speaking time (pauses excluded), from the spoken text.
The speaker's own values come from real recordings (--real), so "natural" means "like this person", not a fixed number.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/prosody.py --real ref.wav heldout.wav
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import librosa
import numpy as np

VOWELS = re.compile(r"o'|[aeiou]")


def features(path: str | Path, text: str | None = None) -> dict:
    y, sr = librosa.load(path, sr=16000)
    hop = 160
    # pYIN, not YIN: YIN's octave jumps inflated the spread to ~6 semitones on real speech
    f0, is_voiced, _ = librosa.pyin(y, fmin=65, fmax=330, sr=sr, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0]
    n = min(len(f0), len(rms))
    f0, is_voiced, db = np.nan_to_num(f0[:n]), is_voiced[:n], 20 * np.log10(rms[:n] + 1e-9)
    speaking = db > max(np.percentile(db, 10) + 12, db.max() - 35)
    voiced = speaking & is_voiced & (f0 > 0)
    st = 12 * np.log2(f0[voiced] / np.median(f0[voiced])) if voiced.sum() > 10 else np.zeros(1)
    out = {
        "pitch_median_hz": round(float(np.median(f0[voiced])), 1) if voiced.sum() > 10 else None,
        "pitch_st": round(float(np.std(st)), 2),
        "pitch_range_st": round(float(np.percentile(st, 95) - np.percentile(st, 5)), 2),
        "speaking_s": round(float(speaking.sum() * hop / sr), 2),
        "loud_db": round(float(np.mean(db[speaking])), 1) if speaking.any() else None,  # mean level while speaking
    }
    if text:
        vowels = len(VOWELS.findall(text.lower()))
        out["rate"] = round(vowels / max(out["speaking_s"], 0.1), 2)
    return out


def penalty(f: dict, real: dict) -> float:
    """0 = within the speaker's natural band; grows as the take gets flatter or more extreme than the real person."""
    p = 0.0
    lo, hi = real["pitch_st"] * 0.75, real["pitch_st"] * 1.35
    if f["pitch_st"] < lo:
        p += (lo - f["pitch_st"]) / real["pitch_st"] * 2  # flat: robot-like
    elif f["pitch_st"] > hi:
        p += (f["pitch_st"] - hi) / real["pitch_st"] * 2  # overdone
    if "rate" in f and real.get("rate"):
        ratio = f["rate"] / real["rate"]
        if ratio < 0.8 or ratio > 1.25:
            p += abs(np.log(ratio)) * 2
    return round(p, 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", nargs="+", required=True, help="real recordings of the speaker")
    ap.add_argument("--texts", nargs="*", help="their transcripts (same order), for the speaking rate")
    ap.add_argument("--save", type=Path, help="write the speaker profile JSON here")
    args = ap.parse_args()
    feats = [features(p, t) for p, t in zip(args.real, args.texts or [None] * len(args.real))]
    profile = {k: round(float(np.mean([f[k] for f in feats if f.get(k) is not None])), 2)
               for k in ("pitch_median_hz", "pitch_st", "pitch_range_st", "rate") if any(f.get(k) for f in feats)}
    print(json.dumps({"files": feats, "profile": profile}, indent=1))
    if args.save:
        args.save.write_text(json.dumps(profile, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
