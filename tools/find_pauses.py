"""Find the pauses in a speech clip (to cut reference clips at sentence ends, never mid-word).

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/find_pauses.py clip.wav [--min 0.12]
"""
import argparse

import librosa
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--min", type=float, default=0.12, help="shortest pause to report, seconds")
args = ap.parse_args()
for f in args.files:
    y, sr = librosa.load(f, sr=16000)
    hop = 160
    db = 20 * np.log10(librosa.feature.rms(y=y, frame_length=400, hop_length=hop)[0] + 1e-9)
    floor = np.percentile(db, 10)
    quiet = db < max(floor + 10, db.max() - 40)
    pauses, start = [], None
    for i, q in enumerate(np.append(quiet, False)):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if (i - start) * hop / sr >= args.min:
                pauses.append((start * hop / sr, i * hop / sr))
            start = None
    print(f"{f}: {len(y) / sr:.2f} s, noise floor {floor:.0f} dB")
    for a, b in pauses:
        print(f"   pause {a:5.2f} - {b:5.2f}  ({b - a:.2f} s)")
