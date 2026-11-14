"""The owner's REAL recordings cut into sentences: the male, same-speaker ceiling for every score.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/bench_owner.py
Writes out/bench-owner-real/ (24 kHz mono) + manifest.json. Cut points are the sentence pauses found with tools/find_pauses.py.
"""
import json
from pathlib import Path

import librosa
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
CLIPS = [  # file, start, end (s), transcript
    ("E:/tools/voiceclone/work/ref.wav", 0.0, 5.75, "Bir ming uch yuz o'ttiz oltinchi yilda Kesh viloyatida tug'ilgan."),
    ("E:/tools/voiceclone/work/ref.wav", 5.9, 10.2, "U o'ttiz besh yil davomida ulkan saltanatni boshqardi."),
    ("E:/tools/voiceclone/work/heldout.wav", 0.0, 7.2, "Sint daryosi bo'yidagi jangda u otda daryoga sakrab, narigi qirg'oqqa o'tdi."),
    ("E:/tools/voiceclone/work/heldout.wav", 7.6, 13.1, "Najmiddin Kubro esa shu tuproqda tug'ilibmiz, shu tuproqda o'lamiz degan."),
]
out = ROOT / "out" / "bench-owner-real"
out.mkdir(parents=True, exist_ok=True)
lines = []
for i, (path, a, b, text) in enumerate(CLIPS, 1):
    y, sr = librosa.load(path, sr=24000)
    name = f"r{i:02d}.wav"
    sf.write(out / name, y[int(a * sr):int(b * sr)], sr)
    lines.append({"file": name, "id": f"r{i:02d}", "category": "owner", "seed": 0, "text": text, "spoken": text,
                  "seconds": round(b - a, 2)})
(out / "manifest.json").write_text(json.dumps({"source": "owner's real voice", "lines": lines}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
print(len(lines), "clips ->", out)
