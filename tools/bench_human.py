"""Human reference: real Uzbek speakers from Google FLEURS (CC-BY-4.0) as an eval folder, scored like our voice.

    python tools/bench_human.py data/human/fleurs_uz --n 60
Picks up to N utterances from different speakers, both genders, 3-12 s long, and writes out/bench-human-fleurs/
(24 kHz mono copies + manifest.json with the real transcripts). Then: tools/eval_uz.py out/bench-human-fleurs
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("fleurs", type=Path)
ap.add_argument("--n", type=int, default=60)
args = ap.parse_args()
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)
wavs = {p.name: p for p in (args.fleurs / "audio").rglob("*.wav")}
rows = []
with open(args.fleurs / "dev.tsv", encoding="utf-8") as f:
    for r in csv.reader(f, delimiter="\t"):
        # id, file, raw transcription, normalized transcription, phonemes, num_samples, gender
        if len(r) < 7 or r[1] not in wavs:
            continue
        seconds = int(r[5]) / 16000
        if 3 <= seconds <= 12:
            rows.append({"fleurs_id": r[0], "src": r[1], "text": r[2], "seconds": seconds, "gender": r[6]})
# one clip per sentence id (≈ different recordings), alternating genders when both exist
# (uz_uz dev turned out to be all FEMALE: 146 sentences, 0 male)
by_gender = {}
for r in rows:
    by_gender.setdefault(r["gender"], []).append(r)
seen, picked = set(), []
queues = [iter(v) for v in by_gender.values()]
while len(picked) < args.n and queues:
    for q in list(queues):
        r = next((x for x in q if x["fleurs_id"] not in seen), None)
        if r is None:
            queues.remove(q)
            continue
        seen.add(r["fleurs_id"])
        picked.append(r)
        if len(picked) >= args.n:
            break
out = ROOT / "out" / "bench-human-fleurs"
out.mkdir(parents=True, exist_ok=True)
lines = []
for i, r in enumerate(picked, 1):
    name = f"h{i:03d}.wav"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wavs[r["src"]]), "-ar", "24000", "-ac", "1",
                    str(out / name)], check=True)
    lines.append({"file": name, "id": f"h{i:03d}", "category": r["gender"].lower(), "seed": 0, "text": r["text"],
                  "spoken": r["text"], "seconds": round(r["seconds"], 2), "source": f"FLEURS uz_uz dev {r['src']}"})
(out / "manifest.json").write_text(json.dumps({"source": "google/fleurs uz_uz dev (CC-BY-4.0)", "lines": lines},
                                              ensure_ascii=False, indent=1), encoding="utf-8")
print(len(lines), "human utterances ->", out, "| male", sum(l["category"] == "male" for l in lines),
      "female", sum(l["category"] == "female" for l in lines))
