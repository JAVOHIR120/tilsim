"""Best-of-N: for every sentence keep the take with the fewest heard errors, then the most natural, then the closest voice.

    python tools/pick_best.py out/eval-base            (after tools/eval_uz.py)

Writes OUT/best/<id>.wav, OUT/best/manifest.json and OUT/best/_all.mp3 (all picks in a row, for listening),
and prints first-take vs best-of-N averages.
"""
import json
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_uz import summarize  # noqa: E402

out = Path(sys.argv[1])
data = json.loads((out / "eval.json").read_text(encoding="utf-8"))
groups = defaultdict(list)
for row in data["lines"]:
    groups[row["id"]].append(row)


def key(r):
    # 1) fewest heard errors (in 2% steps: smaller CER gaps are mostly ASR noise), 2) highest UTMOS.
    # The prosody penalty was dropped: against the Gemini native judge (60 clips, 2026-11-15) it predicted nothing
    # (r = 0.04), while UTMOS did best (r = +0.41); best-of-3 picked by prosody scored *below* the first take.
    return (round(r.get("cer", 1) / 0.02), -(r.get("utmos") or 0), -(r.get("similarity") or 0))


first_seed = min(r["seed"] for r in data["lines"])
first = [r for r in data["lines"] if r["seed"] == first_seed]
best = [min(rows, key=key) for _, rows in sorted(groups.items())]

dest = out / "best"
dest.mkdir(exist_ok=True)
for r in best:
    shutil.copyfile(out / r["file"], dest / f"{r['id']}.wav")
(dest / "manifest.json").write_text(json.dumps({"from": str(out), "lines": [{**r, "file": f"{r['id']}.wav"} for r in best]},
                                               ensure_ascii=False, indent=1), encoding="utf-8")

a, b = summarize(first), summarize(best)
print(f"{'':14}{'take 1':>10}{'best of ' + str(len(groups[best[0]['id']])):>12}")
for k in ("cer", "wer", "utmos", "similarity"):
    print(f"{k:14}{a[k]:>10}{b[k]:>12}")
print("\nletters correct (take 1 -> best):")
for u, v in b["letters_correct"].items():
    print(f"  {u:3} {a['letters_correct'].get(u, {}).get('ok')} -> {v['ok']}  (n={v['n']})")
print("\nconfusions left in best:", b["top_confusions"])
print("lines still with errors:")
for r in best:
    if r.get("cer", 0) > 0:
        print(f"  {r['id']} s{r['seed']} CER {r['cer']:.3f}  want: {r['ref']}\n{'':>14}heard: {r['hyp']}")

# one file to listen to, in order, with short gaps
try:
    import soundfile as sf
    parts = []
    for r in best:
        y, sr = sf.read(dest / f"{r['id']}.wav", dtype="float32")
        parts += [y, np.zeros(int(0.7 * sr), np.float32)]
    sf.write(dest / "_all.wav", np.concatenate(parts), sr)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4", "-i", str(dest / "_all.wav"),
                    "-codec:a", "libmp3lame", "-b:a", "160k", str(dest / "_all.mp3")], check=True)
    (dest / "_all.wav").unlink()
    print("listening file:", dest / "_all.mp3")
except Exception as exc:  # soundfile/ffmpeg missing: the picks are still saved
    print("no listening file:", exc)
