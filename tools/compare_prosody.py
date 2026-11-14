"""Prosody of several eval folders side by side, limited to the same sentence ids and seeds.

    E:/tools/voiceclone/.venv/Scripts/python.exe tools/compare_prosody.py out/eval-base out/quick-prefix --ids t11,t12 --seeds 0,1
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prosody import features  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folders", nargs="+", type=Path)
ap.add_argument("--ids", default="")
ap.add_argument("--seeds", default="0,1")
args = ap.parse_args()
ids = set(args.ids.split(",")) if args.ids else None
seeds = {int(s) for s in args.seeds.split(",")}
real = json.loads((Path(__file__).resolve().parents[1] / "data/eval/speaker_profile.json").read_text())
print(f"{'real speaker':28} pitch spread {real['pitch_st']:.2f} st   rate {real['rate']:.2f} vowels/s")
for folder in args.folders:
    data = json.loads((folder / "eval.json").read_text(encoding="utf-8"))["lines"]
    rows = [r for r in data if (ids is None or r["id"] in ids) and r.get("seed", 0) in seeds]
    feats = [r if "pitch_st" in r else {**r, **features(folder / r["file"], r["spoken"])} for r in rows]
    n = len(feats)
    pitch = sum(f["pitch_st"] for f in feats) / n
    rate = sum(f["rate"] for f in feats) / n
    flat = sum(f["pitch_st"] < real["pitch_st"] * 0.75 for f in feats)
    print(f"{folder.name:28} pitch spread {pitch:.2f} st   rate {rate:.2f} vowels/s   flat {flat}/{n}", flush=True)
