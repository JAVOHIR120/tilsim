"""Slow down (or speed up) every take in an eval folder with rubberband (formants kept), into a new folder.

    python tools/stretch_folder.py out/grid-p6 out/grid-p6-slow88 --tempo 0.88
The new folder gets a manifest.json, so tools/eval_uz.py can score it like any other run.
"""
import argparse
import json
import subprocess
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--tempo", type=float, required=True, help="<1 slower, >1 faster")
args = ap.parse_args()
args.dst.mkdir(parents=True, exist_ok=True)
manifest = json.loads((args.src / "manifest.json").read_text(encoding="utf-8"))
for row in manifest["lines"]:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "4", "-i", str(args.src / row["file"]),
                    "-af", f"rubberband=tempo={args.tempo}:formant=preserved:pitchq=quality:transients=smooth:detector=soft",
                    "-ar", "24000", "-ac", "1", str(args.dst / row["file"])], check=True)
    row["seconds"] = round(row["seconds"] / args.tempo, 2)
manifest["stretch_tempo"] = args.tempo
(args.dst / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
print("stretched", len(manifest["lines"]), "files ->", args.dst)
