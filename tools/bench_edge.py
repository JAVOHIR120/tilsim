"""Benchmark voices: the same test sentences in Azure's Uzbek neural voices (free, through edge-tts, no key).

    python tools/bench_edge.py data/eval/test_all.tsv          (global Python 3.12, which has edge_tts)
Writes out/bench-azure-sardor/ and out/bench-azure-madina/ with a manifest.json each, ready for tools/eval_uz.py.
Only the generic test sentences are sent to Microsoft, nothing personal. Text goes in as uznorm's spoken form
(numbers as words), so every voice reads the same words.
"""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import edge_tts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uznorm import normalize  # noqa: E402

VOICES = {"sardor": "uz-UZ-SardorNeural", "madina": "uz-UZ-MadinaNeural"}


async def one(sem, voice, text, mp3: Path, wav: Path):
    async with sem:
        for attempt in range(4):
            try:
                await edge_tts.Communicate(text, voice).save(str(mp3))
                break
            except Exception as exc:  # network hiccup: retry
                if attempt == 3:
                    raise
                print("retry", mp3.name, exc, flush=True)
                await asyncio.sleep(2 * (attempt + 1))
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(mp3), "-ar", "24000", "-ac", "1",
                    str(wav)], check=True)
    mp3.unlink()


async def main(tsv: Path):
    items = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            item_id, category, text = line.split("\t", 2)
            items.append({"id": item_id, "category": category, "text": text, "spoken": normalize(text)})
    sem = asyncio.Semaphore(6)
    for name, voice in VOICES.items():
        out = ROOT / "out" / f"bench-azure-{name}"
        out.mkdir(parents=True, exist_ok=True)
        rows = [{**it, "file": f"{it['id']}.wav", "seed": 0} for it in items]
        await asyncio.gather(*(one(sem, voice, r["spoken"], out / f"{r['id']}.mp3", out / r["file"]) for r in rows))
        (out / "manifest.json").write_text(json.dumps({"voice": voice, "lines": rows}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        print(name, len(rows), "files ->", out, flush=True)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))
