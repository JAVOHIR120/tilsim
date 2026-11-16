"""Pick emotion-bank candidates from an emo_lab run: clean takes first, then the ones whose voice moved the right way.

    python tools/emo_pick.py out/emo/A                    # shortlist per emotion -> OUT/shortlist.json (+ ids for the judge)
    python tools/emo_pick.py out/emo/A --judged           # after gemini_judge --mode emotion: final ranking per emotion

1. Clean: CER <= --max-cer, voice similarity >= --min-sim, UTMOS >= --min-mos, 2.0-8.0 s long.
2. Acoustic fit: each take's pitch level (semitones vs the neutral takes), pitch range, speaking rate and loudness,
   as z-scores against the neutral takes, weighted by the direction that emotion moves the voice in speech research
   (e.g. excited: higher, wider, faster, louder; calm: lower, narrower, slower, softer). Only a pre-filter, so the
   paid judge hears the likeliest takes; the judge (and the founder's ear) decide.
3. --judged: adds the blind judge's label (gemini-emotion.json): a take counts when the judge's first or second label
   is the intended emotion, it is not over-acted and natural >= 7.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

LABEL = {"neytral": "neutral", "iliq": "warm", "quvnoq": "cheerful", "hayajonli": "excited", "xotirjam": "calm",
         "jiddiy": "serious", "hamdard": "sympathetic", "hayron": "surprised"}
# direction weights: (pitch level, pitch range, rate, loudness)
DIRECTION = {
    "iliq": (0.5, 0.5, -0.3, 0.0), "quvnoq": (1.0, 0.8, 0.4, 0.5), "hayajonli": (1.0, 1.0, 0.8, 1.0),
    "hayron": (0.8, 1.0, 0.0, 0.3), "xotirjam": (-0.6, -0.6, -1.0, -0.8), "hamdard": (-0.4, 0.0, -0.8, -0.8),
    "jiddiy": (-0.6, -0.8, -0.2, 0.2), "neytral": (0.0, 0.0, 0.0, 0.0),
}


def semitones(hz: float, ref: float) -> float:
    return 12 * math.log2(hz / ref)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--max-cer", type=float, default=0.02)
    ap.add_argument("--min-sim", type=float, default=0.86)
    ap.add_argument("--min-mos", type=float, default=3.3)
    ap.add_argument("--per-emotion", type=int, default=12)
    ap.add_argument("--judged", action="store_true")
    args = ap.parse_args()
    rows = json.loads((args.out / "eval.json").read_text(encoding="utf-8"))["lines"]
    emo_of = lambda r: r["category"]  # noqa: E731
    neutral = [r for r in rows if emo_of(r) == "neytral" and r.get("pitch_hz")]
    if not neutral:
        raise SystemExit("no neutral takes to compare with")
    ref_hz = sorted(r["pitch_hz"] for r in neutral)[len(neutral) // 2]

    def feats(r):
        return (semitones(r["pitch_hz"], ref_hz), r["pitch_range_st"], r["rate"], r["loud_db"])

    cols = list(zip(*[feats(r) for r in neutral]))
    mu = [sum(c) / len(c) for c in cols]
    sd = [max(1e-3, (sum((x - m) ** 2 for x in c) / max(1, len(c) - 1)) ** 0.5) for c, m in zip(cols, mu)]
    judged = {}
    jpath = args.out / "gemini-emotion.json"
    if args.judged and jpath.exists():
        judged = json.loads(jpath.read_text(encoding="utf-8"))
    by = defaultdict(list)
    dropped = defaultdict(int)
    for r in rows:
        e = emo_of(r)
        ok = (r.get("cer", 1) <= args.max_cer and r.get("similarity", 0) >= args.min_sim
              and r.get("utmos", 0) >= args.min_mos and 2.0 <= r["seconds"] <= 8.0 and r.get("pitch_hz"))
        if not ok:
            dropped[e] += 1
            continue
        z = [(x - m) / s for x, m, s in zip(feats(r), mu, sd)]
        fit = sum(w * v for w, v in zip(DIRECTION[e], z))
        j = judged.get(r["file"], {})
        hit = 1.0 if j.get("emotion") == LABEL[e] else 0.5 if j.get("second") == LABEL[e] else 0.0
        if j and (j.get("overacted") or (j.get("natural") or 0) < 7):
            hit *= 0.5
        by[e].append({"file": r["file"], "tag": r.get("tag"), "params": {k: r.get(k) for k in ("exaggeration", "cfg_weight", "temperature")},
                      "fit": round(fit, 2), "z": [round(v, 2) for v in z], "cer": r.get("cer"), "sim": r.get("similarity"),
                      "mos": r.get("utmos"), "seconds": r["seconds"], "text": r["text"],
                      **({"judge": j.get("emotion"), "second": j.get("second"), "intensity": j.get("intensity"),
                          "natural": j.get("natural"), "overacted": j.get("overacted"), "hit": hit} if j else {})})
    short, ids = {}, []
    for e, cands in sorted(by.items()):
        key = (lambda c: (c.get("hit", 0), c.get("intensity") or 0, c["fit"])) if args.judged else (lambda c: c["fit"])
        cands.sort(key=key, reverse=True)
        keep = cands if args.judged else cands[: args.per_emotion if e != "neytral" else 10]
        short[e] = keep
        ids += [c["file"] for c in keep]
        total = len(cands) + dropped[e]
        line = f"{e:10} clean {len(cands)}/{total}  "
        if args.judged and cands:
            hits = sum(c.get("hit", 0) == 1.0 for c in cands if "judge" in c)
            n = sum("judge" in c for c in cands)
            line += f"judge first-label hits {hits}/{n}  best: {cands[0]['file']} ({cands[0].get('judge')}, " \
                    f"int {cands[0].get('intensity')}, nat {cands[0].get('natural')})"
        else:
            line += "top fit " + ", ".join(f"{c['fit']:+.1f}" for c in keep[:5])
        print(line)
    name = "ranking.json" if args.judged else "shortlist.json"
    (args.out / name).write_text(json.dumps(short, ensure_ascii=False, indent=1), encoding="utf-8")
    if not args.judged:
        (args.out / "shortlist_files.txt").write_text("\n".join(ids), encoding="utf-8")
        print(f"{len(ids)} takes shortlisted -> {args.out / 'shortlist_files.txt'}")


if __name__ == "__main__":
    main()
