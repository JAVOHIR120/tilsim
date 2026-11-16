"""Blind emotion-recognition score of an emo_lab bank run (gemini_judge --mode emotion): accuracy and confusions per mood.

    python tools/emo_score.py out/emo/B1 [out/emo/B2 ...]
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABEL = {"neytral": "neutral", "iliq": "warm", "quvnoq": "cheerful", "hayajonli": "excited", "xotirjam": "calm",
         "jiddiy": "serious", "hamdard": "sympathetic", "hayron": "surprised"}

BY_CATEGORY = "--by-category" in sys.argv   # the intended mood is the text's category (neutral voice on mood text)
for folder in [Path(a) for a in sys.argv[1:] if not a.startswith("--")]:
    rows = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))["lines"]
    judged = json.loads((folder / "gemini-emotion.json").read_text(encoding="utf-8"))
    by = defaultdict(list)
    for r in rows:
        j = judged.get(r["file"])
        if j and "emotion" in j:
            tag = {"ili": "iliq", "quv": "quvnoq", "hay": "hayajonli", "xot": "xotirjam", "jid": "jiddiy", "ham": "hamdard",
                   "hrn": "hayron", "neu": "neytral"}.get((r.get("tag") or "")[:3])
            by[r["category"] if BY_CATEGORY else (r.get("emotion") or tag or r["category"])].append(j)
    total = hit = top2 = 0
    print(folder)
    for e, js in by.items():
        h = sum(j["emotion"] == LABEL[e] for j in js)
        t2 = sum(LABEL[e] in (j["emotion"], j.get("second")) for j in js)
        nat = sum(j.get("natural") or 0 for j in js) / len(js)
        over = sum(bool(j.get("overacted")) for j in js)
        conf = Counter(j["emotion"] for j in js if j["emotion"] != LABEL[e]).most_common(3)
        print(f"  {e:10} {h}/{len(js)} first, {t2}/{len(js)} top-2, natural {nat:.1f}, overacted {over}  wrong: {conf}")
        total, hit, top2 = total + len(js), hit + h, top2 + t2
    print(f"  ALL       {hit}/{total} = {100 * hit / max(total, 1):.0f}% first label, {100 * top2 / max(total, 1):.0f}% top-2")
