"""Publish where the Tilsim voice server is, for the website and clients/tilsim.py (they read this gist).

    python tools/set_api_address.py --laptop https://x-y-z.trycloudflare.com     # start-tilsim.ps1 does this
    python tools/set_api_address.py --laptop none                               # laptop server stopped
    python tools/set_api_address.py --cloud https://javohir120--tilsim-api.us-east.modal.run
    python tools/set_api_address.py                                              # just show what is there

Readers try the laptop first (free, and its health check costs nothing), then the cloud (Modal credit).
"api" keeps the old single-address field for pages and clients cached before 2026-11-15.
Uses the GitHub CLI (`gh`), which is already signed in on this PC.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GIST = "5801b83645819ab8d22340b5b60798f2"
FILE = "tilsim-api.json"


def read() -> dict:
    out = subprocess.run(["gh", "api", f"gists/{GIST}"], capture_output=True, check=True).stdout
    try:
        return json.loads(json.loads(out)["files"][FILE]["content"])
    except (KeyError, ValueError):
        return {}


def write(cfg: dict) -> None:
    cfg["api"] = cfg.get("laptop") or cfg.get("cloud")
    cfg["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    body = {"files": {FILE: {"content": json.dumps(cfg, ensure_ascii=False)}}}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(body, f)
    try:
        subprocess.run(["gh", "api", "-X", "PATCH", f"gists/{GIST}", "--input", f.name], capture_output=True, check=True)
    finally:
        Path(f.name).unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laptop", help="tunnel URL, or 'none'")
    ap.add_argument("--cloud", help="Modal URL, or 'none'")
    args = ap.parse_args()
    cfg = read()
    for field in ("laptop", "cloud"):
        value = getattr(args, field)
        if value is not None:
            cfg[field] = None if value.lower() in ("none", "null", "") else value.rstrip("/")
    if args.laptop is not None or args.cloud is not None:
        write(cfg)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(read(), ensure_ascii=False))


if __name__ == "__main__":
    main()
