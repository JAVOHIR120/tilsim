"""Tilsim API keys: create, look up, rate-limit and count usage.

Keys are stored only as SHA-256 hashes in data/api_keys.json (git-ignored). The plain key is shown once, at creation.

    python engine/apikeys.py create "Telegram bot" --out .secrets/telegram-bot.key   # key goes to the file only
    python engine/apikeys.py list
    python engine/apikeys.py revoke k_1a2b3c4d

A key looks like tlsm_<32 url-safe characters>. Send it as `X-API-Key: tlsm_...` or `Authorization: Bearer tlsm_...`.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import secrets
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "api_keys.json"
PREFIX = "tlsm_"


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def month_now() -> str:
    return time.strftime("%Y-%m")


class KeyStore:
    """Thread-safe; every change is written atomically (temp file + rename), so a crash never leaves half a file."""

    def __init__(self, path: Path = DEFAULT_PATH):
        self.path = Path(path)
        self.lock = threading.Lock()
        self.recent: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"keys": []}
        self.by_hash = {k["hash"]: k for k in self.data["keys"]}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)

    def create(self, name: str, per_minute: int = 30, monthly_chars: int | None = None,
               max_chars: int = 5000) -> tuple[dict, str]:
        key = PREFIX + secrets.token_urlsafe(24)
        rec = {"id": "k_" + secrets.token_hex(4), "name": name, "hash": _hash(key), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "active": True, "per_minute": per_minute, "monthly_chars": monthly_chars, "max_chars": max_chars, "usage": {}}
        with self.lock:
            self.data["keys"].append(rec)
            self.by_hash[rec["hash"]] = rec
            self._save()
        return rec, key

    def find(self, key: str | None) -> dict | None:
        """The active key record for this plain key, else None."""
        if not key or not key.startswith(PREFIX):
            return None
        rec = self.by_hash.get(_hash(key.strip()))
        return rec if rec and rec.get("active") else None

    def get(self, key_id: str) -> dict | None:
        return next((k for k in self.data["keys"] if k["id"] == key_id or k["name"] == key_id), None)

    def revoke(self, key_id: str) -> bool:
        with self.lock:
            rec = self.get(key_id)
            if not rec:
                return False
            rec["active"] = False
            self._save()
            return True

    def rate_wait(self, rec: dict, now: float | None = None) -> int:
        """0 if this key may make a request now (and counts it), else seconds to wait. Window: 60 s."""
        now = time.time() if now is None else now
        with self.lock:
            q = self.recent[rec["id"]]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= rec.get("per_minute", 30):
                return int(60 - (now - q[0])) + 1
            q.append(now)
            return 0

    def quota_left(self, rec: dict) -> int | None:
        """Characters left this month, or None for no monthly limit."""
        limit = rec.get("monthly_chars")
        if limit is None:
            return None
        used = rec.get("usage", {}).get(month_now(), {}).get("chars", 0)
        return max(0, limit - used)

    def record(self, rec: dict, chars: int, seconds: float) -> None:
        with self.lock:
            m = rec.setdefault("usage", {}).setdefault(month_now(), {"requests": 0, "chars": 0, "audio_seconds": 0.0})
            m["requests"] += 1
            m["chars"] += chars
            m["audio_seconds"] = round(m["audio_seconds"] + seconds, 2)
            self._save()

    def usage(self, rec: dict) -> dict:
        month = month_now()
        m = rec.get("usage", {}).get(month, {"requests": 0, "chars": 0, "audio_seconds": 0.0})
        return {"key": {"id": rec["id"], "name": rec["name"]}, "month": month, **m,
                "limits": {"per_minute": rec.get("per_minute"), "monthly_chars": rec.get("monthly_chars"),
                           "max_chars": rec.get("max_chars"), "chars_left": self.quota_left(rec)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, default=DEFAULT_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="make a new key")
    c.add_argument("name")
    c.add_argument("--per-minute", type=int, default=30)
    c.add_argument("--monthly-chars", type=int, default=None, help="characters per month; default: no limit")
    c.add_argument("--max-chars", type=int, default=5000, help="longest text in one job")
    c.add_argument("--out", type=Path, help="write the key to this file instead of printing it")
    sub.add_parser("list", help="show keys (never the key itself)")
    r = sub.add_parser("revoke", help="switch a key off")
    r.add_argument("key_id", help="k_... id or the key's name")
    args = ap.parse_args()

    store = KeyStore(args.store)
    if args.cmd == "create":
        rec, key = store.create(args.name, args.per_minute, args.monthly_chars, args.max_chars)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(key + "\n", encoding="utf-8")
            print(f"created {rec['id']} «{rec['name']}»; key written to {args.out}")
        else:
            print(f"created {rec['id']} «{rec['name']}». Key (shown once, keep it secret):\n{key}")
    elif args.cmd == "list":
        for k in store.data["keys"]:
            u = k.get("usage", {}).get(month_now(), {})
            print(f"{k['id']}  {'on ' if k['active'] else 'off'}  {k['name']:<24} {k['per_minute']}/min  "
                  f"month: {u.get('requests', 0)} req, {u.get('chars', 0)} chars, {u.get('audio_seconds', 0)} s")
    elif args.cmd == "revoke":
        print("revoked" if store.revoke(args.key_id) else "no such key")


if __name__ == "__main__":
    main()
