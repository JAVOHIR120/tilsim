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
    """Thread-safe. Every change re-reads the file, edits one record and writes it back atomically (temp + rename).
    So the server and the key CLI, two separate writers, never undo each other's work.

    The server re-reads the file every REFRESH seconds, or on an unknown key, so new and revoked keys take effect
    without a restart. `before_read` / `after_write` hooks let a cloud volume reload and commit around
    each read and write (deploy/modal_app.py)."""

    REFRESH = 30

    def __init__(self, path: Path = DEFAULT_PATH, before_read=None, after_write=None):
        self.path = Path(path)
        self.before_read, self.after_write = before_read, after_write
        self.lock = threading.RLock()
        self.recent: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self._reload()

    def _read(self) -> dict:
        if self.before_read:
            try:
                self.before_read()
            except Exception:  # noqa: BLE001 (a busy cloud volume: use what is on disk now)
                pass
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {"keys": []}

    def _index(self, data: dict) -> None:
        self.data = data
        self.by_hash = {k["hash"]: k for k in data["keys"]}
        self.loaded_at = time.time()

    def _reload(self) -> None:
        self._index(self._read())

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
        self._index(data)
        if self.after_write:
            self.after_write()

    def _edit(self, key_id: str, change) -> dict | None:
        """Read the newest file, apply change(record) to one key, write it back. Returns that record."""
        with self.lock:
            data = self._read()
            rec = next((k for k in data["keys"] if key_id in (k["id"], k["name"])), None)
            if rec is None:
                return None
            change(rec)
            self._write(data)
            return rec

    def create(self, name: str, per_minute: int = 30, monthly_chars: int | None = None,
               max_chars: int = 5000) -> tuple[dict, str]:
        key = PREFIX + secrets.token_urlsafe(24)
        rec = {"id": "k_" + secrets.token_hex(4), "name": name, "hash": _hash(key), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "active": True, "per_minute": per_minute, "monthly_chars": monthly_chars, "max_chars": max_chars, "usage": {}}
        with self.lock:
            data = self._read()
            data["keys"].append(rec)
            self._write(data)
        return rec, key

    def find(self, key: str | None) -> dict | None:
        """The active key record for this plain key, else None."""
        if not key or not key.startswith(PREFIX):
            return None
        h = _hash(key.strip())
        with self.lock:
            age = time.time() - self.loaded_at
            rec = self.by_hash.get(h)
            if age > self.REFRESH or (rec is None and age > 5):  # pick up new or revoked keys
                self._reload()
                rec = self.by_hash.get(h)
        return rec if rec and rec.get("active") else None

    def get(self, key_id: str) -> dict | None:
        return next((k for k in self.data["keys"] if key_id in (k["id"], k["name"])), None)

    def revoke(self, key_id: str) -> bool:
        return self._edit(key_id, lambda r: r.update(active=False)) is not None

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
        def bump(r: dict) -> None:
            m = r.setdefault("usage", {}).setdefault(month_now(), {"requests": 0, "chars": 0, "audio_seconds": 0.0})
            m["requests"] += 1
            m["chars"] += chars
            m["audio_seconds"] = round(m["audio_seconds"] + seconds, 2)
        fresh = self._edit(rec["id"], bump)
        if fresh is not None:
            rec["usage"] = fresh["usage"]  # keep the caller's copy current (quota checks read it)

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
