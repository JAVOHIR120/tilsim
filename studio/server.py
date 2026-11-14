"""Ovoz studiyasi: a local recording studio for building your TTS voice dataset.

    python studio/server.py                       # voice "ovozim", script data/scripts/starter_uz.tsv
    python studio/server.py --voice ovozim --script data/scripts/starter_uz.tsv --port 8765

Then open http://127.0.0.1:8765 in Chrome or Edge. Takes are saved as 16-bit mono WAV (48 kHz) in
data/recordings/<voice>/wavs/<id>.wav, and data/recordings/<voice>/metadata.csv holds `id|text|spoken`
(LJSpeech style, which every TTS trainer reads). A re-recorded take moves the old file to trash/, never deletes it.
Standard library only; it listens on 127.0.0.1, so nothing leaves this computer.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
import wave
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uznorm import check, normalize  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MAX_WAV_BYTES = 40 * 1024 * 1024
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".wav": "audio/wav"}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("|", " ")).strip()


class Store:
    """The script (what to read) plus what has been recorded so far."""

    def __init__(self, voice_dir: Path, script_paths: list[Path]):
        self.dir = voice_dir
        self.wavs = voice_dir / "wavs"
        self.trash = voice_dir / "trash"
        self.meta_path = voice_dir / "metadata.csv"
        self.takes_log = voice_dir / "takes.jsonl"
        self.wavs.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.script = self._load_script(script_paths)
        self.meta = self._load_meta()
        self.durations = {p.stem: self._duration(p) for p in self.wavs.glob("*.wav")}

    @staticmethod
    def _load_script(paths: list[Path]) -> list[dict]:
        items, seen = [], set()
        for path in paths:
            n = 0
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                n += 1
                if "\t" in line:
                    item_id, text = line.split("\t", 1)
                else:
                    item_id, text = f"{path.stem}_{n:04d}", line
                item_id = item_id.strip()
                if not ID_RE.match(item_id) or item_id in seen:
                    raise SystemExit(f"{path}: bad or repeated id {item_id!r}")
                seen.add(item_id)
                items.append({"id": item_id, "text": clean_text(text)})
        return items

    def _load_meta(self) -> dict[str, dict]:
        meta = {}
        if self.meta_path.exists():
            for line in self.meta_path.read_text(encoding="utf-8").splitlines():
                parts = line.split("|")
                if len(parts) >= 2 and ID_RE.match(parts[0]):
                    meta[parts[0]] = {"text": parts[1], "spoken": parts[2] if len(parts) > 2 else normalize(parts[1])}
        return meta

    @staticmethod
    def _duration(path: Path) -> float:
        try:
            with wave.open(str(path)) as w:
                return w.getnframes() / w.getframerate()
        except (wave.Error, OSError, EOFError):
            return 0.0

    def state(self) -> dict:
        items = []
        for item in self.script:
            text = self.meta.get(item["id"], {}).get("text", item["text"])
            spoken = normalize(text)
            items.append({"id": item["id"], "text": text, "spoken": spoken, "issues": check(spoken),
                          "edited": text != item["text"], "recorded": item["id"] in self.durations,
                          "duration": round(self.durations.get(item["id"], 0.0), 2)})
        return {"voice": self.dir.name, "items": items, "totalSeconds": round(sum(self.durations.values()), 1)}

    def save(self, item_id: str, text: str, wav_bytes: bytes, stats: dict) -> dict:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() < 16000 or w.getnframes() == 0:
                raise ValueError("expected non-empty 16-bit mono WAV at 16 kHz or more")
            duration = w.getnframes() / w.getframerate()
        text = clean_text(text)
        spoken = normalize(text)
        with self.lock:
            target = self.wavs / f"{item_id}.wav"
            if target.exists():  # keep the old take, just in case
                self.trash.mkdir(exist_ok=True)
                target.replace(self.trash / f"{item_id}.{time.strftime('%Y%m%d-%H%M%S')}.wav")
            tmp = target.with_suffix(".tmp")
            tmp.write_bytes(wav_bytes)
            tmp.replace(target)
            self.durations[item_id] = duration
            self.meta[item_id] = {"text": text, "spoken": spoken}
            self._write_meta()
            with self.takes_log.open("a", encoding="utf-8") as log:
                log.write(json.dumps({"id": item_id, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "duration": round(duration, 3), **stats}, ensure_ascii=False) + "\n")
        return {"id": item_id, "duration": round(duration, 2), "spoken": spoken,
                "totalSeconds": round(sum(self.durations.values()), 1)}

    def remove(self, item_id: str) -> None:
        with self.lock:
            target = self.wavs / f"{item_id}.wav"
            if target.exists():
                self.trash.mkdir(exist_ok=True)
                target.replace(self.trash / f"{item_id}.{time.strftime('%Y%m%d-%H%M%S')}.wav")
            self.durations.pop(item_id, None)
            self._write_meta()

    def _write_meta(self) -> None:
        order = [i["id"] for i in self.script] + sorted(set(self.meta) - {i["id"] for i in self.script})
        lines = [f"{i}|{self.meta[i]['text']}|{self.meta[i]['spoken']}"
                 for i in order if i in self.meta and i in self.durations]
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        tmp.replace(self.meta_path)


def make_handler(store: Store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "OvozStudio/1"

        def log_message(self, fmt, *args):  # quieter console: only errors
            if args and str(args[1]).startswith(("4", "5")):
                super().log_message(fmt, *args)

        def _send(self, status: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

        def _id(self, query: dict) -> str | None:
            item_id = query.get("id", [""])[0]
            return item_id if ID_RE.match(item_id) else None

        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path == "/api/state":
                return self._json(store.state())
            if url.path == "/api/audio":
                item_id = self._id(query)
                path = store.wavs / f"{item_id}.wav" if item_id else None
                if not path or not path.exists():
                    return self._json({"error": "not found"}, 404)
                return self._send(200, path.read_bytes(), "audio/wav")
            name = "index.html" if url.path in ("/", "/index.html") else url.path.removeprefix("/static/")
            path = (STATIC / name).resolve()
            if STATIC.resolve() not in path.parents or not path.is_file():
                return self._json({"error": "not found"}, 404)
            return self._send(200, path.read_bytes(), CONTENT_TYPES.get(path.suffix, "application/octet-stream"))

        def do_POST(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_WAV_BYTES:
                return self._json({"error": "too large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            body = self.rfile.read(length) if length else b""
            try:
                if url.path == "/api/normalize":
                    text = json.loads(body or b"{}").get("text", "")
                    spoken = normalize(clean_text(text))
                    return self._json({"spoken": spoken, "issues": check(spoken)})
                item_id = self._id(query)
                if not item_id:
                    return self._json({"error": "bad id"}, 400)
                if url.path == "/api/save":
                    text = unquote(self.headers.get("X-Text", ""))
                    stats = json.loads(unquote(self.headers.get("X-Stats", "")) or "{}")
                    return self._json(store.save(item_id, text, body, stats if isinstance(stats, dict) else {}))
                if url.path == "/api/remove":
                    store.remove(item_id)
                    return self._json({"ok": True})
            except (ValueError, wave.Error, EOFError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, 400)
            return self._json({"error": "not found"}, 404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", default="ovozim", help="folder name for this voice (default: ovozim)")
    parser.add_argument("--script", nargs="+", type=Path, default=[ROOT / "data/scripts/starter_uz.tsv"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/recordings")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not re.match(r"^[A-Za-z0-9_-]+$", args.voice):
        raise SystemExit("--voice: letters, digits, - and _ only")
    store = Store(args.data / args.voice, [p if p.is_absolute() else ROOT / p for p in args.script])
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
    done = len(store.durations)
    print(f"Ovoz studiyasi: http://127.0.0.1:{args.port}  ({done}/{len(store.script)} recorded, "
          f"saving to {store.dir})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
