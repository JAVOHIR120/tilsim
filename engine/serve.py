"""Tilsim voice API: the website's text box sends Uzbek text here and gets back an MP3 in the owner's voice.

    E:/tools/voiceclone/.venv/Scripts/python.exe engine/serve.py            # http://127.0.0.1:8790
    curl -X POST http://127.0.0.1:8790/v1/tts -H "Content-Type: application/json" \
         -d "{\"text\": \"Assalomu alaykum!\"}" -o salom.mp3

Endpoints
    GET  /v1/health  -> {"ok", "ready", "busy", "queue", "max_chars", "voice"}
    POST /v1/tts     {"text": "..."} -> audio/mpeg; X-Tilsim-Seconds, X-Tilsim-Spoken (URL-encoded) headers

One GPU worker makes one take at a time (4 GB card). Every request waits in a short queue.
When the queue is full the server answers 503, so the laptop is never overloaded.
Limits protect the owner's voice and the laptop:
  - at most MAX_CHARS characters per request;
  - per-visitor rate limits (the tunnel passes the visitor's IP in CF-Connecting-IP);
  - only the Tilsim site may call it from a browser (CORS).
The model's inaudible watermark stays on every take (generate_continuing).
Texts are never written to disk. The log keeps only time, length, duration and a hashed visitor id.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.parse
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT))

MAX_CHARS = 300            # what a visitor may type
MAX_SPOKEN = 700           # after numbers become words ("1 250 000" grows a lot)
QUEUE_MAX = 4              # waiting requests; more -> 503 "busy"
PER_IP = [(600, 8), (86400, 60)]  # (window seconds, max requests) per visitor
CACHE_SIZE = 48            # same text again -> same MP3, no GPU work
ORIGINS = {"https://javohir120.github.io", "http://localhost:8780", "http://127.0.0.1:8780"}
LOG = ROOT / "out" / "serve.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_priority_above_normal() -> None:
    """The owner wants voicelab jobs at AboveNormal (fast, and the desktop still responds)."""
    if sys.platform == "win32":
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = ctypes.c_void_p  # a 64-bit pseudo handle; the int default truncates it
        k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        if not k32.SetPriorityClass(k32.GetCurrentProcess(), 0x8000):
            log("could not raise process priority")


def to_mp3(audio, sr: int) -> bytes:
    """float32 mono -> MP3 at -16 LUFS (same loudness as the site's demo clips)."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-f", "f32le", "-ar", str(sr), "-ac", "1",
           "-i", "-", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-ar", str(sr), "-codec:a", "libmp3lame", "-b:a", "96k",
           "-f", "mp3", "-"]
    return subprocess.run(cmd, input=audio.astype("float32").tobytes(), capture_output=True, check=True).stdout


class Voice:
    """The loaded model plus the owner's voice profile; used by one worker thread only."""

    def __init__(self, profile_name: str):
        import torch
        import uz_tts

        self.torch, self.uz = torch, uz_tts
        profile = json.loads((ROOT / "data" / "voices" / f"{profile_name}.json").read_text(encoding="utf-8"))
        self.params = {**uz_tts.PARAMS, **profile.get("params", {})}
        t0 = time.time()
        self.engine = uz_tts.load_engine()
        self.engine.ENC_COND_LEN = int(float(profile.get("prompt_sec", 6)) * 16000)
        self.engine.prepare_conditionals(profile["ref"], exaggeration=self.params["exaggeration"])
        self.prefix = uz_tts.Prefix(self.engine, profile["ref"], profile["prefix_end"], profile["prefix_text"])
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        log(f"model ready on {self.engine.device} in {time.time() - t0:.0f} s, VRAM {vram:.2f} GB")

    def say(self, text: str) -> tuple[str, bytes, float]:
        """One take; a take whose length is far off for its text (cut short or runaway) is made again, up to 3 tries."""
        letters = max(1, sum(ch.isalpha() for ch in self.uz.normalize(text)))
        best = None
        for seed in (0, 1, 2):
            spoken, audio = self.uz.speak(self.engine, text, seed=seed, prefix=self.prefix, **self.params)
            seconds = len(audio) / self.engine.sr
            rate = letters / max(seconds, 1e-3)  # the owner speaks ~12-14 letters/s
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()  # 4 GB card: keep cached blocks from piling up
            if 7 <= rate <= 22:
                best = (spoken, audio, seconds)
                break
            if best is None or abs(rate - 13) < abs(letters / max(best[2], 1e-3) - 13):
                best = (spoken, audio, seconds)
            log(f"retake: seed {seed} gave {rate:.1f} letters/s")
        spoken, audio, seconds = best
        return spoken, to_mp3(audio, self.engine.sr), seconds


class Service:
    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        self.jobs: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self.voice: Voice | None = None
        self.busy = 0
        self.hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.cache: collections.OrderedDict[str, tuple[str, bytes, float]] = collections.OrderedDict()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        try:
            self.voice = Voice(self.profile_name)
            self.voice.say("Salom.")  # warm-up: first CUDA kernels are slow
            log("warm-up done, accepting requests")
        except Exception as e:  # noqa: BLE001
            log(f"model failed to load: {e!r}")
            return
        while True:
            text, fut = self.jobs.get()
            self.busy = 1
            try:
                fut.set_result(self.voice.say(text))
            except Exception as e:  # noqa: BLE001
                fut.set_exception(e)
            finally:
                self.busy = 0

    def allow(self, who: str) -> int:
        """0 if this visitor may make a request now, else seconds to wait."""
        now = time.time()
        with self.lock:
            q = self.hits[who]
            while q and now - q[0] > PER_IP[-1][0]:
                q.popleft()
            for window, limit in PER_IP:
                recent = [t for t in q if now - t <= window]
                if len(recent) >= limit:
                    return int(window - (now - recent[0])) + 1
            q.append(now)
            return 0

    def submit(self, text: str) -> Future:
        fut: Future = Future()
        self.jobs.put_nowait((text, fut))  # raises queue.Full
        return fut


class Handler(BaseHTTPRequestHandler):
    service: Service
    server_version = "Tilsim/1"

    def log_message(self, fmt, *args):  # quiet default access log; we log our own lines
        pass

    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Expose-Headers", "X-Tilsim-Seconds, X-Tilsim-Spoken, Retry-After")

    def _json(self, code: int, body: dict, extra: dict | None = None) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self._cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):  # noqa: N802 (CORS preflight)
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        if self.headers.get("Access-Control-Request-Private-Network"):
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        s = self.service
        if self.path.split("?")[0] in ("/v1/health", "/"):
            self._json(200, {"ok": True, "ready": s.voice is not None, "busy": s.busy, "queue": s.jobs.qsize(),
                             "max_chars": MAX_CHARS, "voice": "Asoschi"})
        else:
            self._json(404, {"error": "topilmadi"})

    def do_POST(self):  # noqa: N802
        if self.path.split("?")[0] != "/v1/tts":
            return self._json(404, {"error": "topilmadi"})
        s = self.service
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 8000:
                return self._json(413, {"error": "So‘rov juda katta."})
            text = str(json.loads(self.rfile.read(length) or b"{}").get("text", "")).strip()
        except (ValueError, json.JSONDecodeError):
            return self._json(400, {"error": "Noto‘g‘ri so‘rov."})
        if not text:
            return self._json(400, {"error": "Matn yozing."})
        if len(text) > MAX_CHARS:
            return self._json(400, {"error": f"Matn {MAX_CHARS} belgidan oshmasin."})
        if s.voice is None:
            return self._json(503, {"error": "Ovoz modeli yuklanmoqda, bir daqiqadan keyin urinib ko‘ring."},
                              {"Retry-After": "30"})
        from uznorm import normalize
        if len(normalize(text)) > MAX_SPOKEN:
            return self._json(400, {"error": "Matn juda uzun (raqamlar so‘zga aylanganda). Qisqaroq yozing."})

        ip = self.headers.get("CF-Connecting-IP") or self.client_address[0]
        who = hashlib.sha256(f"tilsim:{ip}".encode()).hexdigest()[:10]
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = s.cache.get(key)
        t0 = time.time()
        if cached is None:
            wait = s.allow(who)
            if wait:
                return self._json(429, {"error": f"Juda ko‘p so‘rov. {wait} soniyadan keyin urinib ko‘ring."},
                                  {"Retry-After": str(wait)})
            try:
                fut = s.submit(text)
            except queue.Full:
                return self._json(503, {"error": "Hozir navbat to‘la. Birozdan keyin urinib ko‘ring."},
                                  {"Retry-After": "20"})
            try:
                cached = fut.result(timeout=180)
            except Exception as e:  # noqa: BLE001
                log(f"{who} failed: {e!r}")
                return self._json(500, {"error": "Ovoz yaratishda xato bo‘ldi. Qayta urinib ko‘ring."})
            s.cache[key] = cached
            while len(s.cache) > CACHE_SIZE:
                s.cache.popitem(last=False)
        else:
            s.cache.move_to_end(key)
        spoken, mp3, seconds = cached
        log(f"{who} {len(text)} chars -> {seconds:.1f} s audio in {time.time() - t0:.1f} s"
            f"{' (cache)' if time.time() - t0 < 0.05 else ''}")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(mp3)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Tilsim-Seconds", f"{seconds:.2f}")
        self.send_header("X-Tilsim-Spoken", urllib.parse.quote(spoken))
        self.end_headers()
        self.wfile.write(mp3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1: the tunnel reaches it locally")
    ap.add_argument("--voice", default="ovozim")
    args = ap.parse_args()
    set_priority_above_normal()
    Handler.service = Service(args.voice)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log(f"listening on http://{args.host}:{args.port} (model loading in the background)")
    server.serve_forever()


if __name__ == "__main__":
    main()
