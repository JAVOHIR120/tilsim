"""Tilsim voice API: Uzbek text -> speech in the owner's voice, on this laptop's GPU.

    E:/tools/voiceclone/.venv/Scripts/python.exe engine/serve.py      # http://127.0.0.1:8790 (start-tilsim.bat adds the tunnel)

Two kinds of callers:
  - the website's text box: no key, browser Origin must be the Tilsim site, small public limits;
  - your own apps with an API key (engine/apikeys.py): bigger texts, OGG/WAV, jobs for long texts, usage numbers.

Endpoints (full guide with examples: docs/API.md)
    GET  /v1/health                 public: is the model loaded, queue length
    GET  /v1/voices                 the voices you can use
    POST /v1/tts                    {"text", "voice"?, "format"?: mp3|wav|ogg, "seed"?} -> audio bytes (short texts)
    POST /v1/normalize              {"text"} -> how Tilsim will read it (numbers as words, sentences); no GPU
    POST /v1/jobs                   same body as /v1/tts, for long texts -> 202 {"id", ...}
    GET  /v1/jobs/{id}              status + progress (sentences done / total)
    GET  /v1/jobs/{id}/audio        the finished audio
    GET  /v1/usage                  this key's requests, characters and audio seconds this month

One GPU worker makes one sentence at a time (4 GB card). Requests wait in a priority queue:
API keys first, then the website, then background jobs. A sentence whose speed is far off for its length
(cut short or runaway) is made again with another seed, up to 3 tries.
Texts are never written to disk. The log keeps time, key name or a hashed visitor id, length and duration.
"""
from __future__ import annotations

import argparse
import collections
import ctypes
import hashlib
import itertools
import json
import queue
import secrets
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
from apikeys import KeyStore  # noqa: E402
from uznorm import normalize, split_sentences  # noqa: E402

# website (no key)
DEMO_MAX_CHARS = 300
DEMO_PER_IP = [(600, 8), (86400, 60)]   # (window seconds, max requests) per visitor
DEMO_WAITING_MAX = 4
ORIGINS = {"https://javohir120.github.io", "http://localhost:8780", "http://127.0.0.1:8780"}
# API keys
SYNC_MAX_CHARS = 600      # /v1/tts must answer within the tunnel's 100 s; longer texts go to /v1/jobs
MAX_SPOKEN_RATIO = 2.5    # numbers grow when written as words; cap the spoken text at this x the limit
ACTIVE_JOBS_PER_KEY = 3
JOB_KEEP_SECONDS = 3600
# shared
QUEUE_MAX = 12
CACHE_SIZE = 32         # raw float audio: ~1 MB per 10 s clip
PRIORITY = {"key": 0, "demo": 1, "job": 2}
FORMATS = {"mp3": "audio/mpeg", "wav": "audio/wav", "ogg": "audio/ogg"}
VOICES = {"asoschi": {"profile": "ovozim", "name": "Asoschi", "language": "uz", "gender": "male",
                      "description": "Loyiha asoschisining ovozi: tinch, aniq, iliq."}}
DEFAULT_VOICE = "asoschi"
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


def encode(audio, sr: int, fmt: str) -> bytes:
    """float32 mono -> MP3 / WAV / OGG-Opus, all at -16 LUFS (the site's demo clips use the same loudness).
    OGG-Opus mono is what Telegram's sendVoice wants."""
    codec = {"mp3": ["-ar", str(sr), "-codec:a", "libmp3lame", "-b:a", "96k", "-f", "mp3"],
             "wav": ["-ar", str(sr), "-codec:a", "pcm_s16le", "-f", "wav"],
             "ogg": ["-ar", "48000", "-codec:a", "libopus", "-b:a", "48k", "-f", "ogg"]}[fmt]
    # loudnorm crashes ffmpeg on pure digital silence (seen 2026-11-15, exit 0xC0000409), so skip it there
    level = ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"] if audio.size and float(abs(audio).max()) > 1e-4 else []
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "2", "-f", "f32le", "-ar", str(sr), "-ac", "1",
           "-i", "-", *level, *codec, "-"]
    return subprocess.run(cmd, input=audio.astype("float32").tobytes(), capture_output=True, check=True).stdout


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


class Voice:
    """The loaded model plus one voice profile; used by the single worker thread only."""

    def __init__(self, profile_name: str):
        import numpy as np
        import torch
        import uz_tts

        self.np, self.torch, self.uz = np, torch, uz_tts
        profile = json.loads((ROOT / "data" / "voices" / f"{profile_name}.json").read_text(encoding="utf-8"))
        self.params = {**uz_tts.PARAMS, **profile.get("params", {})}
        t0 = time.time()
        self.engine = uz_tts.load_engine()
        self.sr = self.engine.sr
        self.engine.ENC_COND_LEN = int(float(profile.get("prompt_sec", 6)) * 16000)
        self.engine.prepare_conditionals(profile["ref"], exaggeration=self.params["exaggeration"])
        self.prefix = uz_tts.Prefix(self.engine, profile["ref"], profile["prefix_end"], profile["prefix_text"])
        vram = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
        log(f"model ready on {self.engine.device} in {time.time() - t0:.0f} s, VRAM {vram:.2f} GB")

    def sentence(self, text: str, seed: int):
        """One spoken sentence (already normalised). Retake if its speed is implausible for its length."""
        from uznorm import for_model
        letters = max(1, sum(ch.isalpha() for ch in text))
        best, best_gap = None, 1e9
        for attempt in range(3):
            self.torch.manual_seed(seed + 1000 * attempt)  # attempt 0 = the same seed engine/uz_tts.speak uses
            wav = self.uz.generate_continuing(self.engine, self.prefix, for_model(text), **self.params)
            wav = self.uz.trim(wav, self.sr)
            if self.torch.cuda.is_available():
                self.torch.cuda.empty_cache()  # 4 GB card: keep cached blocks from piling up
            rate = letters / max(len(wav) / self.sr, 1e-3)  # the owner speaks ~12-14 letters/s
            if 7 <= rate <= 22:
                return wav
            if abs(rate - 13) < best_gap:
                best, best_gap = wav, abs(rate - 13)
            log(f"retake: {rate:.1f} letters/s on try {attempt + 1}")
        return best

    def synth(self, sentences: list[str], seed: int = 0, progress=None):
        pause = self.np.zeros(int(0.25 * self.sr), self.np.float32)
        pieces = []
        for i, s in enumerate(sentences):
            pieces += [self.sentence(s, seed + i), pause]
            if progress:
                progress(i + 1)
        return self.np.concatenate(pieces[:-1]) if pieces else self.np.zeros(0, self.np.float32)


class Service:
    def __init__(self, keys: KeyStore):
        self.keys = keys
        self.voice: Voice | None = None
        self.jobs_q: queue.PriorityQueue = queue.PriorityQueue(maxsize=QUEUE_MAX)
        self.seq = itertools.count()
        self.busy = 0
        self.demo_waiting = 0
        self.lock = threading.Lock()
        self.ip_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
        self.cache: collections.OrderedDict = collections.OrderedDict()
        self.jobs: dict[str, dict] = {}
        threading.Thread(target=self._worker, daemon=True).start()

    # ---------------------------------------------------------------- GPU worker
    def _load(self) -> None:
        self.voice = Voice(VOICES[DEFAULT_VOICE]["profile"])
        self.voice.synth(["Salom."])  # warm-up: first CUDA kernels are slow
        log("warm-up done, accepting requests")

    def _worker(self) -> None:
        try:
            self._load()
        except Exception as e:  # noqa: BLE001
            log(f"model failed to load: {e!r}")
            return
        while True:
            _, _, kind, payload, fut = self.jobs_q.get()
            if kind == "demo":
                with self.lock:
                    self.demo_waiting -= 1
            self.busy = 1
            try:
                if kind == "job":
                    self._run_job(payload)
                else:
                    fut.set_result(self._make(payload))
            except Exception as e:  # noqa: BLE001
                if fut:
                    fut.set_exception(e)
            finally:
                self.busy = 0

    def _make(self, req: dict, progress=None) -> dict:
        """GPU work only: raw float audio. Encoding happens on the caller's thread, so the GPU never waits for ffmpeg."""
        audio = self.voice.synth(req["sentences"], req["seed"], progress)
        return {"spoken": req["spoken"], "pcm": audio, "sr": self.voice.sr, "seconds": round(len(audio) / self.voice.sr, 2)}

    def _run_job(self, job: dict) -> None:
        job["status"], job["started"] = "working", time.time()

        def progress(done):
            job["done"] = done
        try:
            raw = self._make(job["req"], progress)
            result = {"audio": encode(raw["pcm"], raw["sr"], job["req"]["format"]), "seconds": raw["seconds"]}
            job.update(status="done", result=result, seconds=result["seconds"], finished=time.time())
            self.keys.record(job["key"], len(job["req"]["text"]), result["seconds"])
            log(f"job {job['id']} ({job['key']['name']}) {len(job['req']['text'])} chars -> "
                f"{result['seconds']:.1f} s audio in {job['finished'] - job['started']:.0f} s")
        except Exception as e:  # noqa: BLE001
            job.update(status="failed", error="Ovoz yaratishda xato bo‘ldi.", finished=time.time())
            log(f"job {job['id']} failed: {e!r}")

    # ---------------------------------------------------------------- helpers
    def prepare(self, body: dict, max_chars: int) -> dict:
        text = str(body.get("text", "")).strip()
        if not text:
            raise ApiError(400, "empty_text", "Matn yozing (\"text\").")
        if len(text) > max_chars:
            raise ApiError(400, "too_long", f"Matn {max_chars} belgidan oshmasin.")
        voice = str(body.get("voice") or DEFAULT_VOICE).lower()
        if voice not in VOICES:
            raise ApiError(400, "unknown_voice", f"Bunday ovoz yo‘q: {voice}. Ro‘yxat: /v1/voices")
        fmt = str(body.get("format") or "mp3").lower()
        if fmt not in FORMATS:
            raise ApiError(400, "bad_format", "format: mp3, wav yoki ogg.")
        try:
            seed = int(body.get("seed", 0))
        except (TypeError, ValueError):
            raise ApiError(400, "bad_seed", "seed butun son bo‘lsin.") from None
        spoken = normalize(text)
        if len(spoken) > max_chars * MAX_SPOKEN_RATIO:
            raise ApiError(400, "too_long", "Matn so‘zga aylantirilganda juda uzun bo‘ldi (raqamlar ko‘p). Qisqartiring.")
        return {"text": text, "voice": voice, "format": fmt, "seed": seed, "spoken": spoken,
                "sentences": split_sentences(spoken, max_chars=200)}

    def demo_wait(self, who: str) -> int:
        now = time.time()
        with self.lock:
            q = self.ip_hits[who]
            while q and now - q[0] > DEMO_PER_IP[-1][0]:
                q.popleft()
            for window, limit in DEMO_PER_IP:
                recent = [t for t in q if now - t <= window]
                if len(recent) >= limit:
                    return int(window - (now - recent[0])) + 1
            q.append(now)
            return 0

    def submit(self, kind: str, payload, fut: Future | None) -> None:
        with self.lock:
            if kind == "demo" and self.demo_waiting >= DEMO_WAITING_MAX:
                raise ApiError(503, "busy", "Hozir navbat to‘la. Birozdan keyin urinib ko‘ring.", 20)
            try:
                self.jobs_q.put_nowait((PRIORITY[kind], next(self.seq), kind, payload, fut))
            except queue.Full:
                raise ApiError(503, "busy", "Hozir navbat to‘la. Birozdan keyin urinib ko‘ring.", 20) from None
            if kind == "demo":
                self.demo_waiting += 1

    def cached(self, req: dict):
        """Raw audio is cached per text+voice+seed, so asking again in another format needs no GPU work."""
        key = hashlib.sha256(json.dumps([req["text"], req["voice"], req["seed"]]).encode()).hexdigest()
        return key, self.cache.get(key)

    def remember(self, key: str, result: dict) -> None:
        self.cache[key] = result
        self.cache.move_to_end(key)
        while len(self.cache) > CACHE_SIZE:
            self.cache.popitem(last=False)

    def sweep_jobs(self) -> None:
        now = time.time()
        for jid in [j for j, v in self.jobs.items() if v.get("finished") and now - v["finished"] > JOB_KEEP_SECONDS]:
            self.jobs.pop(jid, None)


class Handler(BaseHTTPRequestHandler):
    service: Service
    server_version = "Tilsim/1"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # our own log lines only
        pass

    # ---------------------------------------------------------------- plumbing
    def _cors(self) -> None:
        origin = self.headers.get("Origin", "")
        if origin in ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Expose-Headers",
                             "X-Tilsim-Seconds, X-Tilsim-Spoken, X-Tilsim-Chars, Retry-After")

    def _send(self, status: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, obj: dict, extra: dict | None = None) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", extra)

    def _error(self, e: ApiError) -> None:
        self._json(e.status, {"error": e.message, "code": e.code},
                   {"Retry-After": str(e.retry_after)} if e.retry_after else None)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 64_000:
            raise ApiError(413, "too_large", "So‘rov juda katta.")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, UnicodeDecodeError):
            raise ApiError(400, "bad_json", "JSON noto‘g‘ri.") from None
        if not isinstance(body, dict):
            raise ApiError(400, "bad_json", "JSON obyekt yuboring: {\"text\": \"...\"}.")
        return body

    def _caller(self) -> tuple[str, dict | None]:
        """("key", record) for a valid API key, ("demo", None) for the website. Anything else is refused."""
        raw = self.headers.get("X-API-Key") or ""
        auth = self.headers.get("Authorization") or ""
        if not raw and auth.lower().startswith("bearer "):
            raw = auth[7:].strip()
        if raw:
            rec = self.service.keys.find(raw)
            if rec is None:
                raise ApiError(401, "bad_key", "API kalit noto‘g‘ri yoki o‘chirilgan.")
            return "key", rec
        if self.headers.get("Origin", "") in ORIGINS:
            return "demo", None
        raise ApiError(401, "no_key", "API kalit kerak: X-API-Key sarlavhasida yuboring.")

    def _visitor(self) -> str:
        ip = self.headers.get("CF-Connecting-IP") or self.client_address[0]
        return hashlib.sha256(f"tilsim:{ip}".encode()).hexdigest()[:10]

    def _ready(self) -> None:
        if self.service.voice is None:
            raise ApiError(503, "loading", "Ovoz modeli yuklanmoqda, bir daqiqadan keyin urinib ko‘ring.", 30)

    def _key_limits(self, rec: dict, chars: int) -> None:
        wait = self.service.keys.rate_wait(rec)
        if wait:
            raise ApiError(429, "rate_limited", f"Daqiqasiga {rec['per_minute']} so‘rovdan oshdi. {wait} s kuting.", wait)
        left = self.service.keys.quota_left(rec)
        if left is not None and chars > left:
            raise ApiError(402, "quota_exceeded", f"Oylik limit tugadi: {left} belgi qoldi.")

    # ---------------------------------------------------------------- routes
    def do_OPTIONS(self):  # noqa: N802 (CORS preflight for the website)
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        if self.headers.get("Access-Control-Request-Private-Network"):
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):  # noqa: N802
        s = self.service
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        try:
            if path in ("/", "/v1/health"):
                return self._json(200, {"ok": True, "ready": s.voice is not None, "busy": s.busy,
                                        "queue": s.jobs_q.qsize(), "max_chars": DEMO_MAX_CHARS, "voice": "Asoschi",
                                        "docs": "https://github.com/JAVOHIR120/tilsim/blob/main/docs/API.md"})
            if path == "/v1/voices":
                return self._json(200, {"voices": [{"id": vid, "sample_rate": 24000,
                                                    **{k: v for k, v in info.items() if k != "profile"}}
                                                   for vid, info in VOICES.items()], "default": DEFAULT_VOICE})
            kind, rec = self._caller()
            if kind != "key":
                raise ApiError(401, "no_key", "Bu manzil faqat API kalit bilan ishlaydi.")
            if path == "/v1/usage":
                return self._json(200, s.keys.usage(rec))
            parts = path.split("/")
            if len(parts) in (4, 5) and parts[:3] == ["", "v1", "jobs"]:
                job = s.jobs.get(parts[3])
                if not job or job["key"]["id"] != rec["id"]:
                    raise ApiError(404, "no_job", "Bunday vazifa yo‘q (yoki 1 soatdan eski).")
                if len(parts) == 4:
                    return self._json(200, self._job_view(job))
                if parts[4] == "audio":
                    if job["status"] != "done":
                        raise ApiError(409, "not_ready", f"Vazifa hali tayyor emas: {job['status']}.")
                    r = job["result"]
                    return self._send(200, r["audio"], FORMATS[job["req"]["format"]],
                                      {"X-Tilsim-Seconds": f"{r['seconds']:.2f}",
                                       "Content-Disposition": f'attachment; filename="tilsim-{job["id"]}.{job["req"]["format"]}"'})
            raise ApiError(404, "not_found", "Bunday manzil yo‘q. Qo‘llanma: docs/API.md")
        except ApiError as e:
            self._error(e)

    def do_POST(self):  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        try:
            # read the body first: on a kept-alive connection an unread body would corrupt the next request
            body = self._body()
        except ApiError as e:
            self.close_connection = True
            return self._error(e)
        try:
            if path == "/v1/tts":
                return self._tts(body)
            if path == "/v1/jobs":
                return self._new_job(body)
            if path == "/v1/normalize":
                kind, _ = self._caller()
                req = self.service.prepare(body, 5000 if kind == "key" else DEMO_MAX_CHARS)
                return self._json(200, {"text": req["text"], "spoken": req["spoken"], "sentences": req["sentences"]})
            raise ApiError(404, "not_found", "Bunday manzil yo‘q. Qo‘llanma: docs/API.md")
        except ApiError as e:
            self._error(e)

    def _tts(self, body: dict) -> None:
        s = self.service
        kind, rec = self._caller()
        self._ready()
        req = s.prepare(body, SYNC_MAX_CHARS if kind == "key" else DEMO_MAX_CHARS)
        if kind == "demo":
            req["format"] = "mp3"
        who = rec["name"] if rec else self._visitor()
        cache_key, result = s.cached(req)
        hit = result is not None
        t0 = time.time()
        if not hit:
            if kind == "key":
                self._key_limits(rec, len(req["text"]))
            else:
                wait = s.demo_wait(who)
                if wait:
                    raise ApiError(429, "rate_limited", f"Juda ko‘p so‘rov. {wait} soniyadan keyin urinib ko‘ring.", wait)
            fut: Future = Future()
            s.submit(kind, req, fut)
            try:
                result = fut.result(timeout=170)
            except Exception as e:  # noqa: BLE001
                log(f"{who} failed: {e!r}")
                raise ApiError(500, "failed", "Ovoz yaratishda xato bo‘ldi. Qayta urinib ko‘ring.") from None
            s.remember(cache_key, result)
            if rec:
                s.keys.record(rec, len(req["text"]), result["seconds"])
        try:
            audio = encode(result["pcm"], result["sr"], req["format"])
        except Exception as e:  # noqa: BLE001
            log(f"{who} encode failed: {e!r}")
            raise ApiError(500, "failed", "Audio faylga o‘girishda xato bo‘ldi.") from None
        log(f"{who} {len(req['text'])} chars {req['format']} -> {result['seconds']:.1f} s audio in "
            f"{time.time() - t0:.1f} s{' (cache)' if hit else ''}")
        self._send(200, audio, FORMATS[req["format"]],
                   {"X-Tilsim-Seconds": f"{result['seconds']:.2f}", "X-Tilsim-Chars": str(len(req["text"])),
                    "X-Tilsim-Spoken": urllib.parse.quote(result["spoken"])})

    def _new_job(self, body: dict) -> None:
        s = self.service
        kind, rec = self._caller()
        if kind != "key":
            raise ApiError(401, "no_key", "Vazifalar faqat API kalit bilan ishlaydi.")
        self._ready()
        req = s.prepare(body, rec.get("max_chars", 5000))
        s.sweep_jobs()
        active = sum(1 for j in s.jobs.values() if j["key"]["id"] == rec["id"] and j["status"] in ("queued", "working"))
        if active >= ACTIVE_JOBS_PER_KEY:
            raise ApiError(429, "too_many_jobs", f"Bir vaqtda {ACTIVE_JOBS_PER_KEY} tadan ko‘p vazifa bo‘lmaydi.", 30)
        self._key_limits(rec, len(req["text"]))
        job = {"id": "job_" + secrets.token_hex(6), "key": rec, "req": req, "status": "queued", "done": 0,
               "total": len(req["sentences"]), "created": time.time()}
        s.submit("job", job, None)
        s.jobs[job["id"]] = job
        log(f"job {job['id']} queued ({rec['name']}): {len(req['text'])} chars, {job['total']} sentences")
        self._json(202, self._job_view(job))

    @staticmethod
    def _job_view(job: dict) -> dict:
        view = {"id": job["id"], "status": job["status"], "format": job["req"]["format"],
                "progress": {"done": job["done"], "total": job["total"]},
                "status_url": f"/v1/jobs/{job['id']}", "audio_url": f"/v1/jobs/{job['id']}/audio"}
        if job["status"] == "done":
            view["seconds"] = job["seconds"]
        if job["status"] == "failed":
            view["error"] = job.get("error")
        return view


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1: the tunnel reaches it locally")
    args = ap.parse_args()
    set_priority_above_normal()
    keys = KeyStore()
    Handler.service = Service(keys)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log(f"listening on http://{args.host}:{args.port}, {sum(k['active'] for k in keys.data['keys'])} active API keys "
        f"(model loading in the background)")
    server.serve_forever()


if __name__ == "__main__":
    main()
