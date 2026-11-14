"""Tilsim API client: standard library only, Python 3.9+. Copy this one file into any project.

    from tilsim import Tilsim
    t = Tilsim()                                    # key from TILSIM_API_KEY or .secrets/tilsim-api-key.txt
    t.save("Assalomu alaykum!", "salom.mp3")        # short text -> file (format from the file extension)
    t.save_long(open("kitob.txt").read(), "kitob.ogg", on_progress=print)   # long text through a job

    python clients/tilsim.py "Narxi 85 000 so‘m." -o narx.mp3
    python clients/tilsim.py --file matn.txt -o matn.ogg       # long texts switch to a job automatically
    python clients/tilsim.py --usage

The server's address changes every time start-tilsim.bat opens a new tunnel. The client reads the current one
from the same gist the website uses. Set TILSIM_API_URL to pin an address (e.g. http://127.0.0.1:8790 on this PC).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GIST_API = "https://api.github.com/gists/5801b83645819ab8d22340b5b60798f2"
LOCAL = "http://127.0.0.1:8790"
SYNC_MAX_CHARS = 600
KEY_FILE = Path(__file__).resolve().parents[1] / ".secrets" / "tilsim-api-key.txt"


class TilsimError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: float | None = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after


def _alive(url: str, timeout: float = 6) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(f"{url}/v1/health", headers={"User-Agent": "tilsim-client"}),
                                    timeout=timeout) as r:
            return r.status == 200
    except OSError:
        return False


def find_base_url(timeout: float = 10) -> str:
    """TILSIM_API_URL; else the owner's laptop if it answers (free); else the cloud copy (always on, uses Modal
    credit; its first request after a pause wakes the GPU, up to ~1 min); else this PC."""
    if os.environ.get("TILSIM_API_URL"):
        return os.environ["TILSIM_API_URL"].rstrip("/")
    try:
        req = urllib.request.Request(GIST_API, headers={"Accept": "application/vnd.github+json", "User-Agent": "tilsim-client"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            cfg = json.loads(json.loads(r.read())["files"]["tilsim-api.json"]["content"])
        if cfg.get("laptop") and _alive(cfg["laptop"].rstrip("/")):
            return cfg["laptop"].rstrip("/")
        if cfg.get("cloud"):
            return cfg["cloud"].rstrip("/")
        if cfg.get("api"):
            return cfg["api"].rstrip("/")
    except (OSError, ValueError, KeyError):
        pass
    return LOCAL


class Tilsim:
    def __init__(self, api_key: str | None = None, base_url: str | None = None, timeout: float = 170):
        key = api_key or os.environ.get("TILSIM_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else None)
        if not key:
            raise ValueError("API kalit yo‘q: Tilsim(api_key=...) bering yoki TILSIM_API_KEY ni o‘rnating.")
        self.key, self.timeout = key, timeout
        self.base = (base_url or find_base_url()).rstrip("/")

    # ------------------------------------------------------------ low level
    def _call(self, method: str, path: str, body: dict | None = None, retries: int = 2):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"X-API-Key": self.key, "Content-Type": "application/json",
                                              "User-Agent": "tilsim-client"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read())
            except ValueError:
                err = {"code": "http_error", "error": e.reason}
            retry = float(e.headers.get("Retry-After") or 0) or None
            if e.code in (429, 503) and retries > 0 and retry and retry <= 60:
                time.sleep(retry)  # rate limit or busy queue: wait as the server asks, then try again
                return self._call(method, path, body, retries - 1)
            raise TilsimError(e.code, err.get("code", "error"), err.get("error", str(e.reason)), retry) from None
        except urllib.error.URLError as e:
            raise TilsimError(0, "offline", f"Server bilan aloqa yo‘q ({self.base}): {e.reason}. "
                                            "start-tilsim.bat ishlab turibdimi?") from None

    def _json(self, method: str, path: str, body: dict | None = None) -> dict:
        return json.loads(self._call(method, path, body)[0])

    # ------------------------------------------------------------ API
    def health(self) -> dict:
        return self._json("GET", "/v1/health")

    def voices(self) -> list[dict]:
        return self._json("GET", "/v1/voices")["voices"]

    def usage(self) -> dict:
        return self._json("GET", "/v1/usage")

    def normalize(self, text: str) -> dict:
        return self._json("POST", "/v1/normalize", {"text": text})

    def tts(self, text: str, format: str = "mp3", voice: str = "asoschi", seed: int = 0) -> bytes:  # noqa: A002
        """Short text (up to 600 characters) -> audio bytes."""
        audio, _ = self._call("POST", "/v1/tts", {"text": text, "format": format, "voice": voice, "seed": seed})
        return audio

    def job(self, text: str, format: str = "mp3", voice: str = "asoschi", seed: int = 0) -> dict:  # noqa: A002
        return self._json("POST", "/v1/jobs", {"text": text, "format": format, "voice": voice, "seed": seed})

    def wait(self, job: dict, poll: float = 2.0, on_progress=None) -> bytes:
        while True:
            view = self._json("GET", job["status_url"])
            if on_progress:
                on_progress(view["progress"]["done"], view["progress"]["total"])
            if view["status"] == "done":
                return self._call("GET", job["audio_url"])[0]
            if view["status"] == "failed":
                raise TilsimError(500, "failed", view.get("error", "vazifa bajarilmadi"))
            time.sleep(poll)

    def speak(self, text: str, format: str = "mp3", on_progress=None, **kw) -> bytes:  # noqa: A002
        """Any length: short texts in one call, long ones through a job."""
        if len(text) <= SYNC_MAX_CHARS:
            return self.tts(text, format, **kw)
        return self.wait(self.job(text, format, **kw), on_progress=on_progress)

    def save(self, text: str, path: str | Path, on_progress=None, **kw) -> Path:
        path = Path(path)
        fmt = path.suffix.lstrip(".").lower() or "mp3"
        path.write_bytes(self.speak(text, fmt, on_progress=on_progress, **kw))
        return path

    save_long = save


def main() -> None:
    ap = argparse.ArgumentParser(description="Tilsim: o‘zbekcha matn -> ovoz")
    ap.add_argument("text", nargs="?", help="matn (yoki --file)")
    ap.add_argument("--file", type=Path, help="matn fayli (UTF-8)")
    ap.add_argument("-o", "--out", type=Path, default=Path("tilsim.mp3"), help="natija: .mp3, .wav yoki .ogg")
    ap.add_argument("--seed", type=int, default=0, help="boshqa variant uchun boshqa son")
    ap.add_argument("--normalize", action="store_true", help="faqat qanday o‘qilishini ko‘rsat")
    ap.add_argument("--usage", action="store_true", help="shu oydagi foydalanish")
    ap.add_argument("--url", help="server manzili (odatda gistdan o‘zi topiladi)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    t = Tilsim(base_url=args.url)
    try:
        if args.usage:
            print(json.dumps(t.usage(), ensure_ascii=False, indent=1))
            return
        text = args.file.read_text(encoding="utf-8") if args.file else args.text
        if not text:
            ap.error("matn bering yoki --file")
        if args.normalize:
            print(json.dumps(t.normalize(text), ensure_ascii=False, indent=1))
            return
        t0 = time.time()
        path = t.save(text, args.out, seed=args.seed,
                      on_progress=lambda d, n: print(f"\r{d}/{n} gap", end="", flush=True))
        print(f"\n{path} tayyor ({path.stat().st_size // 1024} KB, {time.time() - t0:.0f} s). Server: {t.base}")
    except TilsimError as e:
        sys.exit(f"Xato: {e.message} [{e.code}]")


if __name__ == "__main__":
    main()
