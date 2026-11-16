"""Read single members of a huge remote zip without downloading all of it (HTTP Range requests).

    python tools/remote_zip.py URL --list                      # table of contents summary (~6 MB for 55k entries)
    python tools/remote_zip.py URL --get names.txt --out DIR --max-mb 80

Used for KazEmoTTS (issai/KazEmoTTS, CC BY 4.0, one 10.5 GB zip): only the few clips the emotion lab needs are fetched.
Every byte read is counted, and the run stops before passing --max-mb.
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path


class RangeFile(io.RawIOBase):
    """A seekable, read-only file over HTTP Range requests, with a byte budget."""

    def __init__(self, url: str, max_bytes: int):
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.url = r.geturl()          # the signed CDN address after redirects
            self.size = int(r.headers["Content-Length"])
        self.pos, self.read_bytes, self.max_bytes = 0, 0, max_bytes

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if self.read_bytes + n > self.max_bytes:
            raise RuntimeError(f"byte budget reached ({self.read_bytes / 1e6:.1f} MB read, next read {n / 1e6:.1f} MB)")
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{self.pos + n - 1}"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 3:
                    raise
        self.pos += len(data)
        self.read_bytes += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--names-out", type=Path, help="with --list: write every member name here")
    ap.add_argument("--get", type=Path, help="file with member names to fetch, one per line")
    ap.add_argument("--out", type=Path, default=Path("."))
    ap.add_argument("--max-mb", type=float, default=80)
    args = ap.parse_args()
    f = RangeFile(args.url, int(args.max_mb * 1e6))
    z = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 16))
    infos = z.infolist()
    print(f"{len(infos)} members, archive {f.size / 1e9:.2f} GB; read so far {f.read_bytes / 1e6:.1f} MB", flush=True)
    if args.list:
        tops = Counter("/".join(i.filename.split("/")[:3]) for i in infos if not i.is_dir())
        for k, v in sorted(tops.items())[:80]:
            print(f"  {v:6d}  {k}")
        if args.names_out:
            args.names_out.write_text("\n".join(f"{i.filename}\t{i.file_size}" for i in infos), encoding="utf-8")
    if args.get:
        want = [ln.split("\t")[0] for ln in args.get.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for name in want:
            target = args.out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))
        print(f"fetched {len(want)} members into {args.out}; {f.read_bytes / 1e6:.1f} MB read in total", flush=True)


if __name__ == "__main__":
    sys.exit(main())
