"""A small WebSocket (RFC 6455) layer for the stdlib HTTP server in engine/serve.py. Standard library only.

Server side only: no extensions, no fragmentation on send. Client frames must be masked (all browsers and
WebSocket libraries do this). Text and binary messages, ping/pong and close are handled.
"""
from __future__ import annotations

import base64
import hashlib
import struct
import threading

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
OP_CONT, OP_TEXT, OP_BIN, OP_CLOSE, OP_PING, OP_PONG = 0x0, 0x1, 0x2, 0x8, 0x9, 0xA
MAX_MESSAGE = 1 << 20  # 1 MB per client message: text arrives in small pieces


class Closed(Exception):
    pass


def accept_key(client_key: str) -> str:
    return base64.b64encode(hashlib.sha1((client_key + GUID).encode()).digest()).decode()


class WebSocket:
    def __init__(self, rfile, wfile):
        self.rfile, self.wfile = rfile, wfile
        self.send_lock = threading.Lock()
        self.closed = False

    # ------------------------------------------------------------ sending
    def _send_frame(self, opcode: int, payload: bytes) -> None:
        n = len(payload)
        if n < 126:
            header = struct.pack("!BB", 0x80 | opcode, n)
        elif n < 1 << 16:
            header = struct.pack("!BBH", 0x80 | opcode, 126, n)
        else:
            header = struct.pack("!BBQ", 0x80 | opcode, 127, n)
        with self.send_lock:
            if self.closed:
                raise Closed()
            try:
                self.wfile.write(header + payload)
                self.wfile.flush()
            except OSError:
                self.closed = True
                raise Closed() from None

    def send_text(self, text: str) -> None:
        self._send_frame(OP_TEXT, text.encode("utf-8"))

    def send_bytes(self, data: bytes) -> None:
        self._send_frame(OP_BIN, data)

    def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed:
            return
        try:
            self._send_frame(OP_CLOSE, struct.pack("!H", code) + reason.encode("utf-8")[:120])
        except Closed:
            pass
        self.closed = True

    # ------------------------------------------------------------ receiving
    def _read(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if data is None or len(data) < n:
            self.closed = True
            raise Closed()
        return data

    def recv(self) -> tuple[str, bytes | str]:
        """Next complete message as ("text", str) or ("binary", bytes). Raises Closed when the peer leaves."""
        parts, kind = [], None
        while True:
            b1, b2 = self._read(2)
            fin, opcode = b1 & 0x80, b1 & 0x0F
            masked, n = b2 & 0x80, b2 & 0x7F
            if n == 126:
                n = struct.unpack("!H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack("!Q", self._read(8))[0]
            if n > MAX_MESSAGE:
                self.close(1009, "message too big")
                raise Closed()
            mask = self._read(4) if masked else b"\0\0\0\0"
            payload = bytearray(self._read(n))
            for i in range(n):
                payload[i] ^= mask[i % 4]
            if opcode == OP_PING:
                self._send_frame(OP_PONG, bytes(payload))
                continue
            if opcode == OP_PONG:
                continue
            if opcode == OP_CLOSE:
                self.close()
                raise Closed()
            if opcode in (OP_TEXT, OP_BIN):
                kind = "text" if opcode == OP_TEXT else "binary"
                parts = [bytes(payload)]
            elif opcode == OP_CONT:
                parts.append(bytes(payload))
            if fin and kind:
                data = b"".join(parts)
                if sum(map(len, parts)) > MAX_MESSAGE:
                    self.close(1009, "message too big")
                    raise Closed()
                return (kind, data.decode("utf-8", "replace") if kind == "text" else data)
