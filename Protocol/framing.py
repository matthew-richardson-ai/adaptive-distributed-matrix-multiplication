import json
import socket
import struct
from typing import Any, Dict, Tuple

from Protocol.constants import (
    MAX_HEADER_BYTES,
    MAX_PAYLOAD_BYTES,
    PROTOCOL_MAGIC,
    PROTOCOL_VERSION,
)

_PREFIX = struct.Struct("!IQ")


class ProtocolError(Exception):
    pass


def recv_exact(sock: socket.socket, size: int) -> bytes:
    if size < 0:
        raise ProtocolError("Negative receive size.")
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Socket closed before the full frame was received.")
        data.extend(chunk)
    return bytes(data)


def send_message(
    sock: socket.socket,
    message_type: str,
    metadata: Dict[str, Any] | None = None,
    payload: bytes = b"",
) -> None:
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise TypeError("payload must be bytes-like")

    header = {
        "magic": PROTOCOL_MAGIC,
        "version": PROTOCOL_VERSION,
        "type": message_type,
    }
    if metadata:
        header.update(metadata)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    payload_bytes = bytes(payload)

    if len(header_bytes) > MAX_HEADER_BYTES:
        raise ProtocolError("Header exceeds configured maximum size.")
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        raise ProtocolError("Payload exceeds configured maximum size.")

    sock.sendall(_PREFIX.pack(len(header_bytes), len(payload_bytes)))
    sock.sendall(header_bytes)
    if payload_bytes:
        sock.sendall(payload_bytes)


def recv_message(sock: socket.socket) -> Tuple[Dict[str, Any], bytes]:
    prefix = recv_exact(sock, _PREFIX.size)
    header_len, payload_len = _PREFIX.unpack(prefix)

    if header_len <= 0 or header_len > MAX_HEADER_BYTES:
        raise ProtocolError(f"Invalid header length: {header_len}")
    if payload_len < 0 or payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"Invalid payload length: {payload_len}")

    header_bytes = recv_exact(sock, header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("Header is not valid UTF-8 JSON.") from exc

    if header.get("magic") != PROTOCOL_MAGIC:
        raise ProtocolError("Invalid protocol magic.")
    if header.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Unsupported protocol version: {header.get('version')}"
        )
    if "type" not in header:
        raise ProtocolError("Message type is missing.")

    payload = recv_exact(sock, payload_len) if payload_len else b""
    return header, payload
