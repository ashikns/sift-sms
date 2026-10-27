"""SPMH v1 weights container (SPEC.md). Mirrored by spamcore's HeadCodec."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

MAGIC = b"SPMH"
FORMAT = 1
ENTRY = struct.Struct("<16sIIQ")
DTYPES = {1: np.dtype("<f4"), 2: np.dtype("<i4"), 3: np.dtype("u1"), 4: np.dtype("<f8")}
CODES = {v: k for k, v in DTYPES.items()}


def _align(n: int, a: int = 16) -> int:
    return (n + a - 1) // a * a


def encode(blocks: dict[str, np.ndarray]) -> bytes:
    names = list(blocks)
    header_len = 8 + ENTRY.size * len(names)
    offset = _align(header_len)
    entries, payloads = [], []
    for name in names:
        arr = np.ascontiguousarray(blocks[name])
        dt = arr.dtype.newbyteorder("<") if arr.dtype.byteorder not in ("<", "|") else arr.dtype
        code = CODES.get(np.dtype(dt.str))
        if code is None:
            raise ValueError(f"unsupported dtype {arr.dtype} for {name}")
        raw = arr.astype(DTYPES[code], copy=False).tobytes()
        bname = name.encode("ascii")
        if len(bname) > 16:
            raise ValueError(f"block name too long: {name}")
        entries.append(ENTRY.pack(bname.ljust(16, b"\0"), code, arr.size, offset))
        payloads.append((offset, raw))
        offset = _align(offset + len(raw))
    buf = bytearray(offset)
    buf[0:8] = MAGIC + struct.pack("<HH", FORMAT, len(names))
    for i, e in enumerate(entries):
        buf[8 + i * ENTRY.size: 8 + (i + 1) * ENTRY.size] = e
    for off, raw in payloads:
        buf[off: off + len(raw)] = raw
    buf += struct.pack("<I", zlib.crc32(bytes(buf)) & 0xFFFFFFFF)
    return bytes(buf)


def decode(data: bytes) -> dict[str, np.ndarray]:
    if len(data) < 12 or data[:4] != MAGIC:
        raise ValueError("not an SPMH file")
    (crc,) = struct.unpack("<I", data[-4:])
    if zlib.crc32(data[:-4]) & 0xFFFFFFFF != crc:
        raise ValueError("SPMH CRC mismatch")
    fmt, n = struct.unpack("<HH", data[4:8])
    if fmt != FORMAT:
        raise ValueError(f"unsupported SPMH format {fmt}")
    out = {}
    for i in range(n):
        name, code, count, off = ENTRY.unpack_from(data, 8 + i * ENTRY.size)
        dt = DTYPES[code]
        out[name.rstrip(b"\0").decode("ascii")] = np.frombuffer(data, dtype=dt, count=count, offset=off).copy()
    return out


def write(path: Path, blocks: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(encode(blocks))
    tmp.replace(path)


def read(path: Path) -> dict[str, np.ndarray]:
    return decode(path.read_bytes())
