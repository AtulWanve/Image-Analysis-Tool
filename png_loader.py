"""Minimal stdlib-only PNG loader.

Scoped to Issue #1 acceptance criteria:
  1. Load PNG files and read image data.
  2. Handle invalid PNG files gracefully (FileNotFoundError / ValueError only).
"""

import struct
import zlib

from png_validation import validate_png_file

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
_VALID_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_MODES = {0: "L", 2: "RGB", 3: "RGB", 4: "LA", 6: "RGBA"}


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_scanlines(raw: bytes, width: int, height: int, filter_bpp: int, row_len: int) -> bytes:
    expected = height * (1 + row_len)
    if len(raw) != expected:
        raise ValueError(
            f"Corrupt PNG image data: expected {expected} decompressed bytes, got {len(raw)}."
        )
    out = bytearray(len(raw) - height)  # filter bytes removed; only used for length check
    result_rows = []
    prev = bytearray(row_len)
    pos = 0
    for _ in range(height):
        filt = raw[pos]
        pos += 1
        cur_enc = raw[pos : pos + row_len]
        pos += row_len
        if filt > 4:
            raise ValueError(f"Corrupt PNG image data: unknown filter type {filt}.")
        cur = bytearray(row_len)
        for i in range(row_len):
            a = cur[i - filter_bpp] if i >= filter_bpp else 0
            b = prev[i]
            c = prev[i - filter_bpp] if i >= filter_bpp else 0
            if filt == 0:  # None
                val = cur_enc[i]
            elif filt == 1:  # Sub
                val = (cur_enc[i] + a) & 0xFF
            elif filt == 2:  # Up
                val = (cur_enc[i] + b) & 0xFF
            elif filt == 3:  # Average
                val = (cur_enc[i] + ((a + b) >> 1)) & 0xFF
            else:  # Paeth
                val = (cur_enc[i] + _paeth_predictor(a, b, c)) & 0xFF
            cur[i] = val
        result_rows.append(bytes(cur))
        prev = cur
    _ = out  # keep intent explicit: output is rows below
    return b"".join(result_rows)


def _unpack_low_bit_rows(packed_rows: bytes, width: int, height: int, bit_depth: int) -> bytes:
    """Unpack 1/2/4-bit packed scanlines to one byte per sample."""
    per_byte = 8 // bit_depth
    mask = (1 << bit_depth) - 1
    row_packed_len = (width * bit_depth + 7) // 8
    if len(packed_rows) != height * row_packed_len:
        raise ValueError("Corrupt PNG image data: packed row length mismatch.")
    out = bytearray()
    pos = 0
    for _ in range(height):
        row = packed_rows[pos : pos + row_packed_len]
        pos += row_packed_len
        samples = []
        for byte in row:
            for k in range(per_byte):
                shift = 8 - bit_depth * (k + 1)
                samples.append((byte >> shift) & mask)
                if len(samples) == width:
                    break
            if len(samples) == width:
                break
        if len(samples) != width:
            raise ValueError("Corrupt PNG image data: cannot unpack low-bit-depth row.")
        out.extend(samples)
    return bytes(out)


def load_png(file_path: str) -> dict:
    """Load a PNG file and read image data.

    Returns dict with keys: width, height, bit_depth, color_type, mode, pixels.
    pixels is row-major bytes, top-to-bottom, without filter bytes or padding.
    Palette (color type 3) is expanded to RGB. tRNS is ignored.

    Raises:
        FileNotFoundError: file does not exist.
        ValueError: invalid PNG (extension, empty, signature, structure, data).
    """
    validate_png_file(file_path)

    try:
        with open(file_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        raise
    except OSError as e:
        raise ValueError(f"Cannot read PNG file: {e}") from e

    if len(data) < 8 or data[:8] != PNG_SIGNATURE:
        raise ValueError("Invalid PNG signature.")

    ihdr = None
    plte = None
    idat_parts = []
    seen_iend = False
    offset = 8

    try:
        while offset + 8 <= len(data):
            length = struct.unpack(">I", data[offset : offset + 4])[0]
            ctype = data[offset + 4 : offset + 8]
            offset += 8
            if offset + length + 4 > len(data):
                raise ValueError("Truncated PNG chunk: file ends unexpectedly.")
            chunk = data[offset : offset + length]
            crc_stored = struct.unpack(">I", data[offset + length : offset + length + 4])[0]
            crc_calc = zlib.crc32(ctype + chunk) & 0xFFFFFFFF
            if crc_stored != crc_calc:
                raise ValueError(f"Corrupt PNG chunk {ctype!r}: CRC mismatch.")
            offset += length + 4

            if ctype == b"IHDR":
                if ihdr is not None:
                    raise ValueError("Corrupt PNG: duplicate IHDR chunk.")
                if offset != 8 + 25:  # 8 sig + 4 len + 4 type + 13 data + 4 crc
                    raise ValueError("Corrupt PNG: IHDR must be the first chunk.")
                if length != 13:
                    raise ValueError("Corrupt PNG: IHDR must be 13 bytes.")
                ihdr = struct.unpack(">IIBBBBB", chunk)
            elif ctype == b"PLTE":
                if plte is not None:
                    raise ValueError("Corrupt PNG: duplicate PLTE chunk.")
                if length % 3 != 0 or length == 0 or length > 768:
                    raise ValueError("Corrupt PNG: invalid PLTE chunk.")
                plte = chunk
            elif ctype == b"IDAT":
                idat_parts.append(chunk)
            elif ctype == b"IEND":
                seen_iend = True
                break
            # All other chunks (including tRNS) are skipped after CRC check.
    except ValueError:
        raise
    except Exception as e:  # struct.error etc. -> graceful ValueError
        raise ValueError(f"Corrupt PNG: cannot parse chunks ({e}).") from e

    if ihdr is None:
        raise ValueError("Corrupt PNG: missing IHDR chunk.")
    if not idat_parts:
        raise ValueError("Corrupt PNG: missing IDAT chunk.")
    if not seen_iend:
        raise ValueError("Corrupt PNG: missing IEND chunk.")

    width, height, bit_depth, color_type, comp, filt, interlace = ihdr
    if width == 0 or height == 0:
        raise ValueError("Corrupt PNG: invalid image dimensions.")
    if comp != 0:
        raise ValueError("Corrupt PNG: unsupported compression method.")
    if filt != 0:
        raise ValueError("Corrupt PNG: unsupported filter method.")
    if interlace != 0:
        raise ValueError("Interlaced PNGs are not supported.")
    if color_type not in _VALID_DEPTHS or bit_depth not in _VALID_DEPTHS[color_type]:
        raise ValueError(
            f"Corrupt PNG: invalid bit depth {bit_depth} for color type {color_type}."
        )
    if color_type == 3 and plte is None:
        raise ValueError("Corrupt PNG: palette image missing PLTE chunk.")

    if bit_depth < 8:
        filter_bpp = 1
        row_len = (width * bit_depth * _CHANNELS[color_type] + 7) // 8
    elif bit_depth == 8:
        filter_bpp = _CHANNELS[color_type]
        row_len = width * filter_bpp
    else:  # 16-bit
        filter_bpp = _CHANNELS[color_type] * 2
        row_len = width * filter_bpp

    try:
        decompressed = zlib.decompress(b"".join(idat_parts))
    except zlib.error as e:
        raise ValueError(f"Corrupt PNG image data: zlib decompression failed ({e}).") from e

    rows = _unfilter_scanlines(decompressed, width, height, filter_bpp, row_len)

    if color_type == 3:
        # Palette: unpack indices then expand to RGB via PLTE.
        indices = rows if bit_depth == 8 else _unpack_low_bit_rows(rows, width, height, bit_depth)
        n_entries = len(plte) // 3
        pixels = bytearray()
        for idx in indices:
            if idx >= n_entries:
                raise ValueError("Corrupt PNG: palette index out of range.")
            pixels.extend(plte[idx * 3 : idx * 3 + 3])
        pixels = bytes(pixels)
        mode = "RGB"
    elif bit_depth < 8:  # grayscale 1/2/4-bit
        pixels = _unpack_low_bit_rows(rows, width, height, bit_depth)
        mode = _MODES[color_type]
    else:
        pixels = rows
        mode = _MODES[color_type]

    return {
        "width": width,
        "height": height,
        "bit_depth": bit_depth,
        "color_type": color_type,
        "mode": mode,
        "pixels": pixels,
    }
