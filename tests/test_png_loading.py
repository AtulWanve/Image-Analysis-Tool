"""Unit tests for Issue #1: PNG image loading (stdlib only)."""

import os
import struct
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from png_loader import load_png

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def _make_png(width, height, color_type, bit_depth, rows, palette=None, interlace=0):
    """Build a minimal valid PNG. rows = list of unfiltered row bytes (no filter byte)."""
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, interlace)
    out = bytearray(PNG_SIG)
    out += _chunk(b"IHDR", ihdr)
    if palette is not None:
        out += _chunk(b"PLTE", palette)
    raw = b"".join(b"\x00" + r for r in rows)
    out += _chunk(b"IDAT", zlib.compress(raw))
    out += _chunk(b"IEND", b"")
    return bytes(out)


def _write_temp(suffix, data):
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(data)
        tmp.close()
        return tmp.name
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


class TestPngLoading(unittest.TestCase):
    def test_load_valid_rgb(self):
        # 2x2 RGB pixels: red, green / blue, white
        rows = [bytes([255, 0, 0, 0, 255, 0]), bytes([0, 0, 255, 255, 255, 255])]
        path = _write_temp(".png", _make_png(2, 2, 2, 8, rows))
        try:
            img = load_png(path)
            self.assertEqual(img["width"], 2)
            self.assertEqual(img["height"], 2)
            self.assertEqual(img["mode"], "RGB")
            self.assertEqual(img["pixels"], b"".join(rows))
        finally:
            os.unlink(path)

    def test_load_valid_rgba(self):
        rows = [bytes([10, 20, 30, 40])]
        path = _write_temp(".png", _make_png(1, 1, 6, 8, rows))
        try:
            img = load_png(path)
            self.assertEqual((img["width"], img["height"]), (1, 1))
            self.assertEqual(img["mode"], "RGBA")
            self.assertEqual(img["pixels"], rows[0])
        finally:
            os.unlink(path)

    def test_load_valid_grayscale(self):
        rows = [bytes([0, 127, 255])]
        path = _write_temp(".png", _make_png(3, 1, 0, 8, rows))
        try:
            img = load_png(path)
            self.assertEqual(img["mode"], "L")
            self.assertEqual(img["pixels"], rows[0])
        finally:
            os.unlink(path)

    def test_load_valid_palette_expands_to_rgb(self):
        palette = bytes([255, 0, 0, 0, 255, 0])  # idx0=red, idx1=green
        rows = [bytes([0, 1])]
        path = _write_temp(".png", _make_png(2, 1, 3, 8, rows, palette=palette))
        try:
            img = load_png(path)
            self.assertEqual(img["mode"], "RGB")
            self.assertEqual(img["pixels"], bytes([255, 0, 0, 0, 255, 0]))
        finally:
            os.unlink(path)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_png("does_not_exist_12345.png")

    def test_wrong_extension(self):
        rows = [bytes([1, 2, 3])]
        path = _write_temp(".bin", _make_png(1, 1, 2, 8, rows))
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = _write_temp(".png", b"")
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)

    def test_invalid_signature(self):
        path = _write_temp(".png", b"NOTAPNG!" + b"\x00" * 20)
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)

    def test_truncated_file(self):
        good = _make_png(1, 1, 2, 8, [bytes([1, 2, 3])])
        path = _write_temp(".png", good[:30])  # cut mid-chunk
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)

    def test_missing_ihdr(self):
        raw = b"".join([b"\x00" + bytes([1, 2, 3])])
        data = PNG_SIG + _chunk(b"IDAT", zlib.compress(raw)) + _chunk(b"IEND", b"")
        path = _write_temp(".png", data)
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)

    def test_missing_idat(self):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        data = PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IEND", b"")
        path = _write_temp(".png", data)
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)

    def test_bad_crc(self):
        good = _make_png(1, 1, 2, 8, [bytes([1, 2, 3])])
        bad = bytearray(good)
        bad[-8] ^= 0xFF  # corrupt IDAT CRC
        path = _write_temp(".png", bytes(bad))
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)

    def test_corrupt_idat(self):
        ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        data = PNG_SIG + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", b"not-zlib-data") + _chunk(b"IEND", b"")
        path = _write_temp(".png", data)
        try:
            with self.assertRaises(ValueError):
                load_png(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
