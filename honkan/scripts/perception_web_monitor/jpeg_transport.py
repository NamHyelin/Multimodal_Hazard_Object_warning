"""Bounded, timestamp-preserving reader for the receiver's local JPEG stream."""
import struct

HEADER = struct.Struct('!4sIII')
MAX_JPEG_BYTES = 2_000_000


def read_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError('JPEG source disconnected')
        data.extend(chunk)
    return bytes(data)


def read_frame(sock):
    magic, length, sec, nsec = HEADER.unpack(read_exact(sock, HEADER.size))
    if magic != b'JPG1' or not 4 <= length <= MAX_JPEG_BYTES or nsec >= 1_000_000_000:
        raise ValueError('Invalid JPEG transport header')
    jpeg = read_exact(sock, length)
    if not jpeg.startswith(b'\xff\xd8') or not jpeg.endswith(b'\xff\xd9'):
        raise ValueError('Invalid JPEG transport payload')
    return sec, nsec, jpeg
