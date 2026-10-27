import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jpeg_transport import read_frame


class ProtocolTest(unittest.TestCase):
    def test_fragmented_header_payload_and_consecutive_frames(self):
        a, b = socket.socketpair()
        jpeg = b'\xff\xd8payload\xff\xd9'
        frame = struct.pack('!4sIII', b'JPG1', len(jpeg), 123, 456) + jpeg
        def send():
            with a:
                for index in range(0, len(frame * 2), 3):
                    a.sendall((frame * 2)[index:index + 3])
        sender = threading.Thread(target=send)
        sender.start()
        with b:
            self.assertEqual(read_frame(b), (123, 456, jpeg))
            self.assertEqual(read_frame(b), (123, 456, jpeg))
            with self.assertRaises(EOFError):
                read_frame(b)
        sender.join()

    def test_rejects_unbounded_frame_before_reading_payload(self):
        a, b = socket.socketpair()
        with a, b:
            a.sendall(struct.pack('!4sIII', b'JPG1', 2000001, 1, 0))
            with self.assertRaises(ValueError):
                read_frame(b)


@unittest.skipUnless(os.environ.get('JPEG_STREAM_FIXTURE'), 'C++ fixture not configured')
class SenderTest(unittest.TestCase):
    def setUp(self):
        self.process = subprocess.Popen([os.environ['JPEG_STREAM_FIXTURE']],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.address = '\0' + self.process.stdout.readline().strip()

    def tearDown(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            self.fail('Sender did not stop within two seconds')
        self.process.stdout.close()

    def publish(self, sec, nsec, size=64):
        self.process.stdin.write(f'{sec} {nsec} {size}\n')
        self.process.stdin.flush()
        self.assertEqual(self.process.stdout.readline().strip(), 'published')

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2)
        sock.connect(self.address)
        return sock

    def test_cpp_python_timestamp_roundtrip_and_reconnect(self):
        self.publish(123, 987654321)
        with self.connect() as client:
            sec, nsec, jpeg = read_frame(client)
            self.assertEqual((sec, nsec, len(jpeg)), (123, 987654321, 64))
        self.publish(124, 1)
        with self.connect() as client:
            sec, nsec, _ = read_frame(client)
            self.assertEqual((sec, nsec), (124, 1))

    def test_slow_viewer_does_not_block_publication_or_accumulate_old_frames(self):
        with self.connect():
            start = time.monotonic()
            for sec in range(1, 21):
                self.publish(sec, 0, 1000000)
            self.assertLess(time.monotonic() - start, 2)
            time.sleep(.7)
        with self.connect() as client:
            sec, nsec, jpeg = read_frame(client)
            self.assertEqual((sec, nsec, len(jpeg)), (20, 0, 1000000))


if __name__ == '__main__':
    unittest.main()
