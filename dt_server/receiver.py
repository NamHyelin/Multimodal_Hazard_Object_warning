#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

BASE_DIR = Path("/home/morita/hyelin")
DEFAULT_EXCHANGE_DIR = BASE_DIR / "dt_exchange"


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    os.replace(tmp, path)


class LatestUDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, exchange_dir: Path, log_stale: bool = False):
        self.exchange_dir = exchange_dir
        self.latest_path = exchange_dir / "latest.json"
        self.log_stale = log_stale
        self.latest_seq = -1
        self.accepted = 0
        self.stale = 0
        self.transport = None

        exchange_dir.mkdir(parents=True, exist_ok=True)

        # Start fresh for a new live run.
        try:
            self.latest_path.unlink()
        except FileNotFoundError:
            pass

    def connection_made(self, transport):
        self.transport = transport
        print(f"[UDP RECEIVER READY] exchange_dir={self.exchange_dir}", flush=True)
        print(f"[UDP RECEIVER READY] latest_json={self.latest_path}", flush=True)
        print(f"[UDP RECEIVER READY] latest_seq={self.latest_seq}", flush=True)

    def datagram_received(self, data: bytes, addr):
        t0 = time.perf_counter()

        try:
            payload = json.loads(data.decode("utf-8"))
            seq = int(payload.get("latest_only_seq", -1))
            frame = int(payload.get("frame_index", -1))
        except Exception as exc:
            print(f"[RX DROP INVALID] from={addr} error={exc}", flush=True)
            return

        if seq <= self.latest_seq:
            self.stale += 1
            if self.log_stale:
                print(
                    f"[RX DROP STALE] frame={frame} seq={seq} "
                    f"latest_seq={self.latest_seq} from={addr}",
                    flush=True,
                )
            return

        # Accept only monotonically newer state.
        self.latest_seq = seq
        self.accepted += 1

        try:
            atomic_write_json(self.latest_path, payload)
        except Exception as exc:
            print(
                f"[RX WRITE ERROR] frame={frame} seq={seq} error={exc}",
                flush=True,
            )
            return

        dt_ms = (time.perf_counter() - t0) * 1000.0
        print(
            f"[RX UDP] frame={frame} seq={seq} bytes={len(data)} "
            f"write={dt_ms:.3f}ms accepted={self.accepted} stale_dropped={self.stale}",
            flush=True,
        )

    def error_received(self, exc):
        print(f"[UDP ERROR] {exc}", flush=True)


async def main_async(args):
    loop = asyncio.get_running_loop()
    exchange_dir = Path(args.exchange_dir).expanduser().resolve()

    transport, protocol = await loop.create_datagram_endpoint(
        lambda: LatestUDPProtocol(exchange_dir, log_stale=args.log_stale),
        local_addr=(args.host, args.port),
    )

    try:
        await asyncio.Future()
    finally:
        transport.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--exchange_dir",
        default=str(DEFAULT_EXCHANGE_DIR),
        help="Stable directory shared with dt_server.py",
    )
    parser.add_argument("--log_stale", action="store_true")
    args = parser.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[UDP RECEIVER] stopped.", flush=True)


if __name__ == "__main__":
    main()
