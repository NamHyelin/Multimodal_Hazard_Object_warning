#!/usr/bin/env python3
"""Shared run-directory and single-instance helpers for live Florence pipelines."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


def make_run_id(requested: str | None) -> str:
    if requested:
        if not _SAFE_RUN_ID.fullmatch(requested):
            raise ValueError(
                "--run_id may contain only letters, digits, dot, underscore, and dash"
            )
        return requested

    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{os.getpid()}"


def make_node_name(prefix: str, run_id: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", run_id)
    return f"{prefix}_{normalized}"


class InstanceLock:
    """Hold a non-blocking process lock for one logical live pipeline."""

    def __init__(self, pipeline: str, *identity: str):
        key = "\n".join((pipeline, *identity))
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        lock_dir = Path("/tmp/hyelin_pipeline_locks")
        lock_dir.mkdir(parents=True, exist_ok=True)
        self.path = lock_dir / f"{pipeline}_{digest}.lock"
        self.file = self.path.open("a+", encoding="utf-8")

        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.file.seek(0)
            owner = self.file.read().strip() or "unknown"
            self.file.close()
            raise RuntimeError(
                f"{pipeline} is already running for the same topics "
                f"(lock={self.path}, owner_pid={owner})"
            ) from exc

        self.file.seek(0)
        self.file.truncate()
        self.file.write(str(os.getpid()))
        self.file.flush()

    def close(self) -> None:
        if self.file.closed:
            return
        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


def create_session_directory(
    output_root: str,
    pipeline: str,
    run_id: str,
    metadata: dict,
) -> Path:
    session_dir = Path(output_root) / pipeline / run_id
    session_dir.mkdir(parents=True, exist_ok=False)

    metadata_path = session_dir / "metadata.json"
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, default=str)
        stream.write("\n")

    return session_dir

