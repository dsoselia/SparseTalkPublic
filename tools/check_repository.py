"""Check that tracked files do not include model or data artifacts."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin", ".zip", ".tar"}
FORBIDDEN_PARTS = {"runs", "outputs", "checkpoints", "datasets", "hf_cache", "__pycache__"}
MAX_SOURCE_BYTES = 10 * 1024 * 1024


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    failures = []
    for raw in tracked:
        if not raw:
            continue
        relative = Path(raw.decode())
        path = ROOT / relative
        generated_name = any(
            relative.name.lower().endswith(suffix)
            or f"{suffix}." in relative.name.lower()
            for suffix in FORBIDDEN_SUFFIXES
        )
        if generated_name or relative.name.startswith(".gio") or FORBIDDEN_PARTS.intersection(relative.parts):
            failures.append(f"tracked generated/model artifact: {relative}")
        elif path.is_file() and path.stat().st_size > MAX_SOURCE_BYTES:
            failures.append(f"tracked file exceeds 10 MiB: {relative}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"OK: {len(tracked) - 1} tracked paths, no model/data artifacts found")


if __name__ == "__main__":
    main()
