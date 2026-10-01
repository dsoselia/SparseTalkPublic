from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPLATTALK_ENGINE = ROOT / "engines" / "splattalk"
LLAVA_ENGINE = ROOT / "engines" / "llava"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def require_cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    return torch.cuda.get_device_name(0)


def require_external_run_dir(path: str | Path) -> Path:
    run_dir = Path(path).expanduser().resolve()
    if run_dir == ROOT or ROOT in run_dir.parents:
        raise ValueError(f"run directory must be outside the source repository: {run_dir}")
    return run_dir


def require_run_local_output(path: str | Path, run_dir: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    root = require_external_run_dir(run_dir)
    if output == root or root not in output.parents:
        raise ValueError(f"output must be beneath the run directory: {output}")
    return output


def scene_ids(spec: str | Path) -> list[str]:
    path = Path(spec)
    values = path.read_text().splitlines() if path.is_file() else str(spec).split(",")
    scenes = [value.strip() for value in values if value.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene IDs must be nonempty and unique")
    if any("/" in scene or "\\" in scene or scene in {".", ".."} for scene in scenes):
        raise ValueError("scene IDs must be directory names")
    return scenes


def dense_path(root: str | Path, scene_id: str) -> Path:
    return Path(root) / scene_id / "feat_fs" / f"{scene_id}.pt"


def sparse_path(root: str | Path, scene_id: str) -> Path:
    return Path(root) / scene_id / "feat_fs" / f"{scene_id}.pt"
