import hashlib
import json
import os
from pathlib import Path

import torch


def require_h200():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to execute on CPU")
    name = torch.cuda.get_device_name(0)
    expected = os.environ.get("SPLATTALK_EXPECTED_GPU", "").lower()
    if expected and expected not in name.lower():
        raise RuntimeError(f"an NVIDIA {expected.upper()} is required, got {name}")
    return {
        "name": name,
        "cuda": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def tensor_digest(tensor, metadata=None):
    digest = hashlib.sha256()
    if metadata is not None:
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_unique_scenes(path):
    scenes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be non-empty and unique")
    return scenes


def assert_output_below(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"writable path escapes detected-object root: {path}")
    return path
