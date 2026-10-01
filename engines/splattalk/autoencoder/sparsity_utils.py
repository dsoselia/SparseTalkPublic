import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

import torch


ALIGNED_KEYS = ("features", "points", "covariances", "opacities")
SELECTION_VARIABLES = ["source_count", "scene_id", "requested_ratio", "global_seed"]


def parse_ratio(value):
    try:
        ratio = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError(f"invalid ratio: {value}") from exc
    if not Decimal("0") < ratio <= Decimal("1"):
        raise ValueError("ratio must be in (0, 1]")
    canonical = format(ratio.normalize(), "f")
    return ratio, canonical


def derive_seed(scene_id, canonical_ratio, global_seed):
    payload = f"{scene_id}|{canonical_ratio}|{int(global_seed)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def retained_row_count(source_count, ratio, chunk_size):
    if source_count <= 0:
        raise ValueError("source_count must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if source_count < chunk_size:
        raise ValueError("source_count must be at least one complete chunk")
    if ratio == Decimal("1"):
        return source_count
    retained = int(Decimal(source_count) * ratio) // chunk_size * chunk_size
    return max(retained, chunk_size)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_payload(payload, require_sparsity=False):
    if not isinstance(payload, dict):
        raise ValueError("feature payload must be a dictionary")
    normalized = dict(payload)
    if "opacities" not in normalized and "opacitites" in normalized:
        normalized["opacities"] = normalized["opacitites"]
    missing = [key for key in ALIGNED_KEYS if key not in normalized]
    if missing:
        raise ValueError(f"missing aligned tensors: {missing}")
    if require_sparsity:
        for key in ("selected_idx", "sparsity"):
            if key not in normalized:
                raise ValueError(f"missing sparsity field: {key}")
    return normalized


def validate_aligned(payload, expected_feature_dim=None):
    payload = normalize_payload(payload)
    tensors = {key: payload[key] for key in ALIGNED_KEYS}
    if any(not torch.is_tensor(value) for value in tensors.values()):
        raise ValueError("all aligned values must be tensors")
    count = tensors["features"].shape[0]
    if count == 0:
        raise ValueError("feature payload is empty")
    for key, value in tensors.items():
        if value.shape[0] != count:
            raise ValueError(f"row mismatch for {key}: {value.shape[0]} != {count}")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"non-finite values in {key}")
    if tensors["features"].ndim != 2:
        raise ValueError("features must have shape N x C")
    if expected_feature_dim is not None and tensors["features"].shape[1] != expected_feature_dim:
        raise ValueError(
            f"feature dimension {tensors['features'].shape[1]} != {expected_feature_dim}"
        )
    if tuple(tensors["points"].shape[1:]) != (3,):
        raise ValueError("points must have shape N x 3")
    if tuple(tensors["covariances"].shape[1:]) != (3, 3):
        raise ValueError("covariances must have shape N x 3 x 3")
    if tensors["opacities"].ndim != 1:
        raise ValueError("opacities must have shape N")
    return count


def expected_indices(scene_id, source_count, ratio_value, global_seed, chunk_size):
    ratio, canonical = parse_ratio(ratio_value)
    retained = retained_row_count(source_count, ratio, chunk_size)
    derived = derive_seed(scene_id, canonical, global_seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derived)
    indices = torch.randperm(source_count, generator=generator)[:retained]
    return indices, ratio, canonical, derived


def create_sparse_payload(dense_payload, scene_id, ratio_value, global_seed, chunk_size, source_digest):
    dense_payload = normalize_payload(dense_payload)
    source_count = validate_aligned(dense_payload, expected_feature_dim=256)
    indices, ratio, canonical, derived = expected_indices(
        scene_id, source_count, ratio_value, global_seed, chunk_size
    )
    retained = len(indices)
    result = {key: dense_payload[key][indices] for key in ALIGNED_KEYS}
    result["selected_idx"] = indices
    result["sparsity"] = {
        "selection_method": "random",
        "scene_id": scene_id,
        "requested_ratio": canonical,
        "effective_ratio": retained / source_count,
        "source_count": source_count,
        "retained_count": retained,
        "global_seed": int(global_seed),
        "derived_seed": derived,
        "chunk_size": int(chunk_size),
        "dense_source_sha256": source_digest,
        "selection_variables": list(SELECTION_VARIABLES),
    }
    return result


def verify_sparse_payload(dense_path, sparse_path, scene_id, ratio_value, global_seed, chunk_size):
    dense = normalize_payload(torch.load(dense_path, map_location="cpu", weights_only=False))
    sparse = normalize_payload(
        torch.load(sparse_path, map_location="cpu", weights_only=False), require_sparsity=True
    )
    source_count = validate_aligned(dense, expected_feature_dim=256)
    retained_count = validate_aligned(sparse, expected_feature_dim=256)
    expected, _, canonical, derived = expected_indices(
        scene_id, source_count, ratio_value, global_seed, chunk_size
    )
    selected = sparse["selected_idx"]
    if selected.dtype != torch.int64 or selected.ndim != 1:
        raise ValueError("selected_idx must be a one-dimensional int64 tensor")
    if not torch.equal(selected, expected):
        raise ValueError("selected_idx does not match recomputed random permutation")
    metadata = sparse["sparsity"]
    expected_metadata = {
        "selection_method": "random",
        "scene_id": scene_id,
        "requested_ratio": canonical,
        "source_count": source_count,
        "retained_count": retained_count,
        "effective_ratio": retained_count / source_count,
        "global_seed": int(global_seed),
        "derived_seed": derived,
        "chunk_size": int(chunk_size),
        "dense_source_sha256": sha256_file(dense_path),
        "selection_variables": SELECTION_VARIABLES,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata mismatch for {key}")
    for key in ALIGNED_KEYS:
        if not torch.equal(sparse[key], dense[key][selected]):
            raise ValueError(f"sparse tensor mismatch for {key}")
    return metadata


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
