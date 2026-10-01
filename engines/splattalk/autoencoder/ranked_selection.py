import hashlib
import json
from pathlib import Path

import torch

from anchor_selection import select_anchors, stable_seed
from sparsity_utils import ALIGNED_KEYS, normalize_payload, sha256_file, validate_aligned


SELECTION_VERSION = "sub_block_ranked_v1"
MAX_BUDGET = 729
METHODS = (
    "voxel_random",
    "fps",
    "semantic_kcenter",
    "joint_kcenter",
    "entropy_topk",
    "opacity_topk",
)
METHOD_INPUTS = {
    "voxel_random": ["points"],
    "fps": ["points"],
    "semantic_kcenter": ["points", "features"],
    "joint_kcenter": ["points", "features"],
    "entropy_topk": ["decoded_features"],
    "opacity_topk": ["opacities"],
}


def tensor_digest(indices, metadata):
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    digest.update(indices.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def ranking_set_name(method, global_seed=0, alpha=0.5):
    if method == "voxel_random":
        return f"voxel_random_seed_{global_seed}"
    if method == "joint_kcenter":
        return f"joint_kcenter_alpha_{str(alpha).replace('.', 'p')}"
    return method


def hyperparameters(method, voxel_size, alpha, global_seed):
    values = {"max_budget": MAX_BUDGET}
    if method in {"voxel_random", "semantic_kcenter", "joint_kcenter"}:
        values["voxel_size"] = float(voxel_size)
    if method == "voxel_random":
        values["global_seed"] = int(global_seed)
    if method == "semantic_kcenter":
        values["alpha"] = 0.0
    if method == "joint_kcenter":
        values["alpha"] = float(alpha)
    if method == "entropy_topk":
        values.update({"feature_dim": 3584, "score": "softmax_entropy_descending"})
    if method == "opacity_topk":
        values["score"] = "opacity_descending"
    return values


def stable_descending(values, budget=MAX_BUDGET):
    if values.ndim != 1 or not torch.isfinite(values).all().item():
        raise ValueError("ranking scores must be finite and one-dimensional")
    return torch.argsort(values.cpu(), descending=True, stable=True)[:budget].to(torch.int64)


def decoded_entropy(features, chunk_size=4096):
    if features.ndim != 2 or features.shape[1] != 3584:
        raise ValueError("decoded entropy features must have shape N x 3584")
    scores = []
    for start in range(0, len(features), chunk_size):
        block = features[start : start + chunk_size].float().cuda(non_blocking=True)
        probabilities = torch.softmax(block, dim=1)
        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)
        scores.append(entropy.cpu())
    return torch.cat(scores)


def _validate_dense_pair(encoded, decoded):
    count = validate_aligned(encoded, expected_feature_dim=256)
    if validate_aligned(decoded, expected_feature_dim=3584) != count:
        raise ValueError("encoded and decoded source row counts differ")
    for key in ALIGNED_KEYS[1:]:
        if not torch.equal(encoded[key], decoded[key]):
            raise ValueError(f"encoded/decoded source alignment mismatch: {key}")
    return count


def create_ranking(
    encoded_path,
    scene_id,
    method,
    decoded_path=None,
    voxel_size=0.01,
    alpha=0.5,
    global_seed=0,
):
    if method not in METHODS:
        raise ValueError(f"unknown ranked selector: {method}")
    encoded = normalize_payload(torch.load(encoded_path, map_location="cpu", weights_only=False))
    source_count = validate_aligned(encoded, expected_feature_dim=256)
    if source_count < MAX_BUDGET:
        raise ValueError("dense source contains fewer than 729 rows")
    decoded_digest = None
    diagnostics = {}
    derived_seed = None
    if method == "entropy_topk":
        if decoded_path is None:
            raise ValueError("entropy_topk requires a decoded source")
        decoded = normalize_payload(torch.load(decoded_path, map_location="cpu", weights_only=False))
        _validate_dense_pair(encoded, decoded)
        selected = stable_descending(decoded_entropy(decoded["features"]))
        decoded_digest = sha256_file(decoded_path)
    elif method == "opacity_topk":
        selected = stable_descending(encoded["opacities"].float())
    else:
        selected, diagnostics = select_anchors(
            encoded["points"].cuda(non_blocking=True),
            encoded["features"].cuda(non_blocking=True),
            MAX_BUDGET,
            method,
            voxel_size=voxel_size,
            alpha=alpha,
            global_seed=global_seed,
            scene_id=scene_id,
        )
        if method == "voxel_random":
            derived_seed = stable_seed(scene_id, method, MAX_BUDGET, global_seed)
            if diagnostics.get("derived_seed") != derived_seed:
                raise ValueError("voxel-random derived seed mismatch")
    if selected.dtype != torch.int64 or len(selected) != MAX_BUDGET:
        raise ValueError("ranked selector must produce 729 int64 indices")
    if len(torch.unique(selected)) != MAX_BUDGET:
        raise ValueError("ranked selector produced duplicate indices")
    metadata = {
        "selection_version": SELECTION_VERSION,
        "selection_method": method,
        "ranking_scheme": "nested_prefix_v1",
        "ranking_set": ranking_set_name(method, global_seed, alpha),
        "scene_id": scene_id,
        "source_count": source_count,
        "max_budget": MAX_BUDGET,
        "selection_inputs": METHOD_INPUTS[method],
        "hyperparameters": hyperparameters(method, voxel_size, alpha, global_seed),
        "dense_source_sha256": sha256_file(encoded_path),
        "decoded_source_sha256": decoded_digest,
        "derived_seed": derived_seed,
        "diagnostics": diagnostics,
    }
    metadata["ranking_sha256"] = tensor_digest(selected, metadata)
    return {"selected_idx": selected.cpu(), "ranking": metadata}


def verify_ranking(
    encoded_path,
    ranking,
    scene_id,
    method,
    decoded_path=None,
    voxel_size=0.01,
    alpha=0.5,
    global_seed=0,
):
    validate_ranking(
        encoded_path, ranking, scene_id, method, voxel_size, alpha, global_seed
    )
    expected = create_ranking(
        encoded_path, scene_id, method, decoded_path, voxel_size, alpha, global_seed
    )
    if (ranking["ranking"] != expected["ranking"]
            or not torch.equal(ranking["selected_idx"], expected["selected_idx"])):
        raise ValueError("ranking does not match deterministic recomputation")
    return ranking["ranking"]


def validate_ranking(
    encoded_path,
    ranking,
    scene_id,
    method,
    voxel_size=0.01,
    alpha=0.5,
    global_seed=0,
):
    if not isinstance(ranking, dict) or set(ranking) != {"selected_idx", "ranking"}:
        raise ValueError("ranked payload has unexpected fields")
    selected = ranking["selected_idx"]
    metadata = ranking["ranking"]
    if selected.dtype != torch.int64 or selected.ndim != 1 or len(selected) != MAX_BUDGET:
        raise ValueError("ranked selected_idx must contain 729 int64 rows")
    if len(torch.unique(selected)) != len(selected):
        raise ValueError("ranked selected_idx contains duplicates")
    if metadata.get("selection_version") != SELECTION_VERSION:
        raise ValueError("ranked selection version mismatch")
    if metadata.get("selection_method") != method or metadata.get("scene_id") != scene_id:
        raise ValueError("ranked method or scene mismatch")
    if metadata.get("ranking_scheme") != "nested_prefix_v1":
        raise ValueError("ranked selection scheme mismatch")
    source_count = metadata.get("source_count")
    if not isinstance(source_count, int) or source_count < MAX_BUDGET:
        raise ValueError("invalid ranked source count")
    if selected.min().item() < 0 or selected.max().item() >= source_count:
        raise ValueError("ranked index outside source range")
    if metadata.get("max_budget") != MAX_BUDGET:
        raise ValueError("ranked maximum budget mismatch")
    if metadata.get("ranking_set") != ranking_set_name(method, global_seed, alpha):
        raise ValueError("ranked set name mismatch")
    if metadata.get("selection_inputs") != METHOD_INPUTS[method]:
        raise ValueError("ranked selection inputs mismatch")
    if metadata.get("hyperparameters") != hyperparameters(
        method, voxel_size, alpha, global_seed
    ):
        raise ValueError("ranked hyperparameters mismatch")
    expected_seed = (
        stable_seed(scene_id, method, MAX_BUDGET, global_seed)
        if method == "voxel_random" else None
    )
    if metadata.get("derived_seed") != expected_seed:
        raise ValueError("ranked derived seed mismatch")
    diagnostics = metadata.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise ValueError("ranked diagnostics must be a dictionary")
    if method == "voxel_random" and diagnostics.get("derived_seed") != expected_seed:
        raise ValueError("ranked diagnostic seed mismatch")
    if metadata.get("dense_source_sha256") != sha256_file(encoded_path):
        raise ValueError("ranked encoded-source digest mismatch")
    decoded_digest = metadata.get("decoded_source_sha256")
    if method == "entropy_topk":
        if not isinstance(decoded_digest, str) or len(decoded_digest) != 64:
            raise ValueError("entropy ranking requires a decoded-source digest")
    elif decoded_digest is not None:
        raise ValueError("unexpected decoded-source digest")
    digest_fields = {key: value for key, value in metadata.items() if key != "ranking_sha256"}
    if metadata.get("ranking_sha256") != tensor_digest(selected, digest_fields):
        raise ValueError("ranked digest mismatch")
    return metadata


def prefix_digest(selected, ranking_sha256, budget):
    fields = {
        "ranking_sha256": ranking_sha256,
        "requested_budget": int(budget),
        "order_transform": "ranked",
    }
    return tensor_digest(selected, fields)


def materialize_payload(encoded, ranking, budget):
    if budget <= 0 or budget > MAX_BUDGET:
        raise ValueError("ranked sub-block budget must be in [1, 729]")
    encoded = normalize_payload(encoded)
    source_count = validate_aligned(encoded, expected_feature_dim=256)
    ranked = ranking["ranking"]
    selected = ranking["selected_idx"][:budget].clone()
    metadata = {
        "selection_version": SELECTION_VERSION,
        "selection_method": ranked["selection_method"],
        "ranking_scheme": ranked["ranking_scheme"],
        "ranking_set": ranked["ranking_set"],
        "scene_id": ranked["scene_id"],
        "requested_budget": budget,
        "source_count": source_count,
        "retained_count": budget,
        "consumed_count": budget,
        "effective_ratio": budget / source_count,
        "selection_inputs": ranked["selection_inputs"],
        "hyperparameters": ranked["hyperparameters"],
        "dense_source_sha256": ranked["dense_source_sha256"],
        "decoded_source_sha256": ranked["decoded_source_sha256"],
        "ranking_sha256": ranked["ranking_sha256"],
        "ranking_prefix_sha256": prefix_digest(selected, ranked["ranking_sha256"], budget),
        "feature_layout": "subblock_1xk",
        "order_transform": "ranked",
    }
    payload = {key: encoded[key][selected] for key in ALIGNED_KEYS}
    payload.update({"selected_idx": selected, "sparsity": metadata})
    validate_aligned(payload, expected_feature_dim=256)
    return payload
