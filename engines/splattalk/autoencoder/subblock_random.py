import hashlib
import json

import torch

from sparsity_utils import ALIGNED_KEYS, normalize_payload, sha256_file, validate_aligned


SELECTION_VERSION = "sub_block_preselected_v1"
SELECTION_METHOD = "uniform_random"
MAX_BUDGET = 729
LAYOUTS = ("subblock_1xk", "legacy_27x27")


def derive_permutation_seed(scene_id, global_seed):
    value = f"{scene_id}|{SELECTION_METHOD}|{int(global_seed)}|{SELECTION_VERSION}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def tensor_digest(indices, metadata):
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(indices.contiguous().numpy().tobytes())
    return digest.hexdigest()


def expected_ranking(scene_id, source_count, global_seed, source_digest, max_budget=MAX_BUDGET):
    if source_count < max_budget:
        raise ValueError(f"source has {source_count} rows, below ranking budget {max_budget}")
    permutation_seed = derive_permutation_seed(scene_id, global_seed)
    generator = torch.Generator(device="cpu").manual_seed(permutation_seed)
    selected = torch.randperm(source_count, generator=generator)[:max_budget]
    metadata = {
        "selection_version": SELECTION_VERSION,
        "selection_method": SELECTION_METHOD,
        "ranking_scheme": "nested_prefix_v1",
        "scene_id": scene_id,
        "global_seed": int(global_seed),
        "permutation_seed": permutation_seed,
        "source_count": source_count,
        "max_budget": max_budget,
        "dense_source_sha256": source_digest,
    }
    metadata["ranking_sha256"] = tensor_digest(selected, metadata)
    return {"selected_idx": selected, "ranking": metadata}


def create_ranking(dense_path, scene_id, global_seed, max_budget=MAX_BUDGET):
    dense = normalize_payload(torch.load(dense_path, map_location="cpu", weights_only=False))
    source_count = validate_aligned(dense, expected_feature_dim=256)
    return expected_ranking(
        scene_id, source_count, global_seed, sha256_file(dense_path), max_budget
    )


def create_legacy_ranking(dense_path, legacy_sparse_path, scene_id):
    dense = normalize_payload(torch.load(dense_path, map_location="cpu", weights_only=False))
    legacy = normalize_payload(
        torch.load(legacy_sparse_path, map_location="cpu", weights_only=False),
        require_sparsity=True,
    )
    source_count = validate_aligned(dense, expected_feature_dim=256)
    validate_aligned(legacy, expected_feature_dim=256)
    selected = legacy["selected_idx"].to(torch.int64)
    if len(selected) != MAX_BUDGET or len(torch.unique(selected)) != MAX_BUDGET:
        raise ValueError("legacy control ranking must contain exactly 729 unique rows")
    for key in ALIGNED_KEYS:
        if not torch.equal(legacy[key], dense[key][selected]):
            raise ValueError(f"legacy control tensor mismatch: {key}")
    metadata = {
        "selection_version": SELECTION_VERSION,
        "selection_method": SELECTION_METHOD,
        "ranking_scheme": "legacy_729_control",
        "scene_id": scene_id,
        "global_seed": 0,
        "permutation_seed": int(legacy["sparsity"].get("derived_seed", 0)),
        "source_count": source_count,
        "max_budget": MAX_BUDGET,
        "dense_source_sha256": sha256_file(dense_path),
    }
    metadata["ranking_sha256"] = tensor_digest(selected, metadata)
    return {"selected_idx": selected, "ranking": metadata}


def verify_ranking(dense_path, ranking, scene_id, global_seed=None):
    if not isinstance(ranking, dict) or set(ranking) != {"selected_idx", "ranking"}:
        raise ValueError("ranking payload has unexpected fields")
    metadata = ranking["ranking"]
    selected = ranking["selected_idx"]
    if selected.dtype != torch.int64 or selected.ndim != 1:
        raise ValueError("ranking selected_idx must be one-dimensional int64")
    if metadata.get("selection_version") != SELECTION_VERSION:
        raise ValueError("ranking version mismatch")
    if metadata.get("selection_method") != SELECTION_METHOD:
        raise ValueError("ranking method mismatch")
    if metadata.get("scene_id") != scene_id:
        raise ValueError("ranking scene mismatch")
    if metadata.get("max_budget") != len(selected) or len(selected) != MAX_BUDGET:
        raise ValueError("ranking budget mismatch")
    if len(torch.unique(selected)) != len(selected):
        raise ValueError("ranking contains duplicate indices")
    if metadata.get("dense_source_sha256") != sha256_file(dense_path):
        raise ValueError("ranking source digest mismatch")
    if metadata.get("ranking_sha256") != tensor_digest(
        selected, {key: value for key, value in metadata.items() if key != "ranking_sha256"}
    ):
        raise ValueError("ranking digest mismatch")
    if metadata.get("ranking_scheme") == "nested_prefix_v1":
        if global_seed is None or metadata.get("global_seed") != int(global_seed):
            raise ValueError("ranking global seed mismatch")
        expected = create_ranking(dense_path, scene_id, global_seed)
        if not torch.equal(selected, expected["selected_idx"]):
            raise ValueError("ranking does not match recomputed permutation")
    elif metadata.get("ranking_scheme") != "legacy_729_control":
        raise ValueError("unknown ranking scheme")
    source_count = metadata.get("source_count")
    if selected.min().item() < 0 or selected.max().item() >= source_count:
        raise ValueError("ranking index outside source range")
    return metadata


def ordered_prefix(ranking, budget, order_transform="ranked"):
    if budget <= 0 or budget > len(ranking["selected_idx"]):
        raise ValueError("sub-block budget must be in [1, 729]")
    selected = ranking["selected_idx"][:budget].clone()
    if order_transform == "ranked":
        return selected
    if order_transform == "shuffle_k64_v1":
        if budget != 64:
            raise ValueError("shuffle_k64_v1 is defined only for budget 64")
        seed = int.from_bytes(
            hashlib.sha256(
                f"{ranking['ranking']['scene_id']}|order_shuffle_k64_v1".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return selected[torch.randperm(budget, generator=generator)]
    raise ValueError(f"unknown order transform: {order_transform}")


def materialize_payload(dense, ranking, budget, feature_layout="subblock_1xk", order_transform="ranked"):
    if feature_layout not in LAYOUTS:
        raise ValueError(f"unknown feature layout: {feature_layout}")
    if feature_layout == "legacy_27x27" and budget != 729:
        raise ValueError("legacy layout requires exactly 729 rows")
    dense = normalize_payload(dense)
    source_count = validate_aligned(dense, expected_feature_dim=256)
    selected = ordered_prefix(ranking, budget, order_transform)
    rank_metadata = ranking["ranking"]
    prefix_fields = {
        "ranking_sha256": rank_metadata["ranking_sha256"],
        "requested_budget": budget,
        "order_transform": order_transform,
    }
    metadata = {
        "selection_version": SELECTION_VERSION,
        "selection_method": SELECTION_METHOD,
        "scene_id": rank_metadata["scene_id"],
        "requested_budget": budget,
        "source_count": source_count,
        "retained_count": budget,
        "consumed_count": budget,
        "effective_ratio": budget / source_count,
        "global_seed": rank_metadata["global_seed"],
        "permutation_seed": rank_metadata["permutation_seed"],
        "ranking_scheme": rank_metadata["ranking_scheme"],
        "dense_source_sha256": rank_metadata["dense_source_sha256"],
        "ranking_sha256": rank_metadata["ranking_sha256"],
        "ranking_prefix_sha256": tensor_digest(selected, prefix_fields),
        "feature_layout": feature_layout,
        "order_transform": order_transform,
        "selection_inputs": ["source_count", "scene_id", "global_seed"],
    }
    payload = {key: dense[key][selected] for key in ALIGNED_KEYS}
    payload["selected_idx"] = selected
    payload["sparsity"] = metadata
    validate_aligned(payload, expected_feature_dim=256)
    return payload


def verify_sparse_payload(dense_path, ranking_path, sparse_path, budget, global_seed=None):
    ranking = torch.load(ranking_path, map_location="cpu", weights_only=False)
    scene_id = ranking["ranking"]["scene_id"]
    verify_ranking(dense_path, ranking, scene_id, global_seed)
    dense = normalize_payload(torch.load(dense_path, map_location="cpu", weights_only=False))
    sparse = normalize_payload(
        torch.load(sparse_path, map_location="cpu", weights_only=False), require_sparsity=True
    )
    expected = materialize_payload(
        dense,
        ranking,
        budget,
        sparse["sparsity"].get("feature_layout"),
        sparse["sparsity"].get("order_transform"),
    )
    if sparse["sparsity"] != expected["sparsity"]:
        raise ValueError("sub-block metadata verification failed")
    if not torch.equal(sparse["selected_idx"], expected["selected_idx"]):
        raise ValueError("sub-block selected_idx verification failed")
    for key in ALIGNED_KEYS:
        if not torch.equal(sparse[key], expected[key]):
            raise ValueError(f"sub-block tensor verification failed: {key}")
    return sparse["sparsity"]
