import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from detected_object_sampling import (
    BACKGROUND_POOL,
    MAX_BUDGET,
    POLICY_STUDY_VERSION,
    SELECTION_VERSION,
)
from detected_object_sampling.association import validate_association
from detected_object_sampling.common import canonical_digest, sha256_file, tensor_digest


METHOD_RANDOM = "object_balanced_random"
METHOD_FPS = "object_balanced_fps"
METHODS = (METHOD_RANDOM, METHOD_FPS)
SELECTION_INPUTS = [
    "points",
    "opacities",
    "posed_rgb",
    "florence_2_large_object_detection",
    "sam2_1_hiera_large_segmentation",
    "alpha_compositing_mass",
]


def stable_seed(scene_id, method, global_seed, pool_id, seed_version=SELECTION_VERSION):
    value = f"{scene_id}|{method}|{int(global_seed)}|{seed_version}|pool:{int(pool_id)}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def stable_pool_tie(scene_id, pool_id, seed_version=SELECTION_VERSION):
    value = f"{scene_id}|pool_tie|{int(pool_id)}|{seed_version}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_policy(policy, selection_version=SELECTION_VERSION):
    if not isinstance(policy, dict):
        raise ValueError("allocation policy must be a dictionary")
    if set(policy) != {"lambda", "q_bg", "alpha"}:
        raise ValueError("allocation policy fields must be lambda, q_bg, and alpha")
    if selection_version not in {SELECTION_VERSION, POLICY_STUDY_VERSION}:
        raise ValueError(f"unsupported detected-object selection version: {selection_version}")
    if any(not isinstance(policy[key], (int, float)) or not math.isfinite(policy[key]) for key in policy):
        raise ValueError("allocation parameters must be finite numbers")
    if not 0 <= policy["lambda"] <= 1 or not 0 <= policy["q_bg"] <= 1:
        raise ValueError("lambda and q_bg must be in [0, 1]")
    if policy["alpha"] <= 0:
        raise ValueError("gamma (internal alpha) must be positive")
    return policy


def pool_weights(pool_ids, visible_mass, policy, selection_version=SELECTION_VERSION):
    validate_policy(policy, selection_version)
    pools = sorted(int(value) for value in torch.unique(pool_ids).tolist())
    foreground = [pool for pool in pools if pool != BACKGROUND_POOL]
    if not foreground:
        return {BACKGROUND_POOL: 1.0}
    sizes = {}
    for pool in foreground:
        selected = pool_ids == pool
        size = float(visible_mass[selected].double().sum().item())
        if not math.isfinite(size) or size <= 0:
            raise ValueError(f"foreground pool {pool} has non-positive visible size")
        sizes[pool] = size
    powered_sum = sum(size ** policy["alpha"] for size in sizes.values())
    foreground_weights = {}
    for pool in foreground:
        equal = 1.0 / len(foreground)
        size_weight = sizes[pool] ** policy["alpha"] / powered_sum
        foreground_weights[pool] = (1.0 - policy["lambda"]) * equal + policy["lambda"] * size_weight
    weights = {pool: (1.0 - policy["q_bg"]) * value for pool, value in foreground_weights.items()}
    if BACKGROUND_POOL in pools:
        weights[BACKGROUND_POOL] = policy["q_bg"]
    else:
        scale = sum(weights.values())
        weights = {pool: value / scale for pool, value in weights.items()}
    return weights


def random_pool_order(
    indices, scene_id, global_seed, pool_id, limit=MAX_BUDGET,
    seed_version=SELECTION_VERSION,
):
    indices = torch.sort(indices.cpu().to(torch.int64)).values
    generator = torch.Generator(device="cpu")
    generator.manual_seed(stable_seed(
        scene_id, METHOD_RANDOM, global_seed, pool_id, seed_version
    ))
    return indices[torch.randperm(len(indices), generator=generator)[:limit]]


def fps_pool_order(indices, points, visible_mass, limit=MAX_BUDGET):
    indices = torch.sort(indices.cpu().to(torch.int64)).values
    if not len(indices):
        return indices
    pool_points = points[indices].to(device="cuda", dtype=torch.float32, non_blocking=True)
    pool_mass = visible_mass[indices].to(device="cuda", dtype=torch.float32, non_blocking=True)
    count = min(limit, len(indices))
    chosen = torch.empty(count, dtype=torch.int64, device="cuda")
    chosen[0] = torch.argmax(pool_mass)
    minimum_distance = torch.sum((pool_points - pool_points[chosen[0]]) ** 2, dim=1)
    minimum_distance[chosen[0]] = -1
    for position in range(1, count):
        candidate = torch.argmax(minimum_distance)
        chosen[position] = candidate
        distance = torch.sum((pool_points - pool_points[candidate]) ** 2, dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[chosen[: position + 1]] = -1
    return indices[chosen.cpu()]


def local_orders(
    association, points, scene_id, method, global_seed=0, limit=MAX_BUDGET,
    seed_version=SELECTION_VERSION,
):
    if method not in METHODS:
        raise ValueError(f"unsupported object sampler: {method}")
    pool_ids = association["pool_ids"]
    visible_mass = association["total_visible_mass"]
    if points.ndim != 2 or points.shape != (len(pool_ids), 3) or not torch.isfinite(points).all():
        raise ValueError("dense points are not aligned finite N x 3 coordinates")
    result = {}
    for pool in sorted(int(value) for value in torch.unique(pool_ids).tolist()):
        indices = torch.nonzero(pool_ids == pool, as_tuple=False).flatten()
        if method == METHOD_RANDOM:
            result[pool] = random_pool_order(
                indices, scene_id, global_seed, pool, limit, seed_version
            )
        else:
            result[pool] = fps_pool_order(indices, points, visible_mass, limit)
    return result


def deficit_interleave(
    orders, weights, scene_id, budget=MAX_BUDGET, tie_version=SELECTION_VERSION
):
    if budget <= 0 or budget > MAX_BUDGET:
        raise ValueError(f"budget must be in [1, {MAX_BUDGET}]")
    active = {pool for pool, order in orders.items() if len(order) and weights.get(pool, 0) > 0}
    if not active:
        raise ValueError("no non-empty positive-weight pools")
    offsets = defaultdict(int)
    deficits = defaultdict(float)
    tie_order = {pool: stable_pool_tie(scene_id, pool, tie_version) for pool in active}
    selected = []
    selected_pools = []
    while active and len(selected) < budget:
        total_weight = sum(weights[pool] for pool in active)
        if total_weight <= 0:
            raise ValueError("active pool weights sum to zero")
        for pool in active:
            deficits[pool] += weights[pool] / total_weight
        pool = min(active, key=lambda item: (-deficits[item], tie_order[item]))
        selected.append(int(orders[pool][offsets[pool]].item()))
        selected_pools.append(pool)
        offsets[pool] += 1
        deficits[pool] -= 1.0
        if offsets[pool] == len(orders[pool]):
            active.remove(pool)
    if len(selected) != budget:
        raise ValueError(f"pool orders exhausted after {len(selected)} of {budget} selections")
    indices = torch.tensor(selected, dtype=torch.int64)
    pools = torch.tensor(selected_pools, dtype=torch.int32)
    if len(torch.unique(indices)) != budget:
        raise ValueError("deficit scheduler produced duplicate Gaussian indices")
    return indices, pools


def build_ranking(
    association,
    dense_payload,
    scene_id,
    method,
    policy,
    global_seed=0,
    selection_version=SELECTION_VERSION,
    seed_version=None,
    ranking_set=None,
    orders=None,
):
    metadata = validate_association(association, scene_id)
    validate_policy(policy, selection_version)
    seed_version = seed_version or selection_version
    if method == METHOD_FPS and global_seed != 0:
        raise ValueError("deterministic FPS uses global_seed=0")
    points = dense_payload["points"]
    if len(points) != metadata["source_count"]:
        raise ValueError("association and dense source row counts differ")
    if orders is None:
        orders = local_orders(
            association, points, scene_id, method, global_seed,
            seed_version=seed_version,
        )
    weights = pool_weights(
        association["pool_ids"], association["total_visible_mass"], policy,
        selection_version,
    )
    indices, pool_sequence = deficit_interleave(
        orders, weights, scene_id, tie_version=seed_version
    )
    policy_sha256 = canonical_digest(policy)
    default_ranking_set = (
        f"object_random_seed_{global_seed}" if method == METHOD_RANDOM else "object_fps"
    )
    ranking_metadata = {
        "selection_version": selection_version,
        "selection_method": method,
        "scene_id": scene_id,
        "source_count": metadata["source_count"],
        "max_budget": MAX_BUDGET,
        "ranking_scheme": "nested_prefix_v1",
        "ranking_set": ranking_set or default_ranking_set,
        "global_seed": int(global_seed) if method == METHOD_RANDOM else None,
        "pool_random_seeds": (
            {
                str(pool): stable_seed(
                    scene_id, method, global_seed, pool, seed_version
                )
                for pool in orders
            }
            if method == METHOD_RANDOM else None
        ),
        "policy": policy,
        "policy_sha256": policy_sha256,
        "association_sha256": metadata["association_sha256"],
        "frame_digest": metadata["frame_digest"],
        "dense_source_sha256": metadata["input_digests"]["dense_encoded"],
        "selection_inputs": list(SELECTION_INPUTS),
        "pool_weights": {str(pool): value for pool, value in sorted(weights.items())},
        "pool_counts": {
            str(pool): int((association["pool_ids"] == pool).sum().item()) for pool in sorted(orders)
        },
    }
    if selection_version == POLICY_STUDY_VERSION:
        ranking_metadata["seed_version"] = seed_version
    digest_metadata = dict(ranking_metadata)
    ranking_metadata["pool_sequence_sha256"] = tensor_digest(pool_sequence, {
        "scene_id": scene_id, "policy_sha256": policy_sha256, "method": method,
    })
    ranking_metadata["ranking_sha256"] = tensor_digest(indices, {
        **digest_metadata,
        "pool_sequence_sha256": ranking_metadata["pool_sequence_sha256"],
    })
    return {
        "selected_idx": indices,
        "assigned_pool_sequence": pool_sequence,
        "ranking": ranking_metadata,
    }


def validate_ranking(ranking, association, dense_path, scene_id, expected_method=None):
    if not isinstance(ranking, dict) or set(ranking) != {
        "selected_idx", "assigned_pool_sequence", "ranking"
    }:
        raise ValueError("invalid ranking payload fields")
    metadata = ranking["ranking"]
    method = metadata.get("selection_method")
    selection_version = metadata.get("selection_version")
    if selection_version not in {SELECTION_VERSION, POLICY_STUDY_VERSION} or metadata.get("scene_id") != scene_id:
        raise ValueError("ranking identity mismatch")
    if method not in METHODS or (expected_method and method != expected_method):
        raise ValueError("ranking method mismatch")
    if selection_version == POLICY_STUDY_VERSION:
        if metadata.get("seed_version") != SELECTION_VERSION:
            raise ValueError("policy-study rankings must preserve the published detector seed namespace")
        ranking_set = metadata.get("ranking_set")
        seed = metadata.get("global_seed")
        if method == METHOD_RANDOM:
            if not isinstance(seed, int) or not ranking_set.endswith(f"/object_random_seed_{seed}"):
                raise ValueError("policy-study random ranking-set identity mismatch")
        elif seed is not None or not ranking_set.endswith("/object_fps"):
            raise ValueError("policy-study FPS ranking-set identity mismatch")
    indices = ranking["selected_idx"]
    pools = ranking["assigned_pool_sequence"]
    if indices.dtype != torch.int64 or indices.shape != (MAX_BUDGET,):
        raise ValueError("ranking must contain 729 int64 indices")
    if pools.dtype != torch.int32 or pools.shape != (MAX_BUDGET,):
        raise ValueError("ranking pool sequence must contain 729 int32 IDs")
    if len(torch.unique(indices)) != MAX_BUDGET:
        raise ValueError("ranking indices are not unique")
    source_count = metadata.get("source_count")
    if not isinstance(source_count, int) or source_count < MAX_BUDGET:
        raise ValueError("ranking source_count is invalid")
    if indices.min().item() < 0 or indices.max().item() >= source_count:
        raise ValueError("ranking index is outside source range")
    association_metadata = validate_association(association, scene_id, dense_path)
    if metadata.get("association_sha256") != association_metadata["association_sha256"]:
        raise ValueError("ranking association digest mismatch")
    if metadata.get("dense_source_sha256") != sha256_file(dense_path):
        raise ValueError("ranking dense-source digest mismatch")
    validate_policy(metadata.get("policy"), selection_version)
    if metadata.get("policy_sha256") != canonical_digest(metadata["policy"]):
        raise ValueError("ranking policy digest mismatch")
    expected_pool_digest = tensor_digest(pools, {
        "scene_id": scene_id,
        "policy_sha256": metadata["policy_sha256"],
        "method": method,
    })
    if metadata.get("pool_sequence_sha256") != expected_pool_digest:
        raise ValueError("ranking pool-sequence digest mismatch")
    digest_metadata = {
        key: value for key, value in metadata.items()
        if key not in {"ranking_sha256", "pool_sequence_sha256"}
    }
    expected_ranking_digest = tensor_digest(indices, {
        **digest_metadata, "pool_sequence_sha256": expected_pool_digest,
    })
    if metadata.get("ranking_sha256") != expected_ranking_digest:
        raise ValueError("ranking digest mismatch")
    assigned = association["pool_ids"][indices]
    if not torch.equal(assigned, pools):
        raise ValueError("ranking pool sequence is not aligned with association assignments")
    return metadata


def recompute_and_verify(ranking, association, dense_payload, dense_path, scene_id):
    metadata = validate_ranking(
        ranking, association, dense_path, scene_id, ranking["ranking"].get("selection_method")
    )
    expected = build_ranking(
        association,
        dense_payload,
        scene_id,
        metadata["selection_method"],
        metadata["policy"],
        metadata.get("global_seed") or 0,
        selection_version=metadata["selection_version"],
        seed_version=metadata.get("seed_version") or metadata["selection_version"],
        ranking_set=metadata["ranking_set"],
    )
    if expected["ranking"] != metadata:
        raise ValueError("recomputed ranking metadata mismatch")
    if not torch.equal(expected["selected_idx"], ranking["selected_idx"]):
        raise ValueError("recomputed ranking indices mismatch")
    if not torch.equal(expected["assigned_pool_sequence"], ranking["assigned_pool_sequence"]):
        raise ValueError("recomputed ranking pool sequence mismatch")
    return metadata


def diagnostics(indices, association):
    pool_ids = association["pool_ids"]
    mass = association["total_visible_mass"].double()
    foreground = sorted(int(value) for value in torch.unique(pool_ids).tolist() if value != 0)
    selected_pools = set(int(value) for value in pool_ids[indices].tolist())
    sizes = {
        pool: float(mass[pool_ids == pool].sum().item()) for pool in foreground
    }
    small_count = max(1, math.ceil(len(foreground) / 4)) if foreground else 0
    small = set(sorted(foreground, key=lambda pool: (sizes[pool], pool))[:small_count])
    total_mass = float(mass.sum().item())
    bg_total = float(mass[pool_ids == BACKGROUND_POOL].sum().item())
    selected_mass = float(mass[indices].sum().item())
    selected_bg = float(mass[indices[pool_ids[indices] == BACKGROUND_POOL]].sum().item())
    return {
        "foreground_instance_coverage": (
            len(selected_pools.intersection(foreground)) / len(foreground) if foreground else 1.0
        ),
        "small_instance_coverage": (
            len(selected_pools.intersection(small)) / len(small) if small else 1.0
        ),
        "retained_visible_contribution": selected_mass / total_mass if total_mass else 0.0,
        "retained_structural_contribution": selected_bg / bg_total if bg_total else 1.0,
    }
