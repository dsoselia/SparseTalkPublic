import hashlib
import math

import torch


METHODS = (
    "uniform_random",
    "voxel_random",
    "fps",
    "semantic_kcenter",
    "joint_kcenter",
    "adaptive_anchors",
)
METHOD_VERSION = "coverage_aware_anchors_v1"
SELECTION_INPUTS = ["points", "features"]


def stable_seed(scene_id, method, budget, global_seed):
    value = f"{scene_id}|{method}|{budget}|{global_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def normalize_inputs(points, features):
    return normalize_xyz(points), normalize_embeddings(features)


def normalize_xyz(points):
    xyz_min = points.amin(dim=0)
    xyz_span = (points.amax(dim=0) - xyz_min).clamp_min(1e-12)
    return (points - xyz_min) / xyz_span


def normalize_embeddings(features):
    return torch.nn.functional.normalize(features, dim=1, eps=1e-12)


def voxelize(xyz, voxel_size):
    if not 0 < voxel_size <= 1:
        raise ValueError("voxel_size must be in (0, 1]")
    max_bin = max(0, math.ceil(1 / voxel_size) - 1)
    coordinates = torch.floor(xyz / voxel_size).to(torch.int64).clamp_(0, max_bin)
    voxels, inverse, counts = torch.unique(
        coordinates, dim=0, sorted=True, return_inverse=True, return_counts=True
    )
    return voxels, inverse, counts


def voxel_statistics(embeddings, inverse, counts):
    order = torch.argsort(inverse, stable=True)
    sorted_embeddings = embeddings[order]
    centroids = torch.segment_reduce(sorted_embeddings, reduce="mean", lengths=counts)
    centroids = torch.nn.functional.normalize(centroids, dim=1, eps=1e-12)
    distances = 1 - (embeddings * centroids[inverse]).sum(dim=1)
    sorted_distances = distances[order]
    variances = torch.segment_reduce(sorted_distances, reduce="mean", lengths=counts)
    min_distance = torch.segment_reduce(sorted_distances, reduce="min", lengths=counts)
    original_indices = torch.arange(len(embeddings), device=embeddings.device, dtype=torch.int64)
    sentinel = torch.full_like(original_indices, len(embeddings))
    tied = torch.where(
        torch.isclose(distances, min_distance[inverse], rtol=1e-6, atol=1e-8),
        original_indices,
        sentinel,
    )
    representatives = torch.segment_reduce(
        tied[order].to(torch.float64), reduce="min", lengths=counts
    ).to(torch.int64)
    if (representatives == len(embeddings)).any():
        raise RuntimeError("failed to choose a representative for every voxel")
    return representatives, variances


def _deterministic_argmax(scores, original_indices, available):
    masked = scores.masked_fill(~available, -torch.inf)
    maximum = masked.max()
    tied = available & torch.isclose(masked, maximum, rtol=1e-6, atol=1e-8)
    return torch.where(tied, original_indices, original_indices.max() + 1).min()


def kcenter(pool_indices, xyz, embeddings, budget, alpha):
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if budget <= 0 or budget > len(pool_indices):
        raise ValueError("k-center budget must fit candidate pool")
    pool_indices = pool_indices.to(torch.int64)
    pool_xyz = xyz[pool_indices]
    pool_embeddings = embeddings[pool_indices]
    global_centroid = torch.nn.functional.normalize(
        pool_embeddings.mean(dim=0, keepdim=True), dim=1, eps=1e-12
    )[0]
    centroid_distance = 1 - pool_embeddings @ global_centroid
    first_local = torch.where(
        torch.isclose(centroid_distance, centroid_distance.min(), rtol=1e-6, atol=1e-8),
        pool_indices,
        pool_indices.max() + 1,
    ).min()
    first_position = torch.where(pool_indices == first_local)[0][0]
    available = torch.ones(len(pool_indices), dtype=torch.bool, device=pool_indices.device)
    minimum_spatial = torch.full(
        (len(pool_indices),), torch.inf, dtype=torch.float32, device=pool_indices.device
    )
    minimum_semantic = minimum_spatial.clone()
    selected = []
    position = first_position
    for _ in range(budget):
        original = pool_indices[position]
        selected.append(original)
        available[position] = False
        spatial = torch.linalg.vector_norm(pool_xyz - pool_xyz[position], dim=1) / math.sqrt(3)
        semantic = (1 - pool_embeddings @ pool_embeddings[position]) / 2
        torch.minimum(minimum_spatial, spatial, out=minimum_spatial)
        torch.minimum(minimum_semantic, semantic, out=minimum_semantic)
        if len(selected) < budget:
            score = alpha * minimum_spatial + (1 - alpha) * minimum_semantic
            next_original = _deterministic_argmax(score, pool_indices, available)
            position = torch.where(pool_indices == next_original)[0][0]
    return torch.stack(selected)


def farthest_point_sampling(xyz, budget):
    indices = torch.arange(len(xyz), device=xyz.device, dtype=torch.int64)
    centroid = xyz.mean(dim=0)
    first_distance = torch.linalg.vector_norm(xyz - centroid, dim=1)
    first = _deterministic_argmax(first_distance, indices, torch.ones_like(indices, dtype=torch.bool))
    available = torch.ones(len(xyz), dtype=torch.bool, device=xyz.device)
    minimum = torch.full((len(xyz),), torch.inf, device=xyz.device)
    selected = []
    position = first
    for _ in range(budget):
        selected.append(position)
        available[position] = False
        distance = torch.linalg.vector_norm(xyz - xyz[position], dim=1)
        torch.minimum(minimum, distance, out=minimum)
        if len(selected) < budget:
            position = _deterministic_argmax(minimum, indices, available)
    return torch.stack(selected)


def voxel_stratified_random(inverse, counts, budget, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    inverse_cpu = inverse.cpu()
    members = [torch.where(inverse_cpu == voxel)[0] for voxel in range(len(counts))]
    voxel_order = torch.randperm(len(members), generator=generator).tolist()
    for voxel in range(len(members)):
        order = torch.randperm(len(members[voxel]), generator=generator)
        members[voxel] = members[voxel][order]
    selected = []
    offsets = [0] * len(members)
    while len(selected) < budget:
        progressed = False
        for voxel in voxel_order:
            if offsets[voxel] < len(members[voxel]):
                selected.append(members[voxel][offsets[voxel]])
                offsets[voxel] += 1
                progressed = True
                if len(selected) == budget:
                    break
        if not progressed:
            raise RuntimeError("voxel random allocation exhausted before budget")
    return torch.stack(selected).to(inverse.device)


def uniform_random(source_count, budget, seed, device):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(source_count, generator=generator)[:budget].to(device)


def allocate_extras(capacities, weights, total):
    allocation = torch.zeros_like(capacities)
    remaining = int(total)
    active = capacities > 0
    while remaining > 0 and active.any():
        active_weights = weights.clone().clamp_min(0)
        active_weights[~active] = 0
        if active_weights.sum() == 0:
            active_weights = active.to(weights.dtype)
        quotas = active_weights / active_weights.sum() * remaining
        base = torch.floor(quotas).to(torch.int64)
        base = torch.minimum(base, capacities - allocation)
        allocation += base
        assigned = int(base.sum())
        remaining -= assigned
        active = allocation < capacities
        if remaining == 0:
            break
        remainders = quotas - torch.floor(quotas)
        order = sorted(
            torch.where(active)[0].tolist(), key=lambda i: (-float(remainders[i]), i)
        )
        if not order:
            break
        for voxel in order:
            allocation[voxel] += 1
            remaining -= 1
            if remaining == 0:
                break
        active = allocation < capacities
    if remaining:
        raise RuntimeError("adaptive allocation exhausted before budget")
    return allocation


def adaptive_selection(representatives, variances, inverse, counts, xyz, embeddings, budget, alpha):
    if len(representatives) >= budget:
        return kcenter(representatives, xyz, embeddings, budget, alpha)
    ordered_representatives = kcenter(
        representatives, xyz, embeddings, len(representatives), alpha
    )
    capacities = counts.cpu().to(torch.int64) - 1
    extras = allocate_extras(capacities, variances.cpu(), budget - len(representatives))
    selected = ordered_representatives.tolist()
    representative_by_voxel = representatives.cpu()
    inverse_cpu = inverse.cpu()
    for voxel, extra_count in enumerate(extras.tolist()):
        if extra_count == 0:
            continue
        members = torch.where(inverse_cpu == voxel)[0].to(embeddings.device)
        representative = representative_by_voxel[voxel].to(embeddings.device)
        remaining = members[members != representative]
        chosen = [representative]
        minimum = 1 - embeddings[remaining] @ embeddings[representative]
        available = torch.ones(len(remaining), dtype=torch.bool, device=embeddings.device)
        for _ in range(extra_count):
            next_original = _deterministic_argmax(minimum, remaining, available)
            position = torch.where(remaining == next_original)[0][0]
            chosen.append(next_original)
            available[position] = False
            distance = 1 - embeddings[remaining] @ embeddings[next_original]
            torch.minimum(minimum, distance, out=minimum)
        selected.extend(chosen[1:])
    if len(selected) != budget:
        raise RuntimeError(f"adaptive selector produced {len(selected)} rows, expected {budget}")
    return torch.as_tensor(selected, dtype=torch.int64, device=embeddings.device)


def select_anchors(
    points,
    features,
    budget,
    method,
    voxel_size=0.01,
    alpha=0.5,
    global_seed=0,
    scene_id="",
):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if budget <= 0 or budget > len(features):
        raise ValueError("budget must be positive and no larger than source rows")
    xyz = normalize_xyz(points.float())
    voxels, inverse, counts = voxelize(xyz, voxel_size)
    seed = stable_seed(scene_id, method, budget, global_seed)
    if method == "uniform_random":
        selected = uniform_random(len(features), budget, seed, xyz.device)
    elif method == "voxel_random":
        selected = voxel_stratified_random(inverse, counts, budget, seed)
    elif method == "fps":
        selected = farthest_point_sampling(xyz, budget)
    else:
        embeddings = normalize_embeddings(features.float())
        representatives, variances = voxel_statistics(embeddings, inverse, counts)
        if method == "semantic_kcenter":
            if len(representatives) < budget:
                raise ValueError("voxel candidate pool is smaller than budget; reduce voxel_size")
            selected = kcenter(representatives, xyz, embeddings, budget, alpha=0)
        elif method == "joint_kcenter":
            if len(representatives) < budget:
                raise ValueError("voxel candidate pool is smaller than budget; reduce voxel_size")
            selected = kcenter(representatives, xyz, embeddings, budget, alpha=alpha)
        else:
            selected = adaptive_selection(
                representatives, variances, inverse, counts, xyz, embeddings, budget, alpha
            )
    if len(torch.unique(selected)) != budget:
        raise RuntimeError("selector returned duplicate indices")
    diagnostics = {
        "occupied_voxels": len(voxels),
        "selected_voxels": len(torch.unique(inverse[selected])),
        "voxel_coverage": len(torch.unique(inverse[selected])) / len(voxels),
        "derived_seed": seed,
    }
    return selected.cpu(), diagnostics


def representation_metrics(points, features, selected_idx, voxel_size, row_chunk=2048, anchor_chunk=2048):
    """Compute nearest-anchor diagnostics without materializing an N x B matrix."""
    xyz, embeddings = normalize_inputs(points.float(), features.float())
    selected_idx = selected_idx.to(xyz.device)
    anchor_xyz = xyz[selected_idx]
    anchor_embeddings = embeddings[selected_idx]
    spatial_values = []
    semantic_values = []
    for row_start in range(0, len(xyz), row_chunk):
        row_xyz = xyz[row_start : row_start + row_chunk]
        row_embeddings = embeddings[row_start : row_start + row_chunk]
        spatial_min = torch.full((len(row_xyz),), torch.inf, device=xyz.device)
        semantic_min = torch.full_like(spatial_min, torch.inf)
        for anchor_start in range(0, len(anchor_xyz), anchor_chunk):
            xyz_block = anchor_xyz[anchor_start : anchor_start + anchor_chunk]
            embedding_block = anchor_embeddings[anchor_start : anchor_start + anchor_chunk]
            torch.minimum(
                spatial_min,
                torch.cdist(row_xyz, xyz_block).amin(dim=1) / math.sqrt(3),
                out=spatial_min,
            )
            torch.minimum(
                semantic_min,
                (1 - row_embeddings @ embedding_block.T).amin(dim=1) / 2,
                out=semantic_min,
            )
        spatial_values.append(spatial_min.cpu())
        semantic_values.append(semantic_min.cpu())
    spatial = torch.cat(spatial_values)
    semantic = torch.cat(semantic_values)

    nearest_other = []
    anchor_ids = torch.arange(len(anchor_xyz), device=xyz.device)
    for row_start in range(0, len(anchor_xyz), row_chunk):
        rows = anchor_embeddings[row_start : row_start + row_chunk]
        maximum_similarity = torch.full((len(rows),), -torch.inf, device=xyz.device)
        row_ids = anchor_ids[row_start : row_start + len(rows)]
        for anchor_start in range(0, len(anchor_embeddings), anchor_chunk):
            block = anchor_embeddings[anchor_start : anchor_start + anchor_chunk]
            similarities = rows @ block.T
            block_ids = anchor_ids[anchor_start : anchor_start + len(block)]
            similarities.masked_fill_(row_ids[:, None] == block_ids[None, :], -torch.inf)
            torch.maximum(
                maximum_similarity, similarities.amax(dim=1), out=maximum_similarity
            )
        nearest_other.append(maximum_similarity.cpu())
    redundancy = torch.cat(nearest_other)
    if len(anchor_xyz) == 1:
        redundancy.fill_(0)

    _, inverse, counts = voxelize(xyz, voxel_size)
    covered = len(torch.unique(inverse[selected_idx]))
    return {
        "mean_nearest_spatial_distance": float(spatial.mean()),
        "max_nearest_spatial_distance": float(spatial.max()),
        "mean_nearest_semantic_distance": float(semantic.mean()),
        "embedding_reconstruction_error": float(semantic.mean()),
        "mean_selected_nearest_neighbor_cosine_similarity": float(redundancy.mean()),
        "occupied_voxels": len(counts),
        "selected_voxels": covered,
        "voxel_coverage": covered / len(counts),
    }
