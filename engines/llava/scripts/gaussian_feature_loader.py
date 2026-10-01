import hashlib

import torch

import gaussian_utils
from preselected_gaussians import validate_and_prepare


def prepare_single_payload(payload, scene_id, ntokens, selection_mode):
    if selection_mode == "preselected":
        return validate_and_prepare(payload, scene_id, ntokens)
    if selection_mode != "entropy":
        raise ValueError(f"unknown Gaussian selection mode: {selection_mode}")

    if isinstance(payload, dict):
        image_features = payload["features"].half()
        xyz = payload["points"]
        if isinstance(xyz, list):
            xyz = torch.cat(xyz, dim=0)
    else:
        image_features = payload
        xyz = torch.zeros(image_features.shape[0], 3, device=image_features.device)
    source_count = len(image_features)
    original_indices = torch.arange(source_count, dtype=torch.int64)
    invalid = torch.logical_or(torch.isnan(image_features), torch.isinf(image_features))
    invalid_rows = torch.unique(torch.where(invalid)[0])
    mask = torch.ones(len(image_features), dtype=torch.bool)
    mask[invalid_rows] = False
    image_features = image_features[mask]
    xyz = xyz[mask]
    original_indices = original_indices[mask]

    if image_features.ndim > 2:
        indices = torch.randperm(len(image_features))[:44]
        image_features = image_features[indices]
    if image_features.ndim == 2:
        sample_gaussians = min(ntokens, len(image_features) // 729)
        indices = gaussian_utils.select_high_entropy_gaussians(
            image_features, sample_gaussians * 729
        )
        image_features = image_features[indices]
        xyz = xyz[indices]
        consumed_indices = original_indices[indices]
        image_features = image_features.reshape(-1, 27, 27, 3584).permute(0, 3, 1, 2)
        xyz.reshape(-1, 27, 27, 3).permute(0, 3, 1, 2)
        index_bytes = consumed_indices.numpy().astype("<i8", copy=False).tobytes()
        audit = {
            "selection_version": "published_entropy_v1",
            "selection_method": "entropy",
            "scene_id": scene_id,
            "source_count": source_count,
            "valid_source_count": len(original_indices),
            "retained_count": len(original_indices),
            "requested_budget": ntokens * 729,
            "consumed_count": len(consumed_indices),
            "consumed_selected_idx": consumed_indices.tolist(),
            "selected_idx_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "score": "softmax_entropy_topk",
            "block_count": sample_gaussians,
            "tensor_layout": "legacy_27x27",
        }
        return image_features, audit
    return image_features, None
