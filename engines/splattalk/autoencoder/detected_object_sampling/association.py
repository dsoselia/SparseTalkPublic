#!/usr/bin/env python3
"""Associate RGB detections across views through Gaussian alpha contributions."""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parent
AUTOENCODER = PACKAGE.parent
REPOSITORY = AUTOENCODER.parent
for path in (str(AUTOENCODER), str(REPOSITORY)):
    if path not in sys.path:
        sys.path.insert(0, path)

from sparsity_utils import normalize_payload, sha256_file as payload_digest, validate_aligned
from detected_object_sampling import BACKGROUND_POOL, SELECTION_VERSION
from detected_object_sampling.common import (
    atomic_json,
    atomic_torch_save,
    canonical_digest,
    require_h200,
    sha256_file,
)
from detected_object_sampling.detection import validate_detection_scene


SUPPORT_MASS_FRACTION = 0.95
MAX_SUPPORT = 2048
MAX_TRACKS = 512
WEIGHT_SCALE = 1_000_000


def association_digest(payload, metadata):
    tensors = torch.cat([
        payload["pool_ids"].to(torch.float64),
        payload["confidence"].double(),
        payload["total_visible_mass"].double(),
        payload["winning_instance_mass"].double(),
    ])
    return canonical_digest({
        "selection_version": SELECTION_VERSION,
        "scene_id": metadata["scene_id"],
        "frame_digest": metadata["frame_digest"],
        "input_digests": metadata["input_digests"],
        "track_digest": metadata["track_digest"],
        "tensor_sha256": __import__("hashlib").sha256(
            tensors.contiguous().numpy().tobytes()
        ).hexdigest(),
    })


def load_cameras(scene_root, detection_manifest):
    scene_root = Path(scene_root)
    colors = sorted((scene_root / "color").glob("*.jpg"), key=lambda path: int(path.stem))
    poses_path = scene_root / "extrinsics.npy"
    intrinsic_path = scene_root / "intrinsic/intrinsic_color.txt"
    poses = torch.from_numpy(np.load(poses_path)).float()
    intrinsic = torch.from_numpy(np.loadtxt(intrinsic_path)[:3, :3]).float()
    if poses.shape != (len(colors), 4, 4):
        raise ValueError("RGB and camera-pose counts differ")
    selected = {}
    for frame in detection_manifest["frames"]:
        ordinal = frame["ordinal"]
        if ordinal < 0 or ordinal >= len(colors) or colors[ordinal].stem != frame["frame_stem"]:
            raise ValueError("detection frame and sorted RGB ordinal are misaligned")
        if sha256_file(colors[ordinal]) != frame["image_sha256"]:
            raise ValueError("detection source RGB digest changed")
        pose = poses[ordinal]
        if not torch.isfinite(pose).all():
            raise ValueError("selected camera pose is non-finite")
        selected[frame["frame_stem"]] = pose
    with __import__("PIL.Image", fromlist=["Image"]).open(colors[0]) as image:
        width, height = image.size
    intrinsic[0] /= width
    intrinsic[1] /= height
    if not torch.isfinite(intrinsic).all():
        raise ValueError("camera intrinsic is non-finite")
    return selected, intrinsic, {
        "extrinsics": sha256_file(poses_path),
        "intrinsic": sha256_file(intrinsic_path),
    }


def proposal_channels(proposal_map, proposal_count):
    if proposal_map.dtype != np.uint16 or proposal_map.ndim != 2:
        raise ValueError("proposal map must be a 2D uint16 array")
    if set(np.unique(proposal_map).tolist()) != set(range(proposal_count + 1)):
        raise ValueError("proposal map IDs do not match frame metadata")
    channels = np.stack(
        [proposal_map == map_id for map_id in range(proposal_count + 1)], axis=0
    ).astype(np.float32)
    if not np.array_equal(channels.sum(axis=0), np.ones(proposal_map.shape, dtype=np.float32)):
        raise ValueError("disjoint proposal channels do not form a pixel partition")
    return torch.from_numpy(channels)


def render_feature_image(payload, extrinsic, intrinsic, feature_values, image_shape):
    from src.model.decoder.cuda_splatting import render_cuda

    device = feature_values.device
    count = len(payload["points"])
    sh = torch.zeros((1, count, 3, 1), dtype=torch.float32, device=device)
    _, _, rendered = render_cuda(
        extrinsic[None].to(device=device, dtype=torch.float32),
        intrinsic[None].to(device=device, dtype=torch.float32),
        torch.tensor([0.5], dtype=torch.float32, device=device),
        torch.tensor([15.0], dtype=torch.float32, device=device),
        image_shape,
        torch.zeros((1, 3), dtype=torch.float32, device=device),
        payload["points"][None].to(device=device, dtype=torch.float32),
        payload["covariances"][None].to(device=device, dtype=torch.float32),
        sh,
        payload["opacities"][None].to(device=device, dtype=torch.float32),
        feature_values[None],
        scale_invariant=True,
        use_sh=False,
    )
    if not torch.isfinite(rendered).all():
        raise ValueError("rasterizer produced non-finite output")
    return rendered[0]


def frame_contributions(payload, extrinsic, intrinsic, channels, image_shape):
    count = len(payload["points"])
    channel_count = channels.shape[0]
    if channel_count > 256:
        raise ValueError("detector proposals exceed rasterizer feature channels")
    features = torch.zeros(
        (count, 256), dtype=torch.float32, device="cuda", requires_grad=True
    )
    rendered = render_feature_image(payload, extrinsic, intrinsic, features, image_shape)
    if rendered.shape != (256, *image_shape):
        raise ValueError(f"unexpected rendered feature shape: {tuple(rendered.shape)}")
    masks = channels.to(device="cuda", dtype=torch.float32)
    loss = (rendered[:channel_count] * masks).sum()
    gradient = torch.autograd.grad(loss, features, create_graph=False)[0][:, :channel_count]
    if not torch.isfinite(gradient).all():
        raise ValueError("non-finite rasterizer feature gradient")
    minimum = float(gradient.min().item())
    if minimum < -1e-6:
        raise ValueError(f"negative alpha contribution below tolerance: {minimum}")
    return gradient.clamp_min_(0).detach()


def validate_frame_gradients(payload, extrinsic, intrinsic, channels, image_shape, contribution):
    channel_count = channels.shape[0]
    count = len(payload["points"])
    direct_features = torch.zeros(
        (count, 256), dtype=torch.float32, device="cuda", requires_grad=True
    )
    rendered = render_feature_image(payload, extrinsic, intrinsic, direct_features, image_shape)
    direct_loss = rendered[0].sum()
    direct = torch.autograd.grad(direct_loss, direct_features)[0][:, 0]
    torch.testing.assert_close(
        contribution.sum(dim=1), direct, rtol=1e-5, atol=1e-6,
        msg="proposal-partition alpha masses differ from the all-pixel render",
    )
    total = contribution.sum(dim=1)
    candidates = torch.argsort(total, descending=True, stable=True)[:3].tolist()
    epsilon = 1e-3
    masks = channels.to(device="cuda", dtype=torch.float32)
    for gaussian_index in candidates:
        channel = int(torch.argmax(contribution[gaussian_index]).item())
        values = []
        for sign in (-1.0, 1.0):
            features = torch.zeros((count, 256), dtype=torch.float32, device="cuda")
            features[gaussian_index, channel] = sign * epsilon
            output = render_feature_image(payload, extrinsic, intrinsic, features, image_shape)
            values.append(float((output[:channel_count] * masks).sum().item()))
        finite_difference = (values[1] - values[0]) / (2 * epsilon)
        analytic = float(contribution[gaussian_index, channel].item())
        if not np.isclose(finite_difference, analytic, rtol=2e-3, atol=2e-4):
            raise ValueError(
                "rasterizer finite-difference validation failed: "
                f"g={gaussian_index}, c={channel}, analytic={analytic}, fd={finite_difference}"
            )


def sparse_support(values, mass_fraction=SUPPORT_MASS_FRACTION, maximum=MAX_SUPPORT):
    values = values.detach().cpu().double()
    if values.ndim != 1 or not torch.isfinite(values).all() or values.min().item() < 0:
        raise ValueError("proposal contribution must be a finite nonnegative vector")
    total = float(values.sum().item())
    if total <= 0:
        return {
            "indices": torch.empty(0, dtype=torch.int64),
            "weights": torch.empty(0, dtype=torch.int32),
            "raw_mass": 0.0,
        }
    order = torch.argsort(values, descending=True, stable=True)
    positive = order[values[order] > 0]
    cumulative = torch.cumsum(values[positive], dim=0)
    count = int(torch.searchsorted(cumulative, total * mass_fraction, right=False).item()) + 1
    count = min(count, maximum, len(positive))
    indices = positive[:count].to(torch.int64)
    normalized = values[indices] / total
    quantized = torch.round(normalized * WEIGHT_SCALE).to(torch.int64)
    keep = quantized > 0
    indices = indices[keep]
    quantized = quantized[keep]
    if not len(indices):
        indices = positive[:1].to(torch.int64)
        quantized = torch.ones(1, dtype=torch.int64)
    stable = torch.argsort(indices, stable=True)
    return {
        "indices": indices[stable],
        "weights": quantized[stable].to(torch.int32),
        "raw_mass": total,
    }


def weighted_jaccard(first_indices, first_weights, second_indices, second_weights):
    first_indices = np.asarray(first_indices, dtype=np.int64)
    second_indices = np.asarray(second_indices, dtype=np.int64)
    first_weights = np.asarray(first_weights, dtype=np.int64)
    second_weights = np.asarray(second_weights, dtype=np.int64)
    i = j = 0
    intersection = 0
    union = 0
    while i < len(first_indices) and j < len(second_indices):
        if first_indices[i] == second_indices[j]:
            intersection += min(first_weights[i], second_weights[j])
            union += max(first_weights[i], second_weights[j])
            i += 1
            j += 1
        elif first_indices[i] < second_indices[j]:
            union += first_weights[i]
            i += 1
        else:
            union += second_weights[j]
            j += 1
    union += int(first_weights[i:].sum()) + int(second_weights[j:].sum())
    return intersection / union if union else 0.0


def track_label(track):
    return min(track["label_mass"], key=lambda label: (-track["label_mass"][label], label))


def refresh_match_support(track):
    if not track["contributions"]:
        track["match_indices"] = np.empty(0, dtype=np.int64)
        track["match_weights"] = np.empty(0, dtype=np.int32)
        return
    indices = torch.tensor(sorted(track["contributions"]), dtype=torch.int64)
    values = torch.tensor([track["contributions"][int(index)] for index in indices], dtype=torch.float64)
    support = sparse_support(values)
    track["match_indices"] = indices[support["indices"]].numpy()
    track["match_weights"] = support["weights"].numpy()


def add_proposal(track, proposal, frame_index):
    raw_mass = float(proposal["support"]["raw_mass"])
    track["visible_mass"] += raw_mass
    track["label_mass"][proposal["canonical_label"]] += raw_mass
    track["observations"] += 1
    track["last_frame"] = frame_index
    for index, weight in zip(
        proposal["support"]["indices"].tolist(), proposal["support"]["weights"].tolist()
    ):
        track["contributions"][int(index)] += raw_mass * int(weight) / WEIGHT_SCALE
    refresh_match_support(track)


def new_track(track_id, proposal, frame_index):
    track = {
        "track_id": int(track_id),
        "first_frame": int(frame_index),
        "last_frame": int(frame_index),
        "observations": 0,
        "visible_mass": 0.0,
        "label_mass": defaultdict(float),
        "contributions": defaultdict(float),
        "match_indices": np.empty(0, dtype=np.int64),
        "match_weights": np.empty(0, dtype=np.int32),
    }
    add_proposal(track, proposal, frame_index)
    return track


def match_proposals(proposals, tracks):
    if not proposals or not tracks:
        return {}, set()
    from scipy.optimize import linear_sum_assignment

    ordered_tracks = sorted(tracks.values(), key=lambda value: value["track_id"])
    proposal_count = len(proposals)
    track_count = len(ordered_tracks)
    invalid = -(10**18)
    scores = np.full((proposal_count, track_count + proposal_count), invalid, dtype=np.int64)
    for row in range(proposal_count):
        scores[row, track_count + row] = 0
    similarities = {}
    for row, proposal in enumerate(proposals):
        p_indices = proposal["support"]["indices"].numpy()
        p_weights = proposal["support"]["weights"].numpy()
        for column, track in enumerate(ordered_tracks):
            similarity = weighted_jaccard(
                p_indices, p_weights, track["match_indices"], track["match_weights"]
            )
            threshold = 0.15 if proposal["canonical_label"] == track_label(track) else 0.40
            quantized = int(round(similarity * WEIGHT_SCALE))
            if quantized < int(round(threshold * WEIGHT_SCALE)):
                continue
            tie = row * 20_000 + track["track_id"]
            scores[row, column] = quantized * 1_000_000 - tie
            similarities[(row, column)] = quantized
    rows, columns = linear_sum_assignment(-scores)
    matches = {}
    matched_tracks = set()
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column < track_count and (row, column) in similarities:
            track_id = ordered_tracks[column]["track_id"]
            matches[row] = track_id
            matched_tracks.add(track_id)
    return matches, matched_tracks


def prune_tracks(tracks, maximum=MAX_TRACKS):
    if len(tracks) <= maximum:
        return [], 0.0
    ordered = sorted(tracks.values(), key=lambda value: (-value["visible_mass"], value["track_id"]))
    keep = {track["track_id"] for track in ordered[:maximum]}
    removed = [track_id for track_id in sorted(tracks) if track_id not in keep]
    removed_mass = sum(tracks[track_id]["visible_mass"] for track_id in removed)
    for track_id in removed:
        del tracks[track_id]
    return removed, removed_mass


def assign_gaussians(tracks, total_mass, confidence_threshold):
    count = len(total_mass)
    winner_mass = torch.zeros(count, dtype=torch.float64)
    winner_track = torch.zeros(count, dtype=torch.int32)
    for track_id in sorted(tracks):
        track = tracks[track_id]
        if not track["contributions"]:
            continue
        indices = torch.tensor(sorted(track["contributions"]), dtype=torch.int64)
        values = torch.tensor(
            [track["contributions"][int(index)] for index in indices], dtype=torch.float64
        )
        current = winner_mass[indices]
        replace = values > current
        winner_mass[indices[replace]] = values[replace]
        winner_track[indices[replace]] = int(track_id)
    # Quantized sparse supports can overshoot the per-Gaussian all-pixel mass by
    # a small amount. Preserve the physical invariant before computing confidence.
    winner_mass = torch.minimum(winner_mass, total_mass.double().clamp_min(0))
    raw_confidence = winner_mass / total_mass.double().clamp_min(torch.finfo(torch.float64).tiny)
    confidence = torch.round(raw_confidence * WEIGHT_SCALE) / WEIGHT_SCALE
    accepted = (total_mass > 0) & (confidence >= confidence_threshold) & (winner_track > 0)
    pool_ids = torch.where(accepted, winner_track, torch.zeros_like(winner_track))
    confidence = torch.where(accepted, confidence, torch.zeros_like(confidence))
    accepted_mass = torch.where(accepted, winner_mass, torch.zeros_like(winner_mass))
    return pool_ids.to(torch.int32), confidence.float(), accepted_mass.float()


def build_association(
    scene_id,
    dense_path,
    scene_root,
    detection_root,
    output_path,
    confidence_threshold=0.5,
    validate_gradients=False,
):
    hardware = require_h200()
    dense_path = Path(dense_path)
    if not dense_path.is_file():
        raise FileNotFoundError(dense_path)
    detection_scene = Path(detection_root) / scene_id
    detection = validate_detection_scene(detection_scene, scene_id)
    payload = normalize_payload(torch.load(dense_path, map_location="cpu", weights_only=False))
    source_count = validate_aligned(payload, expected_feature_dim=256)
    poses, intrinsic, camera_digests = load_cameras(scene_root, detection)
    image_shape = tuple(detection["image_shape"])
    render_payload = {
        "points": payload["points"].float().cuda(non_blocking=True),
        "covariances": payload["covariances"].float().cuda(non_blocking=True),
        "opacities": payload["opacities"].float().cuda(non_blocking=True),
    }
    tracks = {}
    next_track_id = 1
    total_mass = torch.zeros(source_count, dtype=torch.float64)
    overflow_track_count = 0
    overflow_visible_mass = 0.0
    frame_stats = []
    for frame_index, frame in enumerate(detection["frames"]):
        stem = frame["frame_stem"]
        with np.load(detection_scene / "frames" / f"{stem}.npz", allow_pickle=False) as data:
            proposal_map = data["proposal_map"]
        frame_metadata = json.loads(
            (detection_scene / "frames" / f"{stem}.json").read_text()
        )
        channels = proposal_channels(proposal_map, len(frame_metadata["proposals"]))
        contribution = frame_contributions(
            render_payload, poses[stem], intrinsic, channels, image_shape
        )
        if validate_gradients and frame_index == 0:
            validate_frame_gradients(
                render_payload, poses[stem], intrinsic, channels, image_shape, contribution
            )
        frame_total = contribution.sum(dim=1).cpu().double()
        total_mass.add_(frame_total)
        proposals = []
        for proposal_index, metadata in enumerate(frame_metadata["proposals"], 1):
            support = sparse_support(contribution[:, proposal_index])
            if support["raw_mass"] <= 0:
                continue
            proposals.append({
                "canonical_label": metadata["canonical_label"],
                "map_id": metadata["map_id"],
                "proposal_order": proposal_index - 1,
                "support": support,
            })
        matches, _ = match_proposals(proposals, tracks)
        matched_count = 0
        proposal_tracks = {}
        for proposal_index, proposal in enumerate(proposals):
            if proposal_index in matches:
                track_id = matches[proposal_index]
                add_proposal(tracks[track_id], proposal, frame_index)
                matched_count += 1
            else:
                track_id = next_track_id
                tracks[track_id] = new_track(track_id, proposal, frame_index)
                next_track_id += 1
            proposal_tracks[str(proposal["map_id"])] = track_id
        removed, removed_mass = prune_tracks(tracks)
        overflow_track_count += len(removed)
        overflow_visible_mass += removed_mass
        frame_stats.append({
            "frame_stem": stem,
            "proposal_count": len(frame_metadata["proposals"]),
            "contributing_proposal_count": len(proposals),
            "matched_count": matched_count,
            "new_track_count": len(proposals) - matched_count,
            "active_track_count": len(tracks),
            "pruned_track_count": len(removed),
            "proposal_track_ids": proposal_tracks,
        })
        del contribution, frame_total
        torch.cuda.empty_cache()
    if not tracks:
        raise ValueError(f"scene has no detected foreground tracks: {scene_id}")
    pool_ids, confidence, winning_mass = assign_gaussians(
        tracks, total_mass, confidence_threshold
    )
    foreground_pool_ids = sorted(
        int(value) for value in torch.unique(pool_ids).tolist() if value != BACKGROUND_POOL
    )
    if not foreground_pool_ids:
        raise ValueError(f"scene has no confidently assigned foreground track: {scene_id}")
    track_records = []
    for track_id in sorted(tracks):
        track = tracks[track_id]
        selected = pool_ids == track_id
        track_records.append({
            "track_id": track_id,
            "canonical_label": track_label(track),
            "label_mass": dict(sorted(track["label_mass"].items())),
            "first_frame": track["first_frame"],
            "last_frame": track["last_frame"],
            "observations": track["observations"],
            "persistence": track["observations"] / len(detection["frames"]),
            "accumulated_visible_mass": track["visible_mass"],
            "support_size": len(track["contributions"]),
            "match_support_size": len(track["match_indices"]),
            "assigned_gaussian_count": int(selected.sum().item()),
            "assigned_visible_mass": float(total_mass[selected].sum().item()),
        })
    background = pool_ids == BACKGROUND_POOL
    pool_stats = [{
        "pool_id": BACKGROUND_POOL,
        "canonical_label": "structural/background",
        "gaussian_count": int(background.sum().item()),
        "visible_mass": float(total_mass[background].sum().item()),
    }]
    pool_stats.extend({
        "pool_id": record["track_id"],
        "canonical_label": record["canonical_label"],
        "gaussian_count": record["assigned_gaussian_count"],
        "visible_mass": record["assigned_visible_mass"],
    } for record in track_records if record["track_id"] in foreground_pool_ids)
    frame_record = [{
        "ordinal": frame["ordinal"],
        "frame_stem": frame["frame_stem"],
        "image_sha256": frame["image_sha256"],
    } for frame in detection["frames"]]
    track_identity = [{
        key: record[key] for key in (
            "track_id", "canonical_label", "first_frame", "last_frame", "observations",
            "support_size", "match_support_size", "assigned_gaussian_count",
        )
    } for record in track_records]
    track_digest = canonical_digest(track_identity)
    metadata = {
        "selection_version": SELECTION_VERSION,
        "scene_id": scene_id,
        "source_count": source_count,
        "frame_count": len(frame_record),
        "image_shape": list(image_shape),
        "near": 0.5,
        "far": 15.0,
        "confidence_threshold": float(confidence_threshold),
        "association_quantization": 1e-6,
        "proposal_support_mass_fraction": SUPPORT_MASS_FRACTION,
        "maximum_proposal_support": MAX_SUPPORT,
        "same_label_match_threshold": 0.15,
        "different_label_match_threshold": 0.40,
        "maximum_tracks": MAX_TRACKS,
        "frame_record": frame_record,
        "frame_digest": canonical_digest(frame_record),
        "input_digests": {
            "dense_encoded": payload_digest(dense_path),
            "detection": detection["detection_sha256"],
            **camera_digests,
        },
        "detector_models": detection["models"],
        "foreground_pool_ids": foreground_pool_ids,
        "detected_track_count": len(tracks),
        "assigned_foreground_track_count": len(foreground_pool_ids),
        "overflow_track_count": overflow_track_count,
        "overflow_visible_mass": overflow_visible_mass,
        "background_visible_mass_fraction": (
            float(total_mass[background].sum().item() / total_mass.sum().item())
            if total_mass.sum().item() else 1.0
        ),
        "pool_stats": pool_stats,
        "tracks": track_records,
        "track_identity": track_identity,
        "track_digest": track_digest,
        "frame_stats": frame_stats,
        "gradient_validation": bool(validate_gradients),
        "hardware": hardware,
    }
    result = {
        "pool_ids": pool_ids,
        "confidence": confidence,
        "total_visible_mass": total_mass.float(),
        "winning_instance_mass": winning_mass,
        "association": metadata,
    }
    metadata["association_sha256"] = association_digest(result, metadata)
    atomic_torch_save(output_path, result)
    atomic_json(Path(output_path).with_suffix(".json"), metadata)
    return result


def validate_association(payload, scene_id, dense_path=None):
    required = {
        "pool_ids", "confidence", "total_visible_mass", "winning_instance_mass", "association"
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("association payload fields mismatch")
    metadata = payload["association"]
    count = metadata.get("source_count")
    if metadata.get("selection_version") != SELECTION_VERSION or metadata.get("scene_id") != scene_id:
        raise ValueError("association identity mismatch")
    if not isinstance(count, int) or count <= 0:
        raise ValueError("invalid association source count")
    for key in ("pool_ids", "confidence", "total_visible_mass", "winning_instance_mass"):
        value = payload[key]
        if not torch.is_tensor(value) or value.ndim != 1 or len(value) != count:
            raise ValueError(f"invalid association tensor: {key}")
        if not torch.isfinite(value).all():
            raise ValueError(f"non-finite association tensor: {key}")
    if payload["pool_ids"].dtype != torch.int32 or payload["pool_ids"].min().item() < 0:
        raise ValueError("pool IDs must be nonnegative int32")
    if payload["confidence"].min().item() < 0 or payload["confidence"].max().item() > 1:
        raise ValueError("association confidence is outside [0, 1]")
    if payload["total_visible_mass"].min().item() < 0:
        raise ValueError("visible mass must be nonnegative")
    if (payload["winning_instance_mass"] < 0).any() or (
        payload["winning_instance_mass"] > payload["total_visible_mass"] + 1e-5
    ).any():
        raise ValueError("winning-track mass is outside valid bounds")
    foreground = set(metadata.get("foreground_pool_ids", []))
    observed = set(int(value) for value in torch.unique(payload["pool_ids"]).tolist())
    if observed != foreground | {BACKGROUND_POOL}:
        raise ValueError("association foreground pool metadata is incomplete")
    if not 1 <= metadata.get("detected_track_count", 0) <= MAX_TRACKS:
        raise ValueError("detected track count is invalid")
    if metadata.get("frame_digest") != canonical_digest(metadata.get("frame_record")):
        raise ValueError("association frame digest mismatch")
    if metadata.get("track_digest") != canonical_digest(metadata.get("track_identity")):
        raise ValueError("association track digest mismatch")
    if metadata.get("association_sha256") != association_digest(payload, metadata):
        raise ValueError("association payload digest mismatch")
    forbidden = {"aggregation", "instance_masks_zip", "sensor_data"}
    if forbidden.intersection(metadata.get("input_digests", {})):
        raise ValueError("detected association contains forbidden oracle inputs")
    if dense_path is not None and metadata["input_digests"]["dense_encoded"] != sha256_file(dense_path):
        raise ValueError("association dense-source digest mismatch")
    return metadata


def compare_duplicate_associations(first, second):
    if not torch.equal(first["pool_ids"], second["pool_ids"]):
        raise ValueError("duplicate association mismatch: pool_ids")
    # CUDA atomic accumulation requires tolerances for continuous masses.
    torch.testing.assert_close(first["confidence"], second["confidence"], rtol=1e-3, atol=1.1e-6)
    torch.testing.assert_close(
        first["total_visible_mass"], second["total_visible_mass"], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        first["winning_instance_mass"], second["winning_instance_mass"],
        rtol=1e-3, atol=1e-6,
    )
    first_meta = first["association"]
    second_meta = second["association"]
    if first_meta["track_digest"] != second_meta["track_digest"]:
        raise ValueError("duplicate association track metadata mismatch")
    return {
        "pool_assignments_exact": True,
        "track_metadata_exact": True,
        "confidence_tolerance": {"rtol": 1e-3, "atol": 1.1e-6},
        "total_mass_tolerance": {"rtol": 1e-5, "atol": 1e-6},
        "winning_mass_tolerance": {"rtol": 1e-3, "atol": 1e-6},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--scene-data-root", required=True)
    parser.add_argument("--detection-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--validate-gradients", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    require_h200()
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("confidence threshold must be in [0, 1]")
    output = Path(args.output)
    dense_path = Path(args.dense_root) / args.scene / "feat_fs" / f"{args.scene}.pt"
    if output.exists() and not args.overwrite:
        existing = torch.load(output, map_location="cpu", weights_only=False)
        validate_association(existing, args.scene, dense_path)
        result = existing
    else:
        result = build_association(
            args.scene,
            dense_path,
            Path(args.scene_data_root) / args.scene,
            args.detection_root,
            output,
            args.confidence_threshold,
            args.validate_gradients,
        )
    print(json.dumps(result["association"], sort_keys=True))


if __name__ == "__main__":
    main()
