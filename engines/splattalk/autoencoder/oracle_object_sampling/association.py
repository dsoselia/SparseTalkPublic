#!/usr/bin/env python3
import argparse
import io
import json
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PACKAGE = Path(__file__).resolve().parent
AUTOENCODER = PACKAGE.parent
REPOSITORY = AUTOENCODER.parent
for path in (str(AUTOENCODER), str(REPOSITORY)):
    if path not in sys.path:
        sys.path.insert(0, path)

from sparsity_utils import normalize_payload, sha256_file as payload_digest, validate_aligned
from oracle_object_sampling import BACKGROUND_POOL, SELECTION_VERSION, STRUCTURAL_LABELS
from oracle_object_sampling.common import (
    atomic_json,
    atomic_torch_save,
    canonical_digest,
    require_h200,
    sha256_file,
)


def association_digest(payload, metadata):
    digest_fields = {
        "selection_version": SELECTION_VERSION,
        "scene_id": metadata["scene_id"],
        "frame_digest": metadata["frame_digest"],
        "input_digests": metadata["input_digests"],
    }
    tensors = torch.cat([
        payload["pool_ids"].to(torch.float64),
        payload["confidence"].double(),
        payload["total_visible_mass"].double(),
        payload["winning_instance_mass"].double(),
    ])
    return canonical_digest({
        **digest_fields,
        "tensor_sha256": __import__("hashlib").sha256(
            tensors.contiguous().numpy().tobytes()
        ).hexdigest(),
    })


def uniformly_spaced_indices(count, requested):
    if count < requested:
        raise ValueError(f"need at least {requested} observations, found {count}")
    values = np.rint(np.linspace(0, count - 1, requested)).astype(np.int64)
    if len(np.unique(values)) != requested:
        raise ValueError("uniform frame selection produced duplicate indices")
    return values.tolist()


def read_exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError("truncated ScanNet sensor stream")
    return value


def read_sens_cameras(path, frame_count):
    with Path(path).open("rb") as stream:
        version = struct.unpack("<I", read_exact(stream, 4))[0]
        if version != 4:
            raise ValueError(f"unsupported ScanNet sensor-data version: {version}")
        sensor_name_length = struct.unpack("<Q", read_exact(stream, 8))[0]
        sensor_name = read_exact(stream, sensor_name_length).decode("utf-8", errors="strict")
        intrinsic_color = np.frombuffer(
            read_exact(stream, 16 * 4), dtype="<f4"
        ).copy().reshape(4, 4)
        extrinsic_color = np.frombuffer(
            read_exact(stream, 16 * 4), dtype="<f4"
        ).copy().reshape(4, 4)
        intrinsic_depth = np.frombuffer(
            read_exact(stream, 16 * 4), dtype="<f4"
        ).copy().reshape(4, 4)
        extrinsic_depth = np.frombuffer(
            read_exact(stream, 16 * 4), dtype="<f4"
        ).copy().reshape(4, 4)
        color_compression = struct.unpack("<i", read_exact(stream, 4))[0]
        depth_compression = struct.unpack("<i", read_exact(stream, 4))[0]
        color_width, color_height, depth_width, depth_height = struct.unpack(
            "<IIII", read_exact(stream, 16)
        )
        depth_shift = struct.unpack("<f", read_exact(stream, 4))[0]
        total_frames = struct.unpack("<Q", read_exact(stream, 8))[0]
        valid_poses = {}
        for frame_index in range(total_frames):
            pose = np.frombuffer(
                read_exact(stream, 16 * 4), dtype="<f4"
            ).copy().reshape(4, 4)
            read_exact(stream, 16)  # color and depth timestamps
            color_size, depth_size = struct.unpack("<QQ", read_exact(stream, 16))
            pose_tensor = torch.from_numpy(pose)
            if torch.isfinite(pose_tensor).all():
                valid_poses[frame_index] = pose_tensor
            stream.seek(color_size + depth_size, 1)
        imu_frames = struct.unpack("<Q", read_exact(stream, 8))[0]
        imu_frame_size = 5 * 3 * 8 + 8
        for _ in range(imu_frames):
            read_exact(stream, imu_frame_size)
        if stream.tell() != Path(path).stat().st_size:
            raise ValueError("ScanNet sensor stream length does not match frame records")
        valid_indices = sorted(valid_poses)
        selected_positions = uniformly_spaced_indices(len(valid_indices), frame_count)
        selected = [valid_indices[position] for position in selected_positions]
        poses = {frame_index: valid_poses[frame_index] for frame_index in selected}
    header = {
        "version": version,
        "sensor_name": sensor_name,
        "intrinsic_color": intrinsic_color.tolist(),
        "extrinsic_color": extrinsic_color.tolist(),
        "intrinsic_depth": intrinsic_depth.tolist(),
        "extrinsic_depth": extrinsic_depth.tolist(),
        "color_compression": color_compression,
        "depth_compression": depth_compression,
        "color_width": color_width,
        "color_height": color_height,
        "depth_width": depth_width,
        "depth_height": depth_height,
        "depth_shift": depth_shift,
        "total_frames": total_frames,
        "valid_pose_frames": len(valid_poses),
        "imu_frames": imu_frames,
    }
    return selected, poses, torch.from_numpy(intrinsic_color[:3, :3]), header, valid_poses


def validate_reference_cameras(poses, intrinsic, reference_scene_root):
    reference_scene_root = Path(reference_scene_root)
    color_files = sorted(
        (reference_scene_root / "color").glob("*.jpg"), key=lambda path: int(path.stem)
    )
    reference_poses = torch.from_numpy(np.load(reference_scene_root / "extrinsics.npy")).float()
    reference_intrinsic = torch.from_numpy(
        np.loadtxt(reference_scene_root / "intrinsic/intrinsic_color.txt")[:3, :3]
    ).float()
    if len(color_files) != len(reference_poses):
        raise ValueError("reference RGB and pose counts differ")
    finite_matches = 0
    matched_nonfinite = 0
    for ordinal, path in enumerate(color_files):
        frame_index = int(path.stem)
        reference_pose = reference_poses[ordinal]
        if frame_index not in poses:
            if torch.isfinite(reference_pose).all():
                raise ValueError(
                    f"finite reference frame {frame_index} has no finite pose in the sensor stream"
                )
            matched_nonfinite += 1
            continue
        if not torch.isfinite(reference_pose).all():
            raise ValueError(
                f"reference frame {frame_index} is non-finite but the sensor pose is finite"
            )
        torch.testing.assert_close(
            poses[frame_index].float(), reference_pose, rtol=0, atol=1e-6
        )
        finite_matches += 1
    if finite_matches == 0:
        raise ValueError("reference camera validation found no finite common frames")
    torch.testing.assert_close(intrinsic.float(), reference_intrinsic, rtol=0, atol=1e-5)
    return {
        "reference_frames": len(color_files),
        "finite_matches": finite_matches,
        "matched_nonfinite": matched_nonfinite,
    }


def load_instance_catalog(aggregation_path):
    value = json.loads(Path(aggregation_path).read_text())
    groups = value.get("segGroups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("aggregation JSON has no segGroups")
    catalog = {}
    for group in groups:
        object_id = group.get("objectId")
        label = str(group.get("label", "")).strip().lower()
        if not isinstance(object_id, int) or object_id < 0 or not label:
            raise ValueError("invalid ScanNet aggregation group")
        mask_id = object_id + 1
        if mask_id in catalog:
            raise ValueError(f"duplicate one-based instance ID {mask_id}")
        catalog[mask_id] = {
            "object_id": object_id,
            "mask_id": mask_id,
            "label": label,
            "structural": label in STRUCTURAL_LABELS,
        }
    foreground = [mask_id for mask_id, item in sorted(catalog.items()) if not item["structural"]]
    if len(foreground) + 1 > 256:
        raise ValueError("scene has more foreground instances than rasterizer feature channels")
    return catalog, foreground


def zip_png_map(archive):
    mapping = {}
    for name in archive.namelist():
        path = Path(name)
        if path.suffix.lower() != ".png":
            continue
        if path.stem in mapping:
            raise ValueError(f"duplicate mask frame in ZIP: {path.stem}")
        mapping[path.stem] = name
    if not mapping:
        raise ValueError("instance ZIP contains no PNG masks")
    return mapping


def mask_channels(mask, catalog, foreground, image_shape):
    if mask.ndim != 2:
        raise ValueError("instance mask must be single-channel")
    height, width = image_shape
    resized = Image.fromarray(mask.astype(np.int32, copy=False)).resize(
        (width, height), resample=Image.Resampling.NEAREST
    )
    values = np.asarray(resized, dtype=np.int64)
    known = set(catalog)
    observed = set(np.unique(values).tolist())
    unknown = observed - known - {0}
    if unknown:
        raise ValueError(f"mask contains IDs absent from aggregation metadata: {sorted(unknown)}")
    channels = np.zeros((len(foreground) + 1, height, width), dtype=np.float32)
    foreground_channel = {mask_id: index + 1 for index, mask_id in enumerate(foreground)}
    assigned = np.zeros((height, width), dtype=bool)
    for mask_id, channel in foreground_channel.items():
        selected = values == mask_id
        channels[channel, selected] = 1.0
        assigned |= selected
    channels[0, ~assigned] = 1.0
    if not np.array_equal(channels.sum(axis=0), np.ones((height, width), dtype=np.float32)):
        raise ValueError("instance masks do not form a pixel partition")
    return torch.from_numpy(channels), sorted(observed)


def render_feature_image(payload, extrinsic, intrinsic, feature_values, image_shape):
    from src.model.decoder.cuda_splatting import render_cuda

    device = feature_values.device
    count = len(payload["points"])
    sh = torch.zeros((1, count, 3, 1), dtype=torch.float32, device=device)
    image, _, rendered = render_cuda(
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
    if not torch.isfinite(image).all() or not torch.isfinite(rendered).all():
        raise ValueError("rasterizer produced non-finite output")
    return rendered[0]


def frame_contributions(payload, extrinsic, intrinsic, channels, image_shape):
    count = len(payload["points"])
    channel_count = channels.shape[0]
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
    minimum = gradient.min().item()
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
        msg="mask-partition alpha masses differ from the all-pixel render",
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


def quantized_assignment(contributions, foreground, confidence_threshold):
    if contributions.ndim != 2 or contributions.shape[1] != len(foreground) + 1:
        raise ValueError("contribution matrix shape does not match object catalog")
    if not torch.isfinite(contributions).all() or contributions.min().item() < 0:
        raise ValueError("contribution matrix must be finite and nonnegative")
    total = contributions.sum(dim=1)
    foreground_mass = contributions[:, 1:]
    if foreground_mass.shape[1]:
        raw_scores = foreground_mass / total.clamp_min(torch.finfo(torch.float64).tiny)[:, None]
        scores = torch.round(raw_scores * 1_000_000.0) / 1_000_000.0
        confidence, winner = torch.max(scores, dim=1)
        winning_mass = foreground_mass.gather(1, winner[:, None]).squeeze(1)
        assigned = torch.tensor(foreground, dtype=torch.int32)[winner]
        accepted = (total > 0) & (confidence >= confidence_threshold)
        pool_ids = torch.where(accepted, assigned, torch.zeros_like(assigned))
        confidence = torch.where(accepted, confidence, torch.zeros_like(confidence))
        winning_mass = torch.where(accepted, winning_mass, torch.zeros_like(winning_mass))
    else:
        pool_ids = torch.zeros(len(total), dtype=torch.int32)
        confidence = torch.zeros(len(total), dtype=torch.float64)
        winning_mass = torch.zeros(len(total), dtype=torch.float64)
    return pool_ids, confidence.float(), total.float(), winning_mass.float()


def build_association(
    scene_id,
    dense_path,
    scans_root,
    sens_path,
    output_path,
    frame_count=100,
    image_shape=(180, 320),
    confidence_threshold=0.5,
    validate_gradients=False,
    reference_scene_root=None,
    sens_source_url=None,
):
    require_h200()
    dense_path = Path(dense_path)
    scan = Path(scans_root) / scene_id
    aggregation_path = scan / f"{scene_id}.aggregation.json"
    mask_zip_path = scan / f"{scene_id}_2d-instance-filt.zip"
    sens_path = Path(sens_path)
    for path in (dense_path, aggregation_path, mask_zip_path, sens_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = normalize_payload(torch.load(dense_path, map_location="cpu", weights_only=False))
    source_count = validate_aligned(payload, expected_feature_dim=256)
    catalog, foreground = load_instance_catalog(aggregation_path)
    selected_ordinals, poses, intrinsic, sensor_header, valid_poses = read_sens_cameras(
        sens_path, frame_count
    )
    original_width = sensor_header["color_width"]
    original_height = sensor_header["color_height"]
    raw_intrinsic = intrinsic.clone()
    intrinsic[0] /= original_width
    intrinsic[1] /= original_height
    if not all(torch.isfinite(pose).all() for pose in poses.values()) or not torch.isfinite(intrinsic).all():
        raise ValueError("camera matrices contain non-finite values")
    reference_camera_matches = None
    if reference_scene_root is not None:
        reference_camera_matches = validate_reference_cameras(
            valid_poses, raw_intrinsic, reference_scene_root
        )
    selected_frames = [str(index) for index in selected_ordinals]
    accumulated = torch.zeros((source_count, len(foreground) + 1), dtype=torch.float64)
    render_payload = {
        "points": payload["points"].float().cuda(non_blocking=True),
        "covariances": payload["covariances"].float().cuda(non_blocking=True),
        "opacities": payload["opacities"].float().cuda(non_blocking=True),
    }
    observed_ids = set()
    with zipfile.ZipFile(mask_zip_path) as archive:
        mask_files = zip_png_map(archive)
        missing = [frame for frame in selected_frames if frame not in mask_files]
        if missing:
            raise ValueError(f"selected RGB frames have no matching instance masks: {missing[:5]}")
        for frame_number, (ordinal, stem) in enumerate(zip(selected_ordinals, selected_frames)):
            with archive.open(mask_files[stem]) as stream:
                with Image.open(io.BytesIO(stream.read())) as image:
                    if image.size != (original_width, original_height):
                        raise ValueError(
                            f"RGB/mask dimensions differ for {scene_id}/{stem}: "
                            f"{(original_width, original_height)} != {image.size}"
                        )
                    mask = np.asarray(image, dtype=np.int64)
            channels, frame_ids = mask_channels(mask, catalog, foreground, image_shape)
            observed_ids.update(frame_ids)
            contribution = frame_contributions(
                render_payload, poses[ordinal], intrinsic, channels, image_shape
            )
            if validate_gradients and frame_number == 0:
                validate_frame_gradients(
                    render_payload, poses[ordinal], intrinsic, channels, image_shape, contribution
                )
            accumulated.add_(contribution.cpu().double())
            del contribution
            torch.cuda.empty_cache()

    pool_ids, confidence, total_mass, winning_mass = quantized_assignment(
        accumulated, foreground, confidence_threshold
    )
    pool_stats = []
    for pool_id in [BACKGROUND_POOL, *foreground]:
        selected = pool_ids == pool_id
        item = catalog.get(pool_id, {
            "object_id": None, "mask_id": 0, "label": "structural/background", "structural": True
        })
        pool_stats.append({
            **item,
            "pool_id": pool_id,
            "gaussian_count": int(selected.sum().item()),
            "visible_mass": float(total_mass[selected].double().sum().item()),
            "observed_in_selected_frames": bool(pool_id in observed_ids) if pool_id else True,
        })
    frame_record = {
        "sorted_observation_count": sensor_header["total_frames"],
        "selected_ordinals": selected_ordinals,
        "selected_frame_stems": selected_frames,
        "source": "scannet_v1_sens",
    }
    metadata = {
        "selection_version": SELECTION_VERSION,
        "scene_id": scene_id,
        "source_count": source_count,
        "frame_count": frame_count,
        "image_shape": list(image_shape),
        "near": 0.5,
        "far": 15.0,
        "confidence_threshold": float(confidence_threshold),
        "association_quantization": 1e-6,
        "mask_id_convention": "one_based_object_id_plus_one",
        "structural_labels": sorted(STRUCTURAL_LABELS),
        "foreground_pool_ids": foreground,
        "frame_record": frame_record,
        "frame_digest": canonical_digest(frame_record),
        "input_digests": {
            "dense_encoded": payload_digest(dense_path),
            "aggregation": sha256_file(aggregation_path),
            "instance_masks_zip": sha256_file(mask_zip_path),
            "sensor_data": sha256_file(sens_path),
        },
        "pool_stats": pool_stats,
        "gradient_validation": bool(validate_gradients),
        "reference_camera_matches": reference_camera_matches,
        "sensor_header": sensor_header,
        "sensor_source_url": sens_source_url,
    }
    result = {
        "pool_ids": pool_ids,
        "confidence": confidence,
        "total_visible_mass": total_mass,
        "winning_instance_mass": winning_mass,
        "association": metadata,
    }
    metadata["association_sha256"] = association_digest(result, metadata)
    atomic_torch_save(output_path, result)
    atomic_json(Path(output_path).with_suffix(".json"), metadata)
    return result


def validate_association(payload, scene_id, dense_path=None):
    if not isinstance(payload, dict):
        raise ValueError("association payload must be a dictionary")
    required = {"pool_ids", "confidence", "total_visible_mass", "winning_instance_mass", "association"}
    if set(payload) != required:
        raise ValueError(f"association payload fields mismatch: {set(payload)}")
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
        payload["winning_instance_mass"] > payload["total_visible_mass"] + 1e-6
    ).any():
        raise ValueError("winning-instance mass is outside valid bounds")
    foreground = set(metadata.get("foreground_pool_ids", []))
    if any(int(value) not in foreground | {BACKGROUND_POOL} for value in torch.unique(payload["pool_ids"]).tolist()):
        raise ValueError("association contains an unknown pool ID")
    if metadata.get("frame_digest") != canonical_digest(metadata.get("frame_record")):
        raise ValueError("association frame digest mismatch")
    if metadata.get("association_sha256") != association_digest(payload, metadata):
        raise ValueError("association payload digest mismatch")
    if dense_path is not None:
        if metadata["input_digests"]["dense_encoded"] != sha256_file(dense_path):
            raise ValueError("association dense-source digest mismatch")
    return metadata


def compare_duplicate_associations(first, second):
    if not torch.equal(first["pool_ids"], second["pool_ids"]):
        raise ValueError("duplicate association mismatch: pool_ids")
    torch.testing.assert_close(
        first["confidence"], second["confidence"], rtol=0, atol=1.1e-6
    )
    for key in ("total_visible_mass", "winning_instance_mass"):
        torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-6)
    return {
        "pool_assignments_exact": True,
        "confidence_atol": 1.1e-6,
        "raw_mass_tolerance": {"rtol": 1e-5, "atol": 1e-6},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--dense-root", required=True)
    parser.add_argument("--scans-root", required=True)
    parser.add_argument("--sens-path", required=True)
    parser.add_argument("--sens-source-url")
    parser.add_argument("--reference-scene-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-count", type=int, default=100)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--validate-gradients", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("confidence threshold must be in [0, 1]")
    output = Path(args.output)
    dense_path = Path(args.dense_root) / args.scene / "feat_fs" / f"{args.scene}.pt"
    if output.exists() and not args.overwrite:
        existing = torch.load(output, map_location="cpu", weights_only=False)
        validate_association(existing, args.scene, dense_path)
        print(json.dumps(existing["association"], sort_keys=True))
        return
    result = build_association(
        args.scene,
        dense_path,
        args.scans_root,
        args.sens_path,
        output,
        frame_count=args.frame_count,
        image_shape=(args.height, args.width),
        confidence_threshold=args.confidence_threshold,
        validate_gradients=args.validate_gradients,
        reference_scene_root=args.reference_scene_root,
        sens_source_url=args.sens_source_url,
    )
    print(json.dumps(result["association"], sort_keys=True))


if __name__ == "__main__":
    main()
