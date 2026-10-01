"""Create a run-local ScanNet-style tree and frame language targets."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sparsetalk.paths import LLAVA_ENGINE, SPLATTALK_ENGINE, atomic_json, file_sha256, require_cuda
from sparsetalk.selection import _load_decoder


def validate_scene(scene_root: Path) -> list[Path]:
    color = scene_root / "color"
    if not color.is_dir():
        raise ValueError(f"missing color directory: {color}")
    frames = [path for path in color.iterdir() if path.suffix.lower() == ".jpg"]
    if not frames:
        raise ValueError(f"no .jpg RGB frames in {color}")
    try:
        frames.sort(key=lambda path: int(path.stem))
    except ValueError as error:
        raise ValueError("frame filenames must have numeric stems") from error
    if len({int(path.stem) for path in frames}) != len(frames):
        raise ValueError("duplicate numeric frame IDs")
    depth = scene_root / "depth"
    if not depth.is_dir() or any(not (depth / f"{path.stem}.png").is_file() for path in frames):
        raise ValueError(f"depth frames do not align with RGB frames: {scene_root}")
    intrinsic_path = scene_root / "intrinsic" / "intrinsic_color.txt"
    if not intrinsic_path.is_file():
        raise ValueError(f"missing color intrinsics: {intrinsic_path}")
    intrinsic = np.loadtxt(intrinsic_path)
    if intrinsic.shape not in {(3, 3), (4, 4)} or not np.isfinite(intrinsic).all():
        raise ValueError("color intrinsic must be finite 3x3 or 4x4")
    extrinsics_path = scene_root / "extrinsics.npy"
    if not extrinsics_path.is_file():
        raise ValueError(f"missing extrinsics: {extrinsics_path}")
    extrinsics = np.load(extrinsics_path, mmap_mode="r")
    if extrinsics.shape != (len(frames), 4, 4) or not np.isfinite(extrinsics).all():
        raise ValueError("extrinsics must be finite [frame_count,4,4] in numeric frame order")
    return frames


def _link_scene(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("color", "depth", "intrinsic", "extrinsics.npy"):
        destination = target / name
        if destination.is_symlink():
            if destination.resolve() != (source / name).resolve():
                raise ValueError(f"existing scene link points elsewhere: {destination}")
        elif destination.exists():
            raise ValueError(f"run-local input conflicts with scene link: {destination}")
        else:
            destination.symlink_to((source / name).resolve())


def _atomic_tensor(path: Path, value: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def prepare_scenes(
    source_root: Path, scene_ids: list[str], run_dir: Path, split: str,
    model_base: str, autoencoder_checkpoint: Path | None,
    save_ov: bool, batch_size: int,
) -> dict:
    require_cuda()
    if split not in {"scanqa", "train"}:
        raise ValueError("split must be scanqa or train")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not autoencoder_checkpoint and not save_ov:
        raise ValueError("provide an autoencoder checkpoint or --save-ov")
    sources = {scene: source_root / scene for scene in scene_ids}
    frames_by_scene = {scene: validate_scene(path) for scene, path in sources.items()}
    target_root = run_dir / "inputs"
    index_name = "test_idx.txt" if split == "scanqa" else "train_idx.txt"
    index_path = target_root / index_name
    index_contents = "\n".join(scene_ids) + "\n"
    if index_path.exists() and index_path.read_text() != index_contents:
        raise ValueError(f"existing scene index differs: {index_path}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if not index_path.exists():
        index_path.write_text(index_contents)
    for scene, source in sources.items():
        _link_scene(source, target_root / split / scene)

    if str(LLAVA_ENGINE) not in sys.path:
        sys.path.insert(0, str(LLAVA_ENGINE))
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    _, model, processor, _ = load_pretrained_model(
        model_base, None, get_model_name_from_path(model_base),
        device_map="cuda", attn_implementation=None,
    )
    model.eval().requires_grad_(False)
    if {parameter.device.type for parameter in model.parameters()} != {"cuda"}:
        raise RuntimeError("LLaVA model was not fully loaded on CUDA")
    encoder = _load_decoder(autoencoder_checkpoint) if autoencoder_checkpoint else None
    manifest = []
    for scene in scene_ids:
        frames = frames_by_scene[scene]
        target = target_root / split / scene
        metadata_path = target / "frame_features.json"
        configuration = {
            "scene_id": scene,
            "source_scene": str(sources[scene].resolve()),
            "frames": [path.name for path in frames],
            "model_base": model_base,
            "autoencoder_sha256": file_sha256(autoencoder_checkpoint) if autoencoder_checkpoint else None,
            "save_ov": save_ov,
        }
        ov_path = target / "language_feats_ov" / f"{scene}.pt"
        encoded_path = target / "language_feats_256" / f"{scene}.pt"
        if metadata_path.is_file():
            existing = json.loads(metadata_path.read_text())
            if existing.get("configuration") == configuration and (
                not save_ov or (ov_path.is_file() and existing.get("ov_sha256") == file_sha256(ov_path))
            ) and (
                encoder is None or (encoded_path.is_file() and existing.get("encoded_sha256") == file_sha256(encoded_path))
            ):
                manifest.append(existing)
                continue
            old = existing.get("configuration", {})
            upgrade = (
                encoder is not None and save_ov and old.get("autoencoder_sha256") is None
                and all(old.get(key) == configuration[key] for key in (
                    "scene_id", "source_scene", "frames", "model_base", "save_ov"
                ))
                and ov_path.is_file() and existing.get("ov_sha256") == file_sha256(ov_path)
                and not encoded_path.exists()
            )
            if upgrade:
                ov_features = torch.load(ov_path, map_location="cpu", weights_only=True, mmap=True)
                if tuple(ov_features.shape) != (len(frames), 3584, 27, 27):
                    raise ValueError(f"stored LLaVA features have the wrong shape: {ov_path}")
                compressed = torch.empty((len(frames), 256, 27, 27), dtype=torch.float16)
                for start in range(0, len(frames), batch_size):
                    block = ov_features[start:start + batch_size].permute(0, 2, 3, 1).reshape(-1, 3584)
                    with torch.inference_mode():
                        encoded_block = encoder.encode(block.float().cuda())
                    compressed[start:start + min(batch_size, len(frames) - start)] = (
                        encoded_block.reshape(-1, 27, 27, 256).permute(0, 3, 1, 2).half().cpu()
                    )
                _atomic_tensor(encoded_path, compressed)
                atomic_json(encoded_path.with_name("metadata.json"), {
                    "validated": True, "scene_id": scene,
                    "frames": [path.name for path in frames],
                    "shape": [len(frames), 256, 27, 27],
                    "payload_sha256": file_sha256(encoded_path),
                })
                item = {
                    "configuration": configuration,
                    "ov_sha256": existing["ov_sha256"],
                    "encoded_sha256": file_sha256(encoded_path),
                }
                atomic_json(metadata_path, item)
                manifest.append(item)
                continue
            raise ValueError(f"existing frame-feature metadata differs: {metadata_path}")
        ov = torch.empty((len(frames), 3584, 27, 27), dtype=torch.float16) if save_ov else None
        encoded = torch.empty((len(frames), 256, 27, 27), dtype=torch.float16) if encoder else None
        for start in range(0, len(frames), batch_size):
            images = [Image.open(path).convert("RGB") for path in frames[start:start + batch_size]]
            pixels = torch.stack([
                processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
                for image in images
            ]).cuda().half()
            with torch.inference_mode():
                features = model.encode_images(pixels)
                if tuple(features.shape[1:]) != (729, 3584):
                    raise ValueError(f"unexpected LLaVA image feature shape: {tuple(features.shape)}")
                if not torch.isfinite(features).all().item():
                    raise ValueError("non-finite LLaVA image features")
                if ov is not None:
                    ov[start:start + len(images)] = (
                        features.reshape(len(images), 27, 27, 3584).permute(0, 3, 1, 2).half().cpu()
                    )
                if encoded is not None:
                    compressed = encoder.encode(features.reshape(-1, 3584).float())
                    if not torch.isfinite(compressed).all().item():
                        raise ValueError("non-finite encoded frame features")
                    encoded[start:start + len(images)] = (
                        compressed.reshape(len(images), 27, 27, 256).permute(0, 3, 1, 2).half().cpu()
                    )
        if ov is not None:
            _atomic_tensor(ov_path, ov)
        if encoded is not None:
            _atomic_tensor(encoded_path, encoded)
            atomic_json(encoded_path.with_name("metadata.json"), {
                "validated": True, "scene_id": scene,
                "frames": [path.name for path in frames],
                "shape": [len(frames), 256, 27, 27],
                "payload_sha256": file_sha256(encoded_path),
            })
        item = {
            "configuration": configuration,
            "ov_sha256": file_sha256(ov_path) if save_ov else None,
            "encoded_sha256": file_sha256(encoded_path) if encoder else None,
        }
        atomic_json(metadata_path, item)
        manifest.append(item)
    summary = {"scene_count": len(scene_ids), "input_root": str(target_root), "scenes": manifest}
    atomic_json(run_dir / "prepare_manifest.json", summary)
    return summary
