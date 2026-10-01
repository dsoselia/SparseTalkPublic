#!/usr/bin/env python3
"""Prompt-free Florence-2 and SAM2 detection for detected-object sampling."""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from PIL import Image

PACKAGE = Path(__file__).resolve().parent
AUTOENCODER = PACKAGE.parent
REPOSITORY = AUTOENCODER.parent
for path in (str(AUTOENCODER), str(REPOSITORY)):
    if path not in sys.path:
        sys.path.insert(0, path)

from detected_object_sampling import (
    FLORENCE_REVISION,
    SAM2_CODE_COMMIT,
    SAM2_WEIGHTS_REVISION,
    SELECTION_VERSION,
    STRUCTURAL_LABELS,
)
from detected_object_sampling.common import (
    atomic_json,
    canonical_digest,
    require_h200,
    sha256_file,
)


LABEL_SYNONYMS = {
    "armchairs": "armchair",
    "backpacks": "backpack",
    "beds": "bed",
    "benches": "bench",
    "books": "book",
    "bookshelves": "bookshelf",
    "bookcases": "bookcase",
    "bottles": "bottle",
    "cabinets": "cabinet",
    "ceilings": "ceiling",
    "chairs": "chair",
    "clocks": "clock",
    "couches": "sofa",
    "cupboards": "cabinet",
    "curtains": "curtain",
    "desks": "desk",
    "doors": "door",
    "drawers": "drawer",
    "floors": "floor",
    "lamps": "lamp",
    "monitors": "monitor",
    "nightstands": "nightstand",
    "office chairs": "office chair",
    "ottomans": "ottoman",
    "pictures": "picture",
    "pillows": "pillow",
    "plants": "plant",
    "refrigerators": "refrigerator",
    "shelves": "shelf",
    "sinks": "sink",
    "sofas": "sofa",
    "stools": "stool",
    "tables": "table",
    "televisions": "television",
    "toilets": "toilet",
    "towels": "towel",
    "trash cans": "trash can",
    "walls": "wall",
    "windows": "window",
}


def normalize_label(value):
    label = re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()
    label = re.sub(r"\s+", " ", label)
    return LABEL_SYNONYMS.get(label, label)


def uniformly_spaced_indices(count, requested):
    if count < requested:
        raise ValueError(f"need at least {requested} finite observations, found {count}")
    values = np.rint(np.linspace(0, count - 1, requested)).astype(np.int64)
    if len(np.unique(values)) != requested:
        raise ValueError("uniform frame selection produced duplicate indices")
    return values.tolist()


def selected_observations(scene_root, frame_count):
    scene_root = Path(scene_root)
    color_files = sorted((scene_root / "color").glob("*.jpg"), key=lambda path: int(path.stem))
    poses = np.load(scene_root / "extrinsics.npy")
    if poses.shape != (len(color_files), 4, 4):
        raise ValueError("RGB and extrinsics counts differ")
    finite_ordinals = [index for index, pose in enumerate(poses) if np.isfinite(pose).all()]
    positions = uniformly_spaced_indices(
        len(finite_ordinals), min(frame_count, len(finite_ordinals))
    )
    selected = []
    for position in positions:
        ordinal = finite_ordinals[position]
        image_path = color_files[ordinal]
        selected.append({
            "ordinal": ordinal,
            "frame_stem": image_path.stem,
            "image_path": str(image_path.resolve()),
            "image_sha256": sha256_file(image_path),
        })
    return selected


def directory_digest(path):
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    records = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or ".git" in item.parts or ".locks" in item.parts:
            continue
        records.append({
            "path": item.relative_to(root).as_posix(),
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        })
    if not records:
        raise ValueError(f"model directory contains no files: {root}")
    return canonical_digest(records), records


def box_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def mask_iou(first, second):
    intersection = np.logical_and(first, second).sum(dtype=np.int64)
    union = np.logical_or(first, second).sum(dtype=np.int64)
    return float(intersection / union) if union else 0.0


def proposal_priority(proposal):
    return (
        -float(proposal["sam_score"]),
        int(proposal["mask_area_render"]),
        proposal["canonical_label"],
        int(proposal["raw_index"]),
    )


def deterministic_mask_nms(proposals, maximum=128):
    kept = []
    for candidate in sorted(proposals, key=proposal_priority):
        reject = False
        for existing in kept:
            threshold = (
                0.80
                if candidate["canonical_label"] == existing["canonical_label"]
                else 0.95
            )
            if mask_iou(candidate["mask"], existing["mask"]) >= threshold:
                reject = True
                break
        if not reject:
            kept.append(candidate)
            if len(kept) == maximum:
                break
    return kept


def arbitrate_overlaps(proposals, image_shape, minimum_pixels=64):
    height, width = image_shape
    proposal_map = np.zeros((height, width), dtype=np.uint16)
    records = []
    for proposal in sorted(proposals, key=proposal_priority):
        visible = np.logical_and(proposal["mask"], proposal_map == 0)
        visible_area = int(visible.sum(dtype=np.int64))
        if visible_area == 0:
            continue
        map_id = len(records) + 1
        if map_id > np.iinfo(np.uint16).max:
            raise ValueError("proposal map exceeds uint16 capacity")
        proposal_map[visible] = map_id
        record = {key: value for key, value in proposal.items() if key != "mask"}
        record.update({"map_id": map_id, "visible_area_render": visible_area})
        records.append(record)
    if len(records) > 128:
        raise ValueError("proposal cap was not enforced")
    observed = set(np.unique(proposal_map).tolist())
    if observed != set(range(len(records) + 1)):
        raise ValueError("proposal map IDs are not dense and aligned")
    return proposal_map, records


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def frame_input_digest(frames):
    return canonical_digest([
        {key: value for key, value in frame.items() if key != "output_digests"}
        for frame in frames
    ])


class DetectorModels:
    def __init__(self, florence_model, sam2_repo, sam2_checkpoint):
        require_h200()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        from transformers import AutoModelForCausalLM, AutoProcessor

        # The pinned remote module contains a guarded FlashAttention import. Transformers
        # 4.40 scans it as unconditional before evaluating the guard. This import-only stub
        # lets that scan pass while package metadata remains absent, so Florence selects SDPA.
        if importlib.util.find_spec("flash_attn") is None:
            compatibility_stub = types.ModuleType("flash_attn")
            compatibility_stub.__path__ = []
            compatibility_stub.__spec__ = importlib.machinery.ModuleSpec(
                "flash_attn", loader=None, is_package=True
            )
            sys.modules["flash_attn"] = compatibility_stub

        self.processor = AutoProcessor.from_pretrained(
            florence_model, trust_remote_code=True, local_files_only=True
        )
        self.florence = AutoModelForCausalLM.from_pretrained(
            florence_model,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager",
        ).eval().cuda()
        sam2_repo = Path(sam2_repo).resolve()
        if str(sam2_repo) not in sys.path:
            sys.path.insert(0, str(sam2_repo))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        sam = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(Path(sam2_checkpoint).resolve()),
            device="cuda",
            mode="eval",
        )
        self.sam_predictor = SAM2ImagePredictor(sam)

    @torch.inference_mode()
    def florence_boxes(self, image):
        task = "<OD>"
        inputs = self.processor(text=task, images=image, return_tensors="pt")
        moved = {}
        for key, value in inputs.items():
            value = value.cuda(non_blocking=True)
            moved[key] = value.to(torch.bfloat16) if key == "pixel_values" else value
        generated = self.florence.generate(
            **moved,
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
        text = self.processor.batch_decode(generated, skip_special_tokens=False)[0]
        parsed = self.processor.post_process_generation(
            text, task=task, image_size=image.size
        )
        result = parsed.get(task)
        if not isinstance(result, dict):
            raise ValueError("Florence-2 returned an invalid OD schema")
        boxes = result.get("bboxes", [])
        labels = result.get("labels", [])
        if len(boxes) != len(labels):
            raise ValueError("Florence-2 boxes and labels are unaligned")
        return boxes, labels, text

    @torch.inference_mode()
    def sam_mask(self, image_array, box):
        masks, scores, _ = self.sam_predictor.predict(
            box=np.asarray(box, dtype=np.float32), multimask_output=False
        )
        if masks.shape[0] != 1 or len(scores) != 1:
            raise ValueError("SAM2 must return exactly one mask per box")
        mask = np.asarray(masks[0], dtype=bool).squeeze()
        if mask.ndim != 2:
            raise ValueError(f"SAM2 returned an invalid mask shape: {mask.shape}")
        return mask, float(scores[0])


def filter_and_segment(models, image, boxes, labels, image_shape):
    width, height = image.size
    total_area = width * height
    image_array = np.asarray(image.convert("RGB"))
    models.sam_predictor.set_image(image_array)
    candidates = []
    rejected = {
        "invalid_box": 0,
        "box_area": 0,
        "sam_score": 0,
        "small_mask": 0,
        "structural": 0,
    }
    for raw_index, (raw_box, raw_label) in enumerate(zip(boxes, labels)):
        try:
            box = [float(value) for value in raw_box]
        except (TypeError, ValueError):
            rejected["invalid_box"] += 1
            continue
        if (
            len(box) != 4
            or not np.isfinite(box).all()
            or box[0] < 0
            or box[1] < 0
            or box[2] > width
            or box[3] > height
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            rejected["invalid_box"] += 1
            continue
        area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / total_area
        if area_ratio < 0.0005 or area_ratio > 0.95:
            rejected["box_area"] += 1
            continue
        canonical = normalize_label(raw_label)
        mask, score = models.sam_mask(image_array, box)
        if not np.isfinite(score) or score < 0.80:
            rejected["sam_score"] += 1
            continue
        resized = np.asarray(
            Image.fromarray(mask).resize(
                (image_shape[1], image_shape[0]), resample=Image.Resampling.NEAREST
            ),
            dtype=bool,
        )
        area_render = int(resized.sum(dtype=np.int64))
        if area_render < 64:
            rejected["small_mask"] += 1
            continue
        if canonical in STRUCTURAL_LABELS:
            rejected["structural"] += 1
            continue
        candidates.append({
            "raw_index": raw_index,
            "bbox": box,
            "raw_label": str(raw_label),
            "canonical_label": canonical,
            "sam_score": score,
            "box_area_ratio": area_ratio,
            "mask_area_original": int(mask.sum(dtype=np.int64)),
            "mask_area_render": area_render,
            "mask": resized,
        })
    kept = deterministic_mask_nms(candidates)
    proposal_map, records = arbitrate_overlaps(kept, image_shape)
    return proposal_map, records, rejected, len(candidates), len(kept)


def validate_detection_scene(output_root, scene_id, expected_frame_count=100):
    output_root = Path(output_root)
    manifest_path = output_root / "detection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("selection_version") != SELECTION_VERSION
        or manifest.get("scene_id") != scene_id
        or manifest.get("requested_frame_count") != expected_frame_count
    ):
        raise ValueError("detection manifest identity mismatch")
    if (
        manifest.get("frame_count") != len(manifest.get("frames", []))
        or not 1 <= manifest["frame_count"] <= expected_frame_count
    ):
        raise ValueError("detection manifest actual frame count is invalid")
    if manifest.get("frame_digest") != frame_input_digest(manifest.get("frames")):
        raise ValueError("detection frame digest mismatch")
    frame_digests = []
    foreground = 0
    for frame in manifest["frames"]:
        stem = frame["frame_stem"]
        map_path = output_root / "frames" / f"{stem}.npz"
        metadata_path = output_root / "frames" / f"{stem}.json"
        with np.load(map_path, allow_pickle=False) as data:
            proposal_map = data["proposal_map"]
        metadata = json.loads(metadata_path.read_text())
        if proposal_map.dtype != np.uint16 or tuple(proposal_map.shape) != tuple(manifest["image_shape"]):
            raise ValueError(f"invalid proposal map for {scene_id}/{stem}")
        proposals = metadata.get("proposals")
        if not isinstance(proposals, list) or len(proposals) > 128:
            raise ValueError(f"invalid proposal metadata for {scene_id}/{stem}")
        if set(np.unique(proposal_map).tolist()) != set(range(len(proposals) + 1)):
            raise ValueError(f"proposal IDs are unaligned for {scene_id}/{stem}")
        for index, proposal in enumerate(proposals, 1):
            if proposal.get("map_id") != index or proposal.get("canonical_label") in STRUCTURAL_LABELS:
                raise ValueError(f"invalid proposal record for {scene_id}/{stem}")
            foreground += 1
        record = {
            "frame_stem": stem,
            "map_sha256": sha256_file(map_path),
            "metadata_sha256": sha256_file(metadata_path),
        }
        if record != frame.get("output_digests"):
            raise ValueError(f"detection frame output digest mismatch for {scene_id}/{stem}")
        frame_digests.append(record)
    expected = canonical_digest({
        "scene_id": scene_id,
        "frame_digest": manifest["frame_digest"],
        "models": manifest["models"],
        "outputs": frame_digests,
    })
    if manifest.get("detection_sha256") != expected:
        raise ValueError("detection scene digest mismatch")
    if foreground == 0:
        raise ValueError(f"scene has no detected foreground proposals: {scene_id}")
    return manifest


def compare_duplicate_detections(first_root, second_root, scene_id):
    first = validate_detection_scene(first_root, scene_id)
    second = validate_detection_scene(second_root, scene_id)
    stable_keys = ("frames", "frame_digest", "models", "image_shape", "generation")
    for key in stable_keys:
        if key == "frames":
            continue
        if first.get(key) != second.get(key):
            raise ValueError(f"duplicate detector metadata mismatch: {key}")
    score_count = 0
    for frame_a, frame_b in zip(first["frames"], second["frames"]):
        if {k: v for k, v in frame_a.items() if k != "output_digests"} != {
            k: v for k, v in frame_b.items() if k != "output_digests"
        }:
            raise ValueError("duplicate detector frame-input mismatch")
        stem = frame_a["frame_stem"]
        with np.load(Path(first_root) / "frames" / f"{stem}.npz", allow_pickle=False) as value:
            map_a = value["proposal_map"]
        with np.load(Path(second_root) / "frames" / f"{stem}.npz", allow_pickle=False) as value:
            map_b = value["proposal_map"]
        if not np.array_equal(map_a, map_b):
            raise ValueError(f"duplicate detector map mismatch: {stem}")
        meta_a = json.loads((Path(first_root) / "frames" / f"{stem}.json").read_text())
        meta_b = json.loads((Path(second_root) / "frames" / f"{stem}.json").read_text())
        if len(meta_a["proposals"]) != len(meta_b["proposals"]):
            raise ValueError(f"duplicate detector proposal-count mismatch: {stem}")
        for proposal_a, proposal_b in zip(meta_a["proposals"], meta_b["proposals"]):
            score_a = proposal_a.pop("sam_score")
            score_b = proposal_b.pop("sam_score")
            if proposal_a != proposal_b:
                raise ValueError(f"duplicate detector proposal mismatch: {stem}")
            if not np.isclose(score_a, score_b, rtol=1e-5, atol=1e-6):
                raise ValueError(f"duplicate SAM score mismatch: {stem}")
            score_count += 1
    return {
        "discrete_maps_exact": True,
        "proposal_metadata_exact_except_scores": True,
        "sam_scores_compared": score_count,
        "score_tolerance": {"rtol": 1e-5, "atol": 1e-6},
    }


def run_detection(
    scene_id,
    scene_root,
    output_root,
    florence_model,
    sam2_repo,
    sam2_checkpoint,
    frame_count=100,
    image_shape=(180, 320),
):
    require_h200()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    observations = selected_observations(scene_root, frame_count)
    florence_digest, florence_files = directory_digest(florence_model)
    sam_repo_commit = __import__("subprocess").check_output(
        ["git", "-C", str(sam2_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if sam_repo_commit != SAM2_CODE_COMMIT:
        raise ValueError(f"SAM2 code commit mismatch: {sam_repo_commit}")
    sam_checkpoint_digest = sha256_file(sam2_checkpoint)
    models = DetectorModels(florence_model, sam2_repo, sam2_checkpoint)
    frame_records = []
    total_proposals = 0
    for observation in observations:
        image_path = Path(observation["image_path"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        boxes, labels, generated_text = models.florence_boxes(image)
        proposal_map, proposals, rejected, filtered_count, nms_count = filter_and_segment(
            models, image, boxes, labels, image_shape
        )
        stem = observation["frame_stem"]
        map_path = output_root / "frames" / f"{stem}.npz"
        metadata_path = output_root / "frames" / f"{stem}.json"
        atomic_npz(map_path, proposal_map=proposal_map)
        atomic_json(metadata_path, {
            "scene_id": scene_id,
            "frame_stem": stem,
            "ordinal": observation["ordinal"],
            "source_image_size": list(image.size[::-1]),
            "render_shape": list(image_shape),
            "florence_raw_count": len(boxes),
            "post_filter_count": filtered_count,
            "post_nms_count": nms_count,
            "retained_count": len(proposals),
            "rejected": rejected,
            "generated_text_sha256": hashlib.sha256(generated_text.encode("utf-8")).hexdigest(),
            "proposals": proposals,
        })
        record = {
            **observation,
            "output_digests": {
                "frame_stem": stem,
                "map_sha256": sha256_file(map_path),
                "metadata_sha256": sha256_file(metadata_path),
            },
        }
        frame_records.append(record)
        total_proposals += len(proposals)
    model_record = {
        "florence": {
            "model_id": "microsoft/Florence-2-large",
            "revision": FLORENCE_REVISION,
            "tree_sha256": florence_digest,
            "file_count": len(florence_files),
        },
        "sam2": {
            "weights_id": "facebook/sam2.1-hiera-large",
            "weights_revision": SAM2_WEIGHTS_REVISION,
            "checkpoint_sha256": sam_checkpoint_digest,
            "code_commit": sam_repo_commit,
        },
    }
    manifest = {
        "selection_version": SELECTION_VERSION,
        "scene_id": scene_id,
        "requested_frame_count": frame_count,
        "frame_count": len(observations),
        "image_shape": list(image_shape),
        "frames": frame_records,
        "frame_digest": frame_input_digest(frame_records),
        "models": model_record,
        "generation": {
            "prompt": "<OD>",
            "num_beams": 3,
            "do_sample": False,
            "dtype": "bfloat16",
            "attention_backend": "eager",
            "flash_attn_import_guard": "transformers_4_40_dynamic_import_scan_v1",
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "sam_score_threshold": 0.80,
            "same_label_nms_iou": 0.80,
            "cross_label_nms_iou": 0.95,
            "maximum_proposals_per_frame": 128,
            "minimum_render_mask_pixels": 64,
            "minimum_box_area_ratio": 0.0005,
            "maximum_box_area_ratio": 0.95,
        },
        "total_retained_proposals": total_proposals,
    }
    manifest["detection_sha256"] = canonical_digest({
        "scene_id": scene_id,
        "frame_digest": manifest["frame_digest"],
        "models": model_record,
        "outputs": [frame["output_digests"] for frame in frame_records],
    })
    atomic_json(output_root / "detection_manifest.json", manifest)
    validate_detection_scene(output_root, scene_id, frame_count)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--scene-data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--florence-model", required=True)
    parser.add_argument("--sam2-repo", required=True)
    parser.add_argument("--sam2-checkpoint", required=True)
    parser.add_argument("--frame-count", type=int, default=100)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    require_h200()
    output = Path(args.output_root) / args.scene
    if (output / "detection_manifest.json").is_file() and not args.overwrite:
        result = validate_detection_scene(output, args.scene, args.frame_count)
    else:
        if output.exists():
            import shutil
            shutil.rmtree(output)
        result = run_detection(
            args.scene,
            Path(args.scene_data_root) / args.scene,
            output,
            args.florence_model,
            args.sam2_repo,
            args.sam2_checkpoint,
            args.frame_count,
            (args.height, args.width),
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
