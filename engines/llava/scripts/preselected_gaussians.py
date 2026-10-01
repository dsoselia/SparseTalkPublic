import json
import hashlib
import math
import string
from pathlib import Path

import torch


ALIGNED_KEYS = ("features", "points", "covariances", "opacities")
EXPECTED_SELECTION_VARIABLES = [
    "source_count",
    "scene_id",
    "requested_ratio",
    "global_seed",
]
ANCHOR_METHODS = {
    "uniform_random",
    "voxel_random",
    "fps",
    "semantic_kcenter",
    "joint_kcenter",
    "adaptive_anchors",
}
ALLOWED_SELECTION_METHODS = {"random", *ANCHOR_METHODS}
ANCHOR_VERSION = "coverage_aware_anchors_v1"
SUBBLOCK_VERSION = "sub_block_preselected_v1"
RANKED_VERSION = "sub_block_ranked_v1"
PUBLIC_VERSION = "sparsetalk_public_v1"
TRAINED_FIELD_VERSION = "trained_field_preselected_v1"
RANKED_METHODS = {
    "voxel_random", "fps", "semantic_kcenter", "joint_kcenter",
    "entropy_topk", "opacity_topk",
}
RANKED_INPUTS = {
    "voxel_random": ["points"], "fps": ["points"],
    "semantic_kcenter": ["points", "features"],
    "joint_kcenter": ["points", "features"],
    "entropy_topk": ["decoded_features"], "opacity_topk": ["opacities"],
}
BLIND_METHOD = "blind"
SUBBLOCK_LAYOUTS = {"subblock_1xk", "legacy_27x27"}
EXPECTED_ANCHOR_INPUTS = ["points", "features"]
EXPECTED_SUBBLOCK_INPUTS = ["source_count", "scene_id", "global_seed"]
ALLOWED_SELECTION_METHODS.add(BLIND_METHOD)
ALLOWED_SELECTION_METHODS.update(RANKED_METHODS)
ALLOWED_SELECTION_METHODS.add("entropy")
PUBLIC_METHODS = {
    "uniform_random", "voxel_random", "fps", "semantic_kcenter",
    "joint_kcenter", "adaptive_anchors", "entropy_topk", "opacity_topk",
    "object_balanced_random", "object_balanced_fps",
}
ALLOWED_SELECTION_METHODS.update(PUBLIC_METHODS)


def valid_digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in string.hexdigits for character in value))


def tensor_digest(indices, metadata):
    digest = hashlib.sha256()
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    digest.update(indices.cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_selection_metadata(metadata, scene_id, retained):
    method = metadata.get("selection_method")
    if method not in ALLOWED_SELECTION_METHODS:
        raise ValueError(f"unsupported preselected selection_method: {method}")
    if metadata.get("scene_id") != scene_id:
        raise ValueError("preselected payload scene_id mismatch")
    source_count = metadata.get("source_count")
    effective_ratio = metadata.get("effective_ratio")
    if not isinstance(source_count, int) or source_count < retained:
        raise ValueError("invalid source_count metadata")
    if not isinstance(effective_ratio, (int, float)) or not math.isclose(
        effective_ratio, retained / source_count, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("effective_ratio metadata mismatch")
    version = metadata.get("selection_version")
    if version == PUBLIC_VERSION:
        if method not in PUBLIC_METHODS or retained <= 0:
            raise ValueError("invalid SparseTalkPublic method or retained count")
        if metadata.get("requested_budget") != retained or metadata.get("consumed_count") != retained:
            raise ValueError("SparseTalkPublic count mismatch")
        if metadata.get("feature_layout") != "subblock_1xk":
            raise ValueError("SparseTalkPublic requires subblock_1xk")
        for key in ("ranking_sha256", "ranking_prefix_sha256", "dense_source_sha256"):
            if not valid_digest(metadata.get(key)):
                raise ValueError(f"invalid SparseTalkPublic digest: {key}")
        source_digests = metadata.get("source_digests")
        if not isinstance(source_digests, dict) or source_digests.get("dense_sha256") != metadata["dense_source_sha256"]:
            raise ValueError("SparseTalkPublic source digest mismatch")
        if not isinstance(metadata.get("parameters"), dict):
            raise ValueError("SparseTalkPublic parameters must be a dictionary")
        if method == "entropy_topk" and not valid_digest(source_digests.get("decoder_sha256")):
            raise ValueError("entropy ranking requires decoder digest")
        if method.startswith("object_") and not valid_digest(source_digests.get("association_sha256")):
            raise ValueError("object ranking requires association digest")
    elif version == RANKED_VERSION:
        if method not in RANKED_METHODS:
            raise ValueError("unsupported ranked sub-block method")
        if retained <= 0 or retained > 729:
            raise ValueError("ranked sub-block retained count must be in [1, 729]")
        if metadata.get("requested_budget") != retained:
            raise ValueError("ranked sub-block requested_budget mismatch")
        if metadata.get("consumed_count") != retained:
            raise ValueError("ranked sub-block consumed_count mismatch")
        if metadata.get("selection_inputs") != RANKED_INPUTS[method]:
            raise ValueError("ranked sub-block selection inputs mismatch")
        if metadata.get("ranking_scheme") != "nested_prefix_v1":
            raise ValueError("ranked sub-block ranking scheme mismatch")
        if metadata.get("feature_layout") != "subblock_1xk":
            raise ValueError("ranked sub-block requires subblock_1xk layout")
        if metadata.get("order_transform") != "ranked":
            raise ValueError("ranked sub-block order transform mismatch")
        if not isinstance(metadata.get("ranking_set"), str) or not metadata["ranking_set"]:
            raise ValueError("ranked sub-block ranking_set is invalid")
        for key in ("dense_source_sha256", "ranking_sha256", "ranking_prefix_sha256"):
            if not valid_digest(metadata.get(key)):
                raise ValueError(f"invalid ranked sub-block digest: {key}")
        decoded_digest = metadata.get("decoded_source_sha256")
        if method == "entropy_topk":
            if not valid_digest(decoded_digest):
                raise ValueError("entropy_topk requires a decoded-source digest")
        elif decoded_digest is not None:
            raise ValueError("unexpected ranked decoded-source digest")
        parameters = metadata.get("hyperparameters")
        expected_keys = {
            "voxel_random": {"max_budget", "voxel_size", "global_seed"},
            "fps": {"max_budget"},
            "semantic_kcenter": {"max_budget", "voxel_size", "alpha"},
            "joint_kcenter": {"max_budget", "voxel_size", "alpha"},
            "entropy_topk": {"max_budget", "feature_dim", "score"},
            "opacity_topk": {"max_budget", "score"},
        }[method]
        if not isinstance(parameters, dict) or set(parameters) != expected_keys:
            raise ValueError("ranked sub-block hyperparameters mismatch")
        if parameters.get("max_budget") != 729:
            raise ValueError("ranked maximum budget mismatch")
        if "voxel_size" in parameters and not 0 < parameters["voxel_size"] <= 1:
            raise ValueError("ranked voxel_size is invalid")
        if "alpha" in parameters and not 0 <= parameters["alpha"] <= 1:
            raise ValueError("ranked alpha is invalid")
        if method == "entropy_topk" and (
            parameters.get("feature_dim") != 3584
            or parameters.get("score") != "softmax_entropy_descending"
        ):
            raise ValueError("entropy_topk hyperparameters are invalid")
        if method == "opacity_topk" and parameters.get("score") != "opacity_descending":
            raise ValueError("opacity_topk hyperparameters are invalid")
    elif method == "uniform_random" and version in {SUBBLOCK_VERSION, TRAINED_FIELD_VERSION}:
        if retained <= 0 or retained > 729:
            raise ValueError("sub-block retained count must be in [1, 729]")
        if metadata.get("requested_budget") != retained:
            raise ValueError("sub-block requested_budget mismatch")
        if metadata.get("consumed_count") != retained:
            raise ValueError("sub-block consumed_count mismatch")
        if metadata.get("selection_inputs") != EXPECTED_SUBBLOCK_INPUTS:
            raise ValueError("unexpected sub-block selection inputs")
        if metadata.get("feature_layout") not in SUBBLOCK_LAYOUTS:
            raise ValueError("invalid sub-block feature layout")
        if metadata.get("feature_layout") == "legacy_27x27" and retained != 729:
            raise ValueError("legacy sub-block layout requires 729 rows")
        if metadata.get("ranking_scheme") not in {"nested_prefix_v1", "legacy_729_control"}:
            raise ValueError("invalid sub-block ranking scheme")
        if not isinstance(metadata.get("global_seed"), int):
            raise ValueError("sub-block global_seed must be an integer")
        if not isinstance(metadata.get("permutation_seed"), int):
            raise ValueError("sub-block permutation_seed must be an integer")
        for key in ("dense_source_sha256", "ranking_sha256", "ranking_prefix_sha256"):
            if not valid_digest(metadata.get(key)):
                raise ValueError(f"invalid sub-block digest: {key}")
        if metadata.get("order_transform") not in {"ranked", "shuffle_k64_v1"}:
            raise ValueError("invalid sub-block order transform")
        if version == TRAINED_FIELD_VERSION:
            if metadata.get("source_selection_version") != "sparse_training_prehead_v1":
                raise ValueError("trained-field source selection version mismatch")
            if metadata.get("permutation_seed_scheme") != "sparse_training_prehead_v1":
                raise ValueError("trained-field permutation seed scheme mismatch")
            for key in (
                "source_field_sha256",
                "source_checkpoint_sha256",
                "source_ranking_sha256",
            ):
                if not valid_digest(metadata.get(key)):
                    raise ValueError(f"invalid trained-field digest: {key}")
            if metadata["dense_source_sha256"] != metadata["source_field_sha256"]:
                raise ValueError("trained-field compatibility source digest mismatch")
    elif method == "random":
        if metadata.get("selection_variables") != EXPECTED_SELECTION_VARIABLES:
            raise ValueError("unexpected selection variables")
        for key in ("requested_ratio", "global_seed", "derived_seed"):
            if key not in metadata:
                raise ValueError(f"random metadata missing {key}")
    else:
        if metadata.get("selection_version") != ANCHOR_VERSION:
            raise ValueError("unsupported anchor selection version")
        if metadata.get("selection_inputs") != EXPECTED_ANCHOR_INPUTS:
            raise ValueError("anchor selection inputs must be points and features only")
        if metadata.get("requested_budget") != retained:
            raise ValueError("anchor requested_budget must equal retained rows")
        if retained % 729:
            raise ValueError("anchor budget must be a multiple of 729")
        digest = metadata.get("dense_source_sha256")
        if not valid_digest(digest):
            raise ValueError("invalid dense source digest")
        parameters = metadata.get("hyperparameters")
        required = {"voxel_size", "pca_dim", "alpha", "global_seed"}
        if not isinstance(parameters, dict) or set(parameters) != required:
            raise ValueError("invalid anchor hyperparameters")
        if parameters["pca_dim"] != 0:
            raise ValueError("version 1 anchor payload must have PCA disabled")
        if not isinstance(parameters["global_seed"], int):
            raise ValueError("anchor global_seed must be an integer")
        if not isinstance(parameters["voxel_size"], (int, float)) or not (
            0 < parameters["voxel_size"] <= 1
        ):
            raise ValueError("invalid anchor voxel_size")
        if not isinstance(parameters["alpha"], (int, float)) or not (
            0 <= parameters["alpha"] <= 1
        ):
            raise ValueError("invalid anchor alpha")
    return method


def validate_and_prepare(payload, scene_id, ntokens, expected_feature_dim=3584):
    if not isinstance(payload, dict):
        raise ValueError("preselected payload must be a dictionary")
    if "opacities" not in payload and "opacitites" in payload:
        payload = {**payload, "opacities": payload["opacitites"]}
    missing = [key for key in (*ALIGNED_KEYS, "selected_idx", "sparsity") if key not in payload]
    if missing:
        raise ValueError(f"preselected payload missing fields: {missing}")
    metadata = payload["sparsity"]
    retained = payload["features"].shape[0]
    if retained == 0:
        raise ValueError("preselected payload is empty")
    for key in ALIGNED_KEYS:
        tensor = payload[key]
        if not torch.is_tensor(tensor) or tensor.shape[0] != retained:
            raise ValueError(f"unaligned tensor: {key}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"non-finite values in {key}")
    if payload["features"].ndim != 2 or payload["features"].shape[1] != expected_feature_dim:
        raise ValueError(f"features must have shape N x {expected_feature_dim}")
    if tuple(payload["points"].shape[1:]) != (3,):
        raise ValueError("points must have shape N x 3")
    if tuple(payload["covariances"].shape[1:]) != (3, 3):
        raise ValueError("covariances must have shape N x 3 x 3")
    if payload["opacities"].ndim != 1:
        raise ValueError("opacities must have shape N")

    selected = payload["selected_idx"]
    source_count = metadata.get("source_count")
    if selected.dtype != torch.int64 or selected.ndim != 1 or len(selected) != retained:
        raise ValueError("selected_idx must be aligned one-dimensional int64")
    if retained != metadata.get("retained_count"):
        raise ValueError("retained_count metadata mismatch")
    if not isinstance(source_count, int) or source_count < retained:
        raise ValueError("invalid source_count metadata")
    if selected.min().item() < 0 or selected.max().item() >= source_count:
        raise ValueError("selected_idx is outside dense source range")
    if len(torch.unique(selected)) != retained:
        raise ValueError("selected_idx contains duplicates")

    method = validate_selection_metadata(metadata, scene_id, retained)

    if metadata.get("selection_version") in {
        SUBBLOCK_VERSION,
        RANKED_VERSION,
        TRAINED_FIELD_VERSION,
        PUBLIC_VERSION,
    }:
        if metadata.get("selection_version") == PUBLIC_VERSION:
            expected_prefix = hashlib.sha256(
                selected.cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()
            ).hexdigest()
        else:
            prefix_fields = {
                "ranking_sha256": metadata["ranking_sha256"],
                "requested_budget": retained,
                "order_transform": metadata["order_transform"],
            }
            expected_prefix = tensor_digest(selected, prefix_fields)
        if metadata["ranking_prefix_sha256"] != expected_prefix:
            raise ValueError("sub-block ranking prefix digest mismatch")
        consumed = retained
        layout = metadata["feature_layout"]
        features = payload["features"].half()
        if layout == "subblock_1xk":
            features = features.reshape(1, 1, retained, expected_feature_dim).permute(0, 3, 1, 2)
        else:
            features = features.reshape(1, 27, 27, expected_feature_dim).permute(0, 3, 1, 2)
        chunk_size = None
    else:
        chunk_size = metadata.get("chunk_size")
        if chunk_size != 729:
            raise ValueError("published LLaVA path requires chunk_size 729")
        consumed = min(retained, ntokens * chunk_size)
        if consumed <= 0 or consumed % chunk_size:
            raise ValueError("consumed rows must be a positive multiple of 729")
        features = payload["features"][:consumed].half()
        features = features.reshape(-1, 27, 27, expected_feature_dim).permute(0, 3, 1, 2)
    audit = {
        "selection_method": method,
        "scene_id": scene_id,
        "effective_ratio": metadata["effective_ratio"],
        "source_count": source_count,
        "retained_count": retained,
        "consumed_count": consumed,
        "ntokens": ntokens,
        "consumed_selected_idx": selected[:consumed].tolist(),
    }
    if chunk_size is not None:
        audit["chunk_size"] = chunk_size
    if metadata.get("selection_version") == PUBLIC_VERSION:
        audit.update({
            key: metadata[key] for key in (
                "selection_version", "requested_budget", "ranking_sha256",
                "ranking_prefix_sha256", "dense_source_sha256", "feature_layout",
                "parameters", "source_digests",
            )
        })
    elif metadata.get("selection_version") == RANKED_VERSION:
        audit.update({
            key: metadata[key] for key in (
                "selection_version", "requested_budget", "ranking_scheme", "ranking_set",
                "selection_inputs", "hyperparameters", "dense_source_sha256",
                "decoded_source_sha256", "ranking_sha256", "ranking_prefix_sha256",
                "feature_layout", "order_transform",
            )
        })
    elif metadata.get("selection_version") in {SUBBLOCK_VERSION, TRAINED_FIELD_VERSION}:
        audit.update({
            key: metadata[key] for key in (
                "selection_version", "requested_budget", "global_seed",
                "permutation_seed", "ranking_scheme", "dense_source_sha256",
                "ranking_sha256", "ranking_prefix_sha256", "feature_layout",
                "order_transform", "selection_inputs",
            )
        })
        if metadata.get("selection_version") == TRAINED_FIELD_VERSION:
            audit.update({
                key: metadata[key] for key in (
                    "source_selection_version",
                    "permutation_seed_scheme",
                    "source_field_sha256",
                    "source_checkpoint_sha256",
                    "source_ranking_sha256",
                )
            })
    elif method == "random":
        audit.update({
            "requested_ratio": metadata["requested_ratio"],
            "global_seed": metadata["global_seed"],
            "derived_seed": metadata["derived_seed"],
        })
    else:
        audit.update({
            "selection_version": metadata["selection_version"],
            "requested_budget": metadata["requested_budget"],
            "selection_inputs": metadata["selection_inputs"],
            "hyperparameters": metadata["hyperparameters"],
            "diagnostics": metadata.get("diagnostics", {}),
            "dense_source_sha256": metadata["dense_source_sha256"],
        })
    return features, audit


def blind_audit(scene_id):
    return {
        "selection_version": SUBBLOCK_VERSION,
        "selection_method": BLIND_METHOD,
        "scene_id": scene_id,
        "requested_budget": 0,
        "source_count": 0,
        "retained_count": 0,
        "consumed_count": 0,
        "effective_ratio": 0.0,
        "consumed_selected_idx": [],
        "feature_layout": "text_only",
    }


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def valid_completed_scene(pred_path, gt_path, audit_path, expected_items, timing_path=None):
    paths = [Path(pred_path), Path(gt_path), Path(audit_path)]
    if timing_path is not None:
        paths.append(Path(timing_path))
    if not all(path.is_file() for path in paths):
        return False
    try:
        predictions = json.loads(paths[0].read_text())
        ground_truth = json.loads(paths[1].read_text())
        audit = json.loads(paths[2].read_text())
        timing = json.loads(paths[3].read_text()) if timing_path is not None else None
    except (OSError, json.JSONDecodeError):
        return False
    expected_ids = [str(item.get("question_id")) for item in expected_items]
    pred_ids = [str(item.get("question_id")) for item in predictions]
    gt_ids = [str(item.get("question_id")) for item in ground_truth]
    content_matches = len(predictions) == len(expected_items) and len(ground_truth) == len(expected_items)
    if content_matches:
        for prediction, stored_gt, expected in zip(predictions, ground_truth, expected_items):
            if prediction.get("question") != expected.get("question"):
                content_matches = False
                break
            if stored_gt.get("question") != expected.get("question"):
                content_matches = False
                break
            if stored_gt.get("text") != expected.get("answers"):
                content_matches = False
                break
    return (
        pred_ids == expected_ids
        and gt_ids == expected_ids
        and content_matches
        and audit.get("scene_id") == Path(pred_path).stem
        and audit.get("selection_method") in ALLOWED_SELECTION_METHODS
        and (
            audit.get("consumed_count", -1) > 0
            or (
                audit.get("selection_method") == BLIND_METHOD
                and audit.get("selection_version") == SUBBLOCK_VERSION
                and audit.get("requested_budget") == 0
            )
        )
        and len(audit.get("consumed_selected_idx", [])) == audit.get("consumed_count")
        and (
            timing is None
            or (
                timing.get("scene_id") == Path(pred_path).stem
                and timing.get("question_count") == len(expected_items)
                and timing.get("elapsed_seconds", 0) > 0
            )
        )
    )
