"""Deterministic, nested Gaussian rankings and aligned sparse payloads."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from sparsetalk.paths import SPLATTALK_ENGINE, canonical_sha256, file_sha256, require_cuda

AUTOENCODER = SPLATTALK_ENGINE / "autoencoder"
if str(AUTOENCODER) not in sys.path:
    sys.path.insert(0, str(AUTOENCODER))

from anchor_selection import select_anchors  # noqa: E402
from model import Autoencoder  # noqa: E402
from sparsity_utils import ALIGNED_KEYS, normalize_payload, validate_aligned  # noqa: E402


VERSION = "sparsetalk_public_v1"
CORE_METHODS = (
    "uniform_random", "voxel_random", "fps", "semantic_kcenter",
    "joint_kcenter", "adaptive_anchors", "entropy_topk", "opacity_topk",
)
OBJECT_METHODS = ("object_balanced_random", "object_balanced_fps")
METHODS = CORE_METHODS + OBJECT_METHODS


def stable_seed(scene_id: str, method: str, global_seed: int) -> int:
    source = f"{VERSION}|{scene_id}|{method}|{global_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(source).digest()[:8], "big")


def indices_sha256(indices: torch.Tensor) -> str:
    return hashlib.sha256(indices.cpu().contiguous().numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def atomic_torch_save(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_dense(path: Path) -> dict:
    dense = normalize_payload(torch.load(path, map_location="cpu", weights_only=False, mmap=True))
    validate_aligned(dense, expected_feature_dim=256)
    return dense


def _load_decoder(checkpoint_path: Path) -> Autoencoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("autoencoder checkpoint must be a state dictionary")
    state = {key.removeprefix("module."): value for key, value in checkpoint.items()}
    model = Autoencoder().cuda().eval()
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    return model


def _entropy_scores(features: torch.Tensor, decoder: Autoencoder, batch_size: int) -> torch.Tensor:
    scores = []
    for start in range(0, len(features), batch_size):
        with torch.inference_mode():
            decoded = decoder.decode(features[start:start + batch_size].float().cuda())
            if not torch.isfinite(decoded).all().item():
                raise ValueError("non-finite decoded features")
            probabilities = torch.softmax(decoded.float(), dim=1)
            entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)
            scores.append(entropy.cpu())
    return torch.cat(scores)


def _object_order(
    dense: dict, scene_id: str, method: str, association_path: Path,
    allocation_lambda: float, gamma: float, q_bg: float, global_seed: int,
    dense_sha256: str,
) -> tuple[torch.Tensor, dict]:
    if not association_path.is_file():
        raise FileNotFoundError(association_path)
    association = torch.load(association_path, map_location="cpu", weights_only=False)
    source = str(association.get("association", {}).get("selection_version", ""))
    # Both association implementations carry their own version in metadata.
    metadata = association.get("association") or association.get("metadata") or {}
    source = metadata.get("selection_version", source)
    if metadata.get("input_digests", {}).get("dense_encoded") != dense_sha256:
        raise ValueError("association dense-source digest mismatch")
    package = "oracle_object_sampling" if "oracle" in str(source) else "detected_object_sampling"
    module = __import__(f"{package}.ranking", fromlist=["build_ranking"])
    policy = {"lambda": float(allocation_lambda), "q_bg": float(q_bg), "alpha": float(gamma)}
    ranking = module.build_ranking(
        association, dense, scene_id, method, policy,
        global_seed=int(global_seed) if method == "object_balanced_random" else 0,
    )
    return ranking["selected_idx"], {
        "association_sha256": file_sha256(association_path),
        "association_kind": package,
        "allocation": {"lambda": allocation_lambda, "gamma": gamma, "q_bg": q_bg},
        "pool_sequence_sha256": ranking["ranking"].get("pool_sequence_sha256"),
    }


def create_ranking(
    dense_path: str | Path, scene_id: str, method: str, rank_length: int,
    *, global_seed: int = 0, voxel_size: float = 0.01, alpha: float = 0.5,
    decoder_checkpoint: str | Path | None = None,
    association_path: str | Path | None = None,
    allocation_lambda: float = 0.5, gamma: float = 0.5, q_bg: float = 0.2,
    batch_size: int = 256,
) -> dict:
    require_cuda()
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    path = Path(dense_path)
    dense = _load_dense(path)
    count = len(dense["features"])
    if rank_length <= 0 or rank_length > count:
        raise ValueError("rank length must be in [1, source row count]")
    if method in OBJECT_METHODS and rank_length > 729:
        raise ValueError("object association rankings currently support at most 729 rows")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    parameters: dict = {"rank_length": rank_length, "global_seed": int(global_seed)}
    source_digests: dict = {"dense_sha256": file_sha256(path)}
    if method == "uniform_random":
        seed = stable_seed(scene_id, method, global_seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        indices = torch.randperm(count, generator=generator)[:rank_length]
        parameters["derived_seed"] = seed
    elif method == "opacity_topk":
        indices = torch.argsort(dense["opacities"].float(), descending=True, stable=True)[:rank_length]
    elif method == "entropy_topk":
        if not decoder_checkpoint:
            raise ValueError("entropy_topk requires --autoencoder-checkpoint")
        checkpoint = Path(decoder_checkpoint)
        parameters["entropy_batch_size"] = batch_size
        source_digests["decoder_sha256"] = file_sha256(checkpoint)
        decoder = _load_decoder(checkpoint)
        scores = _entropy_scores(dense["features"], decoder, batch_size)
        indices = torch.argsort(scores, descending=True, stable=True)[:rank_length]
    elif method in OBJECT_METHODS:
        if not association_path:
            raise ValueError("object methods require --association-root")
        indices, object_metadata = _object_order(
            dense, scene_id, method, Path(association_path),
            allocation_lambda, gamma, q_bg, global_seed,
            source_digests["dense_sha256"],
        )
        indices = indices[:rank_length]
        parameters.update(object_metadata["allocation"])
        source_digests["association_sha256"] = object_metadata["association_sha256"]
        parameters["association_kind"] = object_metadata["association_kind"]
        parameters["pool_sequence_sha256"] = object_metadata["pool_sequence_sha256"]
    else:
        parameters.update({"voxel_size": float(voxel_size), "alpha": float(alpha)})
        indices, _ = select_anchors(
            dense["points"].cuda(), dense["features"].cuda(), rank_length,
            method, voxel_size=voxel_size, alpha=alpha,
            global_seed=global_seed, scene_id=scene_id,
        )
    indices = indices.cpu().to(torch.int64).contiguous()
    if indices.shape != (rank_length,) or len(torch.unique(indices)) != rank_length:
        raise ValueError("selector returned invalid or duplicate indices")
    if indices.min().item() < 0 or indices.max().item() >= count:
        raise ValueError("selector returned out-of-range indices")
    metadata = {
        "selection_version": VERSION,
        "selection_method": method,
        "scene_id": scene_id,
        "source_count": count,
        "rank_length": rank_length,
        "parameters": parameters,
        "source_digests": source_digests,
        "index_sha256": indices_sha256(indices),
    }
    metadata["ranking_sha256"] = canonical_sha256(metadata)
    return {"selected_idx": indices, "ranking": metadata}


def validate_ranking(ranking: dict, dense_path: str | Path, scene_id: str,
                     decoder_checkpoint: str | Path | None = None,
                     association_path: str | Path | None = None) -> dict:
    if not isinstance(ranking, dict) or set(ranking) != {"selected_idx", "ranking"}:
        raise ValueError("ranking must contain selected_idx and ranking metadata")
    metadata = ranking["ranking"]
    indices = ranking["selected_idx"]
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("parameters"), dict)
            or not isinstance(metadata.get("source_digests"), dict)):
        raise ValueError("ranking metadata is malformed")
    if metadata.get("selection_version") != VERSION or metadata.get("scene_id") != scene_id:
        raise ValueError("ranking version or scene mismatch")
    if metadata.get("selection_method") not in METHODS:
        raise ValueError("unknown ranking method")
    count = metadata.get("source_count")
    length = metadata.get("rank_length")
    if not isinstance(count, int) or not isinstance(length, int) or not 0 < length <= count:
        raise ValueError("invalid ranking dimensions")
    if not torch.is_tensor(indices) or indices.dtype != torch.int64 or indices.shape != (length,):
        raise ValueError("ranking indices must be one-dimensional int64")
    if len(torch.unique(indices)) != length or indices.min().item() < 0 or indices.max().item() >= count:
        raise ValueError("ranking indices are duplicate or out of range")
    if metadata.get("index_sha256") != indices_sha256(indices):
        raise ValueError("ranking indices digest mismatch")
    unsigned = {key: value for key, value in metadata.items() if key != "ranking_sha256"}
    if metadata.get("ranking_sha256") != canonical_sha256(unsigned):
        raise ValueError("ranking metadata digest mismatch")
    if metadata.get("source_digests", {}).get("dense_sha256") != file_sha256(dense_path):
        raise ValueError("dense source digest mismatch")
    sources = metadata["source_digests"]
    method = metadata["selection_method"]
    if method == "entropy_topk":
        if not isinstance(sources.get("decoder_sha256"), str):
            raise ValueError("entropy ranking lacks decoder digest")
        if decoder_checkpoint and sources["decoder_sha256"] != file_sha256(decoder_checkpoint):
            raise ValueError("decoder checkpoint digest mismatch")
    if method in OBJECT_METHODS:
        if not isinstance(sources.get("association_sha256"), str):
            raise ValueError("object ranking lacks association digest")
        if association_path and sources["association_sha256"] != file_sha256(association_path):
            raise ValueError("association digest mismatch")
    if len(_load_dense(Path(dense_path))["features"]) != count:
        raise ValueError("dense source count mismatch")
    return metadata


def materialize(ranking: dict, dense_path: str | Path, scene_id: str, budget: int) -> dict:
    metadata = validate_ranking(ranking, dense_path, scene_id)
    if not 0 < budget <= metadata["rank_length"]:
        raise ValueError("budget must be positive and no larger than ranking length")
    dense = _load_dense(Path(dense_path))
    selected = ranking["selected_idx"][:budget].clone()
    result = {key: dense[key][selected] for key in ALIGNED_KEYS}
    result["selected_idx"] = selected
    result["ranking"] = ranking
    result["sparsity"] = {
        "selection_version": VERSION,
        "selection_method": metadata["selection_method"],
        "scene_id": scene_id,
        "source_count": metadata["source_count"],
        "requested_budget": budget,
        "retained_count": budget,
        "consumed_count": budget,
        "effective_ratio": budget / metadata["source_count"],
        "feature_layout": "subblock_1xk",
        "ranking_sha256": metadata["ranking_sha256"],
        "ranking_prefix_sha256": indices_sha256(selected),
        "dense_source_sha256": metadata["source_digests"]["dense_sha256"],
        "parameters": metadata["parameters"],
        "source_digests": metadata["source_digests"],
    }
    validate_aligned(result, expected_feature_dim=256)
    return result


def verify_sparse(ranking: dict, dense_path: str | Path, sparse_path: str | Path, scene_id: str) -> dict:
    sparse = normalize_payload(torch.load(sparse_path, map_location="cpu", weights_only=False), require_sparsity=True)
    embedded = sparse.get("ranking")
    if (not isinstance(embedded, dict) or embedded.get("ranking") != ranking.get("ranking")
            or not torch.is_tensor(embedded.get("selected_idx"))
            or not torch.equal(embedded["selected_idx"], ranking.get("selected_idx"))):
        raise ValueError("embedded ranking mismatch")
    budget = validate_aligned(sparse, expected_feature_dim=256)
    expected = materialize(ranking, dense_path, scene_id, budget)
    if sparse["sparsity"] != expected["sparsity"]:
        raise ValueError("sparse metadata mismatch")
    if not torch.equal(sparse["selected_idx"], expected["selected_idx"]):
        raise ValueError("sparse index prefix mismatch")
    for key in ALIGNED_KEYS:
        if not torch.equal(sparse[key], expected[key]):
            raise ValueError(f"sparse {key} does not equal selected dense rows")
    return sparse["sparsity"]
