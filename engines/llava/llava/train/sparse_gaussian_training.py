import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


FEATURE_MODES = {
    "dense_dynamic",
    "sparse_fixed",
    "sparse_pool",
    "sparse_mixed",
    "sparse_ranked",
}
MIXED_BUDGETS = (128, 256, 512, 729)
VIEW_SELECTION_VERSION = "sparse_training_views_v1"
VIEW_SEED = 20260829
DEFAULT_PAYLOAD_CACHE_BYTES = 20 * 2**30


def validate_view_sampling(value, path):
    if (not isinstance(value, dict)
            or value.get("selection_version") != VIEW_SELECTION_VERSION
            or value.get("global_seed") != VIEW_SEED
            or len(value.get("context_indices", [])) != 100
            or len(value.get("target_indices", [])) != 5):
        raise ValueError(f"invalid frozen view metadata: {path}")


def require_cuda_h200():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing sparse training on CPU")
    name = torch.cuda.get_device_name(0)
    return name


def stable_u64(*parts):
    value = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_indices(indices):
    return hashlib.sha256(indices.numpy().astype("<i8", copy=False).tobytes()).hexdigest()


def _load_payload(path):
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError(f"invalid Gaussian feature payload: {path}")
    features = payload["features"]
    if features.ndim != 2 or features.shape[1] != 256:
        raise ValueError(f"expected [N,256] encoded features in {path}, got {tuple(features.shape)}")
    if len(features) == 0 or not torch.isfinite(features).all().item():
        raise ValueError(f"empty or non-finite encoded features: {path}")
    return payload, features


class GaussianFeatureStore:
    def __init__(
        self,
        mode,
        dense_root=None,
        sparse_root=None,
        budget=729,
        pool_seeds=(0, 1, 2, 3, 4),
        global_seed=0,
        cache_bytes=DEFAULT_PAYLOAD_CACHE_BYTES,
    ):
        if mode not in FEATURE_MODES:
            raise ValueError(f"unknown Gaussian feature mode: {mode}")
        self.mode = mode
        self.dense_root = Path(dense_root) if dense_root else None
        self.sparse_root = Path(sparse_root) if sparse_root else None
        self.budget = int(budget)
        self.pool_seeds = tuple(int(seed) for seed in pool_seeds)
        self.global_seed = int(global_seed)
        self.cache_bytes = int(cache_bytes)
        self._digest_cache = {}
        self._validated_paths = set()
        self._payload_cache = OrderedDict()
        self._payload_cache_bytes = 0
        if self.budget <= 0:
            raise ValueError("Gaussian budget must be positive")
        if mode in {"dense_dynamic", "sparse_ranked"} and self.dense_root is None:
            raise ValueError(f"{mode} requires a dense feature root")
        if mode != "dense_dynamic" and self.sparse_root is None:
            raise ValueError(f"{mode} requires a sparse feature root")
        if not self.pool_seeds or len(self.pool_seeds) != len(set(self.pool_seeds)):
            raise ValueError("pool seeds must be nonempty and unique")
        if self.cache_bytes < 0:
            raise ValueError("payload cache size cannot be negative")

    @staticmethod
    def _tensor_bytes(payload):
        return sum(
            value.numel() * value.element_size()
            for value in payload.values()
            if torch.is_tensor(value)
        )

    def _load(self, path):
        path = Path(path).resolve()
        cached = self._payload_cache.pop(path, None)
        if cached is not None:
            self._payload_cache[path] = cached
            return cached
        payload, features = _load_payload(path)
        size = self._tensor_bytes(payload)
        if size <= self.cache_bytes:
            while self._payload_cache and self._payload_cache_bytes + size > self.cache_bytes:
                _, (evicted_payload, _) = self._payload_cache.popitem(last=False)
                self._payload_cache_bytes -= self._tensor_bytes(evicted_payload)
            self._payload_cache[path] = (payload, features)
            self._payload_cache_bytes += size
        return payload, features

    def clear_cache(self):
        self._payload_cache.clear()
        self._payload_cache_bytes = 0

    def _dense_path(self, scene_id):
        return self.dense_root / scene_id / "feat_fs" / f"{scene_id}.pt"

    def _sparse_path(self, scene_id, pool_seed):
        return self.sparse_root / f"seed_{pool_seed}" / scene_id / "feat_fs" / f"{scene_id}.pt"

    def _choice(self, scene_id, sample_id):
        selector = stable_u64("sparse-training-choice-v1", scene_id, sample_id, self.global_seed)
        pool_seed = self.pool_seeds[selector % len(self.pool_seeds)]
        if self.mode == "sparse_mixed":
            budget = MIXED_BUDGETS[(selector // len(self.pool_seeds)) % len(MIXED_BUDGETS)]
        else:
            budget = self.budget
        return pool_seed, budget

    def load(self, scene_id, sample_id):
        if self.mode == "dense_dynamic":
            path = self._dense_path(scene_id)
            payload, features = self._load(path)
            validate_view_sampling(payload.get("view_sampling"), path)
            budget = self.budget
            if len(features) < budget:
                raise ValueError(f"{scene_id} has {len(features)} rows, below budget {budget}")
            seed = stable_u64("dense-dynamic-v1", scene_id, sample_id, self.global_seed)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            indices = torch.randperm(len(features), generator=generator)[:budget]
            selected = features[indices]
            source_digest = self._digest_cache.get(path)
            if source_digest is None:
                source_digest = sha256_file(path)
                self._digest_cache[path] = source_digest
            metadata = {
                "mode": self.mode,
                "scene_id": scene_id,
                "sample_id": str(sample_id),
                "budget": budget,
                "selection_seed": seed,
                "source_count": len(features),
                "source_sha256": source_digest,
            }
            return selected, metadata

        if self.mode == "sparse_ranked":
            from sparsetalk.selection import verify_sparse

            path = self.sparse_root / scene_id / "feat_fs" / f"{scene_id}.pt"
            payload, features = self._load(path)
            ranking = payload.get("ranking")
            if ranking is None:
                raise ValueError(f"public sparse payload lacks its ranking: {path}")
            dense_path = self._dense_path(scene_id)
            if path not in self._validated_paths:
                verify_sparse(ranking, dense_path, path, scene_id)
                self._validated_paths.add(path)
            if self.budget > len(features):
                raise ValueError(f"{path} has fewer than {self.budget} rows")
            metadata = {
                "mode": self.mode,
                "scene_id": scene_id,
                "sample_id": str(sample_id),
                "budget": self.budget,
                "ranking_sha256": ranking["ranking"]["ranking_sha256"],
                "source_count": ranking["ranking"]["source_count"],
                "source_sha256": ranking["ranking"]["source_digests"]["dense_sha256"],
            }
            return features[:self.budget], metadata

        pool_seed, budget = self._choice(scene_id, sample_id)
        if self.mode == "sparse_fixed":
            pool_seed = self.pool_seeds[0]
        path = self._sparse_path(scene_id, pool_seed)
        payload, features = self._load(path)
        if len(features) < budget:
            raise ValueError(f"{path} has {len(features)} rows, below budget {budget}")
        sparsity = payload.get("sparsity")
        selected_idx = payload.get("selected_idx")
        if not isinstance(sparsity, dict) or selected_idx is None:
            raise ValueError(f"sparse payload lacks selection metadata: {path}")
        if selected_idx.dtype != torch.int64 or selected_idx.ndim != 1:
            raise ValueError(f"invalid selected_idx in {path}")
        if len(selected_idx) != len(features):
            raise ValueError(f"feature/index row mismatch in {path}")
        if len(torch.unique(selected_idx)) != len(selected_idx):
            raise ValueError(f"duplicate selected_idx in {path}")
        if sparsity.get("scene_id") != scene_id:
            raise ValueError(f"scene metadata mismatch in {path}")
        if int(sparsity.get("global_seed", -1)) != pool_seed:
            raise ValueError(f"pool seed mismatch in {path}")
        if sparsity.get("selection_version") != "sparse_training_prehead_v1":
            raise ValueError(f"selection version mismatch in {path}")
        validate_view_sampling(sparsity.get("view_sampling"), path)
        source_count = int(sparsity.get("source_count", -1))
        if source_count <= 0 or selected_idx.min().item() < 0 or selected_idx.max().item() >= source_count:
            raise ValueError(f"selected_idx is out of range in {path}")
        expected_seed = stable_u64(
            scene_id,
            "uniform_random",
            pool_seed,
            "sparse_training_prehead_v1",
        )
        if int(sparsity.get("permutation_seed", -1)) != expected_seed:
            raise ValueError(f"permutation seed mismatch in {path}")
        generator = torch.Generator(device="cpu").manual_seed(expected_seed)
        expected_idx = torch.randperm(source_count, generator=generator)[:len(selected_idx)]
        if not torch.equal(selected_idx, expected_idx):
            raise ValueError(f"selected_idx is not the expected ranking prefix in {path}")
        if sparsity.get("ranking_sha256") != sha256_indices(selected_idx):
            raise ValueError(f"ranking digest mismatch in {path}")
        metadata = {
            "mode": self.mode,
            "scene_id": scene_id,
            "sample_id": str(sample_id),
            "budget": budget,
            "pool_seed": pool_seed,
            "ranking_sha256": sparsity.get("ranking_sha256"),
            "source_count": source_count,
            "source_sha256": sparsity.get("dense_source_sha256"),
        }
        return features[:budget], metadata

    @staticmethod
    def format_features(features, legacy_blocks=False):
        if features.ndim != 2 or features.shape[1] != 256:
            raise ValueError("encoded Gaussian features must have shape [K,256]")
        budget = len(features)
        if legacy_blocks:
            if budget % 729:
                raise ValueError("legacy Gaussian layout requires a multiple of 729 rows")
            blocks = budget // 729
            return features.reshape(blocks, 27, 27, 256).permute(0, 3, 1, 2).contiguous()
        return features.transpose(0, 1).reshape(1, 256, 1, budget).contiguous()


class FrozenGaussianDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.ModuleList(
            [
                nn.Linear(256, 512),
                nn.GELU(),
                nn.Linear(512, 1024),
                nn.GELU(),
                nn.Linear(1024, 2048),
                nn.GELU(),
                nn.Linear(2048, 3584),
            ]
        )

    def forward(self, features):
        output = features
        for layer in self.decoder:
            output = layer(output)
        return output


def load_frozen_decoder(checkpoint_path, device, dtype):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("autoencoder checkpoint must be a state dictionary")
    decoder_state = {}
    for key, value in checkpoint.items():
        normalized = key.removeprefix("module.")
        if normalized.startswith("decoder."):
            decoder_state[normalized] = value
    model = FrozenGaussianDecoder()
    incompatible = model.load_state_dict(decoder_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"autoencoder decoder state mismatch: {incompatible}")
    model.requires_grad_(False)
    model.eval()
    return model.to(device=device, dtype=dtype)


def decode_feature_images(decoder, image_features):
    if decoder is None:
        raise RuntimeError("encoded Gaussian features were provided without a frozen decoder")
    decoded = []
    for tensor in image_features:
        if tensor.ndim != 4 or tensor.shape[1] != 256:
            raise ValueError(f"expected [N,256,H,W] encoded features, got {tuple(tensor.shape)}")
        n, _, h, w = tensor.shape
        parameter = next(decoder.parameters())
        rows = tensor.permute(0, 2, 3, 1).reshape(-1, 256).to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.no_grad():
            output = decoder(rows)
        if not torch.isfinite(output).all().item():
            raise ValueError("decoded Gaussian features contain non-finite values")
        decoded.append(output.reshape(n, h, w, 3584).permute(0, 3, 1, 2).contiguous())
    return decoded


def write_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
