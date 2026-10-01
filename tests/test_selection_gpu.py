import sys
from pathlib import Path

import pytest
import torch

from sparsetalk.paths import LLAVA_ENGINE, SPLATTALK_ENGINE
from sparsetalk.selection import (
    CORE_METHODS, _load_decoder, atomic_torch_save, create_ranking,
    materialize, validate_ranking, verify_sparse,
)


@pytest.fixture
def source(tmp_path):
    torch.manual_seed(3)
    count = 800
    payload = {
        "features": torch.randn(count, 256).half(),
        "points": torch.rand(count, 3),
        "covariances": torch.eye(3).expand(count, -1, -1).clone(),
        "opacities": torch.rand(count),
    }
    path = tmp_path / "scene0" / "feat_fs" / "scene0.pt"
    atomic_torch_save(path, payload)
    return path, payload


@pytest.fixture
def decoder_checkpoint(tmp_path):
    sys.path.insert(0, str(SPLATTALK_ENGINE / "autoencoder"))
    from model import Autoencoder

    checkpoint = tmp_path / "autoencoder.pth"
    torch.save({f"module.{key}": value for key, value in Autoencoder().state_dict().items()}, checkpoint)
    return checkpoint


def test_all_core_methods_make_valid_nested_prefixes(source, decoder_checkpoint, tmp_path):
    path, dense = source
    for method in CORE_METHODS:
        ranking = create_ranking(
            path, "scene0", method, 16, global_seed=2,
            decoder_checkpoint=decoder_checkpoint if method == "entropy_topk" else None,
        )
        repeated = create_ranking(
            path, "scene0", method, 16, global_seed=2,
            decoder_checkpoint=decoder_checkpoint if method == "entropy_topk" else None,
        )
        assert torch.equal(ranking["selected_idx"], repeated["selected_idx"])
        assert ranking["ranking"] == repeated["ranking"]
        validate_ranking(ranking, path, "scene0")
        short = materialize(ranking, path, "scene0", 4)
        long = materialize(ranking, path, "scene0", 16)
        assert torch.equal(short["selected_idx"], long["selected_idx"][:4])
        for key in ("features", "points", "covariances", "opacities"):
            assert torch.equal(short[key], dense[key][short["selected_idx"]])
        sparse_path = tmp_path / method / "scene0.pt"
        atomic_torch_save(sparse_path, short)
        verify_sparse(ranking, path, sparse_path, "scene0")


def test_random_seed_independence_and_tamper(source, tmp_path):
    path, _ = source
    first = create_ranking(path, "scene0", "uniform_random", 64, global_seed=0)
    same = create_ranking(path, "scene0", "uniform_random", 64, global_seed=0)
    other = create_ranking(path, "scene0", "uniform_random", 64, global_seed=1)
    assert torch.equal(first["selected_idx"], same["selected_idx"])
    assert not torch.equal(first["selected_idx"], other["selected_idx"])
    sparse = materialize(first, path, "scene0", 8)
    sparse_path = tmp_path / "sparse.pt"
    sparse["features"][0, 0] += 1
    atomic_torch_save(sparse_path, sparse)
    with pytest.raises(ValueError, match="sparse features"):
        verify_sparse(first, path, sparse_path, "scene0")
    first["selected_idx"][0] = first["selected_idx"][1]
    with pytest.raises(ValueError):
        validate_ranking(first, path, "scene0")


def test_source_digest_and_payload_validation(source):
    path, dense = source
    ranking = create_ranking(path, "scene0", "uniform_random", 8)
    dense["opacities"][0] += 0.1
    atomic_torch_save(path, dense)
    with pytest.raises(ValueError, match="dense source digest"):
        validate_ranking(ranking, path, "scene0")
    legacy = dict(dense)
    legacy["opacitites"] = legacy.pop("opacities")
    atomic_torch_save(path, legacy)
    create_ranking(path, "scene0", "uniform_random", 8)
    legacy["features"][0, 0] = float("nan")
    atomic_torch_save(path, legacy)
    with pytest.raises(ValueError, match="non-finite"):
        create_ranking(path, "scene0", "uniform_random", 8)
    legacy["features"][0, 0] = 0
    legacy["points"] = legacy["points"][:-1]
    atomic_torch_save(path, legacy)
    with pytest.raises(ValueError, match="row mismatch"):
        create_ranking(path, "scene0", "uniform_random", 8)


def test_entropy_and_opacity_stable_ties(source, decoder_checkpoint):
    path, dense = source
    dense["opacities"].fill_(0.5)
    atomic_torch_save(path, dense)
    opacity = create_ranking(path, "scene0", "opacity_topk", 8)
    assert torch.equal(opacity["selected_idx"], torch.arange(8, dtype=torch.int64))
    state = torch.load(decoder_checkpoint, map_location="cpu", weights_only=False)
    for value in state.values():
        value.zero_()
    torch.save(state, decoder_checkpoint)
    entropy = create_ranking(path, "scene0", "entropy_topk", 8,
                             decoder_checkpoint=decoder_checkpoint)
    assert torch.equal(entropy["selected_idx"], torch.arange(8, dtype=torch.int64))


def test_decoded_subset_matches_dense_rows(source, decoder_checkpoint):
    path, dense = source
    ranking = create_ranking(path, "scene0", "uniform_random", 32)
    sparse = materialize(ranking, path, "scene0", 8)
    decoder = _load_decoder(decoder_checkpoint)
    with torch.inference_mode():
        full = decoder.decode(dense["features"].float().cuda()).cpu()
        selected = decoder.decode(sparse["features"].float().cuda()).cpu()
    torch.testing.assert_close(selected, full[sparse["selected_idx"]], rtol=1e-5, atol=1e-5)


def test_preselected_uses_exact_prefix_without_entropy(source, decoder_checkpoint, monkeypatch):
    import torch

    sys.path.insert(0, str(LLAVA_ENGINE / "scripts"))
    import gaussian_utils
    from gaussian_feature_loader import prepare_single_payload

    path, _ = source
    ranking = create_ranking(path, "scene0", "uniform_random", 16)
    sparse = materialize(ranking, path, "scene0", 8)
    decoder = _load_decoder(decoder_checkpoint)
    with torch.inference_mode():
        sparse["features"] = decoder.decode(sparse["features"].float().cuda()).cpu()
    monkeypatch.setattr(gaussian_utils, "select_high_entropy_gaussians", lambda *_: (_ for _ in ()).throw(AssertionError("entropy called")))
    features, audit = prepare_single_payload(sparse, "scene0", 44, "preselected")
    assert features.shape == (1, 3584, 1, 8)
    assert audit["consumed_selected_idx"] == sparse["selected_idx"].tolist()


def test_ranked_training_store(source, tmp_path):
    sys.path.insert(0, str(LLAVA_ENGINE))
    from llava.train.sparse_gaussian_training import GaussianFeatureStore

    path, _ = source
    dense_root = path.parents[2]
    sparse_root = tmp_path / "training_sparse"
    ranking = create_ranking(path, "scene0", "uniform_random", 16)
    sparse = materialize(ranking, path, "scene0", 8)
    atomic_torch_save(sparse_root / "scene0" / "feat_fs" / "scene0.pt", sparse)
    store = GaussianFeatureStore("sparse_ranked", dense_root=dense_root,
                                 sparse_root=sparse_root, budget=4)
    features, metadata = store.load("scene0", "question-1")
    assert features.shape == (4, 256)
    assert metadata["ranking_sha256"] == ranking["ranking"]["ranking_sha256"]
