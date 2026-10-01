import sys

import torch

from sparsetalk.paths import SPLATTALK_ENGINE


sys.path.insert(0, str(SPLATTALK_ENGINE / "autoencoder"))


def test_object_allocation_and_orders():
    from detected_object_sampling.ranking import (
        pool_weights as detected_weights, random_pool_order as detected_order,
        validate_policy as validate_detected, deficit_interleave as detected_interleave,
        fps_pool_order as detected_fps,
    )
    from oracle_object_sampling.ranking import (
        pool_weights as mask_weights, random_pool_order as mask_order,
        validate_policy as validate_masks, deficit_interleave as mask_interleave,
        fps_pool_order as mask_fps,
    )

    policy = {"lambda": 0.37, "alpha": 0.6, "q_bg": 0.18}
    validate_detected(policy)
    validate_masks(policy)
    pools = torch.tensor([0, 1, 1, 2, 2, 2], dtype=torch.int32)
    mass = torch.tensor([1.0, 1.0, 2.0, 1.0, 2.0, 3.0])
    for function in (detected_weights, mask_weights):
        weights = function(pools, mass, policy)
        assert abs(sum(weights.values()) - 1) < 1e-12
        assert set(weights) == {0, 1, 2}
    indices = torch.arange(6, dtype=torch.int64)
    for function in (detected_order, mask_order):
        first = function(indices, "scene0", 4, 1)
        again = function(indices, "scene0", 4, 1)
        assert torch.equal(first, again)
    orders = {0: torch.tensor([0]), 1: torch.tensor([1, 2]), 2: torch.tensor([3, 4, 5])}
    for function in (detected_interleave, mask_interleave):
        selected, selected_pools = function(orders, {0: 0.2, 1: 0.3, 2: 0.5}, "scene0", budget=6)
        assert len(selected) == len(selected_pools) == 6
        assert set(selected.tolist()) == set(range(6))
    points = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    for function in (detected_fps, mask_fps):
        selected = function(indices, points, mass, limit=4)
        assert len(torch.unique(selected)) == 4
