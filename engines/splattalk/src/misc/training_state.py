import random

import numpy as np
import torch


FORMAT_VERSION = "splattalk_resumable_training_state_v1"


def capture_training_state(step_tracker):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG state cannot be captured without CUDA")
    return {
        "format_version": FORMAT_VERSION,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "step_tracker_step": None if step_tracker is None else step_tracker.get_step(),
    }


def restore_training_state(state, step_tracker):
    if state.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported resumable training-state format")
    required = (
        "python_random_state",
        "numpy_random_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
        "step_tracker_step",
    )
    missing = [name for name in required if name not in state]
    if missing:
        raise ValueError(f"resumable training state is missing fields: {missing}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_cpu_rng_state"])
    cuda_states = state["torch_cuda_rng_state_all"]
    if len(cuda_states) != torch.cuda.device_count():
        raise ValueError("saved CUDA RNG state does not match the allocated devices")
    torch.cuda.set_rng_state_all(cuda_states)
    if step_tracker is not None and state["step_tracker_step"] is not None:
        step_tracker.set_step(int(state["step_tracker_step"]))
