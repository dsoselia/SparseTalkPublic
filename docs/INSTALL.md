# Installation

Install [CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/) first.
Use separate environments for the Gaussian, LLaVA, and optional detector stages.

## Gaussian selection, extraction, and decoding

```bash
conda create -n sparsetalk-gaussian python=3.10 -y
conda activate sparsetalk-gaussian
# Install CUDA-enabled torch/torchvision for this host first.
python -m pip install -e .
python -m pip install -r requirements/gaussian.txt
python -m pip install --no-build-isolation \
  'git+https://github.com/ngailapdi/diff-gaussian-rasterization-w-depth-feature.git@03bb8803477da3c4f8969605d5740157cd572877'
sparsetalk doctor --run-dir /scratch/sparsetalk-check
```

Build the rasterizer against the active PyTorch/CUDA toolchain. Its separate
non-commercial license is described in [NOTICE.md](../THIRD_PARTY/NOTICE.md).

## LLaVA preparation, inference, and LoRA training

```bash
conda create -n sparsetalk-llava python=3.10 -y
conda activate sparsetalk-llava
# Install CUDA-enabled torch/torchvision for this host first.
python -m pip install -e .
python -m pip install -r requirements/llava.txt
sparsetalk doctor --run-dir /scratch/sparsetalk-check
```

Single-GPU inference and training use PyTorch SDPA.

## Optional detected-object association

Install the Gaussian rasterizer and `requirements/detector.txt` in the detector
environment. External model revisions:

| Component | Revision |
| --- | --- |
| [SAM2 source](https://github.com/facebookresearch/sam2) | `2b90b9f5ceec907a1c18123530e92e794ad901a4` |
| Florence-2-large weights | `21a599d414c4d928c9032694c424fb94458e3594` |
| SAM2.1 Hiera Large weights | `665f8e2ad61cf5f53d65644ff27c8ee525124610` |

Keep model files and `HF_HOME` outside the checkout.
