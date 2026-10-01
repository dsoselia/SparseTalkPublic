# SparseTalkPublic

Gaussian sparsification, decoding, inference, and training with SplatTalk and
LLaVA. Inputs are dense 256D Gaussian features or ScanNet-style posed RGB-D
scenes. Checkpoints and datasets are separate downloads.

## Requirements

- Linux, Python 3.10+, NVIDIA CUDA, and CUDA-enabled PyTorch.
- A matching CUDA toolkit for the Gaussian rasterizer.
- SplatTalk, autoencoder, and LLaVA checkpoints; optional LoRA and Florence/SAM
  checkpoints. LLaVA also accepts a Hugging Face model ID.

See [installation](docs/INSTALL.md), [input formats](docs/FORMATS.md),
[methods](docs/METHODS.md), [training](docs/TRAINING.md), and
[verification](docs/TESTING.md). Install the CLI in each environment:

```bash
python -m pip install -e .
sparsetalk --help
```

`--config examples/config.json` supplies JSON defaults; CLI flags override them.

## Existing Dense Scene

Input: `$DENSE/scene0011_00/feat_fs/scene0011_00.pt`. Use the Gaussian environment
for selection/decoding and the LLaVA environment for inference.

```bash
RUN=/scratch/sparsetalk-example
DENSE=/data/dense_gaussians
AE=/models/autoencoder.pth

sparsetalk select --run-dir "$RUN" --dense-root "$DENSE" \
  --scenes scene0011_00 --method uniform_random --seed 0 \
  --rank-length 729 --budget 128
```

Set `SPARSE` and `RANKINGS` to the printed `sparse_root` and `ranking_root`:

```bash
sparsetalk verify --run-dir "$RUN" --dense-root "$DENSE" \
  --ranking-root "$RANKINGS" --sparse-root "$SPARSE" --scenes scene0011_00
sparsetalk decode --run-dir "$RUN" --sparse-root "$SPARSE" \
  --scenes scene0011_00 --autoencoder-checkpoint "$AE"
sparsetalk infer --run-dir "$RUN" --sparse-root "$SPARSE" \
  --model-base lmms-lab/llava-onevision-qwen2-7b-ov \
  --autoencoder-checkpoint "$AE" \
  --scene scene0011_00 --prompt "What color is the chair?"
```

`infer` prints its output directory under `$RUN/inference/<id>/`, containing
predictions, references, audits, logs, and timings. Add `--adapter /path/to/adapter`
for LoRA or `--questions questions.json` for multiple questions.

## From Posed RGB-D

Each scene needs numeric RGB/depth frames, intrinsics, and ordered extrinsics
([format](docs/FORMATS.md)). `prepare` links the inputs and generates frame
features under `$RUN`.

```bash
RUN=/scratch/sparsetalk-raw
SCENES=scene0011_00
RAW=/data/scannet/scans
AE=/models/autoencoder.pth
GS=/models/splattalk.ckpt
BASE=lmms-lab/llava-onevision-qwen2-7b-ov

sparsetalk prepare --run-dir "$RUN" --scene-root "$RAW" --scenes "$SCENES" \
  --model-base "$BASE" --autoencoder-checkpoint "$AE"
sparsetalk extract --run-dir "$RUN" --scenes "$SCENES" \
  --gaussian-checkpoint "$GS" --context-views 100 --target-views 5
sparsetalk select --run-dir "$RUN" --dense-root "$RUN/dense" \
  --scenes "$SCENES" --method fps --rank-length 729 --budget 128
```

Continue with `verify`, `decode`, and `infer` as above. See
[training](docs/TRAINING.md) for Gaussian, autoencoder, and LoRA workflows.

## Object Associations

`associate --kind detected` uses Florence-2/SAM2; `--kind masks` uses ScanNet
instance masks and matching `.sens` files. Pass the association root to an
object-balanced selector. See [methods](docs/METHODS.md) for parameters.

## Repository Layout

- `src/sparsetalk/`: CLI, validation, ranking, and workflow wrappers.
- `engines/splattalk/`: Gaussian and autoencoder engine.
- `engines/llava/`: LLaVA inference and training engine.
- `tests/`, `tools/`: CUDA tests and smoke checks.
- `THIRD_PARTY/`: attribution and licenses.

`sparsetalk doctor --run-dir "$RUN"` checks CUDA and engine paths. Use a run
directory and `HF_HOME` outside this checkout. Licenses: [THIRD_PARTY/NOTICE.md](THIRD_PARTY/NOTICE.md).
