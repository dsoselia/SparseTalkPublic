# Selection Methods

Each scene/configuration has one ranking. `--budget k` selects its first `k`
rows; `--rank-length` defaults to 729. Prefixes from the same ranking are nested.

| `--method` | Selection inputs | Additional options |
| --- | --- | --- |
| `uniform_random` | source row count | `--seed` |
| `voxel_random` | 3D points | `--seed`, `--voxel-size` |
| `fps` | 3D points | none |
| `semantic_kcenter` | points, encoded 256D features | `--voxel-size` |
| `joint_kcenter` | points, encoded 256D features | `--voxel-size`, `--alpha` |
| `adaptive_anchors` | points, encoded 256D features | `--voxel-size`, `--alpha` |
| `entropy_topk` | encoded features, external autoencoder decoder | `--autoencoder-checkpoint`, `--batch-size` |
| `opacity_topk` | opacity values | none |
| `object_balanced_random` | object association, points, visible mass | `--association-root`, `--seed`, allocation options |
| `object_balanced_fps` | object association, points, visible mass | `--association-root`, allocation options |

Entropy is scored in decoded batches. Ties use source-row order. `--alpha`
controls the spatial-semantic mixture; `--gamma` below controls object size weighting.

## Object associations

Florence-2/SAM2 detections or ScanNet instance masks are associated with
Gaussians through alpha-compositing contributions. Association and ranking
take scene inputs only.

Detected example:

```bash
sparsetalk associate --run-dir "$RUN" --kind detected \
  --scenes scene0011_00 --scene-root "$RAW" --dense-root "$DENSE" \
  --florence-model /models/Florence-2-large \
  --sam2-repo /opt/sam2 --sam2-checkpoint /models/sam2.1_hiera_large.pt
```

ScanNet instance-mask example:

```bash
sparsetalk associate --run-dir "$RUN" --kind masks \
  --scenes scene0011_00 --dense-root "$DENSE" \
  --scans-root /data/scannet/scans --sens-root /data/scannet/scans
```

Pass the resulting association root to `select`:

```bash
sparsetalk select --run-dir "$RUN" --dense-root "$DENSE" \
  --scenes scene0011_00 --method object_balanced_random --seed 0 \
  --association-root "$RUN/associations/detected" \
  --allocation-lambda 0.5 --gamma 0.5 --q-bg 0.2 \
  --rank-length 729 --budget 128
```

`lambda` mixes equal-instance and size-based allocation, `gamma` is the size
exponent, and `q_bg` is the background share. Valid ranges: `lambda, q_bg` in
`[0,1]`, finite `gamma > 0`. Object rankings support up to 729 rows.

## Validation

`verify` checks hashes, indices, prefixes, metadata, and aligned rows.
`--recompute` regenerates the ranking from the decoder/association inputs.
Inference records the exact consumed indices in each scene audit.
