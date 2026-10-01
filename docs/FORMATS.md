# Inputs And Outputs

## Raw scene

`--scene-root` contains one directory per scene:

```text
scene0011_00/
  color/0.jpg, 10.jpg, ...
  depth/0.png, 10.png, ...
  intrinsic/intrinsic_color.txt
  extrinsics.npy
```

Frame stems are numeric and matched by stem. `extrinsics.npy` is a finite
`[frame_count,4,4]` array in numeric RGB-frame order. The intrinsics file is a
finite 3x3 or 4x4 matrix. Depth PNGs use the upstream ScanNet convention.
`prepare` writes links and frame features under `<run>/inputs/{scanqa,train}/<scene>`.
Dataset adapters must produce this frame/camera layout.

## Dense Gaussian payload

`<dense-root>/<scene>/feat_fs/<scene>.pt` is a PyTorch dictionary containing
aligned tensors: `features` `[N,256]`, `points` `[N,3]`, `covariances`
`[N,3,3]`, and `opacities` `[N]`. Legacy `opacitites` is accepted on input.
Rows must be finite, nonempty, and aligned.

## Ranking and sparse payload

`select` writes `int64` indices, parameters, and source/ranking digests under a
configuration-hashed directory. Budget `k` copies the first `k` aligned rows
and their metadata. `verify` checks digests and exact source rows;
`--recompute` regenerates the ranking.

`decode` writes `<sparse-root>/<scene>/feat_dec_fs/<scene>.pt` with decoded
`[k,3584]` features and the unchanged row indices/selection metadata.
Inference consumes the exact prefix as one `(1,3584,1,k)` pseudo-image.

## Questions

`infer --questions` accepts a JSON array with optional `answers`.
`train llava` requires a nonempty first answer.

```json
[
  {
    "question_id": "example-1",
    "scene_id": "scene0011_00",
    "question": "What color is the chair?",
    "answers": ["blue"]
  }
]
```

Predictions are grouped by scene under `<run>/inference/<condition-id>/predictions/`,
with sibling audits and timings. The condition ID covers questions, payloads,
model/adapter identity, and generation settings.
