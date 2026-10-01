# Training Workflows

Training requires CUDA. Outputs use `<run>/training/` and `<run>/engine_outputs/`.

## Autoencoder

Generate 3584D frame features with `prepare --save-ov`. At least two scenes are
required for the train/evaluation split.

```bash
sparsetalk prepare --run-dir "$RUN" --scene-root "$RAW" \
  --scenes train_scenes.txt --split train --model-base "$BASE" --save-ov
sparsetalk train autoencoder --run-dir "$RUN" \
  --feature-root "$RUN/inputs/train" --name my_autoencoder --epochs 100
```

Use the trained checkpoint to add 256D targets from the saved frame features:

```bash
sparsetalk prepare --run-dir "$RUN" --scene-root "$RAW" \
  --scenes train_scenes.txt --split train --model-base "$BASE" --save-ov \
  --autoencoder-checkpoint "$RUN/training/autoencoder/ckpt/my_autoencoder/best_ckpt.pth"
```

`--max-steps 1` limits an autoencoder startup check to one optimizer step.

## Gaussian model

After `prepare --split train` has produced 256D frame targets:

```bash
sparsetalk train gaussian --run-dir "$RUN" --context-views 100 --max-steps 300001
```

`--gaussian-checkpoint` sets an optional starting checkpoint. This uses the
SplatTalk training workflow. Sparse-from-start optimization uses a separate
engine path from the post-hoc selectors.

## Sparse LoRA

You need verified sparse features, a base LLaVA model, vision tower, frozen
autoencoder, and [question/answer JSON](FORMATS.md).

```bash
sparsetalk train llava --run-dir "$RUN" --dense-root "$DENSE" \
  --sparse-root "$SPARSE" --scenes train_scenes.txt \
  --annotations /data/questions_train.json --model-base "$BASE" \
  --vision-tower /models/siglip --autoencoder-checkpoint "$AE" \
  --budget 128 --max-steps 1000 --learning-rate 1e-5
```

The loader checks ranking/source prefixes. The adapter is saved under
`$RUN/training/llava/adapter`.
