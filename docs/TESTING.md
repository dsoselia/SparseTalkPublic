# Verification

Computational tests require CUDA.

```bash
python -m pip install -e '.[test]'
pytest -q tests/test_selection_gpu.py tests/test_object_gpu.py
python tools/check_repository.py
```

Tests cover deterministic prefixes, aligned rows, tampering, malformed inputs,
decoded equivalence, preselected packing, object allocation, and sparse training.

`tools/smoke_external_gpu.py` checks dense-to-inference with external inputs.
Set `SPARSETALK_RAW` and `SPARSETALK_GAUSSIAN` for raw extraction; optional
Florence/SAM or ScanNet mask variables enable association checks.

`tools/smoke_train_gpu.py` runs one optimizer step. Select autoencoder,
Gaussian, or LoRA via `SPARSETALK_SMOKE_STAGE`. Both scripts list required
environment variables in their module headers.
