"""SplatTalk and LLaVA workflow wrappers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from sparsetalk.paths import (
    LLAVA_ENGINE, SPLATTALK_ENGINE, atomic_json, canonical_sha256, dense_path,
    file_sha256, require_cuda, require_run_local_output,
)
from sparsetalk.selection import verify_sparse


def _environment(engine: Path, run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(engine), str(SPLATTALK_ENGINE / "autoencoder"),
        str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", ""),
    )))
    cache = Path(env.get("HF_HOME", Path.home() / ".cache" / "sparsetalk" / "huggingface")).resolve()
    repository = Path(__file__).resolve().parents[2]
    if cache == repository or repository in cache.parents:
        raise ValueError("HF_HOME must be outside the source repository")
    env["HF_HOME"] = str(cache)
    env["TMPDIR"] = str((run_dir / "tmp").resolve())
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.pop("SPLATTALK_REQUIRE_H200", None)
    (run_dir / "tmp").mkdir(parents=True, exist_ok=True)
    Path(env["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def _run(command: list[str], *, engine: Path, run_dir: Path, name: str,
         extra_env: dict | None = None, cwd: Path | None = None) -> None:
    require_cuda()
    run_dir.mkdir(parents=True, exist_ok=True)
    env = _environment(engine, run_dir)
    env.update(extra_env or {})
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(log_dir / f"{name}.command.json", {
        "argv": command, "cwd": str(cwd or engine),
        "gpu": torch.cuda.get_device_name(0),
    })
    with (log_dir / f"{name}.log").open("a") as log:
        subprocess.run(command, cwd=cwd or engine, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def _verify_decoded_rows(decoder, encoded: dict, decoded: dict, batch_size: int = 256) -> None:
    for start in range(0, len(encoded["features"]), batch_size):
        with torch.inference_mode():
            expected = decoder.decode(encoded["features"][start:start + batch_size].float().cuda()).cpu()
        torch.testing.assert_close(
            decoded["features"][start:start + batch_size], expected,
            rtol=1e-5, atol=1e-5,
        )


def extract(run_dir: Path, scenes: list[str], checkpoint: Path, context_views: int, target_views: int) -> Path:
    require_cuda()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if context_views <= 0 or target_views <= 0:
        raise ValueError("view counts must be positive")
    input_root = run_dir / "inputs"
    for scene in scenes:
        if not (input_root / "scanqa" / scene / "language_feats_256" / f"{scene}.pt").is_file():
            raise ValueError(f"prepare encoded frame features before extraction: {scene}")
    output = run_dir / "dense"
    command = [
        sys.executable, "-u", "-m", "src.main", "+experiment=scannet/fvt",
        "output_dir=extract", "mode=test", f"dataset.roots=[{input_root}]",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.num_context_views={context_views}",
        f"dataset.view_sampler.num_target_views={target_views}",
        "dataset.view_sampler.min_distance_between_context_views=25",
        "dataset.view_sampler.max_distance_between_context_views=50",
        "dataset.view_sampler.min_distance_to_context_views=0",
        "dataset.view_sampler.initial_min_distance_between_context_views=15",
        "dataset.view_sampler.initial_max_distance_between_context_views=25",
        "model.encoder.num_views=16", f"checkpointing.load={checkpoint.resolve()}",
        "data_loader.test.num_workers=0", "data_loader.test.persistent_workers=false",
        "wandb.mode=disabled", f"hydra.run.dir={run_dir / 'hydra' / 'extract'}",
    ]
    _run(command, engine=SPLATTALK_ENGINE, run_dir=run_dir, name="extract", cwd=run_dir, extra_env={
        "SPLATTALK_GAUSSIAN_OUTPUT": str(output.resolve()),
        "SPLATTALK_OUTPUT_BASE": str((run_dir / "engine_outputs").resolve()),
        "SPLATTALK_FEATURE_ROOT": str(input_root.resolve()),
    })
    for scene in scenes:
        if not dense_path(output, scene).is_file():
            raise RuntimeError(f"extraction did not write {dense_path(output, scene)}")
    return output


def decode(run_dir: Path, sparse_root: Path, scenes: list[str], checkpoint: Path,
           dense_decoded_root: Path | None = None, batch_size: int = 256) -> None:
    require_cuda()
    sparse_root = require_run_local_output(sparse_root, run_dir)
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    scene_list = run_dir / "decode_scenes.txt"
    scene_list.parent.mkdir(parents=True, exist_ok=True)
    scene_list.write_text("\n".join(scenes) + "\n")
    _run([
        sys.executable, str(SPLATTALK_ENGINE / "autoencoder" / "decode_features.py"),
        "--dataset_path", str(sparse_root), "--dataset_name", "sparsetalk",
        "--ckpt_path", str(checkpoint), "--input_feat_dir", "feat_fs",
        "--output_feat_dir", "feat_dec_fs", "--scene_list", str(scene_list),
        "--batch_size", str(batch_size),
    ], engine=SPLATTALK_ENGINE / "autoencoder", run_dir=run_dir, name="decode", cwd=run_dir)
    from sparsetalk.selection import _load_decoder

    decoder = _load_decoder(checkpoint)
    for scene in scenes:
        encoded = torch.load(sparse_root / scene / "feat_fs" / f"{scene}.pt", map_location="cpu", weights_only=False)
        decoded = torch.load(sparse_root / scene / "feat_dec_fs" / f"{scene}.pt", map_location="cpu", weights_only=False)
        if (decoded["features"].shape != (len(encoded["features"]), 3584)
                or not torch.equal(decoded["selected_idx"], encoded["selected_idx"])
                or decoded["sparsity"] != encoded["sparsity"]
                or not torch.isfinite(decoded["features"]).all().item()):
            raise ValueError(f"decoded sparse payload failed verification: {scene}")
        _verify_decoded_rows(decoder, encoded, decoded, batch_size)
        if dense_decoded_root is not None:
            source = torch.load(
                dense_decoded_root / scene / "feat_dec_fs" / f"{scene}.pt",
                map_location="cpu", weights_only=False,
            )
            torch.testing.assert_close(
                decoded["features"], source["features"][encoded["selected_idx"]],
                rtol=1e-5, atol=1e-5,
            )


def _normalize_questions(path: Path | None, prompt: str | None, scene: str | None) -> list[dict]:
    if path is not None:
        items = json.loads(path.read_text())
        if not isinstance(items, list) or not items:
            raise ValueError("questions JSON must be a nonempty array")
    elif prompt is not None and scene is not None:
        items = [{"scene_id": scene, "question_id": f"{scene}_0", "question": prompt}]
    else:
        raise ValueError("provide --questions or both --prompt and --scene")
    normalized = []
    for index, item in enumerate(items):
        if not item.get("scene_id") or not item.get("question"):
            raise ValueError(f"question {index} needs scene_id and question")
        answers = item.get("answers", [""])
        if not isinstance(answers, list) or not answers:
            raise ValueError(f"question {index} answers must be a nonempty list when supplied")
        normalized.append({
            "scene_id": str(item["scene_id"]),
            "question_id": str(item.get("question_id", f"question_{index}")),
            "question": str(item["question"]),
            "answers": answers,
        })
    if len({item["question_id"] for item in normalized}) != len(normalized):
        raise ValueError("question IDs must be unique")
    return normalized


def infer(run_dir: Path, sparse_root: Path | None, model_base: str,
          decoder_checkpoint: Path | None, adapter: Path | None,
          questions: Path | None, prompt: str | None, scene: str | None,
          selection: str = "preselected", ntokens: int = 44,
          max_new_tokens: int = 1024) -> Path:
    require_cuda()
    if selection not in {"preselected", "text_only", "entropy"}:
        raise ValueError("selection must be preselected, text_only, or entropy")
    if ntokens <= 0 or max_new_tokens <= 0:
        raise ValueError("ntokens and max_new_tokens must be positive")
    if adapter is not None and not adapter.is_dir():
        raise FileNotFoundError(adapter)
    adapter_weights = {}
    if adapter is not None:
        for path in sorted(adapter.rglob("*")):
            if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt"}:
                adapter_weights[str(path.relative_to(adapter))] = file_sha256(path)
        if not adapter_weights:
            raise ValueError(f"adapter directory contains no weight files: {adapter}")
    if selection != "text_only" and sparse_root is None:
        raise ValueError("visual inference requires --sparse-root")
    if selection == "preselected" and (decoder_checkpoint is None or not decoder_checkpoint.is_file()):
        raise ValueError("preselected inference requires --autoencoder-checkpoint for decoded-feature verification")
    records = _normalize_questions(questions, prompt, scene)
    if selection == "preselected":
        from sparsetalk.selection import _load_decoder

        decoder = _load_decoder(decoder_checkpoint)
        for scene_id in {item["scene_id"] for item in records}:
            encoded_path = sparse_root / scene_id / "feat_fs" / f"{scene_id}.pt"
            decoded_path = sparse_root / scene_id / "feat_dec_fs" / f"{scene_id}.pt"
            encoded = torch.load(encoded_path, map_location="cpu", weights_only=False)
            decoded = torch.load(decoded_path, map_location="cpu", weights_only=False)
            if (decoded["sparsity"] != encoded["sparsity"]
                    or not torch.equal(decoded["selected_idx"], encoded["selected_idx"])):
                raise ValueError(f"encoded/decoded selection mismatch: {scene_id}")
            if not torch.isfinite(decoded["features"]).all().item():
                raise ValueError(f"non-finite decoded features: {scene_id}")
            _verify_decoded_rows(decoder, encoded, decoded)
    scene_ids = sorted({item["scene_id"] for item in records})
    input_digests = {}
    if selection != "text_only":
        for scene_id in scene_ids:
            path = sparse_root / scene_id / "feat_dec_fs" / f"{scene_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            input_digests[scene_id] = file_sha256(path)
    condition = {
        "selection": selection,
        "model_base": model_base,
        "adapter": str(adapter.resolve()) if adapter else None,
        "adapter_config_sha256": file_sha256(adapter / "adapter_config.json") if adapter else None,
        "adapter_weight_sha256": adapter_weights,
        "decoder_sha256": file_sha256(decoder_checkpoint) if decoder_checkpoint else None,
        "questions_sha256": canonical_sha256(records),
        "decoded_payloads": input_digests,
        "ntokens": ntokens,
        "max_new_tokens": max_new_tokens,
    }
    inference_root = run_dir / "inference" / canonical_sha256(condition)[:16]
    condition_path = inference_root / "condition.json"
    if condition_path.is_file() and json.loads(condition_path.read_text()) != condition:
        raise ValueError(f"inference condition differs: {condition_path}")
    atomic_json(condition_path, condition)
    input_json = inference_root / "questions.json"
    atomic_json(input_json, records)
    output = inference_root / "predictions"
    audit = inference_root / "audits"
    timing = inference_root / "timings"
    for directory in (output, audit, timing):
        directory.mkdir(parents=True, exist_ok=True)
    scene_dir = sparse_root if sparse_root is not None else inference_root / "empty_scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, "-u", str(LLAVA_ENGINE / "scripts" / "decoding_llm.py"),
        "--scene_dir", str(scene_dir), "--model_base", model_base,
        "--json_path", str(input_json), "--scene_name", "all",
        "--json_save_path", str(output), "--language_feats_dir", "feat_dec_fs",
        "--gaussian-selection", selection, "--selection-audit-dir", str(audit),
        "--inference-timing-dir", str(timing), "--ntokens", str(ntokens),
        "--max-new-tokens", str(max_new_tokens),
        "--device_map", "cuda",
    ]
    if adapter is not None:
        command.extend(("--model_path", str(adapter)))
    _run(command, engine=LLAVA_ENGINE, run_dir=inference_root, name="infer", cwd=run_dir)
    for scene_id in {item["scene_id"] for item in records}:
        if not (output / f"{scene_id}.json").is_file() or not (audit / f"{scene_id}.json").is_file():
            raise RuntimeError(f"inference output or audit missing for {scene_id}")
    return output


def associate_detected(run_dir: Path, scenes: list[str], scene_root: Path, dense_root: Path,
                       florence_model: Path, sam2_repo: Path, sam2_checkpoint: Path,
                       frame_count: int) -> Path:
    root = run_dir / "associations" / "detected"
    detection_root = run_dir / "detections"
    for scene in scenes:
        _run([
            sys.executable, str(SPLATTALK_ENGINE / "autoencoder" / "detected_object_sampling" / "detection.py"),
            "--scene", scene, "--scene-data-root", str(scene_root),
            "--output-root", str(detection_root), "--florence-model", str(florence_model),
            "--sam2-repo", str(sam2_repo), "--sam2-checkpoint", str(sam2_checkpoint),
            "--frame-count", str(frame_count),
        ], engine=SPLATTALK_ENGINE / "autoencoder", run_dir=run_dir,
            name=f"detect_{scene}", cwd=run_dir)
        _run([
            sys.executable, str(SPLATTALK_ENGINE / "autoencoder" / "detected_object_sampling" / "association.py"),
            "--scene", scene, "--dense-root", str(dense_root),
            "--scene-data-root", str(scene_root), "--detection-root", str(detection_root),
            "--output", str(root / f"{scene}.pt"),
        ], engine=SPLATTALK_ENGINE / "autoencoder", run_dir=run_dir,
            name=f"associate_{scene}", cwd=run_dir)
    return root


def associate_masks(run_dir: Path, scenes: list[str], dense_root: Path,
                    scans_root: Path, sens_root: Path, frame_count: int) -> Path:
    root = run_dir / "associations" / "masks"
    for scene in scenes:
        sens = sens_root / scene / f"{scene}.sens"
        _run([
            sys.executable, str(SPLATTALK_ENGINE / "autoencoder" / "oracle_object_sampling" / "association.py"),
            "--scene", scene, "--dense-root", str(dense_root),
            "--scans-root", str(scans_root), "--sens-path", str(sens),
            "--output", str(root / f"{scene}.pt"), "--frame-count", str(frame_count),
        ], engine=SPLATTALK_ENGINE / "autoencoder", run_dir=run_dir,
            name=f"associate_masks_{scene}", cwd=run_dir)
    return root


def _convert_training_annotations(source: Path, output: Path, scenes: list[str]) -> None:
    items = _normalize_questions(source, None, None)
    allowed = set(scenes)
    records = []
    for item in items:
        scene = item["scene_id"]
        if scene not in allowed:
            raise ValueError(f"training question uses unknown scene: {scene}")
        if not item["answers"][0]:
            raise ValueError(f"training question {item['question_id']} has no answer")
        records.append({
            "id": item["question_id"],
            "image": [f"scenes/{scene}/placeholder.jpg"],
            "metadata": {"dataset": "user", "scene_name": scene},
            "conversations": [
                {"from": "human", "value": f"<image>\n{item['question']}"},
                {"from": "gpt", "value": str(item["answers"][0])},
            ],
        })
    atomic_json(output, records)


def train_autoencoder(run_dir: Path, feature_root: Path, name: str, epochs: int,
                      max_steps: int | None = None) -> Path:
    if not name or Path(name).name != name:
        raise ValueError("autoencoder name must be a single directory name")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps <= 0:
        raise ValueError("max steps must be positive")
    output = run_dir / "training" / "autoencoder"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(SPLATTALK_ENGINE / "autoencoder" / "train.py"),
        "--dataset_path", str(feature_root), "--dataset_name", name,
        "--num_epochs", str(epochs),
    ]
    if max_steps is not None:
        command.extend(("--max_steps", str(max_steps)))
    _run(command, engine=SPLATTALK_ENGINE / "autoencoder", run_dir=output,
        name="train_autoencoder", cwd=output)
    checkpoint = output / "ckpt" / name / "best_ckpt.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"autoencoder training did not produce {checkpoint}")
    return checkpoint


def train_gaussian(run_dir: Path, checkpoint: Path | None, max_steps: int, context_views: int) -> Path:
    if max_steps <= 0 or context_views <= 0:
        raise ValueError("max steps and context views must be positive")
    input_root = run_dir / "inputs"
    if not (input_root / "train_idx.txt").is_file():
        raise ValueError("prepare training scenes before Gaussian training")
    command = [
        sys.executable, "-u", "-m", "src.main", "+experiment=scannet/fvt",
        "output_dir=gaussian_train", "mode=train", f"dataset.roots=[{input_root}]",
        f"dataset.view_sampler.num_context_views={context_views}",
        f"trainer.max_steps={max_steps}", "wandb.mode=disabled",
        f"hydra.run.dir={run_dir / 'hydra' / 'train_gaussian'}",
    ]
    if checkpoint:
        command.append(f"checkpointing.load={checkpoint}")
    _run(command, engine=SPLATTALK_ENGINE, run_dir=run_dir,
         name="train_gaussian", cwd=run_dir, extra_env={
        "SPLATTALK_OUTPUT_BASE": str((run_dir / "engine_outputs").resolve()),
        "SPLATTALK_FEATURE_ROOT": str(input_root.resolve()),
        "SPLATTALK_REQUIRE_FEATURE_TARGETS": "1",
    })
    return run_dir / "engine_outputs" / "gaussian_train" / "checkpoints"


def train_llava(run_dir: Path, dense_root: Path, sparse_root: Path, scenes: list[str],
                annotations: Path, model_base: str, vision_tower: str,
                decoder_checkpoint: Path, budget: int, max_steps: int,
                learning_rate: float) -> Path:
    require_cuda()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("sparse LoRA training requires a CUDA GPU with BF16 support")
    if budget <= 0 or max_steps <= 0 or learning_rate <= 0:
        raise ValueError("budget, max steps, and learning rate must be positive")
    if not decoder_checkpoint.is_file():
        raise FileNotFoundError(decoder_checkpoint)
    for scene in scenes:
        sparse = sparse_root / scene / "feat_fs" / f"{scene}.pt"
        payload = torch.load(sparse, map_location="cpu", weights_only=False)
        verify_sparse(payload["ranking"], dense_path(dense_root, scene), sparse, scene)
        if len(payload["features"]) < budget:
            raise ValueError(f"{scene} contains fewer than {budget} selected rows")
    work = run_dir / "training" / "llava"
    work.mkdir(parents=True, exist_ok=True)
    train_json = work / "train.json"
    scene_list = work / "scenes.txt"
    _convert_training_annotations(annotations, train_json, scenes)
    scene_list.write_text("\n".join(scenes) + "\n")
    output = work / "adapter"
    command = [
        sys.executable, str(LLAVA_ENGINE / "llava" / "train" / "train_mem.py"),
        "--model_name_or_path", model_base,
        "--vision_tower", vision_tower, "--version", "qwen_1_5",
        "--lora_enable", "True", "--lora_r", "16", "--lora_alpha", "64",
        "--data_path", str(train_json), "--image_folder", str(work),
        "--mm_projector_type", "mlp2x_gelu", "--mm_vision_select_layer", "-2",
        "--mm_use_im_start_end", "False", "--mm_use_im_patch_token", "False",
        "--image_aspect_ratio", "anyres_max_9", "--mm_patch_merge_type", "spatial_unpad",
        "--bf16", "True", "--attn_implementation", "sdpa", "--output_dir", str(output),
        "--max_steps", str(max_steps), "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "1", "--evaluation_strategy", "no",
        "--save_strategy", "steps", "--save_steps", str(max_steps),
        "--save_total_limit", "2", "--learning_rate", str(learning_rate),
        "--model_max_length", "32768", "--gradient_checkpointing", "True",
        "--dataloader_num_workers", "0", "--lazy_preprocess", "True",
        "--report_to", "none", "--gaussian_feature_mode", "sparse_ranked",
        "--gaussian_dense_root", str(dense_root),
        "--gaussian_sparse_root", str(sparse_root),
        "--gaussian_feature_decoder", str(decoder_checkpoint),
        "--gaussian_budget", str(budget), "--gaussian_scene_list", str(scene_list),
        "--gaussian_legacy_blocks", "False",
    ]
    _run(command, engine=LLAVA_ENGINE, run_dir=work, name="train_llava", cwd=work)
    if not (output / "adapter_config.json").is_file():
        raise RuntimeError("training finished without an adapter_config.json")
    return output
