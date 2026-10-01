"""Command-line interface for Gaussian selection and model workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sparsetalk.paths import (
    LLAVA_ENGINE, SPLATTALK_ENGINE, atomic_json, canonical_sha256,
    dense_path, require_cuda, require_external_run_dir, scene_ids, sparse_path,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparsetalk", description="Sparse Gaussian scene workflows")
    parser.add_argument("--config", type=Path, help="flat JSON defaults; command-line flags take precedence")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(command, *, scenes=False):
        item = sub.add_parser(command)
        item.add_argument("--run-dir", type=Path)
        if scenes:
            item.add_argument("--scenes", help="scene-list file or comma-separated IDs")
        return item

    doctor = common("doctor")
    doctor.add_argument("--autoencoder-checkpoint", type=Path)
    doctor.add_argument("--gaussian-checkpoint", type=Path)
    doctor.add_argument("--model-base")

    prepare = common("prepare", scenes=True)
    prepare.add_argument("--scene-root", type=Path)
    prepare.add_argument("--model-base")
    prepare.add_argument("--autoencoder-checkpoint", type=Path)
    prepare.add_argument("--split", choices=("scanqa", "train"))
    prepare.add_argument("--save-ov", action="store_true", default=None)
    prepare.add_argument("--batch-size", type=int)

    extract = common("extract", scenes=True)
    extract.add_argument("--gaussian-checkpoint", type=Path)
    extract.add_argument("--context-views", type=int)
    extract.add_argument("--target-views", type=int)

    associate = common("associate", scenes=True)
    associate.add_argument("--kind", choices=("detected", "masks"))
    associate.add_argument("--scene-root", type=Path)
    associate.add_argument("--dense-root", type=Path)
    associate.add_argument("--florence-model", type=Path)
    associate.add_argument("--sam2-repo", type=Path)
    associate.add_argument("--sam2-checkpoint", type=Path)
    associate.add_argument("--scans-root", type=Path)
    associate.add_argument("--sens-root", type=Path)
    associate.add_argument("--frame-count", type=int)

    select = common("select", scenes=True)
    select.add_argument("--dense-root", type=Path)
    select.add_argument("--method")
    select.add_argument("--budget", type=int)
    select.add_argument("--rank-length", type=int)
    select.add_argument("--seed", type=int)
    select.add_argument("--voxel-size", type=float)
    select.add_argument("--alpha", type=float)
    select.add_argument("--autoencoder-checkpoint", type=Path)
    select.add_argument("--association-root", type=Path)
    select.add_argument("--allocation-lambda", type=float)
    select.add_argument("--gamma", type=float)
    select.add_argument("--q-bg", type=float)
    select.add_argument("--batch-size", type=int)

    verify = common("verify", scenes=True)
    verify.add_argument("--dense-root", type=Path)
    verify.add_argument("--ranking-root", type=Path)
    verify.add_argument("--sparse-root", type=Path)
    verify.add_argument("--recompute", action="store_true", default=None)
    verify.add_argument("--autoencoder-checkpoint", type=Path)
    verify.add_argument("--association-root", type=Path)

    decode = common("decode", scenes=True)
    decode.add_argument("--sparse-root", type=Path)
    decode.add_argument("--autoencoder-checkpoint", type=Path)
    decode.add_argument("--dense-decoded-root", type=Path)
    decode.add_argument("--batch-size", type=int)

    infer = common("infer")
    infer.add_argument("--sparse-root", type=Path)
    infer.add_argument("--model-base")
    infer.add_argument("--autoencoder-checkpoint", type=Path)
    infer.add_argument("--adapter", type=Path)
    infer.add_argument("--questions", type=Path)
    infer.add_argument("--prompt")
    infer.add_argument("--scene")
    infer.add_argument("--selection", choices=("preselected", "entropy", "text_only"))
    infer.add_argument("--ntokens", type=int)
    infer.add_argument("--max-new-tokens", type=int)

    train = common("train", scenes=True)
    train.add_argument("stage", choices=("autoencoder", "gaussian", "llava"))
    train.add_argument("--feature-root", type=Path)
    train.add_argument("--name")
    train.add_argument("--epochs", type=int)
    train.add_argument("--gaussian-checkpoint", type=Path)
    train.add_argument("--context-views", type=int)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--dense-root", type=Path)
    train.add_argument("--sparse-root", type=Path)
    train.add_argument("--annotations", type=Path)
    train.add_argument("--model-base")
    train.add_argument("--vision-tower")
    train.add_argument("--autoencoder-checkpoint", type=Path)
    train.add_argument("--budget", type=int)
    train.add_argument("--learning-rate", type=float)
    return parser


def _value(args, config: dict, name: str, default=None, *, required=False):
    value = getattr(args, name, None)
    if value is None:
        value = config.get(name, default)
    if required and value is None:
        raise ValueError(f"--{name.replace('_', '-')} is required")
    return value


def _path(args, config, name, default=None, *, required=False) -> Path | None:
    value = _value(args, config, name, default, required=required)
    return Path(value).expanduser().resolve() if value is not None else None


def _scenes(args, config) -> list[str]:
    return scene_ids(_value(args, config, "scenes", required=True))


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    config = json.loads(args.config.read_text()) if args.config else {}
    if not isinstance(config, dict):
        raise ValueError("configuration JSON must be an object")
    run_dir = require_external_run_dir(_path(args, config, "run_dir", required=True))
    if args.command == "doctor":
        gpu = require_cuda()
        paths = {
            "gaussian_engine": SPLATTALK_ENGINE / "src" / "main.py",
            "llava_engine": LLAVA_ENGINE / "llava" / "train" / "train_mem.py",
        }
        for name in ("autoencoder_checkpoint", "gaussian_checkpoint"):
            value = _path(args, config, name)
            if value is not None:
                paths[name] = value
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing engine or checkpoint files: {missing}")
        print(json.dumps({"cuda_device": gpu, "paths": {k: str(v) for k, v in paths.items()}}, indent=2))
        return

    if args.command == "prepare":
        from sparsetalk.prepare import prepare_scenes

        result = prepare_scenes(
            _path(args, config, "scene_root", required=True), _scenes(args, config),
            run_dir, _value(args, config, "split", "scanqa"),
            _value(args, config, "model_base", required=True),
            _path(args, config, "autoencoder_checkpoint"),
            bool(_value(args, config, "save_ov", False)),
            int(_value(args, config, "batch_size", 8)),
        )
        print(json.dumps({"scene_count": result["scene_count"], "input_root": result["input_root"]}, indent=2))
        return

    if args.command == "extract":
        from sparsetalk.workflows import extract

        output = extract(run_dir, _scenes(args, config),
                         _path(args, config, "gaussian_checkpoint", required=True),
                         int(_value(args, config, "context_views", 100)),
                         int(_value(args, config, "target_views", 5)))
        print(output)
        return

    if args.command == "associate":
        from sparsetalk.workflows import associate_detected, associate_masks

        kind = _value(args, config, "kind", required=True)
        dense_root = _path(args, config, "dense_root", run_dir / "dense")
        if kind == "detected":
            output = associate_detected(
                run_dir, _scenes(args, config),
                _path(args, config, "scene_root", required=True), dense_root,
                _path(args, config, "florence_model", required=True),
                _path(args, config, "sam2_repo", required=True),
                _path(args, config, "sam2_checkpoint", required=True),
                int(_value(args, config, "frame_count", 100)),
            )
        else:
            output = associate_masks(
                run_dir, _scenes(args, config), dense_root,
                _path(args, config, "scans_root", required=True),
                _path(args, config, "sens_root", required=True),
                int(_value(args, config, "frame_count", 100)),
            )
        print(output)
        return

    if args.command == "select":
        from sparsetalk.selection import (
            METHODS, atomic_torch_save, create_ranking, materialize,
            validate_ranking, verify_sparse,
        )
        import torch

        require_cuda()
        scenes = _scenes(args, config)
        method = _value(args, config, "method", required=True)
        if method not in METHODS:
            raise ValueError(f"unknown method {method}; choose from {', '.join(METHODS)}")
        budget = int(_value(args, config, "budget", required=True))
        rank_length = int(_value(args, config, "rank_length", 729))
        if not 0 < budget <= rank_length:
            raise ValueError("budget must be positive and no larger than rank length")
        dense_root = _path(args, config, "dense_root", run_dir / "dense")
        decoder = _path(args, config, "autoencoder_checkpoint")
        association_root = _path(args, config, "association_root")
        parameters = {
            "method": method, "rank_length": rank_length,
            "seed": int(_value(args, config, "seed", 0)),
            "voxel_size": float(_value(args, config, "voxel_size", 0.01)),
            "alpha": float(_value(args, config, "alpha", 0.5)),
            "allocation_lambda": float(_value(args, config, "allocation_lambda", 0.5)),
            "gamma": float(_value(args, config, "gamma", 0.5)),
            "q_bg": float(_value(args, config, "q_bg", 0.2)),
            "dense_root": str(dense_root),
            "decoder": str(decoder) if decoder else None,
            "association_root": str(association_root) if association_root else None,
            "batch_size": int(_value(args, config, "batch_size", 256)),
            "scenes": scenes,
        }
        selection_dir = run_dir / "selections" / f"{method}-{canonical_sha256(parameters)[:12]}"
        config_path = selection_dir / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != parameters:
            raise ValueError(f"selection directory has a different configuration: {selection_dir}")
        if not config_path.exists():
            atomic_json(config_path, parameters)
        ranking_root = selection_dir / "rankings"
        sparse_root = selection_dir / f"k_{budget}"
        for scene in scenes:
            source = dense_path(dense_root, scene)
            ranking_path = ranking_root / f"{scene}.pt"
            if ranking_path.is_file():
                ranking = torch.load(ranking_path, map_location="cpu", weights_only=False)
                validate_ranking(
                    ranking, source, scene, decoder,
                    association_root / f"{scene}.pt" if association_root else None,
                )
            else:
                ranking = create_ranking(
                    source, scene, method, rank_length,
                    global_seed=parameters["seed"], voxel_size=parameters["voxel_size"],
                    alpha=parameters["alpha"], decoder_checkpoint=decoder,
                    association_path=association_root / f"{scene}.pt" if association_root else None,
                    allocation_lambda=parameters["allocation_lambda"],
                    gamma=parameters["gamma"], q_bg=parameters["q_bg"],
                    batch_size=parameters["batch_size"],
                )
                atomic_torch_save(ranking_path, ranking)
            output = sparse_path(sparse_root, scene)
            if not output.is_file():
                atomic_torch_save(output, materialize(ranking, source, scene, budget))
            verify_sparse(ranking, source, output, scene)
        print(json.dumps({"selection_dir": str(selection_dir), "ranking_root": str(ranking_root),
                          "sparse_root": str(sparse_root)}, indent=2))
        return

    if args.command == "verify":
        import torch
        from sparsetalk.selection import create_ranking, validate_ranking, verify_sparse

        require_cuda()
        dense_root = _path(args, config, "dense_root", required=True)
        ranking_root = _path(args, config, "ranking_root", required=True)
        sparse_root = _path(args, config, "sparse_root", required=True)
        scenes = _scenes(args, config)
        for scene in scenes:
            source = dense_path(dense_root, scene)
            ranking = torch.load(ranking_root / f"{scene}.pt", map_location="cpu", weights_only=False)
            decoder = _path(args, config, "autoencoder_checkpoint")
            association_root = _path(args, config, "association_root")
            association = association_root / f"{scene}.pt" if association_root else None
            validate_ranking(ranking, source, scene, decoder, association)
            verify_sparse(ranking, source, sparse_path(sparse_root, scene), scene)
            if _value(args, config, "recompute", False):
                metadata = ranking["ranking"]
                parameters = metadata["parameters"]
                expected = create_ranking(
                    source, scene, metadata["selection_method"], metadata["rank_length"],
                    global_seed=parameters["global_seed"],
                    voxel_size=parameters.get("voxel_size", 0.01),
                    alpha=parameters.get("alpha", 0.5),
                    decoder_checkpoint=decoder,
                    association_path=association,
                    allocation_lambda=parameters.get("lambda", 0.5),
                    gamma=parameters.get("gamma", 0.5), q_bg=parameters.get("q_bg", 0.2),
                    batch_size=parameters.get("entropy_batch_size", 256),
                )
                if metadata != expected["ranking"] or not torch.equal(ranking["selected_idx"], expected["selected_idx"]):
                    raise ValueError(f"ranking recomputation mismatch: {scene}")
        print(json.dumps({"verified_scenes": scenes}))
        return

    if args.command == "decode":
        from sparsetalk.workflows import decode

        decode(run_dir, _path(args, config, "sparse_root", required=True), _scenes(args, config),
               _path(args, config, "autoencoder_checkpoint", required=True),
               _path(args, config, "dense_decoded_root"),
               int(_value(args, config, "batch_size", 256)))
        print("decoded and verified")
        return

    if args.command == "infer":
        from sparsetalk.workflows import infer

        output = infer(
            run_dir, _path(args, config, "sparse_root"),
            _value(args, config, "model_base", required=True),
            _path(args, config, "autoencoder_checkpoint"),
            _path(args, config, "adapter"),
            _path(args, config, "questions"),
            _value(args, config, "prompt"), _value(args, config, "scene"),
            _value(args, config, "selection", "preselected"),
            int(_value(args, config, "ntokens", 44)),
            int(_value(args, config, "max_new_tokens", 1024)),
        )
        print(output)
        return

    if args.command == "train":
        from sparsetalk.workflows import train_autoencoder, train_gaussian, train_llava

        if args.stage == "autoencoder":
            output = train_autoencoder(
                run_dir, _path(args, config, "feature_root", run_dir / "inputs" / "scanqa"),
                _value(args, config, "name", "autoencoder"),
                int(_value(args, config, "epochs", 100)),
                int(_value(args, config, "max_steps")) if _value(args, config, "max_steps") is not None else None,
            )
        elif args.stage == "gaussian":
            output = train_gaussian(
                run_dir, _path(args, config, "gaussian_checkpoint"),
                int(_value(args, config, "max_steps", 300001)),
                int(_value(args, config, "context_views", 100)),
            )
        else:
            output = train_llava(
                run_dir, _path(args, config, "dense_root", required=True),
                _path(args, config, "sparse_root", required=True), _scenes(args, config),
                _path(args, config, "annotations", required=True),
                _value(args, config, "model_base", required=True),
                _value(args, config, "vision_tower", required=True),
                _path(args, config, "autoencoder_checkpoint", required=True),
                int(_value(args, config, "budget", required=True)),
                int(_value(args, config, "max_steps", 1000)),
                float(_value(args, config, "learning_rate", 1e-5)),
            )
        print(output)


if __name__ == "__main__":
    main()
