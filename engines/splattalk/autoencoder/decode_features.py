#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model import Autoencoder
from sparsity_utils import ALIGNED_KEYS, normalize_payload, validate_aligned


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("1", "true", "yes", "y"):
        return True
    if value.lower() in ("0", "false", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def read_scenes(data_dir, scene_list):
    if scene_list:
        scenes = [line.strip() for line in Path(scene_list).read_text().splitlines() if line.strip()]
    else:
        scenes = sorted(path.name for path in Path(data_dir).iterdir() if path.name.startswith("scene"))
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError("scene list must be nonempty and unique")
    return scenes


def decode_rows(model, features, batch_size):
    decoded = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].cuda(non_blocking=True)
        with torch.inference_mode():
            decoded.append(model.module.decode(batch).cpu())
    result = torch.cat(decoded, dim=0)
    if not torch.isfinite(result).all().item():
        raise ValueError("decoded features contain non-finite values")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--ckpt_path", default=None)
    parser.add_argument("--input_feat_dir", default="feat_fs")
    parser.add_argument("--output_feat_dir", required=True)
    parser.add_argument("--scene_list")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--overwrite", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--render", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to decode on CPU")
    if args.num_workers != 0:
        raise ValueError("direct payload decoding requires --num_workers 0")
    checkpoint_path = args.ckpt_path or f"ckpt/{args.dataset_name}/best_ckpt.pth"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = nn.DataParallel(Autoencoder()).cuda()
    model.load_state_dict(checkpoint)
    model.eval()

    data_dir = Path(args.dataset_path)
    for scene in read_scenes(data_dir, args.scene_list):
        input_path = data_dir / scene / args.input_feat_dir / f"{scene}.pt"
        output_path = data_dir / scene / args.output_feat_dir / f"{scene}.pt"
        if output_path.exists() and not args.overwrite:
            print(f"Exists: {scene}")
            continue
        payload = normalize_payload(torch.load(input_path, map_location="cpu", weights_only=False))
        validate_aligned(payload, expected_feature_dim=256)
        decoded = decode_rows(model, payload["features"].float(), args.batch_size)

        output = {key: payload[key] for key in ALIGNED_KEYS}
        output["features"] = decoded
        for key in ("selected_idx", "sparsity"):
            if key in payload:
                output[key] = payload[key]
        validate_aligned(output, expected_feature_dim=3584)
        if args.render:
            raise ValueError("render mode is not supported for dictionary sparsity payloads")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".pt.tmp")
        torch.save(output, temporary)
        temporary.replace(output_path)
        print(f"Decoded {scene}: {len(decoded)} rows")


if __name__ == "__main__":
    main()
