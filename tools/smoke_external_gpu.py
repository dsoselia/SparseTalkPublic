"""Optional end-to-end smoke using external data and model paths.

Required environment variables: SPARSETALK_RUN, SPARSETALK_DENSE,
SPARSETALK_SCENE, SPARSETALK_AUTOENCODER, SPARSETALK_MODEL_BASE.
Set SPARSETALK_RAW and SPARSETALK_GAUSSIAN to include raw extraction.
Optional SPARSETALK_GAUSSIAN_PYTHON, SPARSETALK_LLAVA_PYTHON, and
SPARSETALK_DETECTOR_PYTHON select separate environments; install this repo
editable in each.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from sparsetalk.paths import require_cuda


def required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"set {name} before running the external smoke")
    return value


def command(*args: str) -> str:
    environment_name = (
        "SPARSETALK_LLAVA_PYTHON" if args[0] in {"prepare", "infer"}
        else "SPARSETALK_DETECTOR_PYTHON" if args[0] == "associate"
        and "detected" in args else "SPARSETALK_GAUSSIAN_PYTHON"
    )
    python = os.environ.get(environment_name, sys.executable)
    process = subprocess.run(
        [python, "-m", "sparsetalk.cli", *args],
        check=True, text=True, capture_output=True,
    )
    return process.stdout.strip()


def main() -> None:
    require_cuda()
    run = required("SPARSETALK_RUN")
    scene = required("SPARSETALK_SCENE")
    dense = required("SPARSETALK_DENSE")
    autoencoder = required("SPARSETALK_AUTOENCODER")
    model_base = required("SPARSETALK_MODEL_BASE")
    raw = os.environ.get("SPARSETALK_RAW")
    gaussian = os.environ.get("SPARSETALK_GAUSSIAN")
    if bool(raw) != bool(gaussian):
        raise ValueError("set both SPARSETALK_RAW and SPARSETALK_GAUSSIAN for raw smoke")
    if raw:
        command("prepare", "--run-dir", run, "--scene-root", raw,
                "--scenes", scene, "--model-base", model_base,
                "--autoencoder-checkpoint", autoencoder)
        command("extract", "--run-dir", run, "--scenes", scene,
                "--gaussian-checkpoint", gaussian)
        dense = str(Path(run) / "dense")
    selected = json.loads(command(
        "select", "--run-dir", run, "--dense-root", dense,
        "--scenes", scene, "--method", "uniform_random",
        "--rank-length", "729", "--budget", "8",
    ))
    sparse = selected["sparse_root"]
    command("verify", "--run-dir", run, "--dense-root", dense,
            "--ranking-root", selected["ranking_root"], "--sparse-root", sparse,
            "--scenes", scene, "--recompute")
    command("decode", "--run-dir", run, "--sparse-root", sparse,
            "--scenes", scene, "--autoencoder-checkpoint", autoencoder)
    predictions = Path(command("infer", "--run-dir", run, "--sparse-root", sparse,
            "--model-base", model_base, "--autoencoder-checkpoint", autoencoder,
            "--scene", scene,
            "--prompt", "What is in the room?"))
    output = predictions / f"{scene}.json"
    audit = predictions.parent / "audits" / f"{scene}.json"
    if len(json.loads(output.read_text())) != 1:
        raise RuntimeError("smoke prediction coverage is not one")
    if json.loads(audit.read_text())["consumed_count"] != 8:
        raise RuntimeError("smoke audit did not consume eight rows")
    florence = os.environ.get("SPARSETALK_FLORENCE")
    sam_repo = os.environ.get("SPARSETALK_SAM_REPO")
    sam_checkpoint = os.environ.get("SPARSETALK_SAM_CHECKPOINT")
    if all((florence, sam_repo, sam_checkpoint)):
        command("associate", "--kind", "detected", "--run-dir", run,
                "--scenes", scene, "--scene-root", required("SPARSETALK_RAW"),
                "--dense-root", dense, "--florence-model", florence,
                "--sam2-repo", sam_repo, "--sam2-checkpoint", sam_checkpoint)
        object_root = str(Path(run) / "associations" / "detected")
        object_selection = json.loads(command(
            "select", "--run-dir", run, "--dense-root", dense,
            "--scenes", scene, "--method", "object_balanced_random",
            "--association-root", object_root, "--rank-length", "729", "--budget", "8",
        ))
        command("verify", "--run-dir", run, "--dense-root", dense,
                "--ranking-root", object_selection["ranking_root"],
                "--sparse-root", object_selection["sparse_root"],
                "--association-root", object_root, "--scenes", scene, "--recompute")
    scans = os.environ.get("SPARSETALK_SCANS_ROOT")
    sens = os.environ.get("SPARSETALK_SENS_ROOT")
    if scans and sens:
        command("associate", "--kind", "masks", "--run-dir", run,
                "--scenes", scene, "--dense-root", dense,
                "--scans-root", scans, "--sens-root", sens)
        object_root = str(Path(run) / "associations" / "masks")
        object_selection = json.loads(command(
            "select", "--run-dir", run, "--dense-root", dense,
            "--scenes", scene, "--method", "object_balanced_fps",
            "--association-root", object_root, "--rank-length", "729", "--budget", "8",
        ))
        command("verify", "--run-dir", run, "--dense-root", dense,
                "--ranking-root", object_selection["ranking_root"],
                "--sparse-root", object_selection["sparse_root"],
                "--association-root", object_root, "--scenes", scene, "--recompute")
    print(json.dumps({"validated": True, "scene": scene, "artifact_root": run}))


if __name__ == "__main__":
    main()
