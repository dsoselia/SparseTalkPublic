"""One-step training startup smoke with external inputs and weights.

Set SPARSETALK_SMOKE_STAGE to autoencoder, gaussian, or llava. Every stage
requires SPARSETALK_RUN. Autoencoder also needs SPARSETALK_FRAME_FEATURES;
Gaussian needs prior `prepare --split train` output under the run directory.
LoRA needs SPARSETALK_DENSE, SPARSETALK_SPARSE, SPARSETALK_SCENES,
SPARSETALK_TRAIN_ANNOTATIONS, SPARSETALK_MODEL_BASE,
SPARSETALK_VISION_TOWER, SPARSETALK_AUTOENCODER, and SPARSETALK_BUDGET.
"""

from __future__ import annotations

import os
import subprocess
import sys

from sparsetalk.paths import require_cuda


def value(name):
    result = os.environ.get(name)
    if not result:
        raise ValueError(f"set {name} before running the one-step training smoke")
    return result


def main():
    require_cuda()
    stage = os.environ.get("SPARSETALK_SMOKE_STAGE", "llava")
    command = [sys.executable, "-m", "sparsetalk.cli", "train", stage,
               "--run-dir", value("SPARSETALK_RUN"), "--max-steps", "1"]
    if stage == "autoencoder":
        command.extend(("--feature-root", value("SPARSETALK_FRAME_FEATURES"),
                        "--name", "startup_smoke", "--epochs", "1"))
    elif stage == "gaussian":
        command.extend(("--context-views", "16"))
    elif stage == "llava":
        command.extend((
            "--dense-root", value("SPARSETALK_DENSE"),
            "--sparse-root", value("SPARSETALK_SPARSE"),
            "--scenes", value("SPARSETALK_SCENES"),
            "--annotations", value("SPARSETALK_TRAIN_ANNOTATIONS"),
            "--model-base", value("SPARSETALK_MODEL_BASE"),
            "--vision-tower", value("SPARSETALK_VISION_TOWER"),
            "--autoencoder-checkpoint", value("SPARSETALK_AUTOENCODER"),
            "--budget", value("SPARSETALK_BUDGET"),
        ))
    else:
        raise ValueError(f"unknown training smoke stage: {stage}")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
