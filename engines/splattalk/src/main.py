import hashlib
import json
import os
import random
import shutil
from pathlib import Path
from types import SimpleNamespace

import hydra
import numpy as np
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

import argparse

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.global_cfg import set_cfg
from src.loss import get_losses
from src.misc.LocalLogger import LocalLogger
from src.misc.step_tracker import StepTracker
from src.misc.wandb_tools import update_checkpoint_path
from src.model.decoder import get_decoder
from src.model.encoder import get_encoder
from src.model.model_wrapper import ModelWrapper


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def require_requested_gpu() -> None:
    if os.environ.get("SPLATTALK_REQUIRE_H200") != "1":
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing sparse training on CPU")
    name = torch.cuda.get_device_name(0)
    if "H200" not in name:
        raise RuntimeError(f"NVIDIA H200 is required, found {name!r}")


def write_initial_state_manifest(model: torch.nn.Module, path: str | None) -> None:
    if not path:
        return
    digest = hashlib.sha256()
    parameter_count = 0
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        digest.update(b"\n")
        parameter_count += value.numel()
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "splattalk_initial_state_v1",
        "state_sha256": digest.hexdigest(),
        "state_element_count": parameter_count,
    }
    if output.exists():
        if json.loads(output.read_text()) != payload:
            raise ValueError("model initialization differs from the existing run manifest")
        return
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def trusted_resume_weights_only(checkpoint_path):
    """Enable full deserialization for an explicitly trusted checkpoint."""
    if checkpoint_path is None:
        return None
    if os.environ.get("SPLATTALK_TRUST_RESUME_CHECKPOINT") == "1":
        return False
    return None


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main",
)



def train(cfg_dict: DictConfig):
    require_requested_gpu()
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    random.seed(cfg_dict.seed)
    np.random.seed(cfg_dict.seed)
    torch.manual_seed(cfg_dict.seed)
    torch.cuda.manual_seed_all(cfg_dict.seed)

    if cfg.output_dir is None:
        output_dir = Path(
            hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
        )
        print(cyan(f"Saving outputs to {output_dir}."))
        latest_run = output_dir.parents[1] / "latest-run-1"
    else:
        output_base = Path(os.environ.get("SPLATTALK_OUTPUT_BASE", "outputs")).resolve()
        output_dir = (output_base / cfg.output_dir).resolve()
        if output_base != output_dir and output_base not in output_dir.parents:
            raise ValueError("output directory escapes SPLATTALK_OUTPUT_BASE")
        output_dir.mkdir(exist_ok=True, parents=True)
        print(cyan(f"Saving outputs to {output_dir}."))
        latest_run = output_base / "latest-run-1"

    if os.environ.get("SPLATTALK_DISABLE_LATEST_SYMLINK") != "1":
        latest_run.parent.mkdir(exist_ok=True, parents=True)
        if latest_run.is_symlink() or latest_run.is_file():
            latest_run.unlink()
        elif latest_run.exists():
            raise ValueError(f"latest-run path is not a symlink: {latest_run}")
        latest_run.symlink_to(output_dir, target_is_directory=True)

    # Set up logging with wandb.
    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        run = wandb.init(dir=str(output_dir), mode=cfg_dict.wandb.mode)
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model="all",
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
            experiment=run
        )
        callbacks.append(LearningRateMonitor("step", True))

    else:
        run = SimpleNamespace(dir=str(output_dir))
        logger = LocalLogger(output_dir / "local_logs")

    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_last=True,
        )
    )

    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy="ddp_find_unused_parameters_true"
        if torch.cuda.device_count() > 1
        else "auto",
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        limit_val_batches=cfg.trainer.limit_val_batches,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
    )

    # trainer = Trainer(
    #     max_epochs=-1,
    #     accelerator="gpu",
    #     logger=logger,
    #     devices="auto",
    #     strategy="ddp_find_unused_parameters_true"
    #     if torch.cuda.device_count() > 1
    #     else "auto",
    #     callbacks=callbacks,
    #     val_check_interval=cfg.trainer.val_check_interval,
    #     enable_progress_bar=False,
    #     max_steps=cfg.trainer.max_steps,
    #     check_val_every_n_epoch=None,
    # )

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder, 
                        depth_range=[cfg.dataset.near, cfg.dataset.far])
    cfg.test.output_path = output_dir
    
    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        cfg_dict=cfg_dict,
        run_dir=run.dir,
        num_context_views=cfg.dataset.view_sampler.num_context_views,
        dataset_name=cfg.dataset.name,
    )
    write_initial_state_manifest(
        model_wrapper,
        os.environ.get("SPLATTALK_INITIAL_STATE_MANIFEST"),
    )

    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker)

    if cfg.mode == "train":
        trainer.fit(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
            weights_only=trusted_resume_weights_only(checkpoint_path),
        )
    else:

        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
            weights_only=False,
        )


if __name__ == "__main__":
    train()
