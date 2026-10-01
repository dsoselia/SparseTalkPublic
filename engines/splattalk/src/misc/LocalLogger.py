import json
import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only

class LocalLogger(Logger):
    def __init__(self, log_path: Path | str | None = None) -> None:
        super().__init__()
        self.log_path = Path(
            log_path or os.environ.get("SPLATTALK_LOCAL_LOG_PATH", "outputs/local")
        ).resolve()
        self.log_path.mkdir(exist_ok=True, parents=True)
        self.metrics_path = self.log_path / "metrics.jsonl"
        self.experiment = None

    @property
    def name(self):
        return "LocalLogger"

    @property
    def version(self):
        return 0

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        def scalar(value):
            if hasattr(value, "detach"):
                value = value.detach().cpu()
                return value.item() if value.numel() == 1 else value.tolist()
            if isinstance(value, Path):
                return str(value)
            return value

        record = {"step": int(step)}
        record.update({key: scalar(value) for key, value in metrics.items()})
        with self.metrics_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list[Any],
        step: Optional[int] = None,
        **kwargs,
    ):
        # The function signature is the same as the wandb logger's, but the step is
        # actually required.
        assert step is not None
        for index, image in enumerate(images):
            path = self.log_path / f"{key}/{index:0>2}_{step:0>6}.png"
            path.parent.mkdir(exist_ok=True, parents=True)
            Image.fromarray(image).save(path)
