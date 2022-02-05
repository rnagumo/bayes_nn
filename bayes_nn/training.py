import json
import pathlib
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor, optim
from torch.optim import optimizer
from torch.utils.data import dataloader
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from bayes_nn.base_model import BaseModel


@dataclass
class Config:
    seed: int = 0
    batch_size: int = 64
    max_steps: int = 2
    save_interval: int = 2
    test_interval: int = 2
    logdir: str = "./logs/"
    gpus: str = ""
    lr: float = 0.001
    weight_decay: float = 0.0
    model_name: str = ""
    dataset_name: str = ""
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_params: dict[str, dict[str, Any]] = field(default_factory=dict)


class Trainer:
    """Trainer for NN models.

    Args:
        seed: Random seed.
        batch_size: Batch size for training and testing.
        max_steps: Max number of training steps.
        save_interval: Interval steps for saving checkpoints.
        test_interval: Interval steps for testing.
        logdir: Path to log directory.
        gpus: GPUs option.
        lr: Learning rate of optimizer.
        weight_decay: Weight regularization of L2 norm.
        mmodel_name: Model's name.
        dataset_name: Dataset's name.
        model_params: Hyper-parameters of models.
        dataset_params: Hyper-parameters of datasets.
    """

    def __init__(self, **kwargs: Any) -> None:

        self._config = Config(**kwargs)
        self._global_steps = 0
        self._logdir = pathlib.Path("./logs/")
        self._device = torch.device("cpu")

        self._model: Optional[BaseModel] = None
        self._loader_train: Optional[dataloader.DataLoader] = None
        self._loader_test: Optional[dataloader.DataLoader] = None
        self._optimizer: Optional[optimizer.Optimizer] = None
        self._writer: Optional[SummaryWriter] = None

    def run(self, model: BaseModel, dataset_train: Dataset, dataset_test: Dataset) -> None:

        self._make_logdir()
        self._init_writer()
        self._init_device()
        self._set_model(model)
        self._set_dataset(dataset_train, dataset_test)

        try:
            self._start_run()
        except Exception as e:
            self._save_checkpoint()
            raise e
        finally:
            self._quit()

    def _make_logdir(self) -> None:

        self._logdir = pathlib.Path(self._config.logdir)
        self._logdir.mkdir(parents=True)

    def _init_writer(self) -> None:

        self._writer = SummaryWriter(str(self._logdir))

    def _init_device(self) -> None:

        if self._config.gpus:
            self._device = torch.device(f"cuda:{self._config.gpus}")
        else:
            self._device = torch.device("cpu")

    def _set_model(self, model: BaseModel) -> None:

        self._model = model
        self._optimizer = optim.Adam(
            self._model.parameters(), lr=self._config.lr, weight_decay=self._config.weight_decay
        )

    def _set_dataset(self, dataset_train: Dataset, dataset_test: Dataset) -> None:

        if self._config.gpus:
            kwargs = {"num_workers": 0, "pin_memory": True}
        else:
            kwargs = {}

        self._loader_train = dataloader.DataLoader(
            dataset_train,
            shuffle=True,
            batch_size=self._config.batch_size,
            **kwargs,  # type: ignore
        )
        self._loader_test = dataloader.DataLoader(
            dataset_test,
            shuffle=False,
            batch_size=self._config.batch_size,
            **kwargs,  # type: ignore
        )

    def _start_run(self) -> None:

        assert self._model is not None

        torch.manual_seed(self._config.seed)
        random.seed(self._config.seed)

        self._model = self._model.to(self._device)
        self._global_steps = 0
        while self._global_steps < self._config.max_steps:
            self._train()

    def _train(self) -> None:

        assert (
            self._model is not None
            and self._optimizer is not None
            and self._loader_train is not None
            and self._writer is not None
        )

        for data in self._loader_train:

            self._model.train()
            self._optimizer.zero_grad()

            data = (v.to(self._device) for v in data)
            loss_dict = self._model.loss_func(*data)
            loss = torch.stack([_loss for _loss in loss_dict.values()]).sum(0).mean()
            loss.backward()
            self._optimizer.step()

            self._global_steps += 1
            for key, value in loss_dict.items():
                self._writer.add_scalar(f"train/{key}", value.mean(), self._global_steps)

            if self._global_steps % self._config.test_interval == 0:
                self._test()

            if self._global_steps % self._config.save_interval == 0:
                self._save_checkpoint()

            if self._global_steps >= self._config.max_steps:
                break

    def _test(self) -> None:

        assert (
            self._model is not None and self._loader_test is not None and self._writer is not None
        )

        loss_logger: defaultdict[str, Tensor] = defaultdict(Tensor)
        self._model.eval()
        for data in self._loader_test:
            with torch.no_grad():
                data = (v.to(self._device) for v in data)
                loss_dict = self._model.loss_func(*data)

            for key, value in loss_dict.items():
                loss_logger[key] += value.sum()

        for key, value in loss_logger.items():
            self._writer.add_scalar(
                f"test/{key}", value.item() / len(self._loader_test), self._global_steps
            )

    def _save_checkpoint(self) -> None:

        assert self._model is not None and self._optimizer is not None

        model_state_dict = {}
        for k, v in self._model.state_dict().items():
            model_state_dict[k.replace("module.", "")] = v

        optimizer_state_dict = {}
        for k, v in self._optimizer.state_dict().items():
            optimizer_state_dict[k.replace("module.", "")] = v

        state_dict = {
            "steps": self._global_steps,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
        }

        path = self._logdir / f"checkpoint_{self._global_steps}.pt"
        torch.save(state_dict, path)

    def _quit(self) -> None:

        assert self._writer is not None
        self._save_configs()
        self._writer.close()

    def _save_configs(self) -> None:

        config_dict = asdict(self._config)
        path = self._logdir / "config.json"
        with path.open("w") as f:
            json.dump(config_dict, f, indent=4)
