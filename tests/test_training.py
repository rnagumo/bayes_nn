import pathlib
import tempfile
from copy import deepcopy

import torch
from torch import Tensor, nn

from bayes_nn.base_model import BaseModel
from bayes_nn.training import Trainer

X_DIM = 4
Y_DIM = 2


def test_trainer_run() -> None:

    model = TempModel()
    dataset_train = TempDataset()
    dataset_test = TempDataset()
    org_params = deepcopy(model.state_dict())

    with tempfile.TemporaryDirectory() as logdir:
        logdir = logdir + "/tmp/"
        trainer = Trainer(logdir=logdir)
        trainer.run(model, dataset_train, dataset_test)

        root = pathlib.Path(logdir)
        assert (root / "config.json").exists()
        assert list(root.glob("checkpoint_*.pt"))

    for key, value in model.state_dict().items():
        assert not torch.isclose(value, org_params[key]).all()


class TempModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()

        self._fc = nn.Linear(X_DIM, Y_DIM)

    def loss_func(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:

        return {"loss": ((y - self._fc(x) ** 2)).sum(-1)}


class TempDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._data = torch.randn(100, X_DIM)
        self._label = torch.randn(100, Y_DIM)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:

        return self._data[index], self._label[index]

    def __len__(self) -> int:

        return self._data.size(0)
