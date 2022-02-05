from typing import Union

import torch
from torch import Tensor
from torch.utils.data import Dataset

from bayes_nn.models.gaussian_process import GaussianProcess


class GPSingleBatchDataset(Dataset):
    """Gaussian Process dataset class with single batch.

    Args:
        train: Boolean for specifying train or test.
        total_size: Number of total batch.
        x_dim: Dimension size of input x.
        y_dim: Dimension size of output y.
        gp_params: Parameters dict for GP class.
    """

    def __init__(
        self,
        seq_len: int,
        n_sample: int,
        total_size: int = 1,
        x_dim: int = 1,
        y_dim: int = 1,
        l2_scale: float = 0.4,
        variance: float = 1.0,
    ) -> None:
        super().__init__()

        self._seq_len = seq_len
        self._n_sample = n_sample
        self._total_size = total_size
        self._x_dim = x_dim
        self._y_dim = y_dim

        self._gp = GaussianProcess(l2_scale=l2_scale, variance=variance)
        self._x = torch.tensor([])  # `(total_size, x_dim)`
        self._y = torch.tensor([])  # `(total_size, y_dim)`
        self.generate_dataset()

    def generate_dataset(self, x_ub: float = 2.0, x_lb: float = -2.0) -> None:
        """Initialize dataset.

        Args:
            x_ub: Upper bound of x range.
            x_lb: Lower bound of x range.
        """

        x = torch.arange(x_lb, x_ub, (x_ub - x_lb) / self._seq_len)
        _x_index = torch.sort(torch.randperm(self._seq_len)[:self._n_sample])[0]
        x = x[_x_index]
        x = x[None, :, None].repeat(self._total_size, 1, self._x_dim)
        y = self._gp.sample(x, y_dim=self._y_dim)

        self._x = x
        self._y = y

    def __getitem__(self, index: Union[int, list[int]]) -> tuple[Tensor, Tensor]:

        return (self._x[index], self._y[index])

    def __len__(self) -> int:

        return self._total_size


class GPDataset(Dataset):
    """Gaussian Process dataset class.

    Args:
        train: Boolean for specifying train or test.
        total_size: Number of total batch.
        num_context_min: Lower bound of number of context data.
        num_context_max: Upper bound of number of context data.
        num_target_min: Lower bound of number of target data.
        num_target_max: Upper bound of number of target data.
        x_dim: Dimension size of input x.
        y_dim: Dimension size of output y.
        gp_params: Parameters dict for GP class.
    """

    def __init__(
        self,
        train: bool,
        total_size: int,
        num_context_min: int = 3,
        num_context_max: int = 10,
        num_target_min: int = 2,
        num_target_max: int = 10,
        x_dim: int = 1,
        y_dim: int = 1,
        l2_scale: float = 0.4,
        variance: float = 1.0,
    ) -> None:
        super().__init__()

        self._train = train
        self._total_size = total_size
        self._num_context_min = num_context_min
        self._num_context_max = num_context_max
        self._num_target_min = num_target_min
        self._num_target_max = num_target_max
        self._x_dim = x_dim
        self._y_dim = y_dim

        self._gp = GaussianProcess(l2_scale=l2_scale, variance=variance)
        self._x_context = torch.tensor([])  # `(total_size, num_context, x_dim)`
        self._y_context = torch.tensor([])  # `(total_size, num_context, y_dim)`
        self._x_target = torch.tensor([])  # `(total_size, num_target, x_dim)`
        self._y_target = torch.tensor([])  # `(total_size, num_target, y_dim)`

        self.generate_dataset()

    def generate_dataset(
        self,
        x_ub: float = 2.0,
        x_lb: float = -2.0,
        resample_params: bool = False,
        single_params: bool = True,
    ) -> None:
        """Initializes dataset.

        **Note**

        * `num_context` and `num_target` are sampled from uniform distributions. Therefore, these
            two values might be changed at each time this function is called.
        * Target dataset includes context dataset, that is, context is a subset of target.

        Args:
            x_ub: Upper bound of x range.
            x_lb: Lower bound of x range.
            resample_params: If `True`, resample gaussian kernel parameters.
            single_params: If `True`, resampled kernel parameters are single values.
        """

        # Bounds number of dataset
        num_context_max = max(self._num_context_min, self._num_context_max)
        num_target_max = max(self._num_target_min, self._num_target_max)

        # Sample number of data points
        num_context = int(torch.randint(self._num_context_min, num_context_max, ()).item())
        num_target = (
            int(torch.randint(self._num_target_min, num_target_max, ()).item())
            if self._train
            else max(self._num_target_min, self._num_target_max)
        )

        # Sample input x for target
        if self._train:
            # For training, sample random points in range of [x_lb, x_ub]
            x = torch.rand(self._total_size, num_target, self._x_dim)
            x = x * (x_ub - x_lb) + x_lb
        else:
            # For test, sample uniformly distributed array in range of [x_lb, x_ub]
            x = torch.arange(x_lb, x_ub, (x_ub - x_lb) / num_target)
            x = x.view(1, -1, 1).repeat(self._total_size, 1, self._x_dim)

        # Sample y from GP prior
        y = self._gp.sample(
            x, y_dim=self._y_dim, resample_params=resample_params, single_params=single_params
        )

        # Sample random data points as context from target
        _x_context = torch.empty(self._total_size, num_context, self._x_dim)
        _y_context = torch.empty(self._total_size, num_context, self._y_dim)
        for i in range(self._total_size):
            indices = torch.randint(0, num_target, (num_context,))
            _x_context[i] = x[i, indices]
            _y_context[i] = y[i, indices]

        self._x_context = _x_context
        self._y_context = _y_context
        self._x_target = x
        self._y_target = y

    def __getitem__(self, index: Union[int, list[int]]) -> tuple[Tensor, Tensor, Tensor, Tensor]:

        return (
            self._x_context[index],
            self._y_context[index],
            self._x_target[index],
            self._y_target[index],
        )

    def __len__(self) -> int:

        return self._total_size
