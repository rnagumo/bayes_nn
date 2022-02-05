from torch import Tensor, nn


class BaseModel(nn.Module):
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass to calculate target.

        Args:
            x: Input features.

        Returns:
            Tuple of `(mean, var)` of prediction.
        """

        raise NotImplementedError

    def sample(self, x: Tensor) -> Tensor:
        """Sample targets with Monte Carlo sampling.

        Args:
            Predctions with Monte Carlo sampling in 1st dimension.
        """

        raise NotImplementedError

    def loss_func(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Loss function.

        Args:
            x: Features.
            y: Targets.

        Returns:
            Dict of losses in shape of `(batch,)`.
        """

        raise NotImplementedError
