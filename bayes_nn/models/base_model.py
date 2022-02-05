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

    def loss_func(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Loss function.

        Args:
            x: Features.
            y: Targets.

        Returns:
            Dict of losses in shape of `(batch,)`.
        """

        raise NotImplementedError
