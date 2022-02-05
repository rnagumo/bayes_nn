from torch import Tensor, nn


class BaseModel(nn.Module):
    def loss_func(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:
        """Loss function.

        Args:
            x: Features.
            y: Targets.

        Returns:
            Dict of losses in shape of `(batch,)`.
        """

        raise NotImplementedError
