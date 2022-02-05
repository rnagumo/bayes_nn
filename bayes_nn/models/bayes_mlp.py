import torch
from torch import Tensor, nn

from bayes_nn.models.base_model import BaseModel


class BayesMLP(BaseModel):
    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        *,
        dropout: float = 0.05,
        prior_var: float = 1.0,
        mc_samples: int = 500,
    ) -> None:
        super().__init__()

        self._y_dim = y_dim
        self._prior_var = prior_var
        self._mc_samples = mc_samples

        self._fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(x_dim, 50),
            nn.Dropout(dropout),
            nn.Linear(50, 100),
            nn.Dropout(dropout),
            nn.Linear(100, 50),
            nn.Dropout(dropout),
            nn.Linear(50, y_dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:

        batch = x.size(0)

        y = self.sample(x)
        y_mu = y.mean(0)
        y_cov = torch.eye(self._y_dim)[None].repeat(batch, 1, 1) / self._prior_var
        y_cov = y_cov + (y[..., None] @ y[..., None, :]).mean(0) - y_mu[..., None] @ y_mu[:, None]

        return y_mu, y_cov

    def sample(self, x: Tensor) -> Tensor:

        if x.ndim != 2:
            raise ValueError(f"Input dim should be (batch, dim), but given {x.size()}")

        x = x[None].repeat(self._mc_samples, 1, 1)
        y = self._fc(x)

        return y

    def loss_func(self, x: Tensor, y: Tensor) -> dict[str, Tensor]:

        y_pred = self._fc(x)

        return {"mse_loss": ((y - y_pred) ** 2).sum(-1)}
