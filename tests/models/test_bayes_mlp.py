import torch

from bayes_nn.models.bayes_mlp import BayesMLP

BATCH = 24
X_DIM = 5
Y_DIM = 3


def test_forward() -> None:

    x = torch.randn(BATCH, X_DIM)
    model = BayesMLP(X_DIM, Y_DIM)
    y_mu, y_cov = model(x)

    assert y_mu.size() == (BATCH, Y_DIM)
    assert y_cov.size() == (BATCH, Y_DIM, Y_DIM)


def test_loss_func() -> None:

    x = torch.randn(BATCH, X_DIM)
    y = torch.randn(BATCH, Y_DIM)
    model = BayesMLP(X_DIM, Y_DIM)

    loss_dict = model.loss_func(x, y)
    for value in loss_dict.values():
        assert value.size() == (BATCH,)
