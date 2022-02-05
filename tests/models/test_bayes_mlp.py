import torch

from bayes_nn.models.bayes_mlp import BayesMLP

BATCH = 24
X_DIM = 5
Y_DIM = 3
MC_SAMPLES = 17


def test_forward() -> None:

    x = torch.randn(BATCH, X_DIM)
    model = BayesMLP(X_DIM, Y_DIM)
    y_mu, y_cov = model(x)

    assert y_mu.size() == (BATCH, Y_DIM)
    assert y_cov.size() == (BATCH, Y_DIM, Y_DIM)


def test_sample() -> None:

    x = torch.randn(BATCH, X_DIM)
    model = BayesMLP(X_DIM, Y_DIM, mc_samples=MC_SAMPLES)
    y = model.sample(x)

    assert y.size() == (MC_SAMPLES, BATCH, Y_DIM)


def test_loss_func() -> None:

    x = torch.randn(BATCH, X_DIM)
    y = torch.randn(BATCH, Y_DIM)
    model = BayesMLP(X_DIM, Y_DIM)

    loss_dict = model.loss_func(x, y)
    for value in loss_dict.values():
        assert value.size() == (BATCH,)
