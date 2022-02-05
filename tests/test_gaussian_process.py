import pytest
import torch

from bayes_nn.gaussian_process import GaussianProcess

BATCH_SIZE = 5
NUM_POINTS_0 = 10
NUM_POINTS_1 = 9
X_DIM = 3
Y_DIM = 2


def test_gaussian_kernel() -> None:

    x0 = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    x1 = torch.randn(BATCH_SIZE, NUM_POINTS_1, X_DIM)
    model = GaussianProcess()

    kernel = model.gaussian_kernel(x0, x0)
    assert kernel.size() == (BATCH_SIZE, NUM_POINTS_0, NUM_POINTS_0)

    kernel = model.gaussian_kernel(x0, x1)
    assert kernel.size() == (BATCH_SIZE, NUM_POINTS_0, NUM_POINTS_1)


def test_gaussian_kernel_with_raises() -> None:

    x0 = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    model = GaussianProcess()

    with pytest.raises(ValueError):
        x1 = torch.randn(BATCH_SIZE + 5, NUM_POINTS_1, X_DIM)
        model.gaussian_kernel(x0, x1)

    with pytest.raises(ValueError):
        x1 = torch.randn(BATCH_SIZE, NUM_POINTS_1, X_DIM + 5)
        model.gaussian_kernel(x0, x1)


def test_fit() -> None:

    x = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    y = torch.randn(BATCH_SIZE, NUM_POINTS_0, Y_DIM)

    model = GaussianProcess()
    model.fit(x, y)
    assert model._x_train is not None
    assert model._y_train is not None
    assert torch.isclose(model._x_train, x).all()
    assert torch.isclose(model._y_train, y).all()

    with pytest.raises(ValueError):
        x = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
        y = torch.randn(BATCH_SIZE, NUM_POINTS_0)
        model.fit(x, y)

    with pytest.raises(ValueError):
        x = torch.randn(BATCH_SIZE, NUM_POINTS_0)
        y = torch.randn(BATCH_SIZE, NUM_POINTS_0, Y_DIM)
        model.fit(x, y)


def test_predict() -> None:

    x = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    y = torch.randn(BATCH_SIZE, NUM_POINTS_0, Y_DIM)

    model = GaussianProcess()
    model.fit(x, y)
    y_mean, y_cov = model.predict(x)
    assert y_mean.size() == (BATCH_SIZE, NUM_POINTS_0, Y_DIM)
    assert y_cov.size() == (BATCH_SIZE, NUM_POINTS_0, NUM_POINTS_0)

    with pytest.raises(ValueError):
        x = torch.randn(BATCH_SIZE, X_DIM)
        model.predict(x)


def test_predict_prior() -> None:

    x = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    model = GaussianProcess()
    y_mean, y_cov = model.predict(x, Y_DIM)
    assert y_mean.size() == (BATCH_SIZE, NUM_POINTS_0, Y_DIM)
    assert y_cov.size() == (BATCH_SIZE, NUM_POINTS_0, NUM_POINTS_0)
    assert (y_mean == 0).all()


def test_forward() -> None:

    x = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    y = torch.randn(BATCH_SIZE, NUM_POINTS_0, Y_DIM)

    model = GaussianProcess()
    model.fit(x, y)
    y_mean = model(x)
    assert y_mean.size() == (BATCH_SIZE, NUM_POINTS_0, Y_DIM)


def test_sample() -> None:

    x = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    y = torch.randn(BATCH_SIZE, NUM_POINTS_0, Y_DIM)

    # Sample from prior
    model = GaussianProcess()
    y_sample = model.sample(x, Y_DIM, single_params=True)
    assert y_sample.size() == (BATCH_SIZE, NUM_POINTS_0, Y_DIM)

    # Sample params
    y_sample = model.sample(x, Y_DIM, single_params=False)
    assert y_sample.size() == (BATCH_SIZE, NUM_POINTS_0, Y_DIM)

    # Sample from posterior
    x_sample = torch.randn(BATCH_SIZE, NUM_POINTS_0 + 4, X_DIM)
    model.fit(x, y)
    y_sample = model.sample(x_sample)
    assert y_sample.size() == (BATCH_SIZE, NUM_POINTS_0 + 4, Y_DIM)


def test_sample_without_resample() -> None:

    x = torch.randn(BATCH_SIZE, NUM_POINTS_0, X_DIM)
    model = GaussianProcess()

    # Sample from prior
    y_sample = model.sample(x, Y_DIM, resample_params=False)
    assert y_sample.size() == (BATCH_SIZE, NUM_POINTS_0, Y_DIM)
