from bayes_nn.datasets.gp_dataset import GPDataset

TOTAL_SIZE = 100
NUM_CONTEXT_MIN = 3
NUM_CONTEXT_MAX = 7
NUM_TARGET_MIN = 10
NUM_TARGET_MAX = 16
X_DIM = 3
Y_DIM = 2
PARAMS = {
    "total_size": TOTAL_SIZE,
    "num_context_min": NUM_CONTEXT_MIN,
    "num_context_max": NUM_CONTEXT_MAX,
    "num_target_min": NUM_TARGET_MIN,
    "num_target_max": NUM_TARGET_MAX,
    "x_dim": X_DIM,
    "y_dim": Y_DIM,
}


def _base_case(train: bool) -> None:
    indices = [0, 1, 2]
    dataset = GPDataset(train=train, **PARAMS)
    x_ctx, y_ctx, x_tgt, y_tgt = dataset[indices]

    assert x_ctx.size(0) == len(indices)
    assert x_ctx.size(1) <= NUM_CONTEXT_MAX
    assert x_ctx.size(2) == X_DIM

    assert y_ctx.size(0) == len(indices)
    assert y_ctx.size(1) <= NUM_CONTEXT_MAX
    assert y_ctx.size(2) == Y_DIM

    assert x_tgt.size(0) == len(indices)
    assert x_tgt.size(1) <= NUM_TARGET_MAX
    assert x_tgt.size(2) <= X_DIM

    assert y_tgt.size(0) == len(indices)
    assert y_tgt.size(1) <= NUM_TARGET_MAX
    assert y_tgt.size(2) == Y_DIM


def test_get_item() -> None:

    _base_case(True)
    _base_case(False)
