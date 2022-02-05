import datetime
import pathlib

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import optim

import bayes_nn.models as bayes_models

sns.set_context("talk")


def main() -> None:

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%s")
    path = pathlib.Path("./logs/", f"{now}_gp_example")
    path.mkdir(parents=True)

    gp = bayes_models.GaussianProcess()
    x_grid = torch.arange(0, 4, 0.01)
    y_grid = gp.sample(x_grid[None, :, None]).squeeze() + 4.2
    total_size = len(x_grid)
    y_grid = (y_grid - y_grid.mean()) / y_grid.std()

    index = torch.randperm(int(total_size * 0.6))[: int(total_size * 0.1)]
    x_trn = x_grid[index]
    y_trn = y_grid[index]

    model = bayes_models.BayesMLP(1, 1, dropout=0.2)
    optimizer = optim.Adam(model.parameters())

    n_epochs = 5000
    for epoch in range(1, n_epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        loss_dict = model.loss_func(x_trn[:, None], y_trn[:, None])
        loss = torch.stack([_loss for _loss in loss_dict.values()]).sum(0).mean()
        loss.backward()
        optimizer.step()

        # Test
        if epoch % (n_epochs // 5) == 0:
            with torch.no_grad():
                loss_train = (
                    model.loss_func(x_trn[:, None], y_trn[:, None])["mse_loss"].mean().item()
                )
                y_mu, y_cov = model(x_grid[:, None])
                y_sample = model.sample(x_grid[:, None])
                y_mu = y_mu[:, 0]
                y_cov = y_cov[:, 0, 0]
                y_sample = y_sample[..., 0]

            plt.figure(figsize=(8, 4))
            plt.plot(x_grid, y_grid)
            plt.scatter(x_trn, y_trn)
            plt.plot(x_grid, y_mu)
            plt.fill_between(x_grid, y_mu + y_cov, y_mu - y_cov, alpha=0.2)
            plt.title(f"{loss_train=:.2f}, {epoch=}")
            plt.tight_layout()
            plt.savefig(path / f"training_{epoch}.png")
            plt.close()
            print("Saved", str(path / f"training_{epoch}.png"))


if __name__ == "__main__":
    main()
