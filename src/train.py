import argparse
import datetime
import json
import os
import pathlib

import torch

from bayes_nn.experiment import Trainer


def main() -> None:

    args = _init_args()

    config_path = pathlib.Path(os.getenv("CONFIG_PATH", "./bin/config.json"))
    with config_path.open() as f:
        config = json.load(f)

    now = datetime.datetime.now().strftime("%Y%m%d%H%M%s")
    dirname = f"{now}_{args.model_name}_{args.dataset_name}_{args.seed}"
    logdir = str(pathlib.Path(os.getenv("LOGDIR", "./logs/"), dirname))

    use_cuda = torch.cuda.is_available() and args.cuda != "null"
    gpus = args.cuda if use_cuda else ""

    params = vars(args)
    params.update(config)
    params["logdir"] = logdir
    params["gpus"] = gpus
    params.pop("cuda")

    model = None
    dataset_train = None
    dataset_test = None

    trainer = Trainer(**params)
    trainer.run(model, dataset_train, dataset_test)


def _init_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--dataset-name", type=str)
    parser.add_argument("--cuda", type=str, default="null")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=2)
    parser.add_argument("--test-interval", type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    main()
