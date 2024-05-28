import pytorch_lightning as L
from data_loading import BracketDataset, load_data, get_loaders, load_from_csv
from transformer_predictor import TransformerPredictor, train, test
from probe import probe_all_models
from time import time
import argparse
import os
import torch
from torch.utils.data import DataLoader


class Logger:
    def __init__(self):
        self.logs = []
        self.file_name = f"logs/logs_{int(time())}.txt"
        # check if logs directory exists
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # initialize file
        with open(self.file_name, "w") as f:
            f.write("")

    def log(self, msg):
        self.logs.append(msg)
        with open(self.file_name, "a") as f:
            f.write(msg + "\n")


def parse_list(input_str):
    return input_str.split(",")


def get_stack_depths(loader: DataLoader, stack_depths=[5, 15]):
    stack_data = []
    for batch in loader:
        # print(batch)
        for i, x in enumerate(batch["sd"]):
            # if type(x) is not dict:
            #     print(x)
            if x in stack_depths:
                sample = {
                    "x": batch["x"][i],
                    "y": batch["y"][i],
                    "sd": x,
                    "eos": batch["eos"][i],
                    "count": batch["count"][i],
                }
                stack_data.append(sample)

    if len(stack_data) == 0:
        return None, 0

    stack_loader = DataLoader(stack_data, batch_size=64)

    return stack_loader, len(stack_data)


def get_counts(loader: DataLoader, counts=[1, 2, 3]):
    count_data = []
    for batch in loader:
        for i, x in enumerate(batch["count"]):
            if x in counts:
                sample = {
                    "x": batch["x"][i],
                    "y": batch["y"][i],
                    "sd": batch["sd"][i],
                    "eos": batch["eos"][i],
                    "count": x,
                }
                count_data.append(sample)

    if len(count_data) == 0:
        return None, 0

    count_loader = DataLoader(count_data, batch_size=64)

    return count_loader, len(count_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--max_iters", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--transformer_epochs", type=int, default=10)
    parser.add_argument("--data-path", type=str, default="brackets.csv")
    parser.add_argument("--test-data-path", type=str, default="brackets.csv")
    parser.add_argument(
        "--type", type=str, default="stack_depth", choices=["stack_depth", "count"]
    )
    parser.add_argument(
        "--length", type=int, default=512, help="Length of the input sequence"
    )

    args = parser.parse_args()

    logger = Logger()
    logger.log(f"Arguments: {args}")

    dataset = load_data(args.data_path)

    train_loader, val_loader, test_loader = get_loaders(
        dataset, batch_size=args.batch_size
    )

    model = TransformerPredictor(
        input_dim=4,
        model_dim=args.model_dim,
        num_classes=2,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=args.lr,
        warmup=args.warmup,
        max_iters=args.max_iters,
    )

    trainer = L.Trainer(max_epochs=args.transformer_epochs, devices=1)
    # check for saved model
    if os.path.exists(
        f"models/original_tasks_model_{args.length}_{args.num_layers}_{args.num_heads}.pt"
    ):
        model.load_state_dict(
            torch.load(
                f"models/original_tasks_model_{args.length}_{args.num_layers}_{args.num_heads}.pt"
            )
        )
    else:

        res, model, trainer, _, _ = train(model, trainer, train_loader, val_loader)

        # save the model
        torch.save(
            model.state_dict(),
            f"models/original_tasks_model_{args.length}_{args.num_layers}_{args.num_heads}.pt",
        )

    res, _, _ = test(model, trainer, train_loader)
    logger.log(f"Training results: {res}")

    testing_data_df = load_from_csv(args.test_data_path)
    testing_data: BracketDataset = load_data(args.test_data_path)
    testing_data_loader = DataLoader(testing_data, batch_size=args.batch_size)

    if args.type == "stack_depth":
        # get all unique stack depths in the dataset df
        stack_depths = testing_data_df["stack_depth"].unique()

        for stack_depth in stack_depths:
            stack_loader, len_stack = get_stack_depths(
                testing_data_loader, stack_depths=[stack_depth]
            )
            if stack_loader is None:
                continue
            res, _, _ = test(model, trainer, stack_loader)
            logger.log(
                f"Stack depth {stack_depth}, Support: {len_stack}, Accuracy: {res}"
            )

    elif args.type == "count":
        # get all unique count values in the dataset df
        counts = testing_data_df["count"].unique()

        for count in counts:
            count_loader, len_count = get_counts(testing_data_loader, counts=[count])
            if count_loader is None:
                continue
            res, _, _ = test(model, trainer, count_loader)
            logger.log(f"Count {count}, Support: {len_count}, Accuracy: {res}")

    else:
        raise ValueError(f"Invalid type {args.type}")
